import os
import time
import copy
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.lesson_merge import integrate_lesson
from src.utils import cal_time_cost, save_log_data
from src.llm_retriever import LibraryRetrieval
from src.train_eval_utils import check_optimality, divide_insight, generate_solution_with_retrieval, self_verify_test
from src.insight_states import INSIGHT_STATES, normalize_distribution, normalize_insight_state

def run_library_diagnosis(
    iter,
    train_tasks,
    llm_retri, llm_opt, llm_diag, llm_ins, library,
    params,
    paths,
    max_workers = 8,
):
    """
    Run library insight diagnosis through doing tasks
    """

    def _set_task_state(insight, task_id, state):
        state = normalize_insight_state(state)
        if state and state not in INSIGHT_STATES:
            raise ValueError(f"Unknown insight state: {state}")
        insight.distribution = normalize_distribution(getattr(insight, "distribution", None))

        for bucket, values in list(insight.distribution.items()):
            if isinstance(values, list) and task_id in values:
                insight.distribution[bucket] = [tid for tid in values if tid != task_id]

        if state:
            insight.distribution.setdefault(state, [])
            if task_id not in insight.distribution[state]:
                insight.distribution[state].append(task_id)

    def _build_snapshot_retriever(library_snapshot):
        if llm_retri is None:
            return None

        return LibraryRetrieval(
            lib=library_snapshot,
            model=llm_retri.model,
            service=llm_retri.service,
            temperature=getattr(llm_retri, "temp", None),
            document_retriever=getattr(llm_retri, "document_retriever", None),
        )


    def _train_worker(task, taxo_snapshot, library_snapshot, train_output_path):
        """
        Parallelize the entire per-task pipeline (insight retrieval -> formulation generation -> insight retrieval -> program generation -> diagnosis -> lesson extraction -> lesson verification) for training tasks
        Return: (new insights, is_success, is_execution, is_verify, is_diagnosis, distribution_updates)
        """
        snapshot_retri = _build_snapshot_retriever(library_snapshot)
        distribution_updates = []

        #* Retrieve insights (if any) and generate formulation and program
        prev_insights, candidate_formulation, candidate_program, output, runnable, is_time_out = generate_solution_with_retrieval(
                iter, task, library_snapshot, snapshot_retri, llm_opt,
                retrieved_insights=[],
                output_path=train_output_path, verbose=False, save_data=True
        )
        retrieved_ins_ids = list(dict.fromkeys(ins["insight_id"] for ins in prev_insights if "insight_id" in ins))
        task.retri_ins_lst.append(retrieved_ins_ids)
        if not candidate_program:
            task.output_status.append("parse_error")
            task.record_attempt(iter, 1, "parse_error", candidate_formulation, prev_insights)
            return [], False, False, None, None, distribution_updates
        is_optimal, output_status, feedback = check_optimality(task, output, runnable, is_time_out)
        task.record_attempt(iter, 1, output_status, candidate_formulation, prev_insights)
        task.output_status.append(output_status)

        if is_optimal:
            print(f"\n   [Task {task.id}]: Output was optimal. Task succeeds!")
            return [], True, None, None, None, distribution_updates
        else:
            print(feedback)

        form_context, code_context = divide_insight(prev_insights)
        correction = None
        code_fixed_task = False
        if output_status == "run_error":
            code_fixed_task, runnable, correction, corrected_feedback = llm_diag.diagnose_program(
                iter=iter, task=task, failed_program=candidate_program, feedback=feedback,
                save_data=True, output_path=train_output_path,
            )
            if not runnable:
                task.fail_to_execute += 1
                return [], False, False, None, None, distribution_updates
            reflection = llm_diag.reflect_program_correction(
                task, candidate_formulation, candidate_program, feedback, correction, code_context,
            )
            save_log_data(reflection, f"{train_output_path}/Diagnosis/program_reflection_iter_{iter}.json")
            distribution_updates.extend((row["insight_id"], task.id, row["state"]) for row in reflection)
            excluded = {row["insight_id"] for row in reflection if row["state"] != "useful"}
            code_context = [ins for ins in code_context if ins.get("insight_id") not in excluded]
            feedback = corrected_feedback
            candidate_program = correction

        def extract(context, **evidence):
            return llm_ins.generate_lessons(
                iter=iter, task=task, taxonomy=taxo_snapshot, retrieved_insights=context,
                verbose=False, save_data=True, output_path=train_output_path, **evidence,
            )

        if code_fixed_task:
            for _ in range(params.max_verify_attempts):
                lessons = extract(prev_insights, corrected_program=correction)
                if self_verify_test(iter, task, llm_opt, lessons, prev_insights):
                    return lessons, False, True, True, None, distribution_updates
            return [], False, True, False, None, distribution_updates

        # A runnable correction with a wrong objective still needs formulation diagnosis.
        needs_lesson = True
        failed_form = candidate_formulation
        code_lessons = extract(prev_insights, corrected_program=correction) if correction else []
        if snapshot_retri and snapshot_retri.has_available_knowledge("Formulation"):
            states, form_context, needs_lesson, failed_form = llm_diag.diagnose_formulation(
                iter=iter, task=task, feedback=feedback, failed_formulation=candidate_formulation,
                failed_program=candidate_program, retrieved_insights=[form_context, code_context + code_lessons],
                llm_opt=llm_opt, llm_retri=snapshot_retri, save_data=True, output_path=train_output_path,
            )
            distribution_updates.extend((iid, task.id, state) for iid, state in states.items())

        context = form_context + code_context
        task.attempt_history[-1]["verification_insights"] = context
        if not needs_lesson and not code_lessons:
            return [], False, True, None, True, distribution_updates
        for attempt in range(params.max_verify_attempts):
            if correction and attempt:
                code_lessons = extract(context, corrected_program=correction)
            lessons = list(code_lessons)
            if needs_lesson:
                lessons += extract(context, failed_formulation=failed_form)
            if self_verify_test(iter, task, llm_opt, lessons, context):
                return lessons, False, True, True, not needs_lesson, distribution_updates
        return [], False, True, False, not needs_lesson, distribution_updates

    # Experiment metrics
    # temp_lib = []
    train_success_flags = [False] * len(train_tasks)

    iter_verify_count = 0
    iter_verify_success = 0

    iter_diagnose_count = 0
    iter_diagnose_success = 0

    fail_to_verify_lst = []
    fail_to_execute_lst = []


    processed_count = 0

    # Counters for online merge rate calculation
    total_new_insights = 0
    successful_merges = 0


    def _apply_distribution_updates(distribution_updates):
        if not distribution_updates:
            return

        applied = 0
        id_to_insight = {ins.insight_id: ins for ins in library}
        for update in distribution_updates:
            if isinstance(update, dict):
                insight_id = update.get("insight_id")
                task_id = update.get("task_id")
                state = update.get("state")
            else:
                try:
                    insight_id, task_id, state = update
                except (TypeError, ValueError):
                    continue

            target_insight = id_to_insight.get(insight_id)
            if target_insight is None:
                continue
            _set_task_state(target_insight, task_id, state)
            applied += 1

        if applied:
            print(f"[Iteration {iter}] Applied {applied} insight distribution updates")

    def _process_new_insights(new_insights):
        nonlocal processed_count, total_new_insights, successful_merges
        if not new_insights:
            return

        for index, new_insight in enumerate(new_insights):
            total_new_insights += 1
            accepted = integrate_lesson(new_insight, new_insights[index+1:], train_tasks,
                                        library, llm_opt, llm_ins, iter)
            successful_merges += int(accepted)
            processed_count += 1
            if processed_count % 10 == 0:
                try:
                    library_checkpoint = copy.deepcopy(library)
                    library_checkpoint.save(f"{paths.lib_dir}/library_iter{iter}_diag_snap.json", checkpoint=True)
                    library_checkpoint.save_taxonomy(f"{paths.lib_dir}/latest_taxonomy_iter{iter}_snap.json")
                    train_tasks.save_as_json(f"{paths.train_output_dir}/train_tasks_record_iter{iter}_snap.json")
                    print(f"[Iteration {iter}] Saved library snapshot after processing {processed_count} insights")
                except Exception as e:
                    print(f"[Iteration {iter}] Warning: Failed to save snapshot: {e}")

    # Training phase
    train_start_time = time.time()
    # Create a snapshot of the library to avoid data race conditions in parallel threads
    library_snapshot = copy.deepcopy(library)
    completed = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _train_worker,
                task,
                copy.deepcopy(library_snapshot.taxonomy),  # pass a snapshot to avoid concurrent writes
                library_snapshot,  # pass library snapshot to avoid data race
                os.path.join(paths.train_output_dir, f"task_{task.id}"),     # per-task output folder
            ): (idx, task)
            for idx, task in enumerate(train_tasks)
        }

        for future in tqdm(as_completed(futures), total=len(train_tasks), desc=f"[Iteration {iter}] Failure Reflection\n"):
            idx, task = futures[future]
            new_insights, is_success, is_execution, is_verify, is_diagnosis, distribution_updates = future.result()

            if is_success is False and is_verify is not None:
                iter_verify_count += 1
                if is_verify:
                    iter_verify_success += 1

            if is_execution is False:
                fail_to_execute_lst.append(task.id)

            if is_verify is False:
                fail_to_verify_lst.append(task.id)

            if is_diagnosis is not None:
                iter_diagnose_count += 1
                if is_diagnosis is True:
                    iter_diagnose_success += 1

            train_success_flags[idx] = is_success
            for lesson in new_insights:
                lesson["source_attempts"] = [[task.id, iter, 1]]
            completed[idx] = (distribution_updates, new_insights)

    # Source attempts must be complete before checking a replacement on other tasks.
    # Input order also keeps IDs stable when worker completion order changes.
    for idx in sorted(completed):
        distribution_updates, new_insights = completed[idx]
        _apply_distribution_updates(distribution_updates)
        _process_new_insights(new_insights)
    train_duration = cal_time_cost(train_start_time, f"Iteration {iter} Failure Reflection")

    # Update llm_retri with the latest library state
    llm_retri.library = library

    # Calculate the success rate for this iteration
    number_of_train_failures = len(train_success_flags) - sum(train_success_flags)
    train_accuracy = sum(train_success_flags) / len(train_success_flags) if train_success_flags else 0

    # Calculate the self verification success rate of new generated insights
    verify_success_rate = (iter_verify_success / iter_verify_count) if iter_verify_count > 0 else 0

    # Calculate task success rate after diagnosing and resolve problematic insights (no need to generate new insights)
    diagnose_success_rate = (iter_diagnose_success / iter_diagnose_count) if iter_diagnose_count > 0 else 0

    # Calculate online merge success rate: successfully merged insights / total new insights
    online_merge_rate = (successful_merges / total_new_insights) if total_new_insights > 0 else 0

    # Record library learning success log
    iter_metrics = {
        "stage": "Failure Reflection",
        "iter": iter,
        "train_accuracy": round(train_accuracy, 3),
        "library_size": len(library),
        "principles": len(library.principles),
        "lessons": len(library.lessons),
        "number_of_train_failures": number_of_train_failures,
        "self_verify_rate": round(verify_success_rate, 3),
        "diagnose_success_rate": round(diagnose_success_rate, 3),
        "online_merge_rate": round(online_merge_rate, 3),
        "number_of_train_tasks": len(train_tasks) if train_tasks else 0,
        "fail_to_verify_task_ids": fail_to_verify_lst,
        "fail_to_execute_task_ids": fail_to_execute_lst,
        "library_diagnosis_duration (min)": train_duration
    }

    return iter_metrics
