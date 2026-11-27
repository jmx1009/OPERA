import os
import time
import copy
import numpy as np
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils import cal_time_cost
from src.train_eval_utils import check_optimality, generate_solution_with_retrieval, self_verify_test

from src.lesson_merge import consolidate_lessons, integrate_lesson


def run_library_online_learning(
    iter,
    train_tasks,
    llm_retri, llm_opt, llm_diag, llm_ins, library,
    params,
    paths
):
    """
    Run library learning phase on multithread
    """

    # Global counters for online merge rate calculation
    total_online_merge_attempts = 0
    total_online_merge_successes = 0
    def _train_worker(task, taxo_snapshot, train_output_path):
        """
        Parallelize pipeline on the minibatch of tasks (insight retrieval -> formulation generation -> insight retrieval -> program generation -> check optimality -> lesson extraction -> lesson verification) for training tasks
        Return: (new lessons, is_success, is_execution, is_verify)
        """
        failures = []
        statuses, retrieved_ids = [], []
        for attempt in range(1, params.max_solution_attempts + 1):
            context, form, program, output, runnable, timed_out = generate_solution_with_retrieval(
                iter, task, library, llm_retri, llm_opt, output_path=train_output_path,
            )
            if program:
                optimal, status, feedback = check_optimality(task, output, runnable, timed_out)
                if not optimal:
                    failures.append((attempt, status, feedback, form, program, context))
            else:
                status = "parse_error"
            statuses.append(status)
            retrieved_ids.extend(ins["insight_id"] for ins in context if "insight_id" in ins)
            task.record_attempt(iter, attempt, status, form, context)

        task.output_status.append(statuses)
        task.retri_ins_lst.append(list(dict.fromkeys(retrieved_ids)))
        task.confidence = statuses.count("optimal") / len(statuses)
        if all(status == "optimal" for status in statuses):
            task.success_count += 1
            return [], True, None, None

        lessons, processed = [], False
        for attempt, status, feedback, form, program, context in failures:
            correction, code_fixed_task = None, False
            if status == "run_error":
                code_fixed_task, runnable, correction, new_feedback = llm_diag.diagnose_program(
                    iter=iter, task=task, failed_program=program, feedback=feedback,
                    save_data=True, output_path=train_output_path,
                )
                if not runnable:
                    continue
                feedback = new_feedback or feedback
            if not correction and not (task.formulation or task.correct_program):
                continue
            processed = True
            for _ in range(params.max_verify_attempts):
                extract_args = dict(iter=iter, task=task, taxonomy=taxo_snapshot, retrieved_insights=context,
                                    verbose=False, save_data=True, output_path=train_output_path)
                candidates = llm_ins.generate_lessons(corrected_program=correction, **extract_args) if correction else []
                if not code_fixed_task:
                    candidates += llm_ins.generate_lessons(failed_formulation=form, **extract_args)
                if self_verify_test(iter, task, llm_opt, candidates, context, True, train_output_path):
                    for candidate in candidates:
                        candidate["source_attempts"] = [[task.id, iter, attempt]]
                    lessons.extend(candidates)
                    break
        if not processed:
            task.fail_to_execute += 1
            return [], False, False, None
        if not lessons:
            task.fail_to_verify += 1
        lessons, _, _ = consolidate_lessons(lessons, [task], library, llm_opt, llm_ins, f"Task {task.id}")
        return lessons, False, True, bool(lessons)

    # Experiment metrics
    train_success_flags = [False] * len(train_tasks)

    fail_to_execute_lst = []

    iter_verify_count = 0
    iter_verify_success = 0
    fail_to_verify_lst = []


    batch_verified_merge_rate = []
    batch_insight_merge_rate = []

    train_start_time = time.time()

    for start in range(0, len(train_tasks), params.batch_size):
        batch = train_tasks[start:start + params.batch_size]
        batch_results = {}
        batch_train_start_time = time.time()
        with ThreadPoolExecutor(max_workers=params.batch_size) as executor:
            futures = {
                executor.submit(
                    _train_worker,
                    task,
                    copy.deepcopy(library.taxonomy),                # pass a snapshot to avoid concurrent writes
                    os.path.join(paths.train_output_dir, f"task_{task.id}")     # per-task output folder
                ): (start+i, task)
                for i, task in enumerate(batch)
            }

            batch_idx = start // params.batch_size + 1
            for future in tqdm(as_completed(futures), total=len(batch), desc=f"[Iteration {iter}] Experience Library Construction Batch {batch_idx} (tasks {start+1}-{start+len(batch)}) \n"):
                # (start+i, task)
                idx, task = futures[future]
                new_insights, is_success, is_execution, is_verify = future.result()

                if is_success is False and is_verify is not None:
                    iter_verify_count += 1
                    if is_verify:
                        iter_verify_success += 1

                if is_execution is False:
                    fail_to_execute_lst.append(task.id)

                if is_verify is False:
                    fail_to_verify_lst.append(task.id)

                batch_results[idx] = new_insights

                train_success_flags[idx] = is_success

        #* Once this batch is completed, verified lessons enter the experience library
        batch_new_insights = [lesson for idx in sorted(batch_results) for lesson in batch_results[idx]]
        if batch_new_insights:
            print(f"Batch {batch_idx} has {len(batch_new_insights)} verified lessons to consider for merging.")
            consolidated, proposed_count, accepted_count = consolidate_lessons(
                batch_new_insights, batch, library, llm_opt, llm_ins, f"Batch {batch_idx}"
            )
            total_online_merge_attempts += len(consolidated)
            for i, lesson in enumerate(consolidated):
                accepted = integrate_lesson(lesson, consolidated[i+1:], train_tasks, library,
                                            llm_opt, llm_ins, iter)
                total_online_merge_successes += int(accepted)
            if proposed_count:
                batch_verified_merge_rate.append(accepted_count / proposed_count)
            batch_insight_merge_rate.append(1 - len(consolidated) / len(batch_new_insights))

            # Periodically save library snapshot to prevent data loss
            library.save(f"{paths.lib_dir}/library_base_snap.json", checkpoint=True)
            library.save_taxonomy(f"{paths.lib_dir}/latest_taxonomy_base_snap.json")
            train_tasks.save_as_json(f"{paths.train_output_dir}/train_tasks_record_base_snap.json")

        cal_time_cost(batch_train_start_time, f'Iteration {iter} Experience Library Construction Batch {batch_idx} [{start+1}-{start+len(batch)}]')

    train_duration = cal_time_cost(train_start_time, f'Iteration {iter} Experience Library Construction')

    # Calculate the success rate for this iteration
    number_of_train_failures = len(train_success_flags) - sum(train_success_flags)
    train_accuracy = sum(train_success_flags) / len(train_success_flags) if train_success_flags else 0
    verify_success_rate = (iter_verify_success / iter_verify_count) if iter_verify_count > 0 else 0

    # Empty merge groups contribute no verified replacement.
    verified_merge_rate = float(np.mean(batch_verified_merge_rate)) if batch_verified_merge_rate else 0.0
    insight_merge_rate = float(np.mean(batch_insight_merge_rate)) if batch_insight_merge_rate else 0.0

    # Calculate overall online merge rate
    overall_online_merge_rate = (total_online_merge_successes / total_online_merge_attempts) if total_online_merge_attempts > 0 else 0

    # Record library learning success log
    iter_metrics = {
        "stage": "Experience Library Construction",
        "iter": iter,
        "train_accuracy": round(train_accuracy, 3),
        "library_size": len(library),
        "principles": len(library.principles),
        "lessons": len(library.lessons),
        "number_of_train_failures": number_of_train_failures,
        "self_verify_rate": round(verify_success_rate, 3),
        "insight_merge_rate": round(insight_merge_rate, 3),
        "verified_merge_rate": round(verified_merge_rate, 3),
        "online_merge_rate": round(overall_online_merge_rate, 3),
        "number_of_train_tasks": len(train_tasks),
        "fail_to_execute_task_ids": fail_to_execute_lst,
        "fail_to_verify_task_ids": fail_to_verify_lst,
        "online_learning_duration (min)": train_duration,
    }

    return iter_metrics
