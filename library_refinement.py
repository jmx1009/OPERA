import copy
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.auto import tqdm

from src.insight_states import INSIGHT_STATES, iter_insight_ids, normalize_distribution
from src.llm_retriever import LibraryRetrieval
from src.utils import save_log_data


def _latest_retrieved_ids(task):
    if not getattr(task, "retri_ins_lst", None):
        return set()
    return set(iter_insight_ids(task.retri_ins_lst[-1]))


def _normalize_partition(insight, tasks):
    """Use current task evidence while retaining records outside this split."""
    distribution = normalize_distribution(insight.distribution)
    current_ids = {task.id for task in tasks}
    partition = {state: [] for state in INSIGHT_STATES}
    for task in tasks:
        retrieved = insight.insight_id in _latest_retrieved_ids(task)
        status = task.output_status[-1] if task.output_status else None
        if retrieved and (status == "optimal" or
                          (isinstance(status, list) and status and all(s == "optimal" for s in status))):
            state = "useful"
        else:
            # A later exclusion is a retrieval decision, not evidence that the
            # previous assessment was wrong. Keep supported true negatives too.
            state = next((key for key in ("negative", "irrelevant", "useful", "unretrieved")
                          if task.id in distribution[key]), None)
        if state:
            partition[state].append(task.id)
    for state in INSIGHT_STATES:
        distribution[state] = [
            task_id for task_id in distribution[state] if task_id not in current_ids
        ] + partition[state]
    return partition, distribution


def matching_score(should_retrieve, should_avoid, retrieved):
    """Fraction of correctly included applicable or excluded inapplicable cases."""
    required, excluded, matched = map(set, (should_retrieve, should_avoid, retrieved))
    if required & excluded:
        raise ValueError("Retrieval evaluation sets must be disjoint")
    total = len(required) + len(excluded)
    return (len(required & matched) + len(excluded - matched)) / total if total else 0.0


def refinement_counts(library, iteration):
    counts = {kind: {"evaluated": 0, "changed": 0} for kind in ("principle", "lesson")}
    for ins in library:
        result = ins.metadata.get("retrieval_evaluation", {}).get("iterations", {}).get(str(iteration), {})
        evaluated = any(row.get("score") is not None for row in result.get("variant_scores", []))
        counts[ins.insight_type]["evaluated"] += int(evaluated)
        counts[ins.insight_type]["changed"] += int(evaluated and result["performance_gain"] > 0)
    return counts


def run_library_refinement(iter, tasks, config, llm_evolve, verbose=False, save_data=False, output_path=None, max_workers=8):
    """Refine conditions using four states and explicit applicability evidence."""
    current_partitions = {
        ins.insight_id: _normalize_partition(ins, tasks) for ins in llm_evolve.library
    }
    refined_insights = {}
    insight_distributions = {}
    evaluations = {}

    def _save_partial_aggregate():
        if save_data and output_path:
            save_log_data([
                {"insight_id": iid, "original_condition": values[0], "refined_condition": values[1]}
                for iid, values in refined_insights.items()
            ], f"{output_path}/refined_insights_iter{iter}.json")
            save_log_data(insight_distributions, f"{output_path}/refined_insight_distributions_iter{iter}.json")
            save_log_data(evaluations, f"{output_path}/refinement_retrieval_evaluation_iter{iter}.json")

    def _save_progress(insight_id, stage, **extra):
        if save_data and output_path:
            save_log_data(
                {"insight_id": insight_id, "iter": iter, "stage": stage, **extra},
                f"{output_path}/Refinement/progress_insight_{insight_id}_iter_{iter}.json"
            )

    def _usable(condition):
        return isinstance(condition, str) and bool(condition.strip())

    def _process_one_insight(insight):
        partition, distribution = current_partitions[insight.insight_id]
        useful = set(partition["useful"])
        negative = set(partition["negative"])
        irrelevant = set(partition["irrelevant"])
        unretrieved = set(partition["unretrieved"])
        neg_conditions, irrelevant_conditions, unr_conditions = [], [], []
        clause_evidence = []
        for task in tasks.subset_by_ids(partition["negative"]):
            condition = llm_evolve.generate_neg_condition(
                task, insight, iter, verbose=verbose, output_dir=output_path
            )
            if _usable(condition):
                neg_conditions.append(condition)
                clause_evidence.append({"task_id": task.id, "state": "negative", "condition": condition})

        for task in tasks.subset_by_ids(partition["irrelevant"]):
            result = llm_evolve.generate_irrelevant_condition(
                task, insight, iter, verbose=verbose, output_dir=output_path
            )
            if isinstance(result, dict) and _usable(result.get("condition")):
                irrelevant_conditions.append({"task_id": task.id, **result})
                clause_evidence.append({"task_id": task.id, "state": "irrelevant", "condition": result["condition"]})

        for task in tasks.subset_by_ids(partition["unretrieved"]):
            condition = llm_evolve.generate_unr_condition(
                task, insight, iter, verbose=verbose, output_dir=output_path
            )
            if _usable(condition):
                unr_conditions.append(condition)
                clause_evidence.append({"task_id": task.id, "state": "unretrieved", "condition": condition})
        if not (neg_conditions or irrelevant_conditions or unr_conditions):
            return None

        generation_responses = []
        proposals = llm_evolve.refine_insight(
            iter, neg_conditions, irrelevant_conditions, unr_conditions, insight,
            config.params.variant_num, verbose=verbose, output_dir=output_path, return_records=True,
            response_records=generation_responses,
        )
        if not proposals:
            return None
        candidates = [record["new_condition"] for record in proposals]
        variants = llm_evolve.build_library_variant(
            insight.insight_id, candidates, verification_only=True
        )
        required = useful | unretrieved
        excluded = negative | irrelevant
        evaluation_ids = sorted(required | excluded)
        baseline_retrieved = {task.id for task in tasks if task.id in (required | excluded)
                              and insight.insight_id in _latest_retrieved_ids(task)}
        baseline = matching_score(required, excluded, baseline_retrieved)
        best_score, best_condition = baseline, insight.condition
        best_retrieved = baseline_retrieved
        selected_variant_index = None
        variant_scores = []
        _save_progress(
            insight.insight_id, "variants_generated",
            **{f"{state}_task_count": len(partition[state]) for state in INSIGHT_STATES},
            variant_count=len(variants),
        )
        for index, library in enumerate(variants):
            _save_progress(
                insight.insight_id, "verifying_variant",
                variant_index=index + 1, variant_count=len(variants),
                evaluation_task_count=len(evaluation_ids),
            )
            retriever = LibraryRetrieval(
                lib=library, model=config.model, service=config.service, temperature=0
            )
            proposal = proposals[index]
            checks = []
            variant = {"variant_index": index + 1, "path_id": proposal["path_id"],
                       "strategy": proposal["strategy"], "condition": candidates[index], "checks": checks}
            try:
                _, matched_ids = llm_evolve.verify_retrieval(
                    insight.insight_id, tasks, evaluation_ids, retriever, checks=checks,
                )
            except (RuntimeError, ValueError) as exc:
                variant_scores.append({**variant, "score": None, "error": str(exc)})
                continue
            retrieved = set(matched_ids)
            score = matching_score(required, excluded, retrieved)
            variant_scores.append({**variant, "matched_task_ids": sorted(retrieved), "score": score})
            if score > best_score:
                best_score, best_condition, best_retrieved = score, candidates[index], retrieved
                selected_variant_index = index + 1
            _save_progress(
                insight.insight_id, "variant_verified",
                variant_index=index + 1, variant_count=len(variants),
                current_best_performance=best_score,
            )

        evaluation = {
            "insight_type": insight.insight_type,
            "iteration": iter,
            "refinement_model": llm_evolve.model,
            "generation_responses": generation_responses,
            "retrieval_model": config.model,
            "original_condition": insight.condition,
            "clause_evidence": clause_evidence,
            "baseline_matched_task_ids": sorted(baseline_retrieved),
            "baseline_score": baseline,
            "best_score": best_score,
            "selected_variant_index": selected_variant_index,
            "performance_gain": best_score - baseline,
            "variant_scores": variant_scores,
            "should_retrieve": sorted(required),
            "should_avoid": sorted(excluded),
            "irrelevant_without_clause": sorted(irrelevant - {item["task_id"] for item in irrelevant_conditions}),
            "matched_task_ids": sorted(best_retrieved),
            "recovered_unretrieved": sorted(unretrieved & best_retrieved),
        }
        _save_progress(insight.insight_id, "completed", **evaluation)
        return {
            "insight_id": insight.insight_id,
            "orig_condition": insight.condition,
            "latest_condition": best_condition,
            # Retrieval-only verification does not establish a new solving outcome.
            "distribution": distribution,
            "evaluation": evaluation,
        }

    candidates = [
        ins for ins in llm_evolve.library
        if any(current_partitions[ins.insight_id][0][state]
               for state in ("negative", "irrelevant", "unretrieved"))
    ]
    total_gain = 0.0
    refined_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(
        total=len(candidates), desc=f"[Iteration {iter}] Library Refinement"
    ) as progress:
        futures = [executor.submit(_process_one_insight, ins) for ins in candidates]
        for future in as_completed(futures):
            progress.update(1)
            try:
                result = future.result()
            except Exception:
                traceback.print_exc()
                continue
            if not result:
                continue
            refined_count += int(any(v.get("score") is not None for v in result["evaluation"]["variant_scores"]))
            iid = result["insight_id"]
            refined_insights[iid] = [result["orig_condition"], result["latest_condition"]]
            insight_distributions[iid] = result["distribution"]
            evaluations[iid] = result["evaluation"]
            total_gain += result["evaluation"]["performance_gain"]
            _save_partial_aggregate()
            print(
                f"[Iteration {iter}] Insight {iid}: matching score "
                f"{result['evaluation']['baseline_score']:.4f} -> {result['evaluation']['best_score']:.4f}"
            )

    refined_library = copy.deepcopy(llm_evolve.library)
    for insight in refined_library:
        iid = insight.insight_id
        insight.distribution = copy.deepcopy(current_partitions[iid][1])
        if iid in refined_insights:
            insight.condition = refined_insights[iid][1]
            insight.metadata.setdefault("retrieval_evaluation", {}).setdefault("iterations", {})[
                str(iter)
            ] = evaluations[iid]
    _save_partial_aggregate()
    return refined_library, total_gain / refined_count if refined_count else 0.0
