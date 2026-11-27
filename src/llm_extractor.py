"""Extract verified-repair candidates and prepare lesson merges for verification."""
import json
import traceback

from .prompts.prompts_lessons import (
    PROMPT_CONDUCT_MERGE, PROMPT_INS_FROM_FORMU, PROMPT_INS_FROM_PROGRAM, PROMPT_ONLINE_MERGE,
)
from .utils import call_llm_and_parse_with_retry, extract_json_array, extract_json_object, save_log_data


_RULE_FIELDS = ("taxonomy", "condition", "explanation", "example")
_FORMULATION = {"Domain Modeling", "General Formulation"}
_PROGRAM = {"Code Implementation"}


def _task_id_list(task_id):
    if isinstance(task_id, list):
        return [value for value in task_id if value is not None]
    return [] if task_id is None else [task_id]


def _dedup_preserve_order(values):
    distinct = []
    for value in values:
        if value not in distinct:
            distinct.append(value)
    return distinct


def _mapped_task_ids(mapping_ids, merged_ids, extra_task_ids=None):
    parents = [task for key in merged_ids or () for task in _task_id_list(mapping_ids.get(key))]
    return _dedup_preserve_order(parents + _task_id_list(extra_task_ids))


def _fields(record, fields=_RULE_FIELDS):
    return {name: record[name] for name in fields if name in record}


def _pretty(value):
    return json.dumps(value, indent=2, ensure_ascii=False)


def _complete_rule(record, stages):
    taxonomy = record.get("taxonomy")
    if not isinstance(taxonomy, dict) or not taxonomy or set(taxonomy) - stages:
        return False
    if not all(isinstance(groups, dict) and any(groups.values()) for groups in taxonomy.values()):
        return False
    return all(isinstance(record.get(name), str) and record[name].strip() for name in _RULE_FIELDS[1:])


def _valid_ids(ids, sources):
    return isinstance(ids, list) and all(type(value) is int and value in sources for value in ids)


def _candidate_log(rule):
    # Saved candidate files also accept a scalar leaf as a single taxonomic label.
    compact = {}
    for track, groups in rule["taxonomy"].items():
        compact[track] = {name: list(leaves) if isinstance(leaves, (dict, list)) else [leaves]
                          for name, leaves in groups.items()} if isinstance(groups, dict) else {}
    return {**rule, "taxonomy": compact}


class LessonExtractor:
    def __init__(self, model: str, service: str, temperature: float | None = None):
        self.model = model
        self.service = service
        self.temp = temperature

    def extract_lessons(self, text: str) -> list[dict]:
        try:
            return extract_json_array(text)
        except (ValueError, RecursionError):
            return [extract_json_object(text)]

    def _request(self, prompt, parser, retries, verbose, **log):
        try:
            return call_llm_and_parse_with_retry(
                model=self.model, service=self.service, prompt=prompt, parse_fn=parser,
                temperature=self.temp, max_retry=retries, sleep_sec=0.5, verbose=verbose, **log,
            )
        except Exception:
            traceback.print_exc()
            return None

    def generate_lessons(self, iter=None, task=None, corrected_program=None, failed_formulation=None,
                         taxonomy=None, verbose=True, save_data=False, output_path="learning", retrieved_insights=None):
        if failed_formulation:
            if not (task.correct_program or task.formulation):
                return []
            stage, tracks = "Formulation", _FORMULATION
            reference = task.correct_program or f"Reference mathematical formulation:\n{task.formulation}"
            prompt = PROMPT_INS_FROM_FORMU.format(
                problem_description=task.desc, failed_formulation=failed_formulation, correct_program=reference,
                domain_taxo=_pretty(taxonomy["Domain Modeling"]), formulation_taxo=_pretty(taxonomy["General Formulation"]),
            )
        elif corrected_program:
            stage, tracks = "Program", _PROGRAM
            prompt = PROMPT_INS_FROM_PROGRAM.format(corrected_program=corrected_program,
                                                    code_taxo=_pretty(taxonomy["Code Implementation"]))
        else:
            return []
        principles = [rule for rule in retrieved_insights or () if rule.get("insight_type") == "principle"]
        if principles:
            prompt += ("\nRetrieved principles used on this attempt:\n" + json.dumps(principles)
                       + "\nWhen the correction concerns a principle, explain its implementation "
                         "under the task features in the candidate lesson.")
        records = self._request(
            prompt, self.extract_lessons, 12, verbose,
            log_header=f"\n==========\n[Iteration {iter}] Extract candidate lessons for task {task.id}\n==========\n",
            error_message=f"\n   Task {task.id} failed to extract candidate lessons after maximum attempts\n",
        )
        if records is None:
            return []
        lessons = [{**_fields(rule), "insight_type": "lesson", "insight_id": -1, "task_id": task.id, "iteration": iter}
                   for rule in records if _complete_rule(rule, tracks)]
        if save_data:
            save_log_data([_candidate_log(rule) for rule in lessons], f"{output_path}/{stage}/candidate_lessons_iter_{iter}.json")
        return lessons

    def propose_lesson_merge(self, candidate_insights=None, target=None, verbose=False):
        sources = {row["insight_id"]: row["task_id"] for row in candidate_insights}
        inputs = [_fields(row, ("insight_id", *_RULE_FIELDS)) for row in candidate_insights]
        proposals = self._request(
            PROMPT_CONDUCT_MERGE.format(candidate_insights=_pretty(inputs)), extract_json_array, 8, verbose,
            log_header=f"\n==========\nMerge candidate lessons in {target}\n==========\n",
            error_message=f"\n   {target} failed to merge candidate lessons after maximum attempts\n",
        )
        merges, used = [], set()
        for rule in proposals or ():
            ids = rule.get("merged_ids")
            if not _valid_ids(ids, sources) or len(ids) < 2 or len(set(ids)) != len(ids) or used.intersection(ids):
                continue
            if _complete_rule(rule, _FORMULATION | _PROGRAM):
                merges.append({**_fields(rule, ("merged_ids", *_RULE_FIELDS)), "insight_type": "lesson",
                               "task_id": _mapped_task_ids(sources, ids)})
                used.update(ids)
        return merges

    def propose_online_lesson_merge(self, new_insight=None, library=None, verbose=False):
        taxonomy = new_insight[0].get("taxonomy", {})
        matches = library.retrieve_by_taxonomy(query_taxonomy=taxonomy, include_task_id=True)
        if not matches:
            return [], {}, []
        fields = ("insight_type", *_RULE_FIELDS)
        prompt = PROMPT_ONLINE_MERGE.format(
            new_insight=_pretty([_fields(row, fields) for row in new_insight]),
            existing_insights=_pretty([_fields(row, (*fields, "insight_id")) for row in matches]),
        )
        proposal = self._request(prompt, extract_json_object, 5, verbose)
        ids = proposal.get("merged_ids") if proposal else None
        sources = {row["insight_id"]: row["task_id"] for row in matches}
        if not ids or not _valid_ids(ids, sources):
            return [], {}, []
        rule = {**proposal, "taxonomy": taxonomy}
        if not _complete_rule(rule, set(taxonomy)):
            return [], {}, []
        ids = _dedup_preserve_order(ids)
        parents = [row for row in matches if row["insight_id"] in ids]
        new_tasks = _task_id_list(new_insight[0].get("task_id"))
        tasks = _mapped_task_ids(sources, ids, new_tasks)
        merged = {**_fields(rule), "insight_type": "lesson",
                  "related_principle_ids": [row["insight_id"] for row in parents if row["insight_type"] == "principle"],
                  "merged_ids": ids, "task_id": tasks}
        iterations = {key: row["iteration"] for row in matches for key in _task_id_list(row.get("task_id"))}
        return merged, {key: -1 if key in new_tasks else iterations[key] for key in tasks}, parents
