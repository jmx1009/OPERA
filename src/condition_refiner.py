from __future__ import annotations # Enable postponed (lazy) evaluation of type hints
import re
import glob
import json
import numpy as np
from typing import List, Any
import copy
from tqdm import tqdm
import traceback

from concurrent.futures import ThreadPoolExecutor, as_completed

from .insight_library import InsightLibrary
from .insight_states import normalize_reflection_record
from .utils import save_log_data, extract_json_object, call_llm_and_parse_with_retry
from .dataloader import DataLoader
from .llm_retriever import LibraryRetrieval
from .prompts.prompts_refinement import (
    PROMPT_INS_NEG,
    PROMPT_INS_UNR,
    PROMPT_INS_IRRELEVANT,
    PROMPT_INS_REFINEMENT,
)

#* Configure
from .config import training_config
config = training_config()

APPLICABILITY = "This insight applies when "
EXCLUSION = "This insight does NOT apply when "
PARAGRAPH_BREAK = re.compile(r"\r?\n[ \t]*\r?\n(?:[ \t]*\r?\n)*")


def _check_clause(text, supported):
    if not isinstance(text, str) or not any(text.startswith(prefix) for prefix in supported):
        raise ValueError("Start each added clause with the T15/T16 opening supported by its evidence")
    for prefix in (APPLICABILITY, EXCLUSION):
        if prefix in text and prefix not in supported:
            raise ValueError("No task evidence supports this clause direction")
    for body in re.split(f"{APPLICABILITY}|{EXCLUSION}", text)[1:]:
        if not re.search(r"[^\W_]", body):
            raise ValueError("A clause must identify a task feature")


def _parse_task_clause(text, direction):
    record = extract_json_object(text)
    if record == {}:
        return record
    _check_clause(record.get("condition"), {direction})
    if direction == EXCLUSION and record.get("applicable") is True:
        raise ValueError("An applicable record cannot supply an exclusion")
    if not isinstance(record.get("reason"), str) or not record["reason"].strip():
        raise ValueError("Supply the T15 reason for the clause or return an empty object")
    return record


def parse_refinement_candidates(text, original, count, *, applicability, exclusion):
    """Validate the complete T16 response before any candidate is scored."""
    def unique_fields(pairs):
        record = dict(pairs)
        if len(record) != len(pairs):
            raise ValueError("Duplicate fields in a T16 candidate")
        return record

    try:
        records = json.loads(text, object_pairs_hook=unique_fields)
    except json.JSONDecodeError as exc:
        raise ValueError("No valid JSON array of objects found. Return only the T16 candidate array.") from exc
    if not isinstance(records, list):
        raise ValueError("The T16 response must be a JSON array")
    if len(records) != count or count < 1:
        raise ValueError(f"Expected {count} T16 candidates")
    supported = set()
    if applicability:
        supported.add(APPLICABILITY)
    if exclusion:
        supported.add(EXCLUSION)
    path_ids, conditions = set(), set()
    for record in records:
        if not isinstance(record, dict) or record.keys() != {"path_id", "strategy", "new_condition"}:
            raise ValueError("Each T16 candidate must contain exactly path_id, strategy and new_condition")
        path_id = record.get("path_id")
        if type(path_id) not in (int, str) or not str(path_id).strip():
            raise ValueError("Each candidate needs a path identifier")
        key = str(path_id).strip()
        if key in path_ids:
            raise ValueError("Candidate path identifiers must be distinct")
        path_ids.add(key)
        if not isinstance(record.get("strategy"), str) or not record["strategy"].strip():
            raise ValueError("Each candidate needs a strategy")
        condition = record.get("new_condition")
        if not isinstance(condition, str) or not condition.startswith(original):
            raise ValueError("Retain the entire previous condition verbatim")
        suffix = condition[len(original):]
        separator = PARAGRAPH_BREAK.match(suffix)
        if not separator:
            raise ValueError("Append the new clauses in a separate paragraph")
        for paragraph in PARAGRAPH_BREAK.split(suffix[separator.end():].rstrip()):
            _check_clause(paragraph, supported)
        key = " ".join(condition.split())
        if key in conditions:
            raise ValueError("Candidate conditions must be distinct")
        conditions.add(key)
    return records


class ApplicabilityRefiner:
    """
    Propose applicability conditions for stored principles and lessons.
    """
    def __init__(self, lib: "InsightLibrary", model: str, service: str, temperature: float | None = None):
        self.library = lib     # Use an InsightLibrary instance
        # self.client = client
        self.service = service
        self.model = model
        self.temp = temperature

    def verify_retrieval(self, ins_id, tasks, task_ids, llm_retri, *, checks=None):
        target = llm_retri.library.retrieve_insights_by_id(ins_id)[0]
        taxonomy = target["taxonomy"]
        stages = []
        if {"Domain Modeling", "General Formulation"}.intersection(taxonomy):
            stages.append("Formulation")
        if "Code Implementation" in taxonomy:
            stages.append("Program")
        if not stages:
            raise ValueError(f"Missing generation stage for insight {ins_id}")
        matched = []
        for task in tasks.subset_by_ids(task_ids):
            for stage in stages:
                formulation = None
                if stage == "Program":
                    formulation = task.latest_attempted_formulation
                    if not formulation:
                        raise ValueError(f"No attempted formulation for program retrieval on {task.id}")
                responses = [] if checks is not None else None
                if checks is not None:
                    check = {"task_id": task.id, "stage": stage, "responses": responses}
                    checks.append(check)
                insights = llm_retri.retrieve_applicable_insights(
                    task=task, stage=stage, formulation=formulation, config=config,
                    skip_taxonomy_match=True, response_records=responses,
                )
                retrieved = any(ins["insight_id"] == ins_id for ins in insights)
                if checks is not None:
                    check["retrieved"] = retrieved
                if retrieved:
                    matched.append(task.id)
                    break
        return len(matched), matched

    def build_library_variant(self, ins_id, refined_conditions_k, verification_only=False):
        lib_variants = []
        target_insight = None
        if verification_only:
            for ins in self.library:
                if ins.insight_id == ins_id:
                    target_insight = copy.deepcopy(ins.to_dict())
                    break
            if target_insight is None:
                return lib_variants

        for ref_condition in refined_conditions_k:
            if verification_only:
                target_variant = copy.deepcopy(target_insight)
                target_variant["condition"] = ref_condition
                lib_variant = InsightLibrary(insight_list=[target_variant])
                lib_variant.taxonomy = copy.deepcopy(self.library.taxonomy)
            else:
                lib_variant = copy.deepcopy(self.library)
                for ins in lib_variant:
                    if ins.insight_id == ins_id:
                        ins.condition = ref_condition
                        break
            lib_variants.append(lib_variant)
        return lib_variants

    def _load_reflection_evidence(self, task, insight, iter, output_dir="", states=None):
        fallback = ""
        if ((not states or "irrelevant" in states)
                and task.id in insight.metadata.get("legacy_applicable_task_ids", [])):
            fallback = "Legacy reflection record: this insight was applicable and retrieved, but was not adopted. Recheck applicability against the current task description."
        if not output_dir:
            return fallback
        # Prefer the current reflection artifact; earlier checkpoints remain readable.
        records = []
        for filename in (
            f"insight_reflection_iter_{iter}.json",
            f"program_reflection_iter_{iter}.json",
            f"ins_pos_neg_diagnosis_iter_{iter}.json",
        ):
            path = f"{output_dir}/task_{task.id}/Diagnosis/{filename}"
            try:
                with open(path, encoding="utf-8") as handle:
                    records.extend(json.load(handle))
            except FileNotFoundError:
                continue
            except (OSError, ValueError):
                traceback.print_exc()
                return ""
        wanted_states = set(states or [])
        for raw_record in records:
            record = normalize_reflection_record(raw_record)
            if record.get("insight_id") != insight.insight_id:
                continue
            if wanted_states and record["state"] not in wanted_states:
                continue
            evidence = record.get("evidence") or record.get("reason") or ""
            if type(record.get("applicable")) is bool:
                evidence = f"Historical applicability annotation: {str(record['applicable']).lower()}. {evidence}"
            return evidence
        return fallback


    def generate_neg_condition(self, task, insight, iter, verbose=False, output_dir=""):
        neg_evidence = ""
        task_refinement_dir = f"{output_dir}/task_{task.id}/Refinement" if output_dir else None
        neg_evidence = self._load_reflection_evidence(
            task,
            insight,
            iter,
            output_dir=output_dir,
            states={"negative"},
        )

        target_ins = {"condition": insight.condition, "explanation": insight.explanation, "example": insight.example}
        prompt = PROMPT_INS_NEG.format(
            target_insight=json.dumps(target_ins),
            desc=task.desc,
            diag_evidence=neg_evidence
        )
        # print(prompt)

        neg_condition = []
        try:
            error_message = f"\n   insight {insight.insight_id} failed to be refined for {task.id} from LLM after maximum attempts\n"
            # Call the LLM and parse the output
            refined_result = call_llm_and_parse_with_retry(
                model=self.model,
                service=self.service,
                prompt=prompt,
                parse_fn=lambda text: _parse_task_clause(text, EXCLUSION),
                temperature=self.temp,
                max_retry=5,
                sleep_sec=0.5,
                verbose=verbose,
                error_message=error_message
            )
            # print(refined_result)

            if refined_result: # If output {} which means the insight is applicable
                neg_condition = refined_result["condition"]
                if task_refinement_dir:
                    save_log_data(
                        {
                            "task_id": task.id,
                            "insight_id": insight.insight_id,
                            "condition_type": "negative",
                            "generated_condition": neg_condition,
                            "reason": refined_result["reason"],
                            "evidence": neg_evidence,
                        },
                        f"{task_refinement_dir}/neg_condition_insight_{insight.insight_id}_iter_{iter}.json"
                    )
            elif task_refinement_dir:
                save_log_data(
                    {
                        "task_id": task.id,
                        "insight_id": insight.insight_id,
                        "condition_type": "negative",
                        "generated_condition": [],
                        "evidence": neg_evidence,
                    },
                    f"{task_refinement_dir}/neg_condition_insight_{insight.insight_id}_iter_{iter}.json"
                )

        except Exception as err:
            print(f"\n   [WARNING]: Handle malformed LLM outputs after maximum retry as no refinement for insight {insight.insight_id} on {task.id}\n")
            traceback.print_exc() # print error and cause

        return neg_condition

    @staticmethod
    def _parse_irrelevant_condition(text):
        return _parse_task_clause(text, EXCLUSION)

    def generate_irrelevant_condition(self, task, insight, iter, verbose=False, output_dir=""):
        evidence = self._load_reflection_evidence(
            task, insight, iter, output_dir=output_dir, states={"irrelevant"}
        )
        target_ins = {"condition": insight.condition, "explanation": insight.explanation, "example": insight.example}
        prompt = PROMPT_INS_IRRELEVANT.format(
            target_insight=json.dumps(target_ins), desc=task.desc, diag_evidence=evidence
        )
        result = {}
        try:
            result = call_llm_and_parse_with_retry(
                model=self.model,
                service=self.service,
                prompt=prompt,
                parse_fn=self._parse_irrelevant_condition,
                temperature=self.temp,
                max_retry=5,
                sleep_sec=0.5,
                verbose=verbose,
                error_message=f"Insight {insight.insight_id}: irrelevant-condition refinement failed for {task.id}",
            ) or {}
        except Exception:
            traceback.print_exc()
        if output_dir:
            save_log_data(
                {
                    "task_id": task.id,
                    "insight_id": insight.insight_id,
                    "condition_type": "irrelevant",
                    "generated_condition": result.get("condition", ""),
                    "reason": result.get("reason", ""),
                    "evidence": evidence,
                },
                f"{output_dir}/task_{task.id}/Refinement/irrelevant_condition_insight_{insight.insight_id}_iter_{iter}.json"
            )
        return result


    def generate_unr_condition(self, task, insight, iter, verbose=False, output_dir=""):
        # Retrieve insight diagnosis reason
        unr_evidence = ""
        task_refinement_dir = f"{output_dir}/task_{task.id}/Refinement" if output_dir else None
        try:
            files = glob.glob(f"{output_dir}/task_{task.id}/Diagnosis/applicable_insights_iter_{iter}_idx*.json")
            for fp in sorted(files):
                # print(fp)
                with open(fp, "r", encoding="utf-8") as f:
                    unr_ins_diag = json.load(f)
                    unr_ins_diag = unr_ins_diag.get("applicable_insights")
                for diag in unr_ins_diag:
                    if diag.get("insight_id") == insight.insight_id:
                        unr_evidence = diag.get("reason") or diag.get("evidence")
                        break
        except Exception:
            print(f"Didn't find the files for {task.id}!")


        target_ins = {"condition": insight.condition, "explanation": insight.explanation, "example": insight.example}
        prompt = PROMPT_INS_UNR.format(
            target_insight=json.dumps(target_ins),
            desc=task.desc,
            diag_evidence=unr_evidence
        )

        unr_condition = []
        try:
            error_message = f"\n   insight {insight.insight_id} failed to be refined for {task.id} from LLM after maximum attempts\n"
            # Call the LLM and parse the output
            refined_result = call_llm_and_parse_with_retry(
                model=self.model,
                service=self.service,
                prompt=prompt,
                parse_fn=lambda text: _parse_task_clause(text, APPLICABILITY),
                temperature=self.temp,
                max_retry=5,
                sleep_sec=0.5,
                verbose=verbose,
                error_message=error_message
            )
            # print(refined_result)

            if refined_result:
                unr_condition = refined_result["condition"]
                if task_refinement_dir:
                    save_log_data(
                        {
                            "task_id": task.id,
                            "insight_id": insight.insight_id,
                            "condition_type": "unretrieved",
                            "generated_condition": unr_condition,
                            "reason": refined_result["reason"],
                            "evidence": unr_evidence,
                        },
                        f"{task_refinement_dir}/unr_condition_insight_{insight.insight_id}_iter_{iter}.json"
                    )
            elif task_refinement_dir:
                save_log_data(
                    {
                        "task_id": task.id,
                        "insight_id": insight.insight_id,
                        "condition_type": "unretrieved",
                        "generated_condition": [],
                        "evidence": unr_evidence,
                    },
                    f"{task_refinement_dir}/unr_condition_insight_{insight.insight_id}_iter_{iter}.json"
                )

        except Exception as err:
            print(f"\n   [WARNING]: Handle malformed LLM outputs after maximum retry as no refinement for insight {insight.insight_id} on {task.id}\n")
            traceback.print_exc() # print error and cause

        return unr_condition

    def refine_insight(self, iter, neg_condition_lst, irrelevant_condition_lst, unr_condition_lst,
                       insight, path_k=5, verbose=False, output_dir="", *, return_records=False,
                       response_records=None):
        # Format insights for the prompt
        # target_ins = {k: insight.get(k) for k in ("condition", "explanation", "example")}
        target_ins = {"condition": insight.condition, "explanation": insight.explanation, "example": insight.example}

        # Format numbered lists for the prompt
        neg_conditions_str = "\n".join(f"{i}. {cond}" for i, cond in enumerate(neg_condition_lst, start=1)) or ""
        irrelevant_conditions_str = json.dumps(irrelevant_condition_lst, ensure_ascii=False)
        unr_conditions_str = "\n".join(f"{i}. {cond}" for i, cond in enumerate(unr_condition_lst, start=1)) or ""

        # Build refinement prompt
        prompt = PROMPT_INS_REFINEMENT.format(
            original_insight=json.dumps(target_ins),
            neg_conditions=neg_conditions_str,
            irrelevant_conditions=irrelevant_conditions_str,
            unr_conditions=unr_conditions_str,
            path_k=path_k
        )

        generation_error = None
        if response_records is None and output_dir:
            response_records = []
        try:
            custom_header = (f"\n==========\n[Iteration {iter}] Refine insight condition {insight.insight_id}\n==========\n")
            error_message = f"\n   insight {insight.insight_id} failed to be refined from LLM after maximum attempts\n"
            refined_results = call_llm_and_parse_with_retry(
                model=self.model,
                service=self.service,
                prompt=prompt,
                parse_fn=lambda text: parse_refinement_candidates(
                    text, insight.condition, path_k, applicability=bool(unr_condition_lst),
                    exclusion=bool(neg_condition_lst or irrelevant_condition_lst),
                ),
                temperature=self.temp,
                max_retry=5,
                sleep_sec=0.5,
                verbose=verbose,
                log_header=custom_header,
                response_records=response_records,
                error_message=error_message
            )
        except Exception as exc:
            traceback.print_exc()
            refined_results = []
            generation_error = str(exc)

        if output_dir:
            save_log_data(
                {
                    "insight_id": insight.insight_id,
                    "iteration": iter,
                    "original_condition": insight.condition,
                    "model": self.model,
                    "generation_responses": response_records,
                    "generation_error": generation_error,
                    "negative_conditions": neg_condition_lst,
                    "irrelevant_conditions": irrelevant_condition_lst,
                    "unretrieved_conditions": unr_condition_lst,
                    "refined_results": refined_results,
                },
                f"{output_dir}/Refinement/insight_{insight.insight_id}_iter_{iter}.json"
            )

        if return_records:
            return refined_results
        return [record["new_condition"] for record in refined_results]
