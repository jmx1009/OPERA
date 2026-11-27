"""Retrieve library entries for formulation, programming and failure reflection."""
import json
import traceback

from .insight_library import InsightLibrary
from .prompts.prompts_diag import PROMPT_RETRI_INS, PROMPT_RETRI_LABEL
from .prompts.prompts_retri import (
    PROMPT_FULL_CHECK_CODE, PROMPT_FULL_CHECK_MODEL, PROMPT_QUICK_MATCH_CODE, PROMPT_QUICK_MATCH_MODEL,
)
from .utils import call_llm_and_parse_with_retry, extract_json_array, extract_json_object, save_log_data


_STAGE_TRACKS = {"Formulation": ("Domain Modeling", "General Formulation"), "Program": ("Code Implementation",)}


def _selection_parser(batch):
    allowed = {row["insight_id"] for row in batch}

    def parse(text):
        chosen = extract_json_array(text)
        if any(not isinstance(row, dict) or row.get("insight_id") not in allowed for row in chosen):
            raise ValueError("Applicability response contains an unknown insight ID")
        return chosen

    return parse


def _save_matches(task, directory, suffix, taxonomy, candidates):
    save_log_data(taxonomy, f"{directory}/matched_taxo_{suffix}.json")
    record = {"task_id": task.id, "description": task.desc, "matched_insights": candidates}
    save_log_data(record, f"{directory}/matched_insights_{suffix}.json")


def _save_selection(task, directory, suffix, entries, decisions, *, hints=(), ranked=False):
    by_id = {row["insight_id"]: row for row in decisions if "insight_id" in row}
    records = []
    for entry in entries:
        evidence = by_id.get(entry["insight_id"])
        row = dict(entry)
        if ranked:
            row["ranking"] = evidence.get("ranking") if evidence else None
        row["reason"] = evidence.get("reason", "") if evidence else None
        records.append(row)
    records.extend({**hint, "reason": "Derived from retrieved document summaries"} for hint in hints)
    log = {"task_id": task.id, "description": task.desc, "applicable_insights": records}
    save_log_data(log, f"{directory}/applicable_insights_{suffix}.json")


class LibraryRetrieval:
    def __init__(self, lib: InsightLibrary, model: str, service: str, temperature: float | None = None,
                 document_retriever=None):
        self.library = lib
        self.model = model
        self.service = service
        self.temp = temperature
        self.document_retriever = document_retriever

    def has_available_knowledge(self, stage=None):
        tracks = _STAGE_TRACKS.get(stage)
        stored = bool(self.library) if tracks is None else any(
            any(track in (rule.taxonomy or {}) for track in tracks) for rule in self.library
        )
        documents = bool(self.document_retriever and self.document_retriever.has_summaries())
        return stored or documents

    def _combine_insights(self, stored_insights, task_specific_insights):
        unique = {}
        for group in (stored_insights, task_specific_insights):
            for rule in group:
                text = tuple(" ".join((rule.get(key) or "").lower().split()) for key in ("condition", "explanation"))
                taxonomy = json.dumps(rule.get("taxonomy", {}), sort_keys=True, ensure_ascii=False)
                unique.setdefault((taxonomy, *text), rule)
        return list(unique.values())

    def extract_taxonomy(self, text):
        labels = extract_json_object(text)
        if not isinstance(labels, dict):
            raise ValueError(f"The parsed JSON is not an object (dict); got {type(labels).__name__}")
        return labels

    def _taxonomy_fields(self, stage):
        def encode(name):
            return json.dumps(self.library.taxonomy[name], indent=2, ensure_ascii=False)

        if stage == "Program":
            return {"taxo": encode("Code Implementation")}
        return {"domain_taxo": encode("Domain Modeling"), "formu_taxo": encode("General Formulation")}

    def _request_labels(self, prompt, verbose, header, error):
        try:
            return call_llm_and_parse_with_retry(
                model=self.model, service=self.service, temperature=0, prompt=prompt, parse_fn=self.extract_taxonomy,
                max_retry=5, sleep_sec=5, verbose=verbose, log_header=header, error_message=error,
            )
        except Exception:
            print(error.strip())
            traceback.print_exc()
            return {}

    def _process_insights_in_batches(self, matched_insights, prompt_template, prompt_kwargs, batch_size=30,
                                     custom_header="", error_message="", verbose=False, strict=False,
                                     response_records=None):
        if not matched_insights:
            return []
        offsets = range(0, len(matched_insights), batch_size)
        matches = []
        for number, offset in enumerate(offsets, 1):
            batch = matched_insights[offset:offset + batch_size]
            arguments = {**prompt_kwargs, "candidate_insights": json.dumps(batch)}
            prompt = prompt_template.format(**arguments)
            try:
                rows = call_llm_and_parse_with_retry(
                    model=self.model, service=self.service, prompt=prompt, parse_fn=_selection_parser(batch),
                    temperature=0, max_retry=5, sleep_sec=0.5, verbose=verbose,
                    log_header=f"{custom_header}\n[Batch {number}/{len(offsets)}]",
                    error_message=f"{error_message} (Batch {number})", response_records=response_records,
                )
                if isinstance(rows, list):
                    matches.extend(rows)
            except Exception as error:
                if strict:
                    raise RuntimeError("Applicability check failed") from error
                print(f"Applicability check failed in batch {number} of {len(offsets)}.")
                traceback.print_exc()
        return matches

    def quick_match_by_taxonomy(self, iter=None, task=None, stage=None, formulation=None, verbose=False,
                                save_data=False, output_path=None):
        if stage not in _STAGE_TRACKS:
            return {}
        fields = self._taxonomy_fields(stage)
        fields["problem_description"] = task.desc
        if stage == "Program":
            fields["mathematical_model"] = formulation
            prompt = PROMPT_QUICK_MATCH_CODE.format(**fields)
        else:
            prompt = PROMPT_QUICK_MATCH_MODEL.format(**fields)
        labels = self._request_labels(
            prompt, verbose,
            f"\n==========\n[Iteration {iter}] Quickly match library insights by taxnomoy for [{stage}] generation of Task {task.id}\n==========\n",
            f"\n   [{stage}] Task {task.id} failed to extract matched insight labels after maximum attempts\n",
        )
        if not labels:
            return {}
        candidates = self.library.retrieve_by_taxonomy(query_taxonomy=labels)
        if save_data and output_path:
            _save_matches(task, f"{output_path}/{stage}", f"iter_{iter}", labels, candidates)
        return candidates

    def retrieve_applicable_insights(self, iter=None, task=None, stage=None, formulation=None, filter_fn=None,
                                     batch_size=30, config=None, verbose=False, save_data=False,
                                     output_path="learning", skip_taxonomy_match=False, response_records=None):
        attempt = dict(iter=iter, task=task, stage=stage, formulation=formulation, verbose=verbose,
                       save_data=save_data, output_path=output_path)
        hints = self.document_retriever.retrieve_applicable_insights(**attempt) if self.document_retriever is not None else []
        tracks = _STAGE_TRACKS.get(stage, _STAGE_TRACKS["Formulation"])
        rules = [rule for rule in self.library if any(key in (rule.taxonomy or {}) for key in tracks)]
        if not rules:
            return hints
        if skip_taxonomy_match or not config.ablation.taxonomy:
            candidates = [{"insight_id": rule.insight_id, "insight_type": rule.insight_type,
                           "taxonomy": rule.taxonomy, "condition": rule.condition} for rule in rules]
        else:
            labels = self.quick_match_by_taxonomy(**{**attempt, "output_path": output_path if save_data else None})
            candidates = [row for row in labels if any(key in row["taxonomy"] for key in tracks)]

        template = PROMPT_FULL_CHECK_MODEL
        fields = {"problem_description": task.desc}
        if stage == "Program":
            template = PROMPT_FULL_CHECK_CODE
            fields["mathematical_model"] = formulation
        size = getattr(getattr(config, "params", config), "retrieval_batch_size", batch_size)
        decisions = self._process_insights_in_batches(
            candidates, template, fields, batch_size=size, strict=skip_taxonomy_match,
            verbose=verbose, response_records=response_records,
            custom_header=f"\n==========\n[Iteration {iter}] Check the applicability of insights on [{stage}] for Task {task.id}\n==========\n",
            error_message=f"\n   Task {task.id} [{stage}]: failed to extract applicable insights after maximum attempts\n",
        )
        ids = [row["insight_id"] for row in decisions if "insight_id" in row]
        entries = self.library.retrieve_insights_by_id(ids, filter_fn=filter_fn)
        if save_data:
            _save_selection(task, f"{output_path}/{stage}", f"iter_{iter}", entries, decisions, hints=hints)
        return self._combine_insights(entries, hints)

    def retrieve_insights_for_diagnosis(self, iter=None, task=None, formulation=None, diagnosed_issues=(),
                                        filter_fn=None, batch_size=30, verbose=False, save_data=False,
                                        output_path="learning"):
        by_issue = {}
        for issue in diagnosed_issues:
            fields = dict(problem_description=task.desc, failed_formulation=formulation,
                          one_diagnosed_issue=json.dumps(issue))
            prompt = PROMPT_RETRI_LABEL.format(**fields, **self._taxonomy_fields("Formulation"))
            labels = self._request_labels(
                prompt, verbose,
                f"\n==========\n[Iteration {iter}] [Unretrieved Insight] Quickly match library insights by taxnomoy for Task {task.id}\n==========\n for Task {task.id}\n==========\n",
                f"\n   Task {task.id} failed to extract labels for unretrieved insights after maximum attempts\n",
            )
            if not labels:
                continue
            labels = {key: value for key, value in labels.items() if key in _STAGE_TRACKS["Formulation"]}
            candidates = self.library.retrieve_by_taxonomy(query_taxonomy=labels, filter_fn=filter_fn)
            decisions = self._process_insights_in_batches(
                candidates, PROMPT_RETRI_INS, fields, batch_size=batch_size, verbose=verbose,
                custom_header=f"\n==========\n[Iteration {iter}] [Unretrieved Insight] Check applicability for Task {task.id}\n==========\n",
                error_message=f"\n   Task {task.id} [Diagnosis]: Failed to extract applicable insights after maximum attempts\n",
            )
            entries = self.library.retrieve_insights_by_id([row["insight_id"] for row in decisions if "insight_id" in row])
            issue_id = issue.get("id")
            by_issue[issue_id] = entries
            if save_data:
                directory, suffix = f"{output_path}/Diagnosis", f"iter_{iter}_idx_{issue_id}"
                _save_matches(task, directory, suffix, labels, candidates)
                _save_selection(task, directory, suffix, entries, decisions, ranked=True)
        return by_issue
