from __future__ import annotations

import json
import math
import os
import traceback
from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import validate_model
from .domain_corpus import discover_domain_documents
from .insight_library import InsightLibrary
from .prompts.prompts_principles import (
    PROMPT_DOCUMENT_SUMMARY,
    PROMPT_CANDIDATE_PRINCIPLES,
    PROMPT_TASK_SPECIFIC_INSIGHTS,
)
from .utils import (
    call_llm_and_parse_with_retry,
    extract_json_array,
    save_log_data,
)

MODELING_KEYWORDS = {
    "variable",
    "variables",
    "objective",
    "constraint",
    "constraints",
    "binary",
    "integer",
    "continuous",
    "linearization",
    "facility",
    "location",
    "assignment",
    "routing",
    "route",
    "schedule",
    "scheduling",
    "capacity",
    "queue",
    "queueing",
    "charging",
    "energy",
    "soc",
    "battery",
    "flow",
    "equilibrium",
    "airspace",
    "vertiport",
    "uav",
    "evtol",
    "time window",
    "robust",
    "stochastic",
    "decomposition",
    "callback",
    "piecewise",
    "indicator",
    "big-m",
    "big m",
    "payload",
    "uncertainty",
    "fleet",
    "demand",
    "service",
    "depot",
}

LOW_VALUE_PATTERNS = {
    "is important",
    "are important",
    "plays an important role",
    "plays a vital role",
    "is widely used",
    "are widely used",
    "future research",
    "future direction",
    "this paper reviews",
    "this study reviews",
    "this article reviews",
    "the paper discusses",
    "the study discusses",
    "sustainable and smart cities",
    "broadly speaking",
    "in recent years",
}


def _extract_plain_text(text: str) -> str:
    text = _coerce_text(text)
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return "\n".join(
            part
            for item in value
            if (part := _coerce_text(item).strip())
        )
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _truncate(text: str, max_chars: int = 1800) -> str:
    text = _coerce_text(text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _normalize_signature(text: str) -> str:
    return " ".join(_coerce_text(text).lower().split())


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _count_keyword_hits(text: str, keywords: set[str]) -> int:
    lowered = (text or "").lower()
    return sum(1 for keyword in keywords if keyword in lowered)


VALID_TAXONOMY_STAGES = ("Domain Modeling", "General Formulation", "Code Implementation")
_VALID_TAXONOMY_STAGE_SET = set(VALID_TAXONOMY_STAGES)
TAXONOMY_STAGE_DRIFT_MAP = {
    "constraint formulation": "General Formulation",
    "variable definition": "General Formulation",
    "objective specification": "General Formulation",
    "decomposition strategy": "General Formulation",
    "solver & api syntax": "Code Implementation",
    "solver api syntax": "Code Implementation",
    "numerical stability": "Code Implementation",
    "facility location": "Domain Modeling",
    "network flow": "Domain Modeling",
    "routing": "Domain Modeling",
    "scheduling": "Domain Modeling",
    "resource allocation": "Domain Modeling",
}


def _find_case_insensitive_key(mapping: dict, key: str) -> Optional[str]:
    key_norm = str(key).casefold()
    for existing_key in mapping.keys():
        if str(existing_key).casefold() == key_norm:
            return existing_key
    return None


def _merge_taxonomy_branch(stage_branch: dict, level1_name: str, level2_spec: Any) -> None:
    level1_key = _find_case_insensitive_key(stage_branch, level1_name) or level1_name
    if level1_key not in stage_branch:
        stage_branch[level1_key] = deepcopy(level2_spec) if isinstance(level2_spec, (dict, list)) else {}
        return

    existing_spec = stage_branch[level1_key]
    if isinstance(existing_spec, dict):
        if isinstance(level2_spec, dict):
            for level2_name, info in level2_spec.items():
                existing_l2 = _find_case_insensitive_key(existing_spec, level2_name)
                if existing_l2 is None:
                    existing_spec[level2_name] = deepcopy(info)
        elif isinstance(level2_spec, list):
            for level2_name in level2_spec:
                existing_l2 = _find_case_insensitive_key(existing_spec, level2_name)
                if existing_l2 is None:
                    existing_spec[level2_name] = {}
    elif isinstance(existing_spec, list):
        existing_norm = {str(item).casefold() for item in existing_spec}
        if isinstance(level2_spec, dict):
            incoming_names = level2_spec.keys()
        elif isinstance(level2_spec, list):
            incoming_names = level2_spec
        else:
            incoming_names = []
        for level2_name in incoming_names:
            if str(level2_name).casefold() not in existing_norm:
                existing_spec.append(level2_name)
                existing_norm.add(str(level2_name).casefold())


def _resolve_drifted_taxonomy_stage(stage_label: str, reference_taxonomy: Optional[dict] = None) -> Optional[str]:
    mapped_stage = TAXONOMY_STAGE_DRIFT_MAP.get(str(stage_label).casefold())
    if mapped_stage:
        return mapped_stage

    if not isinstance(reference_taxonomy, dict):
        return None

    matches = []
    for stage in VALID_TAXONOMY_STAGES:
        stage_branch = reference_taxonomy.get(stage)
        if isinstance(stage_branch, dict) and _find_case_insensitive_key(stage_branch, stage_label):
            matches.append(stage)
    return matches[0] if len(matches) == 1 else None


def _normalize_taxonomy_stage_drift(taxonomy: Any, reference_taxonomy: Optional[dict] = None) -> dict:
    if not isinstance(taxonomy, dict):
        return {}

    normalized: dict[str, dict] = {}
    for stage_label, stage_value in taxonomy.items():
        if stage_label in _VALID_TAXONOMY_STAGE_SET:
            if isinstance(stage_value, dict):
                normalized.setdefault(stage_label, {})
                for level1_name, level2_spec in stage_value.items():
                    _merge_taxonomy_branch(normalized[stage_label], level1_name, level2_spec)
            continue

        target_stage = _resolve_drifted_taxonomy_stage(stage_label, reference_taxonomy)
        if target_stage is None:
            continue

        normalized.setdefault(target_stage, {})
        if isinstance(stage_value, dict):
            _merge_taxonomy_branch(normalized[target_stage], stage_label, stage_value)
        elif isinstance(stage_value, list):
            _merge_taxonomy_branch(normalized[target_stage], stage_label, stage_value)
        else:
            _merge_taxonomy_branch(normalized[target_stage], stage_label, {})

    return {stage: branch for stage, branch in normalized.items() if branch}


def _has_valid_taxonomy_shape(taxonomy: Any) -> bool:
    if not isinstance(taxonomy, dict) or not taxonomy:
        return False
    return any(
        stage in taxonomy and isinstance(taxonomy.get(stage), dict) and bool(taxonomy.get(stage))
        for stage in VALID_TAXONOMY_STAGES
    )


def _is_actionable_insight_candidate(insight: dict) -> bool:
    if not isinstance(insight, dict):
        return False
    if not _has_valid_taxonomy_shape(insight.get("taxonomy")):
        return False

    condition = _coerce_text(insight.get("condition")).strip()
    explanation = _coerce_text(insight.get("explanation")).strip()
    example = _coerce_text(insight.get("example")).strip()
    if not condition or not explanation or not example:
        return False
    if len(condition.split()) < 8 or len(explanation.split()) < 10:
        return False

    combined = " ".join(part for part in [condition, explanation, example] if part)
    modeling_hits = _count_keyword_hits(combined, MODELING_KEYWORDS)
    low_value_hits = _count_keyword_hits(combined, LOW_VALUE_PATTERNS)
    explanation_lower = explanation.lower()

    if explanation_lower.startswith(("this paper", "this study", "this article", "the paper", "the study")):
        return False
    if modeling_hits == 0:
        return False
    if low_value_hits >= 2 and modeling_hits < 3:
        return False

    return True


@dataclass
class PrincipleSources:
    document_index_path: str
    candidate_principles_path: str
    candidate_taxonomy_path: str


def _hierarchy_settings(config: Any) -> dict[str, Any]:
    bootstrap = getattr(config, "bootstrap", None)
    if bootstrap is None:
        return {}

    return {
        "corpus_root": str(getattr(bootstrap, "corpus_root", "")),
        "chunk_words": int(getattr(bootstrap, "chunk_words", 500)),
        "overlap_words": int(getattr(bootstrap, "overlap_words", 75)),
        "leaf_text_chars": int(getattr(bootstrap, "leaf_text_chars", 2400)),
        "summary_group_size": int(getattr(bootstrap, "summary_group_size", 5)),
        "extraction_max_level": int(getattr(bootstrap, "extraction_max_level", 2)),
        "max_level1_units_per_doc": int(getattr(bootstrap, "max_level1_units_per_doc", 8)),
        "max_level2_units_per_doc": int(getattr(bootstrap, "max_level2_units_per_doc", 5)),
        "long_doc_chunk_threshold": int(getattr(bootstrap, "long_doc_chunk_threshold", 120)),
        "source_unit_selection_method": str(getattr(bootstrap, "source_unit_selection_method", "modeling_density_diversity")),
        "max_candidate_principles_per_summary": int(getattr(bootstrap, "max_candidate_principles_per_summary", 5)),
    }


def _index_matches_settings(config: Any, document_index_path: str) -> bool:
    if not os.path.exists(document_index_path):
        return False

    try:
        with open(document_index_path, "r", encoding="utf-8") as f:
            document_index = json.load(f)
    except Exception:
        traceback.print_exc()
        return False

    build_config = document_index.get("build_config")
    if not isinstance(build_config, dict):
        return False

    expected = _hierarchy_settings(config)
    return all(build_config.get(key) == value for key, value in expected.items())


class DocumentIndex:
    def __init__(self, data: dict):
        self.data = data
        self.units = data.get("units", [])
        self._retrieval_rows = [
            unit for unit in self.units
            if unit.get("retrieval_text") and unit.get("kind") == "summary"
        ]
        retrieval_texts = [row.get("retrieval_text", "") for row in self._retrieval_rows]
        if retrieval_texts:
            self._vectorizer = TfidfVectorizer(stop_words="english", max_features=4096)
            self._matrix = self._vectorizer.fit_transform(retrieval_texts)
        else:
            self._vectorizer = None
            self._matrix = None

    @classmethod
    def load(cls, path: str) -> "DocumentIndex":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    def has_summaries(self) -> bool:
        return bool(self._retrieval_rows)

    def retrieve_units(self, query: str, top_k: int = 6) -> list[dict]:
        if not self._vectorizer or self._matrix is None or not self._retrieval_rows:
            return []

        query_vec = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self._matrix)[0]
        ranked_indices = similarities.argsort()[::-1]

        selected: list[dict] = []
        seen_ids: set[str] = set()
        for idx in ranked_indices[:top_k]:
            unit = dict(self._retrieval_rows[idx])
            unit["score"] = float(similarities[idx])
            unit_id = unit.get("unit_id")
            if unit_id and unit_id not in seen_ids:
                selected.append(unit)
                seen_ids.add(unit_id)

        return selected


class DocumentRetriever:
    def __init__(
        self,
        index: Optional[DocumentIndex],
        model: str,
        service: str,
        taxonomy: Optional[dict],
        max_units: int = 6,
        max_insights: int = 4,
    ):
        self.index = index
        self.model = model
        self.service = service
        self.taxonomy = taxonomy or {}
        self.max_units = max_units
        self.max_insights = max_insights

    def has_summaries(self) -> bool:
        return self.index is not None and self.index.has_summaries()

    def retrieve_applicable_insights(
        self,
        task,
        stage: str,
        formulation: Optional[str] = None,
        iter: Optional[int] = None,
        save_data: bool = False,
        output_path: Optional[str] = None,
        verbose: bool = False,
    ) -> list[dict]:
        if not self.has_summaries():
            return []

        query = task.desc if stage == "Formulation" else f"{task.desc}\n\n{formulation or ''}"
        source_summaries = self.index.retrieve_units(
            query=query,
            top_k=self.max_units,
        )
        if not source_summaries:
            return []

        prompt = PROMPT_TASK_SPECIFIC_INSIGHTS.format(
            stage=stage,
            problem_description=task.desc,
            mathematical_model=formulation or "N/A",
            source_summaries=_json_dumps(
                [
                    {
                        "unit_id": unit.get("unit_id"),
                        "title": unit.get("title"),
                        "kind": unit.get("kind"),
                        "summary": unit.get("summary") or unit.get("retrieval_text"),
                        "source_refs": unit.get("source_refs", []),
                    }
                    for unit in source_summaries
                ]
            ),
            max_insights=self.max_insights,
        )

        try:
            raw_insights = call_llm_and_parse_with_retry(
                model=self.model,
                service=self.service,
                prompt=prompt,
                parse_fn=extract_json_array,
                temperature=0,
                max_retry=5,
                sleep_sec=1,
                verbose=verbose,
                error_message=f"Failed to derive task-specific insights for task {task.id} [{stage}]",
            )
        except Exception as err:

            traceback.print_exc()
            raw_insights = []

        task_specific_insights = []
        for idx, insight in enumerate(raw_insights or []):
            if isinstance(insight, dict):
                insight = {
                    **insight,
                    "taxonomy": _normalize_taxonomy_stage_drift(insight.get("taxonomy"), self.taxonomy),
                }
            if not _is_actionable_insight_candidate(insight):
                continue
            condition = _coerce_text(insight.get("condition")).strip()
            explanation = _coerce_text(insight.get("explanation")).strip()
            example = _coerce_text(insight.get("example")).strip()
            allowed = {"Code Implementation"} if stage == "Program" else {"Domain Modeling", "General Formulation"}
            if not allowed.intersection(insight["taxonomy"]):
                continue
            task_specific_insights.append(
                {
                    "taxonomy": insight.get("taxonomy", {}),
                    "condition": condition,
                    "explanation": explanation,
                    "example": example,
                    "source_ref": insight.get("source_ref"),
                    "memory_id": f"task_specific_{task.id}_{stage}_{idx}",
                    "source_type": "document_summary",
                }
            )

        task_specific_insights = task_specific_insights[:self.max_insights]
        if save_data and output_path:
            stage_dir = os.path.join(output_path, stage)
            save_log_data(source_summaries, os.path.join(stage_dir, f"retrieved_summaries_iter_{iter}.json"))
            save_log_data(task_specific_insights, os.path.join(stage_dir, f"task_specific_insights_iter_{iter}.json"))

        return task_specific_insights


class PrincipleLibraryBuilder:
    def __init__(self, config: Any, model: str, service: str):
        validate_model(model, service)
        self.config = config
        self.bootstrap = getattr(config, "bootstrap", None)
        self.model = model
        self.service = service
        if self.bootstrap is None:
            raise ValueError("The principle library builder requires document settings")

        self.corpus_root = self.bootstrap.corpus_root
        self.document_index_path = self.bootstrap.document_index_path
        self.candidate_principles_path = self.bootstrap.candidate_principles_path
        self.candidate_taxonomy_path = self.bootstrap.candidate_taxonomy_path
        self.chunk_words = getattr(self.bootstrap, "chunk_words", 500)
        self.overlap_words = getattr(self.bootstrap, "overlap_words", 75)
        self.leaf_text_chars = getattr(self.bootstrap, "leaf_text_chars", 2400)
        self.cluster_target_size = getattr(self.bootstrap, "summary_group_size", 5)
        if self.cluster_target_size < 2:
            raise ValueError("Each summary group must allow at least two units.")
        self.extraction_max_level = getattr(self.bootstrap, "extraction_max_level", 2)
        self.max_level1_units_per_doc = getattr(self.bootstrap, "max_level1_units_per_doc", 8)
        self.max_level2_units_per_doc = getattr(self.bootstrap, "max_level2_units_per_doc", 5)
        self.long_doc_chunk_threshold = getattr(self.bootstrap, "long_doc_chunk_threshold", 120)
        self.source_unit_selection_method = getattr(
            self.bootstrap,
            "source_unit_selection_method",
            "modeling_density_diversity",
        )
        self.max_candidate_principles_per_summary = getattr(self.bootstrap, "max_candidate_principles_per_summary", 5)

    def _summarize_cluster(self, title: str, category: str, level: int, excerpts: list[str]) -> str:
        prompt = PROMPT_DOCUMENT_SUMMARY.format(
            title=title,
            category=category,
            level=level,
            excerpts="\n\n".join(f"[Excerpt {idx + 1}]\n{_truncate(text, 2500)}" for idx, text in enumerate(excerpts)),
        )
        return call_llm_and_parse_with_retry(
            model=self.model,
            service=self.service,
            prompt=prompt,
            parse_fn=_extract_plain_text,
            temperature=0,
            max_retry=5,
            sleep_sec=1,
            verbose=False,
            error_message=f"Failed to summarize cluster for {title}",
        )

    def _cluster_nodes(self, nodes: list[dict]) -> list[list[dict]]:
        if len(nodes) <= self.cluster_target_size:
            return [nodes]

        texts = [node.get("retrieval_text", "") for node in nodes]
        vectorizer = TfidfVectorizer(stop_words="english", max_features=2048)
        matrix = vectorizer.fit_transform(texts)
        n_clusters = max(1, math.ceil(len(nodes) / self.cluster_target_size))
        n_clusters = min(n_clusters, len(nodes))

        if n_clusters <= 1:
            return [nodes]

        labels = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit_predict(matrix)
        grouped: dict[int, list[dict]] = defaultdict(list)
        for node, label in zip(nodes, labels):
            grouped[int(label)].append(node)

        ordered_groups = sorted(
            grouped.values(),
            key=lambda group: min(item.get("order", 0) for item in group),
        )
        return ordered_groups

    def _unit_text(self, unit: dict) -> str:
        return _coerce_text(unit.get("summary") or unit.get("retrieval_text")).strip()

    def _unit_modeling_density_score(self, unit: dict) -> float:
        text = self._unit_text(unit)
        if not text:
            return 0.0

        keyword_hits = _count_keyword_hits(text, MODELING_KEYWORDS)
        words = max(1, len(text.split()))
        return keyword_hits / math.sqrt(words)

    def _display_source_title(self, unit: dict) -> str:
        base_title = unit.get("title", unit.get("doc_id", "document"))
        if unit.get("is_document_root"):
            return f"{base_title} [root summary]"

        level = unit.get("level", 0)
        order = unit.get("order", 0)
        return f"{base_title} [level {level} summary #{order}]"

    def _select_representative_units(self, units: list[dict], max_units: int) -> list[dict]:
        if max_units <= 0 or not units:
            return []

        unique_units = []
        seen_ids: set[str] = set()
        for unit in units:
            unit_id = unit.get("unit_id")
            if not unit_id or unit_id in seen_ids:
                continue
            seen_ids.add(unit_id)
            unique_units.append(unit)

        if len(unique_units) <= max_units:
            return sorted(unique_units, key=lambda unit: (unit.get("order", 0), unit.get("unit_id", "")))

        scores = [self._unit_modeling_density_score(unit) for unit in unique_units]
        ranked_indices = sorted(
            range(len(unique_units)),
            key=lambda idx: (-scores[idx], unique_units[idx].get("order", 0), unique_units[idx].get("unit_id", "")),
        )

        if self.source_unit_selection_method != "modeling_density_diversity":
            selected = [unique_units[idx] for idx in ranked_indices[:max_units]]
            return sorted(selected, key=lambda unit: (unit.get("order", 0), unit.get("unit_id", "")))

        texts = [self._unit_text(unit) or unit.get("unit_id", "") for unit in unique_units]
        similarity_matrix = None
        try:
            vectorizer = TfidfVectorizer(stop_words="english", max_features=2048)
            matrix = vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(matrix)
        except ValueError:
            similarity_matrix = None

        max_score = max(scores) if scores else 0.0
        normalized_scores = [score / max_score if max_score > 0 else 0.0 for score in scores]

        selected_indices: list[int] = []
        remaining = set(range(len(unique_units)))
        while remaining and len(selected_indices) < max_units:
            if not selected_indices:
                best_idx = ranked_indices[0]
            else:
                best_idx = None
                best_value = None
                for idx in remaining:
                    redundancy = 0.0
                    if similarity_matrix is not None and selected_indices:
                        redundancy = max(float(similarity_matrix[idx][chosen]) for chosen in selected_indices)
                    mmr_value = 0.75 * normalized_scores[idx] - 0.25 * redundancy
                    candidate = (
                        mmr_value,
                        normalized_scores[idx],
                        -unique_units[idx].get("order", 0),
                    )
                    if best_value is None or candidate > best_value:
                        best_value = candidate
                        best_idx = idx

            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        selected = [unique_units[idx] for idx in selected_indices]
        return sorted(selected, key=lambda unit: (unit.get("order", 0), unit.get("unit_id", "")))

    def _select_document_source_units(
        self,
        document,
        root_node: dict,
        summary_nodes_by_level: dict[int, list[dict]],
        *,
        max_level: int,
    ) -> list[dict]:
        selected: list[dict] = []
        seen_ids: set[str] = set()

        def add_unit(unit: dict, role: str) -> None:
            unit_id = unit.get("unit_id")
            if not unit_id or unit_id in seen_ids:
                return
            seen_ids.add(unit_id)
            unit.setdefault("source_roles", []).append(role)
            selected.append(unit)

        add_unit(root_node, role="document_root")

        if max_level >= 1:
            level1_candidates = [
                unit
                for unit in summary_nodes_by_level.get(1, [])
                if unit.get("unit_id") != root_node.get("unit_id")
            ]
            for unit in self._select_representative_units(level1_candidates, self.max_level1_units_per_doc):
                add_unit(unit, role="subtopic_level_1")

        if max_level >= 2 and len(document.chunks) >= self.long_doc_chunk_threshold:
            level2_candidates = [
                unit
                for unit in summary_nodes_by_level.get(2, [])
                if unit.get("unit_id") != root_node.get("unit_id")
            ]
            for unit in self._select_representative_units(level2_candidates, self.max_level2_units_per_doc):
                add_unit(unit, role="subtopic_level_2")

        return selected

    def _extract_candidate_principles(
        self,
        title: str,
        category: str,
        summary: str,
        taxonomy: dict,
        *,
        source_ref: str,
        source_unit_id: str,
        source_unit_kind: str,
    ) -> list[dict]:
        prompt = PROMPT_CANDIDATE_PRINCIPLES.format(
            title=title,
            category=category,
            summary=_truncate(summary, 5000),
            taxonomy=_json_dumps(taxonomy),
            max_insights=self.max_candidate_principles_per_summary,
        )
        try:
            insights = call_llm_and_parse_with_retry(
                model=self.model,
                service=self.service,
                prompt=prompt,
                parse_fn=extract_json_array,
                temperature=0.2,
                max_retry=5,
                sleep_sec=1,
                verbose=False,
                error_message=f"Failed to extract candidate principles for {title}",
            )
        except Exception as err:

            traceback.print_exc()
            return []

        normalized = []
        for ins in insights or []:
            if isinstance(ins, dict):
                ins = {
                    **ins,
                    "taxonomy": _normalize_taxonomy_stage_drift(ins.get("taxonomy"), taxonomy),
                }
            if not _is_actionable_insight_candidate(ins):
                continue
            condition = _coerce_text(ins.get("condition")).strip()
            explanation = _coerce_text(ins.get("explanation")).strip()
            example = _coerce_text(ins.get("example")).strip()
            normalized.append(
                {
                    "taxonomy": ins["taxonomy"],
                    "condition": condition,
                    "explanation": explanation,
                    "example": example,
                    "insight_type": "principle",
                    "source_title": title,
                    "source_category": category,
                    "source_ref": source_ref,
                    "source_unit_id": source_unit_id,
                    "source_unit_kind": source_unit_kind,
                }
            )
        return normalized[:self.max_candidate_principles_per_summary]

    def _build_document_hierarchy(
        self,
        document,
        *,
        units: list[dict],
    ) -> tuple[dict, dict[int, list[dict]]]:
        leaf_nodes = []
        summary_nodes_by_level: dict[int, list[dict]] = defaultdict(list)

        for chunk in document.chunks:
            unit_id = f"{document.doc_id}_{chunk.chunk_id}"
            leaf_node = {
                "unit_id": unit_id,
                "doc_id": document.doc_id,
                "title": document.title,
                "category": document.category,
                "kind": "chunk",
                "level": 0,
                "order": chunk.order,
                "summary": _truncate(chunk.text, self.leaf_text_chars),
                "retrieval_text": _truncate(chunk.text, self.leaf_text_chars),
                "children": [],
                "source_refs": [document.doc_id],
            }
            units.append(leaf_node)
            leaf_nodes.append(leaf_node)

        current_nodes = leaf_nodes
        current_level = 0
        while len(current_nodes) > 1 or current_level == 0:
            grouped_nodes = self._cluster_nodes(current_nodes)
            next_nodes = []
            for group_idx, group in enumerate(grouped_nodes):
                summary = self._summarize_cluster(
                    title=document.title,
                    category=document.category,
                    level=current_level + 1,
                    excerpts=[node.get("summary") or node.get("retrieval_text", "") for node in group],
                )
                node = {
                    "unit_id": f"{document.doc_id}_summary_l{current_level + 1}_{group_idx}",
                    "doc_id": document.doc_id,
                    "title": document.title,
                    "category": document.category,
                    "kind": "summary",
                    "level": current_level + 1,
                    "order": min(item.get("order", 0) for item in group),
                    "summary": summary,
                    "retrieval_text": summary,
                    "children": [item["unit_id"] for item in group],
                    "source_refs": [document.doc_id],
                }
                units.append(node)
                summary_nodes_by_level[current_level + 1].append(node)
                next_nodes.append(node)
            current_nodes = next_nodes
            current_level += 1

        root_node = current_nodes[0]
        root_node["is_document_root"] = True
        return root_node, summary_nodes_by_level

    def _write_candidate_principles(
        self,
        *,
        source_units: list[dict],
        document_index: dict,
    ) -> PrincipleSources:
        save_log_data(document_index, self.document_index_path)

        library = InsightLibrary()
        candidate_principles: list[dict] = []
        seen_signatures: set[tuple[str, str, str]] = set()

        deduped_source_units: list[dict] = []
        seen_unit_ids: set[str] = set()
        for unit in source_units:
            unit_id = unit.get("unit_id")
            if not unit_id or unit_id in seen_unit_ids:
                continue
            seen_unit_ids.add(unit_id)
            deduped_source_units.append(unit)

        for source_unit in deduped_source_units:
            source_title = self._display_source_title(source_unit)
            source_category = source_unit.get("category", "unknown")
            source_summary = self._unit_text(source_unit)
            if not source_summary:
                continue

            candidate_insights = self._extract_candidate_principles(
                title=source_title,
                category=source_category,
                summary=source_summary,
                taxonomy=library.taxonomy,
                source_ref=source_unit["doc_id"],
                source_unit_id=source_unit.get("unit_id", ""),
                source_unit_kind=source_unit.get("kind", "unknown"),
            )
            for insight in candidate_insights:
                signature = (
                    _json_dumps(insight.get("taxonomy", {})),
                    _normalize_signature(insight.get("condition", "")),
                    _normalize_signature(insight.get("explanation", "")),
                )
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                insight["insight_id"] = len(candidate_principles) + 1
                candidate_principles.append(insight)
                library.update_taxonomy(insight)

        save_log_data(candidate_principles, self.candidate_principles_path)
        library.save_taxonomy(self.candidate_taxonomy_path)

        return PrincipleSources(
            document_index_path=self.document_index_path,
            candidate_principles_path=self.candidate_principles_path,
            candidate_taxonomy_path=self.candidate_taxonomy_path,
        )

    def build(self) -> PrincipleSources:
        documents = discover_domain_documents(
            self.corpus_root,
            chunk_words=self.chunk_words,
            overlap_words=self.overlap_words,
        )
        if not documents:
            raise FileNotFoundError(f"No documents found under corpus root: {self.corpus_root}")

        units: list[dict] = []
        documents_meta: list[dict] = []
        source_units: list[dict] = []

        for document in documents:
            root_node, summary_nodes_by_level = self._build_document_hierarchy(
                document=document,
                units=units,
            )

            extraction_units = self._select_document_source_units(
                document=document,
                root_node=root_node,
                summary_nodes_by_level=summary_nodes_by_level,
                max_level=self.extraction_max_level,
            )
            source_units.extend(extraction_units)

            documents_meta.append(
                {
                    "doc_id": document.doc_id,
                    "title": document.title,
                    "category": document.category,
                    "word_count": len(document.text.split()),
                    "chunk_count": len(document.chunks),
                    "root_unit_id": root_node["unit_id"],
                    "extraction_unit_ids": [unit["unit_id"] for unit in extraction_units],
                }
            )

        document_index = {
            "build_config": _hierarchy_settings(self.config),
            "documents": documents_meta,
            "units": units,
        }

        return self._write_candidate_principles(
            source_units=source_units,
            document_index=document_index,
        )


def prepare_principle_sources(
    config: Any,
    *,
    allow_build: bool = True,
) -> Optional[PrincipleSources]:
    bootstrap = getattr(config, "bootstrap", None)
    if bootstrap is None or not getattr(bootstrap, "enable_principle_library", False):
        return None

    document_index_path = bootstrap.document_index_path
    candidate_principles_path = bootstrap.candidate_principles_path
    candidate_taxonomy_path = bootstrap.candidate_taxonomy_path
    rebuild = getattr(bootstrap, "rebuild_document_index", False)

    artifacts_exist = all(
        os.path.exists(path)
        for path in [document_index_path, candidate_principles_path, candidate_taxonomy_path]
    )
    artifacts_match_config = artifacts_exist and _index_matches_settings(
        config=config,
        document_index_path=document_index_path,
    )

    needs_build = rebuild or not artifacts_exist or not artifacts_match_config
    if needs_build and not allow_build:
        missing_paths = [
            path
            for path in [document_index_path, candidate_principles_path, candidate_taxonomy_path]
            if not os.path.exists(path)
        ]
        if rebuild:
            reason = "bootstrap.rebuild_document_index is true"
        elif missing_paths:
            reason = "missing artifact files:\n" + "\n".join(missing_paths)
        else:
            reason = "existing artifacts do not match the current bootstrap config"
        raise RuntimeError(
            "The document index or candidate principles are unavailable and this run mode does not allow rebuilding. "
            f"Reason: {reason}. Run with run_mode: build_principle_sources or run_mode: full first."
        )

    if needs_build:
        builder = PrincipleLibraryBuilder(config, config.model, config.service)
        return builder.build()

    return PrincipleSources(
        document_index_path=document_index_path,
        candidate_principles_path=candidate_principles_path,
        candidate_taxonomy_path=candidate_taxonomy_path,
    )


def build_document_retriever(
    config: Any,
    taxonomy: Optional[dict],
    model: str,
    service: str,
    *,
    allow_build: bool = True,
    artifacts: Optional[PrincipleSources] = None,
) -> Optional[DocumentRetriever]:
    if artifacts is None:
        artifacts = prepare_principle_sources(config, allow_build=allow_build)
    if artifacts is None:
        return None

    index = DocumentIndex.load(artifacts.document_index_path)
    bootstrap = config.bootstrap
    return DocumentRetriever(
        index=index,
        model=model,
        service=service,
        taxonomy=taxonomy,
        max_units=getattr(bootstrap, "summaries_per_task", 6),
        max_insights=getattr(bootstrap, "task_specific_insight_limit", 4),
    )
