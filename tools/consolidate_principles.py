#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


VALID_STAGES = ("Domain Modeling", "General Formulation", "Code Implementation")
VALID_STAGE_SET = set(VALID_STAGES)
from src.config import MODEL, SERVICE, validate_model
from src.prompts.manuscript import bind_template
from src.principle_library import _is_actionable_insight_candidate
DRIFT_STAGE_MAP = {
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


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def find_ci(mapping: dict, key: str) -> str | None:
    key_norm = str(key).casefold()
    for existing_key in mapping:
        if str(existing_key).casefold() == key_norm:
            return existing_key
    return None


def merge_branch(stage_branch: dict, level1_name: str, level2_spec: Any) -> None:
    level1_key = find_ci(stage_branch, level1_name) or level1_name
    if level1_key not in stage_branch:
        stage_branch[level1_key] = deepcopy(level2_spec) if isinstance(level2_spec, (dict, list)) else {}
        return

    existing_spec = stage_branch[level1_key]
    if isinstance(existing_spec, dict):
        if isinstance(level2_spec, dict):
            for level2_name, info in level2_spec.items():
                if find_ci(existing_spec, level2_name) is None:
                    existing_spec[level2_name] = deepcopy(info)
        elif isinstance(level2_spec, list):
            for level2_name in level2_spec:
                if find_ci(existing_spec, level2_name) is None:
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


def resolve_stage(stage_label: str, reference_taxonomy: dict | None) -> str | None:
    mapped = DRIFT_STAGE_MAP.get(str(stage_label).casefold())
    if mapped:
        return mapped

    if not isinstance(reference_taxonomy, dict):
        return None

    matches = []
    for stage in VALID_STAGES:
        branch = reference_taxonomy.get(stage)
        if isinstance(branch, dict) and find_ci(branch, stage_label):
            matches.append(stage)
    return matches[0] if len(matches) == 1 else None


def normalize_taxonomy(taxonomy: Any, reference_taxonomy: dict | None = None) -> dict:
    if not isinstance(taxonomy, dict):
        return {}

    normalized: dict[str, dict] = {}
    for stage_label, stage_value in taxonomy.items():
        if stage_label in VALID_STAGE_SET:
            if isinstance(stage_value, dict):
                normalized.setdefault(stage_label, {})
                for level1_name, level2_spec in stage_value.items():
                    merge_branch(normalized[stage_label], level1_name, level2_spec)
            continue

        target_stage = resolve_stage(stage_label, reference_taxonomy)
        if target_stage is None:
            continue

        normalized.setdefault(target_stage, {})
        merge_branch(normalized[target_stage], stage_label, stage_value)

    return compact_taxonomy(normalized)


def compact_taxonomy(taxonomy: Any) -> dict:
    if not isinstance(taxonomy, dict):
        return {}

    compact: dict[str, dict] = {}
    for stage, branch in taxonomy.items():
        if stage not in VALID_STAGE_SET or not isinstance(branch, dict):
            continue

        compact[stage] = {}
        for level1_name, level2_spec in branch.items():
            if isinstance(level2_spec, dict):
                compact[stage][level1_name] = list(level2_spec.keys())
            elif isinstance(level2_spec, list):
                compact[stage][level1_name] = list(level2_spec)
            else:
                compact[stage][level1_name] = []

        if not compact[stage]:
            del compact[stage]
    return compact


def taxonomy_paths(taxonomy: Any) -> list[tuple[str, str, str]]:
    paths = []
    if not isinstance(taxonomy, dict):
        return paths

    for stage in VALID_STAGES:
        branch = taxonomy.get(stage)
        if not isinstance(branch, dict):
            continue
        for level1_name, level2_spec in branch.items():
            if isinstance(level2_spec, dict):
                level2_names = list(level2_spec.keys())
            elif isinstance(level2_spec, list):
                level2_names = list(level2_spec)
            else:
                level2_names = []
            if level2_names:
                for level2_name in level2_names:
                    paths.append((stage, level1_name, str(level2_name)))
            else:
                paths.append((stage, level1_name, ""))
    return paths


def primary_group_key(insight: dict) -> tuple[str, str]:
    paths = taxonomy_paths(insight.get("taxonomy"))
    if not paths:
        return ("Uncategorized", "Uncategorized")
    stage, level1_name, _ = paths[0]
    return (stage, level1_name)


def score_insight(insight: dict) -> float:
    condition = str(insight.get("condition") or "")
    explanation = str(insight.get("explanation") or "")
    example = str(insight.get("example") or "")
    combined = " ".join([condition, explanation, example]).lower()

    score = 0.0
    score += min(len(condition.split()) / 35.0, 1.0) * 2.0
    score += min(len(explanation.split()) / 70.0, 1.0) * 2.0
    score += min(len(example.split()) / 45.0, 1.0) * 1.5

    trigger_terms = ("when", "applies", "apply", "if", "exclude", "unless")
    modeling_terms = (
        "variable",
        "objective",
        "constraint",
        "linear",
        "binary",
        "integer",
        "flow",
        "capacity",
        "routing",
        "scheduling",
        "decomposition",
        "solver",
        "gurobi",
    )
    template_terms = ("x_", "sum", "<=", ">=", "=", "for all", "addconstr", "quicksum", "binary", "continuous")
    low_value_terms = ("this paper", "this study", "trend", "future work", "important", "background")

    score += sum(0.4 for term in trigger_terms if term in combined)
    score += min(sum(0.25 for term in modeling_terms if term in combined), 2.0)
    score += min(sum(0.35 for term in template_terms if term in combined), 1.5)
    score -= sum(0.6 for term in low_value_terms if term in combined)

    if insight.get("merged_from"):
        score += min(len(insight.get("merged_from") or []) / 4.0, 1.0)
    return score


def group_cap(group_size: int, max_group_output: int) -> int:
    if group_size <= 0:
        return 0
    if group_size <= 5:
        return group_size
    if group_size <= 15:
        return min(group_size, max(8, max_group_output))
    if group_size <= 30:
        return min(group_size, max(10, max_group_output))
    return min(group_size, max_group_output)


def group_budget(group_size: int, total_size: int, target_size: int, max_group_output: int) -> int:
    cap = group_cap(group_size, max_group_output)
    proportional = max(1, math.ceil(target_size * group_size / max(total_size, 1)))
    return max(1, min(cap, proportional))


def compute_group_budgets(
    ordered_groups: list[tuple[tuple[str, str], list[dict]]],
    target_size: int,
    max_group_output: int,
) -> dict[tuple[str, str], int]:
    total_size = sum(len(items) for _, items in ordered_groups)
    budgets = {
        group_key: group_budget(len(items), total_size, target_size, max_group_output)
        for group_key, items in ordered_groups
    }

    remaining = max(target_size - sum(budgets.values()), 0)
    while remaining > 0:
        progressed = False
        for group_key, items in ordered_groups:
            cap = group_cap(len(items), max_group_output)
            if budgets[group_key] >= cap:
                continue
            budgets[group_key] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return budgets


def rebuild_taxonomy(insights: list[dict]) -> dict:
    taxonomy = {stage: {} for stage in VALID_STAGES}
    for insight in insights:
        for stage, level1_name, level2_name in taxonomy_paths(insight.get("taxonomy")):
            taxonomy.setdefault(stage, {}).setdefault(level1_name, {})
            if level2_name:
                taxonomy[stage][level1_name].setdefault(level2_name, {})
    return {stage: branch for stage, branch in taxonomy.items() if branch}


def summarize_coverage(insights: list[dict]) -> dict:
    top_stage = Counter()
    level1 = Counter()
    source_category = Counter()
    for insight in insights:
        for stage, level1_name, _ in taxonomy_paths(insight.get("taxonomy")):
            top_stage[stage] += 1
            level1[f"{stage} / {level1_name}"] += 1
        source_category.update([insight.get("source_category", "unknown")])
    return {
        "top_stage": dict(top_stage),
        "level1": dict(level1),
        "source_category": dict(source_category),
    }


def candidate_payload(insights: list[dict]) -> list[dict]:
    payload = []
    for insight in insights:
        payload.append(
            {
                "insight_id": insight.get("insight_id"),
                "taxonomy": insight.get("taxonomy", {}),
                "condition": insight.get("condition", ""),
                "explanation": insight.get("explanation", ""),
                "example": insight.get("example", ""),
                "source_category": insight.get("source_category"),
                "source_title": insight.get("source_title"),
            }
        )
    return payload


CONSOLIDATE = bind_template(
    "T3", TOP_LEVEL_CATEGORY="{stage}", MODELING_CATEGORY="{group}", MAX_OUTPUT="{limit}",
    CANDIDATES="{candidates}", NESTED_TAXONOMY="{taxonomy_example}",
)


def build_prompt(group_key: tuple[str, str], insights: list[dict], max_output: int) -> str:
    stage, group = group_key
    return CONSOLIDATE.format(
        stage=stage, group=group, limit=max_output,
        candidates=json.dumps(candidate_payload(insights), ensure_ascii=False, indent=2),
        taxonomy_example=json.dumps({stage: {group: ["specific label"]}}, ensure_ascii=False),
    )


def call_llm_for_group(args: argparse.Namespace, prompt: str) -> list[dict]:
    from src.utils import call_llm_and_parse_with_retry, extract_json_array

    return call_llm_and_parse_with_retry(
        model=args.model,
        service=args.service,
        prompt=prompt,
        parse_fn=extract_json_array,
        temperature=args.temperature,
        max_retry=args.max_retry,
        sleep_sec=args.sleep_sec,
        verbose=args.verbose,
        error_message="Failed to consolidate candidate principles",
    )


def collect_source_metadata(merged_from: list[int], by_id: dict[int, dict]) -> dict:
    source_titles = []
    source_refs = []
    source_categories = []
    source_unit_ids = []
    lineage = []
    for source_id in merged_from:
        source = by_id.get(source_id)
        if not source:
            continue
        source_titles.extend(source.get("source_titles") or [source.get("source_title")])
        source_refs.extend(source.get("source_refs") or [source.get("source_ref")])
        source_categories.extend(source.get("source_categories") or [source.get("source_category")])
        source_unit_ids.extend(source.get("source_unit_ids") or [source.get("source_unit_id")])
        source_lineage = source.get("lineage") or source.get("merged_from") or [source_id]
        lineage.extend(source_lineage)

    def compact(values: list[Any]) -> list[Any]:
        result = []
        seen = set()
        for value in values:
            if value in (None, ""):
                continue
            marker = json.dumps(value, ensure_ascii=False, sort_keys=True)
            if marker in seen:
                continue
            seen.add(marker)
            result.append(value)
        return result

    return {
        "source_titles": compact(source_titles),
        "source_refs": compact(source_refs),
        "source_categories": compact(source_categories),
        "source_unit_ids": compact(source_unit_ids),
        "lineage": compact(lineage),
    }


def normalize_llm_records(
    raw_records: list[dict],
    group_key: tuple[str, str],
    by_id: dict[int, dict],
    reference_taxonomy: dict,
) -> list[dict]:
    normalized = []
    group_stage, group_level1 = group_key

    for raw in raw_records or []:
        if not isinstance(raw, dict):
            continue

        merged_from = raw.get("merged_from") or []
        if isinstance(merged_from, int):
            merged_from = [merged_from]
        if not isinstance(merged_from, list) or not merged_from or any(
                not str(item).isdigit() or int(item) not in by_id for item in merged_from):
            continue
        merged_from = list(dict.fromkeys(map(int, merged_from)))

        fallback_taxonomy: dict[str, dict] = {group_stage: {group_level1: []}}
        for source_id in merged_from:
            for stage, level1_name, level2_name in taxonomy_paths(by_id[source_id].get("taxonomy")):
                if stage == group_stage and level1_name == group_level1:
                    fallback_taxonomy.setdefault(stage, {}).setdefault(level1_name, [])
                    if level2_name and level2_name not in fallback_taxonomy[stage][level1_name]:
                        fallback_taxonomy[stage][level1_name].append(level2_name)

        taxonomy = normalize_taxonomy(raw.get("taxonomy"), reference_taxonomy)
        if not taxonomy_paths(taxonomy):
            taxonomy = fallback_taxonomy

        record = {
            "taxonomy": taxonomy,
            "condition": str(raw.get("condition") or "").strip(),
            "explanation": str(raw.get("explanation") or "").strip(),
            "example": str(raw.get("example") or "").strip(),
            "merged_from": merged_from,
            "compaction_action": raw.get("action") or ("merge" if len(merged_from) > 1 else "keep"),
            "compaction_reason": str(raw.get("reason") or "").strip(),
            "compaction_group": f"{group_stage} / {group_level1}",
        }
        source_metadata = collect_source_metadata(merged_from, by_id)
        if source_metadata.get("lineage"):
            record["merged_from"] = source_metadata["lineage"]
        record.update(source_metadata)

        if record["condition"] and record["explanation"] and record["example"]:
            normalized.append(record)
    return normalized


def compact_group(
    group_key: tuple[str, str],
    insights: list[dict],
    budget: int,
    args: argparse.Namespace,
    by_id: dict[int, dict],
    reference_taxonomy: dict,
) -> tuple[list[dict], dict]:
    if budget <= 0:
        return [], {"input_count": len(insights), "budget": budget, "output_count": 0, "fallback": False}

    batches = [insights[i:i + args.max_input_insights] for i in range(0, len(insights), args.max_input_insights)]
    provisional = []
    for batch in batches:
        batch_budget = min(budget, group_budget(len(batch), len(insights), budget, args.max_group_output))
        raw_records = call_llm_for_group(args, build_prompt(group_key, batch, batch_budget))
        supplied = {r["insight_id"]: by_id[r["insight_id"]] for r in batch}
        provisional.extend(normalize_llm_records(raw_records, group_key, supplied, reference_taxonomy)[:batch_budget])

    if len(provisional) > budget:
        numbered = {i: {**record, "insight_id": i} for i, record in enumerate(provisional, 1)}
        raw_records = call_llm_for_group(args, build_prompt(group_key, list(numbered.values()), budget))
        provisional = normalize_llm_records(raw_records, group_key, numbered, reference_taxonomy)

    return provisional[:budget], {
        "input_count": len(insights),
        "budget": budget,
        "output_count": min(len(provisional), budget),
        "fallback": False,
    }


def trim_to_target(records: list[dict], target_size: int) -> list[dict]:
    if len(records) <= target_size:
        return records

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record.get("compaction_group", "unknown")].append(record)

    keep: list[dict] = []
    kept_ids = set()
    for group_records in grouped.values():
        best = max(group_records, key=score_insight)
        keep.append(best)
        kept_ids.add(id(best))

    remaining = [record for record in records if id(record) not in kept_ids]
    remaining.sort(key=score_insight, reverse=True)
    keep.extend(remaining[: max(target_size - len(keep), 0)])
    return sorted(keep, key=score_insight, reverse=True)[:target_size]


def assign_ids(records: list[dict]) -> list[dict]:
    assigned = []
    for idx, record in enumerate(records, start=1):
        item = deepcopy(record)
        item["insight_id"] = f"principle{idx}"
        item["insight_type"] = "principle"
        assigned.append(item)
    return assigned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate candidate principles for expert review.")
    parser.add_argument("--library", type=Path, default=Path("data/knowledge/candidates.json"))
    parser.add_argument("--taxonomy", type=Path, default=Path("data/knowledge/candidate_taxonomy.json"))
    parser.add_argument("--output-library", type=Path, default=Path("runs/consolidation/principles.json"))
    parser.add_argument("--output-taxonomy", type=Path, default=Path("runs/consolidation/taxonomy.json"))
    parser.add_argument("--report", type=Path, default=Path("runs/consolidation/report.json"))
    parser.add_argument("--target-size", type=int, default=220)
    parser.add_argument("--max-group-output", type=int, default=14)
    parser.add_argument("--max-input-insights", type=int, default=25)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--service", default=SERVICE)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-retry", type=int, default=3)
    parser.add_argument("--sleep-sec", type=float, default=1.0)
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Only compute grouping and budgets; do not call LLM or write consolidated principles.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_model(args.model, args.service)
    library = read_json(args.library)
    reference_taxonomy = read_json(args.taxonomy)
    if not isinstance(library, list):
        raise ValueError(f"Expected a list of candidate principles, got {type(library).__name__}")

    cleaned = []
    for insight in library:
        item = deepcopy(insight)
        item["taxonomy"] = normalize_taxonomy(item.get("taxonomy"), reference_taxonomy)
        if _is_actionable_insight_candidate(item):
            cleaned.append(item)

    by_id = {int(item["insight_id"]): item for item in cleaned if item.get("insight_id") is not None}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for insight in cleaned:
        groups[primary_group_key(insight)].append(insight)

    ordered_groups = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)
    if args.limit_groups is not None:
        ordered_groups = ordered_groups[: args.limit_groups]

    budgets = compute_group_budgets(ordered_groups, args.target_size, args.max_group_output)

    report = {
        "input_library": str(args.library),
        "input_taxonomy": str(args.taxonomy),
        "output_library": str(args.output_library),
        "output_taxonomy": str(args.output_taxonomy),
        "target_size": args.target_size,
        "input_count": len(cleaned),
        "input_coverage": summarize_coverage(cleaned),
        "groups": {
            " / ".join(group_key): {
                "input_count": len(items),
                "budget": budgets[group_key],
            }
            for group_key, items in ordered_groups
        },
    }

    if args.dry_run:
        write_json(args.report, report)
        print(f"Dry run complete. Wrote group budget report to {args.report}")
        return

    compacted: list[dict] = []
    group_results = {}
    for idx, (group_key, items) in enumerate(ordered_groups, start=1):
        print(f"[{idx}/{len(ordered_groups)}] Compacting {' / '.join(group_key)} ({len(items)} -> {budgets[group_key]})")
        records, group_report = compact_group(group_key, items, budgets[group_key], args, by_id, reference_taxonomy)
        compacted.extend(records)
        group_results[" / ".join(group_key)] = group_report

    compacted = trim_to_target(compacted, args.target_size)
    compacted = assign_ids(compacted)
    core_taxonomy = rebuild_taxonomy(compacted)

    represented_ids = set()
    for record in compacted:
        represented_ids.update(record.get("lineage") or record.get("merged_from") or [])
    input_ids = {item.get("insight_id") for item in cleaned if item.get("insight_id") is not None}

    report.update(
        {
            "output_count": len(compacted),
            "output_coverage": summarize_coverage(compacted),
            "groups": group_results,
            "represented_input_count": len(represented_ids.intersection(input_ids)),
            "unrepresented_input_ids": sorted(input_ids - represented_ids),
        }
    )

    write_json(args.output_library, compacted)
    write_json(args.output_taxonomy, core_taxonomy)
    write_json(args.report, report)
    print(f"Wrote compacted library: {args.output_library} ({len(compacted)} insights)")
    print(f"Wrote compacted taxonomy: {args.output_taxonomy}")
    print(f"Wrote compaction report: {args.report}")


if __name__ == "__main__":
    main()
