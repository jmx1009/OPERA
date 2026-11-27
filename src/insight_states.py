"""Canonical insight states and compatibility with earlier checkpoints."""

from copy import deepcopy
from typing import Any


INSIGHT_STATES = ("useful", "negative", "irrelevant", "unretrieved")
DEFAULT_DISTRIBUTION = {state: [] for state in INSIGHT_STATES}
DISTRIBUTION_STATE_ALIASES = {
    "positive": "useful",
    "postive": "useful",
    "retrieved_unapplied": "irrelevant",
    "retrieved-unapplied": "irrelevant",
    "unapplied": "irrelevant",
    "invalid": "irrelevant",
}
LEGACY_APPLICABLE_STATES = {
    "retrieved_unapplied", "retrieved-unapplied", "unapplied", "invalid",
}


def iter_insight_ids(value):
    """Read flat or stage-grouped IDs from a retrieval record, preserving order."""
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_insight_ids(item)
    elif type(value) is int or isinstance(value, str):
        yield value


def normalize_insight_state(state: str) -> str:
    return DISTRIBUTION_STATE_ALIASES.get(state, state)


def normalize_distribution(distribution: Any) -> dict:
    """Merge legacy lists without mutating the input or creating extra states."""
    normalized = deepcopy(DEFAULT_DISTRIBUTION)
    if not isinstance(distribution, dict):
        return normalized
    for raw_state, raw_values in distribution.items():
        state = normalize_insight_state(raw_state)
        if state not in normalized or not isinstance(raw_values, list):
            continue
        for task_id in raw_values:
            if task_id not in normalized[state]:
                normalized[state].append(task_id)
    return normalized


def normalize_reflection_record(record: dict) -> dict:
    """Preserve applicability evidence when a legacy state is renamed."""
    result = deepcopy(record)
    raw_state = result.get("state")
    result["state"] = normalize_insight_state(raw_state)
    if raw_state in LEGACY_APPLICABLE_STATES:
        result["applicable"] = True
    elif result["state"] == "useful":
        result.setdefault("applicable", True)
    return result


def normalize_insight_record(record: dict) -> dict:
    """Convert a library record while retaining legacy retrieval evidence."""
    result = deepcopy(record)
    raw = record.get("distribution")
    if isinstance(raw, dict):
        applicable_ids = list(result.get("legacy_applicable_task_ids") or [])
        for state, values in raw.items():
            if state not in LEGACY_APPLICABLE_STATES or not isinstance(values, list):
                continue
            for task_id in values:
                if task_id not in applicable_ids:
                    applicable_ids.append(task_id)
        if applicable_ids:
            result["legacy_applicable_task_ids"] = applicable_ids
        recovered = raw.get("recovered_unretrieved")
        if recovered:
            evaluation = result.setdefault("retrieval_evaluation", {})
            previous = evaluation.setdefault("recovered_unretrieved", [])
            previous.extend(task_id for task_id in recovered if task_id not in previous)
    result["distribution"] = normalize_distribution(raw)
    return result


def supports_formulation(record: dict) -> bool:
    """Retain only useful stored guidance in the repaired formulation."""
    return normalize_insight_state(record.get("state")) == "useful"


def migrate_records(records):
    if isinstance(records, dict) and records.get("schema_version") == 2:
        return {**records, **{key: migrate_records(records[key]) for key in ("principles", "lessons")}}
    if not isinstance(records, list):
        return records
    result = []
    allowed = set(INSIGHT_STATES) | set(DISTRIBUTION_STATE_ALIASES) | {"recovered_unretrieved"}
    for record in records:
        if not isinstance(record, dict) or not {"condition", "explanation"} <= record.keys():
            result.append(record)
            continue
        raw = record.get("distribution")
        if isinstance(raw, dict) and set(raw) - allowed:
            raise ValueError(f"Unknown distribution keys: {sorted(set(raw) - allowed)}")
        result.append(normalize_insight_record(record))
    return result
