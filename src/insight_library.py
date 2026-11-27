import os
import json
import re
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from copy import deepcopy
from .insight_states import iter_insight_ids, normalize_distribution, normalize_insight_record


def _taxonomy_paths(taxonomy, *, stored=False):
    """Yield exact, case-sensitive retrieval keys from compact or annotated taxonomy records."""
    if not isinstance(taxonomy, dict):
        return
    for track, groups in taxonomy.items():
        if not isinstance(groups, dict):
            continue
        for group, leaves in groups.items():
            if isinstance(leaves, dict):
                labels = (key for key, value in leaves.items() if not stored or key is not None or value is not None)
            elif isinstance(leaves, list):
                labels = leaves
            else:
                continue
            for label in labels:
                yield track, group, str(label)


def lesson_source_attempts(lesson):
    if lesson.get("source_attempts"):
        return [tuple(ref) for ref in lesson["source_attempts"]]
    task_ids = lesson.get("task_id")
    if not isinstance(task_ids, list):
        task_ids = [task_ids]
    return [(tid, lesson.get("iteration"), None) for tid in task_ids if tid is not None]


def lesson_source_task_ids(lesson):
    task_ids = list(lesson.get("source_task_ids") or [])
    task_ids.extend(task_id for task_id, _, _ in lesson_source_attempts(lesson))
    labels = []
    for task_id in task_ids:
        if isinstance(task_id, str) and re.fullmatch(r"[ABEF][0-9]{3}", task_id):
            labels.append("initial lessons")
        else:
            labels.append(task_id)
    return list(dict.fromkeys(labels))


def inherit_lesson_sources(replacement, originals):
    refs = dict.fromkeys(ref for original in originals for ref in lesson_source_attempts(original))
    replacement["source_attempts"] = [list(ref) for ref in refs]
    replacement["task_id"] = list(dict.fromkeys(ref[0] for ref in refs))
    replacement["source_task_ids"] = list(dict.fromkeys(
        task_id for original in originals for task_id in lesson_source_task_ids(original)
    ))
    principles = list(replacement.get("related_principle_ids", []))
    for original in originals:
        principles.extend(original.get("related_principle_ids", []))
    if principles:
        replacement["related_principle_ids"] = list(dict.fromkeys(principles))


class Insight:
    def __init__(self, data: dict):
        data = normalize_insight_record(data)
        # Published appendices use namespaced identifiers. Keep those names and
        # use the original integer IDs for compatibility with task histories.
        identifier = data.get("insight_id")
        published = re.fullmatch(r"(principle|lesson)([1-9]\d*)", str(identifier))
        if published:
            data["knowledge_id"] = identifier
            data["insight_id"] = data.get("legacy_insight_id", int(published[2]))
        kind = data.get("insight_type")
        if kind is None:
            document_source = bool(data.get("source_unit_id") or data.get("source_unit_ids"))
            early_principle = data.get("iteration") == -1 and data.get("task_id") is None
            kind = published[1] if published else ("principle" if document_source or early_principle else "lesson")
        if kind not in {"principle", "lesson"} or (published and published[1] != kind):
            raise ValueError(f"Inconsistent insight type {kind!r} for {identifier!r}")
        self.insight_type = kind
        known_fields = {
            "insight_type",
            "insight_id",
            "taxonomy",
            "condition",
            "explanation",
            "example",
            "version",
            "task_id",
            "iteration",
            "occurrence",
            "distribution",
            "merged_from",
            "lineage",
        }
        self.insight_id = data.get("insight_id")
        self.taxonomy = data.get("taxonomy")
        self.condition = data.get("condition")
        self.explanation = data.get("explanation")
        self.example = data.get("example")
        self.version = data.get("version", 1)
        self.task_id = data.get("task_id")
        self.iteration = data.get("iteration")

        # Counters
        self.occurrence = data.get("occurrence", 0)  # how many times retrieved
        # self.correctness = data.get("correctness", 0)  # how many times led to success

        self.distribution = normalize_distribution(data.get("distribution")) # how the insight work on target tasks

        # Genealogy (only add if present in the data)
        self.merged_from = data.get("merged_from", None)  # list of parent IDs, default is None
        self.lineage = data.get("lineage", None)  # list of all primitive ancestors, default is None

        # Remove attributes if not provided
        if self.merged_from is None:
            del self.merged_from
        if self.lineage is None:
            del self.lineage

        self.metadata = {
            key: value
            for key, value in data.items()
            if key not in known_fields
        }

    def to_dict(self) -> dict:
        result = {
            "insight_type": self.insight_type,
            "insight_id": self.insight_id,
            "taxonomy": self.taxonomy,
            "condition": self.condition,
            "explanation": self.explanation,
            "example": self.example,
            "iteration": self.iteration,
            "version": self.version,
            "task_id": self.task_id,
            "occurrence": self.occurrence,    # usage record
            # "correctness": self.correctness,   # usage record
            "distribution": normalize_distribution(self.distribution)
        }

        # Only include these fields if they are present
        if hasattr(self, 'merged_from'):
            result["merged_from"] = self.merged_from
        if hasattr(self, 'lineage'):
            result["lineage"] = self.lineage
        if self.metadata:
            result.update(self.metadata)
        if self.insight_type == "lesson":
            result["source_task_ids"] = lesson_source_task_ids(result)

        return result


class InsightLibrary:
    def __init__(self, insight_list: list | None = None):
        """
        Build an InsightLibrary from an empty library and a pre-defined taxonomy dictionary
        """
        self._library = []          # type: list[Insight]
        if insight_list:                 # Skip if None or []
            for ins in insight_list:
                self._library.append(Insight(ins))
        ids = [ins.insight_id for ins in self._library]
        if any(type(i) is not int for i in ids) or len(ids) != len(set(ids)):
            raise ValueError("Stored insight IDs must be unique integers within the working library")
        self._next_id = max(ids, default=0) + 1
        self.is_checkpoint = False

        taxonomy_path = Path(__file__).resolve().parents[1] / "data/libraries/base_taxonomy.json"
        with taxonomy_path.open(encoding="utf-8") as f:
            self.taxonomy = json.load(f)
        # self._taxo_lock = threading.RLock()

    @classmethod
    def from_json_file(cls, library_path: str, taxonomy_path: Optional[str] = None) -> "InsightLibrary":
        """
        Read a JSON file (list of dicts) and return an InsightLibrary instance.
        """
        if not os.path.isfile(library_path):
            raise FileNotFoundError(library_path)
        with open(library_path, "r", encoding="utf-8") as f:
            data = json.load(f)     # Data is a list[dict]

        if isinstance(data, dict):
            if data.get("schema_version") != 2:
                raise ValueError("Unsupported working-library checkpoint schema")
            records = []
            for kind in ("principle", "lesson"):
                for record in data[kind + "s"]:
                    if record.get("insight_type", kind) != kind:
                        raise ValueError(f"Conflicting type in {kind} partition")
                    records.append({**record, "insight_type": kind})
            inst = cls(records)
            if "insight_order" in data:
                by_id = {ins.insight_id: ins for ins in inst}
                order = data["insight_order"]
                if len(order) != len(by_id) or set(order) != set(by_id):
                    raise ValueError("Checkpoint order does not match stored insight IDs")
                inst._library = [by_id[iid] for iid in order]
            inst._next_id = max(inst._next_id, data["next_insight_id"])
            inst.is_checkpoint = True
        else:
            inst = cls(insight_list=data)

        if taxonomy_path is not None:
            if not os.path.isfile(taxonomy_path):
                raise FileNotFoundError(taxonomy_path)
            with open(taxonomy_path, "r", encoding="utf-8") as f:
                inst.taxonomy = json.load(f)

        inst.update_taxonomy(inst.to_json())
        return inst

    def __getitem__(self, index: int) -> Insight:
        return self._library[index]

    def __setitem__(self, index: int, new_insight: Insight):
        self._library[index] = new_insight

    def __len__(self):
        return len(self._library)

    def to_json(self) -> list:
        return [ins.to_dict() for ins in self._library]

    @property
    def principles(self):
        return [ins for ins in self._library if ins.insight_type == "principle"]

    @property
    def lessons(self):
        return [ins for ins in self._library if ins.insight_type == "lesson"]

    def attach_principles(self, source):
        """Import missing document rules without overwriting learned conditions or reusing IDs."""
        def identity(ins):
            return json.dumps([
                self._normalize_taxonomy_for_storage(ins.taxonomy), ins.explanation, ins.example,
                ins.metadata.get("source_unit_ids") or [ins.metadata.get("source_unit_id")],
            ], sort_keys=True)

        named = {ins.metadata.get("knowledge_id") for ins in self.principles}
        legacy = {}
        for ins in self.principles:
            if not ins.metadata.get("knowledge_id"):
                legacy.setdefault(identity(ins), []).append(ins)
        pending = {}
        for original in source:
            name = original.metadata.get("knowledge_id", f"principle{original.insight_id}")
            if name not in named:
                pending.setdefault(identity(original), []).append((name, original))
        for key, group in pending.items():
            remaining = legacy.get(key, [])
            unmatched = []
            for name, original in group:
                match = next((ins for ins in remaining if ins.condition == original.condition), None)
                if match:
                    match.metadata["knowledge_id"] = name
                    remaining.remove(match)
                else:
                    unmatched.append((name, original))
            # A unique immutable rule identifies a legacy principle after condition refinement.
            if len(unmatched) == len(remaining) == 1:
                remaining[0].metadata["knowledge_id"] = unmatched[0][0]
                continue
            for name, original in unmatched:
                ins = deepcopy(original)
                ins.insight_type = "principle"
                ins.metadata["knowledge_id"] = name
                ins.insight_id = self._next_id
                self._next_id += 1
                self._library.append(ins)
        self.update_taxonomy({"taxonomy": source.taxonomy})

    def select_types(self, kinds):
        self._library = [ins for ins in self if ins.insight_type in kinds]
        original = self.taxonomy
        self.taxonomy = {track: {} for track in ("Domain Modeling", "General Formulation", "Code Implementation")}
        for ins in self:
            for track, groups in self._normalize_taxonomy_for_storage(ins.taxonomy).items():
                if track not in self.taxonomy:
                    continue
                for group, labels in groups.items():
                    leaves = self.taxonomy[track].setdefault(group, {})
                    definitions = original.get(track, {}).get(group, {})
                    leaves.update({label: definitions.get(label, {}) for label in labels})

    def reserve_historical_ids(self, tasks):
        # Flat checkpoints did not save a high-water mark. Deleted lessons may
        # still be referenced by a task and must never become a new principle.
        for task in tasks:
            for iid in iter_insight_ids(task.retri_ins_lst):
                if type(iid) is int:
                    self._next_id = max(self._next_id, iid + 1)

    def save(self, path: str, *, checkpoint=False):
        data = self.to_json()
        if checkpoint:
            data = {
                "schema_version": 2, "next_insight_id": self._next_id,
                "insight_order": [row["insight_id"] for row in data],
                "principles": [row for row in data if row["insight_type"] == "principle"],
                "lessons": [row for row in data if row["insight_type"] == "lesson"],
            }
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=directory, encoding="utf-8", delete=False) as f:
            try:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.close()
                os.replace(f.name, path)
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)

    def save_taxonomy(self, path: str):
        """
        Save the current taxonomy to a JSON file
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.taxonomy, f, indent=2, ensure_ascii=False)

    def update_usage(self, insight_ids: list, success: bool):
        """
        Increment occurrence for all used insight_ids.
        If `success` is True, also increment correctness.
        """
        # insight_ids = [ins["insight_id"] for ins in update_lst]
        for ins in self._library:
            if ins.insight_id in insight_ids:
                ins.occurrence += 1
                if success:
                    ins.metadata["correctness"] = ins.metadata.get("correctness", 0) + 1

    def retrieve_insights_by_id(self, insight_ids: int | str | list, *, filter_fn=None) -> list[dict]:
        """Return full prompt records in request order for namespaced or runtime IDs."""
        index = {rule.insight_id: rule for rule in self}
        index.update({rule.metadata["knowledge_id"]: rule for rule in self if rule.metadata.get("knowledge_id")})
        keys = insight_ids if isinstance(insight_ids, (int, str)) else list(insight_ids)
        records = []
        for key in iter_insight_ids(keys):
            rule = index.get(key)
            if rule is None or (filter_fn is not None and not filter_fn(rule)):
                continue
            record = {"insight_id": rule.insight_id, "insight_type": rule.insight_type}
            if "knowledge_id" in rule.metadata:
                record["knowledge_id"] = rule.metadata["knowledge_id"]
            record.update({field: getattr(rule, field) for field in ("taxonomy", "condition", "explanation", "example")})
            records.append(record)
        return records

    def retrieve_by_taxonomy(self, query_taxonomy, filter_fn=None, include_task_id=False):
        wanted = set(_taxonomy_paths(query_taxonomy))
        if not wanted:
            return []
        fields = ("insight_id", "insight_type", "taxonomy", "condition")
        if include_task_id:
            fields += ("explanation", "example", "task_id", "iteration")
        records = []
        for rule in self:
            if filter_fn is not None and not filter_fn(rule):
                continue
            if not wanted.isdisjoint(_taxonomy_paths(rule.taxonomy, stored=True)):
                records.append({field: getattr(rule, field) for field in fields})
        return records

    @staticmethod
    def _normalize_taxonomy_for_storage(taxonomy: Any) -> dict:
        if not isinstance(taxonomy, dict):
            return {}
        compact = {}
        for track, groups in taxonomy.items():
            compact[track] = {name: list(leaves) if isinstance(leaves, (list, dict)) else []
                              for name, leaves in groups.items()} if isinstance(groups, dict) else {}
        return compact

    def add_insights(self, new_insights: list, iteration:int=None) -> None:
        for source_ins in new_insights:
            ins = deepcopy(source_ins)
            ins["taxonomy"] = self._normalize_taxonomy_for_storage(ins.get("taxonomy", {}))

            # Merge defaults with the incoming dict (incoming keys win if duplicated)
            enriched = {
                **ins,
                "insight_id":  self._next_id,
                "iteration": iteration,
                "version":     1,
                "occurrence":  0,
                # "correctness": 0
            }

            self._library.append(Insight(enriched))
            self._next_id += 1


    @staticmethod
    def _update_one_stage(stage, ins_label_dict):
        def existing_names(mapping):
            names = {}
            for key in mapping:
                names.setdefault(str(key).casefold(), key)
            return names

        groups = existing_names(stage)
        for name, labels in (ins_label_dict or {}).items():
            group_name = groups.setdefault(str(name).casefold(), name)
            leaves = stage.setdefault(group_name, {})
            if not isinstance(labels, (list, dict)):
                continue
            existing = existing_names(leaves)
            items = labels.items() if isinstance(labels, dict) else ((label, None) for label in labels)
            for key, definition in items:
                spelling = existing.setdefault(str(key).casefold(), key)
                if spelling not in leaves:
                    leaves[spelling] = definition if definition is not None else {}
                elif not leaves[spelling] and definition:
                    leaves[spelling] = deepcopy(definition)

    def update_taxonomy(self, new_labeled_insights):
        records = [new_labeled_insights] if isinstance(new_labeled_insights, dict) else new_labeled_insights
        for record in records or ():
            taxonomy = record.get("taxonomy")
            if isinstance(taxonomy, dict):
                for track in ("Domain Modeling", "General Formulation", "Code Implementation"):
                    groups = taxonomy.get(track)
                    if isinstance(groups, dict):
                        self._update_one_stage(self.taxonomy.setdefault(track, {}), groups)

    def replace_merged_insights(self, existing_insights: List[dict]) -> None:
        """
        Replace existing insights that were merged with new merged insights.
        This removes the old insights from the library.

        Args:
            existing_insights: List of existing insights that were merged
        """
        # Get insight IDs that were merged (from existing_insights)
        merged_insight_ids = [ins["insight_id"] for ins in existing_insights]

        # Remove the merged insights from library
        self._library = [ins for ins in self._library if ins.insight_id not in merged_insight_ids]

        print(f"Removed {len(merged_insight_ids)} existing insights that were merged: {merged_insight_ids}")

    def merge_into_library(
        self,
        all_merged_ids: List[List[int]],
        all_merged_insights: List[dict],
        lib_base: "InsightLibrary"
    ) -> "InsightLibrary":
        """Apply a batch of accepted replacements, keeping principles used to derive lessons."""
        if len(all_merged_ids) != len(all_merged_insights):
            raise ValueError("Each merge group needs one replacement")
        by_id = {ins.insight_id: ins for ins in lib_base}
        removed, referenced = set(), set()
        additions = []
        for ids, proposal in zip(all_merged_ids, all_merged_insights):
            if not ids or len(set(ids)) != len(ids) or not set(ids) <= by_id.keys():
                raise ValueError("A merge group must refer to distinct existing parents")
            parents = [by_id[i] for i in ids]
            lessons = [p for p in parents if p.insight_type == "lesson"]
            consumed = {p.insight_id for p in lessons or parents}
            if consumed & referenced or removed.intersection(ids):
                raise ValueError("A replaced parent belongs to more than one group")
            removed.update(consumed)
            referenced.update(ids)
            record = {field: deepcopy(proposal[field])
                      for field in ("taxonomy", "condition", "explanation", "example")}
            lineage = {i for p in parents for i in (getattr(p, "lineage", None) or [p.insight_id])}
            record.update(
                insight_id=lib_base._next_id + len(additions),
                insight_type="lesson" if lessons else "principle",
                version=max(p.version for p in parents) + 1,
                merged_from=list(ids), lineage=sorted(lineage),
            )
            if lessons:
                record["related_principle_ids"] = [p.insight_id for p in parents if p.insight_type == "principle"]
                inherit_lesson_sources(record, [p.to_dict() for p in lessons])
            else:
                for field in ("source_titles", "source_refs", "source_unit_ids"):
                    values = [v for p in parents
                              for v in (p.metadata.get(field) or [p.metadata.get(field[:-1])]) if v]
                    record[field] = list(dict.fromkeys(values))
            additions.append(Insight(record))
        lib_base._library = [ins for ins in lib_base if ins.insight_id not in removed] + additions
        lib_base._next_id += len(additions)
        lib_base.update_taxonomy([ins.to_dict() for ins in additions])
        return lib_base
