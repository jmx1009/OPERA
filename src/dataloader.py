import json
import math
import re
from copy import deepcopy
from functools import cached_property
from pathlib import Path

import pandas as pd

class Task:
    def __init__(self, task_id, desc, ground_truth, formulation=None, correct_program=None, tag=None, cluster=None):
        self.id = task_id
        self.desc = desc
        self.ground_truth = ground_truth
        self.formulation = formulation
        self.correct_program = correct_program
        self.tag = tag
        self.cluster = cluster

        # Initialize task records
        self.success_count = 0                  # Count of task success
        self.confidence = 0                     # The correct confidence of task
        self.output_status = []                 # List of status for program output per iteration
        self.fail_to_execute = 0               # Count of task failed to correct program
        self.fail_to_verify = 0
        self.retri_ins_lst = []
        self.attempt_history = []
        self._archived_formulation_path = None

    def record_attempt(self, iteration, attempt, status, formulation, insights):
        self.attempt_history.append({
            "iteration": iteration, "attempt": attempt, "status": status,
            "formulation": formulation, "insights": deepcopy(insights),
        })

    @property
    def latest_attempted_formulation(self):
        if self.attempt_history:
            return self.attempt_history[-1].get("formulation")
        return self._archived_formulation

    @cached_property
    def _archived_formulation(self):
        if self._archived_formulation_path is None:
            return None
        try:
            return self._archived_formulation_path.read_text(encoding="utf-8").strip() or None
        except (OSError, UnicodeError):
            return None

    def to_dict(self, mode="learn"):
        identity = {"task_id": self.id, "description": self.desc, "ground_truth": self.ground_truth}
        if mode == "test":
            return {**identity, "output_status": self.output_status, "attempt_history": self.attempt_history,
                    "retrieved_insights": self.retri_ins_lst, "tag": self.tag}
        if mode == "learn":
            return {
                **identity, "formulation": self.formulation, "correct_program": self.correct_program,
                "output_status": self.output_status, "success_count": self.success_count,
                "success_confidence": self.confidence, "fail_to_execute": self.fail_to_execute,
                "fail_to_verify": self.fail_to_verify, "retrieved_insights": self.retri_ins_lst,
                "attempt_history": self.attempt_history, "tag": self.tag, "cluster": self.cluster,
            }

    def restore_progress(self, record, checkpoint, mode):
        self.output_status = record.get("output_status", [])
        self.success_count = record.get("success_count", 0)
        self.confidence = record.get("success_confidence", record.get("confidence", 0))
        self.fail_to_execute = record.get("fail_to_execute", 0)
        self.fail_to_verify = record.get("fail_to_verify", 0)
        self.retri_ins_lst = record.get("retrieved_insights", [])
        self.attempt_history = record.get("attempt_history", [])
        # Legacy runs sometimes replaced a failed formulation with its repair.
        # Only a successful first attempt reliably identifies the artifact on disk.
        match = re.fullmatch(r"train_tasks_record_(?:diag|refine)_iter([1-9]\d*)\.json", checkpoint.name)
        if (mode == "learn" and match and not self.attempt_history and self.output_status
                and self.output_status[-1] == "optimal" and len(self.output_status) == int(match[1]) + 1
                and Path(str(self.id)).name == str(self.id)):
            self._archived_formulation_path = checkpoint.parent / f"task_{self.id}" / f"model_iter_{match[1]}.txt"


def _catalog_rows(catalog, path, mode):
    version = catalog.get("format")
    if version not in {"opera-laflor-v1", "opera-laflor-v2"}:
        raise ValueError(f"Unknown task catalog format in {path}")
    directory = path.parent
    rows, infeasible = [], []
    for entry in catalog["tasks"]:
        if entry["status"] == "infeasible":
            infeasible.append(entry["task_id"])
            continue
        if mode == "learn":
            split = entry["source_split"] if version == "opera-laflor-v1" else entry.get("source_split", "train")
            if split != "train":
                continue
        row = {field: entry[field] for field in ("task_id", "ground_truth")}
        row["description"] = (directory / entry["description"]).read_text(encoding="utf-8")
        if mode == "learn":
            row["formulation"] = (directory / entry["formulation"]).read_text(encoding="utf-8")
            row["correct_program"] = (directory / entry["reference"]).read_text(encoding="utf-8")
        rows.append(row)
    print(f"Public LAFLOR examples: {len(catalog['tasks'])} task IDs, {len(rows)} numerical tasks for {mode}.")
    if infeasible:
        print("Infeasible references, verified separately: " + ", ".join(infeasible))
    return rows


class DataLoader:
    """An ordered task collection backed by a CSV, an OPERA catalog or a checkpoint."""

    def __init__(self, data_path=None, mode="learn", task_list=None, filter_success_num=None, reset=False):
        self.mode = mode
        self.tasks = task_list if task_list is not None else []
        if task_list is not None or not data_path:
            return
        suffix = Path(data_path).suffix
        if suffix == ".json":
            self.load_from_json(data_path, filter_success_num=filter_success_num, reset=reset)
        elif suffix == ".csv":
            self._load_from_csv(data_path, mode)
        else:
            raise ValueError(f"Unsupported file type for data_path: {data_path}")

    def _load_from_csv(self, path, mode):
        for _, row in pd.read_csv(path).iterrows():
            target = row.get("ground_truth")
            if pd.notna(target):
                self.tasks.append(Task(row.get("task_id"), row.get("description"), target,
                                       row.get("formulation") if mode == "learn" else None,
                                       row.get("optimal_code") if mode == "learn" else None))

    def load_from_json(self, path, filter_success_num=None, reset=False):
        checkpoint = Path(path)
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        rows = _catalog_rows(payload, checkpoint, self.mode) if isinstance(payload, dict) else payload
        loaded = []
        for row in rows:
            target = row.get("ground_truth")
            if target is None or (isinstance(target, float) and math.isnan(target)):
                continue
            target = float(target) if isinstance(target, str) else target
            if filter_success_num is not None and row.get("success_count", 0) >= filter_success_num:
                continue
            task = Task(row.get("task_id"), row.get("description"), target,
                        row.get("formulation") if self.mode == "learn" else None,
                        row.get("correct_program") if self.mode == "learn" else None,
                        tag=row.get("tag"), cluster=row.get("cluster"))
            if not reset:
                task.restore_progress(row, checkpoint, self.mode)
            loaded.append(task)
        self.tasks = loaded

    def save_as_json(self, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            stream.write("[\n")
            for index, task in enumerate(self):
                if index:
                    stream.write(",\n")
                record = task.to_dict(self.mode)
                history = json.dumps(record.pop("retrieved_insights", []), ensure_ascii=False, separators=(",", ":"))
                body = json.dumps(record, ensure_ascii=False, indent=2)[:-1]
                stream.write(body + ',\n  "retrieved_insights": ' + history + "\n}")
            stream.write("\n]\n" if self.tasks else "]\n")

    def save_as_csv(self, path):
        pd.DataFrame([task.to_dict() for task in self]).to_csv(path, index=False)

    def slice(self, start: int, end: int):
        return DataLoader(task_list=self.tasks[start:end], mode=self.mode)

    def __getitem__(self, index):
        return self.tasks[index]

    def __len__(self):
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def subset_by_ids(self, id_lst, inplace=False):
        by_id = {task.id: task for task in self}
        selected = [by_id[key] for key in id_lst if key in by_id]
        if inplace:
            self.tasks = selected
            return self
        return DataLoader(task_list=selected, mode=self.mode)
