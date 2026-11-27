"""Contracts for parsing model output, repair retries, task records and taxonomy lookup."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import MODEL, SERVICE
from src.dataloader import DataLoader, Task
from src.insight_library import InsightLibrary
from src.llm_diagnostic import ProgramDiagnostic
from src.llm_programmer import ProgramGenerator
from src.utils import extract_json_array, extract_json_object


class RuntimeContracts(unittest.TestCase):
    def test_generation_and_repair_response_formats(self):
        generator = ProgramGenerator(MODEL, SERVICE)
        repair = ProgramDiagnostic(MODEL, SERVICE)
        alternatives = "```text\nfirst\n```\n```python\nsecond\n```"
        self.assertEqual(generator.extract_text(alternatives), "first")
        self.assertEqual(generator.extract_code(alternatives, "\nfooter"), "second\nfooter")
        self.assertEqual(repair.extract_code(alternatives), "second")
        self.assertEqual(generator.extract_text("```math\nx <= y"), "x <= y")
        bare = "Code follows\nimport gurobipy as gp\nmodel.optimize()\nThe objective is printed below."
        self.assertEqual(generator.extract_code(bare), "import gurobipy as gp\nmodel.optimize()")
        with self.assertRaises(ValueError):
            repair.extract_code(bare)

    def test_json_scope_and_recovery(self):
        text = '{"earlier":true}\n```json\n[unparseable] [{"example":"x[1], }",}]\n```'
        self.assertEqual(extract_json_object(text), {"earlier": True})
        self.assertEqual(extract_json_array(text), [{"example": "x[1], }"}])
        with self.assertRaises(ValueError):
            extract_json_array("```json\n[1, 2]\n```\n[{}]")

    def test_repair_uses_consecutive_normalized_execution_errors(self):
        repair = ProgramDiagnostic(MODEL, SERVICE)
        task = Task("demo", "Find the minimum.", 4)
        failures = [RuntimeError(f'File "generated.py", line {n}: missing variable') for n in (3, 7, 10)]
        with tempfile.TemporaryDirectory() as folder, patch(
            "src.llm_diagnostic.call_llm_and_parse_with_retry", return_value="print(4)"
        ) as request, patch.object(repair, "execute_code", side_effect=failures):
            result = repair.diagnose_program(2, task, "failed", "error", output_path=folder)
            self.assertEqual(result, (None, False, None, None))
            self.assertEqual(request.call_count, 3)
            self.assertEqual((Path(folder) / "corrected_program_iter_2.py").read_text(), "print(4)")

    def test_reset_filters_on_saved_count_and_hides_test_references(self):
        rows = [
            {"task_id": "first", "description": "a", "ground_truth": "0", "success_count": 2},
            {"task_id": "second", "description": "b", "ground_truth": 1, "success_count": 1,
             "formulation": "reference", "correct_program": "gold", "retrieved_insights": [[7]]},
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tasks.json"
            path.write_text(json.dumps(rows))
            items = DataLoader(str(path), mode="test", filter_success_num=2, reset=True)
            self.assertEqual([task.id for task in items], ["second"])
            self.assertEqual((items[0].formulation, items[0].correct_program, items[0].success_count), (None, None, 0))
            self.assertEqual(items[0].retri_ins_lst, [])

    def test_taxonomy_and_namespaced_id_views(self):
        labels = {"General Formulation": {"Capacity": {"Bound": None}}}
        records = [{"insight_type": "principle", "insight_id": "principle2", "taxonomy": labels,
                    "condition": "Capacity is bounded.", "explanation": "Bound x.", "example": "x <= U"}]
        library = InsightLibrary(records)
        query = {"General Formulation": {"Capacity": ["Bound"]}}
        screened, = library.retrieve_by_taxonomy(query)
        self.assertEqual(list(screened), ["insight_id", "insight_type", "taxonomy", "condition"])
        expanded = library.retrieve_by_taxonomy(labels, include_task_id=True)
        self.assertIn("task_id", expanded[0])
        resolved = library.retrieve_insights_by_id([["principle2"], 2])
        self.assertEqual([r["knowledge_id"] for r in resolved], ["principle2", "principle2"])
        self.assertEqual(library.retrieve_by_taxonomy({"general formulation": query["General Formulation"]}), [])
