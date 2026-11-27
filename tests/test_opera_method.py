"""Offline checks of OPERA's two libraries, evidence routing and acceptance rules."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from evaluation import check_evaluation_protocol, evaluate, run_evaluations
from library_online_learning import run_library_online_learning
from library_refinement import refinement_counts, run_library_refinement
from src.config import MODEL, SERVICE
from src.dataloader import DataLoader, Task
from src.insight_library import InsightLibrary
from src.lesson_merge import consolidate_lessons, integrate_lesson
from src.llm_diagnostic import ProgramDiagnostic
from src.condition_refiner import ApplicabilityRefiner
from src.llm_extractor import LessonExtractor
from src.llm_retriever import LibraryRetrieval
from src.train_eval_utils import check_optimality, save_checkpoint, self_verify_test
from src.library_loading import load_working_library
from tools.consolidate_principles import compact_group


def rule(i, kind="lesson", stage="General Formulation", **fields):
    return {"insight_id": i, "insight_type": kind, "taxonomy": {stage: {"Capacity": ["Limits"]}},
            "condition": "Use for bounded resources.", "explanation": f"Apply rule {i}.",
            "example": "x <= capacity", **fields}


class WorkingLibraryTests(unittest.TestCase):
    def test_initial_lesson_source_label_keeps_verification_attempts(self):
        attempts = [["E000", 0, 1], ["LAFLOR_001", 1, 2]]
        lesson = rule(1, source_task_ids=["A000", "initial lessons"], source_attempts=attempts)
        record = InsightLibrary([lesson]).lessons[0].to_dict()
        self.assertEqual(record["source_task_ids"], ["initial lessons", "LAFLOR_001"])
        self.assertEqual(record["source_attempts"], attempts)

    def test_legacy_mixed_ids_and_refined_checkpoint_precedence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            principles = [rule(1, "principle", source_type="candidate_principle", source_unit_id="doc-1"),
                     rule(2, "principle", source_type="candidate_principle", source_unit_id="doc-2")]
            stored = [{**principles[0], "insight_id": 19, "condition": "Already refined."}, rule(1)]
            for row in [*principles, *stored]:
                row.pop("insight_type")
            (root / "principles.json").write_text(json.dumps(principles))
            (root / "library.json").write_text(json.dumps(stored))
            cfg = OmegaConf.create({"bootstrap": {"enable_principle_library": True,
                                   "consolidated_principles_path": str(root / "principles.json"), "principle_taxonomy_path": None}})
            lib = load_working_library(cfg, object(), root / "library.json")
            self.assertEqual([(i.insight_id, i.metadata["knowledge_id"]) for i in lib.principles],
                             [(19, "principle1"), (20, "principle2")])
            self.assertEqual(lib.principles[0].condition, "Already refined.")
            self.assertEqual(lib.lessons[0].insight_id, 1)
            lib.save(root / "working.json", checkpoint=True)
            cfg.bootstrap.consolidated_principles_path = "missing-principles.json"
            restored = load_working_library(cfg, object(), root / "working.json")
            self.assertEqual(restored.to_json(), lib.to_json())
            restored.replace_merged_insights([{"insight_id": 20}])
            restored.save(root / "working.json", checkpoint=True)
            restored = load_working_library(cfg, object(), root / "working.json")
            restored.add_insights([rule(-1)], 1)
            self.assertEqual(restored.lessons[-1].insight_id, 21)
            cfg.bootstrap.enable_principle_library = False
            self.assertEqual([r.insight_id for r in load_working_library(cfg, None, root / "working.json")], [1])

    def test_published_ids_and_conflicting_partition(self):
        lib = InsightLibrary([rule("lesson7", legacy_insight_id=91)])
        self.assertEqual(lib.retrieve_insights_by_id(91)[0]["knowledge_id"], "lesson7")
        with self.assertRaises(ValueError):
            InsightLibrary([rule("principle18", "lesson")])
        with self.assertRaises(ValueError):
            InsightLibrary([rule(1), rule(1, "principle")])

    def test_legacy_rules_with_distinct_conditions_and_nested_attempt_ids(self):
        principle = rule(1, "principle", source_unit_id="summary-1")
        other = {**principle, "insight_id": 2, "condition": "Use when optional."}
        lib = InsightLibrary([rule(4), {**other, "insight_id": 7}])
        task = Task("old", "task", 7)
        task.retri_ins_lst = [[[4], [91, "hint"]]]
        lib.reserve_historical_ids([task])
        principles = InsightLibrary([principle, other])
        lib.attach_principles(principles)
        lib.attach_principles(principles)
        self.assertEqual([(i.insight_id, i.metadata["knowledge_id"]) for i in lib.principles],
                         [(7, "principle2"), (92, "principle1")])
        self.assertEqual([i["insight_id"] for i in lib.retrieve_insights_by_id([[[4], ["principle1"]]])], [4, 92])

    def test_legacy_program_stage_uses_identifiable_attempt_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "train_tasks_record_diag_iter1.json"
            rows = []
            for tid, status in (("s", "optimal"), ("f", "not_optimal")):
                task = Task(tid, "task", 7, formulation="reference model")
                task.output_status = [["optimal"] * 3, status]
                rows.append(task.to_dict())
                directory = root / f"task_{tid}"
                directory.mkdir()
                (directory / "model_iter_1.txt").write_text("archived model")
            path.write_text(json.dumps(rows))
            success, failure = DataLoader(str(path))
            self.assertEqual(success.latest_attempted_formulation, "archived model")
            self.assertIsNone(failure.latest_attempted_formulation)
            self.assertIsNone(DataLoader(str(path), mode="test")[0].latest_attempted_formulation)
            legacy_success = DataLoader(str(path))[0]
            with patch.object(Path, "read_text", side_effect=TimeoutError("archived file offline")):
                self.assertIsNone(legacy_success.latest_attempted_formulation)

    def test_stored_merge_validates_batch_and_preserves_document_parent(self):
        lib = InsightLibrary([
            rule(3, "principle"),
            rule(4, task_id="old", iteration=0, source_task_ids=["archived", "old"]),
            rule(5, source_attempts=[["new", 2, 1]], related_principle_ids=[8]),
        ])
        original = lib.to_json()
        with self.assertRaises(ValueError):
            lib.merge_into_library([[3, 4], [99]], [rule(-1), rule(-2)], lib)
        self.assertEqual(lib.to_json(), original)
        lib.merge_into_library([[3, 4, 5]], [rule(-1)], lib)
        self.assertEqual([i.insight_id for i in lib], [3, 6])
        self.assertEqual(lib.lessons[0].metadata["source_attempts"], [["old", 0, None], ["new", 2, 1]])
        self.assertEqual(lib.lessons[0].to_dict()["source_task_ids"], ["archived", "old", "new"])
        self.assertEqual(lib.lessons[0].metadata["related_principle_ids"], [3, 8])

    def test_both_kinds_refine_with_stage_specific_checks_and_survive_resume(self):
        lib = InsightLibrary([rule(1, "principle", distribution={"negative": ["n"]}),
                                 rule(2, "lesson", "Code Implementation", distribution={"irrelevant": ["n"]})])
        tasks = DataLoader(task_list=[Task("u", "positive case", 7), Task("n", "exclusion case", 7)])
        for task in tasks:
            task.retri_ins_lst = [[1, 2]]
            status = "optimal" if task.id == "u" else "not_optimal"
            task.output_status = [status]
            task.record_attempt(1, 1, status, "attempted model for " + task.id, lib.to_json())
        orig = copy.deepcopy(lib.to_json())
        config = OmegaConf.create({"params": {"variant_num": 2}, "model": MODEL, "service": SERVICE})
        candidates = ["Use for bounded resources.\n\nThis insight does NOT apply when " + t for t in ("fixed.", "unlimited.")]
        evolution = ApplicabilityRefiner(lib, MODEL, SERVICE)
        def clause(**kwargs):
            if "path_id" in kwargs["prompt"]:
                kwargs["response_records"].append({"completion_id": "test-generation"})
                return kwargs["parse_fn"](json.dumps([
                    {"path_id": index, "strategy": f"Test boundary {index}.", "new_condition": condition}
                    for index, condition in enumerate(candidates, 1)]))
            return kwargs["parse_fn"](json.dumps({"condition": "This insight does NOT apply when fixed.",
                                                  "reason": "The resource is fixed in this task."}))
        def selection(**kwargs):
            prompt = kwargs["prompt"]
            kwargs["response_records"].append({"completion_id": "test-retrieval"})
            self.assertNotIn("distribution", prompt)
            self.assertNotIn("source_attempts", prompt)
            is_code = '"insight_id": 2' in prompt
            self.assertEqual(is_code, "attempted model for " in prompt)
            chosen = [] if "fixed." in prompt and "exclusion case" in prompt else [{"insight_id": 2 if is_code else 1}]
            return kwargs["parse_fn"](json.dumps(chosen))
        with tempfile.TemporaryDirectory() as folder, \
                patch("src.condition_refiner.call_llm_and_parse_with_retry", side_effect=clause), \
                patch("src.llm_retriever.call_llm_and_parse_with_retry", side_effect=selection), \
                patch.object(LibraryRetrieval, "quick_match_by_taxonomy", side_effect=AssertionError("taxonomy leak")):
            result, gain = run_library_refinement(1, tasks, config, evolution, max_workers=1)
            self.assertEqual(gain, 0.5)
            self.assertEqual(refinement_counts(result, 1), {k: {"evaluated": 1, "changed": 1} for k in ("principle", "lesson")})
            paths = SimpleNamespace(lib_dir=folder, train_output_dir=folder, metrics_log_path=f"{folder}/metrics.json")
            save_checkpoint(result, tasks, [], paths, "refine_iter1")
            checkpoint = Path(folder) / "library_refine_iter1.json"
            document = json.loads(checkpoint.read_text())
            self.assertEqual([len(document[k]) for k in ("principles", "lessons")], [1, 1])
            restored = load_working_library(config, None, checkpoint)
            records = DataLoader(f"{folder}/train_tasks_record_refine_iter1.json")
            self.assertEqual(records[0].latest_attempted_formulation, "attempted model for u")
            for record in restored:
                self.assertEqual(record.condition, candidates[0])
                audit = record.metadata["retrieval_evaluation"]["iterations"]["1"]
                self.assertEqual([row["condition"] for row in audit["variant_scores"]], candidates)
                self.assertEqual([row["path_id"] for row in audit["variant_scores"]], [1, 2])
                self.assertEqual([row["strategy"] for row in audit["variant_scores"]],
                                 ["Test boundary 1.", "Test boundary 2."])
                self.assertEqual([row["matched_task_ids"] for row in audit["variant_scores"]], [["u"], ["n", "u"]])
                self.assertEqual(audit["selected_variant_index"], 1)
                self.assertEqual(audit["generation_responses"], [{"completion_id": "test-generation"}])
                for variant in audit["variant_scores"]:
                    self.assertEqual({c["task_id"] for c in variant["checks"]}, {"n", "u"})
                    self.assertEqual({c["task_id"] for c in variant["checks"] if c["retrieved"]},
                                     set(variant["matched_task_ids"]))
                    self.assertTrue(all(c["responses"] == [{"completion_id": "test-retrieval"}]
                                        for c in variant["checks"]))
                self.assertEqual(audit["clause_evidence"][0]["task_id"], "n")
                self.assertEqual(audit["original_condition"], orig[record.insight_id-1]["condition"])
                self.assertEqual((audit["refinement_model"], audit["retrieval_model"]), (MODEL, MODEL))
                for field in ("example", "explanation", "taxonomy", "insight_id", "insight_type"):
                    self.assertEqual(record.to_dict()[field], orig[record.insight_id-1][field])
        self.assertEqual(lib.to_json(), orig)

    def test_unscorable_conditions_and_false_exclusions_never_win(self):
        task = Task("t", "Old task", 0)
        lib = InsightLibrary([rule(1, "principle", "Code Implementation", distribution={"negative": ["t"]})])
        evolution = ApplicabilityRefiner(lib, MODEL, SERVICE)
        retriever = LibraryRetrieval(lib, MODEL, SERVICE)
        with self.assertRaisesRegex(ValueError, "No attempted formulation"):
            evolution.verify_retrieval(1, DataLoader(task_list=[task]), ["t"], retriever)
        task.record_attempt(1, 1, "not_optimal", "actual formulation", lib.to_json())
        with patch("src.llm_retriever.call_llm_and_parse_with_retry", side_effect=RuntimeError("unavailable")):
            with self.assertRaisesRegex(RuntimeError, "Applicability check failed"):
                evolution.verify_retrieval(1, DataLoader(task_list=[task]), ["t"], retriever)
        def invalid_response(**kwargs):
            return kwargs["parse_fn"](json.dumps([{"path_id": 1, "strategy": "Rewrite.", "new_condition": "Discard original."}]))
        with patch("src.condition_refiner.call_llm_and_parse_with_retry", side_effect=invalid_response):
            self.assertEqual(evolution.refine_insight(1, ["exclude"], [], [], lib[0], path_k=1), [])


class VerificationTests(unittest.TestCase):
    def test_failed_attempt_uses_its_own_context(self):
        task = Task("t", "Bounded resources", 10, correct_program="reference")
        contexts = [[rule(i, "principle"), {"memory_id": f"hint_{i}", "taxonomy": {"General Formulation": {}},
                                            "explanation": "transient"}] for i in (1, 2, 3)]
        attempts = [(context, "model", "program", value, True, False)
                    for context, value in zip(contexts, (10, 5, 10))]
        extractor = Mock()
        extractor.generate_lessons.return_value = [rule(-1, task_id="t")]
        extractor.propose_online_lesson_merge.return_value = ([], {}, [])
        library = InsightLibrary()
        with tempfile.TemporaryDirectory() as root, \
                patch("library_online_learning.generate_solution_with_retrieval", side_effect=attempts), \
                patch("library_online_learning.self_verify_test", return_value=True) as verify:
            result = run_library_online_learning(
                0, DataLoader(task_list=[task]), None, Mock(), Mock(), extractor, library,
                SimpleNamespace(max_solution_attempts=3, max_verify_attempts=3, batch_size=1),
                SimpleNamespace(train_output_dir=root, lib_dir=root),
            )
        self.assertEqual(result["train_accuracy"], 0)
        self.assertEqual(task.output_status, [["optimal", "not_optimal", "optimal"]])
        self.assertEqual(verify.call_args.args[4], contexts[1])
        self.assertEqual(library.lessons[0].metadata["source_attempts"], [["t", 0, 2]])

    def test_training_uses_a_supplied_reference_formulation(self):
        task = Task("capacity", "Choose a capacity.", 5, formulation="Minimize capacity subject to capacity >= 5.")
        generator = Mock()
        extractor = Mock()
        extractor.generate_lessons.return_value = []
        with tempfile.TemporaryDirectory() as directory, patch(
            "library_online_learning.generate_solution_with_retrieval",
            return_value=([], "failed model", "program", 2, True, False),
        ):
            run_library_online_learning(
                0, DataLoader(task_list=[task]), None, generator, Mock(), extractor, InsightLibrary(),
                SimpleNamespace(max_solution_attempts=1, max_verify_attempts=1, batch_size=1),
                SimpleNamespace(train_output_dir=directory, lib_dir=directory),
            )
        extractor.generate_lessons.assert_called_once()
        self.assertIsNone(task.correct_program)
        self.assertEqual(generator.mock_calls, [])

    def test_online_merge_preserves_principle_and_tests_without_replaced_lesson(self):
        for accepted in (True, False):
            original = rule(7, task_id="t", iteration=0, source_attempts=[["t", 0, 1]])
            principle = rule(8, "principle")
            new = rule(-1, task_id="t", source_attempts=[["t", 0, 1]])
            lib = InsightLibrary([original, principle])
            task = Task("t", "Bounded", 3)
            task.record_attempt(0, 1, "not_optimal", "model", [original, principle, {"memory_id": "hint"}])
            extractor = LessonExtractor(MODEL, SERVICE)
            proposal = {"merged_ids": [7, 8], "condition": "combined", "explanation": "combined", "example": "combined"}
            with patch("src.llm_extractor.call_llm_and_parse_with_retry", return_value=proposal.copy()), \
                    patch("src.lesson_merge.self_verify_test", return_value=accepted) as verify:
                self.assertEqual(integrate_lesson(new, [], [task], lib, object(), extractor, 1), accepted)
            self.assertEqual(verify.call_args.args[4], [principle, {"memory_id": "hint"}])
            self.assertEqual(lib.principles[0].condition, principle["condition"])
            self.assertEqual([row.insight_id for row in lib.lessons], [9] if accepted else [7, 9])
            if accepted:
                self.assertEqual(lib.lessons[0].taxonomy, new["taxonomy"])
                self.assertEqual(lib.lessons[0].metadata["related_principle_ids"], [8])

    def test_batch_proposal_rejected_if_one_source_fails(self):
        tasks = [Task(i, "task", 1) for i in ("a", "b")]
        originals = [rule(-1, task_id=t.id, source_attempts=[[t.id, 0, 1]]) for t in tasks]
        for t in tasks:
            t.record_attempt(0, 1, "not_optimal", t.id, [])
        extractor = Mock()
        extractor.propose_lesson_merge.return_value = [{**rule(-1), "merged_ids": [0, 1]}]
        with patch("src.lesson_merge.self_verify_test", side_effect=[True, False]) as verify:
            result, proposed, accepted = consolidate_lessons(originals, tasks, InsightLibrary(), Mock(), extractor, "batch")
        self.assertEqual((len(result), proposed, accepted), (2, 1, 0))
        self.assertEqual(verify.call_count, 2)

    def test_repair_uses_issue_ids_and_rejects_missing_evidence(self):
        diagnostic = ProgramDiagnostic(MODEL, SERVICE)
        opt = Mock()
        opt.generate_formulation.return_value = "corrected"
        opt.generate_program.return_value = ("program", 4, True, False)
        retriever = Mock()
        retriever.retrieve_insights_for_diagnosis.return_value = {2: [rule(8)], 1: [rule(7), rule(6)]}
        args = dict(task=Task("t", "task", 4), failed_formulation="failed", retrieved_insights=[[], []],
                    diagnosed_issues=[{"id": 1}, {"id": 2}], llm_opt=opt, llm_retri=retriever, previously_retrieved=[6])
        with patch("src.llm_diagnostic.call_llm_and_parse_with_retry", return_value=[]):
            states, kept, needs_lesson, _ = diagnostic._diagnose_unretrieved(**args)
        self.assertEqual((states, kept, needs_lesson), ({}, [], True))
        opt.generate_program.assert_not_called()
        with patch("src.llm_diagnostic.call_llm_and_parse_with_retry",
                   return_value=[{"id": 1, "status": "solved"}, {"id": 2, "status": "unsolved"}]):
            states, kept, needs_lesson, _ = diagnostic._diagnose_unretrieved(**args)
        self.assertEqual((states, [i["insight_id"] for i in kept], needs_lesson), ({7: "unretrieved"}, [7], False))

    def test_objective_contract_and_one_shot_evaluation(self):
        task = Task("t", "task", -100)
        for value, expected in [(-100.05, True), (-99, False), (float("inf"), False), (float("nan"), False), (True, False)]:
            self.assertEqual(check_optimality(task, value, True, False)[0], expected)
        self.assertFalse(check_optimality(task, -100, False, False)[0])
        self.assertFalse(check_optimality(task, -100, True, True)[0])
        zero = Task("z", "task", 0)
        for value in (-0.1, 0, 0.1):
            self.assertTrue(check_optimality(zero, value, True, False)[0])
        for value in (-0.100001, 0.100001):
            self.assertFalse(check_optimality(zero, value, True, False)[0])
        small = Task("s", "task", 0.25)
        self.assertTrue(check_optimality(small, 0.30, True, False)[0])
        large = Task("l", "task", 2**54)
        self.assertFalse(check_optimality(large, 2**54 + 1, True, False)[0])
        self.assertFalse(check_optimality(zero, 1e-12, True, False, 0.01, "relative")[0])
        self.assertTrue(check_optimality(task, -99, True, False, 0.01, "relative")[0])
        self.assertFalse(self_verify_test(0, zero, None, [], []))
        cfg = OmegaConf.load("eval_params.yaml")
        opt = Mock()
        opt.generate_formulation.return_value = "form"
        opt.generate_program.return_value = (None, None, None, None)
        self.assertEqual(evaluate([task], opt, False, None, None, cfg), (0, 0, 1, 0))
        self.assertEqual(len(task.attempt_history), 1)
        opt.generate_formulation.assert_called_once()
        for budget, retries in [(2, 0), (1, 1)]:
            cfg.pass_at_k, cfg.ablation.max_debug_retry = budget, retries
            with self.assertRaises(ValueError):
                evaluate([task], opt, False, None, None, cfg)
        opt.generate_program.assert_called_once()
        self.assertEqual(task.output_status, ["run_error"])

    def test_repeated_evaluation_averages_run_rates_and_keeps_task_histories(self):
        config = OmegaConf.load("eval_params.yaml")
        self.assertEqual(config.evaluation_runs, 3)
        config.max_workers = 1
        tasks = DataLoader(mode="test", task_list=[Task("a", "A", 1), Task("b", "B", 1)])
        # Disjoint successes in the first two runs; both tasks fail in the third.
        outputs = [([], "formulation", "program", z, True, False) for z in (1, 0, 0, 1, 0, 0)]
        with tempfile.TemporaryDirectory() as folder:
            config.output_folder = folder
            with patch("evaluation.generate_solution_with_retrieval", side_effect=outputs) as generate:
                path = run_evaluations(tasks, Mock(), False, None, None, config, {"dataset": "example"})
            summary = json.loads(path.read_text())
            self.assertEqual(generate.call_count, 6)
            self.assertEqual([row["n_success"] for row in summary["runs"]], [1, 1, 0])
            self.assertAlmostEqual(summary["mean_pass_at_1_rate"], 1 / 3)
            self.assertEqual(summary["mean_execution_rate"], 1)
            self.assertEqual(summary["tasks_per_run"], 2)
            for run in summary["runs"]:
                directory = path.parent / run["directory"]
                records = json.loads((directory / "tasks_record_nolib.json").read_text())
                self.assertEqual([len(record["attempt_history"]) for record in records], [1, 1])
                self.assertEqual(json.loads((directory / "metrics.json").read_text())["run"], run["run"])
            self.assertTrue(all(not task.attempt_history for task in tasks))

        for invalid_count in (0, -1, 1.5, True):
            config.evaluation_runs = invalid_count
            with self.assertRaisesRegex(ValueError, "positive integer"):
                check_evaluation_protocol(config)

    def test_evaluation_uses_its_own_tolerance_mode(self):
        cfg = OmegaConf.load("eval_params.yaml")
        self.assertEqual((cfg.tolerance_mode, cfg.tolerance), ("absolute", 0.1))
        result = ([], "formulation", "program", 100.5, True, False)
        with patch("evaluation.generate_solution_with_retrieval", return_value=result):
            def successes():
                return evaluate([Task("q", "task", 100)], Mock(), False, None, None, cfg)[0]
            self.assertEqual(successes(), 0)
            del cfg.tolerance_mode
            cfg.tolerance = 0.01
            self.assertEqual(successes(), 0)
            cfg.tolerance_mode = "relative"
            self.assertEqual(successes(), 1)

    def test_consolidation_keeps_expert_review_input_grounded(self):
        candidate = rule(1, "principle")
        candidate["source_unit_id"] = "document_l1"
        candidate["source_ref"] = "handbook"
        args = SimpleNamespace(max_input_insights=10, max_group_output=10)
        with patch("tools.consolidate_principles.call_llm_for_group", return_value=[]) as call:
            result, _ = compact_group(("General Formulation", "Capacity"), [candidate], 5, args, {1: candidate}, {})
        call.assert_called_once()
        self.assertEqual(result, [])
        bad = {**candidate, "merged_from": [1, 9]}
        with patch("tools.consolidate_principles.call_llm_for_group", return_value=[bad]):
            result, _ = compact_group(("General Formulation", "Capacity"), [candidate], 5, args, {1: candidate}, {})
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
