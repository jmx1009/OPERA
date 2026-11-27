"""Offline regression tests. All model calls are mocked."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.config import MODEL, SERVICE
from src.dataloader import DataLoader, Task
from src.insight_library import InsightLibrary, Insight
from src.insight_states import (
    INSIGHT_STATES, normalize_distribution, normalize_insight_record,
    normalize_reflection_record, supports_formulation,
)
from src.llm_diagnostic import ProgramDiagnostic
from src.condition_refiner import ApplicabilityRefiner, parse_refinement_candidates
from src.prompts.prompts_diag import PROMPT_INS_REFLECTION
from src.prompts.prompts_refinement import PROMPT_INS_IRRELEVANT, PROMPT_INS_REFINEMENT
from library_refinement import _normalize_partition, matching_score, run_library_refinement
from src.insight_states import migrate_records


def make_insight(distribution=None):
    return {
        "insight_id": 1, "taxonomy": {"Domain Modeling": ["Capacity"]},
        "condition": "Original condition", "explanation": "Explanation", "example": "Example",
        "distribution": distribution or {}, "source_refs": ["source"],
    }


class StateTests(unittest.TestCase):
    def test_alias_merge_and_deduplication(self):
        old = {"positive": ["a"], "useful": ["a", "b"], "retrieved_unapplied": ["c"],
               "irrelevant": ["d", "c"], "invalid": ["e"], "unretrieved": ["f"]}
        original = copy.deepcopy(old)
        result = normalize_distribution(old)
        self.assertEqual(result, {"useful": ["a", "b"], "negative": [],
                                  "irrelevant": ["c", "d", "e"], "unretrieved": ["f"]})
        self.assertEqual(old, original)
        self.assertEqual(normalize_distribution(result), result)

    def test_defaults_are_independent(self):
        first, second = normalize_distribution(None), normalize_distribution([])
        first["useful"].append("a")
        self.assertEqual(second["useful"], [])

    def test_legacy_evidence_and_recovery_are_preserved(self):
        old = make_insight({"positive": ["a"], "retrieved_unapplied": ["b"],
                            "irrelevant": ["c"], "recovered_unretrieved": ["d"]})
        new = normalize_insight_record(old)
        self.assertEqual(set(new["distribution"]), set(INSIGHT_STATES))
        self.assertEqual(new["legacy_applicable_task_ids"], ["b"])
        self.assertEqual(new["retrieval_evaluation"]["recovered_unretrieved"], ["d"])
        self.assertEqual(new, normalize_insight_record(new))
        self.assertEqual(new["source_refs"], old["source_refs"])

    def test_load_save_compatibility(self):
        original = make_insight({"postive": ["a"], "invalid": ["b"]})
        library = InsightLibrary(insight_list=[original])
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/library.json"
            library.save(path)
            loaded = InsightLibrary.from_json_file(path)
        self.assertEqual(library.to_json(), loaded.to_json())
        self.assertEqual(loaded[0].distribution["irrelevant"], ["b"])

    def test_only_useful_is_retained_in_repair(self):
        for record, expected in [
            ({"state": "useful"}, True), ({"state": "negative"}, False),
            ({"state": "irrelevant", "applicable": True}, False),
            ({"state": "irrelevant", "applicable": False}, False),
            ({"state": "irrelevant"}, False), ({"state": "retrieved_unapplied"}, False),
        ]:
            with self.subTest(record=record):
                self.assertEqual(supports_formulation(record), expected)

    def test_legacy_reflection_records(self):
        self.assertEqual(normalize_reflection_record({"state": "invalid"}),
                         {"state": "irrelevant", "applicable": True})

    def test_migration_preserves_non_insight_records(self):
        data = [{"task_id": "x", "description": "Unchanged"}, make_insight({"positive": ["x"]})]
        new = migrate_records(data)
        self.assertEqual(new[0], data[0])
        self.assertEqual(new[1]["distribution"]["useful"], ["x"])
        self.assertEqual(new, migrate_records(new))

    def test_migration_rejects_unknown_fields(self):
        with self.assertRaises(ValueError):
            migrate_records([make_insight({"unknown": ["x"]})])


class ReflectionTests(unittest.TestCase):
    def test_parser_accepts_new_and_legacy_states(self):
        result = ProgramDiagnostic._parse_insight_reflection(json.dumps([
            {"insight_id": 1, "state": "positive"},
            {"insight_id": 2, "state": "retrieved_unapplied"},
            {"insight_id": 3, "state": "irrelevant", "evidence": "Unrelated action adopted"},
        ]), [1, 2, 3])
        self.assertEqual([r["state"] for r in result], ["useful", "irrelevant", "irrelevant"])

    def test_parser_rejects_invalid_or_incomplete_reflections(self):
        for records in [[], [{"insight_id": 1, "state": "unknown"}],
                        [{"insight_id": 1, "state": "irrelevant", "applicable": "false"}],
                        [{"insight_id": 1, "state": "unretrieved"}],
                        [{"insight_id": 2, "state": "useful"}],
                        [{"insight_id": 1, "state": "useful"}] * 2]:
            with self.subTest(records=records), self.assertRaises(ValueError):
                ProgramDiagnostic._parse_insight_reflection(json.dumps(records), [1])

    def test_irrelevant_clause_is_exclusion_or_empty(self):
        exclusion = {"condition": "This insight does NOT apply when capacity is fixed.",
                     "reason": "The task fixes capacity before optimization."}
        self.assertEqual(ApplicabilityRefiner._parse_irrelevant_condition(json.dumps(exclusion)), exclusion)
        self.assertEqual(ApplicabilityRefiner._parse_irrelevant_condition("{}"), {})
        for wrong in ({"condition": "This insight applies when ..."}, {**exclusion, "applicable": True},
                      {"condition": exclusion["condition"]}, {**exclusion, "condition": "Check the capacity."}):
            with self.assertRaises(ValueError):
                ApplicabilityRefiner._parse_irrelevant_condition(json.dumps(wrong))

    def test_old_log_fallback_and_new_log_precedence(self):
        insight = Insight(make_insight())
        task = Task("t", "Description", 1)
        evolver = ApplicabilityRefiner(InsightLibrary(), MODEL, SERVICE)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "task_t" / "Diagnosis"
            folder.mkdir(parents=True)
            (folder / "ins_pos_neg_diagnosis_iter_1.json").write_text(json.dumps([
                {"insight_id": 1, "state": "retrieved_unapplied", "evidence": "legacy evidence"}
            ]))
            evidence = evolver._load_reflection_evidence(task, insight, 1, directory, {"irrelevant"})
            self.assertIn("true", evidence)
            self.assertIn("legacy evidence", evidence)
            (folder / "insight_reflection_iter_1.json").write_text(json.dumps([
                {"insight_id": 1, "state": "irrelevant", "applicable": False, "evidence": "current evidence"}
            ]))
            evidence = evolver._load_reflection_evidence(task, insight, 1, directory, {"irrelevant"})
            self.assertIn("false", evidence)
            self.assertIn("current evidence", evidence)

    def test_legacy_applicability_without_logs(self):
        insight = Insight(make_insight({"retrieved_unapplied": ["t"]}))
        evolver = ApplicabilityRefiner(InsightLibrary(), MODEL, SERVICE)
        self.assertIn("applicable", evolver._load_reflection_evidence(
            Task("t", "Description", 1), insight, 1, states={"irrelevant"}))

    def test_diagnosis_retains_only_supported_insights(self):
        diagnostic = ProgramDiagnostic(MODEL, SERVICE)
        task = Task("t", "Description", 1)
        records = [
            {"insight_id": 1, "state": "useful", "applicable": True},
            {"insight_id": 2, "state": "irrelevant", "applicable": True},
            {"insight_id": 3, "state": "irrelevant", "applicable": False},
            {"insight_id": 4, "state": "negative", "applicable": False},
        ]
        optimizer = SimpleNamespace(
            generate_formulation=lambda **kwargs: captured.append(kwargs["retrieved_insights"]) or "corrected",
            generate_program=lambda **kwargs: ("program", 1, True, False),
        )
        captured = []
        def fake_call(**kwargs):
            return kwargs["parse_fn"](json.dumps(records))
        with patch("src.llm_diagnostic.call_llm_and_parse_with_retry", side_effect=fake_call), \
             patch("src.llm_diagnostic.check_optimality", return_value=(True, 0, "correct")):
            result = diagnostic._reflect_insight_usage(
                iter=1, task=task, failed_formulation="failed", diagnosed_issues=[],
                retrieved_insights=[[{"insight_id": i} for i in range(1, 5)] + [{"task_specific_insight": "keep"}], []],
                llm_opt=optimizer,
            )
        self.assertEqual([r.get("insight_id") for r in captured[0]], [None, 1])
        self.assertFalse(result[2])

    def test_successful_filtering_return_contract(self):
        diagnostic = ProgramDiagnostic(MODEL, SERVICE)
        with patch.object(diagnostic, "_diagnose_issues", return_value=[]), \
             patch.object(diagnostic, "_reflect_insight_usage", return_value=(
                 {1: "useful"}, [{"insight_id": 1}], False, "corrected", None)):
            result = diagnostic.diagnose_formulation(
                task=Task("t", "Description", 1), failed_formulation="failed",
                retrieved_insights=[[], []],
            )
        self.assertEqual(result, ({1: "useful"}, [{"insight_id": 1}], False, "corrected"))

    def test_prompt_templates_format_without_obsolete_states(self):
        for template in (PROMPT_INS_REFLECTION, PROMPT_INS_IRRELEVANT, PROMPT_INS_REFINEMENT):
            import string
            fields = {name: "value" for _, name, _, _ in string.Formatter().parse(template) if name}
            formatted = template.format(**fields)
            self.assertNotIn("retrieved_unapplied", formatted)
            self.assertNotIn('"positive"', formatted)


class RefinementTests(unittest.TestCase):
    def test_t16_requires_the_entire_response_and_exact_candidate_fields(self):
        original = "Capacity is a decision."
        candidate = {"path_id": 1, "strategy": "Specify when capacity varies.",
                     "new_condition": original + "\n\nThis insight applies when capacity varies by period."}
        valid = json.dumps([candidate])
        for invalid in (
            "Candidates follow.\n" + valid, valid + "\nPath 1 is best.", "```json\n" + valid + "\n```",
            json.dumps(candidate), json.dumps([None]), json.dumps([{**candidate, "score": 1}]),
            json.dumps([{**candidate, "selected": True}]), valid.replace('"path_id": 1', '"path_id": 2, "path_id": 1'),
        ):
            with self.subTest(text=invalid), self.assertRaises(ValueError):
                parse_refinement_candidates(invalid, original, 1, applicability=True, exclusion=False)
        self.assertEqual(parse_refinement_candidates(" \n" + valid + "\n", original, 1,
                                                     applicability=True, exclusion=False), [candidate])

    def test_t16_validates_only_the_new_paragraphs_against_this_round(self):
        original = "Capacity is a decision.\n\nThis insight applies when pods share a resource."
        positive = "This insight applies when capacity varies by period."
        negative = "This insight does NOT apply when capacity is fixed in advance."
        for suffix, allow_positive, allow_negative in [
                (positive, True, False), (negative, False, True), (positive + "\n\n" + negative, True, True)]:
            record = {"path_id": 1, "strategy": "Qualify the capacity decision.",
                      "new_condition": original + "\n\n" + suffix}
            self.assertEqual(parse_refinement_candidates(json.dumps([record]), original, 1,
                             applicability=allow_positive, exclusion=allow_negative), [record])

    def test_t16_rejects_unscorable_batches_before_retrieval(self):
        original = "Keep the Big–M condition."
        first = {"path_id": 1, "strategy": "Add a capacity trigger.",
                 "new_condition": original + "\n\nThis insight applies when capacity varies."}
        second = {"path_id": 2, "strategy": "Narrow the trigger.",
                  "new_condition": original + "\n\nThis insight applies when usable capacity varies by period."}
        bad_texts = [
            original + "\n\nCheck whether capacity varies.",
            original + "\n\nThis insight applies when ...",
            original + "\nThis insight applies when capacity varies.",
            second["new_condition"].replace("Big–M", "Big-M"),
            second["new_condition"] + "\n\nDetermine the required mass.",
            original + "\n\nThis insight does NOT apply when capacity is fixed.",
            second["new_condition"] + " This insight does NOT apply when the tariff is fixed.",
        ]
        invalid = [[first], [first, {**second, "path_id": "1"}], [first, {**second, "strategy": ""}],
                   [first, {**second, "new_condition": first["new_condition"] + " "}]]
        invalid.extend([first, {**second, "new_condition": text}] for text in bad_texts)
        for batch in invalid:
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                parse_refinement_candidates(json.dumps(batch), original, 2, applicability=True, exclusion=False)
        for applicability, exclusion in ((False, False), (False, True)):
            with self.assertRaises(ValueError):
                parse_refinement_candidates(json.dumps([first]), original, 1,
                                            applicability=applicability, exclusion=exclusion)

    def make_case(self):
        distribution = {"useful": ["u", "outside"], "negative": ["n"],
                        "irrelevant": ["a", "i", "unknown"], "unretrieved": ["r"]}
        library = InsightLibrary([make_insight(distribution)])
        task_list = []
        for task_id in ("u", "n", "a", "i", "unknown", "r"):
            task = Task(task_id, "Description", 1)
            task.output_status = ["optimal" if task_id == "u" else "failed"]
            task.retri_ins_lst = [[] if task_id == "r" else [1]]
            task_list.append(task)
        return library, DataLoader(task_list=task_list)

    def test_partition_preserves_outside_split(self):
        library, tasks = self.make_case()
        partition, distribution = _normalize_partition(library[0], tasks)
        self.assertEqual(partition["useful"], ["u"])
        self.assertIn("outside", distribution["useful"])
        self.assertEqual(set(distribution), set(INSIGHT_STATES))

    def test_scoring_includes_every_irrelevant_task_as_an_exclusion(self):
        positives, exclusions = {"u", "r"}, {"n", "a", "i", "unknown"}
        self.assertEqual(matching_score(positives, exclusions, {"u", "r"}), 1)
        self.assertEqual(matching_score(positives, exclusions, {"u", "a", "i", "n", "unknown"}), 1/6)
        with self.assertRaises(ValueError):
            matching_score({"a"}, {"a"}, set())

    def run_case(self, matches):
        library, tasks = self.make_case()
        evolver = ApplicabilityRefiner(library, MODEL, SERVICE)
        config = SimpleNamespace(params=SimpleNamespace(variant_num=1), model=MODEL, service=SERVICE)
        def irrelevant_condition(task, *args, **kwargs):
            if task.id == "unknown":
                return {}
            return {"condition": "This insight does NOT apply when the resource is fixed.", "reason": "Evidence"}
        with patch.object(evolver, "generate_neg_condition", return_value="This insight does NOT apply when supply is fixed."), \
             patch.object(evolver, "generate_irrelevant_condition", side_effect=irrelevant_condition), \
             patch.object(evolver, "generate_unr_condition", return_value="This insight applies when supply is a decision."), \
             patch.object(evolver, "refine_insight", return_value=[{
                 "path_id": "fixed", "strategy": "Exclude fixed resources.",
                 "new_condition": "Original condition\n\nThis insight does NOT apply when the resource is fixed."}]), \
             patch.object(evolver, "verify_retrieval",
                          side_effect=RuntimeError("Unavailable") if matches is None else None,
                          return_value=(len(matches or []), matches)) as verify:
            result, gain = run_library_refinement(1, tasks, config, evolver, max_workers=1)
        self.assertEqual(set(verify.call_args.args[2]), {"u", "a", "n", "i", "unknown", "r"})
        self.assertEqual(library[0].condition, "Original condition")
        return result[0], gain

    def test_improvement_and_evaluation_metadata(self):
        insight, gain = self.run_case(["u", "a", "r"])
        self.assertTrue(insight.condition.startswith("Original condition\n\n"))
        self.assertAlmostEqual(gain, 4/6)
        self.assertEqual(set(insight.distribution), set(INSIGHT_STATES))
        self.assertIn("outside", insight.distribution["useful"])
        self.assertIn("r", insight.distribution["unretrieved"])
        evaluation = insight.metadata["retrieval_evaluation"]["iterations"]["1"]
        choice = evaluation["variant_scores"][evaluation["selected_variant_index"] - 1]
        self.assertEqual(insight.condition, choice["condition"])
        self.assertEqual(evaluation["baseline_score"], matching_score(
            evaluation["should_retrieve"], evaluation["should_avoid"], evaluation["baseline_matched_task_ids"]))
        self.assertEqual(choice["score"], matching_score(
            evaluation["should_retrieve"], evaluation["should_avoid"], choice["matched_task_ids"]))
        self.assertEqual(evaluation["recovered_unretrieved"], ["r"])
        self.assertEqual(evaluation["irrelevant_without_clause"], ["unknown"])

    def test_no_improvement_retains_condition(self):
        insight, gain = self.run_case(["u", "a", "n", "i", "unknown"])
        self.assertEqual(insight.condition, "Original condition")
        self.assertEqual(gain, 0)
        self.assertIsNone(insight.metadata["retrieval_evaluation"]["iterations"]["1"]["selected_variant_index"])

    def test_failed_verification_has_no_score_or_selected_path(self):
        insight, gain = self.run_case(None)
        evaluation = insight.metadata["retrieval_evaluation"]["iterations"]["1"]
        failure = evaluation["variant_scores"][0]
        self.assertEqual((insight.condition, gain), ("Original condition", 0))
        self.assertEqual((failure["path_id"], failure["score"], failure["error"]), ("fixed", None, "Unavailable"))
        self.assertNotIn("matched_task_ids", failure)
        self.assertIsNone(evaluation["selected_variant_index"])


if __name__ == "__main__":
    unittest.main()
