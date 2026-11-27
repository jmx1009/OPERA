"""Exercise the training entry point, checkpoint handoff and release library selection."""
import copy
import io
import json
import os
import runpy
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from src.condition_refiner import APPLICABILITY, EXCLUSION
from src.insight_library import InsightLibrary
from src.library_loading import load_working_library


ROOT = Path(__file__).resolve().parents[1]


class RoundCheckpointTests(unittest.TestCase):
    def test_default_libraries_and_release_condition_schema(self):
        training = OmegaConf.load(ROOT / "train_config.yaml")
        evaluation = OmegaConf.load(ROOT / "eval_params.yaml")
        baseline_path = ROOT / training.bootstrap.consolidated_principles_path
        release_path = ROOT / evaluation.bootstrap.consolidated_principles_path
        before = json.loads(baseline_path.read_text())
        after = json.loads(release_path.read_text())
        self.assertEqual(len(before), 210)
        self.assertEqual([p["insight_id"] for p in before], [p["insight_id"] for p in after])
        directions = set()
        for original, revised in zip(before, after):
            self.assertEqual({**original, "condition": revised["condition"]}, revised)
            if original["condition"] == revised["condition"]:
                continue
            prefix = original["condition"] + "\n\n"
            self.assertTrue(revised["condition"].startswith(prefix))
            for clause in revised["condition"][len(prefix):].split("\n\n"):
                opening = next((s for s in (APPLICABILITY, EXCLUSION) if clause.startswith(s)), None)
                self.assertIsNotNone(opening)
                self.assertTrue(clause[len(opening):].strip())
                directions.add(opening)
        self.assertEqual(directions, {APPLICABILITY, EXCLUSION})

        for config, lesson_path, expected_lessons, edition in (
            (training, training.file_paths.initial_library_path, 139, before),
            (evaluation, evaluation.library_path, 363, after),
        ):
            library = load_working_library(config, object(), ROOT / lesson_path)
            self.assertEqual(len(library.lessons), expected_lessons)
            self.assertEqual({p.metadata["knowledge_id"]: p.condition for p in library.principles},
                             {p["insight_id"]: p["condition"] for p in edition})

    def test_four_rounds_and_resume_load_both_partitions_from_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config = OmegaConf.load(ROOT / "train_config.yaml")
            # A nonzero slice checks that resumed rounds reuse the saved task set exactly once.
            config.data_slice = [1, 3]
            config.file_paths.lib_dir = str(folder / "library")
            config.file_paths.train_output_dir = str(folder)
            config.file_paths.metrics_log_path = str(folder / "metrics.json")
            for kind, setting, attr in (
                ("principle", config.bootstrap, "consolidated_principles_path"),
                ("lesson", config.file_paths, "initial_library_path"),
            ):
                row = json.loads((ROOT / setting[attr]).read_text())[0]
                path = folder / f"{kind}.json"
                path.write_text(json.dumps([row]))
                setting[attr] = str(path)
            config_path = folder / "training.yaml"
            starting_conditions = {}
            saved_task_ids = []
            visited_rounds = []

            def learn(iteration, tasks, retriever, generator, diagnostic, extractor, library, params, paths):
                self.assertEqual(iteration, 0)
                self.assertIs(retriever.library, library)
                self.assertEqual((len(library.principles), len(library.lessons)), (1, 1))
                saved_task_ids.extend(task.id for task in tasks)
                self.assertEqual(len(saved_task_ids), 2)
                starting_conditions.update({rule.insight_type: rule.condition for rule in library})
                return {"iter": iteration, "stage": "Experience Library Construction"}

            def clause(round_number):
                return f"\n\nThis insight applies when test case {round_number} is active."

            def diagnose(iteration, tasks, retriever, generator, diagnostic, extractor, library,
                         params, paths, **kwargs):
                visited_rounds.append(iteration)
                self.assertTrue(library.is_checkpoint)
                self.assertIs(retriever.library, library)
                self.assertIs(retriever.document_retriever.taxonomy, library.taxonomy)
                self.assertEqual([task.id for task in tasks], saved_task_ids)
                self.assertEqual((len(library.principles), len(library.lessons)), (1, 1))
                for rule in library:
                    self.assertEqual(rule.condition, starting_conditions[rule.insight_type] +
                                     "".join(clause(n) for n in range(1, iteration)))
                return {"iter": iteration, "stage": "Failure Reflection"}

            def refine(*, iter, llm_evolve, **kwargs):
                result = copy.deepcopy(llm_evolve.library)
                for rule in result:
                    rule.condition += clause(iter)
                return result, 0.0

            def run(first_round, last_round, *, repeat_refinement=None):
                config.start_iter = first_round
                config.params.num_iterations = last_round
                config.resume_refinement_iter = repeat_refinement
                OmegaConf.save(config, config_path)
                with patch.dict(os.environ), patch.object(sys, "argv", ["main.py", "--config", str(config_path)]), \
                        patch("library_online_learning.run_library_online_learning", side_effect=learn), \
                        patch("library_diagnosis.run_library_diagnosis", side_effect=diagnose), \
                        patch("library_refinement.run_library_refinement", side_effect=refine), \
                        patch.object(InsightLibrary, "from_json_file", wraps=InsightLibrary.from_json_file) as reads, \
                        redirect_stdout(io.StringIO()):
                    if repeat_refinement is None:
                        runpy.run_path(str(ROOT / "main.py"), run_name="__main__")
                    else:
                        with self.assertRaises(SystemExit) as stopped:
                            runpy.run_path(str(ROOT / "main.py"), run_name="__main__")
                        self.assertEqual(stopped.exception.code, 0)
                return [Path(call.args[0]).name for call in reads.call_args_list
                        if Path(call.args[0]).parent == folder / "library"]

            expected = ["library_base.json", *(f"library_refine_iter{n}.json" for n in (1, 2, 3))]
            self.assertEqual(run(0, 4), expected)
            self.assertEqual(visited_rounds, [1, 2, 3, 4])
            checkpoint = folder / "library/library_refine_iter4.json"
            original = json.loads(checkpoint.read_text())
            visited_rounds.clear()
            self.assertEqual(run(2, 4), expected[1:])
            self.assertEqual(visited_rounds, [2, 3, 4])
            self.assertEqual(json.loads(checkpoint.read_text()), original)
            for kind in ("principle", "lesson"):
                self.assertEqual(original[kind + "s"][0]["condition"],
                                 starting_conditions[kind] + "".join(clause(n) for n in range(1, 5)))
            self.assertEqual(run(0, 4, repeat_refinement=4), ["library_diag_iter4.json"])
            self.assertEqual(json.loads(checkpoint.read_text()), original)
            self.assertEqual(visited_rounds, [2, 3, 4])
