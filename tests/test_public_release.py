"""Check that the released corpus, manuscript cases and public entry points agree."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf

from src.config import MODEL, SERVICE
from src.dataloader import DataLoader
from src.insight_library import InsightLibrary
from src.principle_library import DocumentIndex, PrincipleLibraryBuilder, prepare_principle_sources
from src.library_loading import load_working_library
from src.objective_check import DEFAULT_ABSOLUTE_TOLERANCE, objective_matches


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data/laflor/manifest.json"
PAPER_CASES = set(
    """
LAFLOR_005 LAFLOR_019 LAFLOR_022 LAFLOR_043 LAFLOR_048
LAFLOR_062 LAFLOR_073 LAFLOR_076 LAFLOR_077 LAFLOR_087
LAFLOR_099 LAFLOR_101 LAFLOR_106 LAFLOR_116 LAFLOR_118
LAFLOR_124 LAFLOR_136 LAFLOR_144 LAFLOR_145 LAFLOR_152
""".split()
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


class PublicReleaseTests(unittest.TestCase):
    def test_catalog_covers_the_named_cases_and_sources(self):
        entries = read(CATALOG)["tasks"]
        self.assertEqual([entry["task_id"] for entry in entries], sorted(PAPER_CASES))
        self.assertEqual(len(entries), len(PAPER_CASES))
        self.assertEqual({x["role"] for x in entries}, {"benchmark_example"})
        files = {
            "description": "problem.md",
            "formulation": "formulation.md",
            "reference": "reference.py",
        }
        statements = {}
        for row in entries:
            with self.subTest(task=row["task_id"]):
                for field, name in files.items():
                    self.assertEqual(row[field], f"{row['task_id']}/{name}")
                    self.assertTrue((CATALOG.parent / row[field]).is_file())
                problem = (CATALOG.parent / row["description"]).read_text(
                    encoding="utf-8"
                )
                body = "\n".join(
                    line for line in problem.splitlines() if not line.startswith("# ")
                )
                words = tuple(re.findall(r"\w+", body.casefold()))
                self.assertTrue(words)
                self.assertIsNone(
                    statements.get(words), "Two tasks share a problem statement"
                )
                statements[words] = row["task_id"]
                self.assertTrue(row["paper_evidence"])
                self.assertEqual(
                    row["ground_truth"] is None, row["status"] == "infeasible"
                )
                if "paper_reference" in row:
                    self.assertTrue(objective_matches(row["verified_objective"], row["paper_reference"]))

    def test_freight_charging_and_siting_grading_targets(self):
        rows = {row["task_id"]: row for row in read(CATALOG)["tasks"]}
        overlapping_intervals = []
        for ids in (
            ("LAFLOR_019", "LAFLOR_073"),
            ("LAFLOR_124", "LAFLOR_022", "LAFLOR_076"),
            ("LAFLOR_043", "LAFLOR_145"),
            ("LAFLOR_048", "LAFLOR_101"),
        ):
            targets = sorted((rows[task]["ground_truth"], task) for task in ids)
            for (lower, left), (higher, right) in zip(targets, targets[1:]):
                self.assertFalse(objective_matches(lower, higher))
                if higher - lower <= 2 * DEFAULT_ABSOLUTE_TOLERANCE:
                    overlapping_intervals.append((left, right))
        self.assertEqual(overlapping_intervals, [("LAFLOR_022", "LAFLOR_076")])
        self.assertEqual(rows["LAFLOR_101"]["status"], "optimal")
        self.assertEqual(rows["LAFLOR_048"]["status"], "optimal")
        self.assertEqual(rows["LAFLOR_048"]["ground_truth"], 286)
        self.assertEqual(rows["LAFLOR_043"]["ground_truth"], 61)
        self.assertEqual(rows["LAFLOR_124"]["ground_truth"], 1470)

    def test_fractional_siting_accepts_split_deliveries_within_range(self):
        from tools.reference_checks import verify

        deliveries = [
            {"zone": 2, "site": 2, "packages": 80},
            {"zone": 4, "site": 2, "packages": 70},
            {"zone": 6, "site": 1, "packages": 60.5},
            {"zone": 6, "site": 2, "packages": 9.5},
            {"zone": 6, "site": 3, "packages": 20},
        ]
        result = {
            "open_sites": [1, 2, 3], "accepted_zones": [2, 4, 6],
            "unserved_packages": 150, "site_penalty": 136,
            "objective": 286, "deliveries": deliveries,
        }
        self.assertEqual(verify("LAFLOR_048", result), 286)
        invalid_plans = {
            "out of range": [dict(deliveries[0], site=1), *deliveries[1:]],
            "closed site": [dict(deliveries[0], site=4), *deliveries[1:]],
            "over capacity": [*deliveries[:2], dict(deliveries[2], site=2), *deliveries[3:]],
            "partial order": [*deliveries[:-1], dict(deliveries[-1], packages=19)],
        }
        for error, plan in invalid_plans.items():
            with self.subTest(error=error), self.assertRaises(AssertionError):
                verify("LAFLOR_048", {**result, "deliveries": plan})

    def test_fractional_siting_designs_and_one_rule_changes(self):
        from src.train_eval_utils import is_optimal_with_tolerance
        from tools.laflor_048 import COMPARISONS, enumerate_designs, read_instance

        sites, zones = read_instance()
        designs = enumerate_designs(sites, zones)
        self.assertEqual(
            {d.sites: (d.served, d.objective) for d in designs},
            {(1, 2, 3): (240, 286), (1, 3, 5): (130, 420), (2, 3, 4): (260, 298)},
        )
        reference = min(d.objective for d in designs)
        expected = {
            "single_source": 316, "partial": 266, "unlimited": 136,
            "ungated": 186, "neighbor": 250, "at_most": 258,
            "no_site_cost": 130, "strict": 298, "no_upper": 250,
        }
        self.assertEqual(set(expected), set(COMPARISONS) - {"reference"})
        tolerance = OmegaConf.load(ROOT / "eval_params.yaml").tolerance
        for rule, target in expected.items():
            with self.subTest(rule=rule):
                value = min(d.objective for d in enumerate_designs(sites, zones, rule))
                self.assertEqual(value, target)
                self.assertFalse(is_optimal_with_tolerance(value, reference, tolerance))

    def test_specimen_bookings_require_a_free_aircraft_and_charge_at_launch(self):
        from tools.reference_checks import verify

        plan = {"bookings": ["P1", "E2", "O1", "V2"], "courier_clinics": ["Reed"], "objective": 61}
        self.assertEqual(verify("LAFLOR_043", plan), 61)
        infeasible = {
            "overlap": (["P2", "E2"], ["Reed", "Oak", "Vale"], 118),
            "insufficient charge at visit 1": (["P1", "R1", "E2", "O1", "V2"], [], 38),
        }
        for error, (bookings, clinics, cost) in infeasible.items():
            with self.subTest(error=error), self.assertRaises(AssertionError):
                result = {"bookings": bookings, "courier_clinics": clinics, "objective": cost}
                verify("LAFLOR_043", result)

    def test_pack_exchanges_precede_charging_and_cycles_occupy_two_hours(self):
        from tools.reference_checks import bay_schedules

        self.assertNotIn((1, 0, 0), bay_schedules((0, 1, 0), 1, 0, 0))
        self.assertIn((1, 0, 0), bay_schedules((0, 0, 1), 1, 0, 0))
        self.assertIn((1, 0), bay_schedules((1, 0), 1, 1, 1))
        self.assertNotIn((2, 1, 0), bay_schedules((0, 0, 0), 3, 0, 0))

    def test_pack_tariff_prices_the_metered_peak_and_charging_energy(self):
        from tools.reference_checks import verify

        plan = {
            "starts": {"North": [2, 0, 0, 1, 0, 2, 0], "South": [1, 0, 0, 2, 0, 2, 0]},
            "peak_kw": 11, "energy_cents": 700, "objective": 1470,
        }
        self.assertEqual(verify("LAFLOR_124", plan), 1470)
        with self.assertRaises(AssertionError):
            verify("LAFLOR_124", dict(plan, peak_kw=8, objective=1260))
        with self.assertRaises(AssertionError):
            verify("LAFLOR_124", dict(plan, energy_cents=680, objective=1450))

    def test_executed_program_uses_final_solver_status(self):
        from src.train_eval_utils import execute_code

        printed = "print('Optimal value: 12')\n"
        self.assertEqual(
            execute_code(printed + "print('Optimal value: -3.2e-4')"), -0.00032
        )
        self.assertIsInstance(
            execute_code(printed + "print('Model is infeasible.')"), str
        )
        self.assertIsInstance(execute_code("print('Optimal value: 1e999')"), str)
        self.assertIsInstance(execute_code("print('Optimal value: 2cost')"), str)

    def test_public_training_and_evaluation_views(self):
        train = DataLoader(str(CATALOG), mode="learn")
        evaluation = DataLoader(str(CATALOG), mode="test")
        self.assertEqual({task.id for task in evaluation}, PAPER_CASES)
        self.assertCountEqual([task.id for task in train], PAPER_CASES)
        fare_task = next(task for task in train if task.id == "LAFLOR_152")
        self.assertAlmostEqual(fare_task.ground_truth, 220.767615, places=6)
        self.assertTrue(
            all(task.formulation and task.correct_program for task in train)
        )
        self.assertTrue(
            all(
                task.formulation is None and task.correct_program is None
                for task in evaluation
            )
        )

    def test_training_respects_explicit_catalog_splits(self):
        train, held_out = read(CATALOG)["tasks"][:2]
        for row in (train, held_out):
            for field in ("description", "formulation", "reference"):
                row[field] = str(CATALOG.parent / row[field])
        held_out["source_split"] = "test"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            for version in ("opera-laflor-v1", "opera-laflor-v2"):
                with self.subTest(version=version):
                    train_entry = {**train, "source_split": "train"} if version == "opera-laflor-v1" else train
                    path.write_text(json.dumps({"format": version, "tasks": [train_entry, held_out]}))
                    self.assertEqual([t.id for t in DataLoader(str(path), mode="learn")],
                                     [train["task_id"]])
                    self.assertCountEqual([t.id for t in DataLoader(str(path), mode="test")],
                                          [train["task_id"], held_out["task_id"]])

    def test_summary_and_principle_provenance(self):
        index_file = ROOT / "data/knowledge/summaries.json"
        index = read(index_file)
        units = {r["unit_id"]: r["doc_id"] for r in index["units"]}
        documents = {doc["doc_id"] for doc in index["documents"]}
        selected = [
            uid for d in index["documents"] for uid in d["extraction_unit_ids"]
        ]
        self.assertEqual((len(documents), len(units), len(selected)), (17, 628, 168))
        self.assertEqual(len(set(selected)), 168)
        self.assertEqual(set(units.values()), documents)
        self.assertTrue(set(selected) <= units.keys())
        results = DocumentIndex.load(str(index_file)).retrieve_units(
            "vertiport waiting time and passenger demand"
        )
        self.assertTrue(results)
        self.assertTrue(all(r["unit_id"] in units for r in results))
        raw = read(ROOT / "data/knowledge/candidates.json")
        candidates = {rec["insight_id"]: rec["source_unit_id"] for rec in raw}
        self.assertEqual(len(candidates), 503)
        self.assertTrue(set(candidates.values()) <= set(selected))
        for rule in read(ROOT / "data/libraries/principles.json"):
            sources = {candidates[parent] for parent in rule["merged_from"]}
            self.assertTrue(sources <= set(rule["source_unit_ids"]))
            self.assertEqual(
                set(rule["source_refs"]),
                {units[uid] for uid in rule["source_unit_ids"]},
            )

    def test_paper_taxonomy_and_public_rules(self):
        counts = {
            "principles": {
                "Domain Modeling": 94,
                "General Formulation": 69,
                "Code Implementation": 47,
            },
            "lessons": {
                "Domain Modeling": 167,
                "General Formulation": 140,
                "Code Implementation": 56,
            },
        }
        for kind, expected in counts.items():
            records = read(ROOT / f"data/libraries/{kind}.json")
            taxonomy = read(ROOT / f"data/libraries/{kind[:-1]}_taxonomy.json")
            self.assertEqual(
                Counter(next(iter(r["taxonomy"])) for r in records), expected
            )
            for record in records:
                self.assertTrue(
                    all(
                        isinstance(record.get(k), str) and record[k].strip()
                        for k in ("condition", "explanation", "example")
                    )
                )
                for stage, groups in record["taxonomy"].items():
                    for group, labels in groups.items():
                        self.assertTrue(set(labels) <= taxonomy[stage][group].keys())
        lessons = {
            r["insight_id"]: r for r in read(ROOT / "data/libraries/lessons.json")
        }
        self.assertEqual(lessons["lesson325"].keys(), lessons["lesson324"].keys())
        self.assertEqual(
            lessons["lesson325"]["taxonomy"],
            {"Domain Modeling": {"Network Flow": ["Flow Conservation"]}},
        )

    def test_public_libraries_contain_rules_and_source_attribution(self):
        common = {"insight_type", "insight_id", "taxonomy", "condition", "explanation", "example"}
        attribution = {
            "source_unit_id", "source_unit_ids", "source_ref", "source_refs",
            "source_title", "source_titles", "source_category", "source_categories",
            "source_unit_kind",
        }
        catalogs = {
            "libraries/generic_lessons": {"source_task_ids"},
            "libraries/lessons": {"source_task_ids"},
            "libraries/initial_principles": attribution | {"merged_from"},
            "libraries/principles": attribution | {"merged_from"},
            "knowledge/candidates": attribution,
        }
        for catalog, optional_fields in catalogs.items():
            with self.subTest(catalog=catalog):
                path = ROOT / "data" / (catalog + ".json")
                for record in read(path):
                    allowed = common | optional_fields
                    if record["insight_type"] == "lesson":
                        sources = record["source_task_ids"]
                        self.assertIsInstance(sources, list)
                        self.assertTrue(sources)
                        for source in sources:
                            self.assertRegex(source, r"\A(?:initial lessons|LAFLOR_[0-9]{3})\Z")
                        self.assertEqual(len(sources), len(set(sources)))
                        if catalog == "libraries/generic_lessons":
                            self.assertEqual(sources, ["initial lessons"])
                    self.assertTrue(common <= record.keys() <= allowed)
                for insight in InsightLibrary.from_json_file(path):
                    self.assertIsNone(insight.task_id)
                    self.assertIsNone(insight.iteration)
                    self.assertEqual(insight.occurrence, 0)
                    self.assertEqual(set(insight.distribution), {"useful", "negative", "irrelevant", "unretrieved"})
                    self.assertFalse(any(insight.distribution.values()))

    def test_model_request_uses_the_manuscript_backbone(self):
        from src.utils import call_llm_and_parse_with_retry

        for config_name in ("train_config.yaml", "eval_params.yaml"):
            cfg = OmegaConf.load(ROOT / config_name)
            self.assertEqual((cfg.model, cfg.service), ("gpt-4o", "openai"))
        with patch("openai.OpenAI") as sdk, patch("src.utils._openai_http_client"):
            client = sdk.return_value.__enter__.return_value
            create = client.chat.completions.create
            responses = []
            create.return_value = SimpleNamespace(
                id="test-completion", model="gpt-4o-2024-08-06", created=1234,
                choices=[SimpleNamespace(message=SimpleNamespace(content="[1]"))],
            )
            request = dict(prompt="Return an array.", parse_fn=json.loads, max_retry=1, verbose=False)
            self.assertEqual(call_llm_and_parse_with_retry(MODEL, SERVICE, response_records=responses, **request), [1])
            self.assertEqual(create.call_args.kwargs["model"], MODEL)
            self.assertEqual(create.call_args.kwargs["messages"], [{"role": "user", "content": request["prompt"]}])
            messages = json.dumps(create.call_args.kwargs["messages"], ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":")).encode("utf-8")
            self.assertEqual(responses, [{
                "requested_model": MODEL, "returned_model": "gpt-4o-2024-08-06",
                "completion_id": "test-completion", "created": 1234,
                "messages_sha256": hashlib.sha256(messages).hexdigest(),
                "response_sha256": hashlib.sha256(b"[1]").hexdigest(),
                "parsed_sha256": hashlib.sha256(b"[1]").hexdigest(),
            }])
            with self.assertRaises(ValueError):
                call_llm_and_parse_with_retry("unconfigured-model", SERVICE, **request)
            with self.assertRaises(ValueError):
                call_llm_and_parse_with_retry(MODEL, "unconfigured-service", **request)
            create.assert_called_once()
            sdk.assert_called_once()

    def test_t16_response_evidence_tracks_the_accepted_retry(self):
        from src.condition_refiner import parse_refinement_candidates
        from src.utils import call_llm_and_parse_with_retry

        original = "Resource capacity."
        rows = [{"path_id": 1, "strategy": "Distinguish fixed capacity.",
                 "new_condition": original + "\n\nThis insight does NOT apply when capacity is fixed."}]
        raw = json.dumps(rows, indent=2)
        first = SimpleNamespace(id="test-rejected", choices=[SimpleNamespace(message=SimpleNamespace(content="bad"))])
        second = SimpleNamespace(id="test-accepted", choices=[SimpleNamespace(message=SimpleNamespace(content=raw))])
        responses = []
        with patch("openai.OpenAI") as sdk, patch("src.utils._openai_http_client"):
            create = sdk.return_value.__enter__.return_value.chat.completions.create
            create.side_effect = [first, second]
            result = call_llm_and_parse_with_retry(
                MODEL, SERVICE, prompt="Return T16 candidates.", max_retry=2, sleep_sec=0, verbose=False,
                parse_fn=lambda text: parse_refinement_candidates(text, original, 1, applicability=False, exclusion=True),
                response_records=responses,
            )
        self.assertEqual(result, rows)
        self.assertEqual(len(responses), 1)
        evidence = responses[0]
        self.assertEqual(evidence["completion_id"], "test-accepted")
        self.assertIsNone(evidence["returned_model"])
        self.assertIsNone(evidence["created"])
        self.assertEqual(evidence["requested_model"], MODEL)
        self.assertEqual(evidence["response_sha256"], hashlib.sha256(raw.encode()).hexdigest())
        sent = create.call_args_list[-1].kwargs["messages"]
        self.assertNotEqual(sent, create.call_args_list[0].kwargs["messages"])
        for key, value in (("messages_sha256", sent), ("parsed_sha256", rows)):
            encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
            self.assertEqual(evidence[key], hashlib.sha256(encoded).hexdigest())

    def test_document_builder_extracts_principles_from_a_summary(self):
        config = OmegaConf.load(ROOT / "train_config.yaml")
        candidate = {
            "taxonomy": {"General Formulation": {"Capacity": ["Activation"]}},
            "condition": "A facility can process shipments only when it is open.",
            "explanation": "Bound the total shipment quantity by facility capacity times its binary opening variable.",
            "example": "flow <= capacity * open",
        }
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            corpus = base / "sources"
            corpus.mkdir()
            (corpus / "Capacity.md").write_text(candidate["explanation"], encoding="utf-8")
            config.bootstrap.corpus_root = str(corpus)
            config.bootstrap.document_index_path = str(base / "index.json")
            config.bootstrap.candidate_principles_path = str(base / "principles.json")
            config.bootstrap.candidate_taxonomy_path = str(base / "taxonomy.json")
            with patch("src.principle_library.call_llm_and_parse_with_retry",
                       side_effect=[candidate["explanation"], [candidate]]) as completion:
                artifacts = PrincipleLibraryBuilder(config, MODEL, SERVICE).build()
            self.assertEqual([call.kwargs["model"] for call in completion.call_args_list], [MODEL, MODEL])
            rules = read(Path(artifacts.candidate_principles_path))
            self.assertEqual(len(rules), 1)
            rule = rules[0]
            self.assertEqual({k: rule[k] for k in candidate}, candidate)
            index = DocumentIndex.load(artifacts.document_index_path)
            summaries = index.retrieve_units("facility capacity")
            self.assertEqual([u["unit_id"] for u in summaries], [rule["source_unit_id"]])
            self.assertEqual(summaries[0]["kind"], "summary")
            self.assertEqual(rule["source_ref"], summaries[0]["doc_id"])
            self.assertEqual(prepare_principle_sources(config, allow_build=False), artifacts)

    def test_appendix_records_and_refinement_examples(self):
        from src.prompts.manuscript import TEMPLATES

        self.assertEqual(set(TEMPLATES), {f"T{i}" for i in range(1, 17)})
        self.assertIn("original condition verbatim", TEMPLATES["T16"]["text"])
        records = read(ROOT / "data/libraries/lessons.json") + read(
            ROOT / "data/libraries/principles.json"
        )
        by_id = {r["insight_id"]: r for r in records}
        for insight_id, phrase in {
            "principle17": "user equilibrium",
            "principle18": "captured demand",
            "principle103": "charging duration",
            "lesson278": "monetary charges",
            "lesson307": "waiting time",
            "lesson362": "binary opening",
            "lesson363": "consecutive selected legs",
        }.items():
            with self.subTest(insight_id=insight_id):
                record = by_id[insight_id]
                self.assertIn(phrase, record["explanation"])
        self.assertEqual(by_id["lesson8"]["source_task_ids"], ["initial lessons"])
        self.assertEqual(by_id["lesson108"]["source_task_ids"], ["initial lessons", "LAFLOR_118"])
        self.assertEqual(by_id["lesson363"]["source_task_ids"], ["LAFLOR_145"])
        principle = by_id["principle18"]
        lesson = by_id["lesson363"]
        self.assertIn("Model service quality and captured demand jointly.", principle["explanation"])
        self.assertIn("the demand–waiting state of the selected configuration.", principle["example"])
        self.assertIn("Propagate payload on consecutive selected legs.", lesson["explanation"])
        self.assertIn(r"x^D_{ij}=x^D_{jk}=1 \Rightarrow L_{jk}=L_{ij}-q_j^{-}+q_j^{+}", lesson["example"])
        initial = read(ROOT / "data/libraries/initial_principles.json")
        initial_principle = next(r for r in initial if r["insight_id"] == "principle18")
        for field in ("condition", "explanation", "example"):
            self.assertEqual(principle[field], initial_principle[field])
        examples = read(ROOT / "data/libraries/refinement_cases.json")
        self.assertTrue(all(set(r) == {
            "insight_id", "round", "paper_excerpts", "correct_before", "correct_after", "records"
        } for r in examples))
        self.assertEqual(
            {(r["insight_id"], r["round"]) for r in examples},
            {("lesson8", 1), ("lesson135", 1), ("lesson265", 1), ("lesson265", 4)},
        )
        self.assertTrue(
            all(
                0 <= r["correct_before"] < r["correct_after"] <= r["records"]
                for r in examples
            )
        )

    def test_experience_only_rejects_the_joint_library(self):
        config = OmegaConf.load(ROOT / "eval_params.yaml")
        config.bootstrap.enable_principle_library = False
        with self.assertRaisesRegex(ValueError, "independently"):
            load_working_library(config, None, config.library_path, independent_experience=True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "library.json"
            rule = read(ROOT / "data/libraries/generic_lessons.json")[0]
            library = InsightLibrary([rule])
            library.save(path, checkpoint=True)
            independent = load_working_library(config, None, path, independent_experience=True)
            self.assertEqual((len(independent.principles), len(independent.lessons)), (0, 1))
            library.attach_principles(InsightLibrary.from_json_file(config.bootstrap.consolidated_principles_path))
            library.save(path, checkpoint=True)
            with self.assertRaisesRegex(ValueError, "independently"):
                load_working_library(config, None, path, independent_experience=True)

    def test_configuration_loads_and_paper_ids_resolve(self):
        train, evaluate = (
            OmegaConf.load(ROOT / name)
            for name in ("train_config.yaml", "eval_params.yaml")
        )
        phase_one = prepare_principle_sources(evaluate, allow_build=False)
        final = load_working_library(
            evaluate, phase_one, evaluate.library_path, evaluate.taxonomy_path
        )
        self.assertEqual((len(final.principles), len(final.lessons)), (210, 363))
        for paper_id in ("principle18", "lesson363"):
            self.assertEqual(len(final.retrieve_insights_by_id(paper_id)), 1)
        initial = InsightLibrary.from_json_file(
            train.file_paths.initial_library_path
        )
        self.assertEqual(len(initial.lessons), 139)
        self.assertEqual(
            (evaluate.pass_at_k, evaluate.ablation.max_debug_retry), (1, 0)
        )
        self.assertEqual(
            (train.params.num_iterations, train.params.variant_num), (4, 5)
        )


if __name__ == "__main__":
    unittest.main()
