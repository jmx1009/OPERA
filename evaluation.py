import os
import json
import time
import argparse
from copy import deepcopy
from pathlib import Path
from statistics import fmean
from tempfile import mkdtemp
from typing import List, Tuple, Optional, Any

from tqdm.auto import tqdm

import concurrent.futures

from src.config import validate_model
from src.utils import cal_time_cost
from src.dataloader import DataLoader, Task
from src.llm_programmer import ProgramGenerator
from src.insight_library import InsightLibrary
from src.library_loading import load_working_library
from src.llm_retriever import LibraryRetrieval
from src.train_eval_utils import check_optimality, generate_solution_with_retrieval
from src.principle_library import prepare_principle_sources, build_document_retriever

def check_evaluation_protocol(config):
    runs = getattr(config, "evaluation_runs", 1)
    if type(runs) is not int or runs < 1:
        raise ValueError("evaluation_runs must be a positive integer.")
    if config.pass_at_k != 1 or config.ablation.max_debug_retry != 0:
        raise ValueError("Evaluation requires one modeling attempt and zero repair attempts per task.")


def evaluate(
    tasks: List["Task"],
    llm_opt: "ProgramGenerator",
    use_knowledge: bool,
    library: Optional["InsightLibrary"],
    document_retriever: Optional[Any],
    config: Any,
    *,
    output_dir: Optional[Path] = None,
) -> Tuple[int, int, int, float]:
    """Return successful tasks, runnable tasks, total tasks and the single-attempt success rate."""
    check_evaluation_protocol(config)
    retriever = LibraryRetrieval(
        lib=library, model=llm_opt.model, service=config.service,
        temperature=llm_opt.temp, document_retriever=document_retriever,
    ) if use_knowledge else None

    output_dir = output_dir if output_dir is not None else Path("testing") / config.output_folder

    def process_task(task):
        mode = "lib" if use_knowledge else "nolib"
        path = str(output_dir / f"task_{task.id}" / mode)
        insights, formulation, _, output, runnable, timed_out = generate_solution_with_retrieval(
            0, task, library, retriever, llm_opt, output_path=path, settings=config,
        )
        if formulation:
            success, status, _ = check_optimality(
                task, output, runnable, timed_out, tolerance=config.tolerance,
                tolerance_mode=getattr(config, "tolerance_mode", "absolute"),
            )
        else:
            success, status = False, "parse_error"
        task.record_attempt(0, 1, status, formulation, insights)
        task.retri_ins_lst.append([row["insight_id"] for row in insights if "insight_id" in row])
        task.output_status.append(status)
        return int(success), int(bool(runnable))

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        results = list(tqdm(executor.map(process_task, tasks), total=len(tasks), desc="Evaluating"))
    successes = sum(success for success, _ in results)
    runnable = sum(executed for _, executed in results)
    return successes, runnable, len(results), successes / len(results) if results else 0.0


def load_config(config_file: str):
    from omegaconf import OmegaConf

    return OmegaConf.load(config_file)


def _none_if_null(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if str(value).strip().lower() in {"", "none", "null"}:
        return None
    return value


def apply_cli_overrides(config: Any, args: argparse.Namespace) -> Any:
    from omegaconf import OmegaConf

    simple_overrides = {
        "dataset": args.dataset,
        "data_path": args.data_path,
        "output_folder": args.output_folder,
        "model": args.model,
        "service": args.service,
        "temperature": args.temperature,
        "pass_at_k": args.pass_at_k,
        "evaluation_runs": getattr(args, "evaluation_runs", None),
        "max_workers": args.max_workers,
    }
    for key, value in simple_overrides.items():
        if value is not None:
            OmegaConf.update(config, key, value, merge=True)

    if args.library_path is not None:
        OmegaConf.update(config, "taxonomy_path", None)
        OmegaConf.update(config, "library_path", _none_if_null(args.library_path), merge=True)
    if args.taxonomy_path is not None:
        OmegaConf.update(config, "taxonomy_path", _none_if_null(args.taxonomy_path), merge=True)

    if args.principle_library is not None:
        enabled = args.principle_library == "enabled"
        OmegaConf.update(config, "bootstrap.enable_principle_library", enabled, merge=True)
    if args.rebuild_document_index:
        OmegaConf.update(config, "bootstrap.rebuild_document_index", True, merge=True)
    if args.max_debug_retry is not None:
        OmegaConf.update(config, "ablation.max_debug_retry", args.max_debug_retry, merge=True)
    OmegaConf.resolve(config)
    validate_model(config.model, config.service)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OPERA on an optimization task split.")
    parser.add_argument("--config", default="./eval_params.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-folder", default=None)
    parser.add_argument("--library-path", default=None)
    parser.add_argument("--experience-library", choices=["enabled", "disabled"], default="enabled")
    parser.add_argument("--taxonomy-path", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--service", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--pass-at-k", type=int, default=None)
    parser.add_argument("--evaluation-runs", type=int, default=None,
                        help="Number of full single-attempt evaluations to average.")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--max-debug-retry", type=int, default=None)
    parser.add_argument("--principle-library", choices=["enabled", "disabled"], default=None)
    parser.add_argument("--rebuild-document-index", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, dataset loading, library loading, and principle source files without LLM calls.",
    )
    return parser.parse_args()


def print_startup_summary(
    config: Any,
    test_tasks: "DataLoader",
    *,
    use_library: bool,
    document_retriever: Optional[Any],
    principle_sources: Optional[Any],
) -> None:
    print(
        "\n================  EVALUATION STARTUP CHECK  ================\n"
        f"Dataset              : {config.dataset}\n"
        f"Data path            : {config.data_path}\n"
        f"Tasks loaded         : {len(test_tasks)}\n"
        f"Output folder        : testing/{config.output_folder}\n"
        f"Experience library   : {'enabled' if use_library else 'disabled'}\n"
        f"Principle library    : {bool(getattr(config.bootstrap, 'enable_principle_library', False))}\n"
        f"Principle sources    : {'ready' if principle_sources is not None else 'not used'}\n"
        f"Document retrieval   : {'enabled' if document_retriever is not None else 'disabled'}\n"
        f"Model / service      : {config.model} / {config.service}\n"
        f"Evaluation runs      : {getattr(config, 'evaluation_runs', 1)}\n"
        f"Pass@k / workers     : {config.pass_at_k} / {config.max_workers}\n"
        "============================================================\n"
    )


def resolve_taxonomy_path(config: Any) -> Optional[str]:
    configured_path = getattr(config, "taxonomy_path", None)
    if configured_path:
        return str(configured_path)

    library_path = getattr(config, "library_path", None)
    if not library_path:
        return None

    library_path = str(library_path)
    library_name = os.path.basename(library_path)
    if not library_name.startswith("library_"):
        return None

    candidate = os.path.join(
        os.path.dirname(library_path),
        library_name.replace("library_", "latest_taxonomy_", 1),
    )
    return candidate if os.path.exists(candidate) else None


def run_evaluations(test_tasks, llm_opt, use_knowledge, library, document_retriever, config, metadata):
    """Evaluate a fixed split and library repeatedly, keeping the task histories separate."""
    check_evaluation_protocol(config)
    n_runs = getattr(config, "evaluation_runs", 1)
    folder = Path("testing") / config.output_folder
    folder.mkdir(parents=True, exist_ok=True)
    if n_runs > 1:
        folder = Path(mkdtemp(prefix="evaluation_", dir=folder))
    mode = "lib" if use_knowledge else "nolib"
    runs = []
    for number in range(1, n_runs + 1):
        directory = folder / f"run_{number}" if n_runs > 1 else folder
        directory.mkdir(exist_ok=True)
        tasks = deepcopy(test_tasks) if n_runs > 1 else test_tasks
        start = time.time()
        successes, runnable, total, rate = evaluate(
            tasks, llm_opt, use_knowledge, library, document_retriever, config, output_dir=directory,
        )
        tasks.save_as_json(str(directory / f"tasks_record_{mode}.json"))
        result = {
            "run": number, "directory": str(directory.relative_to(folder)),
            "n_total": total, "n_success": successes, "n_runnable": runnable,
            "pass_at_1_rate": rate, "pass_at_k_rate": rate, "execution_rate": runnable / total,
            "duration": cal_time_cost(start, f"Evaluation {number}/{n_runs}"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start)),
        }
        record = {**metadata, **result}
        (directory / "metrics.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        runs.append(result)
        print(f"Run {number}/{n_runs}: {successes}/{total} successful ({rate:.3%}).")

    summary = {
        **metadata, "evaluation_runs": n_runs, "tasks_per_run": len(test_tasks),
        "mean_pass_at_1_rate": fmean(run["pass_at_1_rate"] for run in runs),
        "mean_execution_rate": fmean(run["execution_rate"] for run in runs), "runs": runs,
    }
    summary_path = folder / "summary.json"
    temporary = summary_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    print(f"Mean Pass@1 over {n_runs} run(s): {summary['mean_pass_at_1_rate']:.3%}. Results: {summary_path}")

    if n_runs == 1:
        # Preserve the cumulative log and task layout used by single-run callers.
        log = Path("testing/all_test_results.json")
        log.parent.mkdir(parents=True, exist_ok=True)
        history = json.loads(log.read_text(encoding="utf-8")) if log.exists() else []
        history.append({**metadata, **runs[0]})
        log.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    principles_enabled = bool(config.bootstrap.enable_principle_library)
    if args.output_folder is None and (args.experience_library == "disabled" or args.principle_library == "disabled"
                                       or (args.library_path is not None and not config.library_path)):
        lessons_enabled = bool(config.library_path) and args.experience_library != "disabled"
        variant = {(True, False): "principles_only", (False, True): "lessons_only",
                   (False, False): "no_library", (True, True): "both"}
        config.output_folder += "_" + variant[principles_enabled, lessons_enabled]
    check_evaluation_protocol(config)
    library_source = "configured library"
    # Skip the lesson catalog when lessons are disabled. A custom checkpoint may supply its own principles.
    if args.experience_library == "disabled" and (
        not principles_enabled or config.library_path == getattr(config, "standalone_lesson_library_path", None)
    ):
        config.library_path, config.taxonomy_path = None, None
        library_source = "configured principle library" if principles_enabled else "no libraries requested"
    principle_sources = prepare_principle_sources(
        config,
        allow_build=not args.dry_run,
    )

    # Load test tasks
    test_tasks = DataLoader(config.data_path, mode="test")

    if not len(test_tasks):
        raise ValueError("The evaluation data contain no tasks with a finite reference objective.")

    taxonomy_path = resolve_taxonomy_path(config)
    library = load_working_library(config, principle_sources, config.library_path, taxonomy_path)
    if args.experience_library == "disabled":
        library.select_types({"principle"})
    use_library = bool(library.lessons)
    print(f"Loaded {len(library.principles)} principles and {len(library.lessons)} lessons ({library_source}).")

    document_retriever = build_document_retriever(
        config=config,
        artifacts=principle_sources,
        taxonomy=library.taxonomy,
        model=config.model,
        service=config.service,
        allow_build=not args.dry_run,
    )
    use_knowledge = bool(library) or document_retriever is not None

    print_startup_summary(
        config,
        test_tasks,
        use_library=use_library,
        document_retriever=document_retriever,
        principle_sources=principle_sources,
    )

    if args.dry_run:
        print("Dry run complete. No LLM calls were made.")
        return

    # Initialize ProgramGenerator
    llm_opt = ProgramGenerator(
        model       = config.model,
        service     = config.service,
        temperature = config.temperature,
        generation_max_retries=1,
    )

    metadata = {
        "dataset": config.dataset,
        "data_path": config.data_path,
        "library_path": config.library_path,
        "taxonomy_path": taxonomy_path if taxonomy_path else "None",
        "library_source": library_source,
        "principles": len(library.principles),
        "lessons": len(library.lessons),
        "tolerance": config.tolerance,
        "tolerance_mode": getattr(config, "tolerance_mode", "absolute"),
        "document_retrieval_enabled": document_retriever is not None,
        "model": config.model,
        "service": config.service,
        "temperature": config.temperature,
        "pass_at_k": config.pass_at_k,
        "taxonomy": config.ablation.taxonomy,
        "include_example": config.ablation.include_example,
        "include_program_insight": config.ablation.include_program_insight,
        "max_debug_retry": config.ablation.max_debug_retry,
    }
    run_evaluations(test_tasks, llm_opt, use_knowledge, library, document_retriever, config, metadata)


if __name__ == "__main__":
    main()
