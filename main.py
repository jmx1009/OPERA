import os
import time
import json

import argparse

parser = argparse.ArgumentParser(description="Learn and refine OPERA's two knowledge libraries.")
parser.add_argument("--config", help="Training settings (default: train_config.yaml).")
parser.add_argument("--dry-run", action="store_true", help="Validate sample and library loading without model calls.")
parser.add_argument("--experience-library-only", action="store_true", help="Train a new experience library from the generic lessons with principles disabled.")
args = parser.parse_args()
if args.config:
    os.environ["OPERA_TRAIN_CONFIG"] = args.config

from src.config import training_config, validate_model
from src.dataloader import DataLoader
from src.utils import cal_time_cost
from src.train_eval_utils import save_checkpoint
from library_online_learning import run_library_online_learning
from library_diagnosis import run_library_diagnosis
from library_refinement import refinement_counts, run_library_refinement
from src.library_loading import load_previous_round_library, load_working_library
from src.llm_programmer import ProgramGenerator
from src.llm_diagnostic import ProgramDiagnostic
from src.llm_extractor import LessonExtractor
from src.llm_retriever import LibraryRetrieval
from src.condition_refiner import ApplicabilityRefiner
from src.principle_library import prepare_principle_sources, build_document_retriever

def print_metrics_summary(metrics_log):
    """
    Print structured summary of metrics from metrics_log.

    Args:
        metrics_log: List of dictionaries containing metrics for each iteration
    """
    print("\n" + "="*80)
    print("METRICS SUMMARY")
    print("="*80)

    if not metrics_log:
        print("No metrics available.")
        return

    # Print header
    print(f"{'Iter':<4} {'Train Acc':<10} {'Library Size':<8} {'Online Merge':<12} {'Self Verify':<12} {'Diagnose':<10} {'Refinement':<12}")
    print("-" * 80)

    # Process each iteration's metrics
    for i, metrics in enumerate(metrics_log):
        # Extract basic metrics
        train_accuracy = metrics.get("train_accuracy", "N/A")
        library_size = metrics.get("library_size", "N/A")
        online_merge_rate = metrics.get("online_merge_rate", "N/A")
        self_verify_rate = metrics.get("self_verify_rate", "N/A")

        # Extract diagnosis metrics if available
        diagnose_success_rate = "N/A"
        refinement_rate = "N/A"

        # Look for failure reflection stage metrics
        if "stage" in metrics and metrics["stage"] in {"Failure Reflection", "Library Diagnosis"}:
            diagnose_success_rate = metrics.get("diagnose_success_rate", "N/A")
            refinement_rate = metrics.get("refinement_rate", "N/A")

        # Format values for display
        def format_value(value):
            if value == "N/A":
                return "N/A"
            elif isinstance(value, (int, float)):
                if isinstance(value, float):
                    return f"{value:.3f}"
                else:
                    return str(value)
            else:
                return str(value)

        # Print row
        print(f"{i:<4} {format_value(train_accuracy):<10} {format_value(library_size):<8} "
                f"{format_value(online_merge_rate):<12} {format_value(self_verify_rate):<12} "
                f"{format_value(diagnose_success_rate):<10} {format_value(refinement_rate):<12}")

    print("="*80)

    # Print detailed breakdown for failure reflection stages
    print("\nDETAILED BREAKDOWN:")
    print("-" * 50)

    for i, metrics in enumerate(metrics_log):
        if "stage" in metrics and metrics["stage"] in {"Failure Reflection", "Library Diagnosis"}:
            print(f"\nIteration {i} - Failure Reflection:")
            print(f"  Diagnose Success Rate: {metrics.get('diagnose_success_rate', 'N/A')}")
            print(f"  Refinement Rate: {metrics.get('refinement_rate', 'N/A')}")

            # Print other relevant metrics for this stage
            if "train_accuracy" in metrics:
                print(f"  Train Accuracy: {metrics.get('train_accuracy', 'N/A')}")
            if "library_size" in metrics:
                print(f"  Library Size: {metrics.get('library_size', 'N/A')}")

    print("\n" + "="*80)

#* Configure
from omegaconf import OmegaConf
config = training_config()
validate_model(config.model, config.service)

if args.experience_library_only:
    config.bootstrap.enable_principle_library = False
    config.output_folder += "_lessons_only"
OmegaConf.resolve(config)

RUN_MODE_ALIASES = {
    "full": "full",
    "build_principle_sources": "build_principle_sources",
    "learning_only": "learning_only",
    "train_only": "learning_only",
}

run_mode = RUN_MODE_ALIASES.get(str(getattr(config, "run_mode", "full")).strip().lower())
if run_mode is None:
    raise ValueError(
        "Unsupported run_mode. Expected one of: "
        "full, build_principle_sources, learning_only."
    )

principle_sources = prepare_principle_sources(
    config,
    allow_build=(run_mode != "learning_only" and not args.dry_run),
)
if run_mode == "build_principle_sources":
    if principle_sources is None:
        raise ValueError(
            "run_mode: build_principle_sources requires bootstrap.enable_principle_library: true."
        )
    print("Document hierarchy and candidate principles are ready.")
    print(f"Document index: {principle_sources.document_index_path}")
    print(f"Candidate principles: {principle_sources.candidate_principles_path}")
    print(f"Candidate taxonomy: {principle_sources.candidate_taxonomy_path}")
    raise SystemExit(0)

# Initialize the LLM agents
llm_opt = ProgramGenerator(model=config.model, service=config.service, temperature=0)
llm_diag = ProgramDiagnostic(model=config.model, service=config.service, temperature=0)
llm_ins = LessonExtractor(model=config.model, service=config.service, temperature=0.7)

temp_online = 0.7 if config.params.max_solution_attempts > 1 else 0
llm_opt_online = ProgramGenerator(model=config.model, service=config.service, temperature=temp_online)
diagnosis_max_workers = getattr(config.params, "diagnosis_max_workers", 12)
refinement_max_workers = getattr(config.params, "refinement_max_workers", 2)
resume_refinement_iter = getattr(config, "resume_refinement_iter", None)


def _run_refinement_only_from_diagnosis(iter_idx):
    train_data_path = f"{config.file_paths.train_output_dir}/train_tasks_record_diag_iter{iter_idx}.json"
    lib_path = f"{config.file_paths.lib_dir}/library_diag_iter{iter_idx}.json"
    taxo_path = f"{config.file_paths.lib_dir}/latest_taxonomy_diag_iter{iter_idx}.json"

    required_paths = [train_data_path, lib_path, taxo_path, config.file_paths.metrics_log_path]
    missing_paths = [path for path in required_paths if not os.path.exists(path)]
    if missing_paths:
        missing = "\n".join(missing_paths)
        raise FileNotFoundError(
            f"Cannot resume refinement for iter {iter_idx}; missing checkpoint files:\n{missing}"
        )

    with open(config.file_paths.metrics_log_path, "r", encoding="utf-8") as f:
        metrics_log = json.load(f)

    diagnosis_metrics = next(
        (
            metrics
            for metrics in reversed(metrics_log)
            if metrics.get("iter") == iter_idx and metrics.get("stage") in {"Failure Reflection", "Library Diagnosis"}
        ),
        None,
    )
    if diagnosis_metrics is None:
        raise ValueError(
            f"Cannot resume refinement for iter {iter_idx}; "
            "no matching failure reflection metrics entry was found."
        )

    print(f"[Iteration {iter_idx}] Resume Library Refinement From Diagnosis Checkpoint")
    train_tasks = DataLoader(train_data_path, mode="learn", filter_success_num=None, reset=False)
    library = load_working_library(
        config, principle_sources, library_path=lib_path, historical_tasks=train_tasks,
        taxonomy_path=taxo_path,
    )

    if args.dry_run:
        print(f"Refinement checkpoint ready: {len(train_tasks)} tasks, {len(library.principles)} principles, {len(library.lessons)} lessons.")
        return
    iter_start_time = time.time()
    llm_evolve = ApplicabilityRefiner(lib=library, model=config.model, service=config.service, temperature=0.7)
    refined_library, avg_refinement_rate = run_library_refinement(
        iter=iter_idx,
        tasks=train_tasks,
        config=config,
        llm_evolve=llm_evolve,
        verbose=False,
        save_data=True,
        output_path=config.file_paths.train_output_dir,
        max_workers=refinement_max_workers,
    )
    print("refinement_rate:", avg_refinement_rate)

    diagnosis_metrics["refinement_rate"] = round(avg_refinement_rate, 3)
    diagnosis_metrics["refinement_by_type"] = refinement_counts(refined_library, iter_idx)
    save_checkpoint(
        library=refined_library,
        tasks=None,
        metrics=metrics_log,
        paths=config.file_paths,
        suffix=f"refine_iter{iter_idx}",
    )

    cal_time_cost(iter_start_time, f"Iteration {iter_idx} Library Refinement")
    print_metrics_summary(metrics_log)
    raise SystemExit(0)


if resume_refinement_iter is not None:
    _run_refinement_only_from_diagnosis(int(resume_refinement_iter))
    raise SystemExit(0)


# 0 (start from online learning), 1 (start from library diagnosis at iter 1)
start_iter = config.start_iter
end_iter = config.params.num_iterations + 1

if start_iter == 0:
    train_tasks = DataLoader(config.file_paths.train_data_path, mode="learn", filter_success_num=None, reset=True)
    library = load_working_library(
        config, principle_sources, getattr(config.file_paths, "initial_library_path", None),
        getattr(config.file_paths, "initial_taxonomy_path", None),
    )
    # Track iteration metrics
    metrics_log = []

else:
    task_suffix = "base" if start_iter == 1 else f"diag_iter{start_iter - 1}"
    train_data_path = f"{config.file_paths.train_output_dir}/train_tasks_record_{task_suffix}.json"
    train_tasks = DataLoader(train_data_path, mode="learn", filter_success_num=None, reset=False)
    library = load_previous_round_library(config, principle_sources, start_iter, train_tasks)
    # Track iteration metrics
    with open(config.file_paths.metrics_log_path, "r", encoding="utf-8") as f:
        metrics_log = json.load(f)

document_retriever = build_document_retriever(
    config=config,
    artifacts=principle_sources,
    taxonomy=library.taxonomy,
    model=config.model,
    service=config.service,
    allow_build=(run_mode != "learning_only" and not args.dry_run),
)

# Run subset
if config.data_slice and start_iter == 0:
    start = config.data_slice[0]
    end = config.data_slice[1]
    train_tasks = train_tasks.slice(start, end)

if args.dry_run:
    print(f"Training startup: {len(train_tasks)} tasks, {len(library.principles)} principles, "
          f"{len(library.lessons)} lessons, {config.params.num_iterations} refinement rounds.")
    raise SystemExit(0)

start_time = time.time()
for iter in range(start_iter, end_iter):
    iter_start_time = time.time()
    if iter > start_iter:
        library = load_previous_round_library(config, principle_sources, iter, train_tasks)
    if document_retriever is not None:
        document_retriever.taxonomy = library.taxonomy
    llm_retri = LibraryRetrieval(
        lib=library,
        model=config.model,
        service=config.service,
        temperature=0,
        document_retriever=document_retriever,
    )

    #* Library online learning for once
    if iter == 0:
        iter_metrics = run_library_online_learning(
            iter,
            train_tasks,
            llm_retri, llm_opt_online, llm_diag, llm_ins, library,
            config.params,
            config.file_paths
        )

        # Save checkpoint
        print(iter_metrics)
        metrics_log.append(iter_metrics)
        save_checkpoint(library=library, tasks=train_tasks, metrics=metrics_log, paths=config.file_paths, suffix="base")
        # directly continue to iter 1
        continue
    # Failure reflection
    iter_metrics = run_library_diagnosis(
        iter,
        train_tasks,
        llm_retri, llm_opt, llm_diag, llm_ins, library,
        config.params,
        config.file_paths,
        max_workers=diagnosis_max_workers
    )

    # Save checkpoint
    print(iter_metrics)
    metrics_log[:] = [entry for entry in metrics_log
                      if (entry.get("iter"), entry.get("stage")) != (iter, iter_metrics["stage"])]
    metrics_log.append(iter_metrics)
    save_checkpoint(library=library, tasks=train_tasks, metrics=metrics_log, paths=config.file_paths, suffix=f"diag_iter{iter}")

    #* Library Refinement
    llm_evolve = ApplicabilityRefiner(lib=library, model=config.model, service=config.service, temperature=0.7)
    refined_library, avg_refinement_rate = run_library_refinement(
        iter=iter, tasks=train_tasks,
        config=config, llm_evolve=llm_evolve,
        verbose=False, save_data=True, output_path=config.file_paths.train_output_dir,
        max_workers=refinement_max_workers
    )
    print("refinement_rate:", avg_refinement_rate)
    # Save iteration metrics log for applicability condition refinement
    last_metrics = metrics_log[-1]
    last_metrics["refinement_rate"] = round(avg_refinement_rate, 3)
    last_metrics["refinement_by_type"] = refinement_counts(refined_library, iter)

    save_checkpoint(library=refined_library, tasks=None, metrics=metrics_log, paths=config.file_paths, suffix=f"refine_iter{iter}")

    iter_duration = cal_time_cost(iter_start_time, f'Iteration {iter} Total Pipeline')

# Count time cost
total_duration = cal_time_cost(start_time, f'The iterative library learning and evolution process for {config.params.num_iterations} iterations')

# Print structured metrics summary
print_metrics_summary(metrics_log)
