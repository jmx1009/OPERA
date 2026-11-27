import os
import re
import math
import json
import sys
import subprocess
from numbers import Real

from .config import training_config
from .objective_check import objective_matches
config = training_config()

def divide_insight(insight):
    """An insight may guide both formulation and program generation."""
    formulation, program = [], []
    for record in insight or ():
        if not isinstance(record, dict):
            continue
        taxonomy = record.get("taxonomy") or {}
        if "General Formulation" in taxonomy or "Domain Modeling" in taxonomy:
            formulation.append(record)
        if "Code Implementation" in taxonomy:
            program.append(record)
    return formulation, program


def generate_solution_with_retrieval(iter, task, library, llm_retri, llm_opt, retrieved_insights=None,
                                     output_path="", verbose=False, save_data=True, settings=None):
    settings = config if settings is None else settings
    formulation_rules, code_rules = divide_insight(retrieved_insights)
    attempt = dict(iter=iter, task=task, output_path=output_path, verbose=verbose, save_data=save_data)
    generation = {**attempt, "abl_params": settings.ablation}

    def retrieve(stage, existing, **details):
        if existing or llm_retri is None or not llm_retri.has_available_knowledge(stage=stage):
            return existing
        return llm_retri.retrieve_applicable_insights(stage=stage, config=settings, **attempt, **details) or []

    formulation_rules = retrieve("Formulation", formulation_rules)
    model = llm_opt.generate_formulation(retrieved_insights=formulation_rules, **generation)
    if not model:
        return formulation_rules + code_rules, None, None, None, None, None

    if llm_retri is not None and settings.ablation.include_program_insight:
        code_rules = retrieve("Program", code_rules, formulation=model)
    program, result, ran, timeout = llm_opt.generate_program(
        formulation=model, retrieved_insights=code_rules, **generation,
    )
    return formulation_rules + code_rules, model, program, result, ran, timeout


def is_optimal_with_tolerance(output, gt, tol=None, mode=None):
    if tol is None:
        tol = config.params.tolerance
    if mode is None:
        mode = getattr(config.params, "tolerance_mode", "absolute")
    return objective_matches(output, gt, tol, mode)


def check_optimality(task, output, runnable, is_time_out, tolerance=None, tolerance_mode=None):
    """Return a status and the same feedback used by diagnosis for this attempt."""
    if not runnable:
        status, message = "run_error", "Failed to generate a runnable program"
    elif is_time_out:
        status, message = "solver_time_out", "Solver timed out without finding an optimal solution"
    elif not isinstance(output, Real):
        status, message = "failure_solve", "Failed to obtain an objective value"
    elif is_optimal_with_tolerance(output, task.ground_truth, tol=tolerance, mode=tolerance_mode):
        return True, "optimal", None
    else:
        return False, "not_optimal", f"Task {task.id} returned objective {output}, outside the verification tolerance."
    return False, status, f"\n   [Task {task.id}]: {message}: \n{output}"


def self_verify_test(iter, task, llm_opt, new_insights, prev_insights, save_data=False, output_path=""):
    if not new_insights:
        return False
    # Proposals and batch-local IDs can collide with the working library.
    # Give temporary entries distinct prompt IDs and preserve the source context.
    context = list(prev_insights or [])
    def signature(ins):
        return tuple(json.dumps(ins.get(k), sort_keys=True) for k in ("taxonomy", "condition", "explanation", "example"))
    seen = {signature(ins) for ins in context}
    for idx, ins in enumerate(new_insights, 1):
        key = signature(ins)
        if key not in seen:
            context.append({**ins, "insight_id": f"candidate{idx}"})
            seen.add(key)
    # Reuse the attempt pipeline with the recorded context and no new retrieval.
    _, model, _, value, ran, timeout = generate_solution_with_retrieval(
        iter, task, None, None, llm_opt, retrieved_insights=context, save_data=save_data,
        output_path=os.path.join(output_path, "self_verify"),
    )
    return bool(model) and check_optimality(task, value, ran, timeout)[0]


def save_checkpoint(library, tasks, metrics, paths, suffix):
    if library is not None:
        library.save(f"{paths.lib_dir}/library_{suffix}.json", checkpoint=True)
        library.save_taxonomy(f"{paths.lib_dir}/latest_taxonomy_{suffix}.json")
    # Save tasks with status record
    if tasks:
        tasks.save_as_json(f"{paths.train_output_dir}/train_tasks_record_{suffix}.json")
    if metrics:
        # Save iteration metrics log
        with open(paths.metrics_log_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


def solver_output(stdout):
    """Read the most recent reported solver status, in full numeric precision."""
    number = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith(("Model is infeasible", "Model is unbounded", "Other status:")):
            break
        if line.startswith("Optimal value"):
            match = re.fullmatch(rf"Optimal value\s*[:=]\s*({number})", line)
            if match and math.isfinite(value := float(match.group(1))):
                return value
            break
    return stdout


def _run_solver(arguments, timeout_sec, source=None):
    try:
        process = subprocess.run(
            [sys.executable, *arguments], input=source, capture_output=True,
            encoding="utf-8", errors="replace", check=True, timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as timeout:
        return timeout
    return solver_output(process.stdout)


def execute_code(code_str, timeout_sec=400):
    return _run_solver(["-u", "-"], timeout_sec, source=code_str)


def execute_solver_file(file_path, timeout_sec=400):
    return _run_solver([file_path], timeout_sec)


def solver_execution_result(output):
    """Preserve solver messages and timeouts alongside numeric objective values."""
    if isinstance(output, subprocess.TimeoutExpired):
        return output, True
    try:
        return float(output), False
    except (ValueError, TypeError):
        return output, False
