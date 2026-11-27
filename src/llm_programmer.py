"""Generate mathematical formulations and executable solver programs for an OPERA task."""
import json
import re
import traceback
from functools import partial

from .prompts.prompts_opt import PROMPT_GENERATE_FORMU, PROMPT_GENERATE_PROGRAM
from .train_eval_utils import execute_code, solver_execution_result
from .utils import call_llm_and_parse_with_retry, save_log_data


_FENCE = re.compile(r"```[^\n]*\n(.*?)\n```", re.DOTALL)
_PYTHON_FENCE = re.compile(r"```python\s*\n(.*?)\n```", re.DOTALL)
_REPAIR_FENCE = re.compile(r"```(?:\w*\s*)?\n(.*?)\n```", re.DOTALL)
_OPEN_FENCE = re.compile(r"```[^\n]*\n(.*)\Z", re.DOTALL)
_FORMULATION_MARKERS = (r"\[", r"\]", "## Parameters", "## Variables", "## Objective",
                        "## Constraints", "Parameters:", "Decision Variables", "Objective Function",
                        "Constraints:", "s.t.", "Subject to")
_SOLVER_MARKERS = ("import gurobipy", "from gurobipy import", "gp.Model", "model.optimize(", "m.optimize(")
_CODE_START = re.compile(r"^\s*(?:import\s+|from\s+\S+\s+import\s+|[A-Za-z_]\w*\s*=)")
_OPTIMIZE = re.compile(r"\b(?:model|m)\.optimize\s*\(")


def _read_fence(text, patterns):
    for pattern in patterns:
        match = pattern.search(text)
        if match is not None:
            return match.group(1).strip()
    return None


def read_program_response(text, *, repair=False):
    """Repair requires a closed fence. Initial generation also accepts bare solver code."""
    generic = _REPAIR_FENCE if repair else _FENCE
    body = _read_fence(text, (_PYTHON_FENCE, generic))
    if body is not None:
        return body
    if not repair:
        source = text.strip()
        if source.casefold().startswith("python\n"):
            source = source.partition("\n")[2].strip()
        lines = source.splitlines()
        start = next((i for i, line in enumerate(lines) if _CODE_START.match(line)), None)
        if start is not None:
            source = "\n".join(lines[start:]).strip()
        if any(marker in source for marker in _SOLVER_MARKERS):
            lines = source.splitlines()
            for i, line in enumerate(lines):
                if _OPTIMIZE.search(line):
                    return "\n".join(lines[:i + 1]).strip()
            return source
    raise ValueError("No valid code fence found. Expected a ```python``` block or a generic ``` block.")


def _prompt_insights(insights, include_example):
    fields = {"insight_id", "insight_type", "knowledge_id", "memory_id", "condition", "explanation"}
    if include_example:
        fields.add("example")
    selected = [{key: value for key, value in record.items() if key in fields} for record in insights or ()]
    return json.dumps(selected, indent=2, ensure_ascii=False)


class ProgramGenerator:
    def __init__(self, model: str, service: str, temperature: float | None = None, generation_max_retries: int = 8):
        self.model = model
        self.service = service
        self.temp = temperature
        self.generation_max_retries = generation_max_retries

    def extract_text(self, text: str):
        model = _read_fence(text, (_FENCE, _OPEN_FENCE))
        if model is not None:
            return model
        if any(marker in text for marker in _FORMULATION_MARKERS):
            return text.strip()
        raise ValueError("No valid ```...``` block found.")

    def extract_code(self, text: str, formatted_output: str = None) -> str:
        return read_program_response(text) + (formatted_output or "")

    execute_code = staticmethod(execute_code)

    def _generate(self, prompt, parser, stage, task_id, iteration, verbose):
        subject = "formulation" if stage == "Formulation" else "code"
        try:
            return call_llm_and_parse_with_retry(
                model=self.model, service=self.service, prompt=prompt, parse_fn=parser,
                temperature=self.temp, max_retry=self.generation_max_retries, sleep_sec=0.5,
                verbose=verbose,
                log_header=f"\n==========\n[Iteration {iteration}]: {stage} Generation for Task {task_id}\n==========\n",
                error_message=f"\n   Task {task_id} failed to extract {subject} from LLM after maximum attempts\n",
            )
        except Exception:
            print(f"Could not generate {subject} for task {task_id}.")
            traceback.print_exc()
            return None

    def generate_formulation(self, iter=None, task=None, retrieved_insights=None, abl_params=False,
                             verbose=False, save_data=False, output_path="learning"):
        prompt = PROMPT_GENERATE_FORMU.format(
            problem_description=task.desc, insights=_prompt_insights(retrieved_insights, abl_params.include_example),
        )
        model = self._generate(prompt, self.extract_text, "Formulation", task.id, iter, verbose)
        if save_data and model is not None:
            save_log_data(model, f"{output_path}/model_iter_{iter}.txt")
        return model

    def generate_program(self, iter=None, task=None, retrieved_insights=None, formulation=None, abl_params=False,
                         verbose=False, save_data=False, output_path="learning"):
        prompt = PROMPT_GENERATE_PROGRAM.format(
            problem_description=task.desc, mathematical_model=formulation,
            insights=_prompt_insights(retrieved_insights, abl_params.include_example),
        )
        parse = partial(self.extract_code, formatted_output=formatted_output)
        program = self._generate(prompt, parse, "Program", task.id, iter, verbose)
        if program is None:
            return None, None, None, None

        try:
            value, timed_out = solver_execution_result(self.execute_code(program))
            runnable = True
        except Exception as error:
            value, runnable, timed_out = getattr(error, "stderr", str(error)), False, None
        if save_data:
            save_log_data(program, f"{output_path}/program_iter_{iter}.py")
            save_log_data(str(value), f"{output_path}/output_iter_{iter}.txt")
        return program, value, runnable, timed_out


formatted_output = (
    "\n\nif model.Status == GRB.OPTIMAL:\n"
    "    print(\"Optimal value:\", model.ObjVal)\n"
    "elif model.Status == GRB.INFEASIBLE:\n"
    "    print(\"Model is infeasible.\")\n"
    "elif model.Status == GRB.UNBOUNDED:\n"
    "    print(\"Model is unbounded.\")\n"
    "else:\n"
    "    print(\"Other status:\", model.Status)\n"
)
