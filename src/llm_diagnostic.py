"""Diagnose modeling failures and verify changes to the formulation or solver program."""
import itertools
import json
import re
import traceback

from .config import training_config
from .insight_states import normalize_reflection_record, supports_formulation
from .llm_programmer import read_program_response
from .prompts.prompts_diag import PROMPT_DIAGNOSE_ISSUES, PROMPT_INS_REFLECTION, PROMPT_VALIDATE_ISSUES, PROMPT_PROGRAM_DIAG
from .train_eval_utils import check_optimality, execute_solver_file, solver_execution_result
from .utils import call_llm_and_parse_with_retry, extract_json_array, save_log_data

config = training_config()


class ProgramDiagnostic:
    def __init__(self, model: str, service: str, temperature: float | None = None):
        self.model = model
        self.service = service
        self.temp = temperature

    def extract_code(self, text: str):
        return read_program_response(text, repair=True)

    def execute_code(self, file_path, timeout_sec=400):
        return execute_solver_file(file_path, timeout_sec)

    def _normalize_execution_error(self, stderr: str) -> str:
        """
        Collapse line-number noise so repeated execution failures can be detected.
        """
        if not stderr:
            return "<empty>"
        normalized = stderr
        normalized = re.sub(r'line \d+', 'line <n>', normalized)
        normalized = re.sub(r'File ".*?"', 'File "<generated>"', normalized)
        normalized = re.sub(r'/[^ \n"]+corrected_program_iter_\d+\.py', '<corrected_program>', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized

    @staticmethod
    def _reference_material(task):
        sections = []
        if task.correct_program:
            sections.append("Reference program:\n" + task.correct_program)
        if task.formulation:
            sections.append("Reference formulation:\n" + task.formulation)
        return "\n\n".join(sections) or "No reference was supplied. Use the stated requirements and execution feedback."

    def _request(self, prompt, parser, retries, verbose, header, error):
        return call_llm_and_parse_with_retry(
            model=self.model, service=self.service, temperature=self.temp, prompt=prompt, parse_fn=parser,
            max_retry=retries, sleep_sec=2, verbose=verbose, log_header=header, error_message=error,
        )

    def _diagnose_issues(self, iter=None, task=None, failed_formulation=None, feedback=None, verbose=False,
                         save_data=False, output_path="learning", failed_program=None):
        prompt = PROMPT_DIAGNOSE_ISSUES.format(
            problem_description=task.desc, failed_formulation=failed_formulation, feedback=feedback,
            correct_program_section=self._reference_material(task),
        )
        if failed_program:
            prompt += f"\nAttempted solver program:\n{failed_program}"
        try:
            issues = self._request(
                prompt, extract_json_array, 5, verbose,
                f"\n==========\n[Iteration {iter}] Diagnose the issues in Task {task.id}\n==========\n",
                f"\n   Task {task.id} failed to diagnose issues from LLM after maximum attempts\n",
            )
        except Exception:
            traceback.print_exc()
            return {}
        if save_data:
            save_log_data(issues, f"{output_path}/Diagnosis/issues_diagnosis_iter_{iter}.json")
        return issues

    @staticmethod
    def _parse_insight_reflection(text, expected_ids):
        records = extract_json_array(text)
        normalized = []
        for record in records:
            record = normalize_reflection_record(record)
            if record.get("state") not in {"useful", "negative", "irrelevant"}:
                raise ValueError("Reflection must label a retrieved insight useful, negative, or irrelevant")
            if "applicable" in record and type(record["applicable"]) is not bool:
                raise ValueError("Legacy applicability evidence must be Boolean")
            normalized.append(record)
        ids = [record["insight_id"] for record in normalized]
        if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
            raise ValueError("Reflection must cover every retrieved insight exactly once")
        return normalized

    def reflect_program_correction(self, task, formulation, failed, feedback, corrected, insights):
        """Attribute execution defects using the original program and its correction blocks."""
        stored = [ins for ins in insights if ins.get("insight_id") is not None]
        if not stored:
            return []
        prompt = (
            "Assess each retrieved implementation rule against the attempted code and observed repair. "
            "Label a correct adopted rule that supports the implementation useful, misleading guidance "
            "negative, and an adopted rule unrelated to this task irrelevant. Return a JSON array "
            "with the supplied insight_id, state and concrete code evidence for each rule.\n"
            + json.dumps({"task": task.desc, "formulation": formulation, "failed_program": failed,
                          "execution_feedback": feedback, "corrected_program": corrected, "insights": stored})
        )
        try:
            return call_llm_and_parse_with_retry(
                model=self.model, service=self.service, prompt=prompt, temperature=0,
                parse_fn=lambda text: self._parse_insight_reflection(text, [ins["insight_id"] for ins in stored]),
                max_retry=5,
            )
        except RuntimeError:
            return []

    def _reflect_insight_usage(self, iter=None, task=None, failed_formulation=None, diagnosed_issues=None,
                               retrieved_insights=(), llm_opt=None, verbose=False, save_data=False, output_path="learning"):
        if diagnosed_issues is None:
            diagnosed_issues = []
        formulation_rules, code_rules = retrieved_insights
        tracked, untracked = [], []
        for rule in formulation_rules:
            if rule.get("insight_id") is None and rule.get("memory_id"):
                rule = {**rule, "insight_id": rule["memory_id"]}
            if rule.get("insight_id") is None:
                untracked.append(rule)
            else:
                tracked.append(rule)
        log_path = f"{output_path}/Diagnosis/insight_reflection_iter_{iter}.json"
        if not tracked:
            if save_data:
                save_log_data([], log_path)
            return {}, untracked, True, failed_formulation, diagnosed_issues

        prompt = PROMPT_INS_REFLECTION.format(
            problem_description=task.desc, failed_formulation=failed_formulation, diagnosed_issues=json.dumps(diagnosed_issues),
            retrieved_insights=tracked, correct_program_section=self._reference_material(task),
        )
        ids = [rule["insight_id"] for rule in tracked]
        try:
            evidence = self._request(
                prompt, lambda text: self._parse_insight_reflection(text, ids), 5, verbose,
                f"\n==========\n[Iteration {iter}] Diagnose the failed mathematical formulation for Task {task.id}\n==========\n",
                f"\n   Task {task.id} failed to diagnose mathematical formulation from LLM after maximum attempts\n",
            )
        except Exception:
            traceback.print_exc()
            return {}, untracked + tracked, True, failed_formulation, diagnosed_issues
        if save_data:
            save_log_data(evidence, log_path)
        states = {row["insight_id"]: row["state"] for row in evidence}
        support = {row["insight_id"] for row in evidence if supports_formulation(row)}
        if len(support) == len(evidence):
            return states, untracked + tracked, True, failed_formulation, diagnosed_issues

        retained = untracked + [rule for rule in tracked if rule["insight_id"] in support]
        attempt = dict(iter=iter, task=task, abl_params=config.ablation, verbose=False, save_data=True, output_path=output_path)
        model = llm_opt.generate_formulation(retrieved_insights=retained, **attempt)
        program, value, ran, timed_out = llm_opt.generate_program(retrieved_insights=code_rules, formulation=model, **attempt)
        if save_data:
            save_log_data(model, f"{output_path}/Diagnosis/formu1_iter_{iter}.py")
        optimal, _, feedback = check_optimality(task=task, output=value, runnable=ran, is_time_out=timed_out)
        if optimal:
            return states, retained, False, model, None
        remaining = self._diagnose_issues(iter=iter, task=task, failed_formulation=model, failed_program=program,
                                          feedback=feedback, verbose=False)
        return states, retained, True, model, remaining

    def _diagnose_unretrieved(
        self, iter=None, task=None, failed_formulation=None, retrieved_insights=None,
        diagnosed_issues=None, llm_opt=None, llm_retri=None, verbose=False,
        save_data=False, output_path="learning", previously_retrieved=(),
    ):
        supporting, code_insights = retrieved_insights
        if not diagnosed_issues or llm_retri is None:
            return {}, [], True, failed_formulation
        excluded = set(previously_retrieved) | {ins.get("insight_id") for ins in supporting}
        candidates = llm_retri.retrieve_insights_for_diagnosis(
            iter=iter, task=task, formulation=failed_formulation, diagnosed_issues=diagnosed_issues,
            filter_fn=lambda ins: ins.insight_id not in excluded, verbose=verbose,
            save_data=save_data, output_path=output_path,
        )
        issues = {row["id"] for row in diagnosed_issues}
        options = [(i, [ins for ins in rows if ins["insight_id"] not in excluded])
                   for i, rows in candidates.items() if i in issues]
        options = [(i, rows) for i, rows in options if rows]
        if not options:
            return {}, [], True, failed_formulation

        best_count, best_insights, best_form = 0, [], failed_formulation
        best_statuses = []
        def generate(context):
            return llm_opt.generate_formulation(
                iter=iter, task=task, retrieved_insights=supporting + context,
                abl_params=config.ablation, save_data=save_data, output_path=output_path,
            )

        def execute(formulation):
            _, value, ran, timed_out = llm_opt.generate_program(
                iter=iter, task=task, retrieved_insights=code_insights, formulation=formulation,
                abl_params=config.ablation, save_data=save_data, output_path=output_path,
            )
            return check_optimality(task, value, ran, timed_out)[0]

        verified = False
        for combo in itertools.islice(itertools.product(*(rows for _, rows in options)), 8):
            chosen = {ins["insight_id"]: ins for ins in combo}
            form = generate(list(chosen.values()))
            if not form:
                continue
            prompt = PROMPT_VALIDATE_ISSUES.format(
                problem_description=task.desc, failed_formulation=failed_formulation,
                diagnosed_issues=json.dumps(diagnosed_issues), new_formulation=form,
            )
            try:
                statuses = call_llm_and_parse_with_retry(
                    model=self.model, service=self.service, prompt=prompt, parse_fn=extract_json_array,
                    temperature=self.temp, max_retry=5, sleep_sec=2, verbose=verbose,
                )
                by_id = {row["id"]: row["status"] for row in statuses}
                if (set(by_id) != issues or len(by_id) != len(statuses)
                        or not set(by_id.values()) <= {"solved", "unsolved"}):
                    continue
            except (ValueError, KeyError, TypeError, RuntimeError):
                continue
            supported = {ins["insight_id"]: ins for (issue_id, _), ins in zip(options, combo)
                         if by_id[issue_id] == "solved"}
            solved = sum(status == "solved" for status in by_id.values())
            if not supported:
                continue
            if solved == len(issues) and execute(form):
                best_insights, best_form, best_statuses = list(supported.values()), form, statuses
                verified = True
                break
            if solved > best_count:
                best_count, best_insights, best_statuses = solved, list(supported.values()), statuses
                best_form = form if chosen.keys() == supported.keys() else None

        if best_insights and not verified:
            best_form = best_form or generate(best_insights)
            if best_form and best_count < len(issues):
                verified = execute(best_form)
        if save_data:
            save_log_data(best_form or failed_formulation, f"{output_path}/Diagnosis/formu2_iter_{iter}.py")
            save_log_data(best_statuses, f"{output_path}/Diagnosis/issues_status_iter_{iter}.json")
        return ({ins["insight_id"]: "unretrieved" for ins in best_insights}, best_insights,
                not verified, best_form or failed_formulation)


    def diagnose_formulation(self, iter=None, task=None, feedback=None, failed_formulation=None, retrieved_insights=(),
                             llm_opt=None, llm_retri=None, verbose=False, save_data=False, output_path="learning",
                             failed_program=None):
        run = dict(iter=iter, task=task, save_data=save_data, output_path=output_path)
        issues = self._diagnose_issues(failed_formulation=failed_formulation, failed_program=failed_program,
                                      feedback=feedback, verbose=False, **run)
        states, retained, needs_retrieval, model, remaining = self._reflect_insight_usage(
            failed_formulation=failed_formulation, diagnosed_issues=issues, retrieved_insights=retrieved_insights,
            llm_opt=llm_opt, verbose=verbose, **run,
        )
        if not needs_retrieval:
            return states, retained, False, model
        missing, additions, needs_lesson, repaired = self._diagnose_unretrieved(
            failed_formulation=model, retrieved_insights=[retained, retrieved_insights[1]],
            previously_retrieved=[rule.get("insight_id") for rule in retrieved_insights[0]],
            diagnosed_issues=remaining, llm_opt=llm_opt, llm_retri=llm_retri, verbose=False, **run,
        )
        states.update(missing)
        return states, retained + additions, needs_lesson, repaired

    def diagnose_program(self, iter=None, task=None, failed_program=None, feedback=None, verbose=False,
                         save_data=False, output_path="learning"):
        program = failed_program
        last_error, consecutive = None, 0
        runnable = False
        path = f"{output_path}/corrected_program_iter_{iter}.py"
        for attempt in range(1, 9):
            prompt = PROMPT_PROGRAM_DIAG.format(failed_program=program, feedback=feedback)
            try:
                program = self._request(
                    prompt, self.extract_code, 3, verbose,
                    f"\n==========\n[Iteration {iter}] Diagnose and correct the failed program for Task {task.id} at attempt {attempt} \n==========\n",
                    f"\n   Task {task.id} failed to extract code from LLM after maximum attempts\n",
                )
            except Exception:
                print(f"Task {task.id}: the program correction response could not be parsed.")
                traceback.print_exc()
                return False, None, None, None
            # The corrected file is also the input to the Python subprocess.
            save_log_data(program, path)
            try:
                value = self.execute_code(path)
                runnable = True
                value, timed_out = solver_execution_result(value)
                optimal, _, feedback = check_optimality(task=task, output=value, runnable=True, is_time_out=timed_out)
                return optimal, True, program, feedback
            except Exception as error:
                stderr = getattr(error, "stderr", None) or str(error)
                feedback = f"Execution error:\n {stderr}"
                signature = self._normalize_execution_error(stderr)
                consecutive = consecutive + 1 if signature == last_error else 1
                last_error = signature
                print(f"Task {task.id}, correction attempt {attempt}: {stderr}")
                if consecutive == 3:
                    print(f"Task {task.id}: stopping after three consecutive equivalent execution errors.")
                    break
        else:
            print(f"Task {task.id}: the correction attempt limit was reached.")
        return None, runnable, None, None
