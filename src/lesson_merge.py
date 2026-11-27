"""Verify lesson replacements in the source attempts that justified their parents."""

from .insight_library import inherit_lesson_sources, lesson_source_attempts
from .train_eval_utils import self_verify_test


def _verify_proposal(replacements, siblings, tasks, library, generator, replaced_stored_ids=()):
    refs = dict.fromkeys(ref for lesson in replacements for ref in lesson_source_attempts(lesson))
    by_id = {task.id: task for task in tasks}
    checked = False
    for ref in refs:
        task_id, iteration, attempt = ref
        task = by_id.get(task_id)
        if task is None:  # Generic source tasks may be outside this training split.
            continue
        records = [row for row in task.attempt_history if row["iteration"] == iteration
                   and (attempt is None or row["attempt"] == attempt)]
        if records:
            contexts = [(row["attempt"], row.get("verification_insights", row["insights"])) for row in records]
        elif attempt is not None:
            return False  # Never substitute an unrelated attempt for a recorded source.
        else:
            # Legacy checkpoints only have the stored retrieval IDs for an iteration.
            if not isinstance(iteration, int) or not 0 <= iteration < len(task.retri_ins_lst):
                return False
            ids = task.retri_ins_lst[iteration]
            contexts = [(None, library.retrieve_insights_by_id(ids))]
        for number, context in contexts:
            relevant = [lesson for lesson in [*siblings, *replacements]
                        if any(tid == task_id and source_iter == iteration
                               and (number is None or source_attempt in (None, number))
                               for tid, source_iter, source_attempt in lesson_source_attempts(lesson))]
            retained = [ins for ins in context if ins.get("insight_id") not in replaced_stored_ids]
            if not self_verify_test(iteration, task, generator, relevant, retained):
                return False
            checked = True
    return checked


def consolidate_lessons(lessons, tasks, library, generator, extractor, target):
    if len(lessons) < 2:
        return lessons, 0, 0
    originals = {i: {**lesson, "insight_id": i} for i, lesson in enumerate(lessons)}
    proposals = extractor.propose_lesson_merge(list(originals.values()), target)
    accepted, consumed = [], set()
    for proposal in proposals:
        ids = set(proposal["merged_ids"])
        inherit_lesson_sources(proposal, [originals[i] for i in proposal["merged_ids"]])
        remaining = [row for i, row in originals.items() if i not in consumed | ids]
        # Recheck prior accepted groups together when a sibling is replaced.
        if _verify_proposal([*accepted, proposal], remaining, tasks, library, generator):
            accepted.append(proposal)
            consumed.update(ids)
    for row in accepted:
        row.pop("merged_ids")  # batch-local IDs have no meaning in the stored library
    return [row for i, row in originals.items() if i not in consumed] + accepted, len(proposals), len(accepted)


def integrate_lesson(lesson, siblings, tasks, library, generator, extractor, iteration):
    proposal, _, parents = extractor.propose_online_lesson_merge([lesson], library)
    accepted = False
    if proposal:
        parents = [p for p in parents if p["insight_type"] == "lesson"]
        ids = {p["insight_id"] for p in parents}
        stored = [ins.to_dict() for ins in library.lessons]
        inherit_lesson_sources(proposal, [lesson, *(row for row in stored if row["insight_id"] in ids)])
        unaffected = [row for row in stored if row["insight_id"] not in ids]
        unaffected.extend(row for row in siblings if row is not lesson)
        accepted = _verify_proposal([proposal], unaffected, tasks, library, generator, ids)
    entry = proposal if accepted else lesson
    library.add_insights([entry], iteration)
    library.update_taxonomy(entry)
    if accepted and parents:
        library.replace_merged_insights(parents)
    return accepted
