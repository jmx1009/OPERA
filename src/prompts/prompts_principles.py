from .manuscript import bind_template


PROMPT_DOCUMENT_SUMMARY = bind_template(
    "T1",
    TITLE="{title}",
    CATEGORY="{category}",
    LEVEL="{level}",
    EXCERPTS="{excerpts}",
)


PROMPT_CANDIDATE_PRINCIPLES = bind_template(
    "T2",
    TITLE="{title}",
    CATEGORY="{category}",
    TAXONOMY="{taxonomy}",
    SUMMARY="{summary}",
    MAX_PRINCIPLES="{max_insights}",
)


PROMPT_TASK_SPECIFIC_INSIGHTS = bind_template(
    "T4",
    STAGE="{stage}",
    PROBLEM_DESCRIPTION="{problem_description}",
    FORMULATION="{mathematical_model}",
    SOURCE_SUMMARIES="{source_summaries}",
    MAX_INSIGHTS="{max_insights}",
)
