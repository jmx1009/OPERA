from .manuscript import bind_template


PROMPT_GENERATE_FORMU = bind_template(
    "T7",
    PROBLEM_DESCRIPTION="{problem_description}",
    INSIGHTS="{insights}",
)


PROMPT_GENERATE_PROGRAM = bind_template(
    "T8",
    PROBLEM_DESCRIPTION="{problem_description}",
    FORMULATION="{mathematical_model}",
    INSIGHTS="{insights}",
)
