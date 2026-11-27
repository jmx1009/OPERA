from .manuscript import bind_template


PROMPT_QUICK_MATCH_MODEL = bind_template(
    "T5",
    STAGE="Formulation",
    PROBLEM_DESCRIPTION="{problem_description}",
    FORMULATION="Not available before formulation.",
    TAXONOMY="Domain Modeling: {domain_taxo}\nGeneral Formulation: {formu_taxo}",
)


PROMPT_FULL_CHECK_MODEL = bind_template(
    "T6",
    STAGE="Formulation",
    PROBLEM_DESCRIPTION="{problem_description}",
    FORMULATION="Not available before formulation.",
    CANDIDATE_INSIGHTS="{candidate_insights}",
)

PROMPT_QUICK_MATCH_CODE = bind_template(
    "T5",
    STAGE="Program",
    PROBLEM_DESCRIPTION="{problem_description}",
    FORMULATION="{mathematical_model}",
    TAXONOMY="Code Implementation: {taxo}",
)


PROMPT_FULL_CHECK_CODE = bind_template(
    "T6",
    STAGE="Program",
    PROBLEM_DESCRIPTION="{problem_description}",
    FORMULATION="{mathematical_model}",
    CANDIDATE_INSIGHTS="{candidate_insights}",
)
