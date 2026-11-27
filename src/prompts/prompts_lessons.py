from .manuscript import bind_template


PROMPT_INS_FROM_FORMU = bind_template(
    "T11",
    PROBLEM_DESCRIPTION="{problem_description}",
    FAILED_FORMULATION="{failed_formulation}",
    REFERENCE_PROGRAM="{correct_program}",
    DOMAIN_TAXONOMY="{domain_taxo}",
    FORMULATION_TAXONOMY="{formulation_taxo}",
)

PROMPT_INS_FROM_PROGRAM = bind_template(
    "T12",
    CODE_TAXONOMY="{code_taxo}",
    CORRECTED_PROGRAM="{corrected_program}",
)


PROMPT_CONDUCT_MERGE = bind_template(
    "T13",
    BATCH_OR_ONLINE="batch",
    NEW_LESSONS="{candidate_insights}",
    STORED_INSIGHTS="[]",
)


PROMPT_ONLINE_MERGE = bind_template(
    "T13",
    BATCH_OR_ONLINE="online",
    NEW_LESSONS="{new_insight}",
    STORED_INSIGHTS="{existing_insights}",
)
