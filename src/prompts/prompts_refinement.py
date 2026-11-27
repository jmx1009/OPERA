from .manuscript import bind_template


PROMPT_INS_NEG = bind_template(
    "T15",
    STATE="negative",
    TARGET_INSIGHT="{target_insight}",
    PROBLEM_DESCRIPTION="{desc}",
    EVIDENCE="{diag_evidence}",
)


PROMPT_INS_UNR = bind_template(
    "T15",
    STATE="unretrieved",
    TARGET_INSIGHT="{target_insight}",
    PROBLEM_DESCRIPTION="{desc}",
    EVIDENCE="{diag_evidence}",
)


PROMPT_INS_IRRELEVANT = bind_template(
    "T15",
    STATE="irrelevant",
    TARGET_INSIGHT="{target_insight}",
    PROBLEM_DESCRIPTION="{desc}",
    EVIDENCE="{diag_evidence}",
)

PROMPT_INS_REFINEMENT = bind_template(
    "T16",
    ORIGINAL_INSIGHT="{original_insight}",
    APPLICABILITY_CLAUSES="{unr_conditions}",
    EXCLUSION_CLAUSES="Negative: {neg_conditions}\nIrrelevant: {irrelevant_conditions}",
    NUMBER_OF_CANDIDATES="{path_k}",
)
