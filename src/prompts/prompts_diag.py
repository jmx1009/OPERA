from .manuscript import bind_template


PROMPT_DIAGNOSE_ISSUES = bind_template(
    "T9",
    PROBLEM_DESCRIPTION="{problem_description}",
    FAILED_FORMULATION="{failed_formulation}",
    REFERENCE_PROGRAM="{correct_program_section}",
    FEEDBACK="{feedback}",
)


PROMPT_INS_REFLECTION = bind_template(
    "T14",
    PROBLEM_DESCRIPTION="{problem_description}",
    FAILED_FORMULATION="{failed_formulation}",
    DIAGNOSED_ISSUES="{diagnosed_issues}",
    REFERENCE_PROGRAM="{correct_program_section}",
    RETRIEVED_INSIGHTS="{retrieved_insights}",
)


PROMPT_RETRI_LABEL = """
You are an expert in Industrial Engineering and Operations Research.

You are given:
1. A problem description for the optimization task
2. A mathematical model proposed by your colleague that failed to yield an optimal solution when solved with the Gurobi optimizer (hereafter referred to as *the failed mathematical model*)
3. A diagnostic report on the proposed mathematical model, identifying the specific issue that prevents optimality


### Problem description
{problem_description}


### The failed mathematical model
(Note: the model is written in LaTeX and presented in a plain-text code block (```))
{failed_formulation}


### The diagnosed issue
{one_diagnosed_issue}


### Your task
Step 1: Carefully review the problem description, and analyse the failed mathematical model with the issue identified in the diagnostic report.

Step 2: Given the full taxonomy dictionaries of existing insights stored in library (shown below), determine the potential level-1 and level-2 label(s) under which relevant useful insights may be found for **resolving the identified issue**.

Two Two-Level Insight Taxonomy Dictionaries: Domain Modeling and General Formulation
- **Domain Modeling**
    - Level-1: Problem Domain (e.g., "Network Flow")
    - Level-2: Domain-specific Technique/Principle (e.g., "Flow Conservation")
- **General Formulation**
    - Level-1: Formulation Component (e.g., "Variable Definition")
    - Level-2: Specific Aspect/Pitfall (e.g., "Continuous vs. Discrete Confusion")


### Taxonomy Dictionary for Domain Modeling
{domain_taxo}

### Taxonomy Dictionary for General Formulation
{formu_taxo}


### STRICT OUTPUT FORMAT
**Return only a JSON object** of your analysis result in Step 2, with the exact structure below:
- Outer keys = "Domain Modeling" or "General Formulation"
- Values = dictionaries whose keys are Level-1 labels from the taxonomy
- Each Level-1 key's value = a list of one or more Level-2 labels from the taxonomy

Note:
- You may assign multiple level-1 and level-2 labels to the issue only when you think they are all potentially applicable.
- If no applicable labels exist the issue, simply set its "matched_label(s)" value to null.

Example 1 - Multiple applicable level-1 and level-2 labels:
{{
    "Domain Modeling": {{
        "Production Planning": ["Inventory Balance Equations", "Time-Indexed Variables"],
        "Resource Allocation": ["Capacity/Resource Balance Equations"]
    }}
{{

Example 2 - Multiple applicable level-1 and level-2 labels from both Domain Modeling and General Formulation:
{{
    "Domain Modeling": {{
        "Facility Location": ["Fixed Charge (Big-M Linking)"]
    }},
    "General Formulation": {{
        "Constraint Formulation": ["Big-M & Indicator Variables"]
    }}
{{

Example 3 - No taxonomy labels apply to the issue:
{{}}


**Guidelines:**
- You must ensure that every label you list exists in the provided taxonomy dictionary exactly as written.
- Only output the JSON object. DO NOT include any explanations, markdown, or extra text before or after the JSON array.

Check the task constraints before returning your answer.
"""


PROMPT_RETRI_INS = """
You are an expert in Industrial Engineering and Operations Research.

You are given:
1. A problem description for the optimization task
2. A mathematical model proposed by your colleague which failed to yield an optimal solution when solved with the Gurobi optimizer (hereafter referred to as *the failed mathematical model*)
3. A diagnostic report on the proposed mathematical model, identifying the specific issue that prevents optimality
4. A collection of insights. Each insight includes:
    - insight_id: the unique identifier of the insight
    - taxonomy: the classification of the modeling/formulation/code-implementation aspect it pertains to
    - condition: the problem-specific context and broader modeling situations in which the insight applies (its applicability condition)


### Problem description
{problem_description}


### The failed mathematical model
Note: the model is written in LaTeX and presented in a plain-text code block (```)
{failed_formulation}


### The diagnosed issue
{one_diagnosed_issue}


### Candidate insights
{candidate_insights}


### Your task
Step 1: Carefully review the problem description, and analyse the failed mathematical model with the issue identified in the diagnostic report.

Step 2: Evaluate each candidate insight individually. Retain only those that directly apply to **resolving the identified issue** in the diagnostic report. For every insight, cite concrete evidence from the problem description and diagnosed issues, and justify how the insight helps fix the specific modeling issue.

Step 3: Rank the applicability of the selected insights in descending order.


### STRICT OUTPUT FORMAT
**Return only a JSON array** of your result from Step 3. Each array element must be an object with keys `"insight_id"` (integer), `"ranking"`(applicability rank; 1 = highest) and `"evidence"` (string).
Example:

```json
[
    {{"insight_id": 1, "ranking": 1, "evidence": "<text>"}},
    {{"insight_id": 5, "ranking": 2, "evidence": "<text>"}},
	{{"insight_id": 7, "ranking": 3, "evidence": "<text>"}}
]
```

**Guidelines:**
- Output only the insights that apply to the identified issue(s).
- Only output the JSON array. DO NOT include any explanations, markdown, or extra text before or after the JSON array.

Check the task constraints before returning your answer.
"""


PROMPT_VALIDATE_ISSUES = """
You are an expert in Industrial Engineering and Operations Research.

You are given:
1. A problem description for the optimization task
2. A mathematical model proposed by your colleague which failed to yield an optimal solution when solved with the Gurobi optimizer (hereafter referred to as *the failed mathematical model*)
3. A diagnostic report on the proposed mathematical model, identifying **all formulation issues** that prevent optimality.
4. The regenerated mathematical model, which was recreated by your colleague based on the diagnostic report.


### Problem description
{problem_description}


### The failed mathematical model
Note: the model is written in LaTeX and presented in a plain-text code block (```)
{failed_formulation}


### The diagnosed issues
{diagnosed_issues}


### The regenerated mathematical model
{new_formulation}


### Your task
Step 1: Carefully review the problem description, and analyse the failed mathematical model with the issues identified in the diagnostic report.

Step 2: Carefully review the regenerated mathematical model and compare it with the failed model to determine whether each previously identified issue has been resolved. For each issue, report:
- "id": A unique id for the issue (integer).
- "status": "solved" or "unsolved".
- "evidence": The evidence supporting the status, including the relevant excerpt from the failed model and the corresponding excerpt from the regenerated model.


### STRICT OUTPUT FORMAT
**Return only a JSON array** of your answer. Each array element must be an object with keys `"id"`, `"status"` and `"evidence"`.

Example:

```json
[
    {{"id": 1, "status": "solved", "evidence": "<text>"}},
	{{"id": 2, "status": "unsolved", "evidence": "<text>"}}
]
```

**Guidelines:**
- Make sure to identify and output each distinct issue. Do NOT miss any issue id in the given diagnostic report.
- Keep each `"evidence"` under 80 words and plain text only.
- Do not include markdown fences, long code excerpts, or repeated quotations from the formulations.
- Only output the JSON array. DO NOT include any explanations, markdown, or extra text before or after the JSON array.

Check the task constraints before returning your answer.
"""


PROMPT_PROGRAM_DIAG = bind_template(
    "T10",
    FAILED_PROGRAM="{failed_program}",
    FEEDBACK="{feedback}",
)
