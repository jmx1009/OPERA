# OPERA method

## Phase 1. Library construction

OPERA keeps a common schema for principles and lessons. The taxonomy specifies the modeling issue. The condition determines when to retrieve the rule. The explanation states the rule and the example illustrates its application. Formulation uses the Domain Modeling and General Formulation branches. Solver programming uses Code Implementation.

The study used 17 references. Document processing used 500-word segments, 75 words of overlap and hierarchical summaries of groups of about five units. From each document the builder selected the root and up to eight level-one summaries. Documents with at least 120 segments could also contribute up to five level-two summaries. Selection weighted modeling density at 0.75 and redundancy at 0.25. Each selected summary could produce up to five candidate principles.

| Manuscript item | Count | Released material |
| --- | ---: | --- |
| Documents | 17 | Document IDs, source-unit lineage and [bibliography](corpus.md) |
| Segments and summaries | 3,035 | 628 summary nodes, including 627 with retrieval text |
| Summaries selected for extraction | 168 | Selection recorded on each document in the index |
| Candidate principles | 503 | [Candidates](../data/knowledge/candidates.json) |
| Principles after consolidation and expert review | 210 | [Starting principle library](../data/libraries/initial_principles.json) |
| Generic lessons used to initialize learning | 139 | [Starting experience library](../data/libraries/generic_lessons.json) |
| Lessons after learning | 363 | [Experience library](../data/libraries/lessons.json) |

Of the 628 summary nodes, 627 contain searchable text. The remaining node was outside the 168 selected for principle extraction. The document hierarchy can be rebuilt from local copies of the references. PDF inputs require Poppler’s `pdftotext`.

```sh
python tools/build_principles.py --corpus path/to/corpus --output runs/phase_one
python tools/consolidate_principles.py --library runs/phase_one/candidates.json --taxonomy runs/phase_one/candidate_taxonomy.json
```

The first command discovers PDF, Markdown and text files recursively. Subdirectories name knowledge categories. The second writes proposed principles, a taxonomy and a consolidation report under `runs/consolidation/`. Its target size is an upper bound. The manuscript's 210 principles are the outcome of consolidation and expert review. Review any new output against its source documents before using it as a library. To inspect the grouping of the 503 supplied candidates offline, run `python tools/consolidate_principles.py --dry-run`.

## Phase 2. Hybrid retrieval and optimization modeling

For each modeling attempt, document retrieval selects six relevant summaries and derives up to four task-specific insights. Taxonomy screening and applicability checks select stored principles and lessons. The formulation stage uses the task description and formulation insights. Solver programming also receives the generated formulation and the applicable implementation insights.

During library learning, each task receives three initial modeling attempts. A failed formulation is compared with the reference formulation and task statement. An implementation error calls for a code correction. OPERA extracts candidate lessons and regenerates the source task with the failed attempt’s insight context and the candidates. The extraction and verification budget is three rounds. Source-task verification determines which lessons can enter the experience library. Proposed merges are checked on the associated available tasks with unaffected insights retained.

## Phase 3. Library refinement

**Failure reflection** labels retrieved insights as useful, negative or irrelevant. Repair can identify unretrieved insights. Diagnosis evaluates up to eight candidate combinations.

**Applicability condition refinement** applies to principles and lessons in each of four rounds. Template T15 asks for an evidence-backed clause and a reason, or an empty object when the task supplies insufficient evidence. An unretrieved task can support “This insight applies when …”. A negative or irrelevant task can support “This insight does NOT apply when …”. T16 requests five distinct candidates. Each preserves the complete current condition verbatim and appends one or more paragraphs using the openings supported by that round's evidence.

Each candidate undergoes an applicability check on the associated tasks in a temporary library containing only the target insight, with taxonomy matching bypassed. Retrieving the insight for a useful or unretrieved task counts as correct, as does excluding it for a negative or irrelevant task. OPERA adopts the highest-scoring candidate only if its fraction of correct decisions exceeds the recorded baseline. A tie with the baseline retains the original condition. The T16 parser requires a single JSON array with exactly the fields `path_id`, `strategy` and `new_condition` in each candidate. The retrieval evaluator supplies the scores.

For each evaluated principle or lesson, the checkpoint records the original condition, task evidence, baseline matches, candidate texts, retrieval decisions and selected path index. The `generation_responses` field links a T16 response to its candidate batch. Each candidate's `checks` records the task, stage and associated model responses. Successful calls record the model name, completion ID and creation time returned by the service, alongside the requested model and SHA-256 digests of the messages, raw response and parsed JSON. Missing response metadata remains null. An unsuccessful retrieval check has a null score. These records can be compared with the original run's saved responses when establishing which experiment produced a checkpoint.

Initial library learning writes the base checkpoint. Round one loads both partitions from that checkpoint. Each later round starts by loading the preceding round's refined principles, lessons and taxonomy. It then performs failure reflection and condition refinement and saves both resulting partitions together. A condition with insufficient evidence or no improvement carries forward. Setting `start_iter` resumes the same sequence from the saved libraries and task records. Setting `resume_refinement_iter` repeats condition selection from the specified round's diagnosis checkpoint.

## Evaluation

In training, source-task verification and evaluation, a successfully executed program passes the numerical check when it returns a finite objective $z$ satisfying $|z-z^*|\leq 0.1$, where $z^*$ is the reference objective. The tolerance is expressed in the task's objective units. For a zero reference, the accepted interval is $[-0.1,0.1]$. For each configuration evaluated in this study, the same tasks are evaluated three times with fixed libraries and settings. Each task receives one modeling attempt per evaluation. For a benchmark with $N$ tasks, let $S_r$ denote the number of successful tasks in evaluation $r$. The reported success rate is $(S_1/N + S_2/N + S_3/N)/3$. Baseline values taken from prior publications retain their original evaluation protocols.

The training and evaluation configurations set `tolerance_mode: absolute` and `tolerance: 0.1`. A configuration that supplies only a numerical tolerance also uses absolute error. Evaluation logs record both the mode and threshold.

| Evaluation benchmark | Paper evaluation tasks | OPERA mean success rate (three evaluations) |
| --- | ---: | ---: |
| NLP4LP | 73 | 89.0% |
| NL4OPT | 64 | 87.5% |
| IndustryOR | 25 | 72.0% |
| MAMO ComplexLP | 34 | 88.2% |
| LogiOR | 92 | 71.7% |
| OptiBench | 403 | 93.8% |
| LAFLOR | 46 | 58.7% |
| All benchmarks | 737 | 86.8% micro-average, 80.1% macro-average |

On the 46-task LAFLOR evaluation set, the reported mean success rates across the three evaluations are 32.6% for *No library*, 50.0% for *Principle library only*, 52.2% for *Experience library only* and 58.7% for *Both libraries*. The public catalog supports a separate demonstration on 20 named examples. [LAFLOR case catalog](../data/laflor/README.md)

## Library provenance

The released taxonomy follows the classification used for the manuscript. The counts below group entries by their top-level modeling category.

| Library | Domain Modeling | General Formulation | Code Implementation |
| --- | ---: | ---: | ---: |
| Principles | 94 | 69 | 47 |
| Lessons | 167 | 140 | 56 |

Each principle links to its source summaries, directly or through its recorded candidate parents. In each published lesson, `source_task_ids` uses `initial lessons` for contributions from the generic starting library and lists individual LAFLOR tasks by ID. A lesson with both sources lists both. Fresh training loads the 210 consolidated conditions and 139 generic lessons. By default, evaluation loads 210 principles and 363 lessons. The training and evaluation principle files share identifiers, taxonomy, explanations, examples and source references. To evaluate a particular training run, pass its joint checkpoint to `evaluation.py --library-path`. [Training principles](../data/libraries/initial_principles.json) · [Evaluation principles](../data/libraries/principles.json)

Appendices A.3 and C give the original records for `principle17`, `principle18`, `principle103`, `lesson278`, `lesson307`, `lesson362` and `lesson363`.

Appendix D gives four round-specific examples with the original condition excerpt, the added clauses and the matching counts. [Refinement examples](../data/libraries/refinement_cases.json)

The workflow uses the sixteen instruction templates from Appendix B. [Instruction templates](../src/prompts/templates.json)
