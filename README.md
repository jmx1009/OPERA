# OPERA

OPERA translates optimization problems into mathematical formulations and executable Gurobi programs. It draws on two kinds of modeling knowledge. **Principles** come from domain and optimization references. **Lessons** come from diagnosing a failed attempt, correcting it and verifying the correction on its source task. Both have applicability conditions that can change as the system encounters new tasks.

This repository accompanies *Automated optimization modeling for low-altitude air mobility: A self-evolving LLM insight library framework* by Maoxin Jiang, Junyi Li and Xiqun Chen. It includes the implementation, 210 principles, 363 lessons and cases for all 20 LAFLOR task IDs named in the manuscript. Each task has its own problem statement, formulation and executable reference. [Case catalog](data/laflor/README.md) · [Method](docs/method.md) · [Citation](CITATION.cff)

## Quick start

From the repository root, use Python 3.10 or newer and a Gurobi installation licensed to solve the supplied models. The following commands install the Python dependencies and run the case checks locally.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python tools/verify_examples.py
python -m unittest discover -s tests
```

The verifier runs each reference program and checks its objective with an independent calculation. The two case studies can be run together.

```sh
python tools/verify_examples.py LAFLOR_152 LAFLOR_145
```

| Manuscript case | Verified outcome |
| --- | --- |
| Fare and vertiport capacity · [LAFLOR_152](data/laflor/LAFLOR_152/problem.md) · `principle18` | Of 81 fare-and-pad designs, pads `(1, 2, 1)` at a fare of $85 maximize peak-hour profit at **220.767615**. A zero-wait capacity screen discards this design. |
| Truck and drone pickups and deliveries · [LAFLOR_145](data/laflor/LAFLOR_145/problem.md) · `lesson363` | The truck stays at the depot. The drone flies `0 → 2 → 3 → 1 → 4` carrying 1.8, 1.0, 2.2 and 1.7 kg on successive legs. Its minimum total cost is **23.038041**. |

### Vertiport siting and split deliveries

In LAFLOR_048, the operator accepts entire daily orders and may split deliveries across sites. S1 and S3 must both deliver to C6 at the optimum. Allowing partial credit gives a penalty of 266, the stated rules give 286, and forcing a single site per order gives 316. These are controlled formulation comparisons on the same inputs. [Problem statement](data/laflor/LAFLOR_048/problem.md) · [Optimality certificate and nine comparisons](data/laflor/LAFLOR_048/formulation.md)

```sh
python tools/verify_examples.py LAFLOR_048
python tools/laflor_048.py
```

## Run OPERA

The default configurations use GPT-4o. The following startup checks load the data, knowledge libraries and retrieval index locally. Live model runs read `OPENAI_API_KEY` from the environment.

```sh
python main.py --dry-run
python evaluation.py --dry-run
```

`python main.py` starts each new run with 210 consolidated principles and 139 generic lessons. After initial library learning, each of the four rounds loads both libraries from the preceding checkpoint, evaluates candidate conditions for both principles and lessons, and saves the selected conditions together. The learning demonstration uses all 20 public cases, including the fare-and-pad task `LAFLOR_152`. [Initial principles](data/libraries/initial_principles.json) · [Generic lessons](data/libraries/generic_lessons.json)

The first command evaluates the 210 principles and 363 lessons supplied with the repository. The second evaluates both libraries saved after the fourth round of a training run. [Evaluation principles](data/libraries/principles.json) · [Published lessons](data/libraries/lessons.json)

```sh
python evaluation.py
python evaluation.py --library-path runs/opera_public/library/library_refine_iter4.json --output-folder opera_retrained
```

Each evaluation run scores the same 20 public cases with the selected libraries fixed. The default `evaluation_runs: 3` repeats the evaluation three times and reports the arithmetic mean of the three Pass@1 success rates. Each run keeps its own task records and metrics under `testing/<output-folder>/evaluation_*/run_1/`, `run_2/` and `run_3/`. The enclosing `summary.json` gives the individual rates and their mean. Use `--evaluation-runs 1` for a single run. The paper reports LAFLOR success rates on its 46-task evaluation set. [Case selection and verification](data/laflor/README.md)

### Library ablations

The *Experience library only* condition learns lessons independently of the principle library. Run `python main.py --experience-library-only` on the public examples before evaluating this condition. The resulting checkpoint is written under `runs/opera_public_lessons_only/library/`. Use `--library-path` to select an independently trained checkpoint of your own.

| Configuration | Evaluation command |
| --- | --- |
| Both libraries | `python evaluation.py` |
| Principle library only | `python evaluation.py --experience-library disabled` |
| Experience library only | `python evaluation.py --principle-library disabled` |
| No library | `python evaluation.py --principle-library disabled --experience-library disabled` |

Each command accepts `--dry-run` to check its inputs. With the principle library enabled, retrieval uses both stored principles and task-specific insights derived from document summaries. In each of the three evaluations, all four conditions use one attempt per task and accept a finite objective within 0.1 objective units of the reference, including a zero reference. [Evaluation criterion](docs/method.md#evaluation)

### Other data and settings

Use `python main.py --config path/to/train.yaml` for a training experiment and `python evaluation.py --config path/to/eval.yaml` for evaluation. The following command loads an external evaluation set.

```sh
python evaluation.py --dataset my_split --data-path path/to/tasks.json --output-folder my_split
```

Each external JSON task has `task_id`, `description` and numeric `ground_truth`. Training tasks additionally carry `formulation` and/or `correct_program`. The grader reads the reference objective after generation. Evaluation passes neither the reference formulation nor the reference program to the model. Training writes task records to `learning/` and both library partitions to `runs/`. Evaluation writes to `testing/`.

## Repository guide

| Location | Contents |
| --- | --- |
| [Library learning](library_online_learning.py) | Source-task verification and integration of new lessons |
| [Diagnosis](library_diagnosis.py) and [refinement](library_refinement.py) | Failure reflection and condition selection for principles and lessons |
| [16 manuscript templates](src/prompts/templates.json) | Formulation, programming, retrieval, reflection, extraction and refinement |
| [Principle and experience libraries](data/libraries/) | The starting generic lessons, public library records and taxonomies |
| [Document knowledge](data/knowledge/) | Searchable summaries, candidate principles and their source metadata |
| [LAFLOR](data/laflor/README.md) | Problem statements, mathematical models, reference programs and case provenance |
| [Method](docs/method.md) | Library construction, retrieval, refinement and evaluation |
| [Paper](paper/) | Reserved location for the authors' manuscript PDF |

The software is distributed under the terms of the [MIT License](LICENSE).
