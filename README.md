# SQL-TransferRL

This repository contains the core implementation accompanying the paper
"Do Code Rewards Transfer to Text-to-SQL? SQL-TransferRL for LLMs." The
defaults below are the paper-aligned settings used by the public entry points.

## Paper-aligned training settings

| Setting | Value |
| --- | --- |
| Learning rate | `1e-4` for Qwen3-0.6B and SmolLM3-3B |
| Optimizer | AdamW, betas `(0.9, 0.95)`, weight decay `0.01` |
| Learning-rate schedule | Constant; no scheduler |
| SQL SFT | One epoch, effective batch size `16` |
| Maximum total sequence length | `512` tokens for WikiSQL, Spider, and BIRD |
| RL batch | One SQL prompt and one code prompt per update |
| Rollouts | One on-policy rollout per prompt |
| RL decoding | Temperature `0.8`, top-p `1.0` |
| Evaluation decoding | Greedy |
| Code coefficient | `0.30` |
| SQL-reference KL | `0.02` for SQL-only RL, matched retention, and SQL-TransferRL |
| Checkpoint interpolation | `0.25` for SQL-only RL, matched retention, and SQL-TransferRL |
| Shared LoRA / adapter width | rank `8` for Qwen3-0.6B, rank `16` for SmolLM3-3B; width `96` |
| Hardware | Eight NVIDIA A100 80 GB GPUs per job |

For the Spider and BIRD entry points, `sft_batch_size=1` and
`sft_gradient_accumulation=16` implement the reported effective SFT batch size
of 16. Their `sql_new_tokens=256` value is the completion budget within the
512-token total sequence limit; it does not increase the context length. When
`--model` selects SmolLM3-3B, the entry points choose rank 16 unless an explicit
`--lora-rank` override is supplied.

## Data protocol

Each official training split is partitioned once into 80% training data and
20% model-selection data. From the training partition, 256 labeled SQL
examples form the disjoint anchor set `D_a`; the remainder form `D_s`. Final
results are reported on the complete official development split, used as the
held-out evaluation set.

| Dataset | Training | Model selection | Held-out evaluation | `D_a` | `D_s` |
| --- | ---: | ---: | ---: | ---: | ---: |
| WikiSQL | 45,084 | 11,271 | 8,421 | 256 | 44,828 |
| Spider | 5,600 | 1,400 | 1,034 | 256 | 5,344 |
| BIRD | 7,542 | 1,886 | 1,534 | 256 | 7,286 |

The auxiliary pool contains 1,362 verified Python tasks from the official
MBPP, APPS, and CodeContests training splits: 369, 493, and 500 tasks,
respectively. Reference solutions are retained only when they pass all
available tests. The default continuation performs exactly 1,362 RL updates.

## Paper controls

- `sql_only`: SQL RL with SQL-reference KL and checkpoint interpolation; no
  code signal.
- `naive_mixed`: SQL and code RL without update alignment, KL, or checkpoint
  interpolation.
- `matched_retention`: the same code transfer as `naive_mixed`, with the same
  KL and interpolation used by the full method.
- `alignment_only`: source normalization, conflict projection, norm matching,
  and the disjoint-SQL agreement gate, without KL or interpolation.
- `target_aligned`: the complete SQL-TransferRL pipeline: update alignment,
  SQL-reference KL, and checkpoint interpolation.

For retention branches, `post_rl_trainable.pt` stores the state before
interpolation and `trainable.pt` stores the deployed state after interpolation.

## Metric names

- WikiSQL `denotation_accuracy` is execution-result accuracy;
  `valid_query_rate` only measures whether SQL parses and executes.
- Spider `execution_accuracy` and `official_structural_exact_match` come from
  the official Test Suite evaluator. Local string equality, executability, and
  single-database denotation are emitted only as diagnostic fields.
- BIRD `execution_accuracy` comes from the official evaluator. The paper's
  `Exact` column is emitted as `normalized_string_exact_match`, while
  `valid_query_rate` is executability rather than answer correctness.

## Prompt format

SQL prompts contain the serialized schema, the natural-language question, and
an instruction to return only one executable SQL query. For example:

```text
Return only an exact, executable solution.
Write one SQLite query.
Columns: c0 (text; player), c1 (real; points).
Question: Which player scored more than 20 points?
Return SQL only.
```

Code prompts request a complete Python 3 program without Markdown or
explanation, followed by `Task:` and the normalized problem statement.

## Entry points

- `src/run_full_experiment.py`: shared SQL-TransferRL training logic and
  WikiSQL defaults.
- `src/run_spider_experiment.py`: Spider data and official Test Suite
  evaluation hooks.
- `src/run_bird_experiment.py`: BIRD data and official evaluation hooks.
- `src/interpolate_checkpoint.py`: post-training SQL-SFT checkpoint
  interpolation.

The repository intentionally keeps configuration in these Python entry points
so that the public defaults and the manuscript can be checked directly. The
main training entry point applies interpolation automatically for every branch
whose paper definition includes retention; the standalone interpolation script
is retained for checkpoint inspection and controlled ablations.
