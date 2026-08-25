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
| Hardware | Eight NVIDIA A100 80 GB GPUs per job |

For the SmolLM3-3B Spider and BIRD entry points, `sft_batch_size=1` and
`sft_gradient_accumulation=16` implement the reported effective SFT batch size
of 16. Their `sql_new_tokens=256` value is the completion budget within the
512-token total sequence limit; it does not increase the context length.

## Data protocol

Each official training split is partitioned once into 80% training data and
20% model-selection data. From the training partition, 256 labeled SQL
examples form the disjoint anchor set `D_a`; the remainder form `D_s`. Final
results are reported on the complete official development split, used as the
held-out evaluation set.

The auxiliary pool contains 1,362 verified Python tasks from the official
MBPP, APPS, and CodeContests training splits: 369, 493, and 500 tasks,
respectively. Reference solutions are retained only when they pass all
available tests.

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
so that the public defaults and the manuscript can be checked directly.
