# SQL-TransferRL

Reference implementation of **SQL-TransferRL**, a target-aligned cross-domain reinforcement-learning method that transfers execution-verified code rewards to Text-to-SQL.

The released code contains the final `target_aligned` method, the shared/private adapter model, verified code-pool construction, WikiSQL/Spider/BIRD data and evaluation utilities, checkpoint interpolation, and focused unit tests. Model weights, datasets, generated predictions, and internal experiment logs are not included.

## Reproduction protocol

### SQL data

The official training split of each benchmark is partitioned once into 80% train and 20% eval. The official development split is used as the final test set.

| Dataset | Train | Eval | Test | Final metrics |
|---|---:|---:|---:|---|
| WikiSQL | 45,084 | 11,271 | 8,421 | denotation, exact match |
| Spider | 5,600 | 1,400 | 1,034 | Test Suite execution, exact match |
| BIRD | 7,542 | 1,886 | 1,534 | execution, exact match |

SQL SFT maps the serialized question and schema to the gold SQL query. WikiSQL schemas use `c0`, `c1`, ... identifiers with column type and header. Spider and BIRD serialize tables, typed columns, primary keys, and foreign keys; BIRD also includes the supplied evidence field. The exact templates are implemented in `src/run_pilot.py`, `src/spider_data.py`, and `src/bird_data.py`.

The SQL supervision set `D_a` contains 256 examples sampled from the train partition and excluded from the ordinary SQL minibatch stream. Its logical batch size is 16. The normalized anchor direction is refreshed every 8 optimizer updates with EMA coefficient 0.9.

### Verified code pool

Only official training splits are used. A task is retained when a Python reference solution passes every merged public/private/generated test, after deduplication by source task ID and normalized statement.

| Source | Hugging Face dataset | Retained tasks |
|---|---|---:|
| MBPP | `google-research-datasets/mbpp` | 369 |
| APPS | `codeparrot/apps` | 493 |
| CodeContests | `deepmind/code_contests` | 500 |
| **Total** |  | **1,362** |

The aggregate filtering manifest is in `manifests/code_pool.json`. Dataset contents are rebuilt locally and are not redistributed.

### Optimization and decoding

All runs use AdamW with betas `(0.9, 0.95)`, weight decay `0.01`, gradient clipping at `1.0`, one SQL-SFT epoch, SQL-reference KL weight `0.02`, code coefficient `0.30`, and post-training interpolation coefficient `0.25`. Code rewards are normalized per source after 32 observations with EMA `0.95`, standard-deviation floor `0.10`, and clipping to `[-2, 2]`.

| Backbone / task | LR | LoRA rank | Adapter width | SFT batch x accumulation | Eval batch | Context | SQL output |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B / WikiSQL | `1e-4` | 8 | 96 | 16 x 1 | 64 | 512 | 48 |
| Qwen3-0.6B / Spider, BIRD | `1e-4` | 8 | 96 | 1 x 16 | 1 | 3,072 | 256 |
| SmolLM3-3B / Spider, BIRD | `5e-5` | 16 | 96 | 1 x 16 | 2 | 3,072 | 256 |

RL rollouts use temperature `0.8`, top-p `1.0`, and at most 128 new code tokens. Final SQL evaluation uses greedy decoding. Full machine-readable settings are in `configs/paper_config.json`.

### Evaluators

- WikiSQL execution and exact match are implemented in `src/run_pilot.py`.
- Spider uses the official Test Suite evaluator at commit `e97acc546ecbee8fa27fa8dbf025ef61493a876c`.
- BIRD uses the official evaluator at commit `483554eae102996f5ec1f4feab4e78ef29c2a394`.

Training and evaluation jobs were run on eight NVIDIA A100 GPUs with 80 GB of memory each.

## Run the final method

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Prepare WikiSQL SQL SFT and the verified code pool, then run the final method:

```bash
PYTHONPATH=src python src/run_full_experiment.py \
  --stage prepare \
  --experiment-dir outputs/qwen_wikisql \
  --model Qwen/Qwen3-0.6B \
  --reference-kl-weight 0.02 \
  --code-sources mbpp,apps,codecontests \
  --code-pilot-cap 500

PYTHONPATH=src python src/run_full_experiment.py \
  --stage branch \
  --branch target_aligned \
  --experiment-dir outputs/qwen_wikisql
```

Interpolate the RL checkpoint toward SQL SFT before deployment:

```bash
python src/interpolate_checkpoint.py \
  --sft outputs/qwen_wikisql/common_sql_trainable.pt \
  --rl outputs/qwen_wikisql/target_aligned/trainable.pt \
  --alpha 0.25 \
  --output outputs/qwen_wikisql/target_aligned/deploy.pt
```

Spider and BIRD use `src/run_spider_experiment.py` and `src/run_bird_experiment.py`. Their data roots, evaluator roots, and the locally built code pool are command-line arguments; no machine-specific paths are embedded in the release.
