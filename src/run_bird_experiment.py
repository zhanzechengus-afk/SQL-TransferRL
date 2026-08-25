#!/usr/bin/env python3
"""BIRD training and evaluation with the official evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bird_data  # noqa: E402
import run_full_experiment as full  # noqa: E402


@dataclass
class BirdConfig(full.FullConfig):
    bird_train_root: str = "data/bird/train"
    bird_train_marker: str = "data/bird/BIRD_TRAIN_DATA_STAGED"
    bird_dev_json: str = "data/bird/dev/dev.json"
    bird_dev_db_dir: str = "data/bird/dev/dev_databases"
    bird_dev_marker: str = "data/bird/BIRD_DEV_PREFLIGHT_PASSED"
    bird_evaluator: str = "third_party/bird/llm/src/evaluation.py"
    bird_evaluator_commit: str = "483554eae102996f5ec1f4feab4e78ef29c2a394"
    bird_eval_num_cpus: int = 16
    bird_eval_timeout: int = 30
    code_pool_source: str = "artifacts/code_pool.jsonl"
    code_manifest_source: str = "manifests/code_pool.json"
    sqlite_progress_steps: int = 2_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_bird_manifest(
    experiment_dir: Path,
    sql_train: Sequence[Mapping[str, Any]],
    sql_anchor: Sequence[Mapping[str, Any]],
    sql_eval: Sequence[Mapping[str, Any]],
    code_manifest: Mapping[str, Any],
) -> None:
    train_ids = sorted({str(row["db_id"]) for row in sql_train})
    anchor_ids = sorted({str(row["db_id"]) for row in sql_anchor})
    eval_ids = sorted({str(row["db_id"]) for row in sql_eval})
    train_root = Path(str(sql_train[0]["_database_path"])).parents[1]
    config = json.loads((experiment_dir / "config.json").read_text())
    full.write_json(
        experiment_dir / "data_manifest.json",
        {
            "sql_dataset": "BIRD train 2023-07-11 and updated dev 2025-11-06",
            "sql_train_root": str(train_root),
            "sql_train_examples": len(sql_train),
            "sql_anchor_examples": len(sql_anchor),
            "sql_eval_examples": len(sql_eval),
            "sql_train_database_ids": train_ids,
            "sql_anchor_database_ids": anchor_ids,
            "sql_eval_database_ids": eval_ids,
            "train_anchor_overlap": sorted(set(train_ids) & set(anchor_ids)),
            "train_eval_overlap": sorted(set(train_ids) & set(eval_ids)),
            "anchor_eval_overlap": sorted(set(anchor_ids) & set(eval_ids)),
            "official_evaluator_commit": config["bird_evaluator_commit"],
            "official_train_archive_sha256": (
                "66e9e3115b59559554013aa3b124156249f30437a6b4e4f96de3d2dfb5ae8cbc"
            ),
            "official_dev_database_sha256": (
                "aeb211c0e39010bbdae3838bb5e8bd27dc446ed77495b1709f85ccc9bf67f2be"
            ),
            "official_dev_revision": "3c11fb193e5439b338e23677fa0aae11e8b85db9",
            "official_gold_execution_ceiling": 0.9967,
            "code": dict(code_manifest),
        },
    )


def run_official_evaluator(
    examples: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    config: BirdConfig,
    output_dir: Path,
) -> dict[str, float]:
    if len(examples) != len(predictions) or not examples:
        raise ValueError("official evaluation requires aligned non-empty rows")
    gold_path = output_dir / "dev_gold.sql"
    prediction_path = output_dir / "predict_dev.json"
    log_path = output_dir / "official_evaluation.log"
    gold_path.write_text(
        "".join(
            f"{bird_data.gold_bird(row).replace(chr(10), ' ')}\t{row['db_id']}\n"
            for row in examples
        ),
        encoding="utf-8",
    )
    payload = {
        str(index): (
            (prediction.replace("\t", " ").replace("\n", " ").strip()
             or "SELECT 1 WHERE 0")
            + "\t----- bird -----\t"
            + str(example["db_id"])
        )
        for index, (example, prediction) in enumerate(zip(examples, predictions))
    }
    full.write_json(prediction_path, payload)
    difficulty_path = bird_data.write_aligned_difficulty_json(
        examples,
        Path(config.bird_dev_json),
        output_dir / "official_difficulty_subset.json",
    )

    evaluator = Path(config.bird_evaluator)
    evaluation_root = str(output_dir) + os.sep
    command = [
        sys.executable,
        str(evaluator),
        "--predicted_sql_path",
        evaluation_root,
        "--ground_truth_path",
        evaluation_root,
        "--data_mode",
        "dev",
        "--db_root_path",
        str(Path(config.bird_dev_db_dir)) + os.sep,
        "--num_cpus",
        str(config.bird_eval_num_cpus),
        "--meta_time_out",
        str(config.bird_eval_timeout),
        "--diff_json_path",
        str(difficulty_path),
    ]
    result = subprocess.run(
        command,
        cwd=evaluator.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=14_400,
        check=False,
    )
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"official BIRD evaluator failed with code {result.returncode}; "
            f"see {log_path}"
        )
    metrics = {"execution": bird_data.parse_official_execution(result.stdout)}
    full.write_json(output_dir / "official_metrics.json", metrics)
    return metrics


@torch.no_grad()
def evaluate_bird(
    branch: str,
    model: full.core.DomainModel,
    tokenizer: Any,
    examples: Sequence[Mapping[str, Any]],
    config: BirdConfig,
    predictions_path: Path,
) -> dict[str, Any]:
    totals = {"exact": 0, "executable": 0, "denotation": 0, "reward": 0.0}
    official_predictions: list[str] = []
    started = time.time()
    model.eval()
    device = next(model.parameters()).device
    for offset in range(0, len(examples), config.eval_batch_size):
        batch_examples = [
            dict(examples[index])
            for index in range(offset, min(offset + config.eval_batch_size, len(examples)))
        ]
        prompt_rows = [
            full.format_prompt(tokenizer, bird_data.bird_prompt(example))
            for example in batch_examples
        ]
        prompt_limit = config.max_length - config.sql_new_tokens
        if prompt_limit < 1:
            raise ValueError("BIRD prompt budget must be positive")
        prompt_rows = [row[-prompt_limit:] for row in prompt_rows]
        width = max(len(row) for row in prompt_rows)
        padded = [
            [tokenizer.pad_token_id] * (width - len(row)) + row for row in prompt_rows
        ]
        attention = [
            [0] * (width - len(row)) + [1] * len(row) for row in prompt_rows
        ]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention, dtype=torch.long, device=device)
        with model.domain("sql"):
            sequences = model.policy.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.sql_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        completions = tokenizer.batch_decode(
            sequences[:, width:], skip_special_tokens=True
        )
        for local_index, (example, text) in enumerate(zip(batch_examples, completions)):
            score = bird_data.score_bird(
                text, example, max_steps=config.sqlite_progress_steps
            )
            for key in totals:
                totals[key] += float(score[key])
            official_predictions.append(str(score["candidate"]))
            full.append_jsonl(
                predictions_path,
                {
                    "branch": branch,
                    "index": offset + local_index,
                    "question_id": example.get("question_id", offset + local_index),
                    "question": example["question"],
                    **score,
                },
            )
        completed = min(offset + len(batch_examples), len(examples))
        if completed % 64 < config.eval_batch_size or completed == len(examples):
            print(f"eval {branch} {completed}/{len(examples)}", flush=True)

    official = run_official_evaluator(
        examples, official_predictions, config, predictions_path.parent
    )
    count = len(examples)
    return {
        "name": branch,
        "examples": count,
        "exact_match": totals["exact"] / count,
        "execution_rate": official["execution"],
        "denotation_accuracy": official["execution"],
        "single_database_execution": totals["executable"] / count,
        "single_database_denotation": totals["denotation"] / count,
        "mean_reward": totals["reward"] / count,
        "official_metric": "BIRD official execution accuracy",
        "official_gold_execution_ceiling": 0.9967,
        "seconds": time.time() - started,
    }


def _code_manifest(config: BirdConfig, row_count: int) -> dict[str, Any]:
    source_manifest = json.loads(Path(config.code_manifest_source).read_text())
    return {
        **dict(source_manifest.get("code", {})),
        "copied_from": config.code_pool_source,
        "copied_manifest": config.code_manifest_source,
        "copied_rows": row_count,
        "sha256": sha256(Path(config.code_pool_source)),
    }


def finish_prepare(config: BirdConfig) -> None:
    experiment_dir = Path(config.experiment_dir)
    checkpoint = experiment_dir / "common_sql_trainable.pt"
    destination = experiment_dir / "code_pool.jsonl"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing BIRD SQL-SFT checkpoint: {checkpoint}")
    if destination.exists():
        raise FileExistsError("refusing to overwrite an existing code pool")
    source = Path(config.code_pool_source)
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("validated M5 code pool is empty")
    shutil.copy2(source, destination)
    sql_train, sql_anchor, sql_eval = bird_data.load_bird_data(config)
    save_bird_manifest(
        experiment_dir,
        sql_train,
        sql_anchor,
        sql_eval,
        _code_manifest(config, len(rows)),
    )
    print(
        json.dumps(
            {"finished_prepare": str(experiment_dir), "code_examples": len(rows)},
            indent=2,
        ),
        flush=True,
    )


def validate_data(config: BirdConfig) -> None:
    sql_train, sql_anchor, sql_eval = bird_data.load_bird_data(config)
    report = {
        "train_examples": len(sql_train),
        "anchor_examples": len(sql_anchor),
        "eval_examples": len(sql_eval),
        "train_database_ids": len({row["db_id"] for row in sql_train}),
        "anchor_database_ids": len({row["db_id"] for row in sql_anchor}),
        "eval_database_ids": len({row["db_id"] for row in sql_eval}),
        "code_pool_rows": sum(
            1 for line in Path(config.code_pool_source).read_text().splitlines() if line
        ),
        "train_marker": config.bird_train_marker,
        "dev_marker": config.bird_dev_marker,
    }
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = full.build_parser()
    for action in parser._actions:
        if action.dest == "stage":
            action.choices = tuple(action.choices) + ("validate_data",)
            break
    parser.set_defaults(
        max_length=3072,
        sql_new_tokens=256,
        sft_batch_size=1,
        sft_gradient_accumulation=16,
        eval_batch_size=2,
        anchor_micro_batch_size=1,
    )
    parser.add_argument("--bird-train-root", default=BirdConfig.bird_train_root)
    parser.add_argument("--bird-train-marker", default=BirdConfig.bird_train_marker)
    parser.add_argument("--bird-dev-json", default=BirdConfig.bird_dev_json)
    parser.add_argument("--bird-dev-db-dir", default=BirdConfig.bird_dev_db_dir)
    parser.add_argument("--bird-dev-marker", default=BirdConfig.bird_dev_marker)
    parser.add_argument("--bird-evaluator", default=BirdConfig.bird_evaluator)
    parser.add_argument(
        "--bird-evaluator-commit", default=BirdConfig.bird_evaluator_commit
    )
    parser.add_argument(
        "--bird-eval-num-cpus", type=int, default=BirdConfig.bird_eval_num_cpus
    )
    parser.add_argument(
        "--bird-eval-timeout", type=int, default=BirdConfig.bird_eval_timeout
    )
    parser.add_argument("--code-pool-source", default=BirdConfig.code_pool_source)
    parser.add_argument(
        "--code-manifest-source", default=BirdConfig.code_manifest_source
    )
    parser.add_argument(
        "--sqlite-progress-steps",
        type=int,
        default=BirdConfig.sqlite_progress_steps,
    )
    return parser


def _config_from_values(values: Mapping[str, Any]) -> BirdConfig:
    names = {item.name for item in fields(BirdConfig)}
    return BirdConfig(**{key: value for key, value in values.items() if key in names})


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[str, str | None, BirdConfig]:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_stages = {"prepare", "prepare_checkpoint", "validate_data"}
    if args.stage not in setup_stages:
        if args.stage == "branch" and not args.branch:
            parser.error("--branch is required for --stage branch")
        saved = json.loads((Path(args.experiment_dir) / "config.json").read_text())
        config = _config_from_values(saved)
    else:
        config = _config_from_values(vars(args))
        if config.top_p != 1.0:
            parser.error("--top-p must be 1.0 for on-policy RL")
        if config.max_length <= config.sql_new_tokens:
            parser.error("--max-length must exceed --sql-new-tokens")
        if config.sqlite_progress_steps < 1:
            parser.error("--sqlite-progress-steps must be positive")
        if config.bird_eval_num_cpus < 1 or config.bird_eval_timeout < 1:
            parser.error("BIRD evaluator CPU count and timeout must be positive")
        if config.smoke:
            config.sql_train_limit = 64
            config.sql_eval_limit = 32
            config.anchor_examples = 16
            config.anchor_batch_size = 4
            config.anchor_micro_batch_size = 1
            config.source_reward_warmup = 2
            config.code_pilot_cap = 4
            config.rl_steps = 8
            config.sft_batch_size = 1
            config.sft_gradient_accumulation = 4
            config.eval_batch_size = 1
    return args.stage, args.branch, config


def install_bird_hooks(config: BirdConfig) -> None:
    full.core.format_prompt = full.format_prompt
    full.core.code_prompt = full.code_prompt
    full.core.score_code = full.score_code
    full.core.sql_prompt = bird_data.bird_prompt
    full.core.gold_wikisql = bird_data.gold_bird
    full.core.score_sql = lambda text, example: bird_data.score_bird(
        text, example, max_steps=config.sqlite_progress_steps
    )
    full.load_sql_data = bird_data.load_bird_data
    full.save_data_manifest = save_bird_manifest
    full.evaluate_full_wikisql = evaluate_bird


def main() -> None:
    stage, branch_name, config = parse_args()
    install_bird_hooks(config)
    if stage == "validate_data":
        validate_data(config)
    elif stage == "prepare":
        full.prepare_checkpoint(config)
        finish_prepare(config)
    elif stage == "prepare_checkpoint":
        full.prepare_checkpoint(config)
    elif stage == "finish_prepare":
        finish_prepare(config)
    elif stage == "evaluate_checkpoint":
        full.evaluate_checkpoint(config)
    else:
        assert branch_name is not None
        full.branch(config, branch_name)


if __name__ == "__main__":
    main()
