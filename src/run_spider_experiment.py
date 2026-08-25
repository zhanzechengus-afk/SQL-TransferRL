#!/usr/bin/env python3
"""Spider training and evaluation with the official Test Suite evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_full_experiment as full  # noqa: E402
import spider_data  # noqa: E402
import spider_score_isolation  # noqa: E402


SMOLLM3_MODEL = "HuggingFaceTB/SmolLM3-3B"
SMOLLM3_REVISION = "a07cc9a04f16550a088caea529712d1d335b0ac1"
SMOLLM3_CACHE_DIR = ""


def format_smollm_prompt(tokenizer: Any, prompt: str) -> list[int]:
    messages = [
        {
            "role": "system",
            "content": "Return only an exact, executable solution. /no_think",
        },
        {"role": "user", "content": prompt},
    ]
    kwargs = {"tokenize": True, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


@dataclass
class SpiderConfig(full.FullConfig):
    spider_data_dir: str = "data/spider"
    spider_test_suite_db_dir: str = "data/spider-test-suite/database"
    spider_evaluator_dir: str = "third_party/test-suite-sql-eval"
    spider_eval_pythonpath: str = ""
    spider_nltk_data: str = ""
    code_pool_source: str = "artifacts/code_pool.jsonl"
    code_manifest_source: str = "manifests/code_pool.json"
    sqlite_progress_steps: int = 2_000_000
    spider_score_timeout: int = 30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_spider_manifest(
    experiment_dir: Path,
    sql_train: Sequence[Mapping[str, Any]],
    sql_anchor: Sequence[Mapping[str, Any]],
    sql_eval: Sequence[Mapping[str, Any]],
    code_manifest: Mapping[str, Any],
) -> None:
    train_ids = sorted({str(row["db_id"]) for row in sql_train})
    anchor_ids = sorted({str(row["db_id"]) for row in sql_anchor})
    eval_ids = sorted({str(row["db_id"]) for row in sql_eval})
    full.write_json(
        experiment_dir / "data_manifest.json",
        {
            "sql_dataset": "Spider 1.0 official 2020-08 data",
            "sql_data_dir": str(sql_train[0]["_database_path"]).split("/database/")[0],
            "sql_train_examples": len(sql_train),
            "sql_anchor_examples": len(sql_anchor),
            "sql_eval_examples": len(sql_eval),
            "sql_train_database_ids": train_ids,
            "sql_anchor_database_ids": anchor_ids,
            "sql_eval_database_ids": eval_ids,
            "train_anchor_overlap": sorted(set(train_ids) & set(anchor_ids)),
            "train_eval_overlap": sorted(set(train_ids) & set(eval_ids)),
            "anchor_eval_overlap": sorted(set(anchor_ids) & set(eval_ids)),
            "official_evaluator_commit": "e97acc546ecbee8fa27fa8dbf025ef61493a876c",
            "official_data_zip_sha256": (
                "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"
            ),
            "official_test_suite_zip_sha256": (
                "9ec24ea8debc6bd04abfe137b5f1a739b5a8836f32c0464e4dfc94eb7f41da96"
            ),
            "code": dict(code_manifest),
        },
    )


def parse_official_metrics(output: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in output.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith("execution"):
            values = re.findall(r"\d+\.\d+", stripped)
            if values:
                metrics["execution"] = float(values[-1])
        elif stripped.startswith("exact match"):
            values = re.findall(r"\d+\.\d+", stripped)
            if values:
                metrics["exact_match"] = float(values[-1])
    if set(metrics) != {"execution", "exact_match"}:
        raise ValueError(f"could not parse official Spider metrics: {metrics}")
    return metrics


def run_official_evaluator(
    examples: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    config: SpiderConfig,
    output_dir: Path,
) -> dict[str, float]:
    if len(examples) != len(predictions) or not examples:
        raise ValueError("official evaluation requires aligned non-empty rows")
    gold_path = output_dir / "official_gold.sql"
    prediction_path = output_dir / "official_predictions.sql"
    log_path = output_dir / "official_evaluation.log"
    gold_path.write_text(
        "".join(f"{spider_data.gold_spider(row).rstrip(';')}\t{row['db_id']}\n" for row in examples),
        encoding="utf-8",
    )
    safe_predictions = [
        (prediction.replace("\t", " ").replace("\n", " ").strip() or "SELECT 1 WHERE 0")
        for prediction in predictions
    ]
    prediction_path.write_text(
        "".join(f"{prediction}\n" for prediction in safe_predictions),
        encoding="utf-8",
    )

    evaluator_dir = Path(config.spider_evaluator_dir)
    command = [
        sys.executable,
        str(evaluator_dir / "evaluation.py"),
        "--gold",
        str(gold_path),
        "--pred",
        str(prediction_path),
        "--etype",
        "all",
        "--db",
        config.spider_test_suite_db_dir,
        "--table",
        str(Path(config.spider_data_dir) / "tables.json"),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            config.spider_eval_pythonpath,
            env.get("PYTHONPATH", ""),
        )
        if value
    )
    env["NLTK_DATA"] = config.spider_nltk_data
    result = subprocess.run(
        command,
        cwd=evaluator_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=7_200,
        check=False,
    )
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"official Spider evaluator failed with code {result.returncode}; "
            f"see {log_path}"
        )
    metrics = parse_official_metrics(result.stdout)
    full.write_json(output_dir / "official_metrics.json", metrics)
    return metrics


@torch.no_grad()
def evaluate_spider(
    branch: str,
    model: full.core.DomainModel,
    tokenizer: Any,
    examples: Sequence[Mapping[str, Any]],
    config: SpiderConfig,
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
            full.format_prompt(tokenizer, spider_data.spider_prompt(example))
            for example in batch_examples
        ]
        prompt_limit = config.max_length - config.sql_new_tokens
        if prompt_limit < 1:
            raise ValueError("Spider prompt budget must be positive")
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
            score = spider_data.score_spider(
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
                    "question": example["question"],
                    **score,
                },
            )
        completed = min(offset + len(batch_examples), len(examples))
        if completed % 64 < config.eval_batch_size or completed == len(examples):
            print(f"eval {branch} {completed}/{len(examples)}", flush=True)

    official = run_official_evaluator(
        examples,
        official_predictions,
        config,
        predictions_path.parent,
    )
    count = len(examples)
    return {
        "name": branch,
        "examples": count,
        "exact_match": official["exact_match"],
        "execution_rate": official["execution"],
        "denotation_accuracy": official["execution"],
        "single_database_exact": totals["exact"] / count,
        "single_database_execution": totals["executable"] / count,
        "single_database_denotation": totals["denotation"] / count,
        "mean_reward": totals["reward"] / count,
        "official_metric": "Spider test-suite execution without value plugging",
        "seconds": time.time() - started,
    }


def _code_manifest(config: SpiderConfig, row_count: int) -> dict[str, Any]:
    source_manifest = json.loads(Path(config.code_manifest_source).read_text())
    return {
        **dict(source_manifest.get("code", {})),
        "copied_from": config.code_pool_source,
        "copied_manifest": config.code_manifest_source,
        "copied_rows": row_count,
        "sha256": sha256(Path(config.code_pool_source)),
    }


def finish_prepare(config: SpiderConfig) -> None:
    experiment_dir = Path(config.experiment_dir)
    checkpoint = experiment_dir / "common_sql_trainable.pt"
    destination = experiment_dir / "code_pool.jsonl"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing Spider SQL-SFT checkpoint: {checkpoint}")
    if destination.exists():
        raise FileExistsError("refusing to overwrite an existing code pool")
    source = Path(config.code_pool_source)
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("validated M5 code pool is empty")
    shutil.copy2(source, destination)
    sql_train, sql_anchor, sql_eval = spider_data.load_spider_data(config)
    save_spider_manifest(
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


def validate_data(config: SpiderConfig) -> None:
    sql_train, sql_anchor, sql_eval = spider_data.load_spider_data(config)
    test_suite_dir = Path(config.spider_test_suite_db_dir)
    eval_ids = {str(row["db_id"]) for row in sql_eval}
    missing_test_suites = sorted(
        database_id
        for database_id in eval_ids
        if not list((test_suite_dir / database_id).glob("*.sqlite"))
    )
    if missing_test_suites:
        raise FileNotFoundError(
            f"missing official Spider test suites: {missing_test_suites}"
        )
    report = {
        "train_examples": len(sql_train),
        "anchor_examples": len(sql_anchor),
        "eval_examples": len(sql_eval),
        "train_database_ids": len({row["db_id"] for row in sql_train}),
        "anchor_database_ids": len({row["db_id"] for row in sql_anchor}),
        "eval_database_ids": len(eval_ids),
        "test_suite_sqlite_files": sum(
            len(list((test_suite_dir / database_id).glob("*.sqlite")))
            for database_id in eval_ids
        ),
        "code_pool_rows": sum(
            1 for line in Path(config.code_pool_source).read_text().splitlines() if line
        ),
    }
    print(json.dumps(report, indent=2))


def train_only_branch(config: SpiderConfig, branch_name: str) -> None:
    """Run the frozen branch update path without spending a full-dev evaluation."""

    experiment_dir = Path(config.experiment_dir)
    branch_dir = experiment_dir / branch_name
    branch_dir.mkdir(parents=True, exist_ok=False)
    full.write_json(
        branch_dir / "config.json",
        {**asdict(config), "branch": branch_name, "evaluation_deferred": True},
    )
    full.core.seed_everything(config.seed + 101)
    tokenizer = full.tokenizer_for(config)
    sql_train, sql_anchor, _ = spider_data.load_spider_data(config)
    code_pool = full.load_saved_code_pool(experiment_dir / "code_pool.jsonl")
    model = full.core.restore_model(
        config, tokenizer, experiment_dir / "common_sql_trainable.pt"
    )
    full.train_branch(
        branch_name,
        model,
        tokenizer,
        sql_train,
        sql_anchor,
        code_pool,
        config,
        branch_dir / "trace.jsonl",
    )
    full.release_cuda_memory()
    full.atomic_save_trainable(model, branch_dir / "trainable.pt")
    full.release_cuda_memory()
    full.write_json(
        branch_dir / "TRAIN_ONLY_COMPLETE.json",
        {"branch": branch_name, "updates": len(code_pool)},
    )
    print(
        json.dumps(
            {"trained_without_evaluation": branch_name, "updates": len(code_pool)},
            indent=2,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = full.build_parser()
    for action in parser._actions:
        if action.dest == "stage":
            action.choices = tuple(action.choices) + (
                "validate_data",
                "train_only_branch",
            )
            break
    parser.set_defaults(
        model=SMOLLM3_MODEL,
        model_revision=SMOLLM3_REVISION,
        model_cache_dir=SMOLLM3_CACHE_DIR,
        max_length=512,
        sql_new_tokens=256,
        sft_batch_size=1,
        sft_gradient_accumulation=16,
        eval_batch_size=2,
        anchor_micro_batch_size=1,
    )
    parser.add_argument(
        "--spider-data-dir", default=SpiderConfig.spider_data_dir
    )
    parser.add_argument(
        "--spider-test-suite-db-dir",
        default=SpiderConfig.spider_test_suite_db_dir,
    )
    parser.add_argument(
        "--spider-evaluator-dir", default=SpiderConfig.spider_evaluator_dir
    )
    parser.add_argument(
        "--spider-eval-pythonpath", default=SpiderConfig.spider_eval_pythonpath
    )
    parser.add_argument(
        "--spider-nltk-data", default=SpiderConfig.spider_nltk_data
    )
    parser.add_argument("--code-pool-source", default=SpiderConfig.code_pool_source)
    parser.add_argument(
        "--code-manifest-source", default=SpiderConfig.code_manifest_source
    )
    parser.add_argument(
        "--sqlite-progress-steps",
        type=int,
        default=SpiderConfig.sqlite_progress_steps,
    )
    parser.add_argument(
        "--spider-score-timeout",
        type=int,
        default=SpiderConfig.spider_score_timeout,
    )
    return parser


def _config_from_values(values: Mapping[str, Any]) -> SpiderConfig:
    names = {item.name for item in fields(SpiderConfig)}
    return SpiderConfig(**{key: value for key, value in values.items() if key in names})


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[str, str | None, SpiderConfig]:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_stages = {"prepare", "prepare_checkpoint", "validate_data"}
    if args.stage not in setup_stages:
        if args.stage in {"branch", "train_only_branch"} and not args.branch:
            parser.error(f"--branch is required for --stage {args.stage}")
        saved = json.loads((Path(args.experiment_dir) / "config.json").read_text())
        config = _config_from_values(saved)
    else:
        config = _config_from_values(vars(args))
        if config.top_p != 1.0:
            parser.error("--top-p must be 1.0 for on-policy RL")
        if config.max_length <= config.sql_new_tokens:
            parser.error("--max-length must exceed --sql-new-tokens")
        if config.kl_token_chunk_size < 1:
            parser.error("--kl-token-chunk-size must be positive")
        if config.sqlite_progress_steps < 1 or config.spider_score_timeout < 1:
            parser.error("Spider scorer progress and wall timeouts must be positive")
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


def install_spider_hooks(config: SpiderConfig) -> None:
    full.format_prompt = format_smollm_prompt
    full.core.format_prompt = format_smollm_prompt
    full.core.code_prompt = full.code_prompt
    full.core.score_code = full.score_code
    full.core.sql_prompt = spider_data.spider_prompt
    full.core.gold_wikisql = spider_data.gold_spider
    full.core.score_sql = lambda text, example: spider_score_isolation.score_spider_isolated(
        text,
        example,
        max_steps=config.sqlite_progress_steps,
        timeout_seconds=config.spider_score_timeout,
    )
    full.load_sql_data = spider_data.load_spider_data
    full.save_data_manifest = save_spider_manifest
    full.evaluate_full_wikisql = evaluate_spider


def main() -> None:
    stage, branch_name, config = parse_args()
    install_spider_hooks(config)
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
    elif stage == "train_only_branch":
        assert branch_name is not None
        train_only_branch(config, branch_name)
    else:
        assert branch_name is not None
        full.branch(config, branch_name)


if __name__ == "__main__":
    main()
