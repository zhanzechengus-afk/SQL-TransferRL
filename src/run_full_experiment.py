#!/usr/bin/env python3
"""WikiSQL training entry point for SQL-TransferRL and its controls."""

from __future__ import annotations

import argparse
import atexit
import ast
import concurrent.futures
import gc
import json
import math
import multiprocessing
import os
import random
import re
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import code_data  # noqa: E402
import run_pilot as core  # noqa: E402


BRANCH_NAMES = (
    "sql_only",
    "naive_mixed",
    "norm_matched",
    "source_normalized",
    "projection_only",
    "target_aligned",
    "code_sft_only",
)

# Full code corpora commonly use ``sys.stdin``. Keep it available while
# rejecting process, network, and filesystem access inside verifier programs.
UNSAFE_CODE = re.compile(
    r"\b(?:os|subprocess|socket|shutil|pathlib|requests|urllib)\b|"
    r"\b(?:open|eval|exec|compile|__import__)\s*\(",
    flags=re.IGNORECASE,
)


@dataclass
class FullConfig:
    model: str = "Qwen/Qwen3-0.6B"
    model_revision: str = ""
    model_cache_dir: str = ""
    seed: int = 13
    learning_rate: float = 1.0e-4
    gradient_clip: float = 1.0
    adapter_dim: int = 96
    adapter_type: str = "output"
    adapter_top_k: int = 6
    lora_rank: int = 8
    max_length: int = 512
    sql_new_tokens: int = 48
    code_new_tokens: int = 128
    code_test_workers: int = 8
    sql_sft_weight: float = 0.20
    code_sft_weight: float = 0.10
    auxiliary_weight: float = 0.30
    max_aux_scale: float = 2.0
    alignment_temperature: float = 0.02
    temperature: float = 0.8
    top_p: float = 1.0
    rl_loss: str = "ema_reinforce"
    reference_kl_weight: float = 0.0
    kl_token_chunk_size: int = 8
    sql_sft_epochs: int = 1
    sft_batch_size: int = 16
    sft_gradient_accumulation: int = 1
    eval_batch_size: int = 64
    sql_train_limit: int = 0
    sql_eval_limit: int = 0
    anchor_examples: int = 256
    anchor_batch_size: int = 16
    anchor_micro_batch_size: int = 0
    anchor_refresh_steps: int = 8
    anchor_ema_beta: float = 0.90
    source_reward_warmup: int = 32
    source_reward_beta: float = 0.95
    source_reward_std_floor: float = 0.10
    source_reward_clip: float = 2.0
    code_sources: str = "mbpp,apps,codecontests,taco"
    code_pilot_cap: int = 0
    rl_steps: int = 0
    experiment_dir: str = "full_experiment"
    smoke: bool = False


@dataclass(frozen=True)
class BranchSpec:
    uses_code: bool
    uses_code_reward: bool = True
    norm_match: bool = False
    source_normalized: bool = False
    project_conflicts: bool = False
    anchor_gate: bool = False


BRANCH_SPECS = {
    "sql_only": BranchSpec(False, uses_code_reward=False),
    "naive_mixed": BranchSpec(True),
    "norm_matched": BranchSpec(True, norm_match=True),
    "source_normalized": BranchSpec(True, norm_match=True, source_normalized=True),
    "projection_only": BranchSpec(
        True, norm_match=True, source_normalized=True, project_conflicts=True
    ),
    "target_aligned": BranchSpec(
        True,
        norm_match=True,
        source_normalized=True,
        project_conflicts=True,
        anchor_gate=True,
    ),
    "code_sft_only": BranchSpec(True, uses_code_reward=False),
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, default=str) + "\n")


def release_cuda_memory() -> None:
    """Release dead training graphs before checkpointing or evaluation."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _load_trainable_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint is not a state dictionary: {path}")
    return value


def validate_trainable_state(state: Mapping[str, Any]) -> dict[str, int]:
    tensors = {
        str(name): value for name, value in state.items() if isinstance(value, torch.Tensor)
    }
    if not tensors or len(tensors) != len(state):
        raise ValueError("trainable checkpoint must contain only non-empty tensor entries")
    for name, tensor in tensors.items():
        if tensor.device.type != "cpu":
            raise ValueError(f"checkpoint tensor is not on CPU: {name}")
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"checkpoint tensor is non-finite: {name}")
    return {
        "tensors": len(tensors),
        "parameters": sum(int(tensor.numel()) for tensor in tensors.values()),
    }


def atomic_save_trainable(model: core.DomainModel, path: Path) -> dict[str, int]:
    """Write, fsync, reload, and atomically publish one trainable checkpoint."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale checkpoint temporary exists: {temporary}")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.trainable_state()
    expected = validate_trainable_state(state)
    try:
        with temporary.open("xb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.stat().st_size <= 0:
            raise IOError(f"checkpoint write produced an empty file: {temporary}")
        loaded = _load_trainable_state(temporary)
        observed = validate_trainable_state(loaded)
        if set(loaded) != set(state):
            raise ValueError("checkpoint reload changed the trainable key set")
        for name, tensor in state.items():
            restored = loaded[name]
            if restored.shape != tensor.shape or restored.dtype != tensor.dtype:
                raise ValueError(f"checkpoint reload changed tensor metadata: {name}")
        if observed != expected:
            raise ValueError(
                f"checkpoint reload count mismatch: expected={expected}, observed={observed}"
            )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    manifest = {**expected, "bytes": path.stat().st_size}
    write_json(path.with_name(path.name + ".manifest.json"), manifest)
    del state
    release_cuda_memory()
    return manifest


def format_prompt(tokenizer: Any, prompt: str) -> list[int]:
    messages = [
        {"role": "system", "content": "Return only an exact, executable solution."},
        {"role": "user", "content": prompt},
    ]
    kwargs = {"tokenize": True, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def code_prompt(example: Mapping[str, Any]) -> str:
    verifier = str(example.get("verifier", "stdin"))
    instruction = (
        "Return one complete Python 3 program."
        if verifier == "stdin"
        else "Return the complete Python function implementation."
    )
    statement = example.get("prompt", example.get("statement", ""))
    return f"{instruction} Do not use Markdown or explanation.\nTask: {statement}"


def normalize_stdout(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "\n".join(normalize_stdout(item) for item in value)
    return "\n".join(line.rstrip() for line in str(value).strip().splitlines())


def limited_process() -> None:
    limits = [
        (resource.RLIMIT_CPU, 4),
        (resource.RLIMIT_FSIZE, 1_000_000),
        (resource.RLIMIT_NOFILE, 32),
    ]
    # A forked macOS child inherits the already-loaded PyTorch address space;
    # lowering RLIMIT_AS before exec then fails. Linux workers retain the cap.
    if sys.platform != "darwin":
        limits.append((resource.RLIMIT_AS, 1_500_000_000))
    for kind, requested_soft_limit in limits:
        _, hard_limit = resource.getrlimit(kind)
        soft_limit = (
            requested_soft_limit
            if hard_limit == resource.RLIM_INFINITY
            else min(requested_soft_limit, hard_limit)
        )
        # Retaining the inherited hard limit works on Linux and avoids a
        # macOS error when a child tries to lower an infinite hard limit.
        resource.setrlimit(kind, (soft_limit, hard_limit))


def run_source(source: str, stdin: str = "") -> tuple[bool, str, str]:
    if UNSAFE_CODE.search(source):
        return False, "", "unsafe construct rejected"
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return False, "", f"SyntaxError: {exc}"
    with tempfile.TemporaryDirectory(prefix="code_reward_") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(source + "\n", encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(path)],
                input=stdin,
                cwd=directory,
                env={"PATH": os.environ.get("PATH", "")},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                preexec_fn=limited_process,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "", "timeout"
    return result.returncode == 0, result.stdout, result.stderr.strip()[-240:]


def has_valid_python_syntax(source: str, timeout_seconds: int = 5) -> bool:
    """Parse generated source outside the training process with a hard timeout."""
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                "import ast, sys; ast.parse(sys.stdin.read())",
            ],
            input=source,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
            preexec_fn=limited_process,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def _function_assertion(test: Mapping[str, Any]) -> str:
    function_name = str(test["function_name"])
    value = test.get("input")
    if isinstance(value, Mapping):
        call = f"{function_name}(**{dict(value)!r})"
    elif isinstance(value, (list, tuple)):
        call = f"{function_name}(*{list(value)!r})"
    else:
        call = f"{function_name}({value!r})"
    return f"assert {call} == {test.get('expected')!r}"


def execute_code_test(
    candidate: str,
    test: Mapping[str, Any],
    setup: str = "",
) -> tuple[bool, str]:
    kind = str(test.get("kind", "stdin"))
    if kind == "assert":
        source = "\n".join(part for part in (setup, candidate, str(test["code"])) if part)
        ok, _, error = run_source(source)
        return ok, error
    if kind == "function":
        source = "\n".join(
            part for part in (setup, candidate, _function_assertion(test)) if part
        )
        ok, _, error = run_source(source)
        return ok, error
    if kind != "stdin":
        return False, f"unsupported test kind: {kind}"
    ok, stdout, error = run_source(candidate, str(test.get("input", "")))
    expected = test.get("expected", test.get("output", ""))
    alternatives = (
        expected
        if isinstance(expected, list) and all(isinstance(item, str) for item in expected)
        else [expected]
    )
    matched = ok and any(
        normalize_stdout(stdout) == normalize_stdout(item) for item in alternatives
    )
    message = "" if matched else (
        error or f"expected={normalize_stdout(expected)!r}, got={normalize_stdout(stdout)!r}"
    )
    return matched, message


_CODE_TEST_EXECUTOR: concurrent.futures.ProcessPoolExecutor | None = None
_CODE_TEST_EXECUTOR_WORKERS = 0


def _initialize_code_test_worker() -> None:
    torch.set_num_threads(1)


def _shutdown_code_test_executor() -> None:
    global _CODE_TEST_EXECUTOR, _CODE_TEST_EXECUTOR_WORKERS
    if _CODE_TEST_EXECUTOR is not None:
        _CODE_TEST_EXECUTOR.shutdown(wait=True, cancel_futures=True)
        _CODE_TEST_EXECUTOR = None
        _CODE_TEST_EXECUTOR_WORKERS = 0


atexit.register(_shutdown_code_test_executor)


def _execute_code_test_job(
    job: tuple[str, Mapping[str, Any], str],
) -> tuple[bool, str]:
    candidate, test, setup = job
    return execute_code_test(candidate, test, setup)


def _code_test_executor(worker_count: int) -> concurrent.futures.ProcessPoolExecutor:
    global _CODE_TEST_EXECUTOR, _CODE_TEST_EXECUTOR_WORKERS
    if _CODE_TEST_EXECUTOR is None or _CODE_TEST_EXECUTOR_WORKERS != worker_count:
        if _CODE_TEST_EXECUTOR is not None:
            _CODE_TEST_EXECUTOR.shutdown(wait=True, cancel_futures=True)
        _CODE_TEST_EXECUTOR = concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_code_test_worker,
        )
        _CODE_TEST_EXECUTOR_WORKERS = worker_count
    return _CODE_TEST_EXECUTOR


class SandboxReferenceVerifier:
    def verify(self, task: code_data.CodeTask) -> code_data.VerificationResult:
        outcomes: dict[str, bool] = {}
        errors: dict[str, str] = {}
        for test in task.tests:
            passed, error = execute_code_test(
                task.reference_solution, test, task.test_setup
            )
            test_id = str(test["test_id"])
            outcomes[test_id] = passed
            if error:
                errors[test_id] = error
        return code_data.VerificationResult(outcomes, errors)


def score_code(
    candidate: str,
    example: Mapping[str, Any],
    max_workers: int = 1,
) -> dict[str, Any]:
    source = core.extract_code(candidate)
    tests = list(example.get("tests", []))
    normalized_tests = [
        {"kind": "assert", "code": test} if isinstance(test, str) else test
        for test in tests
    ]
    setup = str(example.get("test_setup", ""))
    jobs = [(source, test, setup) for test in normalized_tests]
    worker_count = max(1, max_workers)
    if worker_count > 1 and len(jobs) > 1:
        outcomes = list(
            _code_test_executor(worker_count).map(_execute_code_test_job, jobs)
        )
    else:
        outcomes = [_execute_code_test_job(job) for job in jobs]
    passed = 0
    errors: list[str] = []
    for ok, error in outcomes:
        passed += int(ok)
        if error and len(errors) < 2:
            errors.append(error)
    syntax_valid = has_valid_python_syntax(source)
    syntax_bonus = 0.05 if syntax_valid else 0.0
    fraction = passed / len(tests) if tests else 0.0
    return {
        "reward": fraction if fraction > 0 else syntax_bonus,
        "passed": passed,
        "total": len(tests),
        "candidate": source,
        "syntax_valid": syntax_valid,
        "errors": errors,
        "source": example.get("source"),
        "task_id": example.get("task_id"),
    }


def _load_source_rows(source: str) -> code_data.SourceRows:
    canonical = code_data.canonical_source(source)
    if canonical == "mbpp":
        rows = load_dataset("google-research-datasets/mbpp", split="train")
        return code_data.SourceRows(canonical, "train", rows, variant="full")
    if canonical == "apps":
        rows = load_dataset(
            "codeparrot/apps", split="train", streaming=True, trust_remote_code=True
        )
        return code_data.SourceRows(canonical, "train", rows)
    if canonical == "codecontests":
        rows = load_dataset("deepmind/code_contests", split="train", streaming=True)
        return code_data.SourceRows(canonical, "train", rows)
    if canonical == "taco":
        rows = load_dataset(
            "BAAI/TACO", split="train", streaming=True, trust_remote_code=True
        )
        return code_data.SourceRows(canonical, "train", rows)
    if canonical == "xcodeeval":
        rows = load_dataset(
            "NTU-NLP-sg/xCodeEval",
            "program-synthesis",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        return code_data.SourceRows(canonical, "program_synthesis_train", rows)
    raise AssertionError(f"unhandled canonical source: {canonical}")


def _task_training_record(task: code_data.CodeTask) -> dict[str, Any]:
    value = task.to_dict()
    value["prompt"] = task.statement
    value["code"] = task.reference_solution
    return value


def _build_requested_code_source(
    source_and_cap: tuple[str, int | None],
) -> code_data.BuildResult:
    """Validate one requested source in an isolated worker process."""

    source, pilot_cap = source_and_cap
    return code_data.build_code_pool(
        [_load_source_rows(source)],
        SandboxReferenceVerifier(),
        pilot_cap=pilot_cap,
    )


def _merge_code_build_results(
    results: Sequence[code_data.BuildResult],
) -> code_data.BuildResult:
    """Merge independently verified sources with global statement deduplication."""

    tasks: list[code_data.CodeTask] = []
    accepted_uids: set[str] = set()
    accepted_statements: set[str] = set()
    source_manifest: dict[str, dict[str, Any]] = {}

    for result in results:
        for source, raw_stats in result.manifest["sources"].items():
            if source in source_manifest:
                raise ValueError(f"source built more than once: {source}")
            source_manifest[source] = {
                **raw_stats,
                "filter_reasons": dict(raw_stats.get("filter_reasons", {})),
            }

        for task in result.tasks:
            stats = source_manifest[task.source]
            if task.uid in accepted_uids:
                stats["duplicate_source_id"] += 1
                stats["accepted"] -= 1
                continue
            if task.normalized_statement in accepted_statements:
                stats["duplicate_statement"] += 1
                stats["accepted"] -= 1
                continue
            accepted_uids.add(task.uid)
            accepted_statements.add(task.normalized_statement)
            tasks.append(task)

    for stats in source_manifest.values():
        stats["deduplicated"] = (
            stats["duplicate_source_id"] + stats["duplicate_statement"]
        )

    available = [stats.get("available_rows") for stats in source_manifest.values()]
    totals = {
        "available_rows": (
            sum(int(value) for value in available)
            if all(value is not None for value in available)
            else None
        ),
        "rows_seen": sum(stats["rows_seen"] for stats in source_manifest.values()),
        "normalized": sum(stats["normalized"] for stats in source_manifest.values()),
        "filtered": sum(stats["filtered"] for stats in source_manifest.values()),
        "deduplicated": sum(
            stats["deduplicated"] for stats in source_manifest.values()
        ),
        "accepted": len(tasks),
    }
    manifest = {
        "schema_version": 1,
        "mode": (
            "full"
            if all(result.manifest["mode"] == "full" for result in results)
            else "pilot"
        ),
        "verification": "reference solution passed every merged test",
        "deduplication": [
            "canonical source task ID",
            "normalized problem statement",
        ],
        "totals": totals,
        "sources": dict(sorted(source_manifest.items())),
    }
    return code_data.BuildResult(tasks=tuple(tasks), manifest=manifest)


def build_training_code_pool(
    config: FullConfig,
    source_rows: Sequence[code_data.SourceRows] | None = None,
    verifier: code_data.VerifierLike | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requested = [item.strip() for item in config.code_sources.split(",") if item.strip()]
    pilot_cap = config.code_pilot_cap if config.code_pilot_cap > 0 else None
    if source_rows is None and verifier is None and len(requested) > 1:
        worker_count = min(4, len(requested))
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
        ) as executor:
            partial_results = list(
                executor.map(
                    _build_requested_code_source,
                    [(source, pilot_cap) for source in requested],
                )
            )
        result = _merge_code_build_results(partial_results)
    else:
        rows = (
            list(source_rows)
            if source_rows is not None
            else [_load_source_rows(source) for source in requested]
        )
        result = code_data.build_code_pool(
            rows, verifier or SandboxReferenceVerifier(), pilot_cap=pilot_cap
        )
    records = [_task_training_record(task) for task in result.tasks]
    random.Random(config.seed + 5).shuffle(records)
    return records, result.manifest


def load_sql_data(config: FullConfig):
    dataset = load_dataset("Salesforce/wikisql", trust_remote_code=True)
    train = dataset["train"]
    validation_name = "validation" if "validation" in dataset else "dev"
    validation = dataset[validation_name]
    train_count = (
        len(train)
        if config.sql_train_limit <= 0
        else min(config.sql_train_limit, len(train))
    )
    anchor_count = min(config.anchor_examples, max(1, train_count // 20))
    indices = list(range(train_count))
    random.Random(config.seed + 31).shuffle(indices)
    sql_anchor = train.select(sorted(indices[:anchor_count]))
    sql_train = train.select(sorted(indices[anchor_count:]))
    eval_count = (
        len(validation)
        if config.sql_eval_limit <= 0
        else min(config.sql_eval_limit, len(validation))
    )
    return sql_train, sql_anchor, validation.select(range(eval_count))


def _supervised_ids(
    tokenizer: Any, prompt: str, target: str, max_length: int
) -> tuple[list[int], list[int]]:
    if max_length < 2:
        raise ValueError("max_length must be at least two")
    prompt_ids = list(format_prompt(tokenizer, prompt))
    target_ids = list(tokenizer.encode(target, add_special_tokens=False))
    eos = tokenizer.eos_token_id
    if eos is not None and (not target_ids or target_ids[-1] != eos):
        target_ids.append(int(eos))
    if not target_ids:
        raise ValueError("supervised target has no tokens")
    target_budget = max_length - 1
    if eos is not None and len(target_ids) > target_budget:
        target_ids = target_ids[: target_budget - 1] + [int(eos)]
    else:
        target_ids = target_ids[:target_budget]
    prompt_ids = prompt_ids[-(max_length - len(target_ids)) :]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    if not any(label != -100 for label in labels):
        raise ValueError("supervised batch row has no active label")
    return input_ids, labels


def encode_supervised_batch(
    tokenizer: Any,
    prompts: Sequence[str],
    targets: Sequence[str],
    max_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not prompts or len(prompts) != len(targets):
        raise ValueError("prompts and targets must have the same non-zero length")
    rows = [
        _supervised_ids(tokenizer, prompt, target, max_length)
        for prompt, target in zip(prompts, targets)
    ]
    width = max(len(input_ids) for input_ids, _ in rows)
    pad = tokenizer.pad_token_id
    input_batch = [ids + [pad] * (width - len(ids)) for ids, _ in rows]
    label_batch = [labels + [-100] * (width - len(labels)) for _, labels in rows]
    attention = [[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in rows]
    return {
        "input_ids": torch.tensor(input_batch, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
        "labels": torch.tensor(label_batch, dtype=torch.long, device=device),
    }


def sql_batch_loss(
    model: core.DomainModel,
    tokenizer: Any,
    examples: Sequence[Mapping[str, Any]],
    config: FullConfig,
) -> torch.Tensor:
    batch = encode_supervised_batch(
        tokenizer,
        [core.sql_prompt(dict(example)) for example in examples],
        [core.gold_wikisql(dict(example)) for example in examples],
        config.max_length,
        next(model.parameters()).device,
    )
    with model.domain("sql"):
        loss = model.policy(**batch).loss
    core.require_finite(loss, "SQL batch loss")
    return loss


def sql_anchor_gradients(
    model: core.DomainModel,
    tokenizer: Any,
    examples: Sequence[Mapping[str, Any]],
    config: FullConfig,
    shared: Sequence[nn.Parameter],
) -> GradientList:
    """Compute one logical anchor-batch gradient using bounded micro-batches."""

    if not examples:
        raise ValueError("SQL anchor gradient requires examples")
    requested = config.anchor_micro_batch_size
    if requested < 0:
        raise ValueError("anchor micro-batch size must be non-negative")
    micro_batch_size = len(examples) if requested == 0 else min(requested, len(examples))
    weighted: GradientList = [None] * len(shared)
    total_tokens = 0
    for offset in range(0, len(examples), micro_batch_size):
        micro_examples = examples[offset : offset + micro_batch_size]
        batch = encode_supervised_batch(
            tokenizer,
            [core.sql_prompt(dict(example)) for example in micro_examples],
            [core.gold_wikisql(dict(example)) for example in micro_examples],
            config.max_length,
            next(model.parameters()).device,
        )
        active_tokens = int((batch["labels"][..., 1:] != -100).sum().item())
        if active_tokens <= 0:
            raise ValueError("SQL anchor micro-batch has no active target tokens")
        with model.domain("sql"):
            loss = model.policy(**batch).loss
        core.require_finite(loss, "SQL anchor micro-batch loss")
        current = gradients(loss, shared)
        for index, value in enumerate(current):
            if value is None:
                continue
            contribution = value.detach() * active_tokens
            weighted[index] = (
                contribution
                if weighted[index] is None
                else weighted[index] + contribution
            )
        total_tokens += active_tokens
    if total_tokens <= 0:
        raise ValueError("SQL anchor logical batch has no active target tokens")
    return [
        None if value is None else value / total_tokens
        for value in weighted
    ]


def backward_sql_sft_group(
    model: core.DomainModel,
    tokenizer: Any,
    examples: Sequence[Mapping[str, Any]],
    config: FullConfig,
) -> tuple[float, int]:
    """Backpropagate one logical SFT batch through exact micro-batches."""

    if not examples:
        raise ValueError("SQL SFT group requires examples")
    if config.sft_batch_size < 1:
        raise ValueError("SFT micro-batch size must be positive")
    encoded: list[tuple[dict[str, torch.Tensor], int]] = []
    total_tokens = 0
    for offset in range(0, len(examples), config.sft_batch_size):
        micro_examples = examples[offset : offset + config.sft_batch_size]
        batch = encode_supervised_batch(
            tokenizer,
            [core.sql_prompt(dict(example)) for example in micro_examples],
            [core.gold_wikisql(dict(example)) for example in micro_examples],
            config.max_length,
            next(model.parameters()).device,
        )
        active_tokens = int((batch["labels"][..., 1:] != -100).sum().item())
        if active_tokens <= 0:
            raise ValueError("SQL SFT micro-batch has no active target tokens")
        encoded.append((batch, active_tokens))
        total_tokens += active_tokens
    if total_tokens <= 0:
        raise ValueError("SQL SFT logical batch has no active target tokens")

    group_loss = 0.0
    for batch, active_tokens in encoded:
        with model.domain("sql"):
            loss = model.policy(**batch).loss
        core.require_finite(loss, "SQL SFT micro-batch loss")
        weight = active_tokens / total_tokens
        (weight * loss).backward()
        group_loss += weight * float(loss.detach())
    return group_loss, total_tokens


def train_common_sql(
    model: core.DomainModel,
    tokenizer: Any,
    sql_train: Sequence[Mapping[str, Any]],
    config: FullConfig,
    trace_path: Path,
) -> None:
    optimizer = core.optimizer_for(model, config.learning_rate)
    rng = random.Random(config.seed + 11)
    model.train()
    global_step = 0
    if config.sft_batch_size < 1:
        raise ValueError("SFT micro-batch size must be positive")
    if config.sft_gradient_accumulation < 1:
        raise ValueError("SFT gradient accumulation must be positive")
    effective_batch = config.sft_batch_size * config.sft_gradient_accumulation
    for epoch in range(config.sql_sft_epochs):
        indices = list(range(len(sql_train)))
        rng.shuffle(indices)
        for offset in range(0, len(indices), effective_batch):
            group = indices[offset : offset + effective_batch]
            optimizer.zero_grad(set_to_none=True)
            group_loss, target_tokens = backward_sql_sft_group(
                model,
                tokenizer,
                [dict(sql_train[index]) for index in group],
                config,
            )
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                config.gradient_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()
            global_step += 1
            append_jsonl(
                trace_path,
                {
                    "phase": "sql_sft",
                    "epoch": epoch,
                    "step": global_step,
                    "loss": group_loss,
                    "micro_batch_size": config.sft_batch_size,
                    "gradient_accumulation": config.sft_gradient_accumulation,
                    "logical_batch_size": len(group),
                    "target_tokens": target_tokens,
                },
            )
            if global_step % 100 == 0 or offset + effective_batch >= len(indices):
                print(
                    f"sql sft epoch={epoch + 1}/{config.sql_sft_epochs} "
                    f"examples={min(offset + effective_batch, len(indices))}/{len(indices)} "
                    f"loss={group_loss:.4f}",
                    flush=True,
                )


GradientList = list[Optional[torch.Tensor]]


def gradients(
    loss: torch.Tensor,
    parameters: Sequence[nn.Parameter],
    retain_graph: bool = False,
) -> GradientList:
    if not parameters:
        return []
    return core.parameter_gradients(loss, parameters, retain_graph=retain_graph)


def gradient_dot(
    left: Sequence[torch.Tensor | None], right: Sequence[torch.Tensor | None]
) -> torch.Tensor:
    values = [
        (first.float() * second.float()).sum()
        for first, second in zip(left, right)
        if first is not None and second is not None
    ]
    if values:
        return torch.stack(values).sum()
    for collection in (left, right):
        for value in collection:
            if value is not None:
                return torch.zeros((), device=value.device)
    return torch.tensor(0.0)


def gradient_norm(values: Sequence[torch.Tensor | None]) -> torch.Tensor:
    squares = [value.float().square().sum() for value in values if value is not None]
    if squares:
        return torch.sqrt(torch.stack(squares).sum())
    return torch.tensor(0.0)


def scale_gradients(
    values: Sequence[torch.Tensor | None], scale: float | torch.Tensor
) -> GradientList:
    return [None if value is None else value * scale for value in values]


def normalize_gradients(values: Sequence[torch.Tensor | None]) -> GradientList:
    norm = gradient_norm(values)
    if float(norm.detach()) == 0.0:
        return [None if value is None else torch.zeros_like(value) for value in values]
    return scale_gradients(values, 1.0 / norm)


def normalized_ema_update(
    previous: Sequence[torch.Tensor | None] | None,
    current: Sequence[torch.Tensor | None],
    beta: float,
) -> GradientList:
    if not 0.0 <= beta < 1.0:
        raise ValueError("EMA beta must be in [0, 1)")
    current_unit = normalize_gradients(current)
    if previous is None:
        return [None if value is None else value.detach().clone() for value in current_unit]
    mixed: GradientList = []
    for old, new in zip(previous, current_unit):
        if old is None:
            mixed.append(None if new is None else (1.0 - beta) * new)
        elif new is None:
            mixed.append(beta * old)
        else:
            mixed.append(beta * old + (1.0 - beta) * new)
    return [
        None if value is None else value.detach()
        for value in normalize_gradients(mixed)
    ]


class SourceRewardNormalizer:
    def __init__(
        self,
        warmup: int = 32,
        beta: float = 0.95,
        std_floor: float = 0.10,
        clip: float = 2.0,
    ) -> None:
        if warmup < 2:
            raise ValueError("source reward warmup must be at least two")
        if not 0.0 <= beta < 1.0:
            raise ValueError("source reward beta must be in [0, 1)")
        if std_floor <= 0.0 or clip <= 0.0:
            raise ValueError("std floor and clipping bound must be positive")
        self.warmup = warmup
        self.beta = beta
        self.std_floor = std_floor
        self.clip = clip
        self.states: dict[str, dict[str, float]] = {}

    def observe(self, source: str, reward: float) -> tuple[float, dict[str, Any]]:
        if not math.isfinite(reward):
            raise FloatingPointError(f"non-finite reward for {source}: {reward}")
        state = self.states.setdefault(
            source,
            {"count": 0.0, "mean": 0.0, "m2": 0.0, "variance": 0.0},
        )
        count = int(state["count"])
        ready = count >= self.warmup
        old_mean = state["mean"]
        std = max(math.sqrt(max(state["variance"], 0.0)), self.std_floor)
        advantage = (
            0.0
            if not ready
            else max(-self.clip, min(self.clip, (reward - old_mean) / std))
        )

        if count < self.warmup:
            new_count = count + 1
            delta = reward - state["mean"]
            state["mean"] += delta / new_count
            state["m2"] += delta * (reward - state["mean"])
            state["count"] = float(new_count)
            if new_count >= 2:
                state["variance"] = state["m2"] / (new_count - 1)
        else:
            new_mean = self.beta * old_mean + (1.0 - self.beta) * reward
            state["variance"] = max(
                0.0,
                self.beta * state["variance"]
                + (1.0 - self.beta) * (reward - old_mean) * (reward - new_mean),
            )
            state["mean"] = new_mean
            state["count"] += 1.0
        return advantage, {
            "source_reward_ready": ready,
            "source_reward_count": count,
            "source_reward_mean": old_mean,
            "source_reward_std": std,
        }


class StableSQLAnchor:
    def __init__(
        self,
        examples: Sequence[Mapping[str, Any]],
        batch_size: int,
        refresh_steps: int,
        beta: float,
        seed: int,
    ) -> None:
        if not examples:
            raise ValueError("stable SQL anchor requires examples")
        if batch_size <= 0 or refresh_steps <= 0:
            raise ValueError("anchor batch size and refresh interval must be positive")
        self.examples = examples
        self.batch_size = batch_size
        self.refresh_steps = refresh_steps
        self.beta = beta
        self.order = list(range(len(examples)))
        random.Random(seed).shuffle(self.order)
        self.cursor = 0
        self.ema: GradientList | None = None
        self.last_refresh = -1

    def _next_batch(self) -> list[dict[str, Any]]:
        indices = []
        for _ in range(self.batch_size):
            indices.append(self.order[self.cursor])
            self.cursor = (self.cursor + 1) % len(self.order)
        return [dict(self.examples[index]) for index in indices]

    def get(
        self,
        step: int,
        model: core.DomainModel,
        tokenizer: Any,
        shared: Sequence[nn.Parameter],
        config: FullConfig,
    ) -> tuple[GradientList, bool]:
        refresh = self.ema is None or step % self.refresh_steps == 0
        if refresh:
            with core.evaluation_policy(model):
                current = sql_anchor_gradients(
                    model,
                    tokenizer,
                    self._next_batch(),
                    config,
                    shared,
                )
            self.ema = normalized_ema_update(self.ema, current, self.beta)
            self.last_refresh = step
        assert self.ema is not None
        return [
            None if value is None else value.detach().clone() for value in self.ema
        ], refresh


def branch_spec(branch: str) -> BranchSpec:
    try:
        return BRANCH_SPECS[branch]
    except KeyError as error:
        raise ValueError(f"unknown branch {branch!r}") from error


def combine_branch_gradients(
    sql_gradients: Sequence[torch.Tensor | None],
    code_gradients: Sequence[torch.Tensor | None],
    spec: BranchSpec,
    config: FullConfig,
    anchor_gradients: Sequence[torch.Tensor | None] | None = None,
) -> tuple[GradientList, dict[str, float]]:
    epsilon = 1.0e-12
    sql_norm = gradient_norm(sql_gradients)
    code_norm = gradient_norm(code_gradients)
    raw_dot = gradient_dot(code_gradients, sql_gradients)
    projected = list(code_gradients)
    projection_coefficient = torch.zeros_like(raw_dot)
    if spec.project_conflicts and float(raw_dot.detach()) < 0.0:
        projection_coefficient = raw_dot / (sql_norm.square() + epsilon)
        projected = [
            None
            if code is None
            else (
                code
                if sql is None
                else code - projection_coefficient.to(code.dtype) * sql
            )
            for code, sql in zip(code_gradients, sql_gradients)
        ]
    projected_norm = gradient_norm(projected)
    norm_scale = 1.0
    if spec.norm_match:
        norm_scale = min(
            config.max_aux_scale,
            float((sql_norm / (projected_norm + epsilon)).detach()),
        )
    alpha = 1.0
    anchor_cosine = 0.0
    if spec.anchor_gate:
        if anchor_gradients is None:
            raise ValueError("target-aligned branch requires stable anchor gradients")
        anchor_norm = gradient_norm(anchor_gradients)
        cosine = gradient_dot(projected, anchor_gradients) / (
            projected_norm * anchor_norm + epsilon
        )
        anchor_cosine = float(cosine.detach())
        alpha = min(
            1.0,
            max(0.0, anchor_cosine) / max(config.alignment_temperature, epsilon),
        )
    coefficient = config.auxiliary_weight * norm_scale * alpha
    if not spec.uses_code or coefficient == 0.0:
        combined = list(sql_gradients)
    else:
        combined = []
        for sql, code in zip(sql_gradients, projected):
            if sql is None:
                combined.append(None if code is None else coefficient * code)
            elif code is None:
                combined.append(sql)
            else:
                combined.append(sql + coefficient * code)
    return combined, {
        "raw_sql_code_dot": float(raw_dot.detach()),
        "projection_coefficient": float(projection_coefficient.detach()),
        "projected_anchor_cosine": anchor_cosine,
        "alpha": alpha,
        "norm_scale": norm_scale,
        "auxiliary_coefficient": coefficient,
        "sql_gradient_norm": float(sql_norm.detach()),
        "code_gradient_norm": float(code_norm.detach()),
        "projected_code_gradient_norm": float(projected_norm.detach()),
    }


def apply_gradient_groups(
    primary_parameters: Sequence[nn.Parameter],
    primary_gradients: Sequence[torch.Tensor | None],
    code_private_parameters: Sequence[nn.Parameter],
    code_private_gradients: Sequence[torch.Tensor | None],
    max_norm: float,
) -> dict[str, float]:
    if len(primary_parameters) != len(primary_gradients):
        raise ValueError("primary parameter/gradient length mismatch")
    if len(code_private_parameters) != len(code_private_gradients):
        raise ValueError("code-private parameter/gradient length mismatch")
    # ``clip_grad_norm_`` mutates ``parameter.grad`` in place. Clone incoming
    # tensors so a caller can safely reuse an unmodified reference update in an
    # invariant test or another branch.
    primary_copies = [
        None if gradient is None else gradient.detach().clone()
        for gradient in primary_gradients
    ]
    code_copies = [
        None if gradient is None else gradient.detach().clone()
        for gradient in code_private_gradients
    ]
    core.assign_gradients(primary_parameters, primary_copies)
    core.assign_gradients(code_private_parameters, code_copies)
    primary_norm = (
        torch.nn.utils.clip_grad_norm_(
            primary_parameters, max_norm, error_if_nonfinite=True
        )
        if primary_parameters
        else torch.tensor(0.0)
    )
    code_norm = (
        torch.nn.utils.clip_grad_norm_(
            code_private_parameters, max_norm, error_if_nonfinite=True
        )
        if code_private_parameters
        else torch.tensor(0.0)
    )
    return {
        "primary_preclip_norm": float(primary_norm.detach()),
        "code_private_preclip_norm": float(code_norm.detach()),
    }


def code_policy_sample(
    model: core.DomainModel,
    tokenizer: Any,
    example: Mapping[str, Any],
    config: FullConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    prompt = code_prompt(example)
    text, sequence, prompt_length = core.generate_candidate(
        model,
        tokenizer,
        "code",
        prompt,
        config.code_new_tokens,
        config,
        sample=True,
    )
    score = score_code(text, example, max_workers=config.code_test_workers)
    log_probability = core.sampled_log_probability(
        model,
        "code",
        sequence,
        prompt_length,
        temperature=config.temperature,
        top_p=config.top_p,
    )
    return log_probability, {
        **score,
        "domain": "code",
        "raw_completion": text,
        "completion_tokens": int(sequence.shape[1] - prompt_length),
        "termination_reason": (
            "eos"
            if tokenizer.eos_token_id is not None
            and int(sequence[0, -1]) == int(tokenizer.eos_token_id)
            else "max_new_tokens"
        ),
        "log_probability": float(log_probability.detach()),
    }


def separated_code_gradients(
    verifier_rl_loss: torch.Tensor,
    code_sft_loss: torch.Tensor | None,
    shared_parameters: Sequence[nn.Parameter],
    code_private_parameters: Sequence[nn.Parameter],
    code_sft_weight: float,
) -> tuple[GradientList, GradientList]:
    """Keep verifier RL on shared parameters and code SFT on private parameters."""

    shared_gradients = gradients(verifier_rl_loss, shared_parameters)
    if not code_private_parameters:
        return shared_gradients, []
    if code_sft_loss is None:
        raise ValueError("code-private parameters require a code SFT loss")
    private_gradients = gradients(
        code_sft_weight * code_sft_loss,
        code_private_parameters,
    )
    return shared_gradients, private_gradients


def train_branch(
    branch: str,
    model: core.DomainModel,
    tokenizer: Any,
    sql_train: Sequence[Mapping[str, Any]],
    sql_anchor: Sequence[Mapping[str, Any]],
    code_pool: Sequence[Mapping[str, Any]],
    config: FullConfig,
    trace_path: Path,
    checkpoint_updates: Sequence[int] = (),
    checkpoint_callback: Optional[
        Callable[[int, core.DomainModel, Any], None]
    ] = None,
) -> None:
    spec = branch_spec(branch)
    if spec.uses_code and not code_pool:
        raise ValueError(f"branch {branch} requires a non-empty code pool")
    if not math.isfinite(config.reference_kl_weight) or config.reference_kl_weight < 0.0:
        raise ValueError("reference_kl_weight must be finite and non-negative")
    if config.anchor_micro_batch_size < 0:
        raise ValueError("anchor_micro_batch_size must be non-negative")
    if config.reference_kl_weight > 0.0 and branch != "target_aligned":
        raise ValueError("reference-policy KL is registered only for an M5 candidate")
    optimizer = core.optimizer_for(model, config.learning_rate)
    shared = model.shared_parameters()
    sql_private = model.private_parameters("sql")
    code_private = model.private_parameters("code")
    primary_parameters = [*shared, *sql_private]
    reference_parameters = (
        core.snapshot_trainable_parameters(model)
        if config.reference_kl_weight > 0.0
        else None
    )
    steps = config.rl_steps if config.rl_steps > 0 else len(code_pool)
    if steps <= 0:
        raise ValueError("RL steps must be positive")
    checkpoint_set = {int(update) for update in checkpoint_updates}
    if checkpoint_set and checkpoint_callback is None:
        raise ValueError("checkpoint updates require a checkpoint callback")
    if any(update < 0 or update > steps for update in checkpoint_set):
        raise ValueError("checkpoint update is outside the training range")
    rng = np.random.default_rng(config.seed + 101)
    sql_indices = rng.integers(0, len(sql_train), size=steps)
    code_indices = np.arange(len(code_pool)) if code_pool else np.zeros(steps, dtype=int)
    rng.shuffle(code_indices)
    code_indices = np.resize(code_indices, steps)
    sql_baseline = 0.0
    code_baseline = 0.0
    source_normalizer = SourceRewardNormalizer(
        config.source_reward_warmup,
        config.source_reward_beta,
        config.source_reward_std_floor,
        config.source_reward_clip,
    )
    anchor = (
        StableSQLAnchor(
            sql_anchor,
            config.anchor_batch_size,
            config.anchor_refresh_steps,
            config.anchor_ema_beta,
            config.seed + 211,
        )
        if spec.anchor_gate
        else None
    )
    model.train()
    if 0 in checkpoint_set:
        assert checkpoint_callback is not None
        checkpoint_callback(0, model, tokenizer)
        model.train()

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        sql_example = dict(sql_train[int(sql_indices[step])])
        torch.manual_seed(config.seed + 10_000 + step)
        torch.cuda.manual_seed_all(config.seed + 10_000 + step)
        sql_rollout_kwargs = (
            {"reference_parameters": reference_parameters}
            if reference_parameters is not None
            else {}
        )
        sql_loss, sql_record = core.policy_rollout_loss(
            model,
            tokenizer,
            "sql",
            sql_example,
            sql_baseline,
            config,
            **sql_rollout_kwargs,
        )
        sql_baseline = 0.9 * sql_baseline + 0.1 * float(sql_record["reward"])
        sql_shared = gradients(sql_loss, shared, retain_graph=True)
        sql_private_gradients = gradients(sql_loss, sql_private)
        combined = list(sql_shared)
        code_private_gradients: GradientList = [None] * len(code_private)
        code_record: dict[str, Any] = {
            "reward": None,
            "source": None,
            "task_id": None,
        }
        normalizer_record: dict[str, Any] = {}
        anchor_refreshed = False
        diagnostics = {
            "raw_sql_code_dot": 0.0,
            "projection_coefficient": 0.0,
            "projected_anchor_cosine": 0.0,
            "alpha": 0.0,
            "norm_scale": 0.0,
            "auxiliary_coefficient": 0.0,
            "sql_gradient_norm": float(gradient_norm(sql_shared).detach()),
            "code_gradient_norm": 0.0,
            "projected_code_gradient_norm": 0.0,
        }

        if spec.uses_code:
            code_example = dict(code_pool[int(code_indices[step])])
            code_record.update(
                {
                    "source": code_example.get("source"),
                    "task_id": code_example.get("task_id"),
                }
            )
            code_sft_value: float | None = None
            code_sft: torch.Tensor | None = None
            if code_private:
                code_sft = core.sft_loss(
                    model,
                    tokenizer,
                    "code",
                    code_prompt(code_example),
                    str(code_example["code"]),
                    config,
                )
                code_sft_value = float(code_sft.detach())
                code_private_gradients = gradients(
                    config.code_sft_weight * code_sft,
                    code_private,
                )
            if spec.uses_code_reward:
                torch.manual_seed(config.seed + 20_000 + step)
                torch.cuda.manual_seed_all(config.seed + 20_000 + step)
                log_probability, sampled_record = code_policy_sample(
                    model, tokenizer, code_example, config
                )
                code_record.update(sampled_record)
                reward = float(code_record["reward"])
                if spec.source_normalized:
                    advantage, normalizer_record = source_normalizer.observe(
                        str(code_record["source"]), reward
                    )
                else:
                    advantage = reward - code_baseline
                    code_baseline = 0.9 * code_baseline + 0.1 * reward
                if not math.isfinite(advantage):
                    raise FloatingPointError(f"non-finite code advantage: {advantage}")
                code_rl_loss = -advantage * log_probability
                core.require_finite(code_rl_loss, "code verifier RL loss")
                code_shared = gradients(code_rl_loss, shared)

                anchor_gradients = None
                if anchor is not None:
                    anchor_gradients, anchor_refreshed = anchor.get(
                        step, model, tokenizer, shared, config
                    )
                combined, diagnostics = combine_branch_gradients(
                    sql_shared, code_shared, spec, config, anchor_gradients
                )
                code_record.update(
                    {
                        "advantage": advantage,
                        "reinforcement_loss": float(code_rl_loss.detach()),
                    }
                )
            code_record["sft_loss"] = code_sft_value

        clipping = apply_gradient_groups(
            primary_parameters,
            [*combined, *sql_private_gradients],
            code_private,
            code_private_gradients,
            config.gradient_clip,
        )
        optimizer.step()
        record = {
            "phase": "rl",
            "branch": branch,
            "step": step,
            "sql_reward": float(sql_record["reward"]),
            "sql_denotation": bool(sql_record.get("denotation", False)),
            "sql_score_error": str(sql_record.get("error", "")),
            "sql_score_timeout": sql_record.get("error") == "score_timeout",
            "sql_reference_kl": sql_record.get("reference_kl", 0.0),
            "sql_reference_kl_loss": sql_record.get("reference_kl_loss", 0.0),
            "code_reward": code_record.get("reward"),
            "code_source": code_record.get("source"),
            "code_task_id": code_record.get("task_id"),
            "code_advantage": code_record.get("advantage"),
            "code_reinforcement_loss": code_record.get("reinforcement_loss"),
            "code_sft_loss": code_record.get("sft_loss"),
            "code_raw_completion": code_record.get("raw_completion"),
            "code_candidate": code_record.get("candidate"),
            "code_syntax_valid": code_record.get("syntax_valid"),
            "code_tests_passed": code_record.get("passed"),
            "code_tests_total": code_record.get("total"),
            "code_errors": code_record.get("errors"),
            "code_completion_tokens": code_record.get("completion_tokens"),
            "code_termination_reason": code_record.get("termination_reason"),
            "sql_baseline": sql_baseline,
            "code_baseline": code_baseline,
            "anchor_refreshed": anchor_refreshed,
            "anchor_batch_size": config.anchor_batch_size,
            "anchor_micro_batch_size": (
                config.anchor_batch_size
                if config.anchor_micro_batch_size == 0
                else min(config.anchor_micro_batch_size, config.anchor_batch_size)
            ),
            **normalizer_record,
            **diagnostics,
            **clipping,
        }
        if torch.cuda.is_available():
            cuda_free_bytes, cuda_total_bytes = torch.cuda.mem_get_info()
            record.update(
                {
                    "cuda_free_bytes": int(cuda_free_bytes),
                    "cuda_total_bytes": int(cuda_total_bytes),
                }
            )
        append_jsonl(trace_path, record)
        completed_updates = step + 1
        if completed_updates in checkpoint_set:
            assert checkpoint_callback is not None
            checkpoint_callback(completed_updates, model, tokenizer)
            model.train()
        if (step + 1) % 25 == 0 or step + 1 == steps:
            print(
                f"{branch} {step + 1}/{steps} sql_r={record['sql_reward']:.2f} "
                f"code_r={record['code_reward']} alpha={record['alpha']:.3f}",
                flush=True,
            )


@torch.no_grad()
def evaluate_full_wikisql(
    branch: str,
    model: core.DomainModel,
    tokenizer: Any,
    examples: Sequence[Mapping[str, Any]],
    config: FullConfig,
    predictions_path: Path,
) -> dict[str, Any]:
    totals = {"exact": 0, "executable": 0, "denotation": 0, "reward": 0.0}
    started = time.time()
    model.eval()
    device = next(model.parameters()).device
    for offset in range(0, len(examples), config.eval_batch_size):
        batch_examples = [
            dict(examples[index])
            for index in range(offset, min(offset + config.eval_batch_size, len(examples)))
        ]
        prompt_rows = [
            format_prompt(tokenizer, core.sql_prompt(example))
            for example in batch_examples
        ]
        prompt_limit = max(64, config.max_length - config.sql_new_tokens)
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
            score = core.score_sql(text, example)
            for key in totals:
                totals[key] += float(score[key])
            append_jsonl(
                predictions_path,
                {
                    "branch": branch,
                    "index": offset + local_index,
                    "question": example["question"],
                    **score,
                },
            )
        completed = min(offset + len(batch_examples), len(examples))
        if completed % 256 < config.eval_batch_size or completed == len(examples):
            print(f"eval {branch} {completed}/{len(examples)}", flush=True)
    count = len(examples)
    return {
        "name": branch,
        "examples": count,
        "exact_match": totals["exact"] / count,
        "execution_rate": totals["executable"] / count,
        "denotation_accuracy": totals["denotation"] / count,
        "mean_reward": totals["reward"] / count,
        "seconds": time.time() - started,
    }


def tokenizer_for(config: FullConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model,
        revision=config.model_revision or None,
        cache_dir=config.model_cache_dir or None,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def save_code_pool(path: Path, code_pool: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for task in code_pool:
            handle.write(json.dumps(task) + "\n")


def load_saved_code_pool(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def save_data_manifest(
    experiment_dir: Path,
    sql_train: Sequence[Mapping[str, Any]],
    sql_anchor: Sequence[Mapping[str, Any]],
    sql_eval: Sequence[Mapping[str, Any]],
    code_manifest: Mapping[str, Any],
) -> None:
    write_json(
        experiment_dir / "data_manifest.json",
        {
            "sql_dataset": "Salesforce/wikisql",
            "sql_train_examples": len(sql_train),
            "sql_anchor_examples": len(sql_anchor),
            "sql_eval_examples": len(sql_eval),
            "code": code_manifest,
        },
    )


def prepare(config: FullConfig) -> None:
    experiment_dir = Path(config.experiment_dir)
    if experiment_dir.exists() and any(experiment_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty experiment directory: {experiment_dir}"
        )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    write_json(experiment_dir / "config.json", asdict(config))
    core.seed_everything(config.seed)
    tokenizer = tokenizer_for(config)
    sql_train, sql_anchor, sql_eval = load_sql_data(config)
    # Code-reference validation is CPU-heavy and independent of SQL SFT. Build
    # it concurrently so preparation does not leave the GPU idle.
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="code-pool"
    ) as executor:
        code_future = executor.submit(build_training_code_pool, config)
        model = core.build_model(config, tokenizer)
        write_json(
            experiment_dir / "model_manifest.json",
            {
                "model": config.model,
                "model_revision": config.model_revision,
                "model_cache_dir": config.model_cache_dir,
                "adapter_type": config.adapter_type,
                "adapter_dim": config.adapter_dim,
                "adapter_top_k": config.adapter_top_k,
                "lora_rank": config.lora_rank,
                "private_parameters": model.private_adapters.parameter_counts(),
            },
        )
        train_common_sql(
            model, tokenizer, sql_train, config, experiment_dir / "prepare_trace.jsonl"
        )
        release_cuda_memory()
        atomic_save_trainable(model, experiment_dir / "common_sql_trainable.pt")
        del model
        release_cuda_memory()
        code_pool, code_manifest = code_future.result()

    if not code_pool:
        raise RuntimeError("verified code pool is empty")
    save_code_pool(experiment_dir / "code_pool.jsonl", code_pool)
    save_data_manifest(
        experiment_dir, sql_train, sql_anchor, sql_eval, code_manifest
    )
    print(
        json.dumps(
            {"prepared": str(experiment_dir), "code_examples": len(code_pool)}, indent=2
        ),
        flush=True,
    )


def prepare_checkpoint(config: FullConfig) -> None:
    """Train only a common SQL-SFT checkpoint for an adapter control."""

    experiment_dir = Path(config.experiment_dir)
    if experiment_dir.exists() and any(experiment_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty experiment directory: {experiment_dir}"
        )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    write_json(experiment_dir / "config.json", asdict(config))
    core.seed_everything(config.seed)
    tokenizer = tokenizer_for(config)
    sql_train, sql_anchor, sql_eval = load_sql_data(config)
    model = core.build_model(config, tokenizer)
    write_json(
        experiment_dir / "model_manifest.json",
        {
            "model": config.model,
            "model_revision": config.model_revision,
            "model_cache_dir": config.model_cache_dir,
            "adapter_type": config.adapter_type,
            "adapter_dim": config.adapter_dim,
            "adapter_top_k": config.adapter_top_k,
            "lora_rank": config.lora_rank,
            "private_parameters": model.private_adapters.parameter_counts(),
        },
    )
    train_common_sql(
        model, tokenizer, sql_train, config, experiment_dir / "prepare_trace.jsonl"
    )
    release_cuda_memory()
    atomic_save_trainable(model, experiment_dir / "common_sql_trainable.pt")
    del model
    release_cuda_memory()
    save_data_manifest(
        experiment_dir,
        sql_train,
        sql_anchor,
        sql_eval,
        {"status": "not_built", "reason": "adapter checkpoint preparation only"},
    )
    print(json.dumps({"prepared_checkpoint": str(experiment_dir)}, indent=2), flush=True)


def finish_prepare(config: FullConfig) -> None:
    """Resume only code-pool preparation after SQL SFT is checkpointed."""

    experiment_dir = Path(config.experiment_dir)
    checkpoint = experiment_dir / "common_sql_trainable.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing SQL-SFT checkpoint: {checkpoint}")
    if (experiment_dir / "code_pool.jsonl").exists():
        raise FileExistsError("refusing to overwrite an existing verified code pool")

    core.seed_everything(config.seed)
    sql_train, sql_anchor, sql_eval = load_sql_data(config)
    code_pool, code_manifest = build_training_code_pool(config)
    if not code_pool:
        raise RuntimeError("verified code pool is empty")
    save_code_pool(experiment_dir / "code_pool.jsonl", code_pool)
    save_data_manifest(
        experiment_dir, sql_train, sql_anchor, sql_eval, code_manifest
    )
    print(
        json.dumps(
            {"finished_prepare": str(experiment_dir), "code_examples": len(code_pool)},
            indent=2,
        ),
        flush=True,
    )


def evaluate_checkpoint(config: FullConfig) -> None:
    """Evaluate the common SQL-SFT checkpoint while code validation runs."""

    experiment_dir = Path(config.experiment_dir)
    output_dir = experiment_dir / "checkpoint_eval"
    output_dir.mkdir(parents=True, exist_ok=False)
    core.seed_everything(config.seed + 89)
    tokenizer = tokenizer_for(config)
    _, _, sql_eval = load_sql_data(config)
    model = core.restore_model(
        config, tokenizer, experiment_dir / "common_sql_trainable.pt"
    )
    summary = evaluate_full_wikisql(
        "checkpoint_eval",
        model,
        tokenizer,
        sql_eval,
        config,
        output_dir / "predictions.jsonl",
    )
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def branch(config: FullConfig, branch_name: str) -> None:
    experiment_dir = Path(config.experiment_dir)
    branch_dir = experiment_dir / branch_name
    branch_dir.mkdir(parents=True, exist_ok=False)
    write_json(branch_dir / "config.json", {**asdict(config), "branch": branch_name})
    core.seed_everything(config.seed + 101)
    tokenizer = tokenizer_for(config)
    sql_train, sql_anchor, sql_eval = load_sql_data(config)
    code_pool = load_saved_code_pool(experiment_dir / "code_pool.jsonl")
    model = core.restore_model(
        config, tokenizer, experiment_dir / "common_sql_trainable.pt"
    )
    train_branch(
        branch_name,
        model,
        tokenizer,
        sql_train,
        sql_anchor,
        code_pool,
        config,
        branch_dir / "trace.jsonl",
    )
    release_cuda_memory()
    atomic_save_trainable(model, branch_dir / "trainable.pt")
    release_cuda_memory()
    summary = evaluate_full_wikisql(
        branch_name,
        model,
        tokenizer,
        sql_eval,
        config,
        branch_dir / "predictions.jsonl",
    )
    write_json(branch_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "prepare",
            "prepare_checkpoint",
            "finish_prepare",
            "evaluate_checkpoint",
            "branch",
        ),
    )
    parser.add_argument("--branch", choices=BRANCH_NAMES)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--model-cache-dir", default="")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--adapter-kind",
        "--adapter-type",
        dest="adapter_type",
        choices=("none", "output", "top_k", "layerwise"),
        default="output",
    )
    parser.add_argument("--adapter-dim", type=int, default=96)
    parser.add_argument("--adapter-top-k", type=int, default=6)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--sql-new-tokens", type=int, default=48)
    parser.add_argument("--code-new-tokens", type=int, default=128)
    parser.add_argument(
        "--code-test-workers",
        type=int,
        default=8,
        help="parallel sandbox workers used to score independent code tests",
    )
    parser.add_argument("--sql-sft-weight", type=float, default=0.20)
    parser.add_argument("--code-sft-weight", type=float, default=0.10)
    parser.add_argument("--auxiliary-weight", type=float, default=0.30)
    parser.add_argument("--max-aux-scale", type=float, default=2.0)
    parser.add_argument("--alignment-temperature", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--reference-kl-weight", type=float, default=0.0)
    parser.add_argument("--kl-token-chunk-size", type=int, default=8)
    parser.add_argument("--sql-sft-epochs", type=int, default=1)
    parser.add_argument("--sft-batch-size", type=int, default=16)
    parser.add_argument("--sft-gradient-accumulation", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--sql-train-limit", type=int, default=0)
    parser.add_argument("--sql-eval-limit", type=int, default=0)
    parser.add_argument("--anchor-examples", type=int, default=256)
    parser.add_argument("--anchor-batch-size", type=int, default=16)
    parser.add_argument(
        "--anchor-micro-batch-size",
        type=int,
        default=0,
        help="execution-only anchor micro-batch; 0 uses the logical batch size",
    )
    parser.add_argument("--anchor-refresh-steps", type=int, default=8)
    parser.add_argument("--anchor-ema-beta", type=float, default=0.90)
    parser.add_argument("--source-reward-warmup", type=int, default=32)
    parser.add_argument("--source-reward-beta", type=float, default=0.95)
    parser.add_argument("--source-reward-std-floor", type=float, default=0.10)
    parser.add_argument("--source-reward-clip", type=float, default=2.0)
    parser.add_argument("--code-sources", default="mbpp,apps,codecontests,taco")
    parser.add_argument(
        "--code-pilot-cap",
        type=int,
        default=0,
        help="accepted tasks per source; 0 consumes the full training split",
    )
    parser.add_argument("--rl-steps", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _config_from_values(values: Mapping[str, Any]) -> FullConfig:
    names = {item.name for item in fields(FullConfig)}
    return FullConfig(**{key: value for key, value in values.items() if key in names})


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[str, str | None, FullConfig]:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.stage not in ("prepare", "prepare_checkpoint"):
        if args.stage == "branch" and not args.branch:
            parser.error("--branch is required for --stage branch")
        saved = json.loads((Path(args.experiment_dir) / "config.json").read_text())
        config = _config_from_values(saved)
    else:
        config = _config_from_values(vars(args))
        if config.top_p != 1.0:
            parser.error(
                "--top-p must be 1.0 for on-policy RL; cached sampling and "
                "full-sequence likelihood recomputation can otherwise disagree "
                "at the nucleus boundary"
            )
        if config.code_pilot_cap < 0:
            parser.error("--code-pilot-cap must be non-negative")
        if not math.isfinite(config.reference_kl_weight) or config.reference_kl_weight < 0.0:
            parser.error("--reference-kl-weight must be finite and non-negative")
        if config.kl_token_chunk_size < 1:
            parser.error("--kl-token-chunk-size must be positive")
        if config.sft_gradient_accumulation < 1:
            parser.error("--sft-gradient-accumulation must be positive")
        if config.anchor_micro_batch_size < 0:
            parser.error("--anchor-micro-batch-size must be non-negative")
        if config.smoke:
            config.sql_train_limit = 64
            config.sql_eval_limit = 32
            config.anchor_examples = 16
            config.anchor_batch_size = 4
            config.source_reward_warmup = 2
            config.code_pilot_cap = 4
            config.rl_steps = 8
            config.sft_batch_size = 4
            config.eval_batch_size = 8
    return args.stage, args.branch, config


def main() -> None:
    stage, branch_name, config = parse_args()
    core.format_prompt = format_prompt
    core.code_prompt = code_prompt
    core.score_code = score_code
    if stage == "prepare":
        prepare(config)
    elif stage == "prepare_checkpoint":
        prepare_checkpoint(config)
    elif stage == "finish_prepare":
        finish_prepare(config)
    elif stage == "evaluate_checkpoint":
        evaluate_checkpoint(config)
    else:
        assert branch_name is not None
        branch(config, branch_name)


if __name__ == "__main__":
    main()
