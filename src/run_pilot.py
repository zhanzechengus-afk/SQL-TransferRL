#!/usr/bin/env python3
"""Core SQL-TransferRL model, rewards, and optimization utilities."""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import math
import os
import random
import re
import resource
import sqlite3
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import sqlglot
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from private_adapters import DomainPrivateAdapters


AGG_OPS = ("", "MAX", "MIN", "COUNT", "SUM", "AVG")
COND_OPS = ("=", ">", "<", "!=")
UNSAFE_CODE = re.compile(
    r"\b(?:os|sys|subprocess|socket|shutil|pathlib|requests|urllib)\b|"
    r"\b(?:open|eval|exec|compile|__import__)\s*\(",
    flags=re.IGNORECASE,
)


@dataclass
class RunConfig:
    model: str
    model_revision: str = ""
    model_cache_dir: str = ""
    seed: int = 13
    warmup_steps: int = 48
    rl_steps: int = 32
    eval_examples: int = 48
    train_examples: int = 384
    learning_rate: float = 1.0e-4
    adapter_dim: int = 128
    adapter_type: str = "output"
    adapter_top_k: int = 6
    lora_rank: int = 8
    max_length: int = 512
    sql_new_tokens: int = 96
    code_new_tokens: int = 160
    sql_sft_weight: float = 0.20
    code_sft_weight: float = 0.10
    auxiliary_weight: float = 0.70
    max_aux_scale: float = 2.0
    alignment_temperature: float = 0.02
    curriculum_probe_examples: int = 96
    curriculum_size: int = 32
    temperature: float = 0.8
    top_p: float = 1.0
    rl_loss: str = "ema_reinforce"
    group_size: int = 2
    reference_kl_weight: float = 0.0
    kl_token_chunk_size: int = 8
    output_dir: str = "results"
    smoke: bool = False


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=json_default) + "\n")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, default=json_default) + "\n")


def sql_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    text = str(value)
    try:
        number = float(text.replace(",", ""))
        if math.isfinite(number):
            return str(number)
    except ValueError:
        pass
    return "'" + text.replace("'", "''") + "'"


def gold_wikisql(example: dict[str, Any]) -> str:
    spec = example["sql"]
    select_column = int(spec["sel"])
    aggregate = AGG_OPS[int(spec["agg"])]
    select_expr = f'"c{select_column}"'
    if aggregate:
        select_expr = f"{aggregate}({select_expr})"
    conditions = []
    for column, operator, value in zip(
        spec.get("conds", {}).get("column_index", []),
        spec.get("conds", {}).get("operator_index", []),
        spec.get("conds", {}).get("condition", []),
    ):
        op = COND_OPS[int(operator)] if int(operator) < len(COND_OPS) else "="
        conditions.append(f'"c{int(column)}" {op} {sql_literal(value)}')
    where = " WHERE " + " AND ".join(conditions) if conditions else ""
    return f"SELECT {select_expr} FROM data{where};"


def schema_text(example: dict[str, Any]) -> str:
    table = example["table"]
    columns = []
    for index, (header, kind) in enumerate(zip(table["header"], table["types"])):
        columns.append(f"c{index} ({kind}; {header})")
    return ", ".join(columns)


def sql_prompt(example: dict[str, Any]) -> str:
    return (
        "Write one SQLite query for the question. Use table data and the c0, c1, ... "
        "column identifiers shown below. Return SQL only.\n"
        f"Columns: {schema_text(example)}\n"
        f"Question: {example['question']}"
    )


def code_prompt(example: dict[str, Any]) -> str:
    return (
        "Solve the following Python programming task. Return only the function "
        "implementation, without Markdown or explanation.\n"
        f"Task: {example['prompt']}"
    )


def code_category(example: dict[str, Any]) -> str:
    text = example["prompt"].lower()
    rules = (
        ("sorting/top-k", ("sort", "largest", "smallest", "maximum", "minimum", "top ")),
        ("set/counting", ("count", "frequency", "unique", "duplicate", "set ", "distinct")),
        ("graph/search", ("graph", "path", "tree", "search", "matrix", "grid")),
        ("dynamic programming", ("subsequence", "subarray", "minimum cost", "maximum sum", "ways")),
        ("string/sequence", ("string", "character", "substring", "word", "palindrome")),
    )
    for category, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return category
    return "arithmetic/other"


def extract_sql(text: str) -> str:
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1)
    match = re.search(r"\bSELECT\b", text, flags=re.IGNORECASE)
    if match:
        text = text[match.start() :]
    if ";" in text:
        text = text.split(";", 1)[0] + ";"
    return text.strip()


def extract_code(text: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1)
    # SmolLM chat templates can expose an empty or populated reasoning block in
    # decoded text even when /no_think is requested. Remove only complete,
    # explicitly tagged control blocks; ordinary Python comments and strings
    # remain untouched.
    while control := re.match(
        r"^\s*<(think|analysis)>.*?</\1>\s*",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        text = text[control.end() :]
    text = text.strip()
    text = re.sub(r"^(?:python|py)\s*\n", "", text, count=1, flags=re.IGNORECASE)
    return text.strip()


def normalize_sql(text: str) -> str:
    return re.sub(r"\s+", " ", extract_sql(text).strip().rstrip(";")).lower()


def create_wikisql_db(example: dict[str, Any]) -> sqlite3.Connection:
    table = example["table"]
    connection = sqlite3.connect(":memory:")
    types = ["REAL" if kind == "real" else "TEXT" for kind in table["types"]]
    columns = ", ".join(f'"c{i}" {kind}' for i, kind in enumerate(types))
    connection.execute(f"CREATE TABLE data ({columns})")
    placeholders = ", ".join("?" for _ in types)
    connection.executemany(f"INSERT INTO data VALUES ({placeholders})", table["rows"])
    return connection


def canonical_rows(rows: Sequence[Sequence[Any]]) -> list[tuple[str, ...]]:
    normalized = [tuple(str(value).strip().lower() for value in row) for row in rows]
    return sorted(normalized)


def score_sql(candidate: str, example: dict[str, Any]) -> dict[str, Any]:
    candidate = extract_sql(candidate)
    gold = gold_wikisql(example)
    exact = normalize_sql(candidate) == normalize_sql(gold)
    parsed = False
    executable = False
    denotation = False
    error = ""
    try:
        sqlglot.parse_one(candidate, read="sqlite")
        parsed = True
        connection = create_wikisql_db(example)
        try:
            gold_rows = connection.execute(gold).fetchall()
            candidate_rows = connection.execute(candidate).fetchall()
            executable = True
            denotation = canonical_rows(candidate_rows) == canonical_rows(gold_rows)
        finally:
            connection.close()
    except Exception as exc:  # SQLite exposes the verifier error for auditing.
        error = f"{type(exc).__name__}: {exc}"[:240]
    reward = 1.0 if denotation else 0.20 if executable else 0.05 if parsed else 0.0
    return {
        "reward": reward,
        "exact": exact,
        "executable": executable,
        "denotation": denotation,
        "candidate": candidate,
        "gold": gold,
        "error": error,
    }


def limited_process() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (1_000_000_000, 1_000_000_000))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1_000_000, 1_000_000))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))


def run_python_test(code: str, imports: Sequence[str], test: str) -> tuple[bool, str]:
    if UNSAFE_CODE.search(code):
        return False, "unsafe construct rejected"
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"
    source = "\n".join([*imports, code, test]) + "\n"
    with tempfile.TemporaryDirectory(prefix="mbpp_") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(source, encoding="utf-8")
        try:
            result = subprocess.run(
                ["python", "-I", str(path)],
                cwd=directory,
                env={"PATH": os.environ.get("PATH", "")},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=4,
                preexec_fn=limited_process,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    message = result.stderr.strip()[-240:]
    return result.returncode == 0, message


def score_code(candidate: str, example: dict[str, Any]) -> dict[str, Any]:
    code = extract_code(candidate)
    imports = list(example.get("test_imports", []))
    tests = list(example.get("test_list", []))
    if not tests:
        return {"reward": 0.0, "passed": 0, "total": 0, "candidate": code}
    passed = 0
    errors = []
    for test in tests:
        ok, error = run_python_test(code, imports, test)
        passed += int(ok)
        if error and len(errors) < 2:
            errors.append(error)
    try:
        ast.parse(code)
        syntax_bonus = 0.05
    except SyntaxError:
        syntax_bonus = 0.0
    fraction = passed / len(tests)
    reward = fraction if fraction > 0 else syntax_bonus
    return {
        "reward": reward,
        "passed": passed,
        "total": len(tests),
        "candidate": code,
        "errors": errors,
    }


class DomainModel(nn.Module):
    """Shared LoRA policy with private SQL/code residual adapters."""

    def __init__(
        self,
        model_name: str,
        adapter_dim: int,
        lora_rank: int,
        adapter_type: str = "output",
        adapter_top_k: int = 6,
        model_revision: str = "",
        model_cache_dir: str = "",
    ) -> None:
        super().__init__()
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=model_revision or None,
            cache_dir=model_cache_dir or None,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        base.config.use_cache = False
        lora = LoraConfig(
            r=lora_rank,
            lora_alpha=2 * lora_rank,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=(
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ),
        )
        self.policy = get_peft_model(base, lora)
        hidden_size = int(base.config.hidden_size)
        canonical_adapter = adapter_type.lower().replace("-", "_")
        self.active_domain = "sql"
        base_model = self.policy.get_base_model()
        layers = getattr(getattr(base_model, "model", None), "layers", None)
        if canonical_adapter == "layerwise":
            if layers is None:
                raise ValueError(
                    f"layerwise adapters cannot locate layers in {type(base_model).__name__}"
                )
            canonical_adapter = "top_k"
            adapter_top_k = len(layers)
        if canonical_adapter == "top_k" and layers is None:
            raise ValueError(
                f"top-k adapters cannot locate layers in {type(base_model).__name__}"
            )

        self.adapter_type = canonical_adapter
        self.private_adapters = DomainPrivateAdapters(
            hidden_size,
            adapter_dim,
            canonical_adapter,
            num_layers=None if layers is None else len(layers),
            top_k=adapter_top_k if canonical_adapter == "top_k" else None,
        )
        if canonical_adapter == "output":
            self.private_adapters.register_output_pre_hook(self.policy.get_output_embeddings())
        elif canonical_adapter == "top_k":
            self.private_adapters.register_block_hooks(layers)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.policy(*args, **kwargs)

    @contextlib.contextmanager
    def domain(self, name: str):
        previous = self.active_domain
        self.active_domain = name
        with self.private_adapters.use_domain(name):
            try:
                yield
            finally:
                self.active_domain = previous

    def shared_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for name, parameter in self.policy.named_parameters()
            if parameter.requires_grad and "lora_" in name
        ]

    def private_parameters(self, domain: str) -> list[nn.Parameter]:
        return self.private_adapters.private_parameters(domain)

    def trainable_state(self) -> dict[str, torch.Tensor]:
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if "lora_" in name or name.startswith("private_adapters.")
        }


def snapshot_trainable_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    """Freeze the initial trainable policy state for reference-policy KL."""

    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def format_prompt(tokenizer: Any, prompt: str) -> list[int]:
    messages = [
        {"role": "system", "content": "You produce exact, executable solutions."},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)


def require_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value.detach()).all()):
        raise FloatingPointError(f"{name} contains NaN or Inf")


@contextlib.contextmanager
def evaluation_policy(model: nn.Module):
    """Run a policy deterministically while preserving its previous mode."""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def sampling_log_probs(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    """Return the exact temperature/top-p distribution used by RL sampling."""
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be finite and positive, got {temperature}")
    if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    filtered = logits.float() / temperature
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(filtered, float("-inf")).scatter(
            -1, sorted_indices, sorted_logits
        )

    log_probs = F.log_softmax(filtered, dim=-1)
    if bool(torch.isnan(log_probs).any()) or not bool(
        torch.isfinite(log_probs).any(dim=-1).all()
    ):
        raise FloatingPointError("sampling policy produced an invalid distribution")
    return log_probs


def encode_supervised(
    tokenizer: Any,
    prompt: str,
    target: str,
    max_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if max_length < 2:
        raise ValueError("max_length must leave room for prompt context and a target token")

    prompt_ids = list(format_prompt(tokenizer, prompt))
    target_ids = list(tokenizer.encode(target, add_special_tokens=False))
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and (not target_ids or target_ids[-1] != eos_token_id):
        target_ids.append(int(eos_token_id))
    if not target_ids:
        raise ValueError("supervised target is empty and the tokenizer has no EOS token")

    # Preserve a target prefix plus EOS. The remaining suffix of the prompt
    # contains the query and the chat generation marker.
    target_budget = max_length - 1
    if eos_token_id is not None and len(target_ids) > target_budget:
        target_ids = target_ids[: target_budget - 1] + [int(eos_token_id)]
    else:
        target_ids = target_ids[:target_budget]
    prompt_budget = max_length - len(target_ids)
    prompt_ids = prompt_ids[-prompt_budget:]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    if not any(label != -100 for label in labels):
        raise ValueError("supervised encoding produced no target labels")
    attention = [1] * len(input_ids)
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([attention], dtype=torch.long, device=device),
        "labels": torch.tensor([labels], dtype=torch.long, device=device),
    }


def sft_loss(
    model: DomainModel,
    tokenizer: Any,
    domain: str,
    prompt: str,
    target: str,
    config: RunConfig,
) -> torch.Tensor:
    batch = encode_supervised(
        tokenizer, prompt, target, config.max_length, next(model.parameters()).device
    )
    with model.domain(domain):
        loss = model.policy(**batch).loss
    require_finite(loss, f"{domain} SFT loss")
    return loss


def bounded_prompt_ids(
    prompt_ids: Sequence[int], max_length: int, max_new_tokens: int
) -> list[int]:
    prompt_budget = max_length - max_new_tokens
    if prompt_budget < 1:
        raise ValueError("max_length must exceed max_new_tokens")
    return list(prompt_ids[-prompt_budget:])


@torch.no_grad()
def generate_candidate(
    model: DomainModel,
    tokenizer: Any,
    domain: str,
    prompt: str,
    max_new_tokens: int,
    config: RunConfig,
    sample: bool,
) -> tuple[str, torch.Tensor, int]:
    unbounded_prompt_ids = format_prompt(tokenizer, prompt)
    prompt_ids = bounded_prompt_ids(
        unbounded_prompt_ids,
        int(getattr(config, "max_length", len(unbounded_prompt_ids) + max_new_tokens)),
        max_new_tokens,
    )
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=next(model.parameters()).device)
    sequences = input_ids
    past_key_values = None
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    with evaluation_policy(model), model.domain(domain):
        for _ in range(max_new_tokens):
            step_input = sequences if past_key_values is None else sequences[:, -1:]
            attention = torch.ones_like(sequences)
            outputs = model.policy(
                input_ids=step_input,
                attention_mask=attention,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = getattr(outputs, "past_key_values", None)
            next_logits = outputs.logits[:, -1, :]
            require_finite(next_logits, "generation logits")
            if sample:
                log_probs = sampling_log_probs(
                    next_logits,
                    temperature=config.temperature,
                    top_p=config.top_p,
                )
                next_token = torch.multinomial(log_probs.exp(), num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            sequences = torch.cat((sequences, next_token), dim=-1)
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break

    completion = sequences[0, input_ids.shape[1] :]
    text = tokenizer.decode(completion, skip_special_tokens=True)
    return text, sequences.detach(), input_ids.shape[1]


def sampled_policy_statistics(
    model: DomainModel,
    domain: str,
    sequence: torch.Tensor,
    prompt_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    reference_parameters: Mapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence = sequence.to(next(model.parameters()).device)
    if sequence.ndim != 2 or not 0 < prompt_length < sequence.shape[1]:
        raise ValueError(
            f"invalid sequence shape {tuple(sequence.shape)} or prompt_length {prompt_length}"
        )
    if reference_parameters is not None and top_p != 1.0:
        raise ValueError("reference-policy KL requires top_p=1.0 shared support")

    attention = torch.ones_like(sequence)

    def current_forward(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        with evaluation_policy(model), model.domain(domain):
            return model(input_ids=ids, attention_mask=mask, use_cache=False).logits

    # Recompute the frozen-base forward during backward instead of retaining a
    # full 3B activation graph for every sampled sequence.
    logits = checkpoint(current_forward, sequence, attention, use_reentrant=False)
    token_logits = logits[:, prompt_length - 1 : -1, :].contiguous()
    del logits
    targets = sequence[:, prompt_length:]
    reference_token_logits: torch.Tensor | None = None
    if reference_parameters is not None:
        with torch.no_grad(), evaluation_policy(model), model.domain(domain):
            reference_logits = torch.func.functional_call(
                model,
                reference_parameters,
                (),
                {
                    "input_ids": sequence,
                    "attention_mask": attention,
                    "use_cache": False,
                },
                strict=False,
            ).logits
            reference_token_logits = reference_logits[
                :, prompt_length - 1 : -1, :
            ].contiguous()
            del reference_logits

    chunk_size = int(getattr(model, "kl_token_chunk_size", 0))
    if chunk_size <= 0:
        chunk_size = 8
    selected_sums: list[torch.Tensor] = []
    kl_sums: list[torch.Tensor] = []

    for offset in range(0, targets.shape[1], chunk_size):
        stop = min(offset + chunk_size, targets.shape[1])
        current_chunk = token_logits[:, offset:stop, :]
        target_chunk = targets[:, offset:stop]

        if reference_token_logits is None:
            def policy_terms(
                current: torch.Tensor, target: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                log_probs = sampling_log_probs(current, temperature, top_p)
                selected = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                return selected.sum(dim=-1), log_probs.new_zeros((current.shape[0],))

            selected_sum, kl_sum = checkpoint(
                policy_terms,
                current_chunk,
                target_chunk,
                use_reentrant=False,
            )
        else:
            reference_chunk = reference_token_logits[:, offset:stop, :]

            def policy_and_kl_terms(
                current: torch.Tensor,
                target: torch.Tensor,
                reference: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                log_probs = sampling_log_probs(current, temperature, top_p)
                selected = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                reference_log_probs = sampling_log_probs(
                    reference, temperature, top_p
                )
                token_kl = (
                    log_probs.exp() * (log_probs - reference_log_probs)
                ).sum(dim=-1)
                return selected.sum(dim=-1), token_kl.sum(dim=-1)

            selected_sum, kl_sum = checkpoint(
                policy_and_kl_terms,
                current_chunk,
                target_chunk,
                reference_chunk,
                use_reentrant=False,
            )
        selected_sums.append(selected_sum)
        kl_sums.append(kl_sum)

    selected = torch.stack(selected_sums).sum(dim=0)
    require_finite(selected, "sampled completion log-probability")
    reference_kl = torch.stack(kl_sums).sum(dim=0).mean()
    require_finite(reference_kl, "reference-policy KL")
    total = selected.mean()
    require_finite(total, "completion log-probability")
    return total, reference_kl


def sampled_log_probability(
    model: DomainModel,
    domain: str,
    sequence: torch.Tensor,
    prompt_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    total, _ = sampled_policy_statistics(
        model,
        domain,
        sequence,
        prompt_length,
        temperature=temperature,
        top_p=top_p,
    )
    return total


def rollout_loss(
    model: DomainModel,
    tokenizer: Any,
    domain: str,
    example: dict[str, Any],
    baseline: float,
    config: RunConfig,
    reference_parameters: Mapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if domain == "sql":
        prompt = sql_prompt(example)
        target = gold_wikisql(example)
        max_tokens = config.sql_new_tokens
    else:
        prompt = code_prompt(example)
        target = example["code"]
        max_tokens = config.code_new_tokens
    text, sequence, prompt_length = generate_candidate(
        model, tokenizer, domain, prompt, max_tokens, config, sample=True
    )
    score = score_sql(text, example) if domain == "sql" else score_code(text, example)
    reference_weight = float(getattr(config, "reference_kl_weight", 0.0))
    if reference_weight < 0.0 or not math.isfinite(reference_weight):
        raise ValueError("reference_kl_weight must be finite and non-negative")
    if reference_weight > 0.0 and reference_parameters is None:
        raise ValueError("positive reference_kl_weight requires a reference snapshot")
    log_probability, reference_kl = sampled_policy_statistics(
        model,
        domain,
        sequence,
        prompt_length,
        temperature=config.temperature,
        top_p=config.top_p,
        reference_parameters=(
            reference_parameters if reference_weight > 0.0 else None
        ),
    )
    advantage = float(score["reward"]) - baseline
    if not math.isfinite(advantage):
        raise FloatingPointError(f"{domain} advantage is not finite: {advantage}")
    reinforcement = -advantage * log_probability
    supervised = sft_loss(model, tokenizer, domain, prompt, target, config)
    weight = config.sql_sft_weight if domain == "sql" else config.code_sft_weight
    reference_loss = reference_weight * reference_kl
    loss = reinforcement + weight * supervised + reference_loss
    require_finite(reinforcement, f"{domain} reinforcement loss")
    require_finite(reference_loss, f"{domain} reference-policy KL loss")
    require_finite(loss, f"{domain} rollout loss")
    record = {
        **score,
        "domain": domain,
        "advantage": advantage,
        "log_probability": float(log_probability.detach()),
        "reinforcement_loss": float(reinforcement.detach()),
        "sft_loss": float(supervised.detach()),
        "reference_kl": float(reference_kl.detach()),
        "reference_kl_loss": float(reference_loss.detach()),
    }
    return loss, record


def group_relative_rollout_loss(
    model: DomainModel,
    tokenizer: Any,
    domain: str,
    example: dict[str, Any],
    config: RunConfig,
    reference_parameters: Mapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if domain == "sql":
        prompt = sql_prompt(example)
        target = gold_wikisql(example)
        max_tokens = config.sql_new_tokens
    else:
        prompt = code_prompt(example)
        target = example["code"]
        max_tokens = config.code_new_tokens

    scores: list[dict[str, Any]] = []
    log_probabilities: list[torch.Tensor] = []
    reference_kls: list[torch.Tensor] = []
    reference_weight = float(getattr(config, "reference_kl_weight", 0.0))
    if reference_weight < 0.0 or not math.isfinite(reference_weight):
        raise ValueError("reference_kl_weight must be finite and non-negative")
    if reference_weight > 0.0 and reference_parameters is None:
        raise ValueError("positive reference_kl_weight requires a reference snapshot")
    for _ in range(max(2, int(config.group_size))):
        text, sequence, prompt_length = generate_candidate(
            model, tokenizer, domain, prompt, max_tokens, config, sample=True
        )
        score = score_sql(text, example) if domain == "sql" else score_code(text, example)
        scores.append(score)
        log_probability, reference_kl = sampled_policy_statistics(
            model,
            domain,
            sequence,
            prompt_length,
            temperature=config.temperature,
            top_p=config.top_p,
            reference_parameters=(
                reference_parameters if reference_weight > 0.0 else None
            ),
        )
        log_probabilities.append(log_probability)
        reference_kls.append(reference_kl)

    rewards = [float(score["reward"]) for score in scores]
    mean_reward = float(np.mean(rewards))
    advantages = [reward - mean_reward for reward in rewards]
    reinforcement = torch.stack(
        [
            -advantage * log_probability
            for advantage, log_probability in zip(advantages, log_probabilities)
        ]
    ).mean()
    supervised = sft_loss(model, tokenizer, domain, prompt, target, config)
    weight = config.sql_sft_weight if domain == "sql" else config.code_sft_weight
    reference_kl = torch.stack(reference_kls).mean()
    reference_loss = reference_weight * reference_kl
    loss = reinforcement + weight * supervised + reference_loss
    require_finite(reinforcement, f"{domain} group-relative reinforcement loss")
    require_finite(reference_loss, f"{domain} reference-policy KL loss")
    require_finite(loss, f"{domain} group-relative rollout loss")
    best = dict(scores[int(np.argmax(rewards))])
    best.update(
        {
            "reward": mean_reward,
            "group_rewards": rewards,
            "domain": domain,
            "advantage": advantages,
            "log_probability": float(
                torch.stack([value.detach() for value in log_probabilities]).mean()
            ),
            "reinforcement_loss": float(reinforcement.detach()),
            "sft_loss": float(supervised.detach()),
            "reference_kl": float(reference_kl.detach()),
            "reference_kl_loss": float(reference_loss.detach()),
        }
    )
    return loss, best


def policy_rollout_loss(
    model: DomainModel,
    tokenizer: Any,
    domain: str,
    example: dict[str, Any],
    baseline: float,
    config: RunConfig,
    reference_parameters: Mapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    loss_name = getattr(config, "rl_loss", "ema_reinforce")
    if loss_name == "group_relative":
        return group_relative_rollout_loss(
            model,
            tokenizer,
            domain,
            example,
            config,
            reference_parameters=reference_parameters,
        )
    if loss_name != "ema_reinforce":
        raise ValueError(f"Unknown RL loss: {loss_name}")
    return rollout_loss(
        model,
        tokenizer,
        domain,
        example,
        baseline,
        config,
        reference_parameters=reference_parameters,
    )


def parameter_gradients(
    loss: torch.Tensor,
    parameters: Sequence[nn.Parameter],
    retain_graph: bool = False,
) -> list[torch.Tensor | None]:
    require_finite(loss, "gradient source loss")
    gradients = list(
        torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=retain_graph)
    )
    for index, gradient in enumerate(gradients):
        if gradient is not None:
            require_finite(gradient, f"gradient {index}")
    return gradients


def dot_product(
    left: Sequence[torch.Tensor | None],
    right: Sequence[torch.Tensor | None],
) -> torch.Tensor:
    values = [
        (a.float() * b.float()).sum()
        for a, b in zip(left, right)
        if a is not None and b is not None
    ]
    return torch.stack(values).sum() if values else torch.tensor(0.0, device="cuda")


def squared_norm(gradients: Sequence[torch.Tensor | None]) -> torch.Tensor:
    values = [gradient.float().square().sum() for gradient in gradients if gradient is not None]
    return torch.stack(values).sum() if values else torch.tensor(0.0, device="cuda")


def subtract_scaled(
    left: Sequence[torch.Tensor | None],
    right: Sequence[torch.Tensor | None],
    scale: torch.Tensor,
) -> list[torch.Tensor | None]:
    output = []
    for a, b in zip(left, right):
        if a is None:
            output.append(None)
        elif b is None:
            output.append(a)
        else:
            output.append(a - scale.to(a.dtype) * b)
    return output


def assign_gradients(
    parameters: Sequence[nn.Parameter],
    gradients: Sequence[torch.Tensor | None],
) -> None:
    for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
        if gradient is not None:
            require_finite(gradient, f"assigned gradient {index}")
        parameter.grad = None if gradient is None else gradient.detach().to(parameter.dtype)


def combine_target_aligned(
    sql_gradients: Sequence[torch.Tensor | None],
    code_gradients: Sequence[torch.Tensor | None],
    anchor_gradients: Sequence[torch.Tensor | None],
    config: RunConfig,
) -> tuple[list[torch.Tensor | None], dict[str, float]]:
    epsilon = 1.0e-12
    raw_dot = dot_product(code_gradients, sql_gradients)
    sql_norm_sq = squared_norm(sql_gradients)
    projection = torch.minimum(raw_dot, torch.zeros_like(raw_dot)) / (sql_norm_sq + epsilon)
    projected = subtract_scaled(code_gradients, sql_gradients, projection)
    projected_anchor_dot = dot_product(projected, anchor_gradients)
    projected_norm = torch.sqrt(squared_norm(projected) + epsilon)
    anchor_norm = torch.sqrt(squared_norm(anchor_gradients) + epsilon)
    sql_norm = torch.sqrt(sql_norm_sq + epsilon)
    cosine = projected_anchor_dot / (projected_norm * anchor_norm + epsilon)
    positive_cosine = torch.clamp(cosine, min=0.0)
    alpha = torch.clamp(
        positive_cosine / max(config.alignment_temperature, epsilon),
        max=1.0,
    )
    norm_scale = torch.clamp(sql_norm / (projected_norm + epsilon), max=config.max_aux_scale)
    coefficient = config.auxiliary_weight * alpha * norm_scale
    combined = []
    for sql_gradient, code_gradient in zip(sql_gradients, projected):
        if sql_gradient is None:
            combined.append(None if code_gradient is None else coefficient.to(code_gradient.dtype) * code_gradient)
        elif code_gradient is None:
            combined.append(sql_gradient)
        else:
            combined.append(sql_gradient + coefficient.to(code_gradient.dtype) * code_gradient)
    diagnostics = {
        "raw_sql_code_dot": float(raw_dot.detach()),
        "projection_coefficient": float(projection.detach()),
        "projected_anchor_cosine": float(cosine.detach()),
        "positive_anchor_cosine": float(positive_cosine.detach()),
        "alpha": float(alpha.detach()),
        "norm_scale": float(norm_scale.detach()),
        "auxiliary_coefficient": float(coefficient.detach()),
        "sql_gradient_norm": float(sql_norm.detach()),
        "code_gradient_norm": float(torch.sqrt(squared_norm(code_gradients) + epsilon).detach()),
        "projected_code_gradient_norm": float(projected_norm.detach()),
        "anchor_gradient_norm": float(anchor_norm.detach()),
    }
    return combined, diagnostics


def average_gradients(
    first: Sequence[torch.Tensor | None],
    second: Sequence[torch.Tensor | None],
) -> list[torch.Tensor | None]:
    output = []
    for left, right in zip(first, second):
        if left is None:
            output.append(None if right is None else 0.5 * right)
        elif right is None:
            output.append(0.5 * left)
        else:
            output.append(0.5 * (left + right))
    return output


def optimizer_for(model: DomainModel, learning_rate: float) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(parameters, lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.01)


def build_code_curriculum(
    model: DomainModel,
    tokenizer: Any,
    sql_anchor: Sequence[dict[str, Any]],
    code_train: Sequence[dict[str, Any]],
    config: RunConfig,
    output_path: Path,
) -> list[dict[str, Any]]:
    """Rank code tasks using only a held-out training SQL anchor gradient."""
    shared = model.shared_parameters()
    anchor_losses = [
        sft_loss(
            model,
            tokenizer,
            "sql",
            sql_prompt(example),
            gold_wikisql(example),
            config,
        )
        for example in sql_anchor[:4]
    ]
    anchor_loss = torch.stack(anchor_losses).mean()
    anchor_gradients = parameter_gradients(anchor_loss, shared)
    anchor_norm = torch.sqrt(squared_norm(anchor_gradients) + 1.0e-12)
    records = []
    probe_count = min(config.curriculum_probe_examples, len(code_train))
    for index, example in enumerate(code_train[:probe_count]):
        loss = sft_loss(
            model,
            tokenizer,
            "code",
            code_prompt(example),
            example["code"],
            config,
        )
        gradients = parameter_gradients(loss, shared)
        gradient_norm = torch.sqrt(squared_norm(gradients) + 1.0e-12)
        cosine = dot_product(gradients, anchor_gradients) / (
            gradient_norm * anchor_norm + 1.0e-12
        )
        records.append(
            {
                "index": index,
                "task_id": int(example["task_id"]),
                "category": code_category(example),
                "anchor_cosine": float(cosine.detach()),
                "positive_score": max(0.0, float(cosine.detach())),
            }
        )
        if (index + 1) % 24 == 0:
            print(f"curriculum probe {index + 1}/{probe_count}", flush=True)
    records.sort(key=lambda item: item["anchor_cosine"], reverse=True)
    selected_count = min(config.curriculum_size, len(records))
    for rank, record in enumerate(records):
        record["rank"] = rank + 1
        record["selected"] = rank < selected_count
    write_json(output_path, records)
    selected = [code_train[record["index"]] for record in records[:selected_count]]
    selected_categories: dict[str, int] = {}
    for example in selected:
        category = code_category(example)
        selected_categories[category] = selected_categories.get(category, 0) + 1
    print(
        "curriculum selected "
        f"{selected_count}/{probe_count}: {json.dumps(selected_categories, sort_keys=True)}",
        flush=True,
    )
    return selected


def warmup(
    model: DomainModel,
    tokenizer: Any,
    sql_train: Sequence[dict[str, Any]],
    code_train: Sequence[dict[str, Any]],
    config: RunConfig,
    trace_path: Path,
) -> None:
    optimizer = optimizer_for(model, config.learning_rate)
    model.train()
    for step in range(config.warmup_steps):
        domain = "sql" if step % 2 == 0 else "code"
        if domain == "sql":
            example = sql_train[(step // 2) % len(sql_train)]
            loss = sft_loss(
                model, tokenizer, "sql", sql_prompt(example), gold_wikisql(example), config
            )
        else:
            example = code_train[(step // 2) % len(code_train)]
            loss = sft_loss(
                model, tokenizer, "code", code_prompt(example), example["code"], config
            )
        optimizer.zero_grad(set_to_none=True)
        require_finite(loss, f"warmup {domain} loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        record = {"phase": "warmup", "step": step, "domain": domain, "loss": float(loss.detach())}
        append_jsonl(trace_path, record)
        if (step + 1) % 8 == 0:
            print(f"warmup {step + 1}/{config.warmup_steps} loss={record['loss']:.4f}", flush=True)


def train_branch(
    branch: str,
    model: DomainModel,
    tokenizer: Any,
    sql_train: Sequence[dict[str, Any]],
    sql_anchor: Sequence[dict[str, Any]],
    code_train: Sequence[dict[str, Any]],
    config: RunConfig,
    trace_path: Path,
) -> None:
    optimizer = optimizer_for(model, config.learning_rate)
    shared = model.shared_parameters()
    sql_private = model.private_parameters("sql")
    code_private = model.private_parameters("code")
    sql_baseline = 0.0
    code_baseline = 0.0
    model.train()
    for step in range(config.rl_steps):
        sql_example = sql_train[(2 * step) % len(sql_train)]
        second_sql = sql_train[(2 * step + 1) % len(sql_train)]
        anchor_example = sql_anchor[step % len(sql_anchor)]
        code_example = code_train[step % len(code_train)]

        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(config.seed + 10_000 + step)
        torch.cuda.manual_seed_all(config.seed + 10_000 + step)
        sql_loss, sql_record = policy_rollout_loss(
            model, tokenizer, "sql", sql_example, sql_baseline, config
        )
        sql_baseline = 0.9 * sql_baseline + 0.1 * float(sql_record["reward"])
        sql_shared = parameter_gradients(sql_loss, shared, retain_graph=True)
        sql_private_gradients = parameter_gradients(sql_loss, sql_private)

        anchor_loss = sft_loss(
            model,
            tokenizer,
            "sql",
            sql_prompt(anchor_example),
            gold_wikisql(anchor_example),
            config,
        )
        anchor_shared = parameter_gradients(anchor_loss, shared)

        if branch == "target_aligned":
            torch.manual_seed(config.seed + 20_000 + step)
            torch.cuda.manual_seed_all(config.seed + 20_000 + step)
            code_loss, code_record = policy_rollout_loss(
                model, tokenizer, "code", code_example, code_baseline, config
            )
            code_baseline = 0.9 * code_baseline + 0.1 * float(code_record["reward"])
            code_shared = parameter_gradients(code_loss, shared, retain_graph=True)
            code_private_gradients = parameter_gradients(code_loss, code_private)
            combined, diagnostics = combine_target_aligned(
                sql_shared, code_shared, anchor_shared, config
            )
            assign_gradients(code_private, code_private_gradients)
        elif branch == "sql_only_2x":
            torch.manual_seed(config.seed + 30_000 + step)
            torch.cuda.manual_seed_all(config.seed + 30_000 + step)
            second_loss, second_record = policy_rollout_loss(
                model, tokenizer, "sql", second_sql, sql_baseline, config
            )
            sql_baseline = 0.9 * sql_baseline + 0.1 * float(second_record["reward"])
            second_shared = parameter_gradients(second_loss, shared, retain_graph=True)
            second_private_gradients = parameter_gradients(second_loss, sql_private)
            combined = average_gradients(sql_shared, second_shared)
            sql_private_gradients = average_gradients(
                sql_private_gradients, second_private_gradients
            )
            code_record = {"reward": None, "domain": "none"}
            diagnostics = {
                "raw_sql_code_dot": 0.0,
                "projection_coefficient": 0.0,
                "projected_anchor_cosine": 0.0,
                "positive_anchor_cosine": 0.0,
                "alpha": 0.0,
                "norm_scale": 0.0,
                "auxiliary_coefficient": 0.0,
            }
            diagnostics["second_sql_reward"] = float(second_record["reward"])
        elif branch == "sql_only":
            combined = sql_shared
            code_record = {"reward": None, "domain": "none"}
            diagnostics = {
                "raw_sql_code_dot": 0.0,
                "projection_coefficient": 0.0,
                "projected_anchor_cosine": 0.0,
                "positive_anchor_cosine": 0.0,
                "alpha": 0.0,
                "norm_scale": 0.0,
                "auxiliary_coefficient": 0.0,
                "second_sql_reward": None,
            }
        else:
            raise ValueError(f"Unknown branch: {branch}")

        assign_gradients(shared, combined)
        assign_gradients(sql_private, sql_private_gradients)
        torch.nn.utils.clip_grad_norm_(
            [*shared, *sql_private, *code_private],
            1.0,
            error_if_nonfinite=True,
        )
        optimizer.step()

        record = {
            "phase": "rl",
            "branch": branch,
            "step": step,
            "sql_reward": float(sql_record["reward"]),
            "sql_denotation": bool(sql_record.get("denotation", False)),
            "code_reward": code_record.get("reward"),
            "sql_baseline": sql_baseline,
            "code_baseline": code_baseline,
            "anchor_loss": float(anchor_loss.detach()),
            **diagnostics,
        }
        append_jsonl(trace_path, record)
        print(
            f"{branch} {step + 1}/{config.rl_steps} "
            f"sql_r={record['sql_reward']:.2f} code_r={record['code_reward']} "
            f"alpha={record['alpha']:.3f}",
            flush=True,
        )


@torch.no_grad()
def evaluate_sql(
    name: str,
    model: DomainModel,
    tokenizer: Any,
    examples: Sequence[dict[str, Any]],
    config: RunConfig,
    predictions_path: Path,
) -> dict[str, Any]:
    totals = {"exact": 0, "executable": 0, "denotation": 0, "reward": 0.0}
    started = time.time()
    for index, example in enumerate(examples):
        text, _, _ = generate_candidate(
            model,
            tokenizer,
            "sql",
            sql_prompt(example),
            config.sql_new_tokens,
            config,
            sample=False,
        )
        score = score_sql(text, example)
        for key in totals:
            totals[key] += float(score[key])
        append_jsonl(
            predictions_path,
            {
                "model": name,
                "index": index,
                "question": example["question"],
                **score,
            },
        )
        if (index + 1) % 12 == 0:
            print(f"eval {name} {index + 1}/{len(examples)}", flush=True)
    count = len(examples)
    return {
        "name": name,
        "examples": count,
        "exact_match": totals["exact"] / count,
        "execution_rate": totals["executable"] / count,
        "denotation_accuracy": totals["denotation"] / count,
        "mean_reward": totals["reward"] / count,
        "seconds": time.time() - started,
    }


def load_real_data(config: RunConfig) -> tuple[list[dict[str, Any]], ...]:
    print("loading WikiSQL and MBPP", flush=True)
    sql = load_dataset("Salesforce/wikisql", trust_remote_code=True)
    try:
        code = load_dataset("google-research-datasets/mbpp", "sanitized")
    except Exception:
        code = load_dataset("google-research-datasets/mbpp")

    train_limit = min(config.train_examples, len(sql["train"]))
    sql_train = [dict(sql["train"][index]) for index in range(train_limit)]
    validation_split = "validation" if "validation" in sql else "dev"
    sql_eval = [
        dict(sql[validation_split][index])
        for index in range(min(config.eval_examples, len(sql[validation_split])))
    ]
    anchor_start = min(train_limit, max(64, train_limit // 2))
    anchor_count = min(max(config.rl_steps, 32), len(sql["train"]) - anchor_start)
    sql_anchor = [
        dict(sql["train"][anchor_start + index]) for index in range(anchor_count)
    ]
    code_split = code["train"]
    code_train = [
        dict(code_split[index])
        for index in range(min(config.train_examples, len(code_split)))
    ]
    return sql_train, sql_anchor, sql_eval, code_train


def build_model(config: RunConfig, tokenizer: Any) -> DomainModel:
    print(f"loading {config.model}", flush=True)
    model = DomainModel(
        config.model,
        config.adapter_dim,
        config.lora_rank,
        getattr(config, "adapter_type", "output"),
        getattr(config, "adapter_top_k", 6),
        model_revision=getattr(config, "model_revision", ""),
        model_cache_dir=getattr(config, "model_cache_dir", ""),
    )
    model.kl_token_chunk_size = int(getattr(config, "kl_token_chunk_size", 8))
    if model.kl_token_chunk_size < 1:
        raise ValueError("kl_token_chunk_size must be positive")
    model.to("cuda")
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"trainable={trainable:,} total={total:,} ({100 * trainable / total:.3f}%)", flush=True)
    return model


def restore_model(config: RunConfig, tokenizer: Any, state_path: Path) -> DomainModel:
    model = build_model(config, tokenizer)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    # Checkpoints created before the shared private-adapter component used the
    # same tensors under ``domain_adapters.*``. Remap only that known prefix so
    # old exploratory checkpoints remain readable without weakening validation.
    state = {
        (
            "private_adapters.domain_adapters." + name[len("domain_adapters.") :]
            if name.startswith("domain_adapters.")
            else name
        ): tensor
        for name, tensor in state.items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    relevant_missing = [
        name
        for name in missing
        if "lora_" in name or name.startswith("private_adapters.")
    ]
    if relevant_missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={relevant_missing}, unexpected={unexpected}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--model-cache-dir", default="")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--warmup-steps", type=int, default=48)
    parser.add_argument("--rl-steps", type=int, default=32)
    parser.add_argument("--eval-examples", type=int, default=48)
    parser.add_argument("--train-examples", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--adapter-dim", type=int, default=128)
    parser.add_argument(
        "--adapter-type",
        "--adapter-kind",
        dest="adapter_type",
        choices=("none", "output", "top_k", "layerwise"),
        default="output",
    )
    parser.add_argument("--adapter-top-k", type=int, default=6)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--alignment-temperature", type=float, default=0.02)
    parser.add_argument("--curriculum-probe-examples", type=int, default=96)
    parser.add_argument("--curriculum-size", type=int, default=32)
    parser.add_argument(
        "--rl-loss", choices=("ema_reinforce", "group_relative"), default="ema_reinforce"
    )
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = RunConfig(**vars(args))
    if config.smoke:
        config.warmup_steps = min(config.warmup_steps, 8)
        config.rl_steps = min(config.rl_steps, 4)
        config.eval_examples = min(config.eval_examples, 8)
        config.train_examples = min(config.train_examples, 48)
        config.sql_new_tokens = 64
        config.code_new_tokens = 96
        config.curriculum_probe_examples = min(config.curriculum_probe_examples, 8)
        config.curriculum_size = min(config.curriculum_size, 4)

    seed_everything(config.seed)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_dir) / f"{timestamp}_seed{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", asdict(config))
    trace_path = run_dir / "trace.jsonl"
    predictions_path = run_dir / "predictions.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(
        config.model,
        revision=getattr(config, "model_revision", "") or None,
        cache_dir=getattr(config, "model_cache_dir", "") or None,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    sql_train, sql_anchor, sql_eval, code_train = load_real_data(config)
    write_json(
        run_dir / "data_manifest.json",
        {
            "sql_dataset": "Salesforce/wikisql",
            "code_dataset": "google-research-datasets/mbpp (sanitized when available)",
            "sql_train_examples": len(sql_train),
            "sql_anchor_examples": len(sql_anchor),
            "sql_eval_examples": len(sql_eval),
            "code_train_examples": len(code_train),
        },
    )

    common = build_model(config, tokenizer)
    warmup(common, tokenizer, sql_train, code_train, config, trace_path)
    common_state = run_dir / "common_trainable.pt"
    torch.save(common.trainable_state(), common_state)
    code_curriculum = build_code_curriculum(
        common,
        tokenizer,
        sql_anchor,
        code_train,
        config,
        run_dir / "curriculum.json",
    )
    summaries = [
        evaluate_sql("common", common, tokenizer, sql_eval, config, predictions_path)
    ]
    del common
    torch.cuda.empty_cache()

    for branch in ("sql_only", "sql_only_2x", "target_aligned"):
        seed_everything(config.seed + 101)
        model = restore_model(config, tokenizer, common_state)
        train_branch(
            branch,
            model,
            tokenizer,
            sql_train,
            sql_anchor,
            code_curriculum if branch == "target_aligned" else code_train,
            config,
            trace_path,
        )
        torch.save(model.trainable_state(), run_dir / f"{branch}_trainable.pt")
        summaries.append(
            evaluate_sql(branch, model, tokenizer, sql_eval, config, predictions_path)
        )
        del model
        torch.cuda.empty_cache()

    result = {
        "run_dir": str(run_dir),
        "config": asdict(config),
        "results": summaries,
        "target_aligned_minus_sql_only": next(
            item["denotation_accuracy"] for item in summaries if item["name"] == "target_aligned"
        )
        - next(item["denotation_accuracy"] for item in summaries if item["name"] == "sql_only"),
        "target_aligned_minus_sql_only_2x": next(
            item["denotation_accuracy"] for item in summaries if item["name"] == "target_aligned"
        )
        - next(item["denotation_accuracy"] for item in summaries if item["name"] == "sql_only_2x"),
    }
    write_json(run_dir / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
