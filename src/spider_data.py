#!/usr/bin/env python3
"""Spider data, prompting, and deterministic single-database SQL rewards."""

from __future__ import annotations

import json
import random
import re
import sqlite3
from collections import Counter
from itertools import permutations
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

import sqlglot
from sqlglot import expressions as exp

import data_protocol


def quote_identifier(value: object) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def schema_text(table: Mapping[str, Any]) -> str:
    table_names = list(table["table_names_original"])
    column_names = list(table["column_names_original"])
    column_types = list(table["column_types"])
    grouped: list[list[str]] = [[] for _ in table_names]
    for index, (table_index, column_name) in enumerate(column_names):
        if int(table_index) >= 0:
            grouped[int(table_index)].append(
                f"{quote_identifier(column_name)} {column_types[index]}"
            )
    parts = [
        f"{quote_identifier(name)}(" + ", ".join(grouped[index]) + ")"
        for index, name in enumerate(table_names)
    ]

    def column_reference(column_index: int) -> str:
        table_index, column_name = column_names[column_index]
        return (
            f"{quote_identifier(table_names[int(table_index)])}."
            f"{quote_identifier(column_name)}"
        )

    primary_keys = ", ".join(
        column_reference(int(index)) for index in table.get("primary_keys", [])
    )
    foreign_keys = ", ".join(
        f"{column_reference(int(left))}={column_reference(int(right))}"
        for left, right in table.get("foreign_keys", [])
    )
    if primary_keys:
        parts.append(f"PK {primary_keys}")
    if foreign_keys:
        parts.append(f"FK {foreign_keys}")
    return "; ".join(parts)


def spider_prompt(example: Mapping[str, Any]) -> str:
    return (
        "Write one SQLite query. Return SQL only.\n"
        f"Question: {example['question']}\n"
        f"Schema: {example['_schema_text']}"
    )


def gold_spider(example: Mapping[str, Any]) -> str:
    return str(example["query"]).strip().rstrip(";") + ";"


def choose_anchor_database_ids(
    examples: Sequence[Mapping[str, Any]], target_examples: int, seed: int
) -> frozenset[str]:
    if target_examples <= 0:
        raise ValueError("anchor target must be positive")
    counts: Counter[str] = Counter(str(row["db_id"]) for row in examples)
    database_ids = sorted(counts)
    if len(database_ids) < 2:
        raise ValueError("Spider anchor split requires at least two databases")
    random.Random(seed).shuffle(database_ids)

    selected: list[str] = []
    selected_count = 0
    for database_id in database_ids:
        count = counts[database_id]
        if selected_count + count <= target_examples:
            selected.append(database_id)
            selected_count += count
    if not selected:
        selected.append(min(database_ids, key=lambda value: counts[value]))
        selected_count = counts[selected[0]]
    if selected_count < target_examples:
        remaining = [value for value in database_ids if value not in selected]
        if remaining:
            closest = min(
                remaining,
                key=lambda value: (
                    abs(selected_count + counts[value] - target_examples),
                    database_ids.index(value),
                ),
            )
            selected.append(closest)
    if len(selected) == len(database_ids):
        selected.pop()
    return frozenset(selected)


def _attach_metadata(
    example: Mapping[str, Any],
    serialized_schema: str,
    database_dir: Path,
) -> dict[str, Any]:
    database_id = str(example["db_id"])
    database_path = database_dir / database_id / f"{database_id}.sqlite"
    if not database_path.is_file():
        raise FileNotFoundError(f"missing Spider database: {database_path}")
    return {
        **dict(example),
        "_schema_text": serialized_schema,
        "_database_path": str(database_path),
    }


def load_spider_data(config: Any):
    data_dir = Path(config.spider_data_dir)
    database_dir = data_dir / "database"
    tables = {
        str(row["db_id"]): row
        for row in json.loads((data_dir / "tables.json").read_text())
    }
    schemas = {database_id: schema_text(table) for database_id, table in tables.items()}
    train_rows: list[dict[str, Any]] = []
    for filename in ("train_spider.json", "train_others.json"):
        train_rows.extend(json.loads((data_dir / filename).read_text()))
    dev_rows = json.loads((data_dir / "dev.json").read_text())

    split = data_protocol.split_training_indices(
        len(train_rows),
        config.anchor_examples,
        config.seed + 31,
        limit=config.sql_train_limit,
    )
    sql_train = [train_rows[index] for index in split.optimization]
    sql_anchor = [train_rows[index] for index in split.anchor]
    sql_model_selection = [train_rows[index] for index in split.model_selection]
    if config.sql_eval_limit > 0:
        dev_rows = dev_rows[: config.sql_eval_limit]

    train_ids = {str(row["db_id"]) for row in sql_train}
    observed_anchor_ids = {str(row["db_id"]) for row in sql_anchor}
    selection_ids = {str(row["db_id"]) for row in sql_model_selection}
    dev_ids = {str(row["db_id"]) for row in dev_rows}
    if (train_ids | observed_anchor_ids | selection_ids) & dev_ids:
        raise ValueError("Spider official train and development database IDs overlap")

    def attach(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            _attach_metadata(row, schemas[str(row["db_id"])], database_dir)
            for row in rows
        ]

    return (
        attach(sql_train),
        attach(sql_anchor),
        attach(sql_model_selection),
        attach(dev_rows),
    )


def extract_spider_sql(text: str) -> str:
    fenced = re.search(
        r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL
    )
    if fenced:
        text = fenced.group(1)
    starts = [
        match
        for pattern in (r"\bWITH\b", r"\bSELECT\b")
        if (match := re.search(pattern, text, flags=re.IGNORECASE)) is not None
    ]
    if starts:
        text = text[min(match.start() for match in starts) :]
    if ";" in text:
        text = text.split(";", 1)[0] + ";"
    return text.strip()


def normalize_sql(text: str) -> str:
    return re.sub(r"\s+", " ", extract_spider_sql(text).rstrip(";").strip()).lower()


def _query_only(candidate: str) -> bool:
    try:
        parsed = sqlglot.parse_one(candidate, read="sqlite")
    except Exception:
        return False
    return isinstance(parsed, exp.Query)


def _execute_read_only(database_path: Path, query: str, max_steps: int) -> list[tuple]:
    uri = f"file:{quote(str(database_path.resolve()), safe='/')}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=2.0)
    calls = 0

    def progress() -> int:
        nonlocal calls
        calls += 1
        return int(calls * 1_000 > max_steps)

    try:
        connection.execute("PRAGMA query_only = ON")
        connection.set_progress_handler(progress, 1_000)
        return connection.execute(query).fetchall()
    finally:
        connection.close()


def _permute_row(row: tuple, permutation: tuple[int, ...]) -> tuple:
    return tuple(row[index] for index in permutation)


def denotations_equal(
    gold_rows: Sequence[tuple], candidate_rows: Sequence[tuple], order_matters: bool
) -> bool:
    if not gold_rows or not candidate_rows:
        return list(gold_rows) == list(candidate_rows)
    if len(gold_rows) != len(candidate_rows):
        return False
    columns = len(gold_rows[0])
    if len(candidate_rows[0]) != columns:
        return False
    candidate_permutations = (
        permutations(range(columns)) if columns <= 6 else [tuple(range(columns))]
    )
    gold = list(gold_rows) if order_matters else Counter(gold_rows)
    for permutation in candidate_permutations:
        permuted = [_permute_row(row, permutation) for row in candidate_rows]
        comparison = permuted if order_matters else Counter(permuted)
        if gold == comparison:
            return True
    return False


def score_spider(
    candidate_text: str,
    example: Mapping[str, Any],
    max_steps: int = 2_000_000,
) -> dict[str, Any]:
    candidate = extract_spider_sql(candidate_text)
    gold = gold_spider(example)
    exact = normalize_sql(candidate) == normalize_sql(gold)
    parsed = _query_only(candidate)
    executable = False
    denotation = False
    error = ""
    if parsed:
        try:
            database_path = Path(str(example["_database_path"]))
            gold_rows = _execute_read_only(database_path, gold, max_steps)
            candidate_rows = _execute_read_only(database_path, candidate, max_steps)
            executable = True
            denotation = denotations_equal(
                gold_rows,
                candidate_rows,
                order_matters="order by" in gold.lower(),
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:240]
    reward = 1.0 if denotation else 0.20 if executable else 0.05 if parsed else 0.0
    return {
        "reward": reward,
        "candidate": candidate,
        "gold": gold,
        "exact": exact,
        "parsed": parsed,
        "executable": executable,
        "denotation": denotation,
        "error": error,
        "db_id": str(example["db_id"]),
    }
