#!/usr/bin/env python3
"""BIRD data loading, prompting, and deterministic SQLite rewards."""

from __future__ import annotations

import json
import random
import re
import sqlite3
from collections import Counter
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

    def primary_key_reference(value: object) -> str:
        if isinstance(value, (list, tuple)):
            references = [primary_key_reference(index) for index in value]
            return "(" + ", ".join(references) + ")"
        return column_reference(int(value))

    primary_keys = ", ".join(
        primary_key_reference(index) for index in table.get("primary_keys", [])
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


def schema_text_from_sqlite(database_path: Path) -> str:
    uri = f"file:{quote(str(database_path.resolve()), safe='/')}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=2.0)
    try:
        table_names = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        parts: list[str] = []
        primary_keys: list[str] = []
        foreign_keys: list[str] = []
        for table_name in table_names:
            quoted_table = quote_identifier(table_name)
            columns = list(connection.execute(f"PRAGMA table_info({quoted_table})"))
            column_text = ", ".join(
                f"{quote_identifier(row[1])} {str(row[2] or 'unknown').lower()}"
                for row in columns
            )
            parts.append(f"{quoted_table}({column_text})")
            key_columns = sorted(
                (row for row in columns if int(row[5]) > 0),
                key=lambda row: int(row[5]),
            )
            for row in key_columns:
                primary_keys.append(f"{quoted_table}.{quote_identifier(row[1])}")
            for row in connection.execute(f"PRAGMA foreign_key_list({quoted_table})"):
                foreign_keys.append(
                    f"{quoted_table}.{quote_identifier(row[3])}="
                    f"{quote_identifier(row[2])}.{quote_identifier(row[4])}"
                )
        if primary_keys:
            parts.append("PK " + ", ".join(primary_keys))
        if foreign_keys:
            parts.append("FK " + ", ".join(foreign_keys))
        return "; ".join(parts)
    finally:
        connection.close()


def bird_prompt(example: Mapping[str, Any]) -> str:
    parts = [
        "Write one SQLite query. Return SQL only.",
        f"Schema: {example['_schema_text']}",
    ]
    evidence = str(example.get("evidence", "")).strip()
    if evidence:
        parts.append(f"Evidence: {evidence}")
    parts.append(f"Question: {example['question']}")
    return "\n".join(parts)


def gold_bird(example: Mapping[str, Any]) -> str:
    return str(example["SQL"]).strip().rstrip(";") + ";"


def choose_anchor_database_ids(
    examples: Sequence[Mapping[str, Any]], target_examples: int, seed: int
) -> frozenset[str]:
    if target_examples <= 0:
        raise ValueError("anchor target must be positive")
    counts: Counter[str] = Counter(str(row["db_id"]) for row in examples)
    database_ids = sorted(counts)
    if len(database_ids) < 2:
        raise ValueError("BIRD anchor split requires at least two databases")
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
    example: Mapping[str, Any], serialized_schema: str, database_path: Path
) -> dict[str, Any]:
    if not database_path.is_file():
        raise FileNotFoundError(f"missing BIRD database: {database_path}")
    return {
        **dict(example),
        "_schema_text": serialized_schema,
        "_database_path": str(database_path),
    }


def write_aligned_difficulty_json(
    examples: Sequence[Mapping[str, Any]],
    source_path: Path,
    output_path: Path,
) -> Path:
    """Write official BIRD difficulty rows in prediction order."""
    source_rows = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("BIRD difficulty source must be a non-empty JSON list")

    rows_by_question_id: dict[str, Mapping[str, Any]] = {}
    for row in source_rows:
        if not isinstance(row, Mapping) or "question_id" not in row:
            raise ValueError("every BIRD difficulty row needs question_id")
        question_id = str(row["question_id"])
        if question_id in rows_by_question_id:
            raise ValueError(f"duplicate BIRD difficulty question_id: {question_id}")
        rows_by_question_id[question_id] = row

    aligned: list[dict[str, Any]] = []
    for example in examples:
        if "question_id" not in example:
            raise ValueError("every BIRD evaluation example needs question_id")
        question_id = str(example["question_id"])
        if question_id not in rows_by_question_id:
            raise KeyError(f"missing BIRD difficulty question_id: {question_id}")
        source_row = rows_by_question_id[question_id]
        if str(source_row.get("db_id")) != str(example.get("db_id")):
            raise ValueError(f"BIRD difficulty db_id mismatch for question {question_id}")
        difficulty = str(source_row.get("difficulty", ""))
        if difficulty not in {"simple", "moderate", "challenging"}:
            raise ValueError(
                f"invalid BIRD difficulty for question {question_id}: {difficulty!r}"
            )
        aligned.append(
            {
                "question_id": source_row["question_id"],
                "db_id": source_row["db_id"],
                "difficulty": difficulty,
            }
        )

    output_path.write_text(
        json.dumps(aligned, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output_path


def load_bird_data(config: Any):
    train_marker = Path(config.bird_train_marker)
    dev_marker = Path(config.bird_dev_marker)
    if not train_marker.is_file():
        raise FileNotFoundError(f"missing validated BIRD train marker: {train_marker}")
    if not dev_marker.is_file():
        raise FileNotFoundError(f"missing validated BIRD dev marker: {dev_marker}")

    train_root = Path(config.bird_train_root)
    train_database_dir = train_root / "train_databases"
    dev_database_dir = Path(config.bird_dev_db_dir)
    train_rows = json.loads((train_root / "train.json").read_text())
    dev_rows = json.loads(Path(config.bird_dev_json).read_text())
    tables = {
        str(row["db_id"]): row
        for row in json.loads((train_root / "train_tables.json").read_text())
    }
    train_schemas = {
        database_id: schema_text(table) for database_id, table in tables.items()
    }

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
        raise ValueError("BIRD official train and development database IDs overlap")

    missing_train_schemas = sorted(
        (train_ids | observed_anchor_ids | selection_ids) - set(train_schemas)
    )
    if missing_train_schemas:
        raise KeyError(f"missing BIRD train schemas: {missing_train_schemas}")
    dev_schemas = {
        database_id: schema_text_from_sqlite(
            dev_database_dir / database_id / f"{database_id}.sqlite"
        )
        for database_id in sorted(dev_ids)
    }

    def attach_train(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            _attach_metadata(
                row,
                train_schemas[str(row["db_id"])],
                train_database_dir
                / str(row["db_id"])
                / f"{row['db_id']}.sqlite",
            )
            for row in rows
        ]

    def attach_dev(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            _attach_metadata(
                row,
                dev_schemas[str(row["db_id"])],
                dev_database_dir / str(row["db_id"]) / f"{row['db_id']}.sqlite",
            )
            for row in rows
        ]

    return (
        attach_train(sql_train),
        attach_train(sql_anchor),
        attach_train(sql_model_selection),
        attach_dev(dev_rows),
    )


def extract_bird_sql(text: str) -> str:
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
    return re.sub(r"\s+", " ", extract_bird_sql(text).rstrip(";").strip()).lower()


def parse_official_execution(output: str) -> float:
    for line in output.splitlines():
        if line.strip().lower().startswith("accuracy"):
            values = re.findall(r"\d+(?:\.\d+)?", line)
            if values:
                value = float(values[-1]) / 100.0
                if 0.0 <= value <= 1.0:
                    return value
    raise ValueError("could not parse official BIRD execution accuracy")


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


def denotations_equal(
    gold_rows: Sequence[tuple], candidate_rows: Sequence[tuple]
) -> bool:
    # The locked official BIRD evaluator compares sets of SQLite result tuples.
    return set(gold_rows) == set(candidate_rows)


def score_bird(
    candidate_text: str,
    example: Mapping[str, Any],
    max_steps: int = 2_000_000,
) -> dict[str, Any]:
    candidate = extract_bird_sql(candidate_text)
    gold = gold_bird(example)
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
            denotation = denotations_equal(gold_rows, candidate_rows)
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
