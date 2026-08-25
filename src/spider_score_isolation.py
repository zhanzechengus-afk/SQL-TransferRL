#!/usr/bin/env python3
"""Wall-clock isolation for the complete Spider online scorer."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import spider_data


_RESULT_KEYS = {
    "reward",
    "candidate",
    "gold",
    "exact",
    "parsed",
    "executable",
    "denotation",
    "error",
    "db_id",
}


def _timeout_result(candidate_text: str, example: Mapping[str, Any]) -> dict[str, Any]:
    candidate = spider_data.extract_spider_sql(candidate_text)
    gold = spider_data.gold_spider(example)
    return {
        "reward": 0.0,
        "candidate": candidate,
        "gold": gold,
        "exact": spider_data.normalize_sql(candidate) == spider_data.normalize_sql(gold),
        "parsed": False,
        "executable": False,
        "denotation": False,
        "error": "score_timeout",
        "db_id": str(example["db_id"]),
    }


def _validate_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("isolated Spider scorer returned a non-object result")
    missing = _RESULT_KEYS - result.keys()
    extra = result.keys() - _RESULT_KEYS
    if missing or extra:
        raise ValueError(
            f"isolated Spider scorer fields differ: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    reward = result["reward"]
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise ValueError("isolated Spider scorer reward is not numeric")
    if not math.isfinite(float(reward)):
        raise ValueError("isolated Spider scorer reward is not finite")
    for key in ("exact", "parsed", "executable", "denotation"):
        if not isinstance(result[key], bool):
            raise ValueError(f"isolated Spider scorer field {key} is not boolean")
    for key in ("candidate", "gold", "error", "db_id"):
        if not isinstance(result[key], str):
            raise ValueError(f"isolated Spider scorer field {key} is not a string")
    return result


def score_spider_isolated(
    candidate_text: str,
    example: Mapping[str, Any],
    max_steps: int = 2_000_000,
    timeout_seconds: float = 30.0,
    worker_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one complete Spider score in a fresh interpreter with a wall timeout."""
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    worker = (
        Path(worker_path)
        if worker_path is not None
        else Path(__file__).with_name("score_spider_worker.py")
    )
    if not worker.is_file():
        raise FileNotFoundError(f"missing isolated Spider scorer worker: {worker}")
    payload = json.dumps(
        {
            "candidate_text": candidate_text,
            "example": dict(example),
            "max_steps": max_steps,
        }
    )
    try:
        completed = subprocess.run(
            [sys.executable, str(worker)],
            input=payload,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
            cwd=worker.parent,
        )
    except subprocess.TimeoutExpired:
        return _timeout_result(candidate_text, example)
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-1_000:]
        raise RuntimeError(
            f"isolated Spider scorer failed with code {completed.returncode}: {detail}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("isolated Spider scorer returned invalid JSON") from exc
    return _validate_result(result)
