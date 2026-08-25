#!/usr/bin/env python3
"""Subprocess entry point for one complete Spider score."""

from __future__ import annotations

import json
import sys

import spider_data


def main() -> None:
    payload = json.load(sys.stdin)
    result = spider_data.score_spider(
        str(payload["candidate_text"]),
        payload["example"],
        max_steps=int(payload["max_steps"]),
    )
    json.dump(result, sys.stdout, allow_nan=False)


if __name__ == "__main__":
    main()
