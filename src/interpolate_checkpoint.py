#!/usr/bin/env python3
"""Interpolate an RL checkpoint toward its SQL-SFT initialization."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def interpolate(
    sft: dict[str, torch.Tensor], rl: dict[str, torch.Tensor], alpha: float
) -> dict[str, torch.Tensor]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if sft.keys() != rl.keys():
        raise ValueError("checkpoint parameter sets do not match")
    return {name: sft[name] + alpha * (rl[name] - sft[name]) for name in sft}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--rl", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sft = torch.load(args.sft, map_location="cpu", weights_only=True)
    rl = torch.load(args.rl, map_location="cpu", weights_only=True)
    state = interpolate(sft, rl, args.alpha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)


if __name__ == "__main__":
    main()
