"""Deterministic training, model-selection, and anchor partitions."""

from __future__ import annotations

import random
from dataclasses import dataclass

MODEL_SELECTION_FRACTION = 0.20


@dataclass(frozen=True)
class TrainingIndices:
    optimization: tuple[int, ...]
    anchor: tuple[int, ...]
    model_selection: tuple[int, ...]

    @property
    def training_partition_size(self) -> int:
        return len(self.optimization) + len(self.anchor)


def split_training_indices(
    total_examples: int,
    anchor_examples: int,
    seed: int,
    limit: int = 0,
) -> TrainingIndices:
    """Apply the paper's 80/20 split, then reserve ``D_a`` from training.

    ``limit`` is an execution-only option for smoke tests. Full runs leave it at
    zero and therefore partition the complete official training split.
    """

    if total_examples < 3:
        raise ValueError("the training split must contain at least three examples")
    if anchor_examples < 1:
        raise ValueError("anchor_examples must be positive")
    if limit < 0:
        raise ValueError("limit must be non-negative")

    indices = list(range(total_examples))
    random.Random(seed).shuffle(indices)
    if limit > 0:
        indices = indices[: min(limit, len(indices))]
    if len(indices) < 3:
        raise ValueError(
            "the selected training subset must contain at least three examples"
        )

    selection_count = int(len(indices) * MODEL_SELECTION_FRACTION + 0.5)
    selection_count = min(max(1, selection_count), len(indices) - 2)
    model_selection = indices[:selection_count]
    training_partition = indices[selection_count:]
    anchor_count = min(anchor_examples, len(training_partition) - 1)
    anchor = training_partition[:anchor_count]
    optimization = training_partition[anchor_count:]

    partitions = (set(optimization), set(anchor), set(model_selection))
    if any(
        left & right
        for index, left in enumerate(partitions)
        for right in partitions[index + 1 :]
    ):
        raise AssertionError("training partitions overlap")
    if sum(len(partition) for partition in partitions) != len(indices):
        raise AssertionError("training partitions do not cover the selected examples")

    return TrainingIndices(
        optimization=tuple(sorted(optimization)),
        anchor=tuple(sorted(anchor)),
        model_selection=tuple(sorted(model_selection)),
    )
