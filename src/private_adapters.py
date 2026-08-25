"""Domain-private residual adapters for decoder-only Transformer models.

The module deliberately does not discover Transformer layers by name. Callers
either apply adapters explicitly or pass an ordered sequence of decoder blocks
to the hook helpers.
"""

from __future__ import annotations

import contextlib
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle


AdapterStructure = Literal["none", "output", "top_k"]
DOMAINS = ("sql", "code")


def _canonical_structure(structure: str) -> AdapterStructure:
    normalized = structure.lower().replace("-", "_")
    aliases = {
        "none": "none",
        "output": "output",
        "top_k": "top_k",
        "top_k_block_output": "top_k",
    }
    try:
        return aliases[normalized]  # type: ignore[return-value]
    except KeyError as error:
        choices = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown adapter structure {structure!r}; choose from {choices}") from error


class BottleneckResidualAdapter(nn.Module):
    """A zero-initialized bottleneck adapter with an explicit residual path."""

    def __init__(self, hidden_size: int, bottleneck_dim: int) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")

        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim
        self.down = nn.Linear(hidden_size, bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, received {hidden.shape[-1]}"
            )
        adapter_dtype = self.down.weight.dtype
        delta = self.up(self.activation(self.down(hidden.to(dtype=adapter_dtype))))
        return hidden + delta.to(dtype=hidden.dtype)


class DomainPrivateAdapters(nn.Module):
    """Independent SQL/code adapters at the output or last ``top_k`` blocks.

    ``output`` applies one adapter immediately before the language-model head.
    ``top_k`` applies one adapter after each of the final ``top_k`` decoder
    blocks. ``none`` is a parameter-free identity path.
    """

    def __init__(
        self,
        hidden_size: int,
        bottleneck_dim: int,
        structure: str = "output",
        *,
        num_layers: int | None = None,
        top_k: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")

        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim
        self.structure = _canonical_structure(structure)
        self.num_layers = num_layers
        self.top_k = top_k
        self.active_domain = "sql"
        self.domain_adapters = nn.ModuleDict()
        self._hook_handles: list[RemovableHandle] = []

        if self.structure == "none":
            self.selected_layer_indices: tuple[int, ...] = ()
            return

        if self.structure == "output":
            self.selected_layer_indices = ()
            self.domain_adapters.update(
                {
                    domain: BottleneckResidualAdapter(hidden_size, bottleneck_dim)
                    for domain in DOMAINS
                }
            )
            return

        if num_layers is None or num_layers <= 0:
            raise ValueError("num_layers must be positive for top_k adapters")
        if top_k is None or not 1 <= top_k <= num_layers:
            raise ValueError("top_k must be between 1 and num_layers")

        self.selected_layer_indices = tuple(range(num_layers - top_k, num_layers))
        self.domain_adapters.update(
            {
                domain: nn.ModuleDict(
                    {
                        str(index): BottleneckResidualAdapter(hidden_size, bottleneck_dim)
                        for index in self.selected_layer_indices
                    }
                )
                for domain in DOMAINS
            }
        )

    def _validate_domain(self, domain: str) -> None:
        if domain not in DOMAINS:
            raise ValueError(f"Unknown domain {domain!r}; expected 'sql' or 'code'")

    def _domain(self, domain: str | None) -> str:
        selected = self.active_domain if domain is None else domain
        self._validate_domain(selected)
        return selected

    @contextlib.contextmanager
    def use_domain(self, domain: str) -> Iterator[None]:
        """Temporarily select the domain used by registered hooks."""

        self._validate_domain(domain)
        previous = self.active_domain
        self.active_domain = domain
        try:
            yield
        finally:
            self.active_domain = previous

    def adapt_output(self, hidden: torch.Tensor, domain: str | None = None) -> torch.Tensor:
        """Apply an output adapter, or return ``hidden`` for ``none``."""

        selected = self._domain(domain)
        if self.structure == "none":
            return hidden
        if self.structure != "output":
            raise RuntimeError("adapt_output is available only for output adapters")
        return self.domain_adapters[selected](hidden)

    def adapt_block_output(
        self,
        layer_index: int,
        output: torch.Tensor | tuple[Any, ...],
        domain: str | None = None,
    ) -> torch.Tensor | tuple[Any, ...]:
        """Adapt a selected decoder-block output while preserving tuple metadata."""

        selected = self._domain(domain)
        if self.structure == "none" or layer_index not in self.selected_layer_indices:
            return output
        if self.structure != "top_k":
            raise RuntimeError("adapt_block_output is available only for top_k adapters")

        adapter = self.domain_adapters[selected][str(layer_index)]
        if isinstance(output, torch.Tensor):
            return adapter(output)
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return (adapter(output[0]), *output[1:])
        raise TypeError("Decoder block output must be a Tensor or a tuple beginning with a Tensor")

    def register_output_pre_hook(self, output_module: nn.Module) -> RemovableHandle | None:
        """Attach an ``output`` adapter before an arbitrary LM output module."""

        if self.structure == "none":
            return None
        if self.structure != "output":
            raise RuntimeError("Output hooks require structure='output'")

        def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError("Output module inputs must begin with a hidden-state Tensor")
            return (self.adapt_output(inputs[0]), *inputs[1:])

        handle = output_module.register_forward_pre_hook(hook)
        self._hook_handles.append(handle)
        return handle

    def register_block_hooks(
        self, blocks: Sequence[nn.Module]
    ) -> tuple[RemovableHandle, ...]:
        """Attach adapters to selected positions in an ordered decoder block list."""

        if self.structure == "none":
            return ()
        if self.structure != "top_k":
            raise RuntimeError("Block hooks require structure='top_k'")
        if len(blocks) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} decoder blocks, received {len(blocks)}")

        handles: list[RemovableHandle] = []
        for index in self.selected_layer_indices:
            def hook(
                _module: nn.Module,
                _inputs: tuple[Any, ...],
                output: torch.Tensor | tuple[Any, ...],
                layer_index: int = index,
            ) -> torch.Tensor | tuple[Any, ...]:
                return self.adapt_block_output(layer_index, output)

            handles.append(blocks[index].register_forward_hook(hook))
        self._hook_handles.extend(handles)
        return tuple(handles)

    def remove_hooks(self) -> None:
        """Remove every hook registered through this component."""

        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def private_parameters(self, domain: str) -> list[nn.Parameter]:
        """Return parameters owned exclusively by one domain."""

        self._validate_domain(domain)
        if self.structure == "none":
            return []
        return list(self.domain_adapters[domain].parameters())

    def parameter_counts(self) -> dict[str, int]:
        """Return SQL, code, and total private-adapter parameter counts."""

        counts = {
            domain: sum(parameter.numel() for parameter in self.private_parameters(domain))
            for domain in DOMAINS
        }
        counts["total"] = sum(counts.values())
        return counts

    def sql_deployment_state_dict(self, *, clone: bool = False) -> OrderedDict[str, torch.Tensor]:
        """Return this component's state dict without code-private parameters."""

        return filter_sql_adapter_state_dict(self.state_dict(), clone=clone)


def filter_sql_adapter_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    adapter_prefix: str = "domain_adapters",
    clone: bool = False,
) -> OrderedDict[str, torch.Tensor]:
    """Drop code-private adapter tensors while retaining shared and SQL state.

    For a full model state dict, pass the full path to this component's
    ``domain_adapters`` attribute via ``adapter_prefix``.
    """

    prefix = adapter_prefix.rstrip(".")
    code_prefix = f"{prefix}.code."
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, tensor in state_dict.items():
        if name.startswith(code_prefix):
            continue
        result[name] = tensor.detach().clone() if clone else tensor
    return result


__all__ = [
    "AdapterStructure",
    "BottleneckResidualAdapter",
    "DomainPrivateAdapters",
    "filter_sql_adapter_state_dict",
]
