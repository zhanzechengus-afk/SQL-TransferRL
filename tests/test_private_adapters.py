from __future__ import annotations

import unittest
from collections import OrderedDict

import torch
import torch.nn as nn

from src.private_adapters import DomainPrivateAdapters, filter_sql_adapter_state_dict


class IdentityBlock(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class PrivateAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.hidden = torch.randn(2, 5, 16)

    def test_none_and_output_preserve_shape_and_start_as_identity(self) -> None:
        disabled = DomainPrivateAdapters(16, 4, "none")
        output = DomainPrivateAdapters(16, 4, "output")

        disabled_hidden = disabled.adapt_output(self.hidden, "sql")
        sql_hidden = output.adapt_output(self.hidden, "sql")
        code_hidden = output.adapt_output(self.hidden, "code")

        self.assertEqual(disabled_hidden.shape, self.hidden.shape)
        self.assertEqual(sql_hidden.shape, self.hidden.shape)
        self.assertEqual(code_hidden.shape, self.hidden.shape)
        torch.testing.assert_close(disabled_hidden, self.hidden)
        torch.testing.assert_close(sql_hidden, self.hidden)
        torch.testing.assert_close(code_hidden, self.hidden)

    def test_sql_and_code_parameters_are_isolated(self) -> None:
        adapters = DomainPrivateAdapters(16, 4, "output")
        sql_parameters = adapters.private_parameters("sql")
        code_parameters = adapters.private_parameters("code")

        self.assertTrue(set(map(id, sql_parameters)).isdisjoint(map(id, code_parameters)))
        with torch.no_grad():
            adapters.domain_adapters["sql"].up.bias.fill_(0.5)

        torch.testing.assert_close(adapters.adapt_output(self.hidden, "sql"), self.hidden + 0.5)
        torch.testing.assert_close(adapters.adapt_output(self.hidden, "code"), self.hidden)

    def test_top_k_adapters_attach_only_to_final_blocks(self) -> None:
        adapters = DomainPrivateAdapters(
            16,
            4,
            "top-k-block-output",
            num_layers=6,
            top_k=2,
        )
        self.assertEqual(adapters.selected_layer_indices, (4, 5))

        blocks = nn.ModuleList(IdentityBlock() for _ in range(6))
        handles = adapters.register_block_hooks(blocks)
        self.assertEqual(len(handles), 2)
        with torch.no_grad():
            adapters.domain_adapters["sql"]["4"].up.bias.fill_(1.0)
            adapters.domain_adapters["sql"]["5"].up.bias.fill_(2.0)

        hidden = self.hidden
        with adapters.use_domain("sql"):
            for block in blocks:
                hidden = block(hidden)
        torch.testing.assert_close(hidden, self.hidden + 3.0)

        with adapters.use_domain("code"):
            hidden = self.hidden
            for block in blocks:
                hidden = block(hidden)
        torch.testing.assert_close(hidden, self.hidden)
        adapters.remove_hooks()

    def test_top_k_preserves_decoder_tuple_outputs(self) -> None:
        adapters = DomainPrivateAdapters(16, 4, "top_k", num_layers=4, top_k=1)
        cache = object()
        output = (self.hidden, cache)

        untouched = adapters.adapt_block_output(2, output, "sql")
        adapted = adapters.adapt_block_output(3, output, "sql")

        self.assertIs(untouched, output)
        self.assertIs(adapted[1], cache)
        torch.testing.assert_close(adapted[0], self.hidden)

    def test_parameter_counts_match_closed_form(self) -> None:
        hidden_size = 16
        bottleneck_dim = 4
        per_adapter = 2 * hidden_size * bottleneck_dim + bottleneck_dim + hidden_size

        none = DomainPrivateAdapters(hidden_size, bottleneck_dim, "none")
        output = DomainPrivateAdapters(hidden_size, bottleneck_dim, "output")
        top_k = DomainPrivateAdapters(
            hidden_size,
            bottleneck_dim,
            "top_k",
            num_layers=8,
            top_k=3,
        )

        self.assertEqual(none.parameter_counts(), {"sql": 0, "code": 0, "total": 0})
        self.assertEqual(
            output.parameter_counts(),
            {"sql": per_adapter, "code": per_adapter, "total": 2 * per_adapter},
        )
        self.assertEqual(
            top_k.parameter_counts(),
            {
                "sql": 3 * per_adapter,
                "code": 3 * per_adapter,
                "total": 6 * per_adapter,
            },
        )

    def test_sql_deployment_filter_drops_only_code_private_state(self) -> None:
        state = OrderedDict(
            {
                "shared.weight": torch.tensor([1.0]),
                "private.domain_adapters.sql.up.weight": torch.tensor([2.0]),
                "private.domain_adapters.code.up.weight": torch.tensor([3.0]),
            }
        )
        filtered = filter_sql_adapter_state_dict(
            state,
            adapter_prefix="private.domain_adapters",
            clone=True,
        )

        self.assertEqual(
            list(filtered),
            ["shared.weight", "private.domain_adapters.sql.up.weight"],
        )
        self.assertIsNot(filtered["shared.weight"], state["shared.weight"])

        adapters = DomainPrivateAdapters(16, 4, "output")
        deployment_state = adapters.sql_deployment_state_dict()
        self.assertTrue(any(name.startswith("domain_adapters.sql.") for name in deployment_state))
        self.assertFalse(any(name.startswith("domain_adapters.code.") for name in deployment_state))


if __name__ == "__main__":
    unittest.main()
