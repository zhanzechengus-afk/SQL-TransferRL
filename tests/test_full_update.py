from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_full_experiment as full  # noqa: E402


def optimizer_state_for(
    optimizer: torch.optim.Optimizer, parameter: nn.Parameter
) -> dict[str, torch.Tensor | float]:
    result = {}
    for key, value in optimizer.state[parameter].items():
        result[key] = value.detach().clone() if isinstance(value, torch.Tensor) else value
    return result


class FullUpdateTests(unittest.TestCase):
    def test_source_normalizer_warms_up_per_source_and_clips(self) -> None:
        normalizer = full.SourceRewardNormalizer(
            warmup=2, beta=0.95, std_floor=0.1, clip=2.0
        )
        first, first_record = normalizer.observe("mbpp", 0.0)
        second, second_record = normalizer.observe("mbpp", 1.0)
        other, other_record = normalizer.observe("apps", 1.0)
        clipped, ready_record = normalizer.observe("mbpp", 10.0)

        self.assertEqual(first, 0.0)
        self.assertEqual(second, 0.0)
        self.assertEqual(other, 0.0)
        self.assertFalse(first_record["source_reward_ready"])
        self.assertFalse(second_record["source_reward_ready"])
        self.assertFalse(other_record["source_reward_ready"])
        self.assertTrue(ready_record["source_reward_ready"])
        self.assertEqual(clipped, 2.0)
        self.assertAlmostEqual(ready_record["source_reward_mean"], 0.5)
        self.assertAlmostEqual(ready_record["source_reward_std"], math.sqrt(0.5))

    def test_normalized_anchor_ema_has_unit_global_norm(self) -> None:
        first = [torch.tensor([3.0, 0.0]), torch.tensor([0.0, 4.0])]
        second = [torch.tensor([0.0, 2.0]), torch.tensor([0.0, 0.0])]
        initial = full.normalized_ema_update(None, first, beta=0.9)
        updated = full.normalized_ema_update(initial, second, beta=0.9)

        self.assertAlmostEqual(float(full.gradient_norm(initial)), 1.0, places=6)
        self.assertAlmostEqual(float(full.gradient_norm(updated)), 1.0, places=6)
        self.assertGreater(float(updated[0][1]), 0.0)

    def test_projection_is_one_way_and_zero_gate_preserves_sql_tensor(self) -> None:
        config = full.FullConfig(auxiliary_weight=0.3, alignment_temperature=0.02)
        sql = [torch.tensor([1.0, 0.0])]
        code = [torch.tensor([-1.0, 1.0])]
        projection_spec = full.branch_spec("projection_only")
        projected, diagnostics = full.combine_branch_gradients(
            sql, code, projection_spec, config
        )

        self.assertAlmostEqual(diagnostics["projection_coefficient"], -1.0)
        torch.testing.assert_close(projected[0], torch.tensor([1.0, 0.3]))

        target_spec = full.branch_spec("target_aligned")
        gated, gated_diagnostics = full.combine_branch_gradients(
            sql,
            code,
            target_spec,
            config,
            anchor_gradients=[torch.tensor([0.0, -1.0])],
        )
        self.assertEqual(gated_diagnostics["alpha"], 0.0)
        self.assertIs(gated[0], sql[0])

        zero_config = full.FullConfig(auxiliary_weight=0.0)
        disabled, disabled_diagnostics = full.combine_branch_gradients(
            sql, code, full.branch_spec("naive_mixed"), zero_config
        )
        self.assertEqual(disabled_diagnostics["auxiliary_coefficient"], 0.0)
        self.assertIs(disabled[0], sql[0])

    def test_code_rl_and_sft_gradients_are_parameter_isolated(self) -> None:
        shared = nn.Parameter(torch.tensor([2.0]))
        private = nn.Parameter(torch.tensor([3.0]))
        verifier_rl = 4.0 * shared.sum()
        code_sft = (shared + private).square().sum()

        shared_gradients, private_gradients = full.separated_code_gradients(
            verifier_rl,
            code_sft,
            [shared],
            [private],
            code_sft_weight=0.25,
        )

        torch.testing.assert_close(shared_gradients[0], torch.tensor([4.0]))
        torch.testing.assert_close(private_gradients[0], torch.tensor([2.5]))

    def test_grouped_clipping_preserves_primary_parameters_and_adam_state(self) -> None:
        torch.manual_seed(4)
        initial = [nn.Parameter(torch.tensor([1.0])), nn.Parameter(torch.tensor([-2.0])), nn.Parameter(torch.tensor([0.5]))]
        sql_only = [nn.Parameter(value.detach().clone()) for value in initial]
        auxiliary = [nn.Parameter(value.detach().clone()) for value in initial]
        optimizer_sql = torch.optim.AdamW(sql_only, lr=0.01, betas=(0.9, 0.95), weight_decay=0.01)
        optimizer_aux = torch.optim.AdamW(auxiliary, lr=0.01, betas=(0.9, 0.95), weight_decay=0.01)
        primary_gradients = [torch.tensor([2.0]), torch.tensor([-3.0])]

        full.apply_gradient_groups(
            sql_only[:2], primary_gradients, sql_only[2:], [None], max_norm=1.0
        )
        optimizer_sql.step()
        full.apply_gradient_groups(
            auxiliary[:2], primary_gradients, auxiliary[2:], [torch.tensor([100.0])], max_norm=1.0
        )
        optimizer_aux.step()

        for sql_parameter, auxiliary_parameter in zip(sql_only[:2], auxiliary[:2]):
            torch.testing.assert_close(sql_parameter, auxiliary_parameter, rtol=0.0, atol=0.0)
            sql_state = optimizer_state_for(optimizer_sql, sql_parameter)
            auxiliary_state = optimizer_state_for(optimizer_aux, auxiliary_parameter)
            self.assertEqual(sql_state.keys(), auxiliary_state.keys())
            for key in sql_state:
                if isinstance(sql_state[key], torch.Tensor):
                    torch.testing.assert_close(
                        sql_state[key], auxiliary_state[key], rtol=0.0, atol=0.0
                    )
                else:
                    self.assertEqual(sql_state[key], auxiliary_state[key])
        self.assertNotEqual(
            float(sql_only[2].detach()), float(auxiliary[2].detach())
        )

    def test_nonfinite_group_gradient_fails_fast(self) -> None:
        parameter = nn.Parameter(torch.tensor([1.0]))
        with self.assertRaises(FloatingPointError):
            full.apply_gradient_groups(
                [parameter], [torch.tensor([float("nan")])], [], [], max_norm=1.0
            )


if __name__ == "__main__":
    unittest.main()
