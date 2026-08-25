from __future__ import annotations

import contextlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_pilot  # noqa: E402


class TinyTokenizer:
    eos_token_id = 4
    eos_token = "<eos>"

    def __init__(self, prompt_ids: list[int] | None = None) -> None:
        self.prompt_ids = prompt_ids or [0]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        del messages, tokenize, add_generation_prompt
        return list(self.prompt_ids)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [1 + (ord(character) % 3) for character in text]

    def decode(self, tokens: torch.Tensor, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(int(token)) for token in tokens)


class TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=0.8)
        self.transition = nn.Parameter(
            torch.tensor(
                [
                    [0.1, 1.6, 0.8, -0.4, -3.0],
                    [0.2, -0.1, 1.4, 0.7, -3.0],
                    [1.1, 0.3, -0.2, 1.5, -3.0],
                    [0.5, 1.0, 0.1, -0.3, -3.0],
                    [0.0, 0.0, 0.0, 0.0, 2.0],
                ],
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = True,
        **kwargs,
    ):
        del attention_mask, past_key_values, use_cache, kwargs
        logits = self.dropout(self.transition[input_ids])
        return SimpleNamespace(logits=logits, past_key_values=None)


class TinyDomainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.policy = TinyPolicy()
        self.active_domain = "sql"

    def forward(self, *args, **kwargs):
        return self.policy(*args, **kwargs)

    @contextlib.contextmanager
    def domain(self, name: str):
        previous = self.active_domain
        self.active_domain = name
        try:
            yield
        finally:
            self.active_domain = previous


def manual_log_probs(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    scaled = logits.float() / temperature
    sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove = cumulative > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.full_like(scaled, float("-inf"))
    filtered.scatter_(0, sorted_indices, sorted_logits)
    return torch.log_softmax(filtered, dim=-1)


class RunPilotTests(unittest.TestCase):
    def test_sampling_tokens_and_logprob_match_manual_policy(self) -> None:
        model = TinyDomainModel()
        tokenizer = TinyTokenizer([0])
        config = SimpleNamespace(temperature=0.7, top_p=0.82)

        torch.manual_seed(1234)
        _, sequence, prompt_length = run_pilot.generate_candidate(
            model,
            tokenizer,
            "sql",
            "ignored",
            max_new_tokens=3,
            config=config,
            sample=True,
        )
        self.assertTrue(model.training)

        torch.manual_seed(1234)
        previous = 0
        manual_tokens: list[int] = []
        manual_total = torch.tensor(0.0)
        for _ in range(3):
            log_probs = manual_log_probs(
                model.policy.transition[previous], config.temperature, config.top_p
            )
            token = int(torch.multinomial(log_probs.exp(), 1))
            manual_tokens.append(token)
            manual_total = manual_total + log_probs[token]
            previous = token

        self.assertEqual(sequence[0, prompt_length:].tolist(), manual_tokens)
        actual_total = run_pilot.sampled_log_probability(
            model,
            "sql",
            sequence,
            prompt_length,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        self.assertTrue(model.training)
        self.assertTrue(torch.allclose(actual_total, manual_total, atol=1e-6, rtol=1e-6))

    def test_advantage_reverses_policy_gradient_direction(self) -> None:
        model = TinyDomainModel()
        sequence = torch.tensor([[0, 1]], dtype=torch.long)

        positive_log_prob = run_pilot.sampled_log_probability(
            model, "sql", sequence, 1, temperature=1.0, top_p=1.0
        )
        positive_gradient = torch.autograd.grad(
            -positive_log_prob, model.policy.transition, retain_graph=False
        )[0]

        negative_log_prob = run_pilot.sampled_log_probability(
            model, "sql", sequence, 1, temperature=1.0, top_p=1.0
        )
        negative_gradient = torch.autograd.grad(
            negative_log_prob, model.policy.transition, retain_graph=False
        )[0]

        self.assertLess(float(positive_gradient[0, 1]), 0.0)
        self.assertGreater(float(negative_gradient[0, 1]), 0.0)
        self.assertTrue(
            torch.allclose(positive_gradient, -negative_gradient, atol=1e-7, rtol=1e-7)
        )

    def test_reference_policy_kl_uses_frozen_trainable_snapshot(self) -> None:
        model = TinyDomainModel()
        sequence = torch.tensor([[0, 1, 2]], dtype=torch.long)
        reference = run_pilot.snapshot_trainable_parameters(model)

        _, initial_kl = run_pilot.sampled_policy_statistics(
            model,
            "sql",
            sequence,
            1,
            temperature=0.8,
            top_p=1.0,
            reference_parameters=reference,
        )
        self.assertAlmostEqual(float(initial_kl.detach()), 0.0, places=6)

        with torch.no_grad():
            model.policy.transition[0, 1].add_(0.5)
        _, changed_kl = run_pilot.sampled_policy_statistics(
            model,
            "sql",
            sequence,
            1,
            temperature=0.8,
            top_p=1.0,
            reference_parameters=reference,
        )
        gradient = torch.autograd.grad(changed_kl, model.policy.transition)[0]

        self.assertGreater(float(changed_kl.detach()), 0.0)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        self.assertAlmostEqual(float(reference["policy.transition"][0, 1]), 1.6)

        with self.assertRaises(ValueError):
            run_pilot.sampled_policy_statistics(
                model,
                "sql",
                sequence,
                1,
                temperature=0.8,
                top_p=0.95,
                reference_parameters=reference,
            )

    def test_long_prompt_keeps_target_and_eos_labels(self) -> None:
        tokenizer = TinyTokenizer(prompt_ids=list(range(20)))
        encoded = run_pilot.encode_supervised(
            tokenizer,
            prompt="ignored",
            target="ab",
            max_length=6,
            device=torch.device("cpu"),
        )

        labels = encoded["labels"][0].tolist()
        active_labels = [label for label in labels if label != -100]
        self.assertEqual(encoded["input_ids"].shape, (1, 6))
        self.assertEqual(active_labels, tokenizer.encode("ab") + [tokenizer.eos_token_id])
        self.assertEqual(labels[-1], tokenizer.eos_token_id)
        self.assertGreater(len(active_labels), 0)

    def test_invalid_sampling_parameters_and_nonfinite_loss_fail_fast(self) -> None:
        with self.assertRaises(ValueError):
            run_pilot.sampling_log_probs(torch.zeros(3), temperature=0.0, top_p=0.9)
        with self.assertRaises(ValueError):
            run_pilot.sampling_log_probs(torch.zeros(3), temperature=1.0, top_p=0.0)
        with self.assertRaises(FloatingPointError):
            run_pilot.require_finite(torch.tensor(float("nan")), "test")


if __name__ == "__main__":
    unittest.main()
