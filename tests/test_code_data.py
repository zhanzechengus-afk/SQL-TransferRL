import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.code_data import (
    BuildResult,
    ReferenceVerificationError,
    SourceRows,
    SplitPolicyError,
    VerificationResult,
    build_code_pool,
    canonical_source,
    merge_test_suites,
    normalize_row,
    require_reference_solution_passes,
    require_training_split,
)


class PassingVerifier:
    def verify(self, task):
        return VerificationResult({test["test_id"]: True for test in task.tests})


def mbpp_row(task_id=601, prompt="Return x plus one."):
    return {
        "task_id": task_id,
        "prompt": prompt,
        "code": "def add_one(x):\n    return x + 1",
        "test_list": ["assert add_one(1) == 2", "assert add_one(5) == 6"],
        "challenge_test_list": ["assert add_one(-1) == 0"],
    }


def apps_row(problem_id=1, question="Echo the input."):
    return {
        "problem_id": problem_id,
        "question": question,
        "solutions": json.dumps(["print(input())"]),
        "input_output": json.dumps({"inputs": ["a\n", "b\n"], "outputs": ["a\n", "b\n"]}),
    }


def codecontests_row():
    duplicate = {"input": ["1\n"], "output": ["1\n"]}
    return {
        "source": "codeforces",
        "name": "echo-one",
        "description": "Print the supplied integer.",
        "solutions": {"language": [3], "solution": ["print(input())"]},
        "public_tests": duplicate,
        "private_tests": duplicate,
        "generated_tests": {"input": ["2\n"], "output": ["2\n"]},
    }


def taco_row():
    return {
        "url": "https://example.test/taco/7",
        "question": "Read two integers and print their sum.",
        "solutions": json.dumps(["a,b=map(int,input().split());print(a+b)"]),
        "input_output": json.dumps({"inputs": ["1 2\n"], "outputs": ["3\n"]}),
        "difficulty": "EASY",
    }


def xcodeeval_row():
    return {
        "src_uid": "CF_42_A",
        "description": "Print twice the input integer.",
        "solution": "x=int(input());print(2*x)",
        "unit_tests": {"inputs": ["3\n", "5\n"], "outputs": ["6\n", "10\n"]},
    }


class SplitPolicyTest(unittest.TestCase):
    def test_only_declared_training_splits_are_allowed(self):
        for source in ("mbpp", "apps", "codecontests", "taco"):
            self.assertEqual(require_training_split(source, "train"), (source, "train"))
        self.assertEqual(
            require_training_split("xcodeeval", "program-synthesis-train"),
            ("xcodeeval", "program_synthesis_train"),
        )

        for source, split in (("mbpp", "test"), ("apps", "validation"), ("taco", "dev")):
            with self.subTest(source=source, split=split), self.assertRaises(SplitPolicyError):
                require_training_split(source, split)

    def test_evaluation_only_sources_are_always_blocked(self):
        for source in ("HumanEval", "openai/openai_humaneval", "EvalPlus", "HumanEval+"):
            with self.subTest(source=source), self.assertRaises(SplitPolicyError):
                canonical_source(source)


class NormalizationTest(unittest.TestCase):
    def test_all_supported_sources_normalize_without_downloading(self):
        fixtures = [
            ("mbpp", "train", mbpp_row()),
            ("apps", "train", apps_row()),
            ("codecontests", "train", codecontests_row()),
            ("taco", "train", taco_row()),
            ("xcodeeval", "program_synthesis_train", xcodeeval_row()),
        ]
        for source, split, row in fixtures:
            with self.subTest(source=source):
                task = normalize_row(source, split, row)
                self.assertTrue(task.uid.startswith(source + ":"))
                self.assertTrue(task.statement)
                self.assertTrue(task.reference_solution)
                self.assertGreater(len(task.tests), 0)
                self.assertEqual(len({test["test_id"] for test in task.tests}), len(task.tests))

    def test_codecontests_merges_all_suites_without_duplicate_tests(self):
        task = normalize_row("codecontests", "train", codecontests_row())
        self.assertEqual(len(task.tests), 2)
        self.assertEqual(task.tests[0]["suites"], ["private", "public"])
        self.assertEqual(task.tests[1]["suites"], ["generated"])

    def test_generic_suite_merge_keeps_assertions(self):
        tests = merge_test_suites(
            [("standard", ["assert f(1) == 1"]), ("challenge", ["assert f(2) == 2"])]
        )
        self.assertEqual([test["kind"] for test in tests], ["assert", "assert"])


class VerificationTest(unittest.TestCase):
    def test_verifier_must_cover_every_test(self):
        task = normalize_row("mbpp", "train", mbpp_row())

        def incomplete(_task):
            return VerificationResult({"t0001": True})

        with self.assertRaises(ReferenceVerificationError) as context:
            require_reference_solution_passes(task, incomplete)
        self.assertEqual(context.exception.reason, "verification_incomplete")

    def test_any_failed_test_rejects_reference(self):
        task = normalize_row("mbpp", "train", mbpp_row())
        outcomes = {test["test_id"]: True for test in task.tests}
        outcomes[task.tests[-1]["test_id"]] = False
        with self.assertRaises(ReferenceVerificationError) as context:
            require_reference_solution_passes(task, lambda _: VerificationResult(outcomes))
        self.assertEqual(context.exception.reason, "reference_failed")


class BuildPoolTest(unittest.TestCase):
    def test_oversized_json_integer_is_filtered_instead_of_aborting_pool(self):
        row = apps_row()
        huge_integer = "9" * 5000
        huge_payload = (
            '{"inputs": [[' + huge_integer + ']], "outputs": ["ok"]}'
        )
        row["input_output"] = huge_payload

        original_loads = json.loads

        def limited_loads(value, *args, **kwargs):
            if value == huge_payload:
                raise ValueError(
                    "Exceeds the limit (4300 digits) for integer string conversion"
                )
            return original_loads(value, *args, **kwargs)

        with mock.patch("src.code_data.json.loads", side_effect=limited_loads):
            result = build_code_pool(
                [SourceRows("apps", "train", [row])], PassingVerifier()
            )

        self.assertEqual(len(result.tasks), 0)
        self.assertEqual(
            result.manifest["sources"]["apps"]["filter_reasons"][
                "unsupported_large_integer"
            ],
            1,
        )

    def test_full_build_deduplicates_id_and_normalized_statement(self):
        first = mbpp_row(601, "Return x plus one.")
        duplicate_id = mbpp_row(601, "A different statement with the same source ID.")
        duplicate_statement = apps_row(9, "  RETURN   x plus one. ")
        unique = apps_row(10, "Echo the input.")
        result = build_code_pool(
            [
                SourceRows("mbpp", "train", [first, duplicate_id]),
                SourceRows("apps", "train", [duplicate_statement, unique]),
            ],
            PassingVerifier(),
        )

        self.assertEqual(len(result.tasks), 2)
        self.assertEqual(result.manifest["mode"], "full")
        self.assertEqual(result.manifest["totals"]["deduplicated"], 2)
        self.assertEqual(result.manifest["sources"]["mbpp"]["duplicate_source_id"], 1)
        self.assertEqual(result.manifest["sources"]["apps"]["duplicate_statement"], 1)

    def test_incomplete_and_failed_verification_are_manifested(self):
        rows = [mbpp_row(601), mbpp_row(602, "Return x plus two.")]

        def verifier(task):
            if task.task_id == "601":
                return VerificationResult({"t0001": True})
            return VerificationResult({test["test_id"]: False for test in task.tests})

        result = build_code_pool([SourceRows("mbpp", "train", rows)], verifier)
        source = result.manifest["sources"]["mbpp"]
        self.assertEqual(source["accepted"], 0)
        self.assertEqual(source["filtered"], 2)
        self.assertEqual(source["filter_reasons"]["verification_incomplete"], 1)
        self.assertEqual(source["filter_reasons"]["reference_failed"], 1)

    def test_pilot_cap_is_explicit_and_full_mode_has_no_limit(self):
        rows = [mbpp_row(601 + index, f"Task number {index}.") for index in range(3)]
        full = build_code_pool([SourceRows("mbpp", "train", rows)], PassingVerifier())
        pilot = build_code_pool(
            [SourceRows("mbpp", "train", rows)], PassingVerifier(), pilot_cap={"mbpp": 1}
        )

        self.assertEqual(len(full.tasks), 3)
        self.assertEqual(full.manifest["mode"], "full")
        self.assertEqual(len(pilot.tasks), 1)
        self.assertEqual(pilot.manifest["mode"], "pilot")
        self.assertTrue(pilot.manifest["sources"]["mbpp"]["cap_reached"])
        self.assertFalse(pilot.manifest["sources"]["mbpp"]["input_exhausted"])
        self.assertEqual(pilot.manifest["sources"]["mbpp"]["available_rows"], 3)

    def test_manifest_is_json_serializable_and_writable(self):
        result = build_code_pool(
            [SourceRows("xcodeeval", "train", [xcodeeval_row()])], PassingVerifier()
        )
        self.assertIsInstance(result, BuildResult)
        json.dumps(result.manifest)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            result.write_manifest(path)
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["totals"]["accepted"], 1)


if __name__ == "__main__":
    unittest.main()
