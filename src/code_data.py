"""Policy-safe construction of executable code-training tasks.

This module deliberately separates dataset ingestion from model training.  It
does not download datasets or execute untrusted programs.  Callers provide rows
from an explicitly named source/split and a sandboxed ``ReferenceVerifier``.
Only tasks whose reference solution is checked against every merged test are
admitted to the returned pool.
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Set, Tuple, Union


class SplitPolicyError(ValueError):
    """Raised when a requested source or split is not eligible for training."""


class NormalizationError(ValueError):
    """Raised when a raw row cannot be converted into an executable task."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


class ReferenceVerificationError(ValueError):
    """Raised when a verifier omits a test or the reference solution fails."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class SourcePolicy:
    source: str
    training_splits: Tuple[str, ...]
    note: str


SOURCE_POLICIES: Dict[str, SourcePolicy] = {
    "mbpp": SourcePolicy(
        "mbpp",
        ("train",),
        "Use task IDs assigned to the official train split; prompting, validation, and test are held out.",
    ),
    "apps": SourcePolicy("apps", ("train",), "Use the official APPS train split only."),
    "codecontests": SourcePolicy(
        "codecontests", ("train",), "Use the official CodeContests train split only."
    ),
    "taco": SourcePolicy("taco", ("train",), "Use the official TACO train split only."),
    "xcodeeval": SourcePolicy(
        "xcodeeval",
        ("train", "program_synthesis_train"),
        "Use program-synthesis training data only; Compact and Titan remain held out.",
    ),
}

_SOURCE_ALIASES = {
    "mbpp": "mbpp",
    "google-research-datasets/mbpp": "mbpp",
    "apps": "apps",
    "codeparrot/apps": "apps",
    "codecontests": "codecontests",
    "code_contests": "codecontests",
    "deepmind/code_contests": "codecontests",
    "google-deepmind/code_contests": "codecontests",
    "taco": "taco",
    "baai/taco": "taco",
    "xcodeeval": "xcodeeval",
    "ntunlp/xcodeeval": "xcodeeval",
}

_BLOCKED_SOURCE_ALIASES = {
    "humaneval",
    "human_eval",
    "openai/openai_humaneval",
    "evalplus",
    "eval_plus",
    "humanevalplus",
    "humaneval+",
    "mbppplus",
    "mbpp+",
}

_SPLIT_ALIASES = {
    "training": "train",
    "program-synthesis-train": "program_synthesis_train",
    "program_synthesis_training": "program_synthesis_train",
}


def canonical_source(source: str) -> str:
    value = unicodedata.normalize("NFKC", str(source)).strip().casefold()
    value = re.sub(r"\s+", "_", value)
    if value in _BLOCKED_SOURCE_ALIASES:
        raise SplitPolicyError(f"{source!r} is evaluation-only and cannot be used for training")
    try:
        return _SOURCE_ALIASES[value]
    except KeyError as exc:
        raise SplitPolicyError(f"unsupported code-training source: {source!r}") from exc


def canonical_split(split: str) -> str:
    value = unicodedata.normalize("NFKC", str(split)).strip().casefold()
    value = re.sub(r"[\s/]+", "_", value)
    return _SPLIT_ALIASES.get(value, value)


def require_training_split(source: str, split: str) -> Tuple[str, str]:
    canonical = canonical_source(source)
    normalized_split = canonical_split(split)
    allowed = SOURCE_POLICIES[canonical].training_splits
    if normalized_split not in allowed:
        raise SplitPolicyError(
            f"split {split!r} is blocked for {canonical}; allowed training splits: {', '.join(allowed)}"
        )
    return canonical, normalized_split


def normalize_identifier(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-._")
    if not text:
        raise NormalizationError("missing_task_id", f"cannot normalize task ID from {value!r}")
    return text


def normalize_statement(statement: str) -> str:
    text = html.unescape(unicodedata.normalize("NFKC", str(statement)))
    text = re.sub(r"```[^\n]*", " ", text)
    text = re.sub(r"[`*_#]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().casefold()
    if not text:
        raise NormalizationError("missing_statement", "empty problem statement")
    return text


def _jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
    except ValueError as exc:
        message = str(exc)
        if "integer string conversion" in message and "Exceeds the limit" in message:
            raise NormalizationError(
                "unsupported_large_integer",
                "JSON test payload contains an integer beyond the sandbox limit",
            ) from exc
        raise


def _as_list(value: Any) -> List[Any]:
    value = _jsonish(value)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _first(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    return None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _test_identity(test: Mapping[str, Any]) -> str:
    return _canonical_json({key: value for key, value in test.items() if key not in {"test_id", "suites"}})


def _raw_tests_from_value(value: Any, suite: str, default_kind: str = "stdin") -> List[Dict[str, Any]]:
    value = _jsonish(value)
    if value is None:
        return []

    if isinstance(value, Mapping):
        inputs = _first(value, ("inputs", "input"))
        outputs = _first(value, ("outputs", "output", "expected"))
        function_name = _first(value, ("fn_name", "function_name", "entry_point"))
        if inputs is not None and outputs is not None:
            input_rows = _as_list(inputs)
            output_rows = _as_list(outputs)
            if len(input_rows) != len(output_rows):
                raise NormalizationError(
                    "malformed_tests",
                    f"suite {suite!r} has {len(input_rows)} inputs and {len(output_rows)} outputs",
                )
            kind = "function" if function_name else default_kind
            tests = []
            for test_input, expected in zip(input_rows, output_rows):
                test: Dict[str, Any] = {
                    "kind": kind,
                    "input": test_input,
                    "expected": expected,
                    "suites": [suite],
                }
                if function_name:
                    test["function_name"] = str(function_name)
                tests.append(test)
            return tests
        for key in ("tests", "test_cases", "unit_tests", "unittests"):
            if key in value:
                return _raw_tests_from_value(value[key], suite, default_kind)
        return []

    tests = []
    for item in _as_list(value):
        item = _jsonish(item)
        if isinstance(item, str):
            tests.append({"kind": "assert", "code": item, "suites": [suite]})
        elif isinstance(item, Mapping):
            assertion = _first(item, ("assertion", "code", "test"))
            if assertion is not None and _first(item, ("input", "inputs")) is None:
                tests.append({"kind": "assert", "code": str(assertion), "suites": [suite]})
                continue
            test_input = _first(item, ("input", "inputs", "arguments"))
            expected = _first(item, ("output", "outputs", "expected"))
            if test_input is not None and expected is not None:
                function_name = _first(item, ("fn_name", "function_name", "entry_point"))
                test = {
                    "kind": "function" if function_name else default_kind,
                    "input": test_input,
                    "expected": expected,
                    "suites": [suite],
                }
                if function_name:
                    test["function_name"] = str(function_name)
                tests.append(test)
    return tests


def merge_test_suites(suites: Sequence[Tuple[str, Any]], default_kind: str = "stdin") -> Tuple[Dict[str, Any], ...]:
    """Merge all available suites and deduplicate identical tests.

    Public, private, generated, challenge, and ordinary tests are all retained.
    When the same test appears in multiple suites, one test record carries every
    contributing suite name.
    """

    merged: List[Dict[str, Any]] = []
    positions: Dict[str, int] = {}
    for suite, value in suites:
        for test in _raw_tests_from_value(value, suite, default_kind):
            identity = _test_identity(test)
            if identity in positions:
                existing = merged[positions[identity]]
                existing["suites"] = sorted(set(existing.get("suites", [])) | set(test["suites"]))
            else:
                positions[identity] = len(merged)
                merged.append(dict(test))
    for index, test in enumerate(merged, start=1):
        test["test_id"] = f"t{index:04d}"
    return tuple(merged)


def _task_id(source: str, row: Mapping[str, Any]) -> str:
    if source == "mbpp":
        value = _first(row, ("task_id", "id"))
    elif source == "apps":
        value = _first(row, ("problem_id", "task_id", "id"))
    elif source == "codecontests":
        value = _first(row, ("task_id", "problem_id", "id", "url"))
        if value is None and row.get("name"):
            value = f"{row.get('source', 'unknown')}:{row['name']}"
    elif source == "taco":
        value = _first(row, ("task_id", "problem_id", "id", "url"))
        if value is None and row.get("name"):
            value = f"{row.get('source', 'unknown')}:{row['name']}"
    else:
        value = _first(row, ("src_uid", "task_id", "problem_id", "id"))
    if value is None:
        raise NormalizationError("missing_task_id", f"{source} row has no stable source ID")
    return normalize_identifier(value)


def _statement(source: str, row: Mapping[str, Any]) -> str:
    names = {
        "mbpp": ("prompt", "text", "question"),
        "apps": ("question", "prompt", "description"),
        "codecontests": ("description", "question", "prompt"),
        "taco": ("question", "description", "prompt"),
        "xcodeeval": ("description", "problem_description", "question", "prompt"),
    }[source]
    value = _first(row, names)
    if value is None or not str(value).strip():
        raise NormalizationError("missing_statement", f"{source} row has no problem statement")
    return str(value).strip()


def _solution_candidates(value: Any) -> List[Tuple[Optional[Any], str]]:
    value = _jsonish(value)
    if value is None:
        return []
    if isinstance(value, str):
        return [(None, value)]
    if isinstance(value, Mapping):
        languages = _as_list(_first(value, ("language", "languages", "lang")))
        solutions = _as_list(_first(value, ("solution", "solutions", "code", "programs")))
        if solutions:
            if languages and len(languages) == len(solutions):
                return [(language, str(solution)) for language, solution in zip(languages, solutions)]
            return [(None, str(solution)) for solution in solutions]
        candidate = _first(value, ("canonical_solution", "source_code"))
        return [] if candidate is None else [(value.get("language"), str(candidate))]
    candidates = []
    for item in _as_list(value):
        if isinstance(item, Mapping):
            solution = _first(item, ("solution", "code", "program", "source_code"))
            if solution is not None:
                candidates.append((_first(item, ("language", "lang")), str(solution)))
        elif item is not None:
            candidates.append((None, str(item)))
    return candidates


def _is_python(language: Any) -> bool:
    if language is None:
        return True
    if isinstance(language, int):
        return language == 3
    value = str(language).strip().casefold().replace("_", "").replace("-", "")
    return value in {"3", "python", "python3", "py", "pypy", "pypy3"}


def _reference_solution(source: str, row: Mapping[str, Any]) -> str:
    if source == "mbpp":
        raw = _first(row, ("code", "canonical_solution", "solution"))
    elif source in {"apps", "taco"}:
        raw = _first(row, ("solutions", "solution", "code"))
    elif source == "codecontests":
        raw = _first(row, ("solutions", "solution", "code"))
    else:
        raw = _first(row, ("solution", "code", "source_code", "canonical_solution", "solutions"))
    candidates = _solution_candidates(raw)
    python = [solution for language, solution in candidates if _is_python(language) and solution.strip()]
    if not python:
        raise NormalizationError("missing_python_solution", f"{source} row has no Python reference solution")
    return python[0].strip()


def _tests(source: str, row: Mapping[str, Any]) -> Tuple[Dict[str, Any], ...]:
    if source == "mbpp":
        suites = [
            ("standard", row.get("test_list")),
            ("challenge", row.get("challenge_test_list")),
            ("tests", row.get("tests")),
        ]
        tests = merge_test_suites(suites, default_kind="assert")
    elif source in {"apps", "taco"}:
        tests = merge_test_suites(
            [
                ("official", row.get("input_output")),
                ("tests", row.get("tests")),
                ("unit_tests", row.get("unit_tests")),
            ]
        )
    elif source == "codecontests":
        tests = merge_test_suites(
            [
                ("public", row.get("public_tests")),
                ("private", row.get("private_tests")),
                ("generated", row.get("generated_tests")),
                ("tests", row.get("tests")),
            ]
        )
    else:
        tests = merge_test_suites(
            [
                ("unit_tests", _first(row, ("unit_tests", "unittest", "unittests"))),
                ("input_output", row.get("input_output")),
                ("tests", _first(row, ("tests", "test_cases"))),
            ]
        )
    if not tests:
        raise NormalizationError("missing_tests", f"{source} row has no executable tests")
    return tests


@dataclass(frozen=True)
class CodeTask:
    uid: str
    source: str
    split: str
    task_id: str
    statement: str
    normalized_statement: str
    reference_solution: str
    verifier: str
    tests: Tuple[Dict[str, Any], ...]
    test_setup: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        return self.statement

    @property
    def code(self) -> str:
        return self.reference_solution

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["tests"] = [dict(test) for test in self.tests]
        return value


def normalize_row(source: str, split: str, row: Mapping[str, Any], variant: Optional[str] = None) -> CodeTask:
    canonical, normalized_split = require_training_split(source, split)
    task_id = _task_id(canonical, row)
    statement = _statement(canonical, row)
    tests = _tests(canonical, row)
    kinds = {str(test["kind"]) for test in tests}
    if kinds == {"assert"}:
        verifier = "assert"
    elif kinds == {"function"}:
        verifier = "function"
    elif kinds == {"stdin"}:
        verifier = "stdin"
    else:
        raise NormalizationError("mixed_test_protocol", f"incompatible test kinds: {sorted(kinds)}")
    metadata = {
        key: row[key]
        for key in ("difficulty", "skill_types", "tags", "source", "url", "name")
        if row.get(key) not in (None, "")
    }
    if variant:
        metadata["variant"] = variant
    return CodeTask(
        uid=f"{canonical}:{task_id}",
        source=canonical,
        split=normalized_split,
        task_id=task_id,
        statement=statement,
        normalized_statement=normalize_statement(statement),
        reference_solution=_reference_solution(canonical, row),
        verifier=verifier,
        tests=tests,
        test_setup=str(_first(row, ("test_setup_code", "test_imports", "setup")) or ""),
        metadata=metadata,
    )


@dataclass(frozen=True)
class VerificationResult:
    """Per-test outcomes returned by a sandboxed reference verifier."""

    outcomes: Mapping[str, bool]
    errors: Mapping[str, str] = field(default_factory=dict)


class ReferenceVerifier(Protocol):
    def verify(self, task: CodeTask) -> VerificationResult:
        """Execute ``task.reference_solution`` against every test in ``task.tests``."""


VerifierLike = Union[ReferenceVerifier, Callable[[CodeTask], VerificationResult]]


def require_reference_solution_passes(task: CodeTask, verifier: VerifierLike) -> VerificationResult:
    if hasattr(verifier, "verify"):
        result = verifier.verify(task)  # type: ignore[attr-defined]
    else:
        result = verifier(task)  # type: ignore[operator]
    if not isinstance(result, VerificationResult):
        raise TypeError("reference verifier must return VerificationResult")

    expected = {str(test["test_id"]) for test in task.tests}
    observed = {str(test_id) for test_id in result.outcomes}
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ReferenceVerificationError(
            "verification_incomplete",
            f"verifier coverage mismatch; missing={missing}, unexpected={unexpected}",
        )
    failed = sorted(test_id for test_id, passed in result.outcomes.items() if not bool(passed))
    if failed:
        raise ReferenceVerificationError(
            "reference_failed", f"reference solution failed tests: {failed}"
        )
    return result


@dataclass(frozen=True)
class SourceRows:
    source: str
    split: str
    rows: Iterable[Mapping[str, Any]]
    variant: Optional[str] = None


@dataclass
class _SourceStats:
    source: str
    splits: Set[str] = field(default_factory=set)
    available_rows: Optional[int] = 0
    rows_seen: int = 0
    normalized: int = 0
    filtered: int = 0
    duplicate_source_id: int = 0
    duplicate_statement: int = 0
    accepted: int = 0
    filter_reasons: Counter = field(default_factory=Counter)
    pilot_cap: Optional[int] = None
    cap_reached: bool = False
    input_exhausted: bool = True

    def add_available(self, rows: Iterable[Mapping[str, Any]]) -> None:
        if self.available_rows is None:
            return
        try:
            self.available_rows += len(rows)  # type: ignore[arg-type]
        except TypeError:
            self.available_rows = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "splits": sorted(self.splits),
            "available_rows": self.available_rows,
            "rows_seen": self.rows_seen,
            "normalized": self.normalized,
            "filtered": self.filtered,
            "deduplicated": self.duplicate_source_id + self.duplicate_statement,
            "duplicate_source_id": self.duplicate_source_id,
            "duplicate_statement": self.duplicate_statement,
            "accepted": self.accepted,
            "filter_reasons": dict(sorted(self.filter_reasons.items())),
            "pilot_cap": self.pilot_cap,
            "cap_reached": self.cap_reached,
            "input_exhausted": self.input_exhausted,
        }


@dataclass(frozen=True)
class BuildResult:
    tasks: Tuple[CodeTask, ...]
    manifest: Dict[str, Any]

    def write_manifest(self, path: Union[str, Path]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )


PilotCap = Optional[Union[int, Mapping[str, int]]]


def _normalized_caps(pilot_cap: PilotCap) -> Dict[str, int]:
    if pilot_cap is None:
        return {}
    if isinstance(pilot_cap, int):
        if pilot_cap <= 0:
            raise ValueError("pilot_cap must be a positive integer or None for a full build")
        return {source: pilot_cap for source in SOURCE_POLICIES}
    caps: Dict[str, int] = {}
    for source, cap in pilot_cap.items():
        canonical = canonical_source(source)
        if not isinstance(cap, int) or cap <= 0:
            raise ValueError(f"pilot cap for {canonical} must be a positive integer")
        caps[canonical] = cap
    return caps


def build_code_pool(
    sources: Sequence[SourceRows],
    verifier: VerifierLike,
    pilot_cap: PilotCap = None,
) -> BuildResult:
    """Build a verified, deduplicated code-training pool.

    ``pilot_cap=None`` consumes every input row.  A positive integer applies the
    same accepted-task cap to each source; a mapping supplies source-specific
    caps.  For streaming inputs, construction stops as soon as a source reaches
    its cap and marks ``input_exhausted=false`` in the manifest.
    """

    if verifier is None:
        raise ValueError("a sandboxed reference verifier is required")
    caps = _normalized_caps(pilot_cap)
    tasks: List[CodeTask] = []
    accepted_uids: Set[str] = set()
    accepted_statements: Set[str] = set()
    stats: Dict[str, _SourceStats] = {}

    for source_rows in sources:
        source, split = require_training_split(source_rows.source, source_rows.split)
        source_stats = stats.setdefault(source, _SourceStats(source=source))
        source_stats.splits.add(split)
        source_stats.add_available(source_rows.rows)
        source_stats.pilot_cap = caps.get(source)

        cap = caps.get(source)
        if cap is not None and source_stats.accepted >= cap:
            source_stats.cap_reached = True
            source_stats.input_exhausted = False
            continue

        exhausted = True
        for row in source_rows.rows:
            if cap is not None and source_stats.accepted >= cap:
                source_stats.cap_reached = True
                exhausted = False
                break
            source_stats.rows_seen += 1
            try:
                task = normalize_row(source, split, row, variant=source_rows.variant)
                source_stats.normalized += 1
                require_reference_solution_passes(task, verifier)
            except (NormalizationError, ReferenceVerificationError) as exc:
                source_stats.filtered += 1
                source_stats.filter_reasons[exc.reason] += 1
                continue

            if task.uid in accepted_uids:
                source_stats.duplicate_source_id += 1
                continue
            if task.normalized_statement in accepted_statements:
                source_stats.duplicate_statement += 1
                continue

            accepted_uids.add(task.uid)
            accepted_statements.add(task.normalized_statement)
            tasks.append(task)
            source_stats.accepted += 1
        source_stats.input_exhausted = source_stats.input_exhausted and exhausted

    source_manifest = {source: stats[source].to_dict() for source in sorted(stats)}
    totals = {
        "available_rows": (
            sum(item.available_rows or 0 for item in stats.values())
            if all(item.available_rows is not None for item in stats.values())
            else None
        ),
        "rows_seen": sum(item.rows_seen for item in stats.values()),
        "normalized": sum(item.normalized for item in stats.values()),
        "filtered": sum(item.filtered for item in stats.values()),
        "deduplicated": sum(
            item.duplicate_source_id + item.duplicate_statement for item in stats.values()
        ),
        "accepted": len(tasks),
    }
    manifest = {
        "schema_version": 1,
        "mode": "full" if pilot_cap is None else "pilot",
        "verification": "reference solution passed every merged test",
        "deduplication": ["canonical source task ID", "normalized problem statement"],
        "totals": totals,
        "sources": source_manifest,
    }
    return BuildResult(tasks=tuple(tasks), manifest=manifest)
