"""Dataset presence and shape checks.

Deliberately shallow. Deep validation belongs to ``antgpu-train --dry-run``,
which parses every record with the trainer's own pydantic model — the only
check that actually predicts whether training will start. This module answers
the cheaper question the user asks first: "is my data here, and does it look
like the right thing?"
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Reading every line of a large dataset just to describe it is wasteful; the
#: authoritative pass happens in validate.
SHAPE_SAMPLE_LINES = 200


@dataclass(frozen=True)
class FileReport:
    path: Path
    exists: bool
    lines: int = 0
    bytes: int = 0
    problems: list[str] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "exists": self.exists,
            "lines": self.lines,
            "bytes": self.bytes,
            "problems": list(self.problems or []),
        }


@dataclass(frozen=True)
class DatasetReport:
    dataset_dir: Path
    train: FileReport
    validation: FileReport
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_json(self) -> dict[str, Any]:
        return {
            "dataset_dir": str(self.dataset_dir),
            "train": self.train.to_json(),
            "validation": self.validation.to_json(),
            "problems": list(self.problems),
        }


def _inspect(path: Path, *, required: bool) -> FileReport:
    if not path.is_file():
        return FileReport(path=path, exists=False)

    problems: list[str] = []
    lines = 0
    checked = 0
    try:
        size = path.stat().st_size
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                lines += 1
                if checked >= SHAPE_SAMPLE_LINES:
                    continue
                checked += 1
                problem = _record_problem(line, number)
                if problem:
                    problems.append(problem)
    except OSError as exc:
        return FileReport(path=path, exists=True, problems=[f"cannot read: {exc}"])
    except UnicodeDecodeError:
        return FileReport(
            path=path,
            exists=True,
            problems=["file is not valid UTF-8 text; it may be a compressed archive rather than JSONL"],
        )

    if required and lines == 0:
        problems.append("file is empty")
    return FileReport(path=path, exists=True, lines=lines, bytes=size, problems=problems)


def _record_problem(line: str, number: int) -> str | None:
    """Return a human-readable problem with one JSONL record, or None."""
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        return f"line {number}: not valid JSON ({exc.msg})"
    if not isinstance(record, dict):
        return f"line {number}: record must be a JSON object"
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return f"line {number}: missing a non-empty 'messages' list"
    for message in messages:
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            return f"line {number}: every message needs 'role' and 'content'"
    if messages[-1].get("role") != "assistant":
        return f"line {number}: the last message must have role 'assistant'"
    if not any(m.get("role") == "user" for m in messages):
        return f"line {number}: at least one message must have role 'user'"
    return None


def inspect_dataset(dataset_dir: Path, spec: dict[str, Any]) -> DatasetReport:
    train_name = str(spec.get("train_file", "train.jsonl"))
    validation_name = str(spec.get("validation_file", "validation.jsonl"))
    min_examples = int(spec.get("min_examples", 1))

    train = _inspect(dataset_dir / train_name, required=True)
    validation = _inspect(dataset_dir / validation_name, required=False)

    problems: list[str] = []
    if not dataset_dir.is_dir():
        problems.append(f"dataset directory {dataset_dir} does not exist")
    if not train.exists:
        problems.append(
            f"{train_name} not found in {dataset_dir}. Copy it there, e.g. "
            f"scp train.jsonl <this-host>:{dataset_dir}/"
        )
    problems.extend(train.problems or [])
    if validation.exists:
        problems.extend(validation.problems or [])
    if train.exists and train.lines < min_examples:
        problems.append(
            f"{train_name} has {train.lines} example(s); this preset needs at least {min_examples}"
        )

    return DatasetReport(
        dataset_dir=dataset_dir,
        train=train,
        validation=validation,
        problems=problems,
    )
