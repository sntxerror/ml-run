from __future__ import annotations

import json

import pytest

from antgpu_run.dataset import inspect_dataset

SPEC = {"train_file": "train.jsonl", "validation_file": "validation.jsonl", "min_examples": 2}


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def pair(i=0):
    return {"messages": [
        {"role": "user", "content": f"question {i}"},
        {"role": "assistant", "content": f"answer {i}"},
    ]}


def test_missing_dataset_explains_where_to_put_it(tmp_path):
    report = inspect_dataset(tmp_path, SPEC)
    assert not report.ok
    assert any("train.jsonl not found" in p for p in report.problems)
    assert any(str(tmp_path) in p for p in report.problems)


def test_well_formed_dataset_passes(tmp_path):
    write_jsonl(tmp_path / "train.jsonl", [pair(i) for i in range(4)])
    report = inspect_dataset(tmp_path, SPEC)
    assert report.ok, report.problems
    assert report.train.lines == 4
    assert not report.validation.exists


def test_optional_validation_file_is_checked_when_present(tmp_path):
    write_jsonl(tmp_path / "train.jsonl", [pair(i) for i in range(4)])
    (tmp_path / "validation.jsonl").write_text("{ broken\n", encoding="utf-8")
    report = inspect_dataset(tmp_path, SPEC)
    assert not report.ok
    assert any("not valid JSON" in p for p in report.problems)


def test_too_few_examples_is_reported(tmp_path):
    write_jsonl(tmp_path / "train.jsonl", [pair(0)])
    report = inspect_dataset(tmp_path, SPEC)
    assert any("needs at least 2" in p for p in report.problems)


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"messages": []}, "non-empty 'messages'"),
        ({"nope": 1}, "non-empty 'messages'"),
        ({"messages": [{"role": "user"}]}, "'role' and 'content'"),
        ({"messages": [{"role": "user", "content": "q"}]}, "role 'assistant'"),
        (
            {"messages": [{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}]},
            "role 'user'",
        ),
    ],
)
def test_malformed_records_are_described_precisely(tmp_path, record, expected):
    write_jsonl(tmp_path / "train.jsonl", [record, pair(1)])
    report = inspect_dataset(tmp_path, SPEC)
    assert any(expected in p for p in report.problems), report.problems


def test_binary_file_is_reported_as_not_jsonl(tmp_path):
    (tmp_path / "train.jsonl").write_bytes(b"\x28\xb5\x2f\xfd\x00\x58\x25\x00\x00")
    report = inspect_dataset(tmp_path, SPEC)
    assert any("not valid UTF-8" in p for p in report.problems)
