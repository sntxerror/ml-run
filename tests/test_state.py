from __future__ import annotations

import pytest

from antgpu_run.state import PREREQUISITE, RunState, Stage, StageBlocked, Status


def test_fresh_run_starts_at_dataset(tmp_path):
    state = RunState(tmp_path)
    assert "dataset ingest" in state.next_action()
    for stage in Stage:
        assert state.read(stage).status is Status.PENDING


def test_every_stage_is_gated_on_its_prerequisite(tmp_path):
    state = RunState(tmp_path)
    for stage, prerequisite in PREREQUISITE.items():
        if prerequisite is None:
            state.require_ready(stage)
            continue
        with pytest.raises(StageBlocked):
            state.require_ready(stage)


def test_full_run_requires_an_approved_test_run(tmp_path):
    state = RunState(tmp_path)
    state.write(Stage.DATASET, Status.OK)
    state.write(Stage.VALIDATE, Status.OK)
    state.write(Stage.TEST_RUN, Status.OK)

    # A successful test run is not enough on its own.
    with pytest.raises(StageBlocked):
        state.require_ready(Stage.RUN)

    state.write(Stage.APPROVAL, Status.OK)
    state.require_ready(Stage.RUN)


def test_failed_prerequisite_blocks_and_names_the_command(tmp_path):
    state = RunState(tmp_path)
    state.write(Stage.DATASET, Status.FAILED, error="no train.jsonl")
    with pytest.raises(StageBlocked) as excinfo:
        state.require_ready(Stage.VALIDATE)
    assert "antgpu-run dataset ingest" in str(excinfo.value)


def test_summary_round_trips(tmp_path):
    state = RunState(tmp_path)
    state.write(Stage.VALIDATE, Status.OK, train_examples=42, message="ok")
    reloaded = state.read(Stage.VALIDATE)
    assert reloaded.ok
    assert reloaded.detail["train_examples"] == 42
    assert reloaded.updated_at


def test_unreadable_summary_reports_failure_rather_than_crashing(tmp_path):
    state = RunState(tmp_path)
    path = state.summary_path(Stage.RUN)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert state.read(Stage.RUN).status is Status.FAILED


def test_next_action_walks_forward(tmp_path):
    state = RunState(tmp_path)
    state.write(Stage.DATASET, Status.OK)
    assert "validate" in state.next_action()
    state.write(Stage.VALIDATE, Status.OK)
    assert "test-run" in state.next_action()
    state.write(Stage.TEST_RUN, Status.OK)
    assert "approve" in state.next_action()
    state.write(Stage.APPROVAL, Status.OK)
    assert "run" in state.next_action()
    state.write(Stage.RUN, Status.OK)
    assert state.next_action() == "run complete"
