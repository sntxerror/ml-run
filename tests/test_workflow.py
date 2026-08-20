from __future__ import annotations

from pathlib import Path

import pytest

from antgpu_run.workflow import WorkflowError, load_preset, resolve_params

PRESET = Path(__file__).resolve().parents[1] / "presets" / "train-pairs-v1.yaml"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith("ANTGPU_RUN_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def preset():
    return load_preset(PRESET)


def test_defaults_resolve(preset):
    params = resolve_params(preset)
    assert params.training_mode == "balanced"
    assert params.run_mode == "interactive_sandbox"
    assert params.settings["method"] == "qlora"


def test_model_must_come_from_the_catalog(preset, monkeypatch):
    monkeypatch.setenv("ANTGPU_RUN_BASE_MODEL", "attacker/backdoored-model")
    with pytest.raises(WorkflowError, match="not in this preset's catalog"):
        resolve_params(preset)


def test_unknown_training_mode_is_rejected(preset, monkeypatch):
    monkeypatch.setenv("ANTGPU_RUN_TRAINING_MODE", "turbo")
    with pytest.raises(WorkflowError, match="not defined"):
        resolve_params(preset)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("ANTGPU_RUN_EPOCHS", "99"),
        ("ANTGPU_RUN_EPOCHS", "0"),
        ("ANTGPU_RUN_MAX_SEQUENCE_LENGTH", "999999"),
        ("ANTGPU_RUN_PER_DEVICE_BATCH_SIZE", "1024"),
        ("ANTGPU_RUN_LEARNING_RATE", "5"),
    ],
)
def test_values_outside_preset_bounds_are_rejected(preset, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(WorkflowError):
        resolve_params(preset)


def test_non_numeric_override_is_rejected(preset, monkeypatch):
    monkeypatch.setenv("ANTGPU_RUN_EPOCHS", "lots")
    with pytest.raises(WorkflowError, match="not a valid float"):
        resolve_params(preset)


def test_in_bounds_override_is_applied(preset, monkeypatch):
    monkeypatch.setenv("ANTGPU_RUN_EPOCHS", "3")
    assert resolve_params(preset).settings["epochs"] == 3.0


def test_unavailable_run_modes_are_refused_with_a_useful_message(preset, monkeypatch):
    for mode in ("keepalive_detached", "job_no_shell"):
        monkeypatch.setenv("ANTGPU_RUN_RUN_MODE", mode)
        with pytest.raises(WorkflowError, match="planned but not yet available"):
            resolve_params(preset)


def test_missing_preset_is_a_clear_error(tmp_path):
    with pytest.raises(WorkflowError, match="not found"):
        load_preset(tmp_path / "nope.yaml")


def test_smoke_only_model_is_never_selected_by_omission(preset):
    smoke_only = [m["id"] for m in preset.base_models if m.get("smoke_only")]
    if not smoke_only:
        pytest.skip("preset declares no smoke-only models")
    chosen = resolve_params(preset).base_model
    assert chosen not in smoke_only, "a smoke-only model must not be the default"


def test_smoke_only_model_remains_explicitly_selectable(preset, monkeypatch):
    smoke_only = [m["id"] for m in preset.base_models if m.get("smoke_only")]
    if not smoke_only:
        pytest.skip("preset declares no smoke-only models")
    monkeypatch.setenv("ANTGPU_RUN_BASE_MODEL", smoke_only[0])
    assert resolve_params(preset).base_model == smoke_only[0]
