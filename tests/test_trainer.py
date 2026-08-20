from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from antgpu_run.trainer import TrainerError, invoke_trainer, render_config, write_config
from antgpu_run.workflow import load_preset, resolve_params

PRESET = Path(__file__).resolve().parents[1] / "presets" / "train-pairs-v1.yaml"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith("ANTGPU_RUN_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def context():
    preset = load_preset(PRESET)
    return preset, resolve_params(preset)


def test_full_run_config_uses_the_named_result_dir(context, monkeypatch):
    preset, _ = context
    monkeypatch.setenv("ANTGPU_RUN_RESULT_NAME", "my-adapter")
    params = resolve_params(preset)
    config = render_config(preset, params, test_run=False)
    assert config["output_dir"].endswith("/my-adapter")
    assert "max_steps" not in config


def test_test_run_config_is_bounded(context):
    preset, params = context
    config = render_config(preset, params, test_run=True)
    assert config["max_steps"] == 2
    assert config["save_total_limit"] == 1
    assert config["output_dir"].endswith("/test-run")


def test_test_run_and_full_run_write_to_different_places(context):
    preset, params = context
    assert (
        render_config(preset, params, test_run=True)["output_dir"]
        != render_config(preset, params, test_run=False)["output_dir"]
    )


def test_rendered_config_matches_the_trainer_schema(context):
    """The runner and the trainer are separate packages; this pins the contract."""
    trainer_config = pytest.importorskip("antgpu_train.config")
    preset, params = context
    for test_run in (True, False):
        trainer_config.TrainingConfig.model_validate(render_config(preset, params, test_run=test_run))


def test_write_config_round_trips(context, tmp_path):
    preset, params = context
    config = render_config(preset, params, test_run=False)
    path = write_config(config, tmp_path / "nested" / "config.yaml")
    assert yaml.safe_load(path.read_text()) == config


def test_missing_trainer_binary_is_explained(monkeypatch, tmp_path):
    monkeypatch.setattr("antgpu_run.trainer.TRAINER_BINARY", "antgpu-train-does-not-exist")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    with pytest.raises(TrainerError, match="built on top of the trainer image"):
        invoke_trainer(config_path)
