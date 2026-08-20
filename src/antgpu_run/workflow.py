"""Workflow preset loading and parameter resolution.

A preset is the ``action.yml`` to a template's runner image: it declares which
models may be chosen, which training modes exist, and the bounds any supplied
value must fall inside. The campaign creator picks names; the preset turns those
into trainer settings.

Parameters arrive as ``ANTGPU_RUN_*`` environment variables set by the control
plane at launch. Nothing here trusts them: every value is checked against the
preset before it reaches a config file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ENV_PREFIX = "ANTGPU_RUN_"

#: Baked into the runner image so the container is self-describing. The control
#: plane holds the same preset; a mismatch is a deployment bug worth surfacing.
DEFAULT_PRESET_PATH = Path("/opt/antgpu/workflow.yaml")


class WorkflowError(Exception):
    """Configuration or parameter problem. Always the user's or operator's."""


@dataclass(frozen=True)
class Paths:
    dataset_dir: Path
    output_dir: Path
    run_dir: Path
    cache_dir: Path


@dataclass(frozen=True)
class Preset:
    kind: str
    name: str
    raw: dict[str, Any] = field(repr=False)

    @property
    def paths(self) -> Paths:
        p = self.raw.get("paths", {})
        return Paths(
            dataset_dir=Path(p.get("dataset_dir", "/workspace/input")),
            output_dir=Path(p.get("output_dir", "/workspace/output")),
            run_dir=Path(p.get("run_dir", "/workspace/run")),
            cache_dir=Path(p.get("cache_dir", "/cache/huggingface")),
        )

    @property
    def base_models(self) -> list[dict[str, Any]]:
        return list(self.raw.get("base_models", []))

    @property
    def training_modes(self) -> dict[str, Any]:
        return dict(self.raw.get("training_modes", {}))

    @property
    def dataset(self) -> dict[str, Any]:
        return dict(self.raw.get("dataset", {}))

    @property
    def test_run(self) -> dict[str, Any]:
        return dict(self.raw.get("test_run", {}))

    @property
    def bounds(self) -> dict[str, Any]:
        return dict(self.raw.get("bounds", {}))

    @property
    def run_modes(self) -> dict[str, Any]:
        return dict(self.raw.get("run_modes", {}))


def load_preset(path: Path | None = None) -> Preset:
    preset_path = path or Path(os.environ.get(f"{ENV_PREFIX}PRESET", DEFAULT_PRESET_PATH))
    if not preset_path.is_file():
        raise WorkflowError(
            f"workflow preset not found at {preset_path}. "
            "This image was built without a preset, or the template points at the wrong path."
        )
    try:
        raw = yaml.safe_load(preset_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise WorkflowError(f"cannot read workflow preset {preset_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise WorkflowError(f"workflow preset {preset_path} must be a mapping")
    kind = str(raw.get("kind", "")).strip()
    if not kind:
        raise WorkflowError(f"workflow preset {preset_path} has no 'kind'")
    return Preset(kind=kind, name=str(raw.get("name", kind)), raw=raw)


def _env(name: str) -> str | None:
    value = os.environ.get(f"{ENV_PREFIX}{name}")
    if value is None:
        return None
    value = value.strip()
    return value or None


def _check_bound(preset: Preset, key: str, value: float) -> float:
    bound = preset.bounds.get(key)
    if not isinstance(bound, dict):
        return value
    low, high = bound.get("min"), bound.get("max")
    if low is not None and value < low:
        raise WorkflowError(f"{key}={value} is below the preset minimum {low}")
    if high is not None and value > high:
        raise WorkflowError(f"{key}={value} is above the preset maximum {high}")
    return value


@dataclass(frozen=True)
class Params:
    """Resolved, validated launch parameters."""

    base_model: str
    training_mode: str
    run_mode: str
    result_name: str
    settings: dict[str, Any]


def resolve_params(preset: Preset) -> Params:
    models = preset.base_models
    if not models:
        raise WorkflowError("workflow preset declares no base_models")

    model_ids = [str(m.get("id")) for m in models]
    # Smoke-only models verify the pipeline; they must never be selected by
    # omission, only by an explicit choice.
    selectable = [str(m.get("id")) for m in models if not m.get("smoke_only")]
    base_model = _env("BASE_MODEL") or (selectable[0] if selectable else "")
    if not base_model:
        raise WorkflowError("workflow preset has no selectable base model")
    if base_model not in model_ids:
        raise WorkflowError(
            f"base_model {base_model!r} is not in this preset's catalog. Allowed: {', '.join(model_ids)}"
        )
    if base_model.endswith(".gguf"):
        raise WorkflowError("GGUF checkpoints cannot be fine-tuned; choose a full Transformers model")

    modes = preset.training_modes
    training_mode = _env("TRAINING_MODE") or str(preset.raw.get("default_training_mode", ""))
    if training_mode not in modes:
        raise WorkflowError(
            f"training_mode {training_mode!r} is not defined. Allowed: {', '.join(sorted(modes))}"
        )

    run_modes = preset.run_modes
    run_mode = _env("RUN_MODE") or str(preset.raw.get("default_run_mode", ""))
    mode_spec = run_modes.get(run_mode)
    if not isinstance(mode_spec, dict):
        raise WorkflowError(
            f"run_mode {run_mode!r} is not defined. Allowed: {', '.join(sorted(run_modes))}"
        )
    if not mode_spec.get("available", False):
        available = sorted(k for k, v in run_modes.items() if isinstance(v, dict) and v.get("available"))
        raise WorkflowError(
            f"run mode {run_mode!r} is planned but not yet available. Currently available: {', '.join(available)}"
        )

    settings = dict(modes[training_mode])
    settings.pop("label", None)

    # Per-launch numeric overrides, each checked against the preset's bounds.
    for key, caster in (
        ("epochs", float),
        ("max_sequence_length", int),
        ("per_device_batch_size", int),
        ("gradient_accumulation_steps", int),
        ("learning_rate", float),
    ):
        raw_value = _env(key.upper())
        if raw_value is None:
            continue
        try:
            value = caster(raw_value)
        except ValueError as exc:
            raise WorkflowError(f"{key}={raw_value!r} is not a valid {caster.__name__}") from exc
        settings[key] = _check_bound(preset, key, value)

    return Params(
        base_model=base_model,
        training_mode=training_mode,
        run_mode=run_mode,
        result_name=_env("RESULT_NAME") or "adapter",
        settings=settings,
    )
