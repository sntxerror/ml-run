"""Delegation to ``antgpu-train``.

The runner never imports the trainer. It renders a config and invokes the CLI as
a subprocess, so the trainer stays a portable container contract and the runner
stays thin enough to sit in front of a different tool for another workflow kind.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .workflow import Params, Preset

TRAINER_BINARY = "antgpu-train"


class TrainerError(Exception):
    pass


@dataclass(frozen=True)
class TrainerResult:
    exit_code: int
    payload: dict[str, Any]
    stderr_tail: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


def render_config(preset: Preset, params: Params, *, test_run: bool) -> dict[str, Any]:
    """Build the trainer config. This is the YAML the user never sees or edits."""
    paths = preset.paths
    settings = dict(params.settings)
    lora = dict(settings.pop("lora", {}) or {})

    config: dict[str, Any] = {
        "schema_version": 1,
        "base_model": params.base_model,
        "dataset_dir": str(paths.dataset_dir),
        "output_dir": str(paths.output_dir / ("test-run" if test_run else params.result_name)),
        "method": settings.get("method", "qlora"),
        "max_sequence_length": int(settings.get("max_sequence_length", 2048)),
        "epochs": float(settings.get("epochs", 1)),
        "per_device_batch_size": int(settings.get("per_device_batch_size", 1)),
        "gradient_accumulation_steps": int(settings.get("gradient_accumulation_steps", 4)),
        "learning_rate": float(settings.get("learning_rate", 2e-4)),
    }
    if lora:
        config["lora"] = lora

    if test_run:
        # A test run must be cheap and conclusive: prove the model loads, the
        # dataset tokenizes, one step runs and a checkpoint writes.
        spec = preset.test_run
        config["max_steps"] = int(spec.get("max_steps", 2))
        config["save_steps"] = int(spec.get("save_steps", 1))
        config["save_total_limit"] = int(spec.get("save_total_limit", 1))
        config["epochs"] = float(spec.get("epochs", 1))

    return config


def write_config(config: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return path


def invoke_trainer(
    config_path: Path,
    *,
    dry_run: bool = False,
    resume: bool = False,
    log_path: Path | None = None,
) -> TrainerResult:
    """Run the trainer, streaming its progress to ``log_path``.

    The trainer writes progress to stderr and a single JSON result to stdout.
    """
    command = [TRAINER_BINARY, "train", "--config", str(config_path)]
    if dry_run:
        command.append("--dry-run")
    if resume:
        command.append("--resume")

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise TrainerError(
            f"{TRAINER_BINARY} is not installed in this image. "
            "The runner image must be built on top of the trainer image."
        ) from exc

    if log_path is not None:
        log_path.write_text(
            f"$ {' '.join(command)}\n\n{completed.stderr}\n{completed.stdout}\n",
            encoding="utf-8",
        )

    payload: dict[str, Any] = {}
    stdout = completed.stdout.strip()
    if stdout:
        try:
            parsed = json.loads(stdout.splitlines()[-1])
            if isinstance(parsed, dict):
                payload = parsed
        except json.JSONDecodeError:
            payload = {"raw_stdout": stdout[-2000:]}

    stderr_tail = "\n".join(completed.stderr.strip().splitlines()[-20:])
    return TrainerResult(exit_code=completed.returncode, payload=payload, stderr_tail=stderr_tail)
