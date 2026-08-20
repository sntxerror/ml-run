"""Run state machine.

State is a set of per-stage ``summary.json`` files under the run directory, the
same shape ``antgpu-ml`` uses for its pipeline stages. There is no daemon and no
database: the files are the state. That keeps the run inspectable over SSH with
nothing but ``cat``, and lets a web view read exactly what the CLI prints.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class Stage(StrEnum):
    DATASET = "dataset"
    VALIDATE = "validate"
    TEST_RUN = "test-run"
    APPROVAL = "approval"
    RUN = "run"


class Status(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    OK = "ok"
    FAILED = "failed"


#: Each stage may only start once its prerequisite has succeeded. This is what
#: makes "no full run before an approved test run" a property of the runner
#: rather than a convention.
PREREQUISITE: dict[Stage, Stage | None] = {
    Stage.DATASET: None,
    Stage.VALIDATE: Stage.DATASET,
    Stage.TEST_RUN: Stage.VALIDATE,
    Stage.APPROVAL: Stage.TEST_RUN,
    Stage.RUN: Stage.APPROVAL,
}

#: The CLI command that advances each stage. Error messages must name a command
#: the user can actually type, not the internal stage name.
STAGE_COMMAND: dict[Stage, str] = {
    Stage.DATASET: "dataset ingest",
    Stage.VALIDATE: "validate",
    Stage.TEST_RUN: "test-run",
    Stage.APPROVAL: "approve",
    Stage.RUN: "run",
}

STAGE_ORDER: tuple[Stage, ...] = (
    Stage.DATASET,
    Stage.VALIDATE,
    Stage.TEST_RUN,
    Stage.APPROVAL,
    Stage.RUN,
)


class StageBlocked(Exception):
    """Raised when a stage is attempted before its prerequisite succeeded."""


@dataclass(frozen=True)
class StageSummary:
    stage: Stage
    status: Status
    updated_at: str
    detail: dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.status is Status.OK

    def to_json(self) -> dict[str, Any]:
        return {
            "stage": str(self.stage),
            "status": str(self.status),
            "updated_at": self.updated_at,
            "detail": self.detail,
        }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write_json_atomic(path: Path, value: object) -> None:
    """Write JSON so a reader never observes a partial file.

    The web view polls these files while a run is writing them, so a torn read
    would surface as a spurious failure.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            json.dump(value, tmp, indent=2, sort_keys=True)
            tmp.write("\n")
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


class RunState:
    """Reads and writes stage summaries under ``run_dir``."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def summary_path(self, stage: Stage) -> Path:
        return self.run_dir / str(stage) / "summary.json"

    def log_path(self, stage: Stage) -> Path:
        return self.run_dir / str(stage) / "output.log"

    def read(self, stage: Stage) -> StageSummary:
        path = self.summary_path(stage)
        if not path.is_file():
            return StageSummary(stage, Status.PENDING, "", {})
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return StageSummary(stage, Status.FAILED, "", {"error": f"unreadable summary: {exc}"})
        try:
            status = Status(raw.get("status", "pending"))
        except ValueError:
            status = Status.FAILED
        detail = raw.get("detail")
        return StageSummary(
            stage=stage,
            status=status,
            updated_at=str(raw.get("updated_at", "")),
            detail=detail if isinstance(detail, dict) else {},
        )

    def write(self, stage: Stage, status: Status, **detail: Any) -> StageSummary:
        summary = StageSummary(stage, status, _utc_now(), dict(detail))
        _write_json_atomic(self.summary_path(stage), summary.to_json())
        return summary

    def all_summaries(self) -> list[StageSummary]:
        return [self.read(stage) for stage in STAGE_ORDER]

    def require_ready(self, stage: Stage) -> None:
        """Raise unless ``stage`` may start now."""
        prerequisite = PREREQUISITE[stage]
        if prerequisite is None:
            return
        previous = self.read(prerequisite)
        if previous.ok:
            return
        command = STAGE_COMMAND[prerequisite]
        if previous.status is Status.PENDING:
            raise StageBlocked(
                f"cannot start {stage}: {prerequisite} has not run yet. Run 'antgpu-run {command}' first."
            )
        raise StageBlocked(
            f"cannot start {stage}: {prerequisite} is {previous.status}. "
            f"Fix it and re-run 'antgpu-run {command}'."
        )

    def next_action(self) -> str:
        """The single next thing the user should do."""
        for stage in STAGE_ORDER:
            summary = self.read(stage)
            if summary.ok:
                continue
            if summary.status is Status.RUNNING:
                return f"{stage} is running"
            if summary.status is Status.FAILED:
                return f"fix {stage}, then re-run 'antgpu-run {STAGE_COMMAND[stage]}'"
            if stage is Stage.APPROVAL:
                return "review results with 'antgpu-run results', then run 'antgpu-run approve'"
            return f"run 'antgpu-run {STAGE_COMMAND[stage]}'"
        return "run complete"
