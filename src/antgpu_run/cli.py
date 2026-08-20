"""``antgpu-run`` command line.

The CLI and the web view are two renderings of the same state files. Nothing
here is authoritative on its own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .dataset import inspect_dataset
from .state import STAGE_COMMAND, RunState, Stage, StageBlocked, Status
from .trainer import TrainerError, invoke_trainer, render_config, write_config
from .workflow import Params, Preset, WorkflowError, load_preset, resolve_params

app = typer.Typer(no_args_is_help=True, help="Run a training workflow: dataset, validate, test, run.")
dataset_app = typer.Typer(no_args_is_help=True, help="Inspect the input dataset.")
app.add_typer(dataset_app, name="dataset")

console = Console()
err_console = Console(stderr=True)

STATUS_STYLE = {
    Status.OK: "green",
    Status.FAILED: "red",
    Status.RUNNING: "yellow",
    Status.PENDING: "dim",
}


def _echo_progress(line: str) -> None:
    """Mirror trainer progress to the terminal as it arrives."""
    console.print(f"[dim]{line}[/dim]", markup=False, highlight=False)


def _effective_model(state: RunState, params: Params) -> tuple[str, bool]:
    """What this run actually used, falling back to what it would use.

    A finished run must not be described by the current environment: env vars
    can change between invocations, and reporting the default over a completed
    run's real model would be a plain lie about what was trained.
    """
    for stage in (Stage.RUN, Stage.TEST_RUN):
        recorded = state.read(stage).detail.get("base_model")
        if recorded:
            return str(recorded), True
    return params.base_model, False


def _fail(message: str) -> None:
    err_console.print(f"[red]error:[/red] {message}")
    raise typer.Exit(code=2)


def _context() -> tuple[Preset, Params, RunState]:
    try:
        preset = load_preset()
        params = resolve_params(preset)
    except WorkflowError as exc:
        _fail(str(exc))
        raise AssertionError  # unreachable; keeps type checkers happy
    return preset, params, RunState(preset.paths.run_dir)


# invoke_without_command is required for --version: without it Typer rejects the
# call for having no subcommand before the callback ever runs.
@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[bool, typer.Option("--version", help="Print version and exit.")] = False,
) -> None:
    if version:
        console.print(__version__)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


@app.command("status")
def status(as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False) -> None:
    """Show where this run is and what to do next."""
    preset, params, state = _context()
    summaries = state.all_summaries()

    model, from_run = _effective_model(state, params)

    if as_json:
        console.print_json(
            data={
                "workflow": preset.kind,
                "base_model": model,
                "base_model_is_from_run": from_run,
                "training_mode": params.training_mode,
                "run_mode": params.run_mode,
                "next_action": state.next_action(),
                "stages": [s.to_json() for s in summaries],
            }
        )
        return

    console.print(
        Panel(
            f"[bold]{preset.name}[/bold]\n"
            f"model: {model}" + ("" if from_run else "  [dim](selected; not yet run)[/dim]") + "\n"
            f"mode:  {params.training_mode}  ({params.run_mode})",
            title="workflow",
            expand=False,
        )
    )

    table = Table(show_header=True, header_style="bold")
    table.add_column("step")
    table.add_column("status")
    table.add_column("detail")
    for summary in summaries:
        detail = summary.detail
        note = detail.get("message") or detail.get("error") or ""
        if not note and summary.ok:
            note = ", ".join(f"{k}={v}" for k, v in list(detail.items())[:3])
        table.add_row(
            STAGE_COMMAND[summary.stage],
            f"[{STATUS_STYLE[summary.status]}]{summary.status}[/]",
            str(note)[:80],
        )
    console.print(table)
    console.print(f"\n[bold]next:[/bold] {state.next_action()}")


@dataset_app.command("status")
def dataset_status(as_json: Annotated[bool, typer.Option("--json")] = False) -> None:
    """Check whether the dataset is present and well-formed."""
    preset, _, _ = _context()
    report = inspect_dataset(preset.paths.dataset_dir, preset.dataset)
    if as_json:
        console.print_json(data=report.to_json())
        raise typer.Exit(code=0 if report.ok else 1)

    console.print(f"dataset directory: {report.dataset_dir}")
    console.print(f"  {preset.dataset.get('train_file', 'train.jsonl')}: "
                  + (f"{report.train.lines} examples" if report.train.exists else "[red]missing[/red]"))
    console.print(f"  {preset.dataset.get('validation_file', 'validation.jsonl')}: "
                  + (f"{report.validation.lines} examples" if report.validation.exists else "[dim]absent (optional)[/dim]"))
    if report.problems:
        console.print("\n[red]problems:[/red]")
        for problem in report.problems:
            console.print(f"  - {problem}")
        raise typer.Exit(code=1)
    console.print("\n[green]dataset looks well-formed[/green]")


@dataset_app.command("ingest")
def dataset_ingest() -> None:
    """Register the dataset for this run once it is in place."""
    preset, _, state = _context()
    preset.paths.dataset_dir.mkdir(parents=True, exist_ok=True)
    report = inspect_dataset(preset.paths.dataset_dir, preset.dataset)

    if not report.ok:
        state.write(Stage.DATASET, Status.FAILED, problems=report.problems, **report.to_json())
        err_console.print("[red]dataset is not usable yet:[/red]")
        for problem in report.problems:
            err_console.print(f"  - {problem}")
        raise typer.Exit(code=1)

    state.write(
        Stage.DATASET,
        Status.OK,
        message=f"{report.train.lines} training examples",
        train_examples=report.train.lines,
        validation_examples=report.validation.lines,
        dataset_dir=str(report.dataset_dir),
    )
    console.print(f"[green]dataset registered[/green]: {report.train.lines} training examples")
    console.print(f"next: {state.next_action()}")


@app.command("validate")
def validate() -> None:
    """Parse every record with the trainer's own loader. No model is downloaded."""
    preset, params, state = _context()
    try:
        state.require_ready(Stage.VALIDATE)
    except StageBlocked as exc:
        _fail(str(exc))

    config_path = write_config(
        render_config(preset, params, test_run=True), preset.paths.run_dir / "validate" / "config.yaml"
    )
    state.write(Stage.VALIDATE, Status.RUNNING, message="parsing dataset")

    console.print("validating dataset against the trainer's schema...")
    try:
        result = invoke_trainer(config_path, dry_run=True, log_path=state.log_path(Stage.VALIDATE))
    except TrainerError as exc:
        state.write(Stage.VALIDATE, Status.FAILED, error=str(exc))
        _fail(str(exc))
        return

    if not result.ok:
        state.write(Stage.VALIDATE, Status.FAILED, error=result.stderr_tail or "validation failed")
        err_console.print(f"[red]validation failed[/red]\n{result.stderr_tail}")
        raise typer.Exit(code=1)

    train_examples = result.payload.get("train_examples", 0)
    state.write(
        Stage.VALIDATE,
        Status.OK,
        message=f"{train_examples} examples parsed",
        **{k: v for k, v in result.payload.items() if k != "dry_run"},
    )
    console.print(f"[green]valid[/green]: {train_examples} training examples parsed")
    console.print(f"next: {state.next_action()}")


@app.command("test-run")
def test_run() -> None:
    """Short real training run proving the model loads and a checkpoint writes."""
    preset, params, state = _context()
    try:
        state.require_ready(Stage.TEST_RUN)
    except StageBlocked as exc:
        _fail(str(exc))

    config = render_config(preset, params, test_run=True)
    config_path = write_config(config, preset.paths.run_dir / "test-run" / "config.yaml")
    state.write(Stage.TEST_RUN, Status.RUNNING, message="running a few steps")

    console.print(f"test run: {params.base_model}, {config.get('max_steps')} steps. "
                  "The base model is downloaded on first use; this can take a while.")
    try:
        result = invoke_trainer(
            config_path,
            log_path=state.log_path(Stage.TEST_RUN),
            echo=_echo_progress,
        )
    except TrainerError as exc:
        state.write(Stage.TEST_RUN, Status.FAILED, error=str(exc))
        _fail(str(exc))
        return

    if not result.ok:
        state.write(Stage.TEST_RUN, Status.FAILED, error=result.stderr_tail or "test run failed")
        err_console.print(f"[red]test run failed[/red]\n{result.stderr_tail}")
        err_console.print(f"\nfull log: {state.log_path(Stage.TEST_RUN)}")
        raise typer.Exit(code=1)

    state.write(Stage.TEST_RUN, Status.OK, message="test run completed", **result.payload)
    console.print("[green]test run completed[/green]")
    console.print("review it with 'antgpu-run results', then 'antgpu-run approve'")


@app.command("results")
def results(
    stage: Annotated[str, typer.Option("--stage", help="Which step to show.")] = "test-run",
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show what a step produced."""
    preset, params, state = _context()
    try:
        target = Stage(stage)
    except ValueError:
        _fail(f"unknown step {stage!r}. Choose from: {', '.join(str(s) for s in Stage)}")
        return

    summary = state.read(target)
    if as_json:
        console.print_json(data=summary.to_json())
        return

    if summary.status is Status.PENDING:
        console.print(f"{target} has not run yet.")
        return

    console.print(Panel(f"[bold]{target}[/bold] — [{STATUS_STYLE[summary.status]}]{summary.status}[/]", expand=False))
    table = Table(show_header=False)
    for key, value in sorted(summary.detail.items()):
        table.add_row(str(key), str(value)[:100])
    console.print(table)

    output_dir = preset.paths.output_dir / ("test-run" if target is Stage.TEST_RUN else params.result_name)
    if output_dir.is_dir():
        console.print(f"\nartifacts: {output_dir}")
        for child in sorted(output_dir.iterdir())[:10]:
            console.print(f"  {child.name}")
    console.print(f"\nlog: {state.log_path(target)}")


@app.command("approve")
def approve() -> None:
    """Accept the test run and unlock the full run."""
    _, _, state = _context()
    try:
        state.require_ready(Stage.APPROVAL)
    except StageBlocked as exc:
        _fail(str(exc))

    test_summary = state.read(Stage.TEST_RUN)
    state.write(
        Stage.APPROVAL,
        Status.OK,
        message="test run approved",
        approved_test_run_at=test_summary.updated_at,
    )
    console.print("[green]approved[/green] — the full run is now unlocked")
    console.print("next: run 'antgpu-run run'")


@app.command("run")
def run(
    resume: Annotated[bool, typer.Option("--resume", help="Resume from the newest checkpoint.")] = False,
) -> None:
    """Full training run. Requires an approved test run."""
    preset, params, state = _context()
    try:
        state.require_ready(Stage.RUN)
    except StageBlocked as exc:
        _fail(str(exc))

    config = render_config(preset, params, test_run=False)
    config_path = write_config(config, preset.paths.run_dir / "run" / "config.yaml")
    state.write(Stage.RUN, Status.RUNNING, message="training")

    console.print(f"training {params.base_model} ({params.training_mode}) -> {config['output_dir']}")
    try:
        result = invoke_trainer(
            config_path,
            resume=resume,
            log_path=state.log_path(Stage.RUN),
            echo=_echo_progress,
        )
    except TrainerError as exc:
        state.write(Stage.RUN, Status.FAILED, error=str(exc))
        _fail(str(exc))
        return

    if not result.ok:
        state.write(Stage.RUN, Status.FAILED, error=result.stderr_tail or "training failed")
        err_console.print(f"[red]training failed[/red]\n{result.stderr_tail}")
        err_console.print(f"\nresume with: antgpu-run run --resume\nfull log: {state.log_path(Stage.RUN)}")
        raise typer.Exit(code=1)

    state.write(Stage.RUN, Status.OK, message="training complete", **result.payload)
    console.print("[green]training complete[/green]")
    console.print(f"artifacts: {config['output_dir']}")
    console.print("[yellow]Download your results before the rental ends — this storage is not permanent.[/yellow]")


@app.command("logs")
def logs(
    stage: Annotated[str, typer.Argument(help="Which step's log to print.")] = "run",
    tail: Annotated[int, typer.Option("--tail", help="Show only the last N lines.")] = 0,
) -> None:
    """Print a step's captured output."""
    preset, _, state = _context()
    try:
        target = Stage(stage)
    except ValueError:
        _fail(f"unknown step {stage!r}. Choose from: {', '.join(str(s) for s in Stage)}")
        return

    path = state.log_path(target)
    if not path.is_file():
        console.print(f"no log for {target} yet ({path})")
        raise typer.Exit(code=1)

    text = path.read_text(encoding="utf-8", errors="replace")
    if tail > 0:
        text = "\n".join(text.splitlines()[-tail:])
    console.print(text, markup=False, highlight=False)


if __name__ == "__main__":
    app()
