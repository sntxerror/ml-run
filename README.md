# antgpu-run

Workflow runner for AntGPU templates. It owns the **run state machine** and
shells out to the tool a workflow kind needs. It is deliberately thin:

- `antgpu-train` stays a portable trainer with no platform dependency.
- `antgpu-ml` stays the corpus/dataset pipeline, with its heavy archive and OCR
  dependencies. Those must not end up in a training image.

| Workflow kind | Status | Delegates to |
|---|---|---|
| `train-pairs-v1` | implemented | `antgpu-train` |
| `dataset-prep-v1` | reserved | `antgpu-ml` |

## The flow

```text
dataset  ->  validate  ->  test-run  ->  approve  ->  run
```

Each step is gated on the previous one. `run` refuses to start until a test run
has succeeded and been approved. State lives in per-stage `summary.json` files
under the run directory, following the same pattern `antgpu-ml` uses, so the
web UI and the CLI read exactly the same files.

```bash
antgpu-run status
antgpu-run dataset status
antgpu-run validate
antgpu-run test-run
antgpu-run results
antgpu-run approve
antgpu-run run
```

## Configuration

Parameters arrive as `ANTGPU_RUN_*` environment variables, set by the control
plane when the launch form is submitted. Anything unset falls back to the
preset's default. Bounds are enforced from the preset, not from user input.
