# The runner sits on top of the trainer image. antgpu-run never imports
# antgpu-train; it invokes the CLI. Building FROM the trainer is what puts that
# CLI on PATH.
ARG TRAINER_IMAGE=antgpu-train:local
FROM ${TRAINER_IMAGE}

# The template's on-start script is executed with /bin/sh and the user works
# over an injected SSH session, so an interactive shell and a pager are part of
# the product here, not conveniences.
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash less ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/antgpu/runner
COPY pyproject.toml README.md ./
COPY src ./src

# Installed into the trainer's own venv so `antgpu-run` and `antgpu-train` share
# an interpreter and one is always able to invoke the other.
#
# uv is used rather than pip: the trainer image builds its venv with `uv sync`,
# which does not seed pip, so `python -m pip` does not exist there.
RUN uv pip install --python /app/.venv/bin/python --no-cache .

# Baked in so the container is self-describing: the run can be inspected without
# asking the control plane what it was launched with.
COPY presets/train-pairs-v1.yaml /opt/antgpu/workflow.yaml

ENV ANTGPU_RUN_PRESET=/opt/antgpu/workflow.yaml \
    HF_HOME=/cache/huggingface \
    PATH=/app/.venv/bin:$PATH

# Mount targets, matching the template's volume contract. Declared so the paths
# exist and are writable even when a volume has not been attached.
RUN mkdir -p /workspace/input /workspace/output /workspace/run /cache/huggingface

WORKDIR /workspace

# The trainer image sets ENTRYPOINT ["antgpu-train", "train"]. This image is an
# environment the user works inside, so it must not start a training run on
# boot. Connect overrides the entrypoint with the on-start script anyway; this
# keeps `docker run` sane for anyone testing the image by hand.
ENTRYPOINT []
CMD ["antgpu-run", "status"]
