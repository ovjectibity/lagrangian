# https://hub.docker.com/r/nvidia/cuda
# FROM nvidia/cuda:13.3.0-cudnn-runtime-ubuntu24.04
FROM nvidia/cuda:13.3.1-cudnn-devel-ubuntu26.04
# FROM nvidia/cuda:13.3.1-tensorrt-devel-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_PYTHON_DOWNLOADS=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    XLA_FLAGS=--xla_dump_to=/tmp/xla_dump \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    WS_PING_TIMEOUT_SECONDS=none \
    PI0_CHECKPOINT_DIR=/opt/lagrangian/pi0

# RUN apt-get update && apt-get install -y python3-pip python3-venv
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        software-properties-common \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.14 \
        python3.14-dev \
        python3.14-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/lagrangian/worker

COPY worker-py/pyproject.toml worker-py/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync --locked --no-dev --no-install-project

COPY worker-py/ ./

ENV PATH="/opt/lagrangian/worker/.venv/bin:${PATH}"

EXPOSE 8765

CMD ["python", "main.py"]
