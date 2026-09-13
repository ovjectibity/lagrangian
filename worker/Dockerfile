# https://hub.docker.com/r/nvidia/cuda
# FROM nvidia/cuda:13.3.0-cudnn-runtime-ubuntu24.04
FROM nvidia/cuda:13.3.1-cudnn-devel-ubuntu26.04
# FROM nvidia/cuda:13.3.1-tensorrt-devel-ubuntu24.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    XLA_FLAGS=--xla_dump_to=/tmp/xla_dump \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    WS_PING_TIMEOUT_SECONDS=none \
    PI0_CHECKPOINT_DIR=/opt/lagrangian/pi0 \
    #Jax specific flags: 
    XLA_PYTHON_CLIENT_PREALLOCATE=false
    # CUDA_LAUNCH_BLOCKING=1
    # JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache

# RUN apt-get update && apt-get install -y python3-pip python3-venv
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        software-properties-common \
        libgl1 \
        libglib2.0-0 \
        curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.14 \
        python3.14-dev \
        python3.14-venv \
    && python3.14 -m venv /opt/venv \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /opt/lagrangian/worker

COPY worker-py/requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY worker-py/ ./

EXPOSE 8765

CMD ["python", "main.py"]
