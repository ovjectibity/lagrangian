FROM nvidia/cuda:13.3.0-cudnn-runtime-ubuntu24.04

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

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        software-properties-common \
        libgl1 \
        libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.14 \
        python3.14-dev \
        python3.14-venv \
    && python3.14 -m ensurepip --upgrade \
    && ln -sf /usr/bin/python3.14 /usr/bin/python \
    && ln -sf /usr/bin/python3.14 /usr/bin/python3 \
    && ln -sf /usr/local/bin/pip3.14 /usr/local/bin/pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/lagrangian/worker

COPY worker-py/requirements.txt ./requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY worker-py/ ./

EXPOSE 8765

CMD ["python", "main.py"]
