FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    XLA_FLAGS=--xla_dump_to=/tmp/xla_dump \
    WS_PING_TIMEOUT_SECONDS=none \
    PI0_CHECKPOINT_DIR=/opt/lagrangian/pi0
    # JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3.11 \
        python3.11-dev \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
        git \
        git-lfs \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/lagrangian/worker/trainer-py

COPY trainer-py/requirements.txt ./requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

RUN git lfs install \
    && git clone https://huggingface.co/lerobot/pi0_libero_base /opt/lagrangian/pi0

COPY trainer-py/ ./

EXPOSE 8765

CMD ["python", "main.py"]
