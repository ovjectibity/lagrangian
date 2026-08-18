#!/usr/bin/env bash
set -euo pipefail

: "${REMOTE_WORKER_IMAGE_URI:?REMOTE_WORKER_IMAGE_URI is required}"
: "${REMOTE_WORKER_CONTAINER_NAME:?REMOTE_WORKER_CONTAINER_NAME is required}"
: "${REMOTE_WORKER_PORT:?REMOTE_WORKER_PORT is required}"
: "${REMOTE_WORKER_SHARED_SECRET:?REMOTE_WORKER_SHARED_SECRET is required}"
: "${LOCAL_WORKER_WORKING_DIR:?LOCAL_WORKER_WORKING_DIR is required}"

HF_TOKEN="${HF_TOKEN:-}"
PI0_CHECKPOINT_DIR="${LOCAL_WORKER_WORKING_DIR}/lagrangian/pi0"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

docker rm -f "${REMOTE_WORKER_CONTAINER_NAME}" || true

PI0_CHECKPOINT_DIR="${PI0_CHECKPOINT_DIR}" bash "${SCRIPT_DIR}/download_pi0_model.sh"
cat >"${LOCAL_WORKER_WORKING_DIR}/lagrangian-worker.env" <<ENV_EOF
WS_HOST=0.0.0.0
WS_PORT=${REMOTE_WORKER_PORT}
WORKER_SHARED_SECRET=${REMOTE_WORKER_SHARED_SECRET}
WS_PING_TIMEOUT_SECONDS=60
HF_TOKEN=${HF_TOKEN}
PI0_CHECKPOINT_DIR=/opt/lagrangian/pi0
CUDA_LAUNCH_BLOCKING=1
ENV_EOF

#   --gpus all \

docker run -d --restart unless-stopped \
  --platform linux/amd64 \
  --name "${REMOTE_WORKER_CONTAINER_NAME}" \
  --env-file "${LOCAL_WORKER_WORKING_DIR}/lagrangian-worker.env" \
  -v "${PI0_CHECKPOINT_DIR}:/opt/lagrangian/pi0" \
  -p "${REMOTE_WORKER_PORT}:${REMOTE_WORKER_PORT}" \
  "${REMOTE_WORKER_IMAGE_URI}"
