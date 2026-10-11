#!/usr/bin/env bash
set -euo pipefail

USE_LATEST_LOCAL_IMAGE=0
USE_GPU=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --use-latest-local-image)
      USE_LATEST_LOCAL_IMAGE=1
      ;;
    --without-gpu)
      USE_GPU=0
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      echo "usage: $0 [--use-latest-local-image] [--without-gpu]" >&2
      exit 1
      ;;
  esac
  shift
done

: "${REMOTE_WORKER_CONTAINER_NAME:?REMOTE_WORKER_CONTAINER_NAME is required}"
: "${REMOTE_WORKER_PORT:?REMOTE_WORKER_PORT is required}"
: "${REMOTE_WORKER_SHARED_SECRET:?REMOTE_WORKER_SHARED_SECRET is required}"
: "${LOCAL_WORKER_WORKING_DIR:?LOCAL_WORKER_WORKING_DIR is required}"

if [[ "${USE_LATEST_LOCAL_IMAGE}" -eq 0 ]]; then
  : "${REMOTE_WORKER_IMAGE_URI:?REMOTE_WORKER_IMAGE_URI is required}"
fi

HF_TOKEN="${HF_TOKEN:-}"
PI0_CHECKPOINT_DIR="${LOCAL_WORKER_WORKING_DIR}/lagrangian/pi0"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DEBUG_MODE="${ENABLE_WORKER_CUDA_DEBUG_MODE:-0}"

if [[ "${USE_LATEST_LOCAL_IMAGE}" -eq 1 ]]; then
  LOCAL_IMAGE_REPOSITORY="${LOCAL_WORKER_IMAGE_REPOSITORY:-lagrangian/remote-worker}"
  IMAGE_URI="$(
    docker image ls --filter "reference=${LOCAL_IMAGE_REPOSITORY}:*" --format '{{.ID}}' \
      | while IFS= read -r image_id; do
          docker image inspect --format '{{.Created}} {{index .RepoTags 0}}' "${image_id}"
        done \
      | sort -r \
      | head -n 1 \
      | cut -d ' ' -f 2-
  )"
  if [[ -z "${IMAGE_URI}" ]]; then
    echo "error: no local image found for ${LOCAL_IMAGE_REPOSITORY}:*" >&2
    exit 1
  fi
  echo "Using latest local worker image: ${IMAGE_URI}"
else
  IMAGE_URI="${REMOTE_WORKER_IMAGE_URI}"
  if [[ "${IMAGE_URI}" != *:* ]]; then
    IMAGE_URI="${IMAGE_URI}:latest"
  fi

  REGISTRY="${IMAGE_URI%%/*}"
  if [[ "${REGISTRY}" == *".dkr.ecr."* ]]; then
    REGION="${REMOTE_WORKER_AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
    if [[ -z "${REGION}" ]]; then
      REGION="$(echo "${REGISTRY}" | sed -E 's#^[0-9]+\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com$#\1#')"
    fi
    if [[ -z "${REGION}" || "${REGION}" == "${REGISTRY}" ]]; then
      echo "error: could not infer AWS region. Set REMOTE_WORKER_AWS_REGION or AWS_DEFAULT_REGION." >&2
      exit 1
    fi
    aws ecr get-login-password --region "${REGION}" \
      | docker login --username AWS --password-stdin "${REGISTRY}"
  fi

  docker pull "${IMAGE_URI}"
fi

docker rm -f "${REMOTE_WORKER_CONTAINER_NAME}" || true

mkdir -p "${LOCAL_WORKER_WORKING_DIR}"
PI0_CHECKPOINT_DIR="${PI0_CHECKPOINT_DIR}" bash "${SCRIPT_DIR}/download_pi0_model.sh"
cat >"${LOCAL_WORKER_WORKING_DIR}/lagrangian-worker.env" <<ENV_EOF
WS_HOST=0.0.0.0
WS_PORT=${REMOTE_WORKER_PORT}
WORKER_SHARED_SECRET=${REMOTE_WORKER_SHARED_SECRET}
WS_PING_TIMEOUT_SECONDS=60
HF_TOKEN=${HF_TOKEN}
PI0_CHECKPOINT_DIR=/opt/lagrangian/pi0
CUDA_LAUNCH_BLOCKING=${CUDA_DEBUG_MODE}
CUDA_DEBUG_MODE=${CUDA_DEBUG_MODE}
ENV_EOF

DOCKER_GPU_ARGS=()
if [[ "${USE_GPU}" -eq 1 ]]; then
  DOCKER_GPU_ARGS=(--gpus all)
else
  echo "Starting local worker without GPU access."
fi

docker run -d --restart unless-stopped \
  "${DOCKER_GPU_ARGS[@]}" \
  --cap-add=SYS_ADMIN \
  --name "${REMOTE_WORKER_CONTAINER_NAME}" \
  --env-file "${LOCAL_WORKER_WORKING_DIR}/lagrangian-worker.env" \
  -v "${PI0_CHECKPOINT_DIR}:/opt/lagrangian/pi0" \
  -p "${REMOTE_WORKER_PORT}:${REMOTE_WORKER_PORT}" \
  "${IMAGE_URI}"
