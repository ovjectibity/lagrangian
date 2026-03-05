#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AWS_ENV_FILE="${REPO_ROOT}/.aws-env"

CREATE_REPO=0
if [[ "${1:-}" == "--create-repo" ]]; then
  CREATE_REPO=1
fi

if [[ -f "${AWS_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "${AWS_ENV_FILE}"
  set +a
fi

if [[ -z "${TRAINING_WORKER_IMAGE_URI:-}" ]]; then
  echo "error: TRAINING_WORKER_IMAGE_URI is required (set in .aws-env or env)." >&2
  exit 1
fi

IMAGE_URI="${TRAINING_WORKER_IMAGE_URI}"
if [[ "${IMAGE_URI}" != *:* ]]; then
  IMAGE_URI="${IMAGE_URI}:latest"
fi
LOCAL_IMAGE="${TRAINING_WORKER_LOCAL_IMAGE:-lagrangian/trainer-worker:$(date +%Y%m%d%H%M%S)}"

ECR_REGISTRY="${IMAGE_URI%%/*}"
REPO_AND_TAG="${IMAGE_URI#*/}"
REPO_NAME="${REPO_AND_TAG%%:*}"

REGION="${TRAINING_WORKER_AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
if [[ -z "${REGION}" ]]; then
  REGION="$(echo "${ECR_REGISTRY}" | sed -E 's#^[0-9]+\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com$#\1#')"
fi
if [[ -z "${REGION}" || "${REGION}" == "${ECR_REGISTRY}" ]]; then
  echo "error: could not infer AWS region. Set TRAINING_WORKER_AWS_REGION or AWS_DEFAULT_REGION." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker not found in PATH." >&2
  exit 1
fi
if ! command -v aws >/dev/null 2>&1; then
  echo "error: aws CLI not found in PATH." >&2
  exit 1
fi

if [[ "${CREATE_REPO}" -eq 1 ]]; then
  if ! aws ecr describe-repositories --region "${REGION}" --repository-names "${REPO_NAME}" >/dev/null 2>&1; then
    echo "Creating ECR repository: ${REPO_NAME}"
    aws ecr create-repository --region "${REGION}" --repository-name "${REPO_NAME}" >/dev/null
  fi
fi

echo "Building image from worker/trainer/Dockerfile as ${LOCAL_IMAGE} ..."
docker build \
  --platform linux/amd64 \
  -f "${REPO_ROOT}/worker/trainer/Dockerfile" \
  "${REPO_ROOT}/worker/trainer" \
  -t "${LOCAL_IMAGE}"

echo "Logging into ECR registry: ${ECR_REGISTRY}"
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

echo "Tagging local image ${LOCAL_IMAGE} as ${IMAGE_URI}"
docker tag "${LOCAL_IMAGE}" "${IMAGE_URI}"

echo "Pushing image to ECR ..."
docker push "${IMAGE_URI}"

echo "Done: ${IMAGE_URI}"
