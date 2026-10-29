#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AWS_ENV_FILE="${REPO_ROOT}/.envrc"

CREATE_REPO=0
PURGE_REPO_IMAGES=0
BUILD_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --create-repo)
      CREATE_REPO=1
      ;;
    --purge-repo-images)
      PURGE_REPO_IMAGES=1
      ;;
    --build-only)
      BUILD_ONLY=1
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      echo "usage: $0 [--build-only] [--create-repo] [--purge-repo-images]" >&2
      exit 1
      ;;
  esac
  shift
done

if [[ "${BUILD_ONLY}" -eq 1 ]] && \
  { [[ "${CREATE_REPO}" -eq 1 ]] || [[ "${PURGE_REPO_IMAGES}" -eq 1 ]]; }; then
  echo "error: --build-only cannot be combined with ECR management flags." >&2
  exit 1
fi

if [[ -f "${AWS_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "${AWS_ENV_FILE}"
  set +a
fi

if [[ -z "${REMOTE_WORKER_IMAGE_URI:-}" ]]; then
  echo "error: REMOTE_WORKER_IMAGE_URI is required (set in .envrc or env)." >&2
  exit 1
fi

IMAGE_URI="${REMOTE_WORKER_IMAGE_URI}"
if [[ "${IMAGE_URI}" != *:* ]]; then
  IMAGE_URI="${IMAGE_URI}:latest"
fi

if [[ "${BUILD_ONLY}" -eq 1 ]]; then
  LOCAL_IMAGE="${IMAGE_URI}"
else
  LOCAL_IMAGE="${REMOTE_WORKER_LOCAL_IMAGE:-lagrangian/remote-worker:$(date +%Y%m%d%H%M%S)}"
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker not found in PATH." >&2
  exit 1
fi

build_worker_image() {
  echo "Building image from worker/Dockerfile as ${LOCAL_IMAGE} ..."
  docker build \
    --platform linux/amd64 \
    --load \
    -f "${REPO_ROOT}/worker/Dockerfile" \
    "${REPO_ROOT}/worker" \
    -t "${LOCAL_IMAGE}"
}

if [[ "${BUILD_ONLY}" -eq 1 ]]; then
  build_worker_image
  echo "Done: ${LOCAL_IMAGE}"
  exit 0
fi

ECR_REGISTRY="${IMAGE_URI%%/*}"
REPO_AND_TAG="${IMAGE_URI#*/}"
REPO_NAME="${REPO_AND_TAG%%:*}"

REGION="${REMOTE_WORKER_AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
if [[ -z "${REGION}" ]]; then
  REGION="$(echo "${ECR_REGISTRY}" | sed -E 's#^[0-9]+\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com$#\1#')"
fi
if [[ -z "${REGION}" || "${REGION}" == "${ECR_REGISTRY}" ]]; then
  echo "error: could not infer AWS region. Set REMOTE_WORKER_AWS_REGION or AWS_DEFAULT_REGION." >&2

  exit 1
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "error: aws CLI not found in PATH." >&2
  exit 1
fi

delete_all_repo_images() {
  local next_token=""
  local image_ids_json

  echo "Deleting all existing images from ECR repository: ${REPO_NAME}"
  while :; do
    if [[ -n "${next_token}" ]]; then
      image_ids_json="$(
        aws ecr list-images \
          --region "${REGION}" \
          --repository-name "${REPO_NAME}" \
          --max-results 100 \
          --next-token "${next_token}" \
          --query 'imageIds' \
          --output json
      )"
      next_token="$(
        aws ecr list-images \
          --region "${REGION}" \
          --repository-name "${REPO_NAME}" \
          --max-results 100 \
          --next-token "${next_token}" \
          --query 'nextToken' \
          --output text
      )"
    else
      image_ids_json="$(
        aws ecr list-images \
          --region "${REGION}" \
          --repository-name "${REPO_NAME}" \
          --max-results 100 \
          --query 'imageIds' \
          --output json
      )"
      next_token="$(
        aws ecr list-images \
          --region "${REGION}" \
          --repository-name "${REPO_NAME}" \
          --max-results 100 \
          --query 'nextToken' \
          --output text
      )"
    fi

    if [[ "${image_ids_json}" != "[]" ]]; then
      aws ecr batch-delete-image \
        --region "${REGION}" \
        --repository-name "${REPO_NAME}" \
        --image-ids "${image_ids_json}" >/dev/null
    fi

    if [[ -z "${next_token}" || "${next_token}" == "None" ]]; then
      break
    fi
  done
}

REPO_EXISTS=1
if ! aws ecr describe-repositories --region "${REGION}" --repository-names "${REPO_NAME}" >/dev/null 2>&1; then
  REPO_EXISTS=0
fi

if [[ "${CREATE_REPO}" -eq 1 && "${REPO_EXISTS}" -eq 0 ]]; then
    echo "Creating ECR repository: ${REPO_NAME}"
    aws ecr create-repository --region "${REGION}" --repository-name "${REPO_NAME}" >/dev/null
  REPO_EXISTS=1
fi

if [[ "${PURGE_REPO_IMAGES}" -eq 1 ]]; then
  if [[ "${REPO_EXISTS}" -eq 0 ]]; then
    echo "Skipping image purge because repository does not exist: ${REPO_NAME}"
  else
    delete_all_repo_images
  fi
fi

build_worker_image

echo "Logging into ECR registry: ${ECR_REGISTRY}"
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

echo "Tagging local image ${LOCAL_IMAGE} as ${IMAGE_URI}"
docker tag "${LOCAL_IMAGE}" "${IMAGE_URI}"

echo "Pushing image to ECR ..."
docker push "${IMAGE_URI}"

echo "Done: ${IMAGE_URI}"
