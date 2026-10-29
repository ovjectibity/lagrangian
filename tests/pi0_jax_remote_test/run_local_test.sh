#!/usr/bin/env bash
# Copies a Python test into the local worker container, runs it, and collects artifacts.
#
# Usage: run_local_test.sh <test-script.py>
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <test-script.py>" >&2
  exit 1
fi

TEST_SCRIPT="$1"
if [[ ! -f "${TEST_SCRIPT}" ]]; then
  echo "error: test script not found: ${TEST_SCRIPT}" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
AWS_ENV_FILE="${REPO_ROOT}/.envrc"

if [[ -f "${AWS_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${AWS_ENV_FILE}"
  set +a
fi

CONTAINER_NAME="${REMOTE_WORKER_CONTAINER_NAME:-lagrangian-worker-py}"
SCRIPT_BASENAME="$(basename "${TEST_SCRIPT}")"
CONTAINER_PATH="/tmp/${SCRIPT_BASENAME}"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
CONTAINER_REQUIREMENTS_PATH="/tmp/requirements.txt"
CONTAINER_ARTIFACT_DIR="/tmp/lagrangian-local-test-artifacts"

echo "Copying ${TEST_SCRIPT} into container ${CONTAINER_NAME} ..."
docker cp "${TEST_SCRIPT}" "${CONTAINER_NAME}:${CONTAINER_PATH}"

if [[ -f "${REQUIREMENTS_FILE}" ]]; then
  echo "Copying requirements.txt into container ${CONTAINER_NAME} ..."
  docker cp "${REQUIREMENTS_FILE}" "${CONTAINER_NAME}:${CONTAINER_REQUIREMENTS_PATH}"

  echo "Installing requirements inside container ${CONTAINER_NAME} ..."
  docker exec "${CONTAINER_NAME}" uv pip install \
    --python /opt/lagrangian/worker/.venv/bin/python \
    -r "${CONTAINER_REQUIREMENTS_PATH}"
fi

echo "Preparing artifact directory inside container ${CONTAINER_NAME} ..."
docker exec "${CONTAINER_NAME}" rm -rf "${CONTAINER_ARTIFACT_DIR}"
docker exec "${CONTAINER_NAME}" mkdir -p "${CONTAINER_ARTIFACT_DIR}"

echo "Running test inside container ..."
set +e
docker exec \
  -w "${CONTAINER_ARTIFACT_DIR}" \
  -e PYTHONPATH=/opt/lagrangian/worker \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "${CONTAINER_NAME}" python3 "${CONTAINER_PATH}"
TEST_EXIT=$?
set -e

ARTIFACT_BASE="${REPO_BASE:-${REPO_ROOT}}"
mkdir -p "${ARTIFACT_BASE}/tmp"
LOCAL_ARTIFACT_DIR="$(mktemp -d "${ARTIFACT_BASE}/tmp/lagrangian-local-test-artifacts.XXXXXX")"
echo "Copying test artifacts to ${LOCAL_ARTIFACT_DIR} ..."
docker cp "${CONTAINER_NAME}:${CONTAINER_ARTIFACT_DIR}/." "${LOCAL_ARTIFACT_DIR}/"
echo "Test artifacts saved to ${LOCAL_ARTIFACT_DIR}"

exit "${TEST_EXIT}"
