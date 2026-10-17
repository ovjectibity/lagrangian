#!/usr/bin/env bash
# Transfers a Python test script to an EC2 instance and runs it inside the worker container.
#
# Usage: run_remote_test.sh <test-script.py> <remote-ip>
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <test-script.py> <remote-ip>" >&2
  exit 1
fi

TEST_SCRIPT="$1"
REMOTE_IP="$2"

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

if [[ -z "${REPO_BASE:-}" ]]; then
  echo "error: REPO_BASE must be set" >&2
  exit 1
fi

SSH_USER="${REMOTE_WORKER_SSH_USER:-ec2-user}"
KEY_FILE="${REMOTE_WORKER_KEY_FILE:-}"
CONTAINER_NAME="${REMOTE_WORKER_CONTAINER_NAME:-lagrangian-worker-py}"

if [[ -n "${KEY_FILE}" && "${KEY_FILE}" != /* ]]; then
  KEY_FILE="${REPO_ROOT}/${KEY_FILE}"
fi

SSH_ARGS=(-o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=10)
if [[ -n "${KEY_FILE}" ]]; then
  SSH_ARGS+=(-i "${KEY_FILE}")
fi

SCRIPT_BASENAME="$(basename "${TEST_SCRIPT}")"
REMOTE_TMP="/tmp/${SCRIPT_BASENAME}"
CONTAINER_PATH="/tmp/${SCRIPT_BASENAME}"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
REMOTE_REQUIREMENTS_TMP="/tmp/requirements.txt"
CONTAINER_REQUIREMENTS_PATH="/tmp/requirements.txt"
CONTAINER_ARTIFACT_DIR="/tmp/lagrangian-remote-test-artifacts"

echo "Transferring ${TEST_SCRIPT} to ${REMOTE_IP}:${REMOTE_TMP} ..."
scp "${SSH_ARGS[@]}" "${TEST_SCRIPT}" "${SSH_USER}@${REMOTE_IP}:${REMOTE_TMP}"

echo "Copying into container ${CONTAINER_NAME} ..."
ssh "${SSH_ARGS[@]}" "${SSH_USER}@${REMOTE_IP}" \
  "sudo docker cp ${REMOTE_TMP} ${CONTAINER_NAME}:${CONTAINER_PATH} && rm -f ${REMOTE_TMP}"

if [[ -f "${REQUIREMENTS_FILE}" ]]; then
  echo "Transferring requirements.txt to ${REMOTE_IP}:${REMOTE_REQUIREMENTS_TMP} ..."
  scp "${SSH_ARGS[@]}" "${REQUIREMENTS_FILE}" "${SSH_USER}@${REMOTE_IP}:${REMOTE_REQUIREMENTS_TMP}"

  echo "Copying requirements.txt into container ${CONTAINER_NAME} ..."
  ssh "${SSH_ARGS[@]}" "${SSH_USER}@${REMOTE_IP}" \
    "sudo docker cp ${REMOTE_REQUIREMENTS_TMP} ${CONTAINER_NAME}:${CONTAINER_REQUIREMENTS_PATH} && rm -f ${REMOTE_REQUIREMENTS_TMP}"

  echo "Installing requirements inside container ${CONTAINER_NAME} ..."
  ssh "${SSH_ARGS[@]}" "${SSH_USER}@${REMOTE_IP}" \
    "sudo docker exec ${CONTAINER_NAME} uv pip install --python /opt/lagrangian/worker/.venv/bin/python -r ${CONTAINER_REQUIREMENTS_PATH}"
fi

echo "Preparing artifact directory inside container ${CONTAINER_NAME} ..."
ssh "${SSH_ARGS[@]}" "${SSH_USER}@${REMOTE_IP}" \
  "sudo docker exec ${CONTAINER_NAME} rm -rf ${CONTAINER_ARTIFACT_DIR} && \
   sudo docker exec ${CONTAINER_NAME} mkdir -p ${CONTAINER_ARTIFACT_DIR}"

echo "Running test inside container ..."
set +e
ssh "${SSH_ARGS[@]}" "${SSH_USER}@${REMOTE_IP}" \
  "sudo docker exec \
  -w ${CONTAINER_ARTIFACT_DIR} \
  -e PYTHONPATH=/opt/lagrangian/worker \
   -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ${CONTAINER_NAME} python3 ${CONTAINER_PATH}"
TEST_EXIT=$?
set -e

mkdir -p "${REPO_BASE}/tmp"
LOCAL_ARTIFACT_DIR="$(mktemp -d "${REPO_BASE}/tmp/lagrangian-remote-test-artifacts.XXXXXX")"
echo "Copying test artifacts to ${LOCAL_ARTIFACT_DIR} ..."
ssh "${SSH_ARGS[@]}" "${SSH_USER}@${REMOTE_IP}" \
  "sudo docker cp ${CONTAINER_NAME}:${CONTAINER_ARTIFACT_DIR}/. -" \
  | tar -xf - -C "${LOCAL_ARTIFACT_DIR}"
echo "Test artifacts saved to ${LOCAL_ARTIFACT_DIR}"

exit ${TEST_EXIT}

# -e TF_GPU_ALLOCATOR=cuda_malloc_async \
# -e XLA_PYTHON_CLIENT_ALLOCATOR=platform \
#  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
# -e XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
