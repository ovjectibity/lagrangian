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

echo "Transferring ${TEST_SCRIPT} to ${REMOTE_IP}:${REMOTE_TMP} ..."
scp "${SSH_ARGS[@]}" "${TEST_SCRIPT}" "${SSH_USER}@${REMOTE_IP}:${REMOTE_TMP}"

echo "Copying into container ${CONTAINER_NAME} ..."
ssh "${SSH_ARGS[@]}" "${SSH_USER}@${REMOTE_IP}" \
  "sudo docker cp ${REMOTE_TMP} ${CONTAINER_NAME}:${CONTAINER_PATH} && rm -f ${REMOTE_TMP}"

echo "Running test inside container ..."
ssh "${SSH_ARGS[@]}" "${SSH_USER}@${REMOTE_IP}" \
  "sudo docker exec -e PYTHONPATH=/opt/lagrangian/worker ${CONTAINER_NAME} python3 ${CONTAINER_PATH}"
