#!/usr/bin/env bash
# Tars a local directory and transfers it to a remote EC2 instance, then untars
# it inside a running container under <container-dir>/<dirname-of-local-dir>.
#
# Usage: transfer_dir_to_remote.sh <local-dir> <remote-ip> <container-name> <container-dir>
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 <local-dir> <remote-ip> <container-name> <container-dir>" >&2
  exit 1
fi

LOCAL_DIR="$1"
REMOTE_IP="$2"
CONTAINER_NAME="$3"
CONTAINER_DIR="$4"

if [[ ! -d "${LOCAL_DIR}" ]]; then
  echo "error: local dir '${LOCAL_DIR}' does not exist." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AWS_ENV_FILE="${REPO_ROOT}/.envrc"

if [[ -f "${AWS_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${AWS_ENV_FILE}"
  set +a
fi

SSH_USER="${REMOTE_WORKER_SSH_USER:-ec2-user}"
KEY_FILE="${REMOTE_WORKER_KEY_FILE:-}"

if [[ -n "${KEY_FILE}" && "${KEY_FILE}" != /* ]]; then
  KEY_FILE="${REPO_ROOT}/${KEY_FILE}"
fi

SSH_ARGS=(-o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=10)
if [[ -n "${KEY_FILE}" ]]; then
  SSH_ARGS+=(-i "${KEY_FILE}")
fi

LOCAL_DIR="$(cd "${LOCAL_DIR}" && pwd)"
DIR_NAME="$(basename "${LOCAL_DIR}")"

TMP_TAR="$(mktemp -t "${DIR_NAME}.XXXXXX").tar.gz"
trap 'rm -f "${TMP_TAR}"' EXIT

echo "Tarring ${LOCAL_DIR} ..."
tar -czf "${TMP_TAR}" -C "$(dirname "${LOCAL_DIR}")" "${DIR_NAME}"

REMOTE_TAR="/tmp/$(basename "${TMP_TAR}")"

echo "Transferring tar to ${REMOTE_IP}:${REMOTE_TAR} ..."
scp "${SSH_ARGS[@]}" "${TMP_TAR}" "${SSH_USER}@${REMOTE_IP}:${REMOTE_TAR}"

echo "Copying tar into container ${CONTAINER_NAME} and extracting under ${CONTAINER_DIR}/${DIR_NAME} ..."
ssh "${SSH_ARGS[@]}" "${SSH_USER}@${REMOTE_IP}" "
  sudo docker exec ${CONTAINER_NAME} mkdir -p ${CONTAINER_DIR} \
  && sudo docker cp ${REMOTE_TAR} ${CONTAINER_NAME}:${CONTAINER_DIR}/$(basename "${TMP_TAR}") \
  && sudo docker exec ${CONTAINER_NAME} tar -xzf ${CONTAINER_DIR}/$(basename "${TMP_TAR}") -C ${CONTAINER_DIR} \
  && sudo docker exec ${CONTAINER_NAME} rm -f ${CONTAINER_DIR}/$(basename "${TMP_TAR}") \
  && rm -f ${REMOTE_TAR}
" </dev/null

echo "Done. Contents available at ${CONTAINER_DIR}/${DIR_NAME} inside ${CONTAINER_NAME}."
