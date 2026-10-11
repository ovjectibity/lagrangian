#!/usr/bin/env bash
# Syncs locally changed worker and vendored simulator files to the running local
# worker container, then restarts it. Changes are lost when the container is
# recreated from its image.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.envrc"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

CONTAINER_NAME="${REMOTE_WORKER_CONTAINER_NAME:-lagrangian-worker-py}"
WORKER_ROOT="/opt/lagrangian/worker"
THIRD_PARTY_ROOT="/opt/lagrangian/third-party"
SITE_PACKAGES_ROOT="${WORKER_ROOT}/.venv/lib/python3.14/site-packages"
NEEDS_UV_SYNC=0
SYNCED_FILES=0

if ! docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "error: container '${CONTAINER_NAME}' does not exist." >&2
  exit 1
fi

changed_files() {
  local repository="$1"

  {
    git -C "${repository}" diff --name-only HEAD
    git -C "${repository}" ls-files --others --exclude-standard
  } | sort -u
}

copy_file() {
  local local_path="$1"
  local container_path="$2"

  docker cp "${local_path}" "${CONTAINER_NAME}:${container_path}"
}

restart_and_wait() {
  local attempt

  docker restart "${CONTAINER_NAME}" >/dev/null
  for attempt in {1..30}; do
    if [[ "$(docker container inspect --format '{{.State.Running}}' "${CONTAINER_NAME}")" == "true" ]]; then
      return
    fi
    sleep 1
  done

  echo "error: container '${CONTAINER_NAME}' did not stay running after the file sync." >&2
  return 1
}

sync_worker_file() {
  local relative_path="$1"
  local local_path="${REPO_ROOT}/worker/worker-py/${relative_path}"

  if [[ ! -f "${local_path}" ]]; then
    echo "  skipping deleted worker file: ${relative_path}"
    return
  fi

  echo "  worker/worker-py/${relative_path}"
  copy_file "${local_path}" "${WORKER_ROOT}/${relative_path}"
  SYNCED_FILES=1

  if [[ "${relative_path}" == "pyproject.toml" || "${relative_path}" == "uv.lock" ]]; then
    NEEDS_UV_SYNC=1
  fi
}

sync_third_party_file() {
  local package_name="$1"
  local relative_path="$2"
  local local_path="${REPO_ROOT}/worker/third-party/${package_name}/${relative_path}"
  local installed_path=""

  if [[ ! -f "${local_path}" ]]; then
    echo "  skipping deleted ${package_name} file: ${relative_path}"
    return
  fi

  echo "  third-party/${package_name}/${relative_path}"
  copy_file "${local_path}" "${THIRD_PARTY_ROOT}/${package_name}/${relative_path}"
  SYNCED_FILES=1

  case "${package_name}:${relative_path}" in
    libero:libero/libero/*)
      installed_path="${SITE_PACKAGES_ROOT}/libero/libero/${relative_path#libero/libero/}"
      ;;
    robosuite:robosuite/*)
      installed_path="${SITE_PACKAGES_ROOT}/robosuite/${relative_path#robosuite/}"
      ;;
    libero:setup.py|libero:pyproject.toml|robosuite:setup.py|robosuite:pyproject.toml)
      NEEDS_UV_SYNC=1
      ;;
  esac

  if [[ -n "${installed_path}" ]]; then
    copy_file "${local_path}" "${installed_path}"
  fi
}

echo "Files to sync to ${CONTAINER_NAME}:"
while IFS= read -r relative_path; do
  [[ -n "${relative_path}" ]] && sync_worker_file "${relative_path}"
done < <(changed_files "${REPO_ROOT}" | sed -n 's#^worker/worker-py/##p')

while IFS= read -r relative_path; do
  [[ -n "${relative_path}" ]] && sync_third_party_file "libero" "${relative_path}"
done < <(changed_files "${REPO_ROOT}/worker/third-party/libero")

while IFS= read -r relative_path; do
  [[ -n "${relative_path}" ]] && sync_third_party_file "robosuite" "${relative_path}"
done < <(changed_files "${REPO_ROOT}/worker/third-party/robosuite")

if [[ "${SYNCED_FILES}" -eq 0 ]]; then
  echo "No changed worker, LIBERO, or robosuite files to sync."
  exit 0
fi

if [[ "${NEEDS_UV_SYNC}" -eq 1 ]]; then
  echo "Restarting container ${CONTAINER_NAME} before dependency synchronization ..."
  restart_and_wait
  echo "Synchronizing changed package metadata and dependencies ..."
  docker exec "${CONTAINER_NAME}" sh -lc \
    "cd ${WORKER_ROOT} && CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync --locked --no-dev --no-install-project"
fi

echo "Restarting container ${CONTAINER_NAME} ..."
restart_and_wait
echo "Done."
