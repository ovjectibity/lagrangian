#!/usr/bin/env bash
# gen_api_spec.sh
#
# Builds the app, starts it on a temporary port,
# fetches GET /api/openapi.json, writes the result
# to docs/user/api_spec/spec.json, then shuts the
# app down.
#
# Run from the project root: ./scripts/gen_api_spec.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── libtorch (required by tch-rs in the app) ─────────

if [[ -f "$REPO_ROOT/.envrc" ]]; then
  set -a
  source "$REPO_ROOT/.envrc"
  set +a
fi
export LIBTORCH="${LIBTORCH:-/Users/avi/Documents/projects/libtorch}"
export LIBTORCH_BYPASS_VERSION_CHECK="${LIBTORCH_BYPASS_VERSION_CHECK:-1}"

# ── Config ───────────────────────────────────────────

PORT=18743
OUT="$REPO_ROOT/docs/user/api_spec/spec.json"
BINARY="$REPO_ROOT/app/target/debug/lagrangian"
TMPDIR_CLIENT="$(mktemp -d)"

log() { echo "[gen_api_spec] $*"; }

cleanup() {
  if [[ -n "${APP_PID:-}" ]]; then
    kill "$APP_PID" 2>/dev/null || true
    wait "$APP_PID" 2>/dev/null || true
    log "App stopped."
  fi
  rm -rf "$TMPDIR_CLIENT"
}
trap cleanup EXIT

# ── Build ────────────────────────────────────────────
# Must cd into app/ so that app/.cargo/config.toml is
# picked up (sets libtorch/mujoco rustflags). Without
# it the build cache key differs and tch is recompiled
# from scratch, which fails on Rust ≥ 1.87.

log "Building app..."
(cd "$REPO_ROOT/app" && cargo build)

# ── Start app ────────────────────────────────────────

log "Starting app on port $PORT..."

CLIENT_DIST_DIR="$TMPDIR_CLIENT" \
ROCKET_PORT="$PORT" \
ROCKET_LOG_LEVEL="off" \
  "$BINARY" &

APP_PID=$!

# ── Wait for readiness (up to 30 s) ─────────────────

log "Waiting for app to be ready..."
for i in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:$PORT/api/openapi.json" \
       -o /dev/null 2>/dev/null; then
    break
  fi
  if ! kill -0 "$APP_PID" 2>/dev/null; then
    echo "ERROR: app process exited unexpectedly" >&2
    exit 1
  fi
  sleep 1
done

if ! curl -sf "http://127.0.0.1:$PORT/api/openapi.json" \
       -o /dev/null 2>/dev/null; then
  echo "ERROR: app did not become ready within 30 s" >&2
  exit 1
fi

# ── Fetch spec ───────────────────────────────────────

log "Fetching spec..."
mkdir -p "$(dirname "$OUT")"
curl -sf "http://127.0.0.1:$PORT/api/openapi.json" \
  | python3 -m json.tool --indent 2 \
  > "$OUT"

log "Spec written to $OUT"
