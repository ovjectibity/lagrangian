#!/usr/bin/env bash
# purge_db.sh
#
# Default: clears all data, drops tables/collections,
#          and stops both DB servers.
#
# --purge-entries-only: truncates all managed tables (including
#   LUTs) and clears the managed MongoDB collections, preserving
#   schemas, validators, indexes, and running servers.
#
# See docs/db-env.md for env vars.

set -euo pipefail

# ── Defaults (override via env) ──────────────────────

POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-lagrangian}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-lagrangian}"
POSTGRES_DB="${POSTGRES_DB:-lagrangian}"

MONGODB_HOST="${MONGODB_HOST:-localhost}"
MONGODB_PORT="${MONGODB_PORT:-27017}"
MONGODB_DB="${MONGODB_DB:-lagrangian}"

# ── Parse args ────────────────────────────────────────

ENTRIES_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --purge-entries-only) ENTRIES_ONLY=true ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

# ── Helpers ──────────────────────────────────────────

log() { echo "[purge_db] $*"; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: '$1' not found in PATH" >&2
    exit 1
  }
}

# ── Pre-flight checks ────────────────────────────────

require_cmd psql
require_cmd mongosh

# ── Postgres ─────────────────────────────────────────

if $ENTRIES_ONLY; then
  log "Truncating managed Postgres tables, including LUTs..."
  PGPASSWORD="$POSTGRES_PASSWORD" \
  psql -X -h "$POSTGRES_HOST" \
       -p "$POSTGRES_PORT" \
       -U "$POSTGRES_USER" \
       -d "$POSTGRES_DB" \
       --set=ON_ERROR_STOP=1 <<'SQL'
BEGIN;
TRUNCATE TABLE
    assets.trajectory_camera_frame_ref,
    assets.checkpoint_ref,
    assets.simulation_model_ref,
    workloads.user_metrics_lut,
    workloads.workload,
    workloads.workload_structure_lut,
    product.access_key,
    product.project,
    product."user",
    product."group";
COMMIT;
SQL
else
  log "Dropping Postgres database '$POSTGRES_DB'..."
  PGPASSWORD="$POSTGRES_PASSWORD" \
  psql -X -h "$POSTGRES_HOST" \
       -p "$POSTGRES_PORT" \
       -U "$POSTGRES_USER" \
       -d postgres \
       --set=ON_ERROR_STOP=1 \
       --set=target_db="$POSTGRES_DB" <<'SQL'
DROP DATABASE IF EXISTS :"target_db";
SQL
  log "Stopping Postgres..."
  if command -v brew >/dev/null 2>&1; then
    brew services stop postgresql@18 2>/dev/null \
      || brew services stop postgresql 2>/dev/null \
      || true
  else
    pg_ctl stop || true
  fi
  log "Postgres stopped."
fi

# ── MongoDB ──────────────────────────────────────────

if $ENTRIES_ONLY; then
  log "Clearing managed MongoDB collections..."
  MONGODB_DB="$MONGODB_DB" \
  mongosh --host "$MONGODB_HOST" \
          --port "$MONGODB_PORT" \
          --quiet --file /dev/stdin <<'MONGO'
const targetDb = db.getSiblingDB(process.env.MONGODB_DB);
for (const name of [
  'workload_logs',
  'workload_graph',
  'workload_user_metrics',
  'simulation_model_trajectory_state',
]) {
  const result = targetDb.getCollection(name).deleteMany({});
  if (!result.acknowledged) {
    throw new Error('Delete was not acknowledged: ' + name);
  }
  print('Cleared: ' + name);
}
MONGO
else
  log "Dropping MongoDB database '$MONGODB_DB'..."
  MONGODB_DB="$MONGODB_DB" \
  mongosh --host "$MONGODB_HOST" \
          --port "$MONGODB_PORT" \
          --quiet --file /dev/stdin <<'MONGO'
const targetDb = db.getSiblingDB(process.env.MONGODB_DB);
const result = targetDb.dropDatabase();
if (result.ok !== 1) {
  throw new Error(JSON.stringify(result));
}
print('Dropped database: ' + process.env.MONGODB_DB);
MONGO
  log "Stopping MongoDB..."
  if command -v brew >/dev/null 2>&1; then
    brew services stop mongodb-community \
      2>/dev/null || true
  else
    mongod --shutdown \
           --dbpath /var/lib/mongodb 2>/dev/null \
      || true
  fi
  log "MongoDB stopped."
fi

log "Done."
