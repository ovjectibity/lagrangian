#!/usr/bin/env bash
# purge_db.sh
#
# Default: clears all data, drops tables/collections,
#          and stops both DB servers.
#
# --purge-entries-only: truncates all tables and clears
#   all MongoDB collections but leaves the schema intact
#   and servers running.
#
# See instructions/specs/db-env.md for env vars.

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

POSTGRES_TABLES=(
  training_runs
  training_experiments
  checkpoint_references
  model_references
  projects
  users
)

if $ENTRIES_ONLY; then
  log "Truncating all Postgres tables..."
  for table in "${POSTGRES_TABLES[@]}"; do
    PGPASSWORD="$POSTGRES_PASSWORD" \
    psql -h "$POSTGRES_HOST" \
         -p "$POSTGRES_PORT" \
         -U "$POSTGRES_USER" \
         -d "$POSTGRES_DB" \
         -c "TRUNCATE TABLE $table CASCADE;" \
      && log "  Truncated: $table"
  done
else
  log "Dropping Postgres database '$POSTGRES_DB'..."
  PGPASSWORD="$POSTGRES_PASSWORD" \
  psql -h "$POSTGRES_HOST" \
       -p "$POSTGRES_PORT" \
       -U "$POSTGRES_USER" \
       -d postgres \
       -c "DROP DATABASE IF EXISTS $POSTGRES_DB;"
  log "Stopping Postgres..."
  if command -v brew >/dev/null 2>&1; then
    brew services stop postgresql@17 2>/dev/null \
      || brew services stop postgresql 2>/dev/null \
      || true
  else
    pg_ctl stop || true
  fi
  log "Postgres stopped."
fi

# ── MongoDB ──────────────────────────────────────────

MONGO_COLLECTIONS=(
  training_graphs
  log_streams
  metrics_streams
)

if $ENTRIES_ONLY; then
  log "Clearing all MongoDB collections..."
  for col in "${MONGO_COLLECTIONS[@]}"; do
    mongosh --host "$MONGODB_HOST" \
            --port "$MONGODB_PORT" \
            --quiet \
            --eval "
db = connect('$MONGODB_HOST:$MONGODB_PORT/$MONGODB_DB');
db['$col'].deleteMany({});
print('Cleared: $col');
"
  done
else
  log "Dropping MongoDB database '$MONGODB_DB'..."
  mongosh --host "$MONGODB_HOST" \
          --port "$MONGODB_PORT" \
          --quiet \
          --eval "
db = connect('$MONGODB_HOST:$MONGODB_PORT/$MONGODB_DB');
db.dropDatabase();
print('Dropped database: $MONGODB_DB');
"
  log "Stopping MongoDB..."
  if command -v brew >/dev/null 2>&1; then
    brew services stop mongodb-community 2>/dev/null \
      || brew services stop mongodb 2>/dev/null \
      || true
  else
    mongod --shutdown \
           --dbpath /var/lib/mongodb 2>/dev/null \
      || true
  fi
  log "MongoDB stopped."
fi

log "Done."
