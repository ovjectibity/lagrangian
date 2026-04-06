#!/usr/bin/env bash
# init_db.sh
#
# Creates the Postgres database + schema and
# the MongoDB database + collections, then
# starts both servers (if not already running).
#
# Required env vars — see instructions/specs/db-env.md

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

# ── Helpers ──────────────────────────────────────────

log() { echo "[init_db] $*"; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: '$1' not found in PATH" >&2
    exit 1
  }
}

# ── Install dependencies (if missing) ────────────────

if command -v brew >/dev/null 2>&1; then
  if ! command -v psql >/dev/null 2>&1; then
    log "Installing PostgreSQL via Homebrew..."
    brew install postgresql
  fi

  if ! command -v mongosh >/dev/null 2>&1; then
    log "Installing MongoDB via Homebrew..."
    brew tap mongodb/brew 2>/dev/null || true
    brew install mongodb-community mongosh
  fi
else
  log "Homebrew not found — skipping automatic install. Please install PostgreSQL and MongoDB manually."
fi

# ── Pre-flight checks ────────────────────────────────

require_cmd psql
require_cmd mongosh

# ── Start Postgres (if not already running) ──────────

if ! pg_isready -h "$POSTGRES_HOST" \
                -p "$POSTGRES_PORT" \
                -q 2>/dev/null; then
  log "Starting Postgres..."
  if command -v brew >/dev/null 2>&1; then
    brew services start postgresql@17 \
      || brew services start postgresql
  else
    pg_ctl start
  fi
  # Wait until ready (up to 15 s)
  for i in $(seq 1 15); do
    pg_isready -h "$POSTGRES_HOST" \
               -p "$POSTGRES_PORT" -q \
      && break
    sleep 1
  done
else
  log "Postgres already running."
fi

# ── Create Postgres role + database (idempotent) ─────
# Connect as the current OS user (superuser) to bootstrap.

PGPASSWORD="" \
psql -h "$POSTGRES_HOST" \
     -p "$POSTGRES_PORT" \
     -U "$(whoami)" \
     -d postgres \
     -tc "SELECT 1 FROM pg_roles \
          WHERE rolname = '$POSTGRES_USER'" \
| grep -q 1 || {
  log "Creating Postgres role '$POSTGRES_USER'..."
  PGPASSWORD="" \
  psql -h "$POSTGRES_HOST" \
       -p "$POSTGRES_PORT" \
       -U "$(whoami)" \
       -d postgres \
       -c "CREATE ROLE $POSTGRES_USER \
           WITH LOGIN \
           PASSWORD '$POSTGRES_PASSWORD';"
}

PGPASSWORD="" \
psql -h "$POSTGRES_HOST" \
     -p "$POSTGRES_PORT" \
     -U "$(whoami)" \
     -d postgres \
     -tc "SELECT 1 FROM pg_database \
          WHERE datname = '$POSTGRES_DB'" \
| grep -q 1 || {
  log "Creating Postgres database '$POSTGRES_DB'..."
  PGPASSWORD="" \
  createdb -h "$POSTGRES_HOST" \
           -p "$POSTGRES_PORT" \
           -U "$(whoami)" \
           --owner="$POSTGRES_USER" \
           "$POSTGRES_DB"
}

# ── Apply Postgres schema ─────────────────────────────

log "Applying Postgres schema..."

PGPASSWORD="" \
psql -h "$POSTGRES_HOST" \
     -p "$POSTGRES_PORT" \
     -U "$(whoami)" \
     -d "$POSTGRES_DB" <<'SQL'

CREATE TABLE IF NOT EXISTS users (
    user_id    TEXT        PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    name       TEXT        NOT NULL,
    email      TEXT        NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT        PRIMARY KEY,
    user_id    BIGINT      NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS training_experiments (
    training_experiment_id TEXT        PRIMARY KEY,
    project_id             TEXT        NOT NULL,
    experiment_name        TEXT        NOT NULL,
    created_at             TIMESTAMPTZ NOT NULL,
    training_run_ids       TEXT[]      NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS training_runs (
    id                     TEXT        PRIMARY KEY,
    project_id             TEXT        NOT NULL,
    training_experiment_id TEXT        NOT NULL,
    created_at             TIMESTAMPTZ NOT NULL,
    run_start_at           TIMESTAMPTZ,
    run_end_at             TIMESTAMPTZ,
    run_status             TEXT        NOT NULL DEFAULT 'not_started',
    logs_id                TEXT        NOT NULL,
    metrics_id             TEXT        NOT NULL,
    training_graph_id      TEXT        NOT NULL,
    checkpoints_refs       TEXT[]      NOT NULL DEFAULT '{}',
    model_refs             TEXT[]      NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS model_references (
    id           TEXT        PRIMARY KEY,
    created_at   TIMESTAMPTZ NOT NULL,
    path         TEXT        NOT NULL,
    model_type   TEXT        NOT NULL
);

CREATE TABLE IF NOT EXISTS checkpoint_references (
    id                     TEXT        PRIMARY KEY,
    created_at             TIMESTAMPTZ NOT NULL,
    path                   TEXT        NOT NULL,
    checkpoint_type        TEXT        NOT NULL,
    training_experiment_id TEXT        NOT NULL,
    training_run_id        TEXT        NOT NULL,
    control_model_params   TEXT,
    value_model_params     TEXT
);

GRANT ALL PRIVILEGES ON ALL TABLES
    IN SCHEMA public TO $POSTGRES_USER;

GRANT ALL PRIVILEGES ON ALL SEQUENCES
    IN SCHEMA public TO $POSTGRES_USER;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT ALL PRIVILEGES ON TABLES
    TO $POSTGRES_USER;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT ALL PRIVILEGES ON SEQUENCES
    TO $POSTGRES_USER;

SQL

log "Postgres schema applied."

# ── Start MongoDB (if not already running) ───────────

if ! mongosh --host "$MONGODB_HOST" \
             --port "$MONGODB_PORT" \
             --eval "db.runCommand({ping:1})" \
             --quiet >/dev/null 2>&1; then
  log "Starting MongoDB..."
  if command -v brew >/dev/null 2>&1; then
    # restart handles both the stopped and
    # error states that brew services reports
    brew services restart mongodb-community
  else
    mongod --fork \
           --logpath /tmp/mongod.log \
           --dbpath /var/lib/mongodb
  fi
  # Wait until ready (up to 15 s)
  for i in $(seq 1 15); do
    mongosh --host "$MONGODB_HOST" \
            --port "$MONGODB_PORT" \
            --eval "db.runCommand({ping:1})" \
            --quiet >/dev/null 2>&1 \
      && break
    sleep 1
  done
else
  log "MongoDB already running."
fi

# ── Create MongoDB collections (idempotent) ──────────

log "Initialising MongoDB database '$MONGODB_DB'..."

mongosh --host "$MONGODB_HOST" \
        --port "$MONGODB_PORT" \
        --quiet \
        --eval "
const db = connect('$MONGODB_HOST:$MONGODB_PORT/$MONGODB_DB');

const cols = [
  'training_graphs',
  'log_streams',
  'metrics_streams',
];

for (const name of cols) {
  const existing = db.getCollectionNames();
  if (!existing.includes(name)) {
    db.createCollection(name);
    print('Created collection: ' + name);
  } else {
    print('Collection already exists: ' + name);
  }
}
"

log "MongoDB initialised."
log "Done."

echo ""
echo "Connection strings:"
echo "  DATABASE_URL=postgres://$POSTGRES_USER:$POSTGRES_PASSWORD@$POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB"
echo "  MONGODB_URI=mongodb://$MONGODB_HOST:$MONGODB_PORT"
