#!/usr/bin/env bash
# seed_db.sh
#
# Inserts the default user (id=1) and project (id=1)
# that the app server assumes on every run.
# Safe to run multiple times — skips rows that exist.
#
# See instructions/specs/db-env.md for env vars.

set -euo pipefail

POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-lagrangian}"

log() { echo "[seed_db] $*"; }

log "Seeding user id=1 and project id=1..."

psql -h "$POSTGRES_HOST" \
     -p "$POSTGRES_PORT" \
     -U "$(whoami)" \
     -d "$POSTGRES_DB" <<'SQL'

INSERT INTO users (user_id, created_at, name, email)
VALUES ('1', NOW(), 'Default User', 'user@lagrangian.local')
ON CONFLICT (user_id) DO NOTHING;

INSERT INTO projects (project_id, user_id, created_at)
VALUES ('1', 1, NOW())
ON CONFLICT (project_id) DO NOTHING;

SQL

log "Done."
