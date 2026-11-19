#!/usr/bin/env bash
# seed_db.sh
#
# Inserts one development group, user, project, and access key.
# Each record uses the string ID '1'.
# Safe to run multiple times — skips rows that exist.
#
# See docs/db-env.md for env vars.

set -euo pipefail

POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-lagrangian}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-lagrangian}"
POSTGRES_DB="${POSTGRES_DB:-lagrangian}"

log() { echo "[seed_db] $*"; }

log "Seeding one group, user, project, and access key..."

PGPASSWORD="$POSTGRES_PASSWORD" \
psql -X -h "$POSTGRES_HOST" \
     -p "$POSTGRES_PORT" \
     -U "$POSTGRES_USER" \
     -d "$POSTGRES_DB" \
     --set=ON_ERROR_STOP=1 <<'SQL'

BEGIN;

INSERT INTO product."group" (
    group_id, name, created_at, updated_at, login_type
) VALUES (
    '1', 'Default Group', NOW(), NOW(), 'direct'
)
ON CONFLICT (group_id) DO NOTHING;

INSERT INTO product."user" (
    user_id, group_id, created_at, updated_at, name, email, role
) VALUES (
    '1', '1', NOW(), NOW(), 'Default User',
    'user@lagrangian.local', 'admin'
)
ON CONFLICT (user_id) DO NOTHING;

INSERT INTO product.project (
    project_id, user_id, created_at, updated_at, name
) VALUES (
    '1', '1', NOW(), NOW(), 'Default Project'
)
ON CONFLICT (project_id) DO NOTHING;

INSERT INTO product.access_key (
    access_key_id, access_key_value, user_id, name,
    created_at, status
) VALUES (
    '1', 'lagrangian-local-access-key', '1', 'Default Access Key',
    NOW(), 'active'
)
ON CONFLICT (access_key_id) DO NOTHING;

COMMIT;
SQL

log "Done."
