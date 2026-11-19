#!/usr/bin/env bash
# init_db.sh
#
# Starts Postgres and MongoDB if needed, then creates the
# databases, tables, collection validators, and indexes.
# Existing legacy tables/collections are left untouched;
# this script does not migrate their data.
#
# Required env vars — see docs/db-env.md

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
    brew services start postgresql@18 \
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
     -d "$POSTGRES_DB" \
     --set=ON_ERROR_STOP=1 \
     --set=app_role="$POSTGRES_USER" <<'SQL'

BEGIN;

CREATE SCHEMA IF NOT EXISTS product;
CREATE SCHEMA IF NOT EXISTS assets;
CREATE SCHEMA IF NOT EXISTS workloads;

CREATE TABLE IF NOT EXISTS product."group" (
    group_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    login_type TEXT NOT NULL CHECK (
        login_type IN ('direct', 'sso_google', 'sso_github')
    )
);

CREATE TABLE IF NOT EXISTS product."user" (
    user_id TEXT PRIMARY KEY,
    group_id TEXT NOT NULL REFERENCES product."group"(group_id),
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin'))
);

CREATE TABLE IF NOT EXISTS product.access_key (
    access_key_id TEXT PRIMARY KEY,
    access_key_value TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL REFERENCES product."user"(user_id),
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('expired', 'active', 'deleted')
    )
);

CREATE TABLE IF NOT EXISTS product.project (
    project_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES product."user"(user_id),
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workloads.workload_structure_lut (
    workload_structure_id INTEGER PRIMARY KEY,
    workload_type TEXT NOT NULL CHECK (
        workload_type IN ('training', 'testing')
    ),
    workload_state_index_names TEXT[] NOT NULL CHECK (
        array_position(workload_state_index_names, NULL) IS NULL
    ),
    -- Rust u32 values use BIGINT with an explicit range check.
    number_of_state_indices BIGINT NOT NULL CHECK (
        number_of_state_indices BETWEEN 0 AND 4294967295
    ),
    method_applicability TEXT NOT NULL CHECK (
        method_applicability IN ('ppo', 'dataset', 'custom')
    ),
    CHECK (
        cardinality(workload_state_index_names)
            = number_of_state_indices
    ),
    UNIQUE (workload_structure_id, workload_type)
);

CREATE TABLE IF NOT EXISTS workloads.user_metrics_lut (
    metric_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    workload_structure_id INTEGER NOT NULL,
    workload_type TEXT NOT NULL CHECK (
        workload_type IN ('training', 'testing')
    ),
    aggregation_functions TEXT[] NOT NULL CHECK (
        aggregation_functions <@ ARRAY['mean']::TEXT[]
        AND array_position(aggregation_functions, NULL) IS NULL
    ),
    aggregation_indices BIGINT[] NOT NULL CHECK (
        0 <= ALL(aggregation_indices)
        AND 4294967295 >= ALL(aggregation_indices)
        AND array_position(aggregation_indices, NULL) IS NULL
    ),
    na_indices BIGINT[] NOT NULL CHECK (
        0 <= ALL(na_indices)
        AND 4294967295 >= ALL(na_indices)
        AND array_position(na_indices, NULL) IS NULL
    ),
    CHECK (
        cardinality(aggregation_functions)
            = cardinality(aggregation_indices)
    ),
    CHECK (NOT (aggregation_indices && na_indices)),
    FOREIGN KEY (workload_structure_id, workload_type)
        REFERENCES workloads.workload_structure_lut (
            workload_structure_id, workload_type
        )
);

-- Bounds depend on another table, so enforce them with triggers.
CREATE OR REPLACE FUNCTION workloads.validate_metric_indices()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
DECLARE
    dimension_count BIGINT;
BEGIN
    SELECT number_of_state_indices INTO dimension_count
    FROM workloads.workload_structure_lut
    WHERE workload_structure_id = NEW.workload_structure_id
    FOR SHARE;

    IF EXISTS (
        SELECT 1
        FROM unnest(NEW.aggregation_indices || NEW.na_indices) AS i
        WHERE i < 0 OR i >= dimension_count
    ) THEN
        RAISE EXCEPTION 'Metric indices exceed structure dimensions'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS validate_metric_indices
    ON workloads.user_metrics_lut;
CREATE TRIGGER validate_metric_indices
BEFORE INSERT OR UPDATE ON workloads.user_metrics_lut
FOR EACH ROW EXECUTE FUNCTION workloads.validate_metric_indices();

CREATE OR REPLACE FUNCTION workloads.validate_structure_dimensions()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM workloads.user_metrics_lut AS metric
        CROSS JOIN LATERAL unnest(
            metric.aggregation_indices || metric.na_indices
        ) AS i
        WHERE metric.workload_structure_id = OLD.workload_structure_id
            AND i >= NEW.number_of_state_indices
    ) THEN
        RAISE EXCEPTION 'Structure dimensions invalidate metric indices'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS validate_structure_dimensions
    ON workloads.workload_structure_lut;
CREATE TRIGGER validate_structure_dimensions
BEFORE UPDATE OF number_of_state_indices
ON workloads.workload_structure_lut
FOR EACH ROW EXECUTE FUNCTION workloads.validate_structure_dimensions();

CREATE TABLE IF NOT EXISTS workloads.workload (
    workload_id TEXT PRIMARY KEY,
    workload_type TEXT NOT NULL CHECK (
        workload_type IN ('training', 'testing')
    ),
    project_id TEXT NOT NULL REFERENCES product.project(project_id),
    user_id TEXT NOT NULL REFERENCES product."user"(user_id),
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    launched_at TIMESTAMPTZ,
    launch_origin TEXT NOT NULL CHECK (launch_origin IN ('api')),
    scheduled_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    status TEXT NOT NULL CHECK (
        status IN (
            'completed', 'execution_failed', 'launch_failed',
            'scheduled', 'ongoing'
        )
    ),
    -- Graph documents live in Mongo; the ORM assembles Workload.
    workload_graph_id TEXT NOT NULL,
    tags TEXT[] NOT NULL CHECK (array_position(tags, NULL) IS NULL)
);

CREATE TABLE IF NOT EXISTS assets.simulation_model_ref (
    simulation_model_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    name TEXT NOT NULL,
    uri_path TEXT NOT NULL,
    model_type TEXT NOT NULL CHECK (
        model_type IN ('mjcf', 'urdf', 'usd')
    ),
    file_type TEXT NOT NULL CHECK (file_type IN ('zip')),
    base_file_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assets.checkpoint_ref (
    checkpoint_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    name TEXT NOT NULL,
    uri_path TEXT NOT NULL,
    workload_id TEXT REFERENCES workloads.workload(workload_id),
    workload_structure_id INTEGER REFERENCES
        workloads.workload_structure_lut(workload_structure_id),
    workload_structure_state BIGINT[] CHECK (
        0 <= ALL(workload_structure_state)
        AND 4294967295 >= ALL(workload_structure_state)
        AND array_position(workload_structure_state, NULL) IS NULL
    ),
    model_type TEXT NOT NULL CHECK (
        model_type IN ('policy', 'value', 'flow_noise')
    ),
    model_spec JSONB NOT NULL CHECK (
        jsonb_typeof(model_spec) = 'object'
    )
);

CREATE TABLE IF NOT EXISTS assets.trajectory_camera_frame_ref (
    camera_frame_id TEXT PRIMARY KEY,
    workload_id TEXT NOT NULL REFERENCES workloads.workload(workload_id),
    workload_structure_id INTEGER NOT NULL REFERENCES
        workloads.workload_structure_lut(workload_structure_id),
    workload_structure_state BIGINT[] NOT NULL CHECK (
        0 <= ALL(workload_structure_state)
        AND 4294967295 >= ALL(workload_structure_state)
        AND array_position(workload_structure_state, NULL) IS NULL
    ),
    camera_id TEXT NOT NULL,
    model_camera_name TEXT NOT NULL,
    uri_path TEXT NOT NULL,
    simulation_model_id TEXT NOT NULL REFERENCES
        assets.simulation_model_ref(simulation_model_id)
);

CREATE INDEX IF NOT EXISTS user_group_idx
    ON product."user"(group_id);
CREATE INDEX IF NOT EXISTS access_key_user_idx
    ON product.access_key(user_id);
CREATE INDEX IF NOT EXISTS project_user_idx
    ON product.project(user_id);
CREATE INDEX IF NOT EXISTS workload_project_idx
    ON workloads.workload(project_id);
CREATE INDEX IF NOT EXISTS workload_user_idx
    ON workloads.workload(user_id);
CREATE INDEX IF NOT EXISTS user_metrics_structure_idx
    ON workloads.user_metrics_lut(workload_structure_id);
CREATE INDEX IF NOT EXISTS checkpoint_workload_idx
    ON assets.checkpoint_ref(workload_id);
CREATE INDEX IF NOT EXISTS camera_frame_workload_idx
    ON assets.trajectory_camera_frame_ref(workload_id);

GRANT USAGE ON SCHEMA product, assets, workloads TO :"app_role";

GRANT ALL PRIVILEGES ON ALL TABLES
    IN SCHEMA product, assets, workloads TO :"app_role";

GRANT ALL PRIVILEGES ON ALL SEQUENCES
    IN SCHEMA product, assets, workloads TO :"app_role";

ALTER DEFAULT PRIVILEGES IN SCHEMA product, assets, workloads
    GRANT ALL PRIVILEGES ON TABLES
    TO :"app_role";

ALTER DEFAULT PRIVILEGES IN SCHEMA product, assets, workloads
    GRANT ALL PRIVILEGES ON SEQUENCES
    TO :"app_role";

COMMIT;

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

# ── Create/update MongoDB validators and indexes ─────

log "Initialising MongoDB database '$MONGODB_DB'..."

MONGODB_DB="$MONGODB_DB" \
mongosh --host "$MONGODB_HOST" \
        --port "$MONGODB_PORT" \
        --quiet \
        --file /dev/stdin <<'MONGO'
const targetDb = db.getSiblingDB(process.env.MONGODB_DB);

const string = { bsonType: 'string' };
// The ORM must convert Chrono timestamps to native BSON dates.
const date = { bsonType: 'date' };
const integer = { bsonType: 'int' };
const coordinate = {
  bsonType: ['int', 'long'],
  minimum: 0,
  maximum: 4294967295,
};
const coordinates = { bsonType: 'array', items: coordinate };
const numberArray = {
  bsonType: 'array',
  items: { bsonType: ['double', 'int', 'long'] },
};
// Nested graph and WorldBody contents are validated by Rust.
const collections = [
  {
    name: 'workload_logs',
    id: 'log_id',
    properties: {
      workload_id: string,
      created_at: date,
      log_timestamp: date,
      log_id: string,
      log_line_text: string,
      user_facing: { bsonType: 'bool' },
      log_type: {
        enum: ['info', 'debug', 'trace', 'error', 'warning'],
      },
      stream: { enum: ['app_server', 'worker'] },
    },
    indexes: [{ workload_id: 1, log_timestamp: 1 }],
  },
  {
    name: 'workload_graph',
    id: 'workload_graph_id',
    properties: {
      workload_graph_id: string,
      graph: { bsonType: 'object' },
      workload_type: { enum: ['training', 'testing'] },
      workload_id: string,
      created_at: date,
    },
    indexes: [{ workload_id: 1 }],
  },
  {
    name: 'workload_user_metrics',
    id: 'metrics_entry_id',
    properties: {
      metrics_entry_id: string,
      created_at: date,
      workload_id: string,
      metric_id: integer,
      workload_structure_state_values: {
        bsonType: 'array',
        items: coordinates,
      },
      metric_values: numberArray,
    },
    // Guard $size so malformed writes are rejected cleanly.
    expression: {
      $cond: [
        {
          $and: [
            { $isArray: '$workload_structure_state_values' },
            { $isArray: '$metric_values' },
          ],
        },
        {
          $eq: [
            { $size: '$workload_structure_state_values' },
            { $size: '$metric_values' },
          ],
        },
        false,
      ],
    },
    indexes: [{ workload_id: 1, metric_id: 1 }],
  },
  {
    name: 'simulation_model_trajectory_state',
    id: 'sim_model_state_entry_id',
    properties: {
      created_at: date,
      sim_model_state_entry_id: string,
      workload_id: string,
      workload_structure_id: integer,
      workload_structure_state: coordinates,
      simulation_model_id: string,
      state: { bsonType: 'object' },
    },
    indexes: [{ workload_id: 1, simulation_model_id: 1 }],
  },
];

for (const collection of collections) {
  const schema = {
    bsonType: 'object',
    required: Object.keys(collection.properties),
    additionalProperties: false,
    properties: { _id: {}, ...collection.properties },
  };
  const validator = collection.expression
    ? { $and: [
        { $jsonSchema: schema },
        { $expr: collection.expression },
      ] }
    : { $jsonSchema: schema };
  const options = {
    validator,
    validationLevel: 'strict',
    validationAction: 'error',
  };

  let result;
  if (targetDb.getCollectionNames().includes(collection.name)) {
    // strict validation covers subsequent writes; existing
    // documents need a separate audit/migration.
    result = targetDb.runCommand({
      collMod: collection.name,
      ...options,
    });
  } else {
    result = targetDb.createCollection(
      collection.name, options,
    );
  }
  if (result.ok !== 1) {
    throw new Error(JSON.stringify(result));
  }
  const stored = targetDb.getCollection(collection.name);
  stored.createIndex(
    { [collection.id]: 1 },
    { unique: true, name: collection.id + '_unique' },
  );
  for (const index of collection.indexes) {
    stored.createIndex(index);
  }
  print('Initialised collection: ' + collection.name);
}
MONGO

log "MongoDB initialised."

POSTGRES_HOST="$POSTGRES_HOST" \
POSTGRES_PORT="$POSTGRES_PORT" \
POSTGRES_USER="$POSTGRES_USER" \
POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
POSTGRES_DB="$POSTGRES_DB" \
bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/seed_luts.sh"

log "Done."

echo ""
echo "Connection strings:"
echo "  DATABASE_URL=postgres://$POSTGRES_USER:$POSTGRES_PASSWORD@$POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB"
echo "  MONGODB_URI=mongodb://$MONGODB_HOST:$MONGODB_PORT"
