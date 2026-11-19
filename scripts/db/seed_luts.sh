#!/usr/bin/env bash
# Seed the static LUT definitions from the Notion data model.
# Existing IDs are immutable: repeated runs leave them untouched.
# Run init_db.sh first to create the schemas and tables.

set -euo pipefail

POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-lagrangian}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-lagrangian}"
POSTGRES_DB="${POSTGRES_DB:-lagrangian}"

command -v psql >/dev/null 2>&1 || {
  echo "ERROR: psql not found in PATH" >&2
  exit 1
}

echo "[seed_luts] Seeding static workload LUTs..."

PGPASSWORD="$POSTGRES_PASSWORD" \
psql -X -h "$POSTGRES_HOST" \
     -p "$POSTGRES_PORT" \
     -U "$POSTGRES_USER" \
     -d "$POSTGRES_DB" \
     --set=ON_ERROR_STOP=1 <<'SQL'

BEGIN;

-- Names are ordered; aggregation/NA coordinates use zero-based
-- positions. Notion's '-' entries represent empty arrays.
INSERT INTO workloads.workload_structure_lut (
    workload_structure_id,
    workload_type,
    method_applicability,
    number_of_state_indices,
    workload_state_index_names
) VALUES
    (
        1, 'training', 'ppo', 3,
        ARRAY[
            'rollout', 'rollout_trajectory',
            'rollout_trajectory_timestep'
        ]
    ),
    (
        2, 'training', 'ppo', 5,
        ARRAY[
            'rollout', 'epoch', 'minibatch',
            'trajectory', 'trajectory_timestep'
        ]
    ),
    (
        3, 'testing', 'dataset', 3,
        ARRAY['task_index', 'trajectory', 'trajectory_timestep']
    ),
    (
        4, 'testing', 'custom', 3,
        ARRAY['task_index', 'trajectory', 'trajectory_timestep']
    )
ON CONFLICT (workload_structure_id) DO NOTHING;

-- Keep the published IDs, including the gap at metric ID 7.
INSERT INTO workloads.user_metrics_lut (
    metric_id,
    workload_type,
    name,
    workload_structure_id,
    aggregation_functions,
    aggregation_indices,
    na_indices
) VALUES
    (
        1, 'training', 'Mean rollout trajectory reward', 1,
        ARRAY['mean'], ARRAY[2]::BIGINT[], ARRAY[]::BIGINT[]
    ),
    (
        2, 'training', 'PPO clip objective', 2,
        ARRAY['mean', 'mean'], ARRAY[3, 4]::BIGINT[],
        ARRAY[]::BIGINT[]
    ),
    (
        3, 'training', 'Mean policy gradient norm', 2,
        ARRAY[]::TEXT[], ARRAY[]::BIGINT[], ARRAY[3, 4]::BIGINT[]
    ),
    (
        4, 'training', 'Mean value gradient norm', 2,
        ARRAY[]::TEXT[], ARRAY[]::BIGINT[], ARRAY[3, 4]::BIGINT[]
    ),
    (
        5, 'training', 'Mean value loss', 2,
        ARRAY['mean', 'mean'], ARRAY[3, 4]::BIGINT[],
        ARRAY[]::BIGINT[]
    ),
    (
        6, 'training', 'Mean rollout policy entropy', 1,
        ARRAY['mean', 'mean'], ARRAY[1, 2]::BIGINT[],
        ARRAY[]::BIGINT[]
    ),
    (
        8, 'testing', 'Mean trajectory reward', 3,
        ARRAY['mean'], ARRAY[2]::BIGINT[], ARRAY[]::BIGINT[]
    ),
    (
        9, 'testing', 'Reward', 3,
        ARRAY[]::TEXT[], ARRAY[]::BIGINT[], ARRAY[]::BIGINT[]
    ),
    (
        10, 'testing', 'Mean policy entropy', 3,
        ARRAY['mean', 'mean'], ARRAY[1, 2]::BIGINT[],
        ARRAY[]::BIGINT[]
    ),
    (
        11, 'testing', 'Mean trajectory reward', 4,
        ARRAY['mean'], ARRAY[2]::BIGINT[], ARRAY[]::BIGINT[]
    ),
    (
        12, 'testing', 'Reward', 4,
        ARRAY[]::TEXT[], ARRAY[]::BIGINT[], ARRAY[]::BIGINT[]
    ),
    (
        13, 'testing', 'Mean policy entropy', 4,
        ARRAY['mean', 'mean'], ARRAY[1, 2]::BIGINT[],
        ARRAY[]::BIGINT[]
    )
ON CONFLICT (metric_id) DO NOTHING;

COMMIT;
SQL

echo "[seed_luts] Done."
