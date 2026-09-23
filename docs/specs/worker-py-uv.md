# Python worker uv migration

## Objective

Replace the Python worker's pip and `requirements.txt` dependency workflow with a
native uv project.

## Requirements

- Declare the worker runtime and dependencies in `worker/worker-py/pyproject.toml`.
- Commit `worker/worker-py/uv.lock` for reproducible container builds.
- Install dependencies in the worker image with `uv sync --locked`.
- Continue using the Python 3.14 interpreter supplied by the CUDA base image's
  Ubuntu packages.
- Preserve Docker layer caching by installing dependencies before copying worker
  source code.
- Exclude local virtual environments and Python-generated files from the Docker
  build context.
- Remove the superseded `worker/worker-py/requirements.txt` file.

## Non-goals

- Packaging the worker as an installable Python distribution.
- Changing worker runtime behavior or dependency versions.
- Migrating test-only Python requirements outside `worker/worker-py`.
