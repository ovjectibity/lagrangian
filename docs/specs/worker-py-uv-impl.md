# Python worker uv migration implementation plan

1. Add a non-package `pyproject.toml` containing the dependencies currently
   declared in `requirements.txt` and requiring Python 3.14 or newer.
2. Generate and commit `uv.lock` from that project definition.
3. Copy a pinned uv executable into the CUDA image and configure uv to use the
   system-provided Python interpreter.
4. Replace the manually created virtual environment and pip installation with a
   cached, locked `uv sync` layer.
5. Put uv's project virtual environment on `PATH`, leaving the existing worker
   command unchanged.
6. Add Docker-context exclusions and remove `requirements.txt`.
7. Validate the project lock and Dockerfile configuration.
