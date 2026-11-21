#!/usr/bin/env bash
# Export OpenAPI without starting the app or connecting to services.
# Requires cargo and uv; the validator runs in an isolated uv tool env.
# Run from any directory; --check verifies the committed artifact.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

OUT="$REPO_ROOT/docs/user/api_spec/spec.json"
CHECK=false
if [[ "${1:-}" == "--check" && "$#" -eq 1 ]]; then
  CHECK=true
elif [[ "$#" -ne 0 ]]; then
  echo "Usage: $0 [--check]" >&2
  exit 1
fi

for command in cargo uvx; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "ERROR: $command not found in PATH" >&2
    exit 1
  }
done

mkdir -p "$(dirname "$OUT")"
TMP_SPEC="$(mktemp "${OUT}.XXXXXX")"
trap 'rm -f "$TMP_SPEC"' EXIT

# Keep app/.cargo/config.toml active and respect CARGO_TARGET_DIR.
(cd "$REPO_ROOT/app" && cargo run --quiet -- --export-openapi) \
  > "$TMP_SPEC"

uvx --python '>=3.10' --from openapi-spec-validator==0.7.2 \
  openapi-spec-validator "$TMP_SPEC"

if $CHECK; then
  if ! cmp -s "$TMP_SPEC" "$OUT"; then
    echo "OpenAPI spec is stale; run scripts/typegen/gen_api_spec.sh" >&2
    exit 1
  fi
  echo "OpenAPI spec is up to date."
elif ! cmp -s "$TMP_SPEC" "$OUT"; then
  chmod 644 "$TMP_SPEC"
  mv "$TMP_SPEC" "$OUT"
  echo "Spec written to $OUT"
else
  echo "OpenAPI spec is unchanged."
fi
