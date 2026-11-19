---
description: Rules for shared types across components (Rust, TypeScript, Python). Use when adding, editing, or generating shared types, or when working with commondef, gen_types, remoteworkermessages, or training types.
---

## Source of truth

`app/src/commondef/` is the single source of truth for all types shared between components (Rust backend, TypeScript client, Python remote worker). Never define shared types anywhere else.

## Type generation

Run `scripts/typegen/gen_types.sh` to regenerate worker types. Client/API generation is deferred. Never edit these generated files manually:

- `worker/worker-py/commondef/records/*.py`
- `worker/worker-py/commondef/remoteworker/*.py`

```bash
./scripts/typegen/gen_types.sh
```
