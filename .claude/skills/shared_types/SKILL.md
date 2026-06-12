---
description: Rules for shared types across components (Rust, TypeScript, Python). Use when adding, editing, or generating shared types, or when working with commondef, gen_types, remoteworkermessages, or training types.
---

## Source of truth

`app/src/commondef/` is the single source of truth for all types shared between components (Rust backend, TypeScript client/SDK, Python remote worker). Never define shared types anywhere else.

## Type generation

Always run `scripts/gen_types.sh` from the project root to regenerate derived type files. Never edit these generated files manually:

- `client/src/common/types.ts`
- `sdk/src/types.ts`
- `worker/trainer/trainer-py/commondef/remoteworkermessages.py`
- `worker/trainer/trainer-py/commondef/training.py` — re-export shim only; delegates entirely to `remoteworkermessages.py`

```bash
./scripts/gen_types.sh
```
