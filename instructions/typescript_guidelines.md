# TypeScript Code Style Guide for Coding Agents

This document defines the standard coding style, formatting rules, and idiomatic practices for AI agents generating TypeScript code. Adherence to these guidelines ensures code is maintainable, type-safe, and compatible with modern TypeScript tooling.

## 1. The Golden Rule: Strictness & Safety
Agents must produce code that compiles under `strict` mode.
* **Strict Null Checks:** Do not assume values are non-null. Handle `null` and `undefined` explicitly.
* **No Implicit Any:** Avoid `any` at all costs. Use `unknown` if the type is truly uncertain, or generics if applicable.
* **Linter:** Code should generally pass standard ESLint + Prettier configurations.

## 2. Naming Conventions

| Item | Case Style | Example |
| :--- | :--- | :--- |
| **Variables** | `camelCase` | `userId`, `isLoading` |
| **Functions** | `camelCase` | `fetchUserData()`, `calculateTotal()` |
| **Classes** | `PascalCase` | `UserManager`, `HttpClient` |
| **Interfaces** | `PascalCase` | `User`, `ApiResponse` (No `I` prefix!) |
| **Types** | `PascalCase` | `UserRole`, `ConfigOptions` |
| **Constants** | `SCREAMING_SNAKE` | `MAX_RETRIES`, `API_ENDPOINT` |
| **Files** | `kebab-case` | `user-profile.ts`, `api-client.ts` |
| **React Components** | `PascalCase` | `UserProfile.tsx`, `SubmitButton.tsx` |

**Note on Interfaces:** Do NOT use Hungarian notation (e.g., `IUser`). Just use `User`.

## 3. Formatting & Layout

### Indentation and Spacing
* **Indentation:** Use **2 spaces**. Do not use tabs.
* **Line Length:** Aim for **80-100 characters**.
* **Quotes:** Use double quotes `"` by default (unless project specifies single).
* **Semicolons:** **Always** use semicolons.

```typescript
// ✅ Correct
function getUser(id: string): User {
  if (!id) {
    return null;
  }
  return db.find(id);
}

// ❌ Incorrect
function getUser(id:string):User // Bad spacing
{ // Brace on new line
    if(!id) return null // Missing semicolon, one-liner
}
```

### Imports
Group imports in the following order:
1.  **External Libraries** (React, Lodash, etc.)
2.  **Absolute/Alias Imports** (`@/components/...`)
3.  **Relative Imports** (`./utils`, `../types`)

```typescript
import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { Button } from "@/components/ui/button";
import { formatDate } from "./date-utils";
```

## 4. Idiomatic TypeScript

### Type Definitions
* **Prefer `interface`** for object definitions (better error messages, extendable).
* **Use `type`** for unions, intersections, primitives, or tuples.

```typescript
// ✅ Interface for objects
interface User {
  id: string;
  name: string;
}

// ✅ Type for unions
type Status = "idle" | "loading" | "error";
```

### Functions
* Use **Arrow Functions** for callbacks and functional components.
* Use explicit return types for exported functions to prevent accidental API changes.

```typescript
// ✅ Explicit return type
export const sum = (a: number, b: number): number => {
  return a + b;
};
```

### Optional Chaining & Nullish Coalescing
* Use `?.` for safe property access.
* Use `??` for providing default values (avoid `||` which can trigger on falsy `0` or `""`).

```typescript
// ✅ Correct
const width = config?.dimensions?.width ?? 100;

// ❌ Incorrect
const width = config && config.dimensions && config.dimensions.width || 100;
```

### Immutability
* Prefer `const` over `let`. Never use `var`.
* Use `ReadonlyArray<T>` or `readonly T[]` for function parameters if the array should not be modified.

## 5. React Specifics (If applicable)

* **Functional Components:** define as `React.FC` or standard functions.
* **Props:** Destructure props in the function signature.
* **Hooks:** Follow the Rules of Hooks (top level only).

```typescript
interface ButtonProps {
  label: string;
  onClick: () => void;
}

export const Button = ({ label, onClick }: ButtonProps) => {
  return <button onClick={onClick}>{label}</button>;
};
```

## 6. Agent-Specific Instructions
1.  **Completeness:** Provide fully working code. Do not leave "TODO" comments for critical logic.
2.  **Explicit Types:** Do not rely on implicit `any`. If a type is complex, define it separately.
3.  **Async/Await:** Prefer `async/await` over `.then()` chains.
4.  **Error Handling:** Use `try/catch` blocks for async operations and handle errors gracefully.
5.  **Post-edit Validation (Required):** After TypeScript/Svelte edits in `client`, run:
    * `npm run check`
    * `npm run build`
    from the `client` directory before finalizing changes. Report failures clearly.
6.  **Centralized CSS (Required):** Keep component styling in `client/src/app.css`. Do not add inline `<style>` blocks in Svelte components unless explicitly requested.

---
### Appendix: Minimal `.prettierrc`
```json
{
  "semi": true,
  "trailingComma": "all",
  "singleQuote": false,
  "printWidth": 100,
  "tabWidth": 2
}
```
