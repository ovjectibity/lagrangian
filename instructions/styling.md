# Rust Code Style Guide for Coding Agents

This document defines the standard coding style, formatting rules, and idiomatic practices for AI agents generating Rust code. Adherence to these guidelines ensures code is idiomatic, maintainable, and compatible with the standard Rust ecosystem.

## 1. The Golden Rule: Tooling Compliance
Agents must produce code that passes standard Rust tooling without warnings or errors.
* **Formatter:** Code must comply with `rustfmt` defaults.
* **Linter:** Code must pass `cargo clippy` without warnings.
* **Compiler:** Code must compile on the latest stable Rust version.

## 2. Naming Conventions
Rust uses specific casing for different identifiers. Agents must strictly follow this table:

| Item | Case Style | Example |
| :--- | :--- | :--- |
| **Variables** | `snake_case` | `user_id`, `buffer_size` |
| **Functions** | `snake_case` | `calculate_hash()`, `fetch_url()` |
| **Structs / Enums** | `UpperCamelCase` | `HttpRequest`, `json::Value` |
| **Traits** | `UpperCamelCase` | `ToString`, `Display` |
| **Constants** | `SCREAMING_SNAKE_CASE` | `MAX_RETRIES`, `DEFAULT_TIMEOUT` |
| **Statics** | `SCREAMING_SNAKE_CASE` | `GLOBAL_CONFIG` |
| **Modules** | `snake_case` | `http_client`, `utils` |
| **Files** | `snake_case` | `main.rs`, `lib.rs`, `user_model.rs` |

## 3. Code formatting & Layout

### Indentation and Spacing
* **Indentation:** Use **4 spaces**. Do not use tabs.
* **Line Length:** Aim for **100 characters** (soft limit). Break long chains if necessary.
* **Blocks:** Open braces `{` go on the **same line** as the statement.

```rust
// ✅ Correct
fn process_data(data: &[u8]) {
    if data.is_empty() {
        return;
    }
}

// ❌ Incorrect
fn process_data(data: &[u8])
{ // Brace on new line
  if data.is_empty() { return; } // One-liner if
}
```

### Import Ordering (`use`)
Group imports in the following order, separated by a blank line:
1.  **Standard Library** (`std`)
2.  **Third-party Crates** (External dependencies)
3.  **Local Modules** (`crate`, `super`, `self`)

```rust
use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::models::User;
```

## 4. Idiomatic Rust (The "Rust Way")

### Error Handling
* **Avoid Panics:** Never use `.unwrap()` or `.expect()` in production logic or library code.
* **Use Propagation:** Prefer the `?` operator to propagate errors.
* **Custom Errors:** For libraries, define a custom `Error` enum (using `thiserror` is acceptable).

```rust
// ✅ Correct
fn read_file() -> Result<String, io::Error> {
    let content = fs::read_to_string("config.toml")?;
    Ok(content)
}

// ❌ Incorrect
fn read_file() -> String {
    fs::read_to_string("config.toml").unwrap() // Dangerous panic risk
}
```

### Iterators vs Loops
Prefer high-level **Iterators** (`map`, `filter`, `fold`) over C-style `for` or `while` loops when transforming collections.

```rust
// ✅ Correct
let squares: Vec<i32> = numbers.iter().map(|x| x * x).collect();

// ❌ Incorrect
let mut squares = Vec::new();
for x in numbers {
    squares.push(x * x);
}
```

### Ownership & Borrowing
* Pass by reference (`&T`) whenever possible to avoid unnecessary cloning.
* Implement `Clone` and `Copy` only when logically appropriate.
* Use `Cow<'a, str>` (Clone on Write) for strings if you might return a literal or an owned string.

## 5. Documentation Style
Agents must document code using standard `rustdoc` syntax.

* **Line Comments:** `//` for implementation details inside functions.
* **Doc Comments:** `///` for public APIs (structs, enums, functions).
* **Module Comments:** `//!` at the top of the file to describe the module.

**Structure of a Doc Comment:**
1.  Short summary sentence.
2.  (Optional) Detailed description.
3.  (Optional) `# Examples` section.
4.  (Optional) `# Panics` or `# Errors` section.

```rust
/// Calculates the factorial of a number.
///
/// # Arguments
/// * `n` - The number to calculate the factorial of.
///
/// # Returns
/// The factorial as a `u64`.
///
/// # Example
/// ```
/// let result = my_crate::factorial(5);
/// assert_eq!(result, 120);
/// ```
pub fn factorial(n: u64) -> u64 { ... }
```

## 6. Agent-Specific Instructions
When generating code, the agent must adhere to these meta-rules:

1.  **Completeness:** Do not use `// ... rest of code` placeholders. Provide full, runnable implementations unless explicitly asked for a snippet.
2.  **Explicit Dependencies:** Always list the required dependencies for `Cargo.toml` at the start or end of the response.
3.  **Type Inference:** Use type inference where clear, but add explicit types for complex chained iterators or ambiguous `collect()` calls.
    * *Good:* `let x = 5;`
    * *Good:* `let names: Vec<String> = users.map(...).collect();`
4.  **Clippy Linting:** Pre-emptively fix common Clippy warnings (e.g., use `if let` instead of `match` for single options, avoid `&String` -> use `&str`).

---

### Appendix: Minimal `rustfmt.toml`
If the agent needs to configure formatting, use these defaults:

```toml
edition = "2021"
max_width = 100
tab_spaces = 4
newline_style = "Unix"
imports_granularity = "Crate"
group_imports = "StdExternalCrate"
```