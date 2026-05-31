use std::io::Write;
use std::process::Command;
use std::sync::mpsc;
use std::time::Duration;

use libloading::Library;

const TIMEOUT: Duration = Duration::from_secs(30);

pub fn validate_and_compile_rust_reward(
    code: &str,
) -> Result<Library, String> {
    let parsed = syn::parse_str::<syn::File>(code)
        .map_err(|e| format!("Syntax error: {e}"))?;

    let has_reward_fn = parsed.items.iter().any(|item| {
        if let syn::Item::Fn(f) = item {
            f.sig.ident == "reward"
        } else {
            false
        }
    });
    if !has_reward_fn {
        return Err(
            "no top-level `fn reward` found".into()
        );
    }

    const BLOCKED: &[&str] =
        &["fs", "net", "process", "env", "os"];
    for item in &parsed.items {
        if let syn::Item::Use(u) = item {
            let segs = use_path_segments(&u.tree);
            match segs.as_slice() {
                [] => {}
                [first, ..] if first != "std" => {
                    return Err(format!(
                        "external crate imports \
                         not allowed: `{first}`"
                    ));
                }
                [_, second, ..] => {
                    if BLOCKED
                        .contains(&second.as_str())
                    {
                        return Err(format!(
                            "import of std::{second}\
                             is not allowed"
                        ));
                    }
                }
                _ => {}
            }
        }
    }

    let dir = std::env::temp_dir()
        .join(format!("reward_{}", unique_tag()));
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("tempdir: {e}"))?;
    let src_path = dir.join("reward.rs");
    let so_path = dir.join("reward.so");

    // catch_unwind prevents index-out-of-bounds
    // panics during the dummy run (empty slices)
    // from aborting the host process.
    let wrapper = "\n\
        #[no_mangle]\n\
        pub extern \"C\" fn reward_c_abi(\n\
            prev: *const f64,\n\
            prev_len: usize,\n\
            curr: *const f64,\n\
            curr_len: usize,\n\
            act: *const f64,\n\
            act_len: usize,\n\
            contacts: *const (usize, usize),\n\
            contacts_len: usize,\n\
        ) -> f64 {\n\
            let r = std::panic::catch_unwind(|| {\n\
                unsafe {\n\
                    reward(\n\
                        std::slice::from_raw_parts(\
                            prev, prev_len),\n\
                        std::slice::from_raw_parts(\
                            curr, curr_len),\n\
                        std::slice::from_raw_parts(\
                            act, act_len),\n\
                        std::slice::from_raw_parts(\
                            contacts, contacts_len),\n\
                    )\n\
                }\n\
            });\n\
            r.unwrap_or(f64::NAN)\n\
        }\n";

    {
        let mut f = std::fs::File::create(&src_path)
            .map_err(|e| format!("write src: {e}"))?;
        write!(f, "{}\n{}", code, wrapper)
            .map_err(|e| format!("write src: {e}"))?;
    }

    let out = Command::new("rustc")
        .args([
            "--crate-type",
            "cdylib",
            "-C",
            "panic=unwind",
            "-o",
            so_path.to_str().unwrap(),
            src_path.to_str().unwrap(),
        ])
        .output()
        .map_err(|e| format!("rustc exec: {e}"))?;

    if !out.status.success() {
        let stderr =
            String::from_utf8_lossy(&out.stderr);
        let _ = std::fs::remove_dir_all(&dir);
        return Err(format!(
            "compilation failed:\n{stderr}"
        ));
    }

    let lib = unsafe {
        Library::new(&so_path)
            .map_err(|e| format!("load .so: {e}"))?
    };

    let so_path_c = so_path.clone();
    let (tx, rx) =
        mpsc::channel::<Result<f64, String>>();
    std::thread::spawn(move || {
        let result = (|| -> Result<f64, String> {
            let lib2 = unsafe {
                Library::new(&so_path_c)
                    .map_err(|e| {
                        format!("load: {e}")
                    })?
            };
            type AbiF = unsafe extern "C" fn(
                *const f64,
                usize,
                *const f64,
                usize,
                *const f64,
                usize,
                *const (usize, usize),
                usize,
            ) -> f64;
            let f: libloading::Symbol<AbiF> = unsafe {
                lib2.get(b"reward_c_abi")
                    .map_err(|e| {
                        format!("symbol: {e}")
                    })?
            };
            let d: &[f64] = &[];
            let dc: &[(usize, usize)] = &[];
            Ok(unsafe {
                f(
                    d.as_ptr(),
                    0,
                    d.as_ptr(),
                    0,
                    d.as_ptr(),
                    0,
                    dc.as_ptr(),
                    0,
                )
            })
        })();
        let _ = tx.send(result);
    });

    let dummy_result = rx.recv_timeout(TIMEOUT);
    let _ = std::fs::remove_dir_all(&dir);

    match dummy_result {
        Ok(Ok(_)) => {}
        Ok(Err(e)) => {
            return Err(format!("dummy run: {e}"));
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            return Err(
                "dummy run timed out".into()
            );
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            return Err("dummy run panicked".into());
        }
    }

    Ok(lib)
}

pub fn validate_python_reward(
    code: &str,
) -> Result<(), String> {
    let syntax_out = Command::new("python3")
        .args([
            "-c",
            "import ast, sys; \
             ast.parse(sys.argv[1])",
            code,
        ])
        .output()
        .map_err(|e| {
            format!("python3 exec: {e}")
        })?;
    if !syntax_out.status.success() {
        let s = String::from_utf8_lossy(
            &syntax_out.stderr,
        );
        return Err(format!("syntax error:\n{s}"));
    }

    if !code.lines().any(|l| {
        l.starts_with("def reward(")
            || l.starts_with("def reward (")
    }) {
        return Err(
            "no top-level `def reward(` found".into()
        );
    }

    let allowed =
        "jax,jax.numpy,numpy,math,\
         functools,itertools,typing,dataclasses";
    let import_script = format!(
        "import ast, sys\n\
         allowed = set({allowed:?}.split(','))\n\
         tree = ast.parse(sys.argv[1])\n\
         for node in ast.walk(tree):\n\
         \x20 if isinstance(node, ast.Import):\n\
         \x20   for a in node.names:\n\
         \x20     b = a.name.split('.')[0]\n\
         \x20     if a.name not in allowed \
             and b not in allowed:\n\
         \x20       print(a.name); sys.exit(1)\n\
         \x20 elif isinstance(\
             node, ast.ImportFrom):\n\
         \x20   mod = node.module or ''\n\
         \x20   b = mod.split('.')[0]\n\
         \x20   if mod not in allowed \
             and b not in allowed:\n\
         \x20     print(mod); sys.exit(1)\n",
        allowed = allowed,
    );
    let import_out = Command::new("python3")
        .args(["-c", &import_script, code])
        .output()
        .map_err(|e| {
            format!("python3 exec: {e}")
        })?;
    if !import_out.status.success() {
        let blocked = String::from_utf8_lossy(
            &import_out.stdout,
        )
        .trim()
        .to_string();
        return Err(format!(
            "import of `{blocked}` is not allowed"
        ));
    }

    let dummy_script =
        "import sys, numpy as np\n\
         exec(sys.argv[1])\n\
         result = reward(\
             np.zeros(10), np.zeros(10), \
             np.zeros(8), np.zeros((3, 3)))\n\
         assert isinstance(float(result), float)\n"
            .to_string();
    let code_owned = code.to_string();
    let (tx, rx) =
        mpsc::channel::<Result<(), String>>();
    std::thread::spawn(move || {
        let out = Command::new("python3")
            .args(["-c", &dummy_script, &code_owned])
            .output();
        let result = match out {
            Ok(o) if o.status.success() => Ok(()),
            Ok(o) => {
                let s = String::from_utf8_lossy(
                    &o.stderr,
                )
                .to_string();
                Err(format!(
                    "dummy run failed:\n{s}"
                ))
            }
            Err(e) => {
                Err(format!("python3 exec: {e}"))
            }
        };
        let _ = tx.send(result);
    });

    match rx.recv_timeout(TIMEOUT) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(e),
        Err(mpsc::RecvTimeoutError::Timeout) => {
            Err("dummy run timed out".into())
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            Err("dummy run panicked".into())
        }
    }
}

fn use_path_segments(
    tree: &syn::UseTree,
) -> Vec<String> {
    match tree {
        syn::UseTree::Path(p) => {
            let mut segs =
                vec![p.ident.to_string()];
            segs.extend(use_path_segments(&p.tree));
            segs
        }
        _ => vec![],
    }
}

fn unique_tag() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!(
        "{:x}_{:x}",
        t.as_secs(),
        t.subsec_nanos(),
    )
}
