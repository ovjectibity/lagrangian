use std::collections::HashMap;
use std::io::Write;
use std::process::Command;
use std::sync::mpsc;
use std::time::Duration;

use libloading::Library;
use tch::Tensor;

const TIMEOUT: Duration = Duration::from_secs(30);

pub fn call_reward_c_abi(
    lib: &Library,
    parts_pre: &HashMap<String, Tensor>,
    parts_post: &HashMap<String, Tensor>,
) -> f64 {
    let build_entries = |parts: &HashMap<String, Tensor>| {
        parts
            .iter()
            .map(|(k, v)| {
                (
                    std::ffi::CString::new(k.as_str()).unwrap_or_default(),
                    Vec::<f64>::try_from(v.shallow_clone()).unwrap_or_default(),
                )
            })
            .collect::<Vec<_>>()
    };
    let pre = build_entries(parts_pre);
    let post = build_entries(parts_post);

    let keys_ptrs = |entries: &[(std::ffi::CString, Vec<f64>)]| -> Vec<*const u8> {
        entries.iter().map(|(k, _)| k.as_ptr() as *const u8).collect()
    };
    let val_ptrs = |entries: &[(std::ffi::CString, Vec<f64>)]| -> Vec<*const f64> {
        entries.iter().map(|(_, v)| v.as_ptr()).collect()
    };
    let val_lens = |entries: &[(std::ffi::CString, Vec<f64>)]| -> Vec<usize> {
        entries.iter().map(|(_, v)| v.len()).collect()
    };

    let pre_keys = keys_ptrs(&pre);
    let pre_vals = val_ptrs(&pre);
    let pre_lens = val_lens(&pre);
    let post_keys = keys_ptrs(&post);
    let post_vals = val_ptrs(&post);
    let post_lens = val_lens(&post);

    unsafe {
        type RawFn = unsafe extern "C" fn(
            *const *const u8,
            *const *const f64,
            *const usize,
            usize,
            *const *const u8,
            *const *const f64,
            *const usize,
            usize,
        ) -> f64;
        let f: libloading::Symbol<RawFn> =
            lib.get(b"reward_c_abi").expect("reward_c_abi");
        f(
            pre_keys.as_ptr(),
            pre_vals.as_ptr(),
            pre_lens.as_ptr(),
            pre.len(),
            post_keys.as_ptr(),
            post_vals.as_ptr(),
            post_lens.as_ptr(),
            post.len(),
        )
    }
}

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
    // panics during the dummy run (empty maps)
    // from aborting the host process.
    let wrapper = "\n\
        #[no_mangle]\n\
        pub extern \"C\" fn reward_c_abi(\n\
            pre_keys: *const *const u8,\n\
            pre_vals: *const *const f64,\n\
            pre_val_lens: *const usize,\n\
            pre_n: usize,\n\
            post_keys: *const *const u8,\n\
            post_vals: *const *const f64,\n\
            post_val_lens: *const usize,\n\
            post_n: usize,\n\
        ) -> f64 {\n\
            let r = std::panic::catch_unwind(|| unsafe {\n\
                let build = |keys: *const *const u8,\n\
                              vals: *const *const f64,\n\
                              lens: *const usize,\n\
                              n: usize| {\n\
                    let mut m = std::collections::HashMap::new();\n\
                    for i in 0..n {\n\
                        let k = std::ffi::CStr::from_ptr(\n\
                            *keys.add(i) as *const i8\n\
                        ).to_str().unwrap_or(\"\");\n\
                        let v = std::slice::from_raw_parts(\n\
                            *vals.add(i), *lens.add(i)\n\
                        );\n\
                        m.insert(k, v);\n\
                    }\n\
                    m\n\
                };\n\
                let parts_pre = build(\n\
                    pre_keys, pre_vals, pre_val_lens, pre_n\n\
                );\n\
                let parts_post = build(\n\
                    post_keys, post_vals, post_val_lens, post_n\n\
                );\n\
                reward(&parts_pre, &parts_post)\n\
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
                *const *const u8,
                *const *const f64,
                *const usize,
                usize,
                *const *const u8,
                *const *const f64,
                *const usize,
                usize,
            ) -> f64;
            let f: libloading::Symbol<AbiF> = unsafe {
                lib2.get(b"reward_c_abi")
                    .map_err(|e| {
                        format!("symbol: {e}")
                    })?
            };
            let nk: *const *const u8 =
                std::ptr::null();
            let nv: *const *const f64 =
                std::ptr::null();
            let nl: *const usize =
                std::ptr::null();
            Ok(unsafe {
                f(nk, nv, nl, 0, nk, nv, nl, 0)
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
        "import sys\n\
         exec(sys.argv[1])\n\
         result = reward({}, {})\n\
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
                let stderr =
                    String::from_utf8_lossy(
                        &o.stderr,
                    );
                // Python reward runs on the GPU
                // worker which has JAX installed;
                // a missing module on the server is
                // not a user code error.
                if stderr.contains(
                    "ModuleNotFoundError",
                ) || stderr.contains("ImportError")
                {
                    Ok(())
                } else {
                    Err(format!(
                        "dummy run failed:\n{stderr}"
                    ))
                }
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
