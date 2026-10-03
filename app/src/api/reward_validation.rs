use std::process::Command;
use std::sync::mpsc;
use std::time::Duration;

const TIMEOUT: Duration = Duration::from_secs(30);

pub fn validate_python_reward(code: &str) -> Result<(), String> {
    let syntax_out = Command::new("python3")
        .args([
            "-c",
            "import ast, sys; \
             ast.parse(sys.argv[1])",
            code,
        ])
        .output()
        .map_err(|e| format!("python3 exec: {e}"))?;
    if !syntax_out.status.success() {
        let s = String::from_utf8_lossy(&syntax_out.stderr);
        return Err(format!("syntax error:\n{s}"));
    }

    if !code
        .lines()
        .any(|l| l.starts_with("def reward(") || l.starts_with("def reward ("))
    {
        return Err("no top-level `def reward(` found".into());
    }

    let allowed = "jax,jax.numpy,numpy,math,\
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
        .map_err(|e| format!("python3 exec: {e}"))?;
    if !import_out.status.success() {
        let blocked = String::from_utf8_lossy(&import_out.stdout)
            .trim()
            .to_string();
        return Err(format!("import of `{blocked}` is not allowed"));
    }

    let dummy_script = "import sys\n\
         exec(sys.argv[1])\n\
         result = reward({}, {})\n\
         assert isinstance(float(result), float)\n"
        .to_string();
    let code_owned = code.to_string();
    let (tx, rx) = mpsc::channel::<Result<(), String>>();
    std::thread::spawn(move || {
        let out = Command::new("python3")
            .args(["-c", &dummy_script, &code_owned])
            .output();
        let result = match out {
            Ok(o) if o.status.success() => Ok(()),
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                // Python reward runs on the GPU
                // worker which has JAX installed;
                // a missing module on the server is
                // not a user code error.
                if stderr.contains("ModuleNotFoundError") || stderr.contains("ImportError") {
                    Ok(())
                } else {
                    Err(format!("dummy run failed:\n{stderr}"))
                }
            }
            Err(e) => Err(format!("python3 exec: {e}")),
        };
        let _ = tx.send(result);
    });

    match rx.recv_timeout(TIMEOUT) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(e),
        Err(mpsc::RecvTimeoutError::Timeout) => Err("dummy run timed out".into()),
        Err(mpsc::RecvTimeoutError::Disconnected) => Err("dummy run panicked".into()),
    }
}
