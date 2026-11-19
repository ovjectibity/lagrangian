#!/usr/bin/env bash
# Generate worker Pydantic models from commondef/records and
# commondef/remoteworker, then export Postgres LUT data.
# API/client generation is handled later.
# Run from any directory: /path/to/scripts/typegen/gen_types.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

for command in typeshare python3; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "ERROR: $command not found in PATH" >&2
    exit 1
  }
done

python3 - "$REPO_ROOT" <<'PY'
import ast
import builtins
from collections import defaultdict
from pathlib import Path
import re
import subprocess
import sys
import tempfile

root = Path(sys.argv[1])
source_root = root / "app/src/commondef"
output_root = root / "worker/worker-py/commondef"


def stage_source(source):
    """Adapt unsupported enums only in temporary Typeshare input."""
    external = []
    tagged_units = []
    for name in ("WorkloadGraph", "WorkloadStopReasonServer"):
        pattern = (
            r"(?P<attrs>(?:#\[[^\]]*\]\s*)+)"
            r"(?P<decl>pub enum " + name + r"\s*\{)"
        )
        match = re.search(pattern, source)
        if not match:
            continue
        attrs = match["attrs"]
        if name == "WorkloadGraph" and not re.search(r"\btag\s*=", attrs):
            position = match.start("decl")
            source = (
                source[:position]
                + '#[serde(tag = "type", content = "content")]\n'
                + source[position:]
            )
            external.append(name)
        elif name == "WorkloadStopReasonServer" and re.search(r"\btag\s*=", attrs):
            body = source[match.end():].split("}", 1)[0]
            if "(" in body or "{" in body:
                raise RuntimeError(name + " is no longer a unit-only enum")
            attrs, count = re.subn(
                r'#\[serde\(\s*tag\s*=\s*"type"\s*,\s*'
                r'content\s*=\s*"content"\s*\)\]\s*',
                "",
                attrs,
            )
            if count != 1:
                raise RuntimeError("Unsupported tagging on " + name)
            source = source[:match.start("attrs")] + attrs + source[match.start("decl"):]
            tagged_units.append(name)
    return source, external, tagged_units


def adapt_models(text, external, tagged_units):
    # Chrono UTC timestamps require an offset, as does AwareDatetime.
    text, timestamps = re.subn(r"\bDateTime\[Utc\]", "AwareDatetime", text)
    module = ast.parse(text)
    if timestamps:
        module.body = [
            node for node in module.body
            if not (isinstance(node, ast.ImportFrom) and node.module == "datetime")
        ]
        for node in module.body:
            if isinstance(node, ast.ImportFrom) and node.module == "pydantic":
                node.names.append(ast.alias(name="AwareDatetime"))
    enums = {
        node.name: node
        for node in module.body
        if isinstance(node, ast.ClassDef)
    }
    external_values = {}
    for name in external:
        external_values[name] = {
            item.targets[0].id: item.value.value
            for item in enums[name + "Types"].body
            if isinstance(item, ast.Assign)
        }
    output = []
    for node in module.body:
        if isinstance(node, ast.ClassDef):
            if any(node.name == name + "Types" for name in external):
                continue
            for name, values in external_values.items():
                fields = {
                    item.target.id: item
                    for item in node.body
                    if isinstance(item, ast.AnnAssign)
                }
                tag = fields.get("type")
                payload = fields.get("content")
                if (
                    tag is not None and payload is not None
                    and isinstance(tag.annotation, ast.Subscript)
                    and isinstance(tag.annotation.slice, ast.Attribute)
                    and isinstance(tag.annotation.slice.value, ast.Name)
                    and tag.annotation.slice.value.id == name + "Types"
                ):
                    key = values[tag.annotation.slice.attr]
                    if not key.isidentifier():
                        raise RuntimeError("Unsupported external enum key: " + key)
                    # Avoid shadowing a payload type with a same-named field.
                    alias = "_" + node.name + "Payload"
                    output.append(ast.Assign(
                        targets=[ast.Name(id=alias, ctx=ast.Store())],
                        value=payload.annotation,
                    ))
                    node.body = [ast.AnnAssign(
                        target=ast.Name(id=key, ctx=ast.Store()),
                        annotation=ast.Name(id=alias, ctx=ast.Load()),
                        simple=1,
                    )]
            if node.name in tagged_units:
                name = node.name
                node.name = name + "Types"
                output.append(node)
                members = [
                    item.targets[0].id for item in node.body
                    if isinstance(item, ast.Assign)
                ]
                literals = ", ".join(node.name + "." + member for member in members)
                default = " = " + literals if len(members) == 1 else ""
                output.extend(ast.parse(
                    "class " + name + "(BaseModel):\n"
                    "    type: Literal[" + literals + "]" + default
                ).body)
                continue
            # Preserve wire keys such as Rust's lambda when dumping models.
            for item in node.body:
                if (
                    isinstance(item, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "model_config"
                            for target in item.targets)
                    and isinstance(item.value, ast.Call)
                ):
                    item.value.keywords.append(ast.keyword(
                        arg="serialize_by_alias", value=ast.Constant(value=True),
                    ))
        output.append(node)
    module.body = output
    return module


def definitions(module):
    names = set()
    for node in module.body:
        if isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(
                target.id for target in node.targets
                if isinstance(target, ast.Name)
            )
    return names


modules = {}
with tempfile.TemporaryDirectory(prefix="lagrangian-typegen-") as temporary:
    staging = Path(temporary)
    for group in ("records", "remoteworker"):
        for source in sorted((source_root / group).rglob("*.rs")):
            if source.name == "mod.rs":
                continue
            relative = source.relative_to(source_root).with_suffix(".py")
            directory = staging / relative.with_suffix("")
            directory.mkdir(parents=True)
            text, external, tagged_units = stage_source(source.read_text())
            (directory / source.name).write_text(text)
            output = directory / "generated.py"
            subprocess.run([
                "typeshare", str(directory), "--lang=python",
                "--output-file", str(output),
            ], check=True)
            modules[relative] = adapt_models(
                output.read_text(), external, tagged_units,
            )

# Resolve imports from the generated definitions instead of maintaining
# a second handwritten list of Rust cross-module dependencies.
owners = {}
for relative, module in modules.items():
    for name in definitions(module):
        if name in owners:
            raise RuntimeError("Duplicate generated type: " + name)
        owners[name] = relative

outputs = {}
for relative, module in modules.items():
    imported = {
        alias.asname or alias.name
        for node in module.body if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    used = {
        node.id for node in ast.walk(module)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    dependencies = defaultdict(list)
    for name in sorted(used - imported - definitions(module) - set(dir(builtins))):
        if name not in owners:
            raise RuntimeError("Unresolved type in " + str(relative) + ": " + name)
        dependencies[owners[name]].append(name)
    imports = [
        ast.ImportFrom(
            module="commondef." + ".".join(owner.with_suffix("").parts),
            names=[ast.alias(name=name) for name in names],
            level=0,
        )
        for owner, names in sorted(dependencies.items())
    ]
    # Generated files start with a docstring and the future import.
    module.body[2:2] = imports
    models = [
        node.name for node in module.body
        if isinstance(node, ast.ClassDef)
        and any(isinstance(base, ast.Name) and base.id == "BaseModel"
                for base in node.bases)
    ]
    # Resolve recursive/forward references after every local alias exists.
    for name in models:
        module.body.extend(ast.parse(name + ".model_rebuild()").body)
    module.body[0].value.value += (
        "\nSource: app/src/commondef/" + str(relative.with_suffix(".rs"))
        + "\nRegenerate with scripts/typegen/gen_types.sh; do not edit."
    )
    code = ast.unparse(ast.fix_missing_locations(module)) + "\n"
    compile(code, str(relative), "exec")
    outputs[relative] = code

# Publish only after every source has generated and passed syntax checks.
for relative, code in outputs.items():
    target = output_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.read_text() != code:
        target.write_text(code)
    print("Generated: " + str(target.relative_to(root)))
for group in ("records", "remoteworker"):
    initializer = output_root / group / "__init__.py"
    if not initializer.exists():
        initializer.write_text('"""Generated worker ' + group + ' models."""\n')
PY

bash "$REPO_ROOT/scripts/typegen/gen_static_luts_worker.sh"

echo "Done."
echo "Reminder: commondef/ioadapter.py is hand-maintained and must follow changes to workload_graph input/output specs."
