#!/usr/bin/env python3
"""Decimate STL meshes for faster MJX collision detection.

MJX treats all collision meshes as convex hulls internally, so using --convex-hull
gives the most honest and performant representation. Quadric decimation (--target-faces)
can further reduce complexity.

Usage:
    python scripts/decimate_stls.py <directory> [--convex-hull] [--target-faces N] [--dry-run]

Examples:
    # Preview what would happen (no files modified):
    python scripts/decimate_stls.py app/assets/trossen_vx300s/assets --dry-run

    # Replace meshes with their convex hulls (recommended for MJX collision):
    python scripts/decimate_stls.py app/assets/trossen_vx300s/assets --convex-hull

    # Decimate all STLs in-place, targeting 200 faces per mesh:
    python scripts/decimate_stls.py app/assets/trossen_vx300s/assets --target-faces 200

    # Both: convex hull first, then decimate to 200 faces:
    python scripts/decimate_stls.py app/assets/trossen_vx300s/assets --convex-hull --target-faces 200
"""

import argparse
import shutil
import sys
from pathlib import Path

import trimesh


def decimate_stl(
    path: Path,
    convex_hull: bool,
    target_faces: int | None,
    dry_run: bool,
) -> None:
    mesh = trimesh.load(path)
    orig_verts = len(mesh.vertices)
    orig_faces = len(mesh.faces)

    # Step 1: convex hull (MJX treats collision meshes as convex anyway)
    if convex_hull:
        mesh = mesh.convex_hull

    # Step 2: quadric decimation if a target was specified and mesh exceeds it
    if target_faces is not None and len(mesh.faces) > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_faces)

    final_verts = len(mesh.vertices)
    final_faces = len(mesh.faces)

    vert_reduction = (1 - final_verts / orig_verts) * 100 if orig_verts > 0 else 0
    face_reduction = (1 - final_faces / orig_faces) * 100 if orig_faces > 0 else 0

    tag = "[DRY RUN] " if dry_run else ""
    print(
        f"{tag}{path.name}: "
        f"verts {orig_verts} -> {final_verts} ({vert_reduction:.1f}% reduction), "
        f"faces {orig_faces} -> {final_faces} ({face_reduction:.1f}% reduction)"
    )

    if not dry_run and (final_verts != orig_verts or final_faces != orig_faces):
        backup = path.with_suffix(".stl.bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        mesh.export(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decimate STL meshes to reduce MJX collision cost.")
    parser.add_argument("directory", type=Path, help="Directory containing STL files")
    parser.add_argument(
        "--convex-hull",
        action="store_true",
        help="Replace meshes with their convex hulls (MJX treats collision meshes as convex anyway)",
    )
    parser.add_argument(
        "--target-faces",
        type=int,
        default=None,
        help="Target face count for quadric decimation",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print stats without modifying files")
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Error: {args.directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    stl_files = sorted(args.directory.glob("*.stl"))
    if not stl_files:
        print(f"No STL files found in {args.directory}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(stl_files)} STL files in {args.directory}")
    if args.convex_hull:
        print("Mode: convex hull")
    if args.target_faces:
        print(f"Target faces: {args.target_faces}")
    if not args.convex_hull and args.target_faces is None:
        print("Warning: no --convex-hull or --target-faces specified, nothing to do", file=sys.stderr)
        sys.exit(1)
    print()

    for stl_path in stl_files:
        decimate_stl(stl_path, args.convex_hull, args.target_faces, args.dry_run)


if __name__ == "__main__":
    main()
