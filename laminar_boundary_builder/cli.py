#!/usr/bin/env python3
"""Command line entry point for the laminar boundary builder."""

from __future__ import annotations

import argparse
import glob
import sys
import tempfile
from pathlib import Path

from .core import (
    load_volume,
    prepare_laminar_project,
    run_laminar_boundary_pipeline,
    write_demo_project,
)


def _collect_swc_paths(args: argparse.Namespace) -> list[str]:
    paths: list[str] = []
    if args.swc:
        paths.extend(args.swc)
    if args.swc_list:
        with Path(args.swc_list).open("r", encoding="utf-8") as handle:
            paths.extend(line.strip() for line in handle if line.strip())
    if args.swc_glob:
        paths.extend(str(path) for path in sorted(glob.glob(args.swc_glob)))
    return paths


def cmd_prepare(args: argparse.Namespace) -> None:
    contours = prepare_laminar_project(
        mask_path=args.mask,
        output_dir=args.output_dir,
        slice_axis=args.slice_axis,
        min_area=args.min_area,
        largest_only=not args.keep_all_contours,
        manual_every=args.manual_every,
    )
    print(f"Prepared {len(contours)} slice contours in {args.output_dir}")
    print(f"Edit: {Path(args.output_dir) / 'manual_landmarks_template.csv'}")


def cmd_build(args: argparse.Namespace) -> None:
    outputs = run_laminar_boundary_pipeline(
        mask_path=args.mask,
        manual_csv=args.manual_csv,
        output_dir=args.output_dir,
        template_path=args.template,
        cell_csv=args.cell_csv,
        swc_paths=_collect_swc_paths(args),
        slice_axis=args.slice_axis,
        min_area=args.min_area,
        largest_only=not args.keep_all_contours,
        resample_points=args.resample_points,
        surface_method=args.surface_method,
        shell_backend=args.shell_backend,
        cut_curve_json=args.cut_curve_json,
        depth_method=args.depth_method,
        max_laplace_voxels=args.max_laplace_voxels,
        boundary_dilation=args.boundary_dilation,
        qc_every=args.qc_every,
        volume_format=args.volume_format,
    )
    print("Laminar boundary build finished.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


def cmd_demo(args: argparse.Namespace) -> None:
    demo_input_dir = Path(args.output_dir) / "demo_input"
    mask_path, manual_csv = write_demo_project(demo_input_dir)
    build_dir = Path(args.output_dir) / "demo_build"
    outputs = run_laminar_boundary_pipeline(
        mask_path=mask_path,
        manual_csv=manual_csv,
        output_dir=build_dir,
        slice_axis=0,
        min_area=20.0,
        resample_points=args.resample_points,
        surface_method=args.surface_method,
        shell_backend=args.shell_backend,
        depth_method=args.depth_method,
        max_laplace_voxels=args.max_laplace_voxels,
        qc_every=args.qc_every,
        volume_format=args.volume_format,
    )
    print("Demo build finished.")
    print(f"demo_mask: {mask_path}")
    print(f"demo_manual_csv: {manual_csv}")
    for key, value in outputs.items():
        print(f"{key}: {value}")


def cmd_selfcheck(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        tempfile.mkdtemp(prefix="laminar_boundary_builder_selfcheck_")
    )
    demo_input_dir = output_dir / "demo_input"
    mask_path, manual_csv = write_demo_project(demo_input_dir)
    build_dir = output_dir / "demo_build"
    outputs = run_laminar_boundary_pipeline(
        mask_path=mask_path,
        manual_csv=manual_csv,
        output_dir=build_dir,
        slice_axis=0,
        min_area=20.0,
        resample_points=24,
        depth_method="distance",
        qc_every=4,
        volume_format="nrrd",
    )

    required_outputs = [
        "boundaries",
        "boundary_summary",
        "outer_surface",
        "inner_surface",
        "lateral_surface",
        "laminar_depth",
        "boundary_labels",
    ]
    missing = [key for key in required_outputs if not Path(outputs[key]).exists()]
    if missing:
        raise RuntimeError("Selfcheck output missing: " + ", ".join(missing))

    shell_cut_dir = output_dir / "demo_shell_cut_marching"
    shell_cut_outputs = run_laminar_boundary_pipeline(
        mask_path=mask_path,
        manual_csv=manual_csv,
        output_dir=shell_cut_dir,
        slice_axis=0,
        min_area=20.0,
        resample_points=24,
        surface_method="shell-cut",
        shell_backend="marching-cubes",
        depth_method="distance",
        qc_every=4,
        volume_format="nrrd",
    )
    shell_missing = [
        key
        for key in ("outer_surface", "inner_surface", "lateral_surface", "qc")
        if not Path(shell_cut_outputs[key]).exists()
    ]
    if shell_missing:
        raise RuntimeError("Selfcheck shell-cut output missing: " + ", ".join(shell_missing))

    try:
        load_volume(output_dir / "missing_input.nrrd")
    except FileNotFoundError:
        missing_file_check = "ok"
    else:
        raise RuntimeError("Selfcheck expected missing input to raise FileNotFoundError.")

    print("Selfcheck finished.")
    print(f"output_dir: {output_dir}")
    print("demo_pipeline: ok")
    print("shell_cut_marching_cubes: ok")
    print(f"missing_file_error: {missing_file_check}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build outer/inner/lateral boundary surfaces and laminar depth fields "
            "from a 3D region mask plus sparse endpoint annotations."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Extract contours and write an editable manual landmark CSV template.",
    )
    prepare.add_argument("--mask", required=True, help="Target region mask volume (.nrrd/.npy/.npz)")
    prepare.add_argument("--output-dir", required=True, help="Output project folder")
    prepare.add_argument(
        "--slice-axis",
        default="coronal",
        help="Slice axis/orientation: coronal, sagittal, horizontal, or 0/1/2",
    )
    prepare.add_argument("--min-area", type=float, default=20.0, help="Small contour filter")
    prepare.add_argument("--manual-every", type=int, default=10, help="Template row spacing")
    prepare.add_argument(
        "--keep-all-contours",
        action="store_true",
        help="Keep multiple contours per slice instead of only the largest one.",
    )
    prepare.set_defaults(func=cmd_prepare)

    build = subparsers.add_parser(
        "build",
        help="Run the full MVP pipeline from mask and manual landmark CSV.",
    )
    build.add_argument("--mask", required=True, help="Target region mask volume (.nrrd/.npy/.npz)")
    build.add_argument("--manual-csv", required=True, help="Endpoint annotation CSV")
    build.add_argument("--output-dir", required=True, help="Output project folder")
    build.add_argument("--template", default=None, help="Optional template image volume")
    build.add_argument("--cell-csv", default=None, help="Optional soma coordinate CSV")
    build.add_argument("--swc", nargs="*", default=None, help="Optional SWC files for dendrite depth")
    build.add_argument("--swc-list", default=None, help="Optional text file with one SWC path per line")
    build.add_argument("--swc-glob", default=None, help="Optional glob pattern for SWC files")
    build.add_argument(
        "--slice-axis",
        default="coronal",
        help="Slice axis/orientation: coronal, sagittal, horizontal, or 0/1/2",
    )
    build.add_argument("--min-area", type=float, default=20.0, help="Small contour filter")
    build.add_argument("--resample-points", type=int, default=80, help="Points per outer/inner curve")
    build.add_argument(
        "--surface-method",
        choices=("shell-cut", "arc-graph", "contour-shell", "mask-constrained", "fast-loft"),
        default="contour-shell",
        help=(
            "Surface method. shell-cut cuts patches from the full mask shell; arc-graph is an "
            "experimental local-arc stitcher with topology QC; contour-shell and fast-loft are "
            "legacy contour stitchers; mask-constrained is the older voxel-shell labeler."
        ),
    )
    build.add_argument(
        "--shell-backend",
        choices=("voxel", "marching-cubes"),
        default="voxel",
        help="Shell backend for shell-cut. voxel is blocky debug baseline; marching-cubes is smoother.",
    )
    build.add_argument(
        "--cut-curve-json",
        default=None,
        help=(
            "Optional shell-cut JSON. Accepts GUI-generated shell_cut_annotations.json "
            "or explicit cut_curves; legacy patch_seeds are optional."
        ),
    )
    build.add_argument(
        "--depth-method",
        choices=("auto", "laplace", "distance"),
        default="auto",
        help="Laminar depth method. auto uses Laplace for small masks, distance for large masks.",
    )
    build.add_argument(
        "--max-laplace-voxels",
        type=int,
        default=250_000,
        help="Largest mask size for automatic Laplace solve.",
    )
    build.add_argument("--boundary-dilation", type=int, default=1, help="Boundary label thickness")
    build.add_argument("--qc-every", type=int, default=10, help="QC overlay interval")
    build.add_argument(
        "--volume-format",
        choices=("nrrd", "npy", "nii", "nii.gz"),
        default="nrrd",
        help="Volume output format. NIfTI output needs nibabel installed.",
    )
    build.add_argument(
        "--keep-all-contours",
        action="store_true",
        help="Keep multiple contours per slice instead of only the largest one.",
    )
    build.set_defaults(func=cmd_build)

    demo = subparsers.add_parser(
        "demo",
        help="Create a tiny synthetic example and run the full pipeline.",
    )
    demo.add_argument("--output-dir", required=True, help="Demo output folder")
    demo.add_argument("--resample-points", type=int, default=48, help="Points per curve")
    demo.add_argument(
        "--surface-method",
        choices=("shell-cut", "arc-graph", "contour-shell", "mask-constrained", "fast-loft"),
        default="contour-shell",
        help="Surface reconstruction method for the demo.",
    )
    demo.add_argument(
        "--shell-backend",
        choices=("voxel", "marching-cubes"),
        default="voxel",
        help="Shell backend for shell-cut. voxel is blocky debug baseline; marching-cubes is smoother.",
    )
    demo.add_argument(
        "--depth-method",
        choices=("auto", "laplace", "distance"),
        default="auto",
        help="Laminar depth method for the demo.",
    )
    demo.add_argument(
        "--max-laplace-voxels",
        type=int,
        default=250_000,
        help="Largest mask size for automatic Laplace solve.",
    )
    demo.add_argument("--qc-every", type=int, default=4, help="QC overlay interval")
    demo.add_argument(
        "--volume-format",
        choices=("nrrd", "npy", "nii", "nii.gz"),
        default="nrrd",
        help="Volume output format. NIfTI output needs nibabel installed.",
    )
    demo.set_defaults(func=cmd_demo)

    selfcheck = subparsers.add_parser(
        "selfcheck",
        help="Run a small smoke test for demo output and missing-file errors.",
    )
    selfcheck.add_argument("--output-dir", default=None, help="Optional selfcheck output folder")
    selfcheck.set_defaults(func=cmd_selfcheck)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
