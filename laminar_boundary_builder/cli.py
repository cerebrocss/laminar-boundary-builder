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
    run_3d_surface_depth_pipeline,
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


def _retired_2d_cli_command() -> None:
    raise RuntimeError(
        "The old 2D slice reconstruction CLI has been removed. "
        "Use the GUI 3D annotation workflow to build surfaces, then run the depth command with its project_config.json."
    )


def cmd_prepare(args: argparse.Namespace) -> None:
    _retired_2d_cli_command()


def cmd_build(args: argparse.Namespace) -> None:
    _retired_2d_cli_command()


def cmd_demo(args: argparse.Namespace) -> None:
    _retired_2d_cli_command()


def cmd_selfcheck(args: argparse.Namespace) -> None:
    try:
        load_volume(Path(tempfile.gettempdir()) / "laminar_boundary_builder_missing_input.nrrd")
    except FileNotFoundError:
        missing_file_check = "ok"
    else:
        raise RuntimeError("Selfcheck expected missing input to raise FileNotFoundError.")

    print("Selfcheck finished.")
    print("imports: ok")
    print(f"missing_file_error: {missing_file_check}")


def cmd_depth(args: argparse.Namespace) -> None:
    outputs = run_3d_surface_depth_pipeline(
        mask_path=args.mask,
        project_config=args.project_config,
        output_dir=args.output_dir,
        template_path=args.template,
        cell_csv=args.cell_csv,
        swc_paths=_collect_swc_paths(args),
        depth_method=args.depth_method,
        max_laplace_voxels=args.max_laplace_voxels,
        boundary_dilation=args.boundary_dilation,
        volume_format=args.volume_format,
    )
    print("3D surface depth finished.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute laminar depth from 3D outer/inner surface builds."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    depth = subparsers.add_parser(
        "depth",
        help="Compute depth from a 3D build_3d/project_config.json with outer and inner OBJ surfaces.",
    )
    depth.add_argument("--mask", required=True, help="Target region mask volume (.nrrd/.npy/.npz)")
    depth.add_argument("--project-config", required=True, help="3D build project_config.json")
    depth.add_argument("--output-dir", required=True, help="Output folder for depth volumes and tables")
    depth.add_argument("--template", default=None, help="Optional template image volume")
    depth.add_argument("--cell-csv", default=None, help="Optional soma coordinate CSV")
    depth.add_argument("--swc", nargs="*", default=None, help="Optional SWC files for dendrite depth")
    depth.add_argument("--swc-list", default=None, help="Optional text file with one SWC path per line")
    depth.add_argument("--swc-glob", default=None, help="Optional glob pattern for SWC files")
    depth.add_argument(
        "--depth-method",
        choices=("auto", "laplace", "distance"),
        default="auto",
        help="Laminar depth method. auto uses Laplace for small masks, distance for large masks.",
    )
    depth.add_argument(
        "--max-laplace-voxels",
        type=int,
        default=250_000,
        help="Largest mask size for automatic Laplace solve.",
    )
    depth.add_argument("--boundary-dilation", type=int, default=1, help="Boundary label thickness")
    depth.add_argument(
        "--volume-format",
        choices=("nrrd", "npy", "nii", "nii.gz"),
        default="nrrd",
        help="Volume output format. NIfTI output needs nibabel installed.",
    )
    depth.set_defaults(func=cmd_depth)

    selfcheck = subparsers.add_parser(
        "selfcheck",
        help="Run a small import and missing-file smoke test.",
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
