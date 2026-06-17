"""
Laminar boundary construction tools.

The goal of this module is deliberately practical:

1. Extract ordered contours from a 3D region mask.
2. Turn sparse human landmarks into outer, inner, and lateral boundary arcs.
3. Propagate those arcs across slices.
4. Loft arcs into simple surface meshes.
5. Build a laminar depth field and sample cells or dendrites from it.

It is designed as a core library first. A napari or Qt editor can sit on top of
these functions later without rewriting the geometry and QC logic.
"""

from __future__ import annotations

import csv
import gc
import gzip
import heapq
import json
import math
import os
import pickle
import re
import sys
import warnings
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import nrrd
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


BOUNDARY_BACKGROUND = 0
BOUNDARY_REGION = 1
BOUNDARY_OUTER = 2
BOUNDARY_INNER = 3
BOUNDARY_LATERAL = 4
BOUNDARY_UNKNOWN = 5

SURFACE_MODE_NORMAL = "normal"
SURFACE_MODE_OUTER_ONLY = "outer_only"
SURFACE_MODE_INNER_ONLY = "inner_only"
SURFACE_MODES = {
    SURFACE_MODE_NORMAL,
    SURFACE_MODE_OUTER_ONLY,
    SURFACE_MODE_INNER_ONLY,
}

SURFACE_BUILD_MASK_CONSTRAINED = "mask_constrained"
SURFACE_BUILD_FAST_LOFT = "fast_loft"
SURFACE_BUILD_CONTOUR_SHELL = "contour_shell"
SURFACE_BUILD_ARC_GRAPH = "arc_graph"
SURFACE_BUILD_SHELL_CUT = "shell_cut"

SURFACE_METHOD_CATEGORY_SHELL_PATCH = "shell_patch"
SURFACE_METHOD_CATEGORY_EXPERIMENTAL_STITCHING = "experimental_stitching"
SURFACE_METHOD_CATEGORY_LEGACY_STITCHING = "legacy_stitching"
SURFACE_METHOD_CATEGORY_LEGACY_MASK_LABELING = "legacy_mask_labeling"
SURFACE_METHOD_CATEGORIES = {
    SURFACE_BUILD_SHELL_CUT: SURFACE_METHOD_CATEGORY_SHELL_PATCH,
    SURFACE_BUILD_ARC_GRAPH: SURFACE_METHOD_CATEGORY_EXPERIMENTAL_STITCHING,
    SURFACE_BUILD_CONTOUR_SHELL: SURFACE_METHOD_CATEGORY_LEGACY_STITCHING,
    SURFACE_BUILD_FAST_LOFT: SURFACE_METHOD_CATEGORY_LEGACY_STITCHING,
    SURFACE_BUILD_MASK_CONSTRAINED: SURFACE_METHOD_CATEGORY_LEGACY_MASK_LABELING,
}
SURFACE_BUILD_LEGACY_METHODS = {
    SURFACE_BUILD_MASK_CONSTRAINED,
    SURFACE_BUILD_FAST_LOFT,
    SURFACE_BUILD_CONTOUR_SHELL,
}

SHELL_BACKEND_VOXEL = "voxel"
SHELL_BACKEND_MARCHING_CUBES = "marching_cubes"
SHELL_BACKENDS = {
    SHELL_BACKEND_VOXEL,
    SHELL_BACKEND_MARCHING_CUBES,
}

SHELL_CUT_ANNOTATION_SCHEMA = "laminar_boundary_builder.shell_cut_annotations.v1"
SURFACE_3D_ANNOTATION_SCHEMA = "laminar_boundary_builder.surface_3d_annotations.v1"
SHELL_CUT_MANUAL_POINT_NAMES = (
    "outer_cut_A",
    "outer_cut_B",
    "inner_cut_A",
    "inner_cut_B",
)


def normalize_surface_mode(value: Optional[str]) -> str:
    mode = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "": SURFACE_MODE_NORMAL,
        "normal": SURFACE_MODE_NORMAL,
        "both": SURFACE_MODE_NORMAL,
        "manual": SURFACE_MODE_NORMAL,
        "outer": SURFACE_MODE_OUTER_ONLY,
        "outer_only": SURFACE_MODE_OUTER_ONLY,
        "outer_cap": SURFACE_MODE_OUTER_ONLY,
        "outer_cap_only": SURFACE_MODE_OUTER_ONLY,
        "no_inner": SURFACE_MODE_OUTER_ONLY,
        "inner": SURFACE_MODE_INNER_ONLY,
        "inner_only": SURFACE_MODE_INNER_ONLY,
        "inner_cap": SURFACE_MODE_INNER_ONLY,
        "inner_cap_only": SURFACE_MODE_INNER_ONLY,
        "no_outer": SURFACE_MODE_INNER_ONLY,
    }
    if mode not in aliases:
        raise ValueError(f"Unknown surface_mode: {value}")
    return aliases[mode]


def normalize_surface_build_method(value: Optional[str]) -> str:
    method = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "": SURFACE_BUILD_MASK_CONSTRAINED,
        "mask": SURFACE_BUILD_MASK_CONSTRAINED,
        "mask_constrained": SURFACE_BUILD_MASK_CONSTRAINED,
        "mask_constrained_voxel_shell": SURFACE_BUILD_MASK_CONSTRAINED,
        "voxel": SURFACE_BUILD_MASK_CONSTRAINED,
        "voxel_shell": SURFACE_BUILD_MASK_CONSTRAINED,
        "fast": SURFACE_BUILD_FAST_LOFT,
        "fast_loft": SURFACE_BUILD_FAST_LOFT,
        "loft": SURFACE_BUILD_FAST_LOFT,
        "legacy": SURFACE_BUILD_FAST_LOFT,
        "contour": SURFACE_BUILD_CONTOUR_SHELL,
        "contour_shell": SURFACE_BUILD_CONTOUR_SHELL,
        "arc": SURFACE_BUILD_ARC_GRAPH,
        "arc_graph": SURFACE_BUILD_ARC_GRAPH,
        "local_arc": SURFACE_BUILD_ARC_GRAPH,
        "local_arc_graph": SURFACE_BUILD_ARC_GRAPH,
        "shell": SURFACE_BUILD_SHELL_CUT,
        "shell_cut": SURFACE_BUILD_SHELL_CUT,
        "surface_patch": SURFACE_BUILD_SHELL_CUT,
        "patch": SURFACE_BUILD_SHELL_CUT,
    }
    if method not in aliases:
        raise ValueError(f"Unknown surface_method: {value}")
    return aliases[method]


def surface_build_method_category(value: Optional[str]) -> str:
    method = normalize_surface_build_method(value)
    return SURFACE_METHOD_CATEGORIES.get(method, "unknown")


def surface_method_registry() -> Dict[str, Dict[str, str]]:
    return {
        SURFACE_BUILD_SHELL_CUT: {
            "role": "main",
            "category": SURFACE_METHOD_CATEGORY_SHELL_PATCH,
            "description": "main shell-cut patch extraction workflow",
        },
        SURFACE_BUILD_ARC_GRAPH: {
            "role": "experimental",
            "category": SURFACE_METHOD_CATEGORY_EXPERIMENTAL_STITCHING,
            "description": "experimental local arc graph stitcher and QC helper",
        },
        SURFACE_BUILD_CONTOUR_SHELL: {
            "role": "legacy",
            "category": SURFACE_METHOD_CATEGORY_LEGACY_STITCHING,
            "description": "legacy whole-contour stitching baseline",
        },
        SURFACE_BUILD_FAST_LOFT: {
            "role": "legacy",
            "category": SURFACE_METHOD_CATEGORY_LEGACY_STITCHING,
            "description": "legacy quick preview loft",
        },
        SURFACE_BUILD_MASK_CONSTRAINED: {
            "role": "legacy",
            "category": SURFACE_METHOD_CATEGORY_LEGACY_MASK_LABELING,
            "description": "legacy voxel label baseline",
        },
    }


def normalize_shell_backend(value: Optional[str]) -> str:
    backend = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "": SHELL_BACKEND_VOXEL,
        "voxel": SHELL_BACKEND_VOXEL,
        "voxel_shell": SHELL_BACKEND_VOXEL,
        "debug_voxel": SHELL_BACKEND_VOXEL,
        "marching": SHELL_BACKEND_MARCHING_CUBES,
        "marching_cube": SHELL_BACKEND_MARCHING_CUBES,
        "marching_cubes": SHELL_BACKEND_MARCHING_CUBES,
        "mc": SHELL_BACKEND_MARCHING_CUBES,
        "smooth": SHELL_BACKEND_MARCHING_CUBES,
        "smooth_shell": SHELL_BACKEND_MARCHING_CUBES,
    }
    if backend not in aliases:
        raise ValueError(f"Unknown shell_backend: {value}")
    return aliases[backend]


def _row_surface_mode(row: Dict[str, str]) -> str:
    return normalize_surface_mode(row.get("surface_mode"))


@dataclass
class VolumeData:
    data: np.ndarray
    header: Dict = field(default_factory=dict)
    affine: Optional[np.ndarray] = None
    source_path: Optional[str] = None


@dataclass
class RegionMaskExtraction:
    mask: np.ndarray
    mask_path: str
    region_label: str
    region_ids: List[int]
    voxel_count: int
    template: Optional[np.ndarray] = None
    template_path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class Contour2D:
    slice_index: int
    contour_id: int
    points: np.ndarray
    area: float
    length: float

    def to_json(self) -> Dict:
        return {
            "slice_index": int(self.slice_index),
            "contour_id": int(self.contour_id),
            "area": float(self.area),
            "length": float(self.length),
            "points": np.asarray(self.points, dtype=float).round(4).tolist(),
        }

    @classmethod
    def from_json(cls, item: Dict) -> "Contour2D":
        return cls(
            slice_index=int(item["slice_index"]),
            contour_id=int(item.get("contour_id", 0)),
            points=np.asarray(item["points"], dtype=float),
            area=float(item.get("area", 0.0)),
            length=float(item.get("length", 0.0)),
        )


@dataclass
class BoundarySlice:
    slice_index: int
    contour_id: int
    outer_arc: np.ndarray
    inner_arc: np.ndarray
    lateral_arcs: List[np.ndarray]
    source: str
    confidence: float
    flags: List[str] = field(default_factory=list)
    outer_path: str = "auto"
    inner_path: str = "auto"
    surface_mode: str = SURFACE_MODE_NORMAL
    mean_snap_distance: float = 0.0
    min_outer_inner_distance: float = math.nan

    def to_summary_row(self) -> Dict:
        return {
            "slice_index": self.slice_index,
            "contour_id": self.contour_id,
            "source": self.source,
            "surface_mode": normalize_surface_mode(self.surface_mode),
            "confidence": round(float(self.confidence), 4),
            "mean_snap_distance": round(float(self.mean_snap_distance), 4),
            "min_outer_inner_distance": round(float(self.min_outer_inner_distance), 4)
            if np.isfinite(self.min_outer_inner_distance)
            else "",
            "outer_points": int(len(self.outer_arc)),
            "inner_points": int(len(self.inner_arc)),
            "lateral_segments": int(len(self.lateral_arcs)),
            "flags": ";".join(self.flags),
        }

    def to_json(self) -> Dict:
        return {
            "slice_index": int(self.slice_index),
            "contour_id": int(self.contour_id),
            "source": self.source,
            "confidence": float(self.confidence),
            "flags": list(self.flags),
            "outer_path": self.outer_path,
            "inner_path": self.inner_path,
            "surface_mode": normalize_surface_mode(self.surface_mode),
            "mean_snap_distance": float(self.mean_snap_distance),
            "min_outer_inner_distance": float(self.min_outer_inner_distance)
            if np.isfinite(self.min_outer_inner_distance)
            else None,
            "outer_arc": np.asarray(self.outer_arc, dtype=float).round(4).tolist(),
            "inner_arc": np.asarray(self.inner_arc, dtype=float).round(4).tolist(),
            "lateral_arcs": [
                np.asarray(arc, dtype=float).round(4).tolist() for arc in self.lateral_arcs
            ],
        }

    @classmethod
    def from_json(cls, item: Dict) -> "BoundarySlice":
        min_distance = item.get("min_outer_inner_distance")
        return cls(
            slice_index=int(item["slice_index"]),
            contour_id=int(item.get("contour_id", 0)),
            outer_arc=np.asarray(item["outer_arc"], dtype=float),
            inner_arc=np.asarray(item["inner_arc"], dtype=float),
            lateral_arcs=[
                np.asarray(arc, dtype=float) for arc in item.get("lateral_arcs", [])
            ],
            source=str(item.get("source", "json")),
            confidence=float(item.get("confidence", 1.0)),
            flags=list(item.get("flags", [])),
            outer_path=str(item.get("outer_path", "auto")),
            inner_path=str(item.get("inner_path", "auto")),
            surface_mode=normalize_surface_mode(item.get("surface_mode")),
            mean_snap_distance=float(item.get("mean_snap_distance", 0.0)),
            min_outer_inner_distance=float(min_distance)
            if min_distance is not None
            else math.nan,
        )


@dataclass
class SurfaceMesh:
    name: str
    vertices: np.ndarray
    faces: np.ndarray


@dataclass
class ShellMesh:
    vertices: np.ndarray
    faces: np.ndarray
    spacing: np.ndarray
    backend: str
    coordinate_space: str = "index"
    face_sources: Optional[List[Dict]] = None


@dataclass
class SurfaceCutCurve:
    curve_id: str
    label_left: str
    label_right: str
    control_points: np.ndarray
    source: str = "annotation_derived"
    confidence: float = 1.0
    snapped_vertices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    snapped_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=float))
    mesh_vertex_path: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    cut_edges: List[Tuple[int, int]] = field(default_factory=list)
    is_closed: bool = False
    closure_error: float = math.nan
    mean_snap_distance: float = math.nan
    max_snap_distance: float = math.nan
    path_length: float = math.nan
    control_polyline_length: float = math.nan
    path_length_ratio: float = math.nan
    status: str = "pending"
    reason: str = ""


@dataclass
class SurfacePatchSeed:
    patch_label: str
    seed_point: np.ndarray
    source: str = "auto_from_annotation"
    snapped_face: Optional[int] = None
    snap_distance: Optional[float] = None
    status: str = "pending"
    reason: str = ""


@dataclass
class ArcSegment:
    arc_id: str
    slice_index: int
    contour_id: int
    label: int
    points: np.ndarray
    scoring_points: np.ndarray
    is_closed: bool
    source: str
    confidence: float
    boundary_source: str
    length: float
    centroid: np.ndarray
    start_point: np.ndarray
    end_point: np.ndarray
    tangent_start: np.ndarray
    tangent_end: np.ndarray
    curvature: np.ndarray


@dataclass
class ArcRange:
    arc_id: str
    start_fraction: float
    end_fraction: float
    start_index: int
    end_index: int
    length: float


@dataclass
class ArcMatch:
    prev_arc_id: str
    next_arc_id: str
    label: int
    prev_sample_indices: np.ndarray
    next_sample_indices: np.ndarray
    prev_fraction_start: float
    prev_fraction_end: float
    next_fraction_start: float
    next_fraction_end: float
    direction: int
    event_type: str
    cost: float
    confidence: float
    coverage_prev: float
    coverage_next: float
    mean_distance: float
    max_distance: float
    length_ratio: float
    reason: str
    accepted: bool = False


@dataclass
class TopologyEvent:
    slice_left: int
    slice_right: int
    arc_id: str
    label: int
    event_type: str
    severity: str
    start_fraction: float
    end_fraction: float
    start_index: int
    end_index: int
    coverage_prev: float
    coverage_next: float
    cost: Optional[float]
    reason: str


def load_volume(path: str | Path) -> VolumeData:
    """Load a 3D volume from NRRD, NumPy, or NIfTI if nibabel is installed."""

    path = _resolve_existing_input_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    suffixes = "".join(path.suffixes).lower()

    if suffixes.endswith(".nrrd") or suffixes.endswith(".nhdr"):
        try:
            data, header = nrrd.read(str(path))
        except Exception:
            data, header = _read_nrrd_fallback(path)
        return VolumeData(data=data, header=header, source_path=str(path))

    if suffixes.endswith(".npy"):
        return VolumeData(data=np.load(path, mmap_mode="r"), source_path=str(path))

    if suffixes.endswith(".npz"):
        loaded = np.load(path)
        key = "data" if "data" in loaded else loaded.files[0]
        return VolumeData(data=loaded[key], source_path=str(path))

    if suffixes.endswith(".nii") or suffixes.endswith(".nii.gz"):
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError(
                "Reading NIfTI needs nibabel. Install nibabel or use NRRD/NPY input."
            ) from exc
        img = nib.load(str(path))
        return VolumeData(
            data=np.asarray(img.get_fdata()),
            affine=img.affine,
            header={"nifti_header": img.header},
            source_path=str(path),
        )

    raise ValueError(f"Unsupported volume format: {path}")


def _voxel_spacing_from_volume(volume: VolumeData) -> np.ndarray:
    """Best-effort voxel spacing used for geometry scoring, not output coordinates."""

    if volume.affine is not None:
        affine = np.asarray(volume.affine, dtype=float)
        if affine.shape[0] >= 3 and affine.shape[1] >= 3:
            spacing = np.linalg.norm(affine[:3, :3], axis=0)
            if np.all(np.isfinite(spacing)) and np.all(spacing > 0):
                return spacing.astype(float)

    header = volume.header or {}
    for key in ("space directions", "space_directions"):
        directions = header.get(key)
        if directions is None:
            continue
        try:
            array = np.asarray(directions, dtype=float)
        except (TypeError, ValueError):
            continue
        if array.ndim == 2 and array.shape[0] >= 3:
            spacing = np.linalg.norm(array[:3, :3], axis=1)
            if np.all(np.isfinite(spacing)) and np.all(spacing > 0):
                return spacing.astype(float)

    for key in ("spacings", "spacing"):
        spacings = header.get(key)
        if spacings is None:
            continue
        try:
            spacing = np.asarray(spacings, dtype=float).reshape(-1)[:3]
        except (TypeError, ValueError):
            continue
        if len(spacing) == 3 and np.all(np.isfinite(spacing)) and np.all(spacing > 0):
            return spacing.astype(float)

    return np.ones(3, dtype=float)


def _read_nrrd_fallback(path: str | Path) -> Tuple[np.ndarray, Dict]:
    """Small NRRD reader fallback for plain single-file raw/gzip volumes."""

    path = Path(path)
    content = path.read_bytes()
    marker = b"\n\n"
    header_end = content.find(marker)
    data_offset = header_end + len(marker)
    if header_end < 0:
        marker = b"\r\n\r\n"
        header_end = content.find(marker)
        data_offset = header_end + len(marker)
    if header_end < 0:
        raise ValueError(f"Could not find NRRD header end in {path}")

    header_text = content[:header_end].decode("ascii", errors="replace")
    header: Dict[str, str] = {}
    for line in header_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("NRRD"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        header[key.strip().lower()] = value.strip()

    if header.get("data file"):
        raise ValueError("Detached NRRD data files are not supported by fallback reader")

    sizes = tuple(int(value) for value in header["sizes"].split())
    dtype = _nrrd_dtype(header["type"], header.get("endian"))
    encoding = header.get("encoding", "raw").lower()
    payload = content[data_offset:]
    if encoding in ("gzip", "gz"):
        payload = gzip.decompress(payload)
    elif encoding not in ("raw", "txt", "text", "ascii"):
        raise ValueError(f"Unsupported NRRD encoding in fallback reader: {encoding}")

    if encoding in ("txt", "text", "ascii"):
        array = np.fromstring(payload.decode("ascii"), sep=" ", dtype=dtype)
    else:
        array = np.frombuffer(payload, dtype=dtype)

    expected = int(np.prod(sizes))
    if array.size < expected:
        raise ValueError(f"NRRD data is shorter than expected: {array.size} < {expected}")
    if array.size > expected:
        array = array[:expected]
    header["_fallback_nrrd_reader"] = "true"
    return array.reshape(sizes, order="F"), header


def _nrrd_dtype(type_name: str, endian: Optional[str]) -> np.dtype:
    name = type_name.strip().lower()
    dtype_map = {
        "uchar": "u1",
        "unsigned char": "u1",
        "uint8": "u1",
        "uint8_t": "u1",
        "signed char": "i1",
        "int8": "i1",
        "int8_t": "i1",
        "short": "i2",
        "short int": "i2",
        "int16": "i2",
        "int16_t": "i2",
        "ushort": "u2",
        "unsigned short": "u2",
        "unsigned short int": "u2",
        "uint16": "u2",
        "uint16_t": "u2",
        "int": "i4",
        "signed int": "i4",
        "int32": "i4",
        "int32_t": "i4",
        "uint": "u4",
        "unsigned int": "u4",
        "uint32": "u4",
        "uint32_t": "u4",
        "longlong": "i8",
        "long long": "i8",
        "int64": "i8",
        "int64_t": "i8",
        "ulonglong": "u8",
        "unsigned long long": "u8",
        "uint64": "u8",
        "uint64_t": "u8",
        "float": "f4",
        "double": "f8",
    }
    if name not in dtype_map:
        raise ValueError(f"Unsupported NRRD type: {type_name}")
    dtype = np.dtype(dtype_map[name])
    if dtype.itemsize > 1:
        if endian == "little":
            dtype = dtype.newbyteorder("<")
        elif endian == "big":
            dtype = dtype.newbyteorder(">")
    return dtype


def save_volume(path: str | Path, data: np.ndarray, reference: Optional[VolumeData] = None) -> None:
    """Save a volume, preserving NRRD header fields when possible."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(path.suffixes).lower()

    if suffixes.endswith(".nrrd") or suffixes.endswith(".nhdr"):
        header = dict(reference.header) if reference and reference.header else {}
        if header.get("_fallback_nrrd_reader"):
            header = {}
        nrrd.write(str(path), data, header=header)
        return

    if suffixes.endswith(".npy"):
        np.save(path, data)
        return

    if suffixes.endswith(".nii") or suffixes.endswith(".nii.gz"):
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError(
                "Writing NIfTI needs nibabel. Install nibabel or choose a .nrrd path."
            ) from exc
        affine = reference.affine if reference is not None and reference.affine is not None else np.eye(4)
        nib.save(nib.Nifti1Image(data, affine), str(path))
        return

    raise ValueError(f"Unsupported output volume format: {path}")


def _candidate_project_roots() -> List[Path]:
    starts: List[Path] = []
    for env_name in ("LAMINAR_BOUNDARY_PROJECT_ROOT", "CONNECTOME_PROJECT_ROOT"):
        env_value = os.environ.get(env_name)
        if env_value:
            starts.append(Path(env_value).expanduser())

    starts.extend([Path.cwd(), Path(__file__).resolve()])
    executable = getattr(sys, "executable", "")
    if executable:
        starts.append(Path(executable).resolve())
    bundle_root = getattr(sys, "_MEIPASS", "")
    if bundle_root:
        starts.append(Path(bundle_root).resolve())

    roots: List[Path] = []
    seen = set()
    for start in starts:
        candidates = [start] if start.is_dir() else [start.parent]
        candidates.extend(start.parents)
        for parent in candidates:
            for root in (
                parent,
                parent / "Resources",
                parent / "Contents" / "Resources",
            ):
                key = str(root)
                if key in seen:
                    continue
                seen.add(key)
                if (root / "data" / "local" / "misc").exists():
                    roots.append(root)
    return roots


def _repo_root() -> Path:
    roots = _candidate_project_roots()
    if roots:
        return roots[0]
    return Path(__file__).resolve().parents[3]


def _resolve_existing_input_path(path: str | Path) -> Path:
    raw = str(path).strip()
    candidate = Path(raw).expanduser()
    if candidate.exists():
        return candidate

    relative_options: List[Path] = []
    if raw.startswith("/data/"):
        relative_options.append(Path(raw[1:]))
    elif not candidate.is_absolute():
        relative_options.append(Path(raw))

    for root in _candidate_project_roots():
        for relative in relative_options:
            resolved = root / relative
            if resolved.exists():
                return resolved
    return candidate


def _default_ontology_candidates() -> List[Path]:
    candidates: List[Path] = []
    for root in _candidate_project_roots() or [_repo_root()]:
        candidates.extend(
            [
                root / "data" / "local" / "misc" / "1.json",
                root / "data" / "local" / "misc" / "info_css" / "1.json",
                root / "data" / "local" / "misc" / "annot.txt",
            ]
        )
    return candidates


def _resolve_ontology_path(ontology_path: str | Path | None = None) -> Path:
    candidates: List[Path] = []
    if ontology_path:
        candidates.append(Path(ontology_path).expanduser())
    for env_name in ("LAMINAR_BOUNDARY_ONTOLOGY_PATH", "NEURONVIS_ONTOLOGY_PATH"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(Path(env_value).expanduser())
    candidates.extend(_default_ontology_candidates())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Could not find ontology JSON/txt. Searched:\n{searched}")


def _default_annotation_candidates() -> List[Path]:
    candidates: List[Path] = []
    for root in _candidate_project_roots() or [_repo_root()]:
        candidates.extend(
            [
                root / "data" / "local" / "misc" / "annotation_10.nrrd",
                root / "data" / "local" / "misc" / "annotation_10_2017.nrrd",
                root / "data" / "local" / "misc" / "annotation_10_cache.pkl",
                root / "data" / "local" / "misc" / "annotation_10_2017_cache.pkl",
            ]
        )
    return candidates


def resolve_annotation_path(annotation_path: str | Path | None = None) -> Path:
    candidates: List[Path] = []
    if annotation_path:
        resolved = _resolve_existing_input_path(annotation_path)
        candidates.append(resolved)
        if resolved.exists():
            return resolved

    for env_name in ("LAMINAR_BOUNDARY_ATLAS_PATH", "NEURONVIS_ATLAS_PATH"):
        env_value = os.environ.get(env_name)
        if env_value:
            resolved = _resolve_existing_input_path(env_value)
            candidates.append(resolved)
            if resolved.exists():
                return resolved

    candidates.extend(_default_annotation_candidates())
    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Could not find Allen annotation atlas. Searched:\n{searched}")


def _load_ontology_roots(ontology_path: str | Path | None = None) -> List[Dict]:
    path = _resolve_ontology_path(ontology_path)
    if path.suffix.lower() == ".txt":
        return _load_ontology_from_annot_txt(path)

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "msg" in payload:
        roots = payload["msg"]
    elif isinstance(payload, list):
        roots = payload
    else:
        raise ValueError(f"Unsupported ontology format: {path}")
    if not roots:
        raise ValueError(f"Ontology is empty: {path}")
    return list(roots)


def _load_ontology_from_annot_txt(path: Path) -> List[Dict]:
    region_map: Dict[int, Dict] = {}
    order: List[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(":", 2)
            if len(parts) != 3:
                continue
            region_id_text, parent_id_text, acronym = parts
            region_id = int(region_id_text)
            parent_id = int(parent_id_text) if parent_id_text else None
            region_map[region_id] = {
                "id": region_id,
                "parent_structure_id": parent_id,
                "acronym": acronym.strip(),
                "name": acronym.strip(),
                "children": [],
            }
            order.append(region_id)

    for region_id in order:
        node = region_map[region_id]
        parent_id = node["parent_structure_id"]
        if parent_id in region_map:
            region_map[parent_id]["children"].append(node)

    roots = [
        region_map[region_id]
        for region_id in order
        if region_map[region_id]["parent_structure_id"] not in region_map
    ]
    if not roots:
        raise ValueError(f"annot.txt is empty or invalid: {path}")
    return roots


def _iter_region_nodes(nodes: Sequence[Dict]) -> Iterable[Dict]:
    for node in nodes:
        yield node
        yield from _iter_region_nodes(node.get("children", []))


def _find_region_node(nodes: Sequence[Dict], token: str) -> Optional[Dict]:
    token = str(token).strip()
    if not token:
        return None

    if token.isdigit():
        target_id = int(token)
        for node in _iter_region_nodes(nodes):
            if int(node.get("id", -1)) == target_id:
                return node

    lowered = token.casefold()
    for key in ("acronym", "name"):
        for node in _iter_region_nodes(nodes):
            value = str(node.get(key, "")).strip()
            if value and value.casefold() == lowered:
                return node
    return None


def _collect_region_ids(node: Dict, include_children: bool) -> List[int]:
    ids = [int(node["id"])]
    if include_children:
        for child in node.get("children", []):
            ids.extend(_collect_region_ids(child, include_children=True))
    return ids


def resolve_region_ids(
    region_text: str,
    include_children: bool = True,
    ontology_path: str | Path | None = None,
) -> List[int]:
    """Resolve one or more region acronyms/names/IDs to atlas label IDs."""

    text = str(region_text).strip()
    if not text:
        raise ValueError("Brain region is required.")

    roots = _load_ontology_roots(ontology_path)
    numeric_tokens = [token for token in re.split(r"[\s,;]+", text) if token]
    if numeric_tokens and all(token.isdigit() for token in numeric_tokens):
        region_ids: List[int] = []
        for token in numeric_tokens:
            node = _find_region_node(roots, token)
            if node is None:
                region_ids.append(int(token))
            else:
                region_ids.extend(_collect_region_ids(node, include_children=include_children))
        return sorted(set(region_ids))

    exact = _find_region_node(roots, text)
    if exact is not None:
        tokens = [text]
    else:
        tokens = [token.strip() for token in re.split(r"[,;\n]+", text) if token.strip()]

    region_ids: List[int] = []
    missing: List[str] = []
    for token in tokens:
        node = _find_region_node(roots, token)
        if node is None:
            missing.append(token)
            continue
        region_ids.extend(_collect_region_ids(node, include_children=include_children))

    if missing:
        raise ValueError(
            "Region not found: "
            + ", ".join(missing)
            + ". Try an acronym like ENT, ENTl, CA1, or a numeric region ID."
        )
    return sorted(set(region_ids))


def _array_from_pickle_payload(payload) -> np.ndarray:
    if isinstance(payload, np.ndarray):
        return payload
    if isinstance(payload, VolumeData):
        return payload.data
    if isinstance(payload, dict):
        for key in ("data", "annotation", "atlas", "array", "volume"):
            value = payload.get(key)
            if isinstance(value, np.ndarray):
                return value
        for value in payload.values():
            if isinstance(value, np.ndarray):
                return value
    if isinstance(payload, (list, tuple)):
        for value in payload:
            if isinstance(value, np.ndarray):
                return value
    raise ValueError("Pickle file does not contain a NumPy array annotation volume.")


def load_annotation_array(path: str | Path | None = None) -> VolumeData:
    """Load an atlas annotation volume from NRRD/NumPy or pickle cache."""

    path = resolve_annotation_path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".pkl"):
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        return VolumeData(data=np.asarray(_array_from_pickle_payload(payload)), source_path=str(path))
    return load_volume(path)


def _apply_hemisphere_filter(mask: np.ndarray, hemisphere: str) -> None:
    key = str(hemisphere or "all").strip().lower()
    if key in ("", "all", "both"):
        return

    z_mid = mask.shape[2] // 2
    if key.startswith("left"):
        mask[:, :, z_mid:] = False
        return
    if key.startswith("right"):
        mask[:, :, :z_mid] = False
        return
    raise ValueError("Hemisphere must be all, left, or right.")


def _extract_label_mask_chunked(
    annotation: np.ndarray,
    region_ids: Sequence[int],
    progress: Callable[[str], None],
) -> np.ndarray:
    mask = np.zeros(annotation.shape, dtype=bool)
    if not region_ids:
        return mask

    plane_size = max(1, int(annotation.shape[1]) * int(annotation.shape[2]))
    rows_per_chunk = max(1, min(annotation.shape[0], 64_000_000 // plane_size))
    total_chunks = math.ceil(annotation.shape[0] / rows_per_chunk)
    ids = np.asarray(region_ids, dtype=annotation.dtype)

    for chunk_index, start in enumerate(range(0, annotation.shape[0], rows_per_chunk), start=1):
        end = min(annotation.shape[0], start + rows_per_chunk)
        mask[start:end] = np.isin(annotation[start:end], ids, assume_unique=True)
        progress(f"Extracting mask chunk {chunk_index}/{total_chunks}...")
    return mask


def extract_region_mask_from_annotation(
    annotation_path: str | Path | None,
    region_text: str,
    output_path: str | Path,
    template_path: str | Path | None = None,
    ontology_path: str | Path | None = None,
    include_children: bool = True,
    hemisphere: str = "all",
    progress: Optional[Callable[[str], None]] = None,
) -> RegionMaskExtraction:
    """Extract a temporary binary mask for a selected brain region."""

    emit = progress or (lambda _message: None)
    emit("Resolving brain region...")
    region_ids = resolve_region_ids(
        region_text,
        include_children=include_children,
        ontology_path=ontology_path,
    )

    emit("Loading atlas annotation volume...")
    annotation_volume = load_annotation_array(annotation_path)
    annotation = np.asarray(annotation_volume.data)
    if annotation.ndim != 3:
        raise ValueError(f"Annotation volume must be 3D, got shape {annotation.shape}.")

    emit(f"Extracting region mask from {len(region_ids)} atlas IDs...")
    mask = _extract_label_mask_chunked(annotation, region_ids, emit)
    del annotation
    annotation_volume.data = np.empty((0,), dtype=np.uint8)
    gc.collect()

    _apply_hemisphere_filter(mask, hemisphere)
    voxel_count = int(np.count_nonzero(mask))
    if voxel_count == 0:
        raise ValueError("The selected region produced an empty mask.")

    warnings_list: List[str] = []
    template = None
    reference: Optional[VolumeData] = None
    if annotation_volume.header:
        header = dict(annotation_volume.header)
        header.setdefault("encoding", "gzip")
        reference = VolumeData(
            data=np.empty((0,), dtype=np.uint8),
            header=header,
            source_path=str(annotation_path),
        )

    if template_path:
        emit("Loading template image...")
        template_volume = load_volume(template_path)
        if template_volume.data.shape == mask.shape:
            template = template_volume.data
            if reference is None:
                header = dict(template_volume.header or {})
                header.setdefault("encoding", "gzip")
                reference = VolumeData(
                    data=np.empty((0,), dtype=np.uint8),
                    header=header,
                    source_path=str(template_path),
                )
        else:
            warnings_list.append(
                f"Template shape {template_volume.data.shape} does not match mask shape {mask.shape}; template ignored."
            )

    if reference is None:
        reference = VolumeData(data=np.empty((0,), dtype=np.uint8), header={"encoding": "gzip"})

    emit("Saving temporary mask...")
    output_path = Path(output_path).expanduser()
    if "".join(output_path.suffixes).lower().endswith(".npy"):
        save_volume(output_path, mask, reference=reference)
        del mask
        gc.collect()
        mask_for_app = np.load(output_path, mmap_mode="r")
    else:
        save_volume(output_path, mask.view(np.uint8), reference=reference)
        mask_for_app = mask
    emit("Temporary mask ready.")

    return RegionMaskExtraction(
        mask=mask_for_app,
        mask_path=str(output_path),
        region_label=str(region_text).strip(),
        region_ids=region_ids,
        voxel_count=voxel_count,
        template=template,
        template_path=str(template_path) if template_path else None,
        warnings=warnings_list,
    )


def _slice_axis_to_int(slice_axis: int | str) -> int:
    if isinstance(slice_axis, int):
        if slice_axis not in (0, 1, 2):
            raise ValueError("slice_axis must be 0, 1, or 2")
        return slice_axis

    mapping = {
        "coronal": 0,
        "sagittal": 1,
        "horizontal": 2,
        "axial": 2,
    }
    key = str(slice_axis).strip().lower()
    if key in ("0", "1", "2"):
        return int(key)
    if key not in mapping:
        raise ValueError(f"Unknown slice axis/orientation: {slice_axis}")
    return mapping[key]


def _other_axes(slice_axis: int) -> Tuple[int, int]:
    axes = [0, 1, 2]
    axes.remove(slice_axis)
    return axes[0], axes[1]


def _take_slice(volume: np.ndarray, slice_index: int, slice_axis: int) -> np.ndarray:
    slicer = [slice(None)] * volume.ndim
    slicer[int(slice_axis)] = int(slice_index)
    return volume[tuple(slicer)]


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.dtype == np.bool_:
        return mask
    return mask > 0


def _plane_to_volume_points(
    plane_points: np.ndarray, slice_index: int, slice_axis: int
) -> np.ndarray:
    """Convert 2D plane contour vertices into volume coordinates.

    Contour vertices are x/y in the 2D plane, where x is column and y is row.
    NumPy volume coordinates are axis ordered.
    """

    plane_points = np.asarray(plane_points, dtype=float)
    row_axis, col_axis = _other_axes(slice_axis)
    points = np.zeros((len(plane_points), 3), dtype=float)
    points[:, slice_axis] = float(slice_index)
    points[:, row_axis] = plane_points[:, 1]
    points[:, col_axis] = plane_points[:, 0]
    return points


def _volume_to_plane_points(points: np.ndarray, slice_axis: int) -> np.ndarray:
    row_axis, col_axis = _other_axes(slice_axis)
    points = np.asarray(points, dtype=float)
    return np.column_stack([points[:, col_axis], points[:, row_axis]])


def _polyline_length(points: np.ndarray, closed: bool = False) -> float:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    length = float(np.linalg.norm(diffs, axis=1).sum())
    if closed:
        length += float(np.linalg.norm(points[0] - points[-1]))
    return length


def _polygon_area(points_2d: np.ndarray) -> float:
    if len(points_2d) < 3:
        return 0.0
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _find_mask_contours(mask_2d: np.ndarray) -> List[np.ndarray]:
    """Find ordered 2D mask contours without importing plotting libraries.

    It traces mask-cell edges directly, which keeps GUI startup light and avoids
    loading a plotting stack just to show the first window.
    """

    mask = np.asarray(mask_2d) > 0
    if mask.ndim != 2 or not np.any(mask):
        return []

    height, width = mask.shape
    edges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

    def is_inside(row: int, col: int) -> bool:
        return 0 <= row < height and 0 <= col < width and bool(mask[row, col])

    rows, cols = np.nonzero(mask)
    for row, col in zip(rows, cols):
        x0, x1 = 2 * col - 1, 2 * col + 1
        y0, y1 = 2 * row - 1, 2 * row + 1
        if not is_inside(row - 1, col):
            edges.append(((x0, y0), (x1, y0)))
        if not is_inside(row, col + 1):
            edges.append(((x1, y0), (x1, y1)))
        if not is_inside(row + 1, col):
            edges.append(((x1, y1), (x0, y1)))
        if not is_inside(row, col - 1):
            edges.append(((x0, y1), (x0, y0)))

    outgoing: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for index, (start, _end) in enumerate(edges):
        outgoing[start].append(index)

    unused = set(range(len(edges)))
    contours: List[np.ndarray] = []
    while unused:
        first_index = unused.pop()
        start, current = edges[first_index]
        loop = [start]

        while current != start:
            loop.append(current)
            candidates = [edge_index for edge_index in outgoing.get(current, []) if edge_index in unused]
            if not candidates:
                break
            next_index = candidates[0]
            unused.remove(next_index)
            current = edges[next_index][1]

        if current == start and len(loop) >= 4:
            points = np.asarray([(x * 0.5, y * 0.5) for x, y in loop], dtype=float)
            contours.append(points)

    contours.sort(key=_polygon_area, reverse=True)
    return contours


def extract_slice_contours(
    mask: np.ndarray,
    slice_axis: int | str = 0,
    min_area: float = 20.0,
    largest_only: bool = True,
) -> List[Contour2D]:
    """Extract ordered mask contours for every slice."""

    slice_axis = _slice_axis_to_int(slice_axis)
    mask = _as_bool_mask(mask)
    contours: List[Contour2D] = []
    contour_counter = 0

    for slice_index in range(mask.shape[slice_axis]):
        mask_2d = _take_slice(mask, slice_index, slice_axis)
        raw_contours = _find_mask_contours(mask_2d)
        slice_contours: List[Contour2D] = []
        for raw in raw_contours:
            area = _polygon_area(raw)
            if area < min_area:
                continue
            points = _plane_to_volume_points(raw, slice_index, slice_axis)
            length = _polyline_length(points, closed=True)
            slice_contours.append(
                Contour2D(
                    slice_index=slice_index,
                    contour_id=contour_counter,
                    points=points,
                    area=area,
                    length=length,
                )
            )
            contour_counter += 1

        if largest_only and slice_contours:
            biggest = max(slice_contours, key=lambda item: item.area)
            biggest.contour_id = 0
            contours.append(biggest)
        else:
            contours.extend(slice_contours)

    return contours


def contours_by_slice(contours: Sequence[Contour2D]) -> Dict[int, List[Contour2D]]:
    by_slice: Dict[int, List[Contour2D]] = {}
    for contour in contours:
        by_slice.setdefault(contour.slice_index, []).append(contour)
    return by_slice


def save_contours_json(contours: Sequence[Contour2D], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([contour.to_json() for contour in contours], handle, indent=2)


def load_contours_json(path: str | Path) -> List[Contour2D]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return [Contour2D.from_json(item) for item in raw]


def resample_polyline(points: np.ndarray, n_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return points.reshape(0, 3)
    if len(points) == 1 or n_points <= 1:
        return np.repeat(points[:1], max(1, n_points), axis=0)

    distances = np.zeros(len(points), dtype=float)
    distances[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    if distances[-1] == 0:
        return np.repeat(points[:1], n_points, axis=0)

    targets = np.linspace(0, distances[-1], n_points)
    out = np.zeros((n_points, points.shape[1]), dtype=float)
    for dim in range(points.shape[1]):
        out[:, dim] = np.interp(targets, distances, points[:, dim])
    return out


def _nearest_index(points: np.ndarray, point: Sequence[float]) -> int:
    tree = cKDTree(np.asarray(points, dtype=float))
    _, index = tree.query(np.asarray(point, dtype=float), k=1)
    return int(index)


def _normalize_contour(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) > 1 and np.linalg.norm(points[0] - points[-1]) < 1e-6:
        points = points[:-1]
    return points


def _closed_contour_points(points: np.ndarray) -> np.ndarray:
    points = _normalize_contour(points)
    if len(points) == 0:
        return points.reshape(0, 3)
    return np.vstack([points, points[:1]])


def _is_closed_polyline(points: np.ndarray) -> bool:
    points = np.asarray(points, dtype=float)
    return len(points) > 2 and np.linalg.norm(points[0] - points[-1]) < 1e-6


def _align_closed_polyline_start(points: np.ndarray, target: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if not _is_closed_polyline(points):
        return points
    unique_points = points[:-1]
    start_index = _nearest_index(unique_points, target)
    shifted = np.vstack([unique_points[start_index:], unique_points[:start_index]])
    return np.vstack([shifted, shifted[:1]])


def _path_indices(n_points: int, start: int, end: int, direction: int) -> np.ndarray:
    if direction not in (1, -1):
        raise ValueError("direction must be 1 or -1")

    start = int(start) % n_points
    end = int(end) % n_points
    if direction == 1:
        if start <= end:
            return np.arange(start, end + 1)
        return np.r_[np.arange(start, n_points), np.arange(0, end + 1)]

    if start >= end:
        return np.arange(start, end - 1, -1)
    return np.r_[np.arange(start, -1, -1), np.arange(n_points - 1, end - 1, -1)]


def _path_from_indices(contour: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.asarray(contour, dtype=float)[np.asarray(indices, dtype=int)]


def _parse_path_choice(choice: str | int | None) -> Optional[int]:
    if choice is None:
        return None
    if isinstance(choice, int):
        return 1 if choice >= 0 else -1

    text = str(choice).strip().lower()
    if text in ("", "auto", "nan", "none"):
        return None
    if text in ("forward", "cw", "clockwise", "+", "+1", "1"):
        return 1
    if text in ("backward", "ccw", "counterclockwise", "-", "-1"):
        return -1
    raise ValueError(f"Unknown arc path choice: {choice}")


def _arc_distance_to_expected(arc: np.ndarray, expected: np.ndarray) -> float:
    if len(arc) == 0 or len(expected) == 0:
        return float("inf")
    n_points = max(2, len(expected))
    sampled = resample_polyline(arc, n_points)
    return float(np.linalg.norm(sampled - expected, axis=1).mean())


def _choose_arc(
    contour: np.ndarray,
    start_index: int,
    end_index: int,
    choice: str | int | None = None,
    expected: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    contour = _normalize_contour(contour)
    n_points = len(contour)
    if n_points < 2:
        raise ValueError("Contour needs at least two points")

    fixed_direction = _parse_path_choice(choice)
    candidates = [fixed_direction] if fixed_direction is not None else [1, -1]

    best = None
    for direction in candidates:
        indices = _path_indices(n_points, start_index, end_index, direction)
        arc = _path_from_indices(contour, indices)
        if expected is not None:
            score = _arc_distance_to_expected(arc, expected)
        else:
            score = _polyline_length(arc, closed=False)
        item = (score, arc, indices, direction)
        if best is None or item[0] < best[0]:
            best = item

    assert best is not None
    score, arc, indices, direction = best
    return arc, indices, direction, float(score)


def _shared_endpoint_count(
    outer_start: int,
    outer_end: int,
    inner_start: int,
    inner_end: int,
    n_points: int,
) -> int:
    outer_endpoints = {int(outer_start) % n_points, int(outer_end) % n_points}
    inner_endpoints = {int(inner_start) % n_points, int(inner_end) % n_points}
    return len(outer_endpoints & inner_endpoints)


def _arc_overlap_excess(
    n_points: int,
    outer_indices: np.ndarray,
    inner_indices: np.ndarray,
    allowed_overlap: int,
) -> int:
    outer_unique = np.unique(np.asarray(outer_indices, dtype=int) % n_points)
    inner_unique = np.unique(np.asarray(inner_indices, dtype=int) % n_points)
    overlap = np.intersect1d(outer_unique, inner_unique)
    return max(0, int(len(overlap)) - int(allowed_overlap))


def _arc_score(arc: np.ndarray, expected: Optional[np.ndarray]) -> float:
    if expected is not None:
        return _arc_distance_to_expected(arc, expected)
    return _polyline_length(arc, closed=False)


def _choose_outer_inner_arcs(
    contour: np.ndarray,
    outer_start: int,
    outer_end: int,
    inner_start: int,
    inner_end: int,
    outer_choice: str | int | None = None,
    inner_choice: str | int | None = None,
    expected_outer: Optional[np.ndarray] = None,
    expected_inner: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray, int, float]:
    contour = _normalize_contour(contour)
    n_points = len(contour)
    if n_points < 2:
        raise ValueError("Contour needs at least two points")

    fixed_outer = _parse_path_choice(outer_choice)
    fixed_inner = _parse_path_choice(inner_choice)
    outer_directions = [fixed_outer] if fixed_outer is not None else [1, -1]
    inner_directions = [fixed_inner] if fixed_inner is not None else [1, -1]
    allowed_overlap = _shared_endpoint_count(
        outer_start, outer_end, inner_start, inner_end, n_points
    )

    best = None
    for outer_direction in outer_directions:
        outer_indices = _path_indices(n_points, outer_start, outer_end, outer_direction)
        outer_arc = _path_from_indices(contour, outer_indices)
        outer_score = _arc_score(outer_arc, expected_outer)
        for inner_direction in inner_directions:
            inner_indices = _path_indices(n_points, inner_start, inner_end, inner_direction)
            inner_arc = _path_from_indices(contour, inner_indices)
            inner_score = _arc_score(inner_arc, expected_inner)
            overlap_excess = _arc_overlap_excess(
                n_points, outer_indices, inner_indices, allowed_overlap
            )
            # A shared endpoint is valid. Shared contour segments usually mean
            # the two surfaces are using the same boundary by mistake.
            score = outer_score + inner_score + overlap_excess * 10_000.0
            item = (
                score,
                outer_arc,
                outer_indices,
                outer_direction,
                float(outer_score),
                inner_arc,
                inner_indices,
                inner_direction,
                float(inner_score),
            )
            if best is None or item[0] < best[0]:
                best = item

    assert best is not None
    (
        _score,
        outer_arc,
        outer_indices,
        outer_direction,
        outer_score,
        inner_arc,
        inner_indices,
        inner_direction,
        inner_score,
    ) = best
    return (
        outer_arc,
        outer_indices,
        outer_direction,
        outer_score,
        inner_arc,
        inner_indices,
        inner_direction,
        inner_score,
    )


def _remaining_lateral_arcs(
    contour: np.ndarray, outer_indices: np.ndarray, inner_indices: np.ndarray
) -> Tuple[List[np.ndarray], List[str]]:
    contour = _normalize_contour(contour)
    n_points = len(contour)
    labels = np.zeros(n_points, dtype=np.uint8)
    flags: List[str] = []

    outer_unique = np.unique(outer_indices % n_points)
    inner_unique = np.unique(inner_indices % n_points)
    labels[outer_unique] = BOUNDARY_OUTER
    overlap = np.intersect1d(outer_unique, inner_unique)
    if len(overlap) > 2:
        flags.append("outer_inner_arc_overlap")
    labels[inner_unique] = np.where(labels[inner_unique] == 0, BOUNDARY_INNER, labels[inner_unique])

    lateral_mask = labels == 0
    if not np.any(lateral_mask):
        return [], flags + ["no_lateral_boundary"]

    doubled = np.r_[lateral_mask, lateral_mask]
    starts: List[int] = []
    ends: List[int] = []
    in_run = False
    for idx, value in enumerate(doubled):
        if idx >= n_points * 2:
            break
        if value and not in_run:
            starts.append(idx)
            in_run = True
        elif not value and in_run:
            ends.append(idx - 1)
            in_run = False
        if starts and starts[0] >= n_points:
            break
    if in_run:
        ends.append(len(doubled) - 1)

    arcs: List[np.ndarray] = []
    seen = set()
    for start, end in zip(starts, ends):
        run = np.arange(start, end + 1) % n_points
        key = tuple(sorted(np.unique(run).tolist()))
        if key in seen:
            continue
        seen.add(key)
        if len(run) >= 2:
            arcs.append(contour[run])

    return arcs, flags


def _allows_degenerate_lateral_contact(flags: Sequence[str]) -> bool:
    return "no_lateral_boundary" in flags and "outer_inner_arc_overlap" not in flags


def _landmark_point(row: Dict[str, str], prefix: str) -> Optional[np.ndarray]:
    keys = [
        (f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"),
        (f"{prefix}_0", f"{prefix}_1", f"{prefix}_2"),
    ]
    for trio in keys:
        if all(key in row and str(row[key]).strip() != "" for key in trio):
            return np.asarray([float(row[key]) for key in trio], dtype=float)
    return None


def _landmark_index(
    row: Dict[str, str], contour: np.ndarray, prefix: str, fallback_point_prefix: str
) -> int:
    index_key = f"{prefix}_index"
    short_key = prefix
    for key in (index_key, short_key):
        if key in row and str(row[key]).strip() != "":
            return int(float(row[key]))

    point = _landmark_point(row, fallback_point_prefix)
    if point is None:
        raise ValueError(
            f"Manual row for slice {row.get('slice_index')} is missing {prefix}_index "
            f"or {fallback_point_prefix}_x/y/z"
        )
    return _nearest_index(contour, point)


def read_manual_landmarks(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No manual landmark rows found in {path}")
    return rows


def _write_csv_rows(
    path: str | Path,
    rows: Sequence[Dict],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        ordered: List[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fieldnames = ordered
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def write_surface_method_registry(path: str | Path) -> None:
    rows = [
        {
            "surface_method": method,
            "role": meta["role"],
            "category": meta["category"],
            "description": meta["description"],
        }
        for method, meta in surface_method_registry().items()
    ]
    _write_csv_rows(
        path,
        rows,
        fieldnames=["surface_method", "role", "category", "description"],
    )


def make_boundary_from_landmark_row(
    contour: Contour2D,
    row: Dict[str, str],
    resample_points: int = 80,
) -> BoundarySlice:
    surface_mode = _row_surface_mode(row)
    if surface_mode != SURFACE_MODE_NORMAL:
        return make_whole_contour_boundary(
            contour,
            surface_mode=surface_mode,
            resample_points=resample_points,
            source="manual",
        )

    points = _normalize_contour(contour.points)
    outer_start = _landmark_index(row, points, "outer_start", "outer_start")
    outer_end = _landmark_index(row, points, "outer_end", "outer_end")
    inner_start = _landmark_index(row, points, "inner_start", "inner_start")
    inner_end = _landmark_index(row, points, "inner_end", "inner_end")

    outer_choice = row.get("outer_path", "auto")
    inner_choice = row.get("inner_path", "auto")
    (
        outer_arc,
        outer_indices,
        outer_direction,
        _outer_score,
        inner_arc,
        inner_indices,
        inner_direction,
        _inner_score,
    ) = _choose_outer_inner_arcs(
        points,
        outer_start,
        outer_end,
        inner_start,
        inner_end,
        outer_choice=outer_choice,
        inner_choice=inner_choice,
    )
    lateral_arcs, flags = _remaining_lateral_arcs(points, outer_indices, inner_indices)

    outer_resampled = resample_polyline(outer_arc, resample_points)
    inner_resampled = resample_polyline(inner_arc, resample_points)
    min_distance = _min_curve_distance(outer_resampled, inner_resampled, trim_fraction=0.05)
    if min_distance < 1.0 and not _allows_degenerate_lateral_contact(flags):
        flags.append("outer_inner_too_close")

    return BoundarySlice(
        slice_index=contour.slice_index,
        contour_id=contour.contour_id,
        outer_arc=outer_resampled,
        inner_arc=inner_resampled,
        lateral_arcs=[resample_polyline(arc, max(8, resample_points // 3)) for arc in lateral_arcs],
        source="manual",
        confidence=1.0,
        flags=flags,
        outer_path=str(outer_direction),
        inner_path=str(inner_direction),
        surface_mode=SURFACE_MODE_NORMAL,
        mean_snap_distance=0.0,
        min_outer_inner_distance=min_distance,
    )


def make_whole_contour_boundary(
    contour: Contour2D,
    surface_mode: str,
    resample_points: int = 80,
    source: str = "manual",
) -> BoundarySlice:
    surface_mode = normalize_surface_mode(surface_mode)
    if surface_mode == SURFACE_MODE_NORMAL:
        raise ValueError("Whole-contour boundary requires outer_only or inner_only mode.")

    ring = resample_polyline(_closed_contour_points(contour.points), resample_points)
    empty = np.empty((0, 3), dtype=float)
    is_outer = surface_mode == SURFACE_MODE_OUTER_ONLY
    return BoundarySlice(
        slice_index=contour.slice_index,
        contour_id=contour.contour_id,
        outer_arc=ring if is_outer else empty,
        inner_arc=empty if is_outer else ring,
        lateral_arcs=[],
        source=source,
        confidence=1.0,
        flags=[surface_mode],
        outer_path="whole" if is_outer else "",
        inner_path="" if is_outer else "whole",
        surface_mode=surface_mode,
        mean_snap_distance=0.0,
        min_outer_inner_distance=math.nan,
    )


def _select_contour_for_row(row: Dict[str, str], candidates: Sequence[Contour2D]) -> Contour2D:
    if not candidates:
        raise ValueError(f"No contour available for slice {row.get('slice_index')}")

    contour_id_text = str(row.get("contour_id", "")).strip()
    if contour_id_text:
        contour_id = int(float(contour_id_text))
        for contour in candidates:
            if contour.contour_id == contour_id:
                return contour
        raise ValueError(
            f"Contour id {contour_id} not found for slice {row.get('slice_index')}"
        )

    return max(candidates, key=lambda contour: contour.area)


def build_manual_boundaries(
    contours: Sequence[Contour2D],
    manual_rows: Sequence[Dict[str, str]],
    resample_points: int = 80,
) -> List[BoundarySlice]:
    by_slice = contours_by_slice(contours)
    boundaries: List[BoundarySlice] = []
    for row in manual_rows:
        slice_index = int(float(row["slice_index"]))
        contour = _select_contour_for_row(row, by_slice.get(slice_index, []))
        boundaries.append(make_boundary_from_landmark_row(contour, row, resample_points))
    return sorted(boundaries, key=lambda boundary: boundary.slice_index)


def validate_endpoint_annotations(
    contours: Sequence[Contour2D],
    manual_boundaries: Sequence[BoundarySlice],
) -> None:
    if not contours or not manual_boundaries:
        return

    first_slice = min(contour.slice_index for contour in contours)
    last_slice = max(contour.slice_index for contour in contours)
    manual_slices = {boundary.slice_index for boundary in manual_boundaries}
    missing = [
        slice_index
        for slice_index in (first_slice, last_slice)
        if slice_index not in manual_slices
    ]
    if missing:
        missing_text = ", ".join(str(slice_index) for slice_index in missing)
        raise ValueError(
            "The first and last contour slices must be manually annotated. "
            f"Missing endpoint slice(s): {missing_text}."
        )


def _min_curve_distance(
    curve_a: np.ndarray, curve_b: np.ndarray, trim_fraction: float = 0.0
) -> float:
    if len(curve_a) == 0 or len(curve_b) == 0:
        return float("inf")
    if trim_fraction > 0 and len(curve_a) > 6 and len(curve_b) > 6:
        trim_a = max(1, int(round(len(curve_a) * trim_fraction)))
        trim_b = max(1, int(round(len(curve_b) * trim_fraction)))
        curve_a = curve_a[trim_a:-trim_a]
        curve_b = curve_b[trim_b:-trim_b]
    tree = cKDTree(np.asarray(curve_b, dtype=float))
    distances, _ = tree.query(np.asarray(curve_a, dtype=float), k=1)
    return float(np.min(distances))


def _snap_expected_arc_to_contour(
    contour: Contour2D,
    expected_arc: np.ndarray,
    resample_points: int,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    points = _normalize_contour(contour.points)
    start_index = _nearest_index(points, expected_arc[0])
    end_index = _nearest_index(points, expected_arc[-1])
    arc, indices, direction, score = _choose_arc(
        points,
        start_index,
        end_index,
        choice=None,
        expected=resample_polyline(expected_arc, resample_points),
    )
    return resample_polyline(arc, resample_points), indices, direction, score


def _snap_expected_arc_pair_to_contour(
    contour: Contour2D,
    expected_outer: np.ndarray,
    expected_inner: np.ndarray,
    resample_points: int,
) -> Tuple[np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray, int, float]:
    points = _normalize_contour(contour.points)
    outer_expected = resample_polyline(expected_outer, resample_points)
    inner_expected = resample_polyline(expected_inner, resample_points)
    outer_start = _nearest_index(points, outer_expected[0])
    outer_end = _nearest_index(points, outer_expected[-1])
    inner_start = _nearest_index(points, inner_expected[0])
    inner_end = _nearest_index(points, inner_expected[-1])
    (
        outer_arc,
        outer_indices,
        outer_direction,
        outer_score,
        inner_arc,
        inner_indices,
        inner_direction,
        inner_score,
    ) = _choose_outer_inner_arcs(
        points,
        outer_start,
        outer_end,
        inner_start,
        inner_end,
        expected_outer=outer_expected,
        expected_inner=inner_expected,
    )
    return (
        resample_polyline(outer_arc, resample_points),
        outer_indices,
        outer_direction,
        outer_score,
        resample_polyline(inner_arc, resample_points),
        inner_indices,
        inner_direction,
        inner_score,
    )


def _boundary_qc(
    outer_arc: np.ndarray,
    inner_arc: np.ndarray,
    mean_snap_distance: float,
    snap_warning_distance: float,
    min_layer_distance: float,
    allow_endpoint_contact: bool = False,
) -> Tuple[float, float, List[str]]:
    flags: List[str] = []
    min_distance = _min_curve_distance(outer_arc, inner_arc, trim_fraction=0.05)
    if min_distance < min_layer_distance and not allow_endpoint_contact:
        flags.append("outer_inner_too_close")
    if mean_snap_distance > snap_warning_distance:
        flags.append("large_snap_distance")

    snap_score = 1.0 / (1.0 + mean_snap_distance / max(1e-6, snap_warning_distance))
    if allow_endpoint_contact:
        distance_score = 1.0
    else:
        distance_score = min(1.0, max(0.0, min_distance / max(1e-6, min_layer_distance)))
    confidence = max(0.0, min(1.0, 0.75 * snap_score + 0.25 * distance_score))
    return confidence, min_distance, flags


def propagate_boundaries(
    contours: Sequence[Contour2D],
    manual_boundaries: Sequence[BoundarySlice],
    resample_points: int = 80,
    snap_warning_distance: float = 6.0,
    min_layer_distance: float = 2.0,
) -> List[BoundarySlice]:
    """Interpolate manual arcs, then snap the prediction back to real contours."""

    if len(manual_boundaries) < 2:
        return sorted(list(manual_boundaries), key=lambda boundary: boundary.slice_index)

    by_slice = contours_by_slice(contours)
    manual_by_slice = {boundary.slice_index: boundary for boundary in manual_boundaries}
    output: Dict[int, BoundarySlice] = dict(manual_by_slice)
    sorted_manual = sorted(manual_boundaries, key=lambda boundary: boundary.slice_index)

    for left, right in zip(sorted_manual[:-1], sorted_manual[1:]):
        span = right.slice_index - left.slice_index
        if span <= 1:
            continue

        left_mode = normalize_surface_mode(left.surface_mode)
        right_mode = normalize_surface_mode(right.surface_mode)
        if left_mode != SURFACE_MODE_NORMAL or right_mode != SURFACE_MODE_NORMAL:
            single_surface_mode = None
            if left_mode == right_mode and left_mode != SURFACE_MODE_NORMAL:
                single_surface_mode = left_mode
            elif left_mode == SURFACE_MODE_NORMAL and right_mode != SURFACE_MODE_NORMAL:
                single_surface_mode = right_mode
            elif left_mode != SURFACE_MODE_NORMAL and right_mode == SURFACE_MODE_NORMAL:
                single_surface_mode = left_mode

            if single_surface_mode is not None:
                for slice_index in range(left.slice_index + 1, right.slice_index):
                    candidates = by_slice.get(slice_index, [])
                    if not candidates:
                        continue
                    contour = max(candidates, key=lambda item: item.area)
                    output[slice_index] = make_whole_contour_boundary(
                        contour,
                        surface_mode=single_surface_mode,
                        resample_points=resample_points,
                        source="auto",
                    )
            continue

        for slice_index in range(left.slice_index + 1, right.slice_index):
            candidates = by_slice.get(slice_index, [])
            if not candidates:
                continue
            contour = max(candidates, key=lambda item: item.area)
            t = (slice_index - left.slice_index) / float(span)
            expected_outer = (1.0 - t) * left.outer_arc + t * right.outer_arc
            expected_inner = (1.0 - t) * left.inner_arc + t * right.inner_arc

            (
                outer_arc,
                outer_indices,
                outer_direction,
                outer_snap,
                inner_arc,
                inner_indices,
                inner_direction,
                inner_snap,
            ) = _snap_expected_arc_pair_to_contour(
                contour, expected_outer, expected_inner, resample_points
            )
            lateral_arcs, lateral_flags = _remaining_lateral_arcs(
                contour.points, outer_indices, inner_indices
            )
            mean_snap = float((outer_snap + inner_snap) * 0.5)
            confidence, min_distance, qc_flags = _boundary_qc(
                outer_arc,
                inner_arc,
                mean_snap,
                snap_warning_distance=snap_warning_distance,
                min_layer_distance=min_layer_distance,
                allow_endpoint_contact=_allows_degenerate_lateral_contact(lateral_flags),
            )
            flags = lateral_flags + qc_flags
            if confidence < 0.5:
                flags.append("low_confidence")

            output[slice_index] = BoundarySlice(
                slice_index=slice_index,
                contour_id=contour.contour_id,
                outer_arc=outer_arc,
                inner_arc=inner_arc,
                lateral_arcs=[
                    resample_polyline(arc, max(8, resample_points // 3)) for arc in lateral_arcs
                ],
                source="auto",
                confidence=confidence,
                flags=flags,
                outer_path=str(outer_direction),
                inner_path=str(inner_direction),
                surface_mode=SURFACE_MODE_NORMAL,
                mean_snap_distance=mean_snap,
                min_outer_inner_distance=min_distance,
            )

    _extend_single_surface_tail(
        output,
        by_slice,
        sorted_manual,
        resample_points=resample_points,
    )
    return [output[key] for key in sorted(output)]


def _extend_single_surface_tail(
    output: Dict[int, BoundarySlice],
    by_slice: Dict[int, List[Contour2D]],
    sorted_manual: Sequence[BoundarySlice],
    resample_points: int,
) -> None:
    if not sorted_manual:
        return

    last = sorted_manual[-1]
    last_mode = normalize_surface_mode(last.surface_mode)
    if last_mode == SURFACE_MODE_NORMAL:
        return

    for slice_index in sorted(by_slice):
        if slice_index <= last.slice_index or slice_index in output:
            continue
        contour = max(by_slice[slice_index], key=lambda item: item.area)
        output[slice_index] = make_whole_contour_boundary(
            contour,
            surface_mode=last_mode,
            resample_points=resample_points,
            source="auto",
        )


def save_boundaries_json(boundaries: Sequence[BoundarySlice], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([boundary.to_json() for boundary in boundaries], handle, indent=2)


def read_boundaries_json(path: str | Path) -> List[BoundarySlice]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return [BoundarySlice.from_json(item) for item in raw]


def write_boundary_summary(boundaries: Sequence[BoundarySlice], path: str | Path) -> None:
    rows = [boundary.to_summary_row() for boundary in boundaries]
    _write_csv_rows(path, rows)


def loft_surface(
    boundaries: Sequence[BoundarySlice],
    arc_name: str,
    resample_points: int = 80,
    lateral_index: int = 0,
) -> SurfaceMesh:
    entries: List[Tuple[BoundarySlice, np.ndarray, bool]] = []
    for boundary in sorted(boundaries, key=lambda item: item.slice_index):
        if arc_name == "outer":
            curve = boundary.outer_arc
        elif arc_name == "inner":
            curve = boundary.inner_arc
        elif arc_name == "lateral":
            if len(boundary.lateral_arcs) <= lateral_index:
                continue
            curve = boundary.lateral_arcs[lateral_index]
        else:
            raise ValueError(f"Unknown arc name: {arc_name}")
        if len(curve) < 2:
            continue
        is_closed = _is_closed_polyline(curve)
        if is_closed:
            # Keep resample_points unique vertices for closed rings. The stored
            # boundary keeps first == last, but surface topology should wrap
            # around instead of carrying a duplicate seam column.
            sampled = resample_polyline(curve, resample_points + 1)[:-1]
        else:
            sampled = resample_polyline(curve, resample_points)
        entries.append((boundary, sampled, is_closed))

    if len(entries) < 2:
        raise ValueError(f"Need at least two curves to build {arc_name} surface")

    aligned_entries: List[Tuple[BoundarySlice, np.ndarray, bool]] = [entries[0]]
    for entry in entries[1:]:
        aligned_entries.append(_align_next_loft_entry(aligned_entries[-1], entry))

    base_vertices = np.vstack([entry[1] for entry in aligned_entries]).astype(float)
    extra_vertices: List[np.ndarray] = []

    def add_vertex(point: np.ndarray) -> int:
        extra_vertices.append(np.asarray(point, dtype=float))
        return len(base_vertices) + len(extra_vertices) - 1

    faces: List[List[int]] = []
    points_per_curve = len(aligned_entries[0][1])
    for entry_index, (left, right) in enumerate(zip(aligned_entries[:-1], aligned_entries[1:])):
        base = entry_index * points_per_curve
        next_base = (entry_index + 1) * points_per_curve
        if _loft_entries_compatible(left, right):
            faces.extend(_full_loft_faces(base, next_base, points_per_curve, left[2]))
        else:
            faces.extend(_transition_loft_faces(left, right, base, next_base, add_vertex))

    for entry_index, entry in enumerate(aligned_entries):
        _boundary, curve, is_closed = entry
        if not is_closed:
            continue
        has_previous = entry_index > 0
        has_next = entry_index + 1 < len(aligned_entries)
        base = entry_index * points_per_curve
        if not has_previous:
            faces.extend(_cap_closed_ring_faces(curve, base, add_vertex, reverse=True))
        if not has_next:
            faces.extend(_cap_closed_ring_faces(curve, base, add_vertex, reverse=False))

    if not faces:
        raise ValueError(f"Need at least two compatible curves to build {arc_name} surface")

    vertices = (
        np.vstack([base_vertices, np.vstack(extra_vertices)]).astype(float)
        if extra_vertices
        else base_vertices
    )
    return SurfaceMesh(name=arc_name, vertices=vertices, faces=np.asarray(faces, dtype=np.int32))


def _align_next_loft_entry(
    left: Tuple[BoundarySlice, np.ndarray, bool],
    right: Tuple[BoundarySlice, np.ndarray, bool],
) -> Tuple[BoundarySlice, np.ndarray, bool]:
    _left_boundary, left_curve, left_closed = left
    right_boundary, right_curve, right_closed = right
    if left_closed == right_closed:
        return right_boundary, _align_loft_curve(left_curve, right_curve, right_closed), right_closed
    if right_closed:
        return right_boundary, _align_closed_curve_to_open_curve(right_curve, left_curve), right_closed
    return right_boundary, _align_open_curve_to_closed_curve(right_curve, left_curve), right_closed


def _loft_entries_compatible(
    left: Tuple[BoundarySlice, np.ndarray, bool],
    right: Tuple[BoundarySlice, np.ndarray, bool],
) -> bool:
    left_boundary, left_curve, left_closed = left
    right_boundary, right_curve, right_closed = right

    if left_closed != right_closed:
        return False
    if right_boundary.slice_index - left_boundary.slice_index > 1:
        return False
    left_mode = normalize_surface_mode(left_boundary.surface_mode)
    right_mode = normalize_surface_mode(right_boundary.surface_mode)
    if left_mode != right_mode:
        return False

    point_distances = np.linalg.norm(left_curve - right_curve, axis=1)
    mean_jump = float(point_distances.mean())
    max_jump = float(point_distances.max())
    centroid_jump = float(np.linalg.norm(left_curve.mean(axis=0) - right_curve.mean(axis=0)))
    scale = max(1.0, min(_polyline_length(left_curve), _polyline_length(right_curve)))
    jump_limit = max(30.0, 0.25 * scale)
    centroid_limit = max(25.0, 0.18 * scale)
    max_jump_limit = max(80.0, 0.35 * scale)
    if mean_jump > jump_limit and centroid_jump > centroid_limit:
        return False
    if mean_jump > 30.0 and max_jump > max_jump_limit:
        return False
    return True


def _full_loft_faces(
    base: int,
    next_base: int,
    points_per_curve: int,
    is_closed: bool,
) -> List[List[int]]:
    left_indices = [base + index for index in range(points_per_curve)]
    right_indices = [next_base + index for index in range(points_per_curve)]
    return _indexed_loft_faces(left_indices, right_indices, is_closed)


def _indexed_loft_faces(
    left_indices: Sequence[int],
    right_indices: Sequence[int],
    is_closed: bool,
) -> List[List[int]]:
    faces: List[List[int]] = []
    points_per_curve = min(len(left_indices), len(right_indices))
    point_range = range(points_per_curve if is_closed else points_per_curve - 1)
    for point_index in point_range:
        next_point = (point_index + 1) % points_per_curve
        a = int(left_indices[point_index])
        b = int(left_indices[next_point])
        c = int(right_indices[point_index])
        d = int(right_indices[next_point])
        faces.append([a, c, b])
        faces.append([b, c, d])
    return faces


def _cap_closed_ring_faces(
    curve: np.ndarray,
    base: int,
    add_vertex: Callable[[np.ndarray], int],
    reverse: bool = False,
) -> List[List[int]]:
    if len(curve) < 3:
        return []
    center_index = add_vertex(np.asarray(curve, dtype=float).mean(axis=0))
    faces: List[List[int]] = []
    for point_index in range(len(curve)):
        next_point = (point_index + 1) % len(curve)
        a = base + point_index
        b = base + next_point
        if reverse:
            faces.append([center_index, b, a])
        else:
            faces.append([center_index, a, b])
    return faces


def _transition_loft_faces(
    left: Tuple[BoundarySlice, np.ndarray, bool],
    right: Tuple[BoundarySlice, np.ndarray, bool],
    base: int,
    next_base: int,
    add_vertex: Callable[[np.ndarray], int],
) -> List[List[int]]:
    left_boundary, left_curve, left_closed = left
    right_boundary, right_curve, right_closed = right
    if right_boundary.slice_index - left_boundary.slice_index > 1:
        return []
    if left_closed and right_closed:
        return []
    if not left_closed and not right_closed:
        return _subdivided_open_transition_faces(left_curve, right_curve, base, next_base, add_vertex)
    if not left_closed and right_closed:
        return _open_to_closed_transition_faces(left_curve, right_curve, base, next_base)
    return _closed_to_open_transition_faces(left_curve, right_curve, base, next_base)


def _align_loft_curve(left_curve: np.ndarray, right_curve: np.ndarray, is_closed: bool) -> np.ndarray:
    if is_closed:
        closed = _closed_contour_points(right_curve)
        aligned = _align_closed_polyline_start(closed, left_curve[0])
        return resample_polyline(aligned, len(right_curve) + 1)[:-1]

    same = _mean_curve_distance(left_curve, right_curve)
    reversed_curve = right_curve[::-1]
    reversed_mean = _mean_curve_distance(left_curve, reversed_curve)
    if reversed_mean < same * 0.5:
        return reversed_curve
    return right_curve


def _align_closed_curve_to_open_curve(closed_curve: np.ndarray, open_curve: np.ndarray) -> np.ndarray:
    closed = _closed_contour_points(closed_curve)
    aligned = _align_closed_polyline_start(closed, open_curve[0])
    return resample_polyline(aligned, len(closed_curve) + 1)[:-1]


def _align_open_curve_to_closed_curve(open_curve: np.ndarray, closed_curve: np.ndarray) -> np.ndarray:
    closed = _closed_contour_points(closed_curve)
    open_start = float(np.linalg.norm(open_curve[0] - closed[0]))
    open_end = float(np.linalg.norm(open_curve[-1] - closed[0]))
    if open_end < open_start:
        return open_curve[::-1]
    return open_curve


def _mean_curve_distance(left_curve: np.ndarray, right_curve: np.ndarray) -> float:
    if len(left_curve) != len(right_curve):
        count = min(len(left_curve), len(right_curve))
        left_curve = resample_polyline(left_curve, count)
        right_curve = resample_polyline(right_curve, count)
    return float(np.linalg.norm(left_curve - right_curve, axis=1).mean())


def _transition_distance_limit(left_curve: np.ndarray, right_curve: np.ndarray) -> float:
    scale = max(1.0, min(_polyline_length(left_curve), _polyline_length(right_curve)))
    return max(35.0, min(70.0, 0.18 * scale))


def _subdivided_open_transition_faces(
    left_curve: np.ndarray,
    right_curve: np.ndarray,
    base: int,
    next_base: int,
    add_vertex: Callable[[np.ndarray], int],
) -> List[List[int]]:
    distances = np.linalg.norm(left_curve - right_curve, axis=1)
    max_distance = float(distances.max()) if len(distances) else 0.0
    if max_distance <= 0.0:
        return _full_loft_faces(base, next_base, len(left_curve), is_closed=False)

    # Large real topology changes should not be bridged by one huge triangle
    # strip. Insert simple intermediate cross-sections so the generated patch is
    # explicit, local, and easy to spot in QC instead of becoming a folded slab.
    step_count = max(1, min(12, int(math.ceil(max_distance / 24.0))))
    faces: List[List[int]] = []
    previous_indices = [base + index for index in range(len(left_curve))]

    for step in range(1, step_count):
        t = step / float(step_count)
        intermediate = (1.0 - t) * left_curve + t * right_curve
        current_indices = [add_vertex(point) for point in intermediate]
        faces.extend(_indexed_loft_faces(previous_indices, current_indices, is_closed=False))
        previous_indices = current_indices

    right_indices = [next_base + index for index in range(len(right_curve))]
    faces.extend(_indexed_loft_faces(previous_indices, right_indices, is_closed=False))
    return faces


def _ring_path_indices(count: int, start: int, end: int) -> List[int]:
    indices = [int(start)]
    index = int(start)
    while index != int(end):
        index = (index + 1) % count
        indices.append(index)
    return indices


def _best_closed_subarc_indices(open_curve: np.ndarray, closed_curve: np.ndarray) -> Optional[List[int]]:
    if len(open_curve) < 2 or len(closed_curve) < 3:
        return None
    start_index = int(np.linalg.norm(closed_curve - open_curve[0], axis=1).argmin())
    end_index = int(np.linalg.norm(closed_curve - open_curve[-1], axis=1).argmin())
    candidates = [
        _ring_path_indices(len(closed_curve), start_index, end_index),
        list(reversed(_ring_path_indices(len(closed_curve), end_index, start_index))),
    ]
    best_indices: Optional[List[int]] = None
    best_mean = math.inf
    for indices in candidates:
        if len(indices) < 2:
            continue
        subarc = closed_curve[indices]
        sampled = resample_polyline(subarc, len(open_curve))
        mean_distance = _mean_curve_distance(open_curve, sampled)
        if mean_distance < best_mean:
            best_mean = mean_distance
            best_indices = indices

    if best_indices is None:
        return None
    limit = _transition_distance_limit(open_curve, closed_curve)
    if best_mean > limit:
        return None
    return best_indices


def _polyline_cumulative_fraction(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return np.zeros(len(points), dtype=float)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(cumulative[-1])
    if total <= 0:
        return np.linspace(0.0, 1.0, len(points))
    return cumulative / total


def _strip_faces_between_polylines(
    left_indices: Sequence[int],
    left_points: np.ndarray,
    right_indices: Sequence[int],
    right_points: np.ndarray,
) -> List[List[int]]:
    if len(left_indices) < 2 or len(right_indices) < 2:
        return []
    left_t = _polyline_cumulative_fraction(left_points)
    right_t = _polyline_cumulative_fraction(right_points)
    faces: List[List[int]] = []
    left_pos = 0
    right_pos = 0
    while left_pos < len(left_indices) - 1 or right_pos < len(right_indices) - 1:
        can_advance_left = left_pos < len(left_indices) - 1
        can_advance_right = right_pos < len(right_indices) - 1
        if not can_advance_right or (
            can_advance_left and left_t[left_pos + 1] <= right_t[right_pos + 1]
        ):
            faces.append(
                [
                    int(left_indices[left_pos]),
                    int(right_indices[right_pos]),
                    int(left_indices[left_pos + 1]),
                ]
            )
            left_pos += 1
        else:
            faces.append(
                [
                    int(left_indices[left_pos]),
                    int(right_indices[right_pos]),
                    int(right_indices[right_pos + 1]),
                ]
            )
            right_pos += 1
    return faces


def _open_to_closed_transition_faces(
    open_curve: np.ndarray,
    closed_curve: np.ndarray,
    open_base: int,
    closed_base: int,
) -> List[List[int]]:
    subarc_indices = _best_closed_subarc_indices(open_curve, closed_curve)
    if subarc_indices is None:
        return []
    open_indices = [open_base + index for index in range(len(open_curve))]
    closed_indices = [closed_base + index for index in subarc_indices]
    return _strip_faces_between_polylines(
        open_indices,
        open_curve,
        closed_indices,
        closed_curve[subarc_indices],
    )


def _closed_to_open_transition_faces(
    closed_curve: np.ndarray,
    open_curve: np.ndarray,
    closed_base: int,
    open_base: int,
) -> List[List[int]]:
    subarc_indices = _best_closed_subarc_indices(open_curve, closed_curve)
    if subarc_indices is None:
        return []
    closed_indices = [closed_base + index for index in subarc_indices]
    open_indices = [open_base + index for index in range(len(open_curve))]
    return _strip_faces_between_polylines(
        closed_indices,
        closed_curve[subarc_indices],
        open_indices,
        open_curve,
    )


def write_ply(path: str | Path, mesh: SurfaceMesh) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(mesh.vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write(f"element face {len(mesh.faces)}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for vertex in mesh.vertices:
            handle.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        for face in mesh.faces:
            handle.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def _write_colored_line_ply(
    path: str | Path,
    vertices: np.ndarray,
    edges: Sequence[Tuple[int, int]],
    vertex_colors: Optional[np.ndarray] = None,
    edge_colors: Optional[np.ndarray] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    if vertex_colors is None:
        vertex_colors = np.full((len(vertices), 3), 255, dtype=np.uint8)
    else:
        vertex_colors = np.asarray(vertex_colors, dtype=np.uint8).reshape(-1, 3)
    if edge_colors is None:
        edge_colors = np.full((len(edges), 3), 255, dtype=np.uint8)
    else:
        edge_colors = np.asarray(edge_colors, dtype=np.uint8).reshape(-1, 3)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element edge {len(edges)}\n")
        handle.write("property int vertex1\nproperty int vertex2\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for vertex, color in zip(vertices, vertex_colors):
            handle.write(
                f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for edge, color in zip(edges, edge_colors):
            handle.write(
                f"{int(edge[0])} {int(edge[1])} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _write_colored_mesh_ply(
    path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: Optional[np.ndarray] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if vertex_colors is None:
        vertex_colors = np.full((len(vertices), 3), 220, dtype=np.uint8)
    else:
        vertex_colors = np.asarray(vertex_colors, dtype=np.uint8).reshape(-1, 3)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for vertex, color in zip(vertices, vertex_colors):
            handle.write(
                f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def write_obj(path: str | Path, mesh: SurfaceMesh) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"o {mesh.name}\n")
        for vertex in mesh.vertices:
            handle.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        for face in mesh.faces:
            # OBJ indices are 1-based.
            handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def _write_colored_obj(
    path: str | Path,
    name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_materials: Optional[Sequence[str]] = None,
    materials: Optional[Dict[str, Tuple[int, int, int]]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if face_materials is None:
        face_materials = ["default"] * len(faces)
    if materials is None:
        materials = {"default": (220, 220, 220)}

    mtl_path = path.with_suffix(".mtl")
    with mtl_path.open("w", encoding="utf-8") as handle:
        for material_name, color in materials.items():
            rgb = np.asarray(color, dtype=float) / 255.0
            handle.write(f"newmtl {material_name}\n")
            handle.write(f"Kd {rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}\n")
            handle.write("Ka 0.000000 0.000000 0.000000\n")
            handle.write("Ks 0.000000 0.000000 0.000000\n")
            handle.write("d 1.000000\n\n")

    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"mtllib {mtl_path.name}\n")
        handle.write(f"o {name}\n")
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        active_material = None
        for face, material_name in zip(faces, face_materials):
            if material_name != active_material:
                handle.write(f"usemtl {material_name}\n")
                active_material = material_name
            handle.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def mask_constrained_surface_patches(
    mask: np.ndarray,
    boundaries: Sequence[BoundarySlice],
    resample_points: int = 80,
    slice_axis: int | str = 0,
    component_seed_distance: float = 6.0,
    max_surface_quads: int = 1_500_000,
) -> Dict[str, SurfaceMesh]:
    """Build patch surfaces directly from exposed voxel faces of the mask."""

    mask = _as_bool_mask(mask)
    quad_count = _count_exposed_voxel_quads(mask)
    if quad_count > max_surface_quads:
        raise ValueError(
            f"Mask surface has {quad_count:,} exposed faces; limit is {max_surface_quads:,}"
        )

    slice_axis_int = _slice_axis_to_int(slice_axis)
    guarded_slice_planes = _large_slice_transition_planes(mask, slice_axis_int)
    if guarded_slice_planes:
        preview = ", ".join(f"{plane:g}" for plane in sorted(guarded_slice_planes)[:8])
        extra = "" if len(guarded_slice_planes) <= 8 else f", +{len(guarded_slice_planes) - 8} more"
        warnings.warn(
            "Skipping large slice-transition cap faces while building surfaces: "
            f"{preview}{extra}",
            RuntimeWarning,
        )

    seed_trees = _surface_seed_trees(boundaries)
    vertices_by_label: Dict[int, List[Tuple[int, int, int]]] = {
        BOUNDARY_OUTER: [],
        BOUNDARY_INNER: [],
        BOUNDARY_LATERAL: [],
    }
    vertex_index_by_label: Dict[int, Dict[Tuple[int, int, int], int]] = {
        BOUNDARY_OUTER: {},
        BOUNDARY_INNER: {},
        BOUNDARY_LATERAL: {},
    }
    faces_by_label: Dict[int, List[List[int]]] = {
        BOUNDARY_OUTER: [],
        BOUNDARY_INNER: [],
        BOUNDARY_LATERAL: [],
    }

    def vertex_index(label: int, vertex: Tuple[int, int, int]) -> int:
        index_by_vertex = vertex_index_by_label[label]
        existing = index_by_vertex.get(vertex)
        if existing is not None:
            return existing
        index = len(vertices_by_label[label])
        index_by_vertex[vertex] = index
        vertices_by_label[label].append(vertex)
        return index

    def add_quad(label: int, vertices: Sequence[Tuple[int, int, int]]) -> None:
        if label not in faces_by_label:
            return
        indices = [vertex_index(label, tuple(vertex)) for vertex in vertices]
        faces_by_label[label].append([indices[0], indices[1], indices[2]])
        faces_by_label[label].append([indices[2], indices[1], indices[3]])

    def emit(side: str, x: int, y_values: np.ndarray, z_values: np.ndarray) -> None:
        if len(y_values) == 0:
            return
        centers = _voxel_face_centers(side, x, y_values, z_values)
        keep = _surface_face_guard_mask(
            side,
            centers,
            slice_axis_int,
            guarded_slice_planes,
        )
        if not np.any(keep):
            return
        if not np.all(keep):
            centers = centers[keep]
            y_values = y_values[keep]
            z_values = z_values[keep]
        labels = _classify_surface_centers(centers, seed_trees)
        for label, y, z in zip(labels, y_values, z_values):
            add_quad(int(label), _voxel_face_vertices2(side, x, int(y), int(z)))

    previous = np.zeros(mask.shape[1:], dtype=bool)
    for x in range(mask.shape[0]):
        current = np.asarray(mask[x], dtype=bool)

        y_values, z_values = np.nonzero(current & ~previous)
        emit("x_minus", x, y_values, z_values)
        if x > 0:
            y_values, z_values = np.nonzero(previous & ~current)
            emit("x_plus", x - 1, y_values, z_values)

        z_values = np.nonzero(current[0, :])[0]
        emit("y_minus", x, np.zeros(len(z_values), dtype=int), z_values)
        z_values = np.nonzero(current[-1, :])[0]
        emit("y_plus", x, np.full(len(z_values), current.shape[0] - 1, dtype=int), z_values)
        y_values, z_values = np.nonzero(current[1:, :] & ~current[:-1, :])
        emit("y_minus", x, y_values + 1, z_values)
        y_values, z_values = np.nonzero(current[:-1, :] & ~current[1:, :])
        emit("y_plus", x, y_values, z_values)

        y_values = np.nonzero(current[:, 0])[0]
        emit("z_minus", x, y_values, np.zeros(len(y_values), dtype=int))
        y_values = np.nonzero(current[:, -1])[0]
        emit("z_plus", x, y_values, np.full(len(y_values), current.shape[1] - 1, dtype=int))
        y_values, z_values = np.nonzero(current[:, 1:] & ~current[:, :-1])
        emit("z_minus", x, y_values, z_values + 1)
        y_values, z_values = np.nonzero(current[:, :-1] & ~current[:, 1:])
        emit("z_plus", x, y_values, z_values)

        previous = current

    y_values, z_values = np.nonzero(previous)
    emit("x_plus", mask.shape[0] - 1, y_values, z_values)

    meshes: Dict[str, SurfaceMesh] = {}
    for label, name in (
        (BOUNDARY_OUTER, "outer"),
        (BOUNDARY_INNER, "inner"),
        (BOUNDARY_LATERAL, "lateral"),
    ):
        faces = faces_by_label[label]
        if not faces:
            continue
        vertices = np.asarray(vertices_by_label[label], dtype=float) * 0.5
        mesh = SurfaceMesh(name=name, vertices=vertices, faces=np.asarray(faces, dtype=np.int32))
        meshes[name] = _keep_seed_connected_surface_components(
            mesh,
            _surface_seed_points_for_label(boundaries, label),
            max_seed_distance=float(component_seed_distance),
        )

    if "outer" not in meshes or "inner" not in meshes:
        missing = ", ".join(name for name in ("outer", "inner") if name not in meshes)
        raise ValueError(f"Voxel surface build produced no {missing} mesh")
    return meshes


def _large_slice_transition_planes(
    mask: np.ndarray,
    slice_axis: int,
    min_changed_faces: int = 1000,
    percentile: float = 99.0,
) -> set[float]:
    """Find abrupt slice-to-slice mask jumps that would become artificial caps."""

    if mask.shape[slice_axis] < 2:
        return set()

    changes: List[int] = []
    for index in range(mask.shape[slice_axis] - 1):
        left = _take_slice(mask, index, slice_axis)
        right = _take_slice(mask, index + 1, slice_axis)
        changes.append(int(np.count_nonzero(left != right)))

    nonzero = np.asarray([value for value in changes if value > 0], dtype=float)
    if len(nonzero) == 0:
        return set()

    threshold = max(float(min_changed_faces), float(np.percentile(nonzero, percentile)))
    return {
        float(index) + 0.5
        for index, changed_faces in enumerate(changes)
        if float(changed_faces) >= threshold
    }


def _surface_side_axis(side: str) -> int:
    if side.startswith("x_"):
        return 0
    if side.startswith("y_"):
        return 1
    if side.startswith("z_"):
        return 2
    raise ValueError(f"Unknown voxel face side: {side}")


def _surface_face_guard_mask(
    side: str,
    centers: np.ndarray,
    slice_axis: int,
    guarded_slice_planes: set[float],
) -> np.ndarray:
    keep = np.ones(len(centers), dtype=bool)
    if not guarded_slice_planes or _surface_side_axis(side) != slice_axis:
        return keep

    guarded = {round(float(value), 3) for value in guarded_slice_planes}
    center_planes = np.round(centers[:, slice_axis].astype(float), 3)
    for index, plane in enumerate(center_planes):
        if float(plane) in guarded:
            keep[index] = False
    return keep


def _count_exposed_voxel_quads(mask: np.ndarray) -> int:
    previous = np.zeros(mask.shape[1:], dtype=bool)
    total = 0
    for x in range(mask.shape[0]):
        current = np.asarray(mask[x], dtype=bool)
        total += int(np.count_nonzero(current & ~previous))
        total += int(np.count_nonzero(previous & ~current))
        total += int(np.count_nonzero(current[0, :]))
        total += int(np.count_nonzero(current[-1, :]))
        total += int(np.count_nonzero(current[1:, :] != current[:-1, :]))
        total += int(np.count_nonzero(current[:, 0]))
        total += int(np.count_nonzero(current[:, -1]))
        total += int(np.count_nonzero(current[:, 1:] != current[:, :-1]))
        previous = current
    total += int(np.count_nonzero(previous))
    return total


def _mask_for_annotated_components(
    mask: np.ndarray,
    boundaries: Sequence[BoundarySlice],
    padding: int = 16,
) -> np.ndarray:
    seed_points = _surface_seed_points_for_label(boundaries, BOUNDARY_OUTER)
    inner_points = _surface_seed_points_for_label(boundaries, BOUNDARY_INNER)
    lateral_points = _surface_seed_points_for_label(boundaries, BOUNDARY_LATERAL)
    if len(inner_points) > 0:
        seed_points = np.vstack([seed_points, inner_points]) if len(seed_points) > 0 else inner_points
    if len(lateral_points) > 0:
        seed_points = np.vstack([seed_points, lateral_points]) if len(seed_points) > 0 else lateral_points
    if len(seed_points) == 0:
        return mask

    slices = _bounding_slices_for_points(seed_points, mask.shape, padding=padding)
    crop = np.asarray(mask[slices], dtype=bool)
    if not np.any(crop):
        return mask

    structure = ndimage.generate_binary_structure(3, 3)
    components, _component_count = ndimage.label(crop, structure=structure)
    origin = np.asarray([item.start for item in slices], dtype=float)
    hits: Dict[int, int] = defaultdict(int)
    for point in seed_points:
        label = _nearest_component_label(components, np.asarray(point, dtype=float) - origin)
        if label > 0:
            hits[int(label)] += 1

    if not hits:
        return mask

    min_hits = max(3, int(math.ceil(sum(hits.values()) * 0.01)))
    keep_labels = [label for label, count in hits.items() if count >= min_hits]
    if not keep_labels:
        keep_labels = [max(hits, key=hits.get)]

    selected = np.zeros(mask.shape, dtype=bool)
    selected[slices] = np.isin(components, np.asarray(keep_labels, dtype=components.dtype))
    if not np.any(selected):
        return mask
    return selected


def _bounding_slices_for_points(
    points: np.ndarray,
    shape: Sequence[int],
    padding: int,
) -> Tuple[slice, slice, slice]:
    points = np.asarray(points, dtype=float)
    mins = np.floor(np.nanmin(points, axis=0)).astype(int) - int(padding)
    maxs = np.ceil(np.nanmax(points, axis=0)).astype(int) + int(padding) + 1
    starts = np.maximum(mins, 0)
    stops = np.minimum(maxs, np.asarray(shape, dtype=int))
    return tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops))  # type: ignore[return-value]


def _nearest_component_label(
    components: np.ndarray,
    point: np.ndarray,
    max_radius: int = 3,
) -> int:
    center = np.rint(point).astype(int)
    shape = np.asarray(components.shape, dtype=int)
    for radius in range(max_radius + 1):
        starts = np.maximum(center - radius, 0)
        stops = np.minimum(center + radius + 1, shape)
        if np.any(starts >= stops):
            continue
        local = components[
            starts[0] : stops[0],
            starts[1] : stops[1],
            starts[2] : stops[2],
        ]
        labels, counts = np.unique(local[local > 0], return_counts=True)
        if len(labels) > 0:
            return int(labels[int(np.argmax(counts))])
    return 0


def _surface_seed_trees(boundaries: Sequence[BoundarySlice]) -> Dict[int, cKDTree]:
    seed_points = {
        BOUNDARY_OUTER: _surface_seed_points_for_label(boundaries, BOUNDARY_OUTER),
        BOUNDARY_INNER: _surface_seed_points_for_label(boundaries, BOUNDARY_INNER),
        BOUNDARY_LATERAL: _surface_seed_points_for_label(boundaries, BOUNDARY_LATERAL),
    }
    trees = {
        label: cKDTree(points)
        for label, points in seed_points.items()
        if len(points) > 0
    }
    if BOUNDARY_OUTER not in trees:
        raise ValueError("Need outer seed points for mask-constrained surfaces")
    if BOUNDARY_INNER not in trees:
        raise ValueError("Need inner seed points for mask-constrained surfaces")
    return trees


def _classify_surface_centers(
    centers: np.ndarray,
    seed_trees: Dict[int, cKDTree],
) -> np.ndarray:
    if len(centers) == 0:
        return np.empty(0, dtype=np.uint8)

    labels = [BOUNDARY_OUTER, BOUNDARY_INNER, BOUNDARY_LATERAL]
    distances = np.full((len(labels), len(centers)), np.inf, dtype=float)
    for row, label in enumerate(labels):
        tree = seed_trees.get(label)
        if tree is None:
            continue
        distances[row], _ = tree.query(centers, k=1)
    best = np.argmin(distances, axis=0)
    return np.asarray([labels[index] for index in best], dtype=np.uint8)


def _surface_seed_points_for_label(
    boundaries: Sequence[BoundarySlice],
    label: int,
) -> np.ndarray:
    arcs: List[np.ndarray] = []
    for boundary in boundaries:
        if label == BOUNDARY_OUTER and len(boundary.outer_arc) > 0:
            arcs.append(boundary.outer_arc)
        elif label == BOUNDARY_INNER and len(boundary.inner_arc) > 0:
            arcs.append(boundary.inner_arc)
        elif label == BOUNDARY_LATERAL:
            arcs.extend(boundary.lateral_arcs)
    return _surface_label_points(arcs)


def _keep_seed_connected_surface_components(
    mesh: SurfaceMesh,
    seed_points: np.ndarray,
    max_seed_distance: float,
) -> SurfaceMesh:
    if len(mesh.faces) == 0 or len(seed_points) == 0:
        return mesh

    parent = np.arange(len(mesh.vertices), dtype=np.int32)
    size = np.ones(len(mesh.vertices), dtype=np.int32)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return int(index)

    def union(left: int, right: int) -> None:
        left_root = find(int(left))
        right_root = find(int(right))
        if left_root == right_root:
            return
        if size[left_root] < size[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        size[left_root] += size[right_root]

    for a, b, c in mesh.faces:
        union(int(a), int(b))
        union(int(a), int(c))

    vertex_roots = np.asarray([find(index) for index in range(len(mesh.vertices))], dtype=np.int32)
    face_roots = vertex_roots[mesh.faces[:, 0]]
    seed_tree = cKDTree(np.asarray(seed_points, dtype=float))
    keep_roots: List[int] = []
    largest_root = int(face_roots[0])
    largest_faces = 0

    for root in np.unique(face_roots):
        face_mask = face_roots == root
        face_count = int(np.count_nonzero(face_mask))
        if face_count > largest_faces:
            largest_faces = face_count
            largest_root = int(root)
        used_vertices = np.unique(mesh.faces[face_mask].reshape(-1))
        distances, _ = seed_tree.query(mesh.vertices[used_vertices], k=1)
        if float(np.min(distances)) <= max_seed_distance:
            keep_roots.append(int(root))

    if not keep_roots:
        keep_roots = [largest_root]

    keep_faces = np.isin(face_roots, np.asarray(keep_roots, dtype=np.int32))
    return _compact_surface_mesh(mesh.name, mesh.vertices, mesh.faces[keep_faces])


def _voxel_face_centers(
    side: str,
    x: int,
    y_values: np.ndarray,
    z_values: np.ndarray,
) -> np.ndarray:
    centers = np.zeros((len(y_values), 3), dtype=float)
    centers[:, 0] = float(x)
    centers[:, 1] = y_values.astype(float)
    centers[:, 2] = z_values.astype(float)
    if side == "x_minus":
        centers[:, 0] -= 0.5
    elif side == "x_plus":
        centers[:, 0] += 0.5
    elif side == "y_minus":
        centers[:, 1] -= 0.5
    elif side == "y_plus":
        centers[:, 1] += 0.5
    elif side == "z_minus":
        centers[:, 2] -= 0.5
    elif side == "z_plus":
        centers[:, 2] += 0.5
    return centers


def _voxel_face_vertices2(side: str, x: int, y: int, z: int) -> List[Tuple[int, int, int]]:
    x0, x1 = 2 * x - 1, 2 * x + 1
    y0, y1 = 2 * y - 1, 2 * y + 1
    z0, z1 = 2 * z - 1, 2 * z + 1
    xc, yc, zc = 2 * x, 2 * y, 2 * z
    if side == "x_minus":
        return [(x0, y0, z0), (x0, y0, z1), (x0, y1, z0), (x0, y1, z1)]
    if side == "x_plus":
        return [(x1, y0, z0), (x1, y1, z0), (x1, y0, z1), (x1, y1, z1)]
    if side == "y_minus":
        return [(x0, y0, z0), (x1, y0, z0), (x0, y0, z1), (x1, y0, z1)]
    if side == "y_plus":
        return [(x0, y1, z0), (x0, y1, z1), (x1, y1, z0), (x1, y1, z1)]
    if side == "z_minus":
        return [(x0, y0, z0), (x0, y1, z0), (x1, y0, z0), (x1, y1, z0)]
    if side == "z_plus":
        return [(x0, y0, z1), (x1, y0, z1), (x0, y1, z1), (x1, y1, z1)]
    raise ValueError(f"Unknown voxel face side: {side}")


def build_voxel_shell_mesh(
    mask: np.ndarray,
    spacing: Optional[np.ndarray] = None,
    max_surface_quads: int = 1_500_000,
) -> ShellMesh:
    """Build one complete voxel shell mesh from all exposed mask faces."""

    mask = _as_bool_mask(mask)
    quad_count = _count_exposed_voxel_quads(mask)
    if quad_count > max_surface_quads:
        raise ValueError(
            f"Mask surface has {quad_count:,} exposed faces; limit is {max_surface_quads:,}"
        )

    vertices: List[Tuple[int, int, int]] = []
    vertex_by_key: Dict[Tuple[int, int, int], int] = {}
    faces: List[List[int]] = []
    face_sources: List[Dict] = []

    def vertex_index(vertex: Tuple[int, int, int]) -> int:
        existing = vertex_by_key.get(vertex)
        if existing is not None:
            return existing
        index = len(vertices)
        vertex_by_key[vertex] = index
        vertices.append(vertex)
        return index

    def add_quad(side: str, x: int, y: int, z: int) -> None:
        quad = _voxel_face_vertices2(side, x, y, z)
        indices = [vertex_index(tuple(vertex)) for vertex in quad]
        faces.append([indices[0], indices[1], indices[2]])
        faces.append([indices[2], indices[1], indices[3]])
        source = {"voxel_x": int(x), "voxel_y": int(y), "voxel_z": int(z), "side": side}
        face_sources.append(dict(source))
        face_sources.append(dict(source))

    def emit(side: str, x: int, y_values: np.ndarray, z_values: np.ndarray) -> None:
        for y, z in zip(y_values, z_values):
            add_quad(side, x, int(y), int(z))

    previous = np.zeros(mask.shape[1:], dtype=bool)
    for x in range(mask.shape[0]):
        current = np.asarray(mask[x], dtype=bool)

        y_values, z_values = np.nonzero(current & ~previous)
        emit("x_minus", x, y_values, z_values)
        if x > 0:
            y_values, z_values = np.nonzero(previous & ~current)
            emit("x_plus", x - 1, y_values, z_values)

        z_values = np.nonzero(current[0, :])[0]
        emit("y_minus", x, np.zeros(len(z_values), dtype=int), z_values)
        z_values = np.nonzero(current[-1, :])[0]
        emit("y_plus", x, np.full(len(z_values), current.shape[0] - 1, dtype=int), z_values)
        y_values, z_values = np.nonzero(current[1:, :] & ~current[:-1, :])
        emit("y_minus", x, y_values + 1, z_values)
        y_values, z_values = np.nonzero(current[:-1, :] & ~current[1:, :])
        emit("y_plus", x, y_values, z_values)

        y_values = np.nonzero(current[:, 0])[0]
        emit("z_minus", x, y_values, np.zeros(len(y_values), dtype=int))
        y_values = np.nonzero(current[:, -1])[0]
        emit("z_plus", x, y_values, np.full(len(y_values), current.shape[1] - 1, dtype=int))
        y_values, z_values = np.nonzero(current[:, 1:] & ~current[:, :-1])
        emit("z_minus", x, y_values, z_values + 1)
        y_values, z_values = np.nonzero(current[:, :-1] & ~current[:, 1:])
        emit("z_plus", x, y_values, z_values)

        previous = current

    y_values, z_values = np.nonzero(previous)
    emit("x_plus", mask.shape[0] - 1, y_values, z_values)

    return ShellMesh(
        vertices=np.asarray(vertices, dtype=float) * 0.5,
        faces=np.asarray(faces, dtype=np.int32).reshape(-1, 3),
        spacing=np.asarray(spacing if spacing is not None else np.ones(3), dtype=float),
        backend="voxel",
        coordinate_space="index",
        face_sources=face_sources,
    )


_MARCHING_CUBE_CORNERS = np.asarray(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.int16,
)
_MARCHING_TETRAHEDRA = (
    (0, 5, 1, 6),
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
)


def _marching_edge_key(left: np.ndarray, right: np.ndarray) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    left_key = tuple(int(value) for value in left)
    right_key = tuple(int(value) for value in right)
    return (left_key, right_key) if left_key <= right_key else (right_key, left_key)


def _add_unique_triangle(
    faces: List[List[int]],
    seen_faces: set[Tuple[int, int, int]],
    a: int,
    b: int,
    c: int,
) -> None:
    if a == b or b == c or a == c:
        return
    key = tuple(sorted((int(a), int(b), int(c))))
    if key in seen_faces:
        return
    seen_faces.add(key)
    faces.append([int(a), int(b), int(c)])


def build_marching_cubes_shell_mesh(
    mask: np.ndarray,
    spacing: Optional[np.ndarray] = None,
    max_surface_quads: int = 1_500_000,
    smoothing_sigma: float = 0.65,
    iso_value: float = 0.5,
) -> ShellMesh:
    """Build a smoother isosurface shell without adding a scikit-image dependency.

    The implementation uses marching tetrahedra inside each cube. It serves the
    same purpose as marching cubes here: turn the mask into a triangle shell that
    is less blocky than the debug voxel backend.
    """

    mask = _as_bool_mask(mask)
    if not np.any(mask):
        raise ValueError("Cannot build a marching-cubes shell from an empty mask")

    padded = np.pad(mask.astype(float), 1, mode="constant", constant_values=0.0)
    field = ndimage.gaussian_filter(padded, sigma=float(smoothing_sigma)) if smoothing_sigma > 0 else padded

    corner_fields = [
        field[
            corner[0] : corner[0] + field.shape[0] - 1,
            corner[1] : corner[1] + field.shape[1] - 1,
            corner[2] : corner[2] + field.shape[2] - 1,
        ]
        for corner in _MARCHING_CUBE_CORNERS
    ]
    cell_min = np.minimum.reduce(corner_fields)
    cell_max = np.maximum.reduce(corner_fields)
    active = (cell_min <= iso_value) & (cell_max >= iso_value) & (cell_max > cell_min)
    active_cells = np.argwhere(active)
    if len(active_cells) == 0:
        raise ValueError("Marching-cubes shell has no active cells; mask may be too small after smoothing")
    if len(active_cells) > max_surface_quads:
        raise ValueError(
            f"Marching-cubes shell has {len(active_cells):,} active cells; limit is {max_surface_quads:,}"
        )

    vertices: List[np.ndarray] = []
    vertex_by_edge: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], int] = {}
    faces: List[List[int]] = []
    seen_faces: set[Tuple[int, int, int]] = set()

    def edge_vertex(corner_a: np.ndarray, value_a: float, corner_b: np.ndarray, value_b: float) -> int:
        key = _marching_edge_key(corner_a, corner_b)
        existing = vertex_by_edge.get(key)
        if existing is not None:
            return existing
        denom = float(value_b - value_a)
        if abs(denom) <= 1e-12:
            t = 0.5
        else:
            t = float(np.clip((iso_value - value_a) / denom, 0.0, 1.0))
        point = np.asarray(corner_a, dtype=float) + t * (np.asarray(corner_b, dtype=float) - np.asarray(corner_a, dtype=float))
        point -= 1.0
        index = len(vertices)
        vertex_by_edge[key] = index
        vertices.append(point)
        return index

    def add_tetra(cell_origin: np.ndarray, tetra: Tuple[int, int, int, int]) -> None:
        corner_ids = list(tetra)
        coords = [cell_origin + _MARCHING_CUBE_CORNERS[corner] for corner in corner_ids]
        values = [float(field[tuple(coord)]) for coord in coords]
        inside = [index for index, value in enumerate(values) if value >= iso_value]
        outside = [index for index, value in enumerate(values) if value < iso_value]

        if len(inside) == 0 or len(inside) == 4:
            return
        if len(inside) == 1:
            center = inside[0]
            verts = [
                edge_vertex(coords[center], values[center], coords[index], values[index])
                for index in outside
            ]
            _add_unique_triangle(faces, seen_faces, verts[0], verts[1], verts[2])
            return
        if len(inside) == 3:
            center = outside[0]
            verts = [
                edge_vertex(coords[center], values[center], coords[index], values[index])
                for index in inside
            ]
            _add_unique_triangle(faces, seen_faces, verts[0], verts[2], verts[1])
            return

        left, right = inside
        outer_left, outer_right = outside
        v00 = edge_vertex(coords[left], values[left], coords[outer_left], values[outer_left])
        v01 = edge_vertex(coords[left], values[left], coords[outer_right], values[outer_right])
        v10 = edge_vertex(coords[right], values[right], coords[outer_left], values[outer_left])
        v11 = edge_vertex(coords[right], values[right], coords[outer_right], values[outer_right])
        _add_unique_triangle(faces, seen_faces, v00, v10, v01)
        _add_unique_triangle(faces, seen_faces, v01, v10, v11)

    for cell_origin in active_cells:
        origin = np.asarray(cell_origin, dtype=np.int64)
        for tetra in _MARCHING_TETRAHEDRA:
            add_tetra(origin, tetra)

    if not vertices or not faces:
        raise ValueError("Marching-cubes shell produced no triangles")

    return ShellMesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int32).reshape(-1, 3),
        spacing=np.asarray(spacing if spacing is not None else np.ones(3), dtype=float),
        backend=SHELL_BACKEND_MARCHING_CUBES,
        coordinate_space="index",
        face_sources=None,
    )


def build_shell_mesh(
    mask: np.ndarray,
    spacing: Optional[np.ndarray] = None,
    shell_backend: str = SHELL_BACKEND_VOXEL,
    max_surface_quads: int = 1_500_000,
) -> ShellMesh:
    backend = normalize_shell_backend(shell_backend)
    if backend == SHELL_BACKEND_VOXEL:
        return build_voxel_shell_mesh(mask, spacing=spacing, max_surface_quads=max_surface_quads)
    if backend == SHELL_BACKEND_MARCHING_CUBES:
        return build_marching_cubes_shell_mesh(mask, spacing=spacing, max_surface_quads=max_surface_quads)
    raise ValueError(f"Unknown shell_backend: {shell_backend}")


def _dedupe_polyline_points(points: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) == 0:
        return points
    kept = [points[0]]
    for point in points[1:]:
        if float(np.linalg.norm(point - kept[-1])) > tolerance:
            kept.append(point)
    return np.asarray(kept, dtype=float)


def _closed_polyline(points: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    points = _dedupe_polyline_points(points, tolerance=tolerance)
    if len(points) == 0:
        return points
    if float(np.linalg.norm(points[0] - points[-1])) > tolerance:
        points = np.vstack([points, points[0]])
    return points


def _oriented_boundary_arcs(
    boundaries: Sequence[BoundarySlice],
    arc_name: str,
) -> List[np.ndarray]:
    arcs: List[np.ndarray] = []
    for boundary in sorted(boundaries, key=lambda item: item.slice_index):
        arc = np.asarray(getattr(boundary, arc_name), dtype=float)
        if len(arc) < 2:
            continue
        if arcs:
            previous = arcs[-1]
            direct = float(np.linalg.norm(arc[0] - previous[0]) + np.linalg.norm(arc[-1] - previous[-1]))
            swapped = float(np.linalg.norm(arc[-1] - previous[0]) + np.linalg.norm(arc[0] - previous[-1]))
            if swapped < direct:
                arc = arc[::-1]
        arcs.append(arc)
    return arcs


def _cut_curve_from_boundary_arcs(
    boundaries: Sequence[BoundarySlice],
    arc_name: str,
    curve_id: str,
    label_left: str,
    label_right: str,
) -> Optional[SurfaceCutCurve]:
    arcs = _oriented_boundary_arcs(boundaries, arc_name)
    if len(arcs) < 2:
        return None

    start_track = np.asarray([arc[0] for arc in arcs], dtype=float)
    end_track = np.asarray([arc[-1] for arc in arcs], dtype=float)
    loop = np.vstack(
        [
            start_track,
            arcs[-1],
            end_track[::-1],
            arcs[0][::-1],
        ]
    )
    loop = _closed_polyline(loop)
    return SurfaceCutCurve(
        curve_id=curve_id,
        label_left=label_left,
        label_right=label_right,
        control_points=loop,
        source="annotation_derived",
    )


def infer_cut_curves_from_boundaries(
    boundaries: Sequence[BoundarySlice],
) -> List[SurfaceCutCurve]:
    curves: List[SurfaceCutCurve] = []
    outer = _cut_curve_from_boundary_arcs(
        boundaries,
        "outer_arc",
        "outer_lateral_boundary",
        "outer",
        "lateral",
    )
    inner = _cut_curve_from_boundary_arcs(
        boundaries,
        "inner_arc",
        "inner_lateral_boundary",
        "inner",
        "lateral",
    )
    if outer is not None:
        curves.append(outer)
    if inner is not None:
        curves.append(inner)
    return curves


def _shell_cut_warning_row(
    warning_type: str,
    severity: str,
    object_id: str,
    message: str,
    suggested_action: str,
) -> Dict:
    return {
        "warning_type": warning_type,
        "severity": severity,
        "object_id": object_id,
        "message": message,
        "suggested_action": suggested_action,
    }


def _is_manual_shell_cut_annotation_data(data: object) -> bool:
    return isinstance(data, dict) and (
        data.get("schema") == SHELL_CUT_ANNOTATION_SCHEMA
        or data.get("annotation_type") == "manual_2d_shell_cut_boundary"
    )


def shell_cut_json_source(path: Optional[str | Path]) -> str:
    if path is None:
        return "annotation_derived"
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and data.get("schema") == SURFACE_3D_ANNOTATION_SCHEMA:
        return "manual_3d"
    if _is_manual_shell_cut_annotation_data(data):
        return "manual_2d"
    return "json"


def _resolve_shell_cut_json_path(
    output_dir: Path,
    surface_method: str,
    cut_curve_json: Optional[str | Path],
) -> Optional[Path]:
    if cut_curve_json:
        return Path(cut_curve_json).expanduser()
    if normalize_surface_build_method(surface_method) != SURFACE_BUILD_SHELL_CUT:
        return None

    candidates = [
        output_dir / "shell_cut_annotations.json",
        output_dir.parent / "shell_cut_annotations.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _read_explicit_shell_cut_json(data: object) -> Tuple[List[SurfaceCutCurve], List[SurfacePatchSeed]]:
    curve_items = data if isinstance(data, list) else data.get("cut_curves", data.get("curves", []))
    seed_items = [] if isinstance(data, list) else data.get(
        "selected_patches",
        data.get("seeds", data.get("patch_seeds", [])),
    )

    curves: List[SurfaceCutCurve] = []
    for index, item in enumerate(curve_items):
        points = item.get("control_points", item.get("points"))
        if points is None:
            raise ValueError(f"Cut curve {index} is missing control_points/points")
        curve_id = str(item.get("curve_id", item.get("id", f"cut_curve_{index}")))
        label_left = str(item.get("label_left", "inner" if "inner" in curve_id else "outer"))
        label_right = str(item.get("label_right", "lateral"))
        curves.append(
            SurfaceCutCurve(
                curve_id=curve_id,
                label_left=label_left,
                label_right=label_right,
                control_points=_closed_polyline(np.asarray(points, dtype=float)),
                source=str(item.get("source", "json")),
                confidence=float(item.get("confidence", 1.0)),
            )
        )

    seeds: List[SurfacePatchSeed] = []
    for index, item in enumerate(seed_items):
        point = item.get("seed_point", item.get("point"))
        if point is None:
            raise ValueError(f"Patch seed {index} is missing seed_point/point")
        seeds.append(
            SurfacePatchSeed(
                patch_label=str(
                    item.get(
                        "patch_label",
                        item.get("surface_name", item.get("label", f"seed_{index}")),
                    )
                ),
                seed_point=np.asarray(point, dtype=float),
                source=str(item.get("source", "json")),
            )
        )
    return curves, seeds


def _manual_shell_cut_contour_lookup(contours: Sequence[Contour2D]) -> Dict[Tuple[int, int], Contour2D]:
    return {
        (int(contour.slice_index), int(contour.contour_id)): contour
        for contour in contours
    }


def _manual_shell_cut_object_id(row: Dict) -> str:
    return f"slice={int(row['slice_index'])},contour={int(row['contour_id'])}"


def _manual_shell_cut_point(
    raw_row: Dict,
    point_name: str,
    contour_points: np.ndarray,
) -> Tuple[int, np.ndarray, List[Dict]]:
    point_items = raw_row.get("points", {})
    item = point_items.get(point_name)
    if item is None:
        raise ValueError(
            f"Shell-cut annotation slice {raw_row.get('slice_index')} is missing {point_name}"
        )

    saved_point = None
    index = None
    if isinstance(item, dict):
        if item.get("index") is not None:
            index = int(item["index"])
        if item.get("point") is not None:
            saved_point = np.asarray(item["point"], dtype=float).reshape(-1)
    else:
        saved_point = np.asarray(item, dtype=float).reshape(-1)

    if index is None:
        if saved_point is None or len(saved_point) != 3:
            raise ValueError(
                f"Shell-cut annotation slice {raw_row.get('slice_index')} has no usable {point_name}"
            )
        index = _nearest_index(contour_points, saved_point)

    if index < 0 or index >= len(contour_points):
        raise ValueError(
            f"Shell-cut annotation slice {raw_row.get('slice_index')} has out-of-range "
            f"{point_name} index {index}"
        )

    point = np.asarray(contour_points[index], dtype=float)
    warnings_out: List[Dict] = []
    if saved_point is not None and len(saved_point) == 3:
        distance = float(np.linalg.norm(saved_point - point))
        if distance > 1.0:
            warnings_out.append(
                _shell_cut_warning_row(
                    "point_index_mismatch",
                    "warning",
                    f"{raw_row.get('slice_index')}:{point_name}",
                    f"{point_name} saved point is {distance:.3f} voxels from its saved contour index",
                    "reload the annotation on the current contour and re-accept the slice",
                )
            )
    return int(index), point, warnings_out


def _parse_manual_shell_cut_row(
    raw_row: Dict,
    row_index: int,
    contour_lookup: Dict[Tuple[int, int], Contour2D],
) -> Tuple[Dict, List[Dict]]:
    try:
        slice_index = int(raw_row["slice_index"])
        contour_id = int(raw_row.get("contour_id", 0))
    except Exception as exc:
        raise ValueError(f"Shell-cut annotation row {row_index} is missing slice_index/contour_id") from exc

    contour = contour_lookup.get((slice_index, contour_id))
    if contour is None:
        raise ValueError(
            f"Shell-cut annotation row {row_index} references missing contour "
            f"slice={slice_index}, contour={contour_id}"
        )

    contour_points = _normalize_contour(contour.points)
    if len(contour_points) < 3:
        raise ValueError(
            f"Shell-cut annotation row {row_index} uses a contour with fewer than 3 points"
        )

    surface_mode = normalize_surface_mode(raw_row.get("surface_mode"))
    if surface_mode != SURFACE_MODE_NORMAL:
        return (
            {
                "slice_index": slice_index,
                "contour_id": contour_id,
                "contour_points": contour_points,
                "surface_mode": surface_mode,
                "outer_path": "whole" if surface_mode == SURFACE_MODE_OUTER_ONLY else "",
                "inner_path": "whole" if surface_mode == SURFACE_MODE_INNER_ONLY else "",
                "points": {},
            },
            [],
        )

    warnings_out: List[Dict] = []
    points: Dict[str, Dict[str, object]] = {}
    for point_name in SHELL_CUT_MANUAL_POINT_NAMES:
        index, point, point_warnings = _manual_shell_cut_point(raw_row, point_name, contour_points)
        warnings_out.extend(point_warnings)
        points[point_name] = {
            "index": index,
            "point": point,
        }

    row = {
        "slice_index": slice_index,
        "contour_id": contour_id,
        "contour_points": contour_points,
        "surface_mode": SURFACE_MODE_NORMAL,
        "outer_path": str(raw_row.get("outer_path", "auto")),
        "inner_path": str(raw_row.get("inner_path", "auto")),
        "points": points,
    }
    return row, warnings_out


def _manual_shell_cut_cap_arc(
    row: Dict,
    prefix: str,
) -> Tuple[np.ndarray, float, float]:
    contour_points = np.asarray(row["contour_points"], dtype=float)
    a_name = f"{prefix}_cut_A"
    b_name = f"{prefix}_cut_B"
    if int(row["points"][a_name]["index"]) == int(row["points"][b_name]["index"]):
        raise ValueError("Shell-cut cap endpoints cannot use the same contour point")

    (
        outer_arc,
        _outer_indices,
        _outer_direction,
        _outer_score,
        inner_arc,
        _inner_indices,
        _inner_direction,
        _inner_score,
    ) = _choose_outer_inner_arcs(
        contour_points,
        int(row["points"]["outer_cut_A"]["index"]),
        int(row["points"]["outer_cut_B"]["index"]),
        int(row["points"]["inner_cut_A"]["index"]),
        int(row["points"]["inner_cut_B"]["index"]),
        outer_choice=row.get("outer_path", "auto"),
        inner_choice=row.get("inner_path", "auto"),
    )
    cap = outer_arc if prefix == "outer" else inner_arc
    cap_length = _polyline_length(cap, closed=False)
    contour_length = _polyline_length(_closed_contour_points(contour_points), closed=False)
    ratio = float(cap_length / contour_length) if contour_length > 0 else math.inf
    return cap, float(cap_length), ratio


def _manual_shell_cut_endpoint_warnings(rows: Sequence[Dict], prefix: str) -> List[Dict]:
    warnings_out: List[Dict] = []
    if len(rows) < 2:
        return warnings_out

    for left, right in zip(rows[:-1], rows[1:]):
        gap = int(right["slice_index"]) - int(left["slice_index"])
        if gap > 1:
            warnings_out.append(
                _shell_cut_warning_row(
                    "missing_slice_gap",
                    "warning",
                    f"{prefix}:slice={left['slice_index']}->{right['slice_index']}",
                    f"{prefix} cut annotations skip {gap - 1} slice(s)",
                    "add intermediate shell-cut annotations if this transition is not intentionally sparse",
                )
            )

        left_a = np.asarray(left["points"][f"{prefix}_cut_A"]["point"], dtype=float)
        left_b = np.asarray(left["points"][f"{prefix}_cut_B"]["point"], dtype=float)
        right_a = np.asarray(right["points"][f"{prefix}_cut_A"]["point"], dtype=float)
        right_b = np.asarray(right["points"][f"{prefix}_cut_B"]["point"], dtype=float)
        same = float(np.linalg.norm(left_a - right_a) + np.linalg.norm(left_b - right_b))
        swapped = float(np.linalg.norm(left_a - right_b) + np.linalg.norm(left_b - right_a))
        if same > 2.0 and swapped + 1e-6 < same * 0.65:
            warnings_out.append(
                _shell_cut_warning_row(
                    "endpoint_swap_warning",
                    "warning",
                    f"{prefix}:slice={left['slice_index']}->{right['slice_index']}",
                    f"{prefix} A/B endpoints look closer if swapped between these slices",
                    "check whether A and B were clicked in opposite order on one slice",
                )
            )

    for suffix in ("A", "B"):
        distances: List[Tuple[int, int, float, float]] = []
        for left, right in zip(rows[:-1], rows[1:]):
            gap = max(1, int(right["slice_index"]) - int(left["slice_index"]))
            left_point = np.asarray(left["points"][f"{prefix}_cut_{suffix}"]["point"], dtype=float)
            right_point = np.asarray(right["points"][f"{prefix}_cut_{suffix}"]["point"], dtype=float)
            distance = float(np.linalg.norm(left_point - right_point))
            distances.append((int(left["slice_index"]), int(right["slice_index"]), distance, distance / gap))

        per_slice = np.asarray([item[3] for item in distances if item[3] > 1e-6], dtype=float)
        if len(per_slice) == 0:
            continue
        typical = float(np.median(per_slice))
        threshold = max(12.0, typical * 3.0)
        for left_slice, right_slice, distance, normalized in distances:
            if normalized > threshold:
                warnings_out.append(
                    _shell_cut_warning_row(
                        f"endpoint_{suffix}_jump",
                        "warning",
                        f"{prefix}:slice={left_slice}->{right_slice}",
                        (
                            f"{prefix}_cut_{suffix} jumps {distance:.3f} voxels "
                            f"({normalized:.3f} per slice)"
                        ),
                        "check whether this endpoint should be re-clicked or an intermediate slice should be added",
                    )
                )
    return warnings_out


def _manual_shell_cut_curve(
    rows: Sequence[Dict],
    prefix: str,
    curve_id: str,
    label_left: str,
    label_right: str,
) -> Tuple[SurfaceCutCurve, List[Dict]]:
    warnings_out = _manual_shell_cut_endpoint_warnings(rows, prefix)
    first = rows[0]
    last = rows[-1]
    a_name = f"{prefix}_cut_A"
    b_name = f"{prefix}_cut_B"

    first_cap, _first_cap_length, first_ratio = _manual_shell_cut_cap_arc(first, prefix)
    last_cap, _last_cap_length, last_ratio = _manual_shell_cut_cap_arc(last, prefix)
    for cap_name, row, ratio in (
        ("first_slice_cap_arc", first, first_ratio),
        ("last_slice_cap_arc", last, last_ratio),
    ):
        if ratio > 0.45:
            warnings_out.append(
                _shell_cut_warning_row(
                    "cap_arc_too_long",
                    "warning",
                    f"{curve_id}:{cap_name}:slice={row['slice_index']}",
                    f"{cap_name} uses {ratio:.3f} of the contour perimeter",
                    "check whether the two cut endpoints are too far apart or on the wrong contour side",
                )
            )

    a_track = np.asarray([row["points"][a_name]["point"] for row in rows], dtype=float)
    b_track = np.asarray([row["points"][b_name]["point"] for row in rows], dtype=float)
    raw_loop = np.vstack([a_track, last_cap, b_track[::-1], first_cap[::-1]])
    if len(raw_loop) > 1:
        closure_error = float(np.linalg.norm(raw_loop[0] - raw_loop[-1]))
        if closure_error > 1e-6:
            warnings_out.append(
                _shell_cut_warning_row(
                    "cut_curve_not_closed",
                    "error",
                    curve_id,
                    f"manual {curve_id} loop has closure error {closure_error:.6f}",
                    "re-save the shell-cut annotation; the first and last cap should meet the A/B tracks",
                )
            )

    return (
        SurfaceCutCurve(
            curve_id=curve_id,
            label_left=label_left,
            label_right=label_right,
            control_points=_closed_polyline(raw_loop),
            source="manual_2d",
        ),
        warnings_out,
    )


def _manual_shell_cut_annotations_to_input(
    data: Dict,
    contours: Optional[Sequence[Contour2D]],
) -> Tuple[List[SurfaceCutCurve], List[SurfacePatchSeed], List[Dict]]:
    if contours is None:
        raise ValueError("Manual shell-cut annotations need contours from the current mask")

    raw_rows = data.get("rows", [])
    if not raw_rows:
        raise ValueError("shell_cut_annotations.json has no rows")

    contour_lookup = _manual_shell_cut_contour_lookup(contours)
    parsed_rows: List[Dict] = []
    warnings_out: List[Dict] = []
    seen_slices: set[int] = set()
    for row_index, raw_row in enumerate(raw_rows):
        row, row_warnings = _parse_manual_shell_cut_row(raw_row, row_index, contour_lookup)
        if int(row["slice_index"]) in seen_slices:
            raise ValueError(f"Duplicate shell-cut annotation for slice {row['slice_index']}")
        seen_slices.add(int(row["slice_index"]))
        parsed_rows.append(row)
        warnings_out.extend(row_warnings)

    parsed_rows = sorted(parsed_rows, key=lambda item: int(item["slice_index"]))
    cut_rows = [
        row
        for row in parsed_rows
        if normalize_surface_mode(row.get("surface_mode")) == SURFACE_MODE_NORMAL
    ]
    for row in parsed_rows:
        surface_mode = normalize_surface_mode(row.get("surface_mode"))
        if surface_mode == SURFACE_MODE_NORMAL:
            continue
        warnings_out.append(
            _shell_cut_warning_row(
                "single_surface_shell_cut_row",
                "warning",
                f"slice={row['slice_index']},contour={row['contour_id']}",
                (
                    f"manual shell-cut row is marked {surface_mode}; it is used in the "
                    "derived boundary CSV but not as a four-point cut-curve slice"
                ),
                "use four shell-cut endpoints on nearby transition slices if the cut curve needs a tighter boundary",
            )
        )
    if len(cut_rows) < 2:
        raise ValueError("Manual shell-cut annotations need at least two four-point cut-curve slices")

    outer, outer_warnings = _manual_shell_cut_curve(
        cut_rows,
        "outer",
        "outer_lateral_boundary",
        "outer",
        "lateral",
    )
    inner, inner_warnings = _manual_shell_cut_curve(
        cut_rows,
        "inner",
        "inner_lateral_boundary",
        "inner",
        "lateral",
    )
    warnings_out.extend(outer_warnings)
    warnings_out.extend(inner_warnings)

    return [outer, inner], [], warnings_out


def read_shell_cut_json_with_qc(
    path: str | Path,
    contours: Optional[Sequence[Contour2D]] = None,
) -> Tuple[List[SurfaceCutCurve], List[SurfacePatchSeed], List[Dict]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if _is_manual_shell_cut_annotation_data(data):
        return _manual_shell_cut_annotations_to_input(data, contours)

    curves, seeds = _read_explicit_shell_cut_json(data)
    return curves, seeds, []


def read_shell_cut_json(path: str | Path) -> Tuple[List[SurfaceCutCurve], List[SurfacePatchSeed]]:
    curves, seeds, _warnings = read_shell_cut_json_with_qc(path)
    return curves, seeds


def read_manual_shell_cut_annotations_json(
    path: str | Path,
    contours: Sequence[Contour2D],
) -> Tuple[List[SurfaceCutCurve], List[SurfacePatchSeed], List[Dict]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not _is_manual_shell_cut_annotation_data(data):
        raise ValueError(f"{path} is not a manual shell-cut annotation JSON")
    return _manual_shell_cut_annotations_to_input(data, contours)



def _shell_vertices_scaled(shell: ShellMesh) -> np.ndarray:
    return np.asarray(shell.vertices, dtype=float) * np.asarray(shell.spacing, dtype=float).reshape(1, 3)


def _edge_key(left: int, right: int) -> Tuple[int, int]:
    left = int(left)
    right = int(right)
    return (left, right) if left < right else (right, left)


def _mesh_edge_pairs(faces: np.ndarray) -> Iterable[Tuple[int, int]]:
    for a, b, c in np.asarray(faces, dtype=np.int64):
        yield _edge_key(int(a), int(b))
        yield _edge_key(int(b), int(c))
        yield _edge_key(int(c), int(a))


def _build_shell_edge_graph(shell: ShellMesh) -> Tuple[Dict[int, List[Tuple[int, float]]], set[Tuple[int, int]]]:
    vertices_scaled = _shell_vertices_scaled(shell)
    adjacency: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    edges = set(_mesh_edge_pairs(shell.faces))
    for left, right in edges:
        weight = float(np.linalg.norm(vertices_scaled[left] - vertices_scaled[right]))
        adjacency[left].append((right, weight))
        adjacency[right].append((left, weight))
    return adjacency, edges


def _shortest_shell_path(
    adjacency: Dict[int, List[Tuple[int, float]]],
    vertices_scaled: np.ndarray,
    start: int,
    goal: int,
    max_visited: int = 200_000,
) -> Optional[List[int]]:
    start = int(start)
    goal = int(goal)
    if start == goal:
        return [start]

    def heuristic(node: int) -> float:
        return float(np.linalg.norm(vertices_scaled[node] - vertices_scaled[goal]))

    heap: List[Tuple[float, float, int]] = [(heuristic(start), 0.0, start)]
    best_cost = {start: 0.0}
    parent: Dict[int, int] = {}
    visited: set[int] = set()

    while heap and len(visited) < max_visited:
        _estimate, cost, node = heapq.heappop(heap)
        if node in visited:
            continue
        if node == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return path
        visited.add(node)
        for neighbor, weight in adjacency.get(node, []):
            if neighbor in visited:
                continue
            next_cost = cost + float(weight)
            if next_cost >= best_cost.get(neighbor, math.inf):
                continue
            best_cost[neighbor] = next_cost
            parent[neighbor] = node
            heapq.heappush(heap, (next_cost + heuristic(neighbor), next_cost, neighbor))
    return None


def snap_cut_curve_to_shell(curve: SurfaceCutCurve, shell: ShellMesh) -> SurfaceCutCurve:
    control_points = _closed_polyline(curve.control_points)
    vertices_scaled = _shell_vertices_scaled(shell)
    tree = cKDTree(vertices_scaled)
    distances, vertex_ids = tree.query(_scaled_points(control_points, shell.spacing), k=1)
    curve.control_points = control_points
    curve.snapped_vertices = np.asarray(vertex_ids, dtype=np.int64)
    curve.snapped_points = np.asarray(shell.vertices[curve.snapped_vertices], dtype=float)
    curve.mean_snap_distance = float(np.mean(distances)) if len(distances) else math.nan
    curve.max_snap_distance = float(np.max(distances)) if len(distances) else math.nan
    curve.closure_error = (
        float(np.linalg.norm((control_points[0] - control_points[-1]) * shell.spacing))
        if len(control_points) > 1
        else math.inf
    )
    curve.is_closed = bool(curve.closure_error <= max(float(np.max(shell.spacing)), 1e-6))
    curve.control_polyline_length = float(_polyline_length(_scaled_points(control_points, shell.spacing)))
    return curve


def trace_cut_curve_on_shell(
    curve: SurfaceCutCurve,
    shell: ShellMesh,
    adjacency: Dict[int, List[Tuple[int, float]]],
    edge_set: set[Tuple[int, int]],
) -> SurfaceCutCurve:
    vertex_ids: List[int] = []
    for vertex_id in np.asarray(curve.snapped_vertices, dtype=np.int64):
        value = int(vertex_id)
        if not vertex_ids or vertex_ids[-1] != value:
            vertex_ids.append(value)
    if len(vertex_ids) > 1 and vertex_ids[0] == vertex_ids[-1]:
        vertex_ids.pop()
    if len(vertex_ids) < 2:
        curve.status = "error"
        curve.reason = "cut curve collapsed to fewer than two shell vertices"
        return curve

    vertices_scaled = _shell_vertices_scaled(shell)
    full_path: List[int] = []
    for left, right in zip(vertex_ids, vertex_ids[1:] + [vertex_ids[0]]):
        if _edge_key(left, right) in edge_set:
            path = [left, right]
        else:
            path = _shortest_shell_path(adjacency, vertices_scaled, left, right)
        if path is None:
            curve.status = "error"
            curve.reason = f"no shell path between snapped vertices {left} and {right}"
            return curve
        if full_path and full_path[-1] == path[0]:
            full_path.extend(path[1:])
        else:
            full_path.extend(path)

    if full_path and full_path[-1] != full_path[0]:
        full_path.append(full_path[0])

    curve.mesh_vertex_path = np.asarray(full_path, dtype=np.int64)
    curve.cut_edges = [
        _edge_key(left, right)
        for left, right in zip(curve.mesh_vertex_path[:-1], curve.mesh_vertex_path[1:])
        if int(left) != int(right)
    ]
    curve.path_length = float(
        _polyline_length(vertices_scaled[np.asarray(curve.mesh_vertex_path, dtype=np.int64)])
    )
    if curve.control_polyline_length > 0:
        curve.path_length_ratio = float(curve.path_length / curve.control_polyline_length)
    else:
        curve.path_length_ratio = math.inf

    max_snap = float(np.max(shell.spacing)) * 2.0
    if not curve.cut_edges:
        curve.status = "error"
        curve.reason = "cut curve produced no blocking edges"
    elif curve.max_snap_distance > max_snap:
        curve.status = "warning"
        curve.reason = "cut curve is far from the shell"
    elif curve.path_length_ratio > 4.0:
        curve.status = "warning"
        curve.reason = "traced shell path is much longer than the input curve"
    else:
        curve.status = "ok"
        curve.reason = ""
    return curve


def _build_face_adjacency(
    faces: np.ndarray,
) -> Tuple[List[List[int]], Dict[Tuple[int, int], Tuple[int, int]], Dict[Tuple[int, int], List[int]]]:
    edge_to_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for face_id, face in enumerate(np.asarray(faces, dtype=np.int64)):
        a, b, c = [int(value) for value in face]
        for edge in (_edge_key(a, b), _edge_key(b, c), _edge_key(c, a)):
            edge_to_faces[edge].append(int(face_id))

    adjacency: List[List[int]] = [[] for _ in range(len(faces))]
    shared_edges: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for edge, face_ids in edge_to_faces.items():
        if len(face_ids) < 2:
            continue
        for left_index, left in enumerate(face_ids):
            for right in face_ids[left_index + 1 :]:
                adjacency[left].append(right)
                adjacency[right].append(left)
                shared_edges[(left, right)] = edge
                shared_edges[(right, left)] = edge
    return adjacency, shared_edges, edge_to_faces


def _flood_fill_faces(
    seed_face: int,
    adjacency: Sequence[Sequence[int]],
    shared_edges: Dict[Tuple[int, int], Tuple[int, int]],
    blocked_edges: set[Tuple[int, int]],
) -> np.ndarray:
    seed_face = int(seed_face)
    visited = np.zeros(len(adjacency), dtype=bool)
    queue: deque[int] = deque([seed_face])
    visited[seed_face] = True
    while queue:
        face = queue.popleft()
        for neighbor in adjacency[face]:
            edge = shared_edges.get((face, int(neighbor)))
            if edge in blocked_edges or visited[int(neighbor)]:
                continue
            visited[int(neighbor)] = True
            queue.append(int(neighbor))
    return np.nonzero(visited)[0].astype(np.int64)


def _face_components_after_cut(
    adjacency: Sequence[Sequence[int]],
    shared_edges: Dict[Tuple[int, int], Tuple[int, int]],
    blocked_edges: set[Tuple[int, int]],
) -> List[np.ndarray]:
    visited = np.zeros(len(adjacency), dtype=bool)
    components: List[np.ndarray] = []
    for face_id in range(len(adjacency)):
        if visited[face_id]:
            continue
        component = _flood_fill_faces(face_id, adjacency, shared_edges, blocked_edges)
        visited[component] = True
        components.append(component)
    return components


def _vertex_to_faces(faces: np.ndarray, vertex_count: int) -> List[List[int]]:
    mapping: List[List[int]] = [[] for _ in range(vertex_count)]
    for face_id, face in enumerate(np.asarray(faces, dtype=np.int64)):
        for vertex_id in face:
            mapping[int(vertex_id)].append(int(face_id))
    return mapping


def snap_seed_to_shell_face(
    seed: SurfacePatchSeed,
    shell: ShellMesh,
    vertex_faces: Sequence[Sequence[int]],
) -> SurfacePatchSeed:
    vertices_scaled = _shell_vertices_scaled(shell)
    seed_scaled = np.asarray(seed.seed_point, dtype=float) * shell.spacing
    vertex_tree = cKDTree(vertices_scaled)
    distance, vertex_id = vertex_tree.query(seed_scaled, k=1)
    candidate_faces = list(vertex_faces[int(vertex_id)])
    if not candidate_faces:
        seed.status = "error"
        seed.reason = "nearest shell vertex has no faces"
        seed.snap_distance = float(distance)
        return seed

    centers = vertices_scaled[shell.faces[np.asarray(candidate_faces, dtype=np.int64)]].mean(axis=1)
    center_distances = np.linalg.norm(centers - seed_scaled.reshape(1, 3), axis=1)
    best_index = int(np.argmin(center_distances))
    seed.snapped_face = int(candidate_faces[best_index])
    seed.snap_distance = float(center_distances[best_index])
    if seed.snap_distance > float(np.max(shell.spacing)) * 3.0:
        seed.status = "warning"
        seed.reason = "seed is far from the shell"
    else:
        seed.status = "ok"
        seed.reason = ""
    return seed


def _shell_triangle_area(vertices: np.ndarray) -> float:
    if len(vertices) != 3:
        return 0.0
    return float(np.linalg.norm(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])) * 0.5)


def _mesh_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    if len(faces) == 0:
        return 0.0
    return float(sum(_shell_triangle_area(vertices[np.asarray(face, dtype=np.int64)]) for face in faces))


def _boundary_edges_for_faces(faces: np.ndarray) -> List[Tuple[int, int]]:
    counts: Counter[Tuple[int, int]] = Counter(_mesh_edge_pairs(faces))
    return [edge for edge, count in counts.items() if count == 1]


def _edge_length(vertices: np.ndarray, edge: Tuple[int, int]) -> float:
    return float(np.linalg.norm(vertices[int(edge[0])] - vertices[int(edge[1])]))


def _extract_shell_submesh(name: str, shell: ShellMesh, face_indices: np.ndarray) -> SurfaceMesh:
    face_indices = np.asarray(face_indices, dtype=np.int64)
    if len(face_indices) == 0:
        return SurfaceMesh(name=name, vertices=np.empty((0, 3), dtype=float), faces=np.empty((0, 3), dtype=np.int32))
    return _compact_surface_mesh(name, shell.vertices, shell.faces[face_indices])


def _surface_cut_curve_row(curve: SurfaceCutCurve) -> Dict:
    repeated = 0
    if len(curve.mesh_vertex_path) > 0:
        counts = Counter(int(value) for value in curve.mesh_vertex_path[:-1])
        repeated = sum(1 for count in counts.values() if count > 1)
    return {
        "curve_id": curve.curve_id,
        "label_left": curve.label_left,
        "label_right": curve.label_right,
        "source": curve.source,
        "confidence": float(curve.confidence),
        "num_control_points": int(len(curve.control_points)),
        "num_snapped_vertices": int(len(curve.snapped_vertices)),
        "num_path_vertices": int(len(curve.mesh_vertex_path)),
        "num_cut_edges": int(len(curve.cut_edges)),
        "is_closed": bool(curve.is_closed),
        "closure_error": float(curve.closure_error),
        "mean_snap_distance": float(curve.mean_snap_distance),
        "max_snap_distance": float(curve.max_snap_distance),
        "path_length": float(curve.path_length),
        "control_polyline_length": float(curve.control_polyline_length),
        "path_length_ratio": float(curve.path_length_ratio),
        "revisited_vertex_count": int(repeated),
        "status": curve.status,
        "reason": curve.reason,
    }


def _surface_patch_seed_row(seed: SurfacePatchSeed) -> Dict:
    point = np.asarray(seed.seed_point, dtype=float)
    return {
        "patch_label": seed.patch_label,
        "source": seed.source,
        "seed_x": float(point[0]),
        "seed_y": float(point[1]),
        "seed_z": float(point[2]),
        "snapped_face": "" if seed.snapped_face is None else int(seed.snapped_face),
        "snap_distance": "" if seed.snap_distance is None else float(seed.snap_distance),
        "status": seed.status,
        "reason": seed.reason,
    }


def _shell_summary_row(
    shell: ShellMesh,
    mask: np.ndarray,
    edge_to_faces: Dict[Tuple[int, int], List[int]],
) -> Dict:
    boundary_edges = sum(1 for face_ids in edge_to_faces.values() if len(face_ids) == 1)
    nonmanifold_edges = sum(1 for face_ids in edge_to_faces.values() if len(face_ids) > 2)
    structure = ndimage.generate_binary_structure(3, 3)
    _components, component_count = ndimage.label(_as_bool_mask(mask), structure=structure)
    return {
        "backend": shell.backend,
        "coordinate_space": shell.coordinate_space,
        "num_vertices": int(len(shell.vertices)),
        "num_faces": int(len(shell.faces)),
        "num_edges": int(len(edge_to_faces)),
        "num_boundary_edges": int(boundary_edges),
        "num_nonmanifold_edges": int(nonmanifold_edges),
        "mask_shape": "x".join(str(int(value)) for value in mask.shape),
        "foreground_voxel_count": int(np.count_nonzero(mask)),
        "mask_component_count": int(component_count),
        "spacing_x": float(shell.spacing[0]),
        "spacing_y": float(shell.spacing[1]),
        "spacing_z": float(shell.spacing[2]),
    }


def _patch_summary_row(name: str, shell: ShellMesh, face_indices: np.ndarray) -> Dict:
    face_indices = np.asarray(face_indices, dtype=np.int64)
    faces = shell.faces[face_indices] if len(face_indices) else np.empty((0, 3), dtype=np.int32)
    boundary_edges = _boundary_edges_for_faces(faces) if len(faces) else []
    vertices_scaled = _shell_vertices_scaled(shell)
    area = _mesh_area(vertices_scaled, faces)
    boundary_length = sum(_edge_length(vertices_scaled, edge) for edge in boundary_edges)
    return {
        "patch_id": name,
        "label": name,
        "num_faces": int(len(faces)),
        "num_vertices": int(len(np.unique(faces.reshape(-1)))) if len(faces) else 0,
        "area": float(area),
        "boundary_length": float(boundary_length),
        "num_boundary_edges": int(len(boundary_edges)),
        "status": "ok" if len(faces) else "error",
        "reason": "" if len(faces) else "patch is empty",
    }


def _face_edges(face: Sequence[int]) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    a, b, c = [int(value) for value in face]
    return (_edge_key(a, b), _edge_key(b, c), _edge_key(c, a))


def _cut_curve_boundary_labels(curve: SurfaceCutCurve) -> set[str]:
    labels: set[str] = set()
    for value in (curve.curve_id, curve.label_left, curve.label_right):
        text = str(value).lower()
        if "outer" in text:
            labels.add("outer")
        if "inner" in text:
            labels.add("inner")
    return labels


def _cut_edge_label_lookup(
    curves: Sequence[SurfaceCutCurve],
) -> Tuple[Dict[Tuple[int, int], set[str]], Dict[Tuple[int, int], set[str]]]:
    edge_labels: Dict[Tuple[int, int], set[str]] = defaultdict(set)
    edge_curve_ids: Dict[Tuple[int, int], set[str]] = defaultdict(set)
    for curve in curves:
        labels = _cut_curve_boundary_labels(curve)
        if not labels:
            continue
        for edge in curve.cut_edges:
            key = _edge_key(edge[0], edge[1])
            edge_labels[key].update(labels)
            edge_curve_ids[key].add(str(curve.curve_id))
    return edge_labels, edge_curve_ids


def _component_cut_touch_info(
    component: np.ndarray,
    shell: ShellMesh,
    edge_labels: Dict[Tuple[int, int], set[str]],
    edge_curve_ids: Dict[Tuple[int, int], set[str]],
) -> Tuple[set[str], set[str]]:
    labels: set[str] = set()
    curve_ids: set[str] = set()
    for face_id in np.asarray(component, dtype=np.int64):
        for edge in _face_edges(shell.faces[int(face_id)]):
            labels.update(edge_labels.get(edge, set()))
            curve_ids.update(edge_curve_ids.get(edge, set()))
    return labels, curve_ids


def _assigned_label_from_cut_touches(labels: set[str]) -> Tuple[str, str]:
    touches_outer = "outer" in labels
    touches_inner = "inner" in labels
    if touches_outer and touches_inner:
        return "lateral", "touches both outer and inner cut curves"
    if touches_outer:
        return "outer", "touches only the outer cut curve"
    if touches_inner:
        return "inner", "touches only the inner cut curve"
    return "lateral", "does not touch a cut curve; kept with lateral as unclaimed shell"


def _label_cut_components(
    components: Sequence[np.ndarray],
    shell: ShellMesh,
    edge_labels: Dict[Tuple[int, int], set[str]],
    edge_curve_ids: Dict[Tuple[int, int], set[str]],
) -> Tuple[Dict[str, np.ndarray], List[Dict]]:
    faces_by_label: Dict[str, List[np.ndarray]] = {
        "outer": [],
        "inner": [],
        "lateral": [],
    }
    rows: List[Dict] = []
    vertices_scaled = _shell_vertices_scaled(shell)
    for component_id, component in enumerate(components):
        component = np.asarray(component, dtype=np.int64)
        labels, curve_ids = _component_cut_touch_info(component, shell, edge_labels, edge_curve_ids)
        assigned_label, reason = _assigned_label_from_cut_touches(labels)
        faces_by_label[assigned_label].append(component)
        rows.append(
            {
                "component_id": int(component_id),
                "num_faces": int(len(component)),
                "area": float(_mesh_area(vertices_scaled, shell.faces[component])),
                "touches_outer_cut": bool("outer" in labels),
                "touches_inner_cut": bool("inner" in labels),
                "cut_curves": ";".join(sorted(curve_ids)),
                "assigned_label": assigned_label,
                "assignment_reason": reason,
            }
        )

    combined: Dict[str, np.ndarray] = {}
    for label, component_faces in faces_by_label.items():
        if component_faces:
            combined[label] = np.concatenate(component_faces).astype(np.int64)
        else:
            combined[label] = np.empty(0, dtype=np.int64)
    return combined, rows


def _shell_cut_warning_rows(
    curves: Sequence[SurfaceCutCurve],
    seeds: Sequence[SurfacePatchSeed],
    patch_rows: Sequence[Dict],
    component_rows: Optional[Sequence[Dict]] = None,
    shell_summary: Optional[Dict] = None,
    overlap_count: int = 0,
    minimum_component_count: int = 3,
    warn_unclaimed_components: bool = True,
) -> List[Dict]:
    rows: List[Dict] = []
    backend = str(shell_summary.get("backend", "")) if shell_summary is not None else ""
    for curve in curves:
        if curve.status != "ok":
            action = "review or redraw the cut curve"
            if backend == SHELL_BACKEND_MARCHING_CUBES:
                action += "; if the curve looks correct in 3D, retry voxel backend as a baseline"
            rows.append(
                {
                    "warning_type": "cut_curve",
                    "severity": "error" if curve.status == "error" else "warning",
                    "object_id": curve.curve_id,
                    "message": curve.reason,
                    "suggested_action": action,
                }
            )
    for seed in seeds:
        if seed.status != "ok":
            rows.append(
                {
                    "warning_type": "seed",
                    "severity": "error" if seed.status == "error" else "warning",
                    "object_id": seed.patch_label,
                    "message": seed.reason,
                    "suggested_action": "move the explicit seed point onto the intended patch or remove the legacy seed override",
                }
            )
    for patch in patch_rows:
        if patch.get("status") != "ok":
            rows.append(
                {
                    "warning_type": "patch",
                    "severity": "error",
                    "object_id": patch.get("patch_id", ""),
                    "message": patch.get("reason", ""),
                    "suggested_action": "check whether the cut curves separate the shell",
                }
            )
    component_rows = list(component_rows or [])
    if len(component_rows) < int(minimum_component_count):
        rows.append(
            {
                "warning_type": "cut_curve_does_not_separate_shell",
                "severity": "error",
                "object_id": "shell",
                "message": f"cut curves split the shell into only {len(component_rows)} component(s)",
                "suggested_action": "check whether the closed cut curve really separates the shell",
            }
        )
    if warn_unclaimed_components:
        for component in component_rows:
            touches_outer = bool(component.get("touches_outer_cut", False))
            touches_inner = bool(component.get("touches_inner_cut", False))
            assigned_label = str(component.get("assigned_label", ""))
            if not touches_outer and not touches_inner:
                rows.append(
                    {
                        "warning_type": "unclaimed_cut_component",
                        "severity": "warning",
                        "object_id": f"component={component.get('component_id', '')}",
                        "message": (
                            f"component does not touch any cut curve and was assigned to {assigned_label}"
                        ),
                        "suggested_action": "inspect shell_with_cut_curves.obj; this may be a disconnected mask fragment",
                    }
                )
    if shell_summary is not None and int(shell_summary.get("num_nonmanifold_edges", 0) or 0) > 0:
        rows.append(
            {
                "warning_type": "nonmanifold_edges",
                "severity": "warning",
                "object_id": str(shell_summary.get("backend", "shell")),
                "message": f"shell has {shell_summary.get('num_nonmanifold_edges')} nonmanifold edge(s)",
                "suggested_action": "inspect shell_with_cut_curves.obj; if flood fill is unstable, retry voxel backend",
            }
        )
    if overlap_count > 0:
        rows.append(
            {
                "warning_type": "outer_inner_overlap",
                "severity": "error",
                "object_id": "outer;inner",
                "message": f"outer and inner flood fills overlap on {overlap_count} faces",
                "suggested_action": "check cut curves and component assignment",
            }
        )
    return rows


def _shell_cut_debug_color(name: str) -> Tuple[int, int, int]:
    normalized = str(name).lower()
    if "outer" in normalized:
        return (230, 70, 55)
    if "inner" in normalized:
        return (55, 115, 230)
    if "lateral" in normalized:
        return (230, 180, 55)
    return (245, 245, 245)


def _append_debug_box(
    vertices: List[np.ndarray],
    faces: List[List[int]],
    colors: List[Tuple[int, int, int]],
    center: np.ndarray,
    radius: float,
    color: Tuple[int, int, int],
) -> None:
    center = np.asarray(center, dtype=float)
    radius = float(radius)
    offsets = np.asarray(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )
    start = len(vertices)
    for offset in offsets:
        vertices.append(center + offset * radius)
        colors.append(color)
    faces.extend(
        [
            [start + 0, start + 1, start + 2],
            [start + 0, start + 2, start + 3],
            [start + 4, start + 6, start + 5],
            [start + 4, start + 7, start + 6],
            [start + 0, start + 4, start + 5],
            [start + 0, start + 5, start + 1],
            [start + 1, start + 5, start + 6],
            [start + 1, start + 6, start + 2],
            [start + 2, start + 6, start + 7],
            [start + 2, start + 7, start + 3],
            [start + 3, start + 7, start + 4],
            [start + 3, start + 4, start + 0],
        ]
    )


def _debug_box_points_mesh(
    point_groups: Sequence[Tuple[np.ndarray, Tuple[int, int, int], str]],
    radius: float,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Tuple[int, int, int]], np.ndarray]:
    vertices: List[np.ndarray] = []
    faces: List[List[int]] = []
    colors: List[Tuple[int, int, int]] = []
    face_materials: List[str] = []
    materials: Dict[str, Tuple[int, int, int]] = {}
    for points, color, material_name in point_groups:
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        materials[material_name] = color
        for point in points:
            face_start = len(faces)
            _append_debug_box(vertices, faces, colors, point, radius, color)
            face_materials.extend([material_name] * (len(faces) - face_start))
    return (
        np.asarray(vertices, dtype=float).reshape(-1, 3),
        np.asarray(faces, dtype=np.int32).reshape(-1, 3),
        face_materials,
        materials,
        np.asarray(colors, dtype=np.uint8).reshape(-1, 3),
    )


def _write_debug_box_points(
    path: str | Path,
    point_groups: Sequence[Tuple[np.ndarray, Tuple[int, int, int], str]],
    radius: float,
) -> None:
    vertices, faces, face_materials, materials, colors = _debug_box_points_mesh(
        point_groups,
        radius,
    )
    _write_colored_mesh_ply(
        path,
        vertices,
        faces,
        vertex_colors=colors,
    )
    _write_colored_obj(
        Path(path).with_suffix(".obj"),
        Path(path).stem,
        vertices,
        faces,
        face_materials=face_materials,
        materials=materials,
    )


def _write_shell_cut_curve_debug_ply(
    path: str | Path,
    shell: ShellMesh,
    curves: Sequence[SurfaceCutCurve],
    source: str,
) -> None:
    point_groups: List[Tuple[np.ndarray, Tuple[int, int, int], str]] = []
    extent = float(np.max(np.ptp(shell.vertices, axis=0))) if len(shell.vertices) else 1.0
    radius = max(0.35, extent * 0.004)

    for curve in curves:
        if source == "control":
            points = np.asarray(curve.control_points, dtype=float).reshape(-1, 3)
        else:
            if len(curve.mesh_vertex_path) == 0:
                continue
            points = np.asarray(shell.vertices[curve.mesh_vertex_path], dtype=float).reshape(-1, 3)
        if len(points) < 2:
            continue
        color = _shell_cut_debug_color(curve.label_left)
        point_groups.append((points, color, curve.label_left))

    _write_debug_box_points(path, point_groups, radius)


def _write_shell_cut_seed_debug_ply(
    path: str | Path,
    shell: ShellMesh,
    seeds: Sequence[SurfacePatchSeed],
) -> None:
    extent = float(np.max(np.ptp(shell.vertices, axis=0))) if len(shell.vertices) else 1.0
    marker_radius = max(1.0, extent * 0.025)
    point_groups: List[Tuple[np.ndarray, Tuple[int, int, int], str]] = []

    for seed in seeds:
        if seed.snapped_face is not None and 0 <= int(seed.snapped_face) < len(shell.faces):
            center = np.mean(shell.vertices[shell.faces[int(seed.snapped_face)]], axis=0)
        else:
            center = np.asarray(seed.seed_point, dtype=float)
        color = _shell_cut_debug_color(seed.patch_label)
        point_groups.append((center.reshape(1, 3), color, seed.patch_label))

    _write_debug_box_points(path, point_groups, marker_radius)


def _shell_cut_curve_point_groups(
    shell: ShellMesh,
    curves: Sequence[SurfaceCutCurve],
) -> List[Tuple[np.ndarray, Tuple[int, int, int], str]]:
    point_groups: List[Tuple[np.ndarray, Tuple[int, int, int], str]] = []
    for curve in curves:
        if len(curve.mesh_vertex_path) == 0:
            continue
        points = np.asarray(shell.vertices[curve.mesh_vertex_path], dtype=float).reshape(-1, 3)
        if len(points) < 2:
            continue
        point_groups.append((points, _shell_cut_debug_color(curve.label_left), curve.label_left))
    return point_groups


def _shell_cut_seed_point_groups(
    shell: ShellMesh,
    seeds: Sequence[SurfacePatchSeed],
) -> List[Tuple[np.ndarray, Tuple[int, int, int], str]]:
    point_groups: List[Tuple[np.ndarray, Tuple[int, int, int], str]] = []
    for seed in seeds:
        if seed.snapped_face is not None and 0 <= int(seed.snapped_face) < len(shell.faces):
            center = np.mean(shell.vertices[shell.faces[int(seed.snapped_face)]], axis=0)
        else:
            center = np.asarray(seed.seed_point, dtype=float)
        material_name = f"{seed.patch_label}_seed"
        point_groups.append((center.reshape(1, 3), _shell_cut_debug_color(seed.patch_label), material_name))
    return point_groups


def _write_shell_with_cut_curves_obj(
    path: str | Path,
    shell: ShellMesh,
    curves: Sequence[SurfaceCutCurve],
    seeds: Sequence[SurfacePatchSeed],
) -> None:
    extent = float(np.max(np.ptp(shell.vertices, axis=0))) if len(shell.vertices) else 1.0
    curve_radius = max(0.45, extent * 0.006)
    seed_radius = max(1.2, extent * 0.03)

    curve_vertices, curve_faces, curve_materials, curve_mtls, _curve_colors = _debug_box_points_mesh(
        _shell_cut_curve_point_groups(shell, curves),
        curve_radius,
    )
    seed_vertices, seed_faces, seed_materials, seed_mtls, _seed_colors = _debug_box_points_mesh(
        _shell_cut_seed_point_groups(shell, seeds),
        seed_radius,
    )

    vertices = [np.asarray(shell.vertices, dtype=float)]
    faces = [np.asarray(shell.faces, dtype=np.int32)]
    face_materials: List[str] = ["shell"] * len(shell.faces)
    materials: Dict[str, Tuple[int, int, int]] = {"shell": (200, 200, 200)}
    materials.update(curve_mtls)
    materials.update(seed_mtls)

    offset = len(shell.vertices)
    if len(curve_vertices) > 0:
        vertices.append(curve_vertices)
        faces.append(curve_faces + offset)
        face_materials.extend(curve_materials)
        offset += len(curve_vertices)
    if len(seed_vertices) > 0:
        vertices.append(seed_vertices)
        faces.append(seed_faces + offset)
        face_materials.extend(seed_materials)

    _write_colored_obj(
        path,
        "shell_with_cut_curves",
        np.vstack(vertices),
        np.vstack(faces),
        face_materials=face_materials,
        materials=materials,
    )


def _write_shell_cut_debug_geometry(
    output_dir: str | Path,
    shell: ShellMesh,
    curves: Sequence[SurfaceCutCurve],
    seeds: Sequence[SurfacePatchSeed],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_shell = SurfaceMesh("full_shell", shell.vertices, shell.faces)
    write_ply(output_dir / "full_shell.ply", full_shell)
    write_obj(output_dir / "full_shell.obj", full_shell)
    _write_shell_cut_curve_debug_ply(
        output_dir / "cut_curves_on_shell.ply",
        shell,
        curves,
        source="snapped",
    )
    _write_shell_cut_curve_debug_ply(
        output_dir / "cut_curve_control_points.ply",
        shell,
        curves,
        source="control",
    )
    if seeds:
        _write_shell_cut_seed_debug_ply(output_dir / "patch_seed_markers.ply", shell, seeds)
    _write_shell_with_cut_curves_obj(
        output_dir / "shell_with_cut_curves.obj",
        shell,
        curves,
        seeds,
    )


def _write_shell_cut_qc_tables(qc: Dict[str, List[Dict]], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_rows(
        output_dir / "shell_summary.csv",
        qc.get("shell_summary", []),
        fieldnames=[
            "backend",
            "coordinate_space",
            "num_vertices",
            "num_faces",
            "num_edges",
            "num_boundary_edges",
            "num_nonmanifold_edges",
            "mask_shape",
            "foreground_voxel_count",
            "mask_component_count",
            "spacing_x",
            "spacing_y",
            "spacing_z",
        ],
    )
    _write_csv_rows(
        output_dir / "cut_curves.csv",
        qc.get("cut_curves", []),
        fieldnames=[
            "curve_id",
            "label_left",
            "label_right",
            "source",
            "confidence",
            "num_control_points",
            "num_snapped_vertices",
            "num_path_vertices",
            "num_cut_edges",
            "is_closed",
            "closure_error",
            "mean_snap_distance",
            "max_snap_distance",
            "path_length",
            "control_polyline_length",
            "path_length_ratio",
            "revisited_vertex_count",
            "status",
            "reason",
        ],
    )
    seed_rows = qc.get("seeds", [])
    if seed_rows:
        _write_csv_rows(
            output_dir / "seeds.csv",
            seed_rows,
            fieldnames=[
                "patch_label",
                "source",
                "seed_x",
                "seed_y",
                "seed_z",
                "snapped_face",
                "snap_distance",
                "status",
                "reason",
            ],
        )
    _write_csv_rows(
        output_dir / "patch_summary.csv",
        qc.get("patch_summary", []),
        fieldnames=[
            "patch_id",
            "label",
            "num_faces",
            "num_vertices",
            "area",
            "boundary_length",
            "num_boundary_edges",
            "status",
            "reason",
        ],
    )
    _write_csv_rows(
        output_dir / "cut_components.csv",
        qc.get("cut_components", []),
        fieldnames=[
            "component_id",
            "num_faces",
            "area",
            "touches_outer_cut",
            "touches_inner_cut",
            "cut_curves",
            "assigned_label",
            "assignment_reason",
            "selected_patch_labels",
        ],
    )
    _write_csv_rows(
        output_dir / "cut_edges.csv",
        qc.get("cut_edges", []),
        fieldnames=["curve_id", "edge_start", "edge_end"],
    )
    _write_csv_rows(
        output_dir / "shell_cut_warnings.csv",
        qc.get("warnings", []),
        fieldnames=[
            "warning_type",
            "severity",
            "object_id",
            "message",
            "suggested_action",
        ],
    )


def safe_surface_name(name: object, fallback: str = "surface") -> str:
    """Return a readable file-safe surface name."""

    text = str(name or "").strip()
    if not text:
        text = fallback
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or fallback


def _unique_surface_name(name: object, used: set[str], fallback: str = "surface") -> str:
    base = safe_surface_name(name, fallback=fallback)
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def _face_component_lookup(components: Sequence[np.ndarray], face_count: int) -> np.ndarray:
    lookup = np.full(int(face_count), -1, dtype=np.int64)
    for component_id, component in enumerate(components):
        lookup[np.asarray(component, dtype=np.int64)] = int(component_id)
    return lookup


def _named_patch_component_rows(
    components: Sequence[np.ndarray],
    shell: ShellMesh,
    edge_curve_ids: Dict[Tuple[int, int], set[str]],
    labels_by_component: Dict[int, set[str]],
) -> List[Dict]:
    rows: List[Dict] = []
    vertices_scaled = _shell_vertices_scaled(shell)
    for component_id, component in enumerate(components):
        component = np.asarray(component, dtype=np.int64)
        curve_ids: set[str] = set()
        for face_id in component:
            for edge in _face_edges(shell.faces[int(face_id)]):
                curve_ids.update(edge_curve_ids.get(edge, set()))
        labels = sorted(labels_by_component.get(int(component_id), set()))
        rows.append(
            {
                "component_id": int(component_id),
                "num_faces": int(len(component)),
                "area": float(_mesh_area(vertices_scaled, shell.faces[component])),
                "touches_outer_cut": False,
                "touches_inner_cut": False,
                "cut_curves": ";".join(sorted(curve_ids)),
                "assigned_label": ";".join(labels),
                "assignment_reason": "selected by 3D seed" if labels else "not selected",
                "selected_patch_labels": ";".join(labels),
            }
        )
    return rows


def named_shell_cut_surface_patches(
    mask: np.ndarray,
    cut_curves: Sequence[SurfaceCutCurve],
    patch_seeds: Sequence[SurfacePatchSeed],
    spacing: Optional[np.ndarray] = None,
    shell_backend: str = "marching_cubes",
    output_qc_dir: Optional[str | Path] = None,
    max_surface_quads: int = 1_500_000,
    input_warnings: Optional[Sequence[Dict]] = None,
) -> Tuple[Dict[str, SurfaceMesh], Dict[str, List[Dict]]]:
    """Cut named patches from the complete shell using user-selected seed faces."""

    mask = _as_bool_mask(mask)
    if not cut_curves:
        raise ValueError("3D shell patch build needs at least one closed cut curve")
    if not patch_seeds:
        raise ValueError("3D shell patch build needs at least one selected surface patch")

    shell = build_shell_mesh(
        mask,
        spacing=spacing,
        shell_backend=shell_backend,
        max_surface_quads=max_surface_quads,
    )
    if len(shell.faces) == 0:
        raise ValueError("3D shell patch build could not build a shell from an empty mask")

    edge_graph, edge_set = _build_shell_edge_graph(shell)
    traced_curves: List[SurfaceCutCurve] = []
    blocked_edges: set[Tuple[int, int]] = set()
    edge_owner_by_curve: Dict[Tuple[int, int], str] = {}
    overlapping_cut_edges: set[Tuple[int, int]] = set()
    for curve in cut_curves:
        traced = trace_cut_curve_on_shell(
            snap_cut_curve_to_shell(curve, shell),
            shell,
            edge_graph,
            edge_set,
        )
        traced_curves.append(traced)
        for edge in traced.cut_edges:
            owner_curve_id = edge_owner_by_curve.get(edge)
            if owner_curve_id is not None and owner_curve_id != traced.curve_id:
                overlapping_cut_edges.add(edge)
            edge_owner_by_curve.setdefault(edge, traced.curve_id)
            blocked_edges.add(edge)
    if overlapping_cut_edges:
        raise ValueError(
            f"closed cut curves overlap on {len(overlapping_cut_edges)} shell edge(s); "
            "closed curves may share picked points, but they cannot reuse the same shell edge"
        )

    adjacency, shared_edges, edge_to_faces = _build_face_adjacency(shell.faces)
    vertex_faces = _vertex_to_faces(shell.faces, len(shell.vertices))
    snapped_seeds = [snap_seed_to_shell_face(seed, shell, vertex_faces) for seed in patch_seeds]
    components = _face_components_after_cut(adjacency, shared_edges, blocked_edges)
    face_to_component = _face_component_lookup(components, len(shell.faces))

    labels_by_component: Dict[int, set[str]] = defaultdict(set)
    component_ids_by_label: Dict[str, set[int]] = defaultdict(set)
    seed_name_lookup: Dict[int, str] = {}
    warning_rows = list(input_warnings or [])
    for seed_index, seed in enumerate(snapped_seeds):
        label = safe_surface_name(seed.patch_label, fallback=f"surface_{seed_index + 1}")
        seed_name_lookup[seed_index] = label
        if seed.status == "error" or seed.snapped_face is None:
            continue
        component_id = int(face_to_component[int(seed.snapped_face)])
        if component_id < 0:
            seed.status = "error"
            seed.reason = "selected seed face is not part of any cut component"
            continue
        labels_by_component[component_id].add(label)
        component_ids_by_label[label].add(component_id)

    meshes: Dict[str, SurfaceMesh] = {}
    for label, component_ids in component_ids_by_label.items():
        face_indices = np.concatenate(
            [np.asarray(components[component_id], dtype=np.int64) for component_id in sorted(component_ids)]
        )
        meshes[label] = _extract_shell_submesh(label, shell, np.unique(face_indices))

    patch_rows = [
        _patch_summary_row(label, shell, np.concatenate([
            np.asarray(components[component_id], dtype=np.int64)
            for component_id in sorted(component_ids)
        ]))
        for label, component_ids in sorted(component_ids_by_label.items())
    ]
    shell_summary = _shell_summary_row(shell, mask, edge_to_faces)
    _edge_labels, edge_curve_ids = _cut_edge_label_lookup(traced_curves)
    component_rows = _named_patch_component_rows(
        components,
        shell,
        edge_curve_ids,
        labels_by_component,
    )
    cut_edge_rows = [
        {
            "curve_id": curve.curve_id,
            "edge_start": int(left),
            "edge_end": int(right),
        }
        for curve in traced_curves
        for left, right in curve.cut_edges
    ]
    warning_rows.extend(
        _shell_cut_warning_rows(
            traced_curves,
            snapped_seeds,
            patch_rows,
            component_rows=component_rows,
            shell_summary=shell_summary,
            minimum_component_count=2,
            warn_unclaimed_components=False,
        )
    )
    if not meshes:
        warning_rows.append(
            _shell_cut_warning_row(
                "no_selected_patch",
                "error",
                "selected_patches",
                "no selected surface patch produced any faces",
                "click inside the shell area that should be saved as a named surface",
            )
        )

    qc = {
        "shell_summary": [shell_summary],
        "cut_curves": [_surface_cut_curve_row(curve) for curve in traced_curves],
        "seeds": [_surface_patch_seed_row(seed) for seed in snapped_seeds],
        "patch_summary": patch_rows,
        "cut_components": component_rows,
        "cut_edges": cut_edge_rows,
        "warnings": warning_rows,
    }

    if output_qc_dir is not None:
        output_qc_path = Path(output_qc_dir)
        _write_shell_cut_qc_tables(qc, output_qc_path)
        _write_shell_cut_debug_geometry(
            output_qc_path,
            shell,
            traced_curves,
            snapped_seeds,
        )

    error_messages = [
        str(row.get("message", ""))
        for row in qc["warnings"]
        if row.get("severity") == "error"
    ]
    if error_messages:
        raise ValueError("3D shell patch build failed: " + "; ".join(error_messages))
    return meshes, qc


def shell_cut_surface_patches(
    mask: np.ndarray,
    boundaries: Sequence[BoundarySlice],
    cut_curves: Optional[Sequence[SurfaceCutCurve]] = None,
    patch_seeds: Optional[Sequence[SurfacePatchSeed]] = None,
    spacing: Optional[np.ndarray] = None,
    shell_backend: str = "voxel",
    output_qc_dir: Optional[str | Path] = None,
    max_surface_quads: int = 1_500_000,
    input_warnings: Optional[Sequence[Dict]] = None,
) -> Tuple[Dict[str, SurfaceMesh], Dict[str, List[Dict]]]:
    """Cut outer/inner/lateral patches from the complete mask shell."""

    mask = _as_bool_mask(mask)
    shell = build_shell_mesh(
        mask,
        spacing=spacing,
        shell_backend=shell_backend,
        max_surface_quads=max_surface_quads,
    )
    if len(shell.faces) == 0:
        raise ValueError("Shell-cut could not build a shell from an empty mask")

    curves = list(cut_curves) if cut_curves is not None else infer_cut_curves_from_boundaries(boundaries)
    if len(curves) < 2:
        raise ValueError("shell-cut needs outer and inner cut curves")

    seeds = list(patch_seeds or [])

    edge_graph, edge_set = _build_shell_edge_graph(shell)
    traced_curves: List[SurfaceCutCurve] = []
    blocked_edges: set[Tuple[int, int]] = set()
    for curve in curves:
        traced = trace_cut_curve_on_shell(
            snap_cut_curve_to_shell(curve, shell),
            shell,
            edge_graph,
            edge_set,
        )
        traced_curves.append(traced)
        blocked_edges.update(traced.cut_edges)

    adjacency, shared_edges, edge_to_faces = _build_face_adjacency(shell.faces)
    vertex_faces = _vertex_to_faces(shell.faces, len(shell.vertices))
    snapped_seeds = [snap_seed_to_shell_face(seed, shell, vertex_faces) for seed in seeds]
    components = _face_components_after_cut(adjacency, shared_edges, blocked_edges)
    edge_labels, edge_curve_ids = _cut_edge_label_lookup(traced_curves)
    faces_by_label, component_rows = _label_cut_components(
        components,
        shell,
        edge_labels,
        edge_curve_ids,
    )

    outer_faces = faces_by_label["outer"]
    inner_faces = faces_by_label["inner"]
    lateral_faces = faces_by_label["lateral"]
    overlap = np.intersect1d(outer_faces, inner_faces)

    meshes = {
        "outer": _extract_shell_submesh("outer", shell, outer_faces),
        "inner": _extract_shell_submesh("inner", shell, inner_faces),
        "lateral": _extract_shell_submesh("lateral", shell, lateral_faces),
    }

    patch_rows = [
        _patch_summary_row("outer", shell, outer_faces),
        _patch_summary_row("inner", shell, inner_faces),
        _patch_summary_row("lateral", shell, lateral_faces),
    ]
    shell_summary = _shell_summary_row(shell, mask, edge_to_faces)
    cut_edge_rows = [
        {
            "curve_id": curve.curve_id,
            "edge_start": int(left),
            "edge_end": int(right),
        }
        for curve in traced_curves
        for left, right in curve.cut_edges
    ]
    warning_rows = list(input_warnings or [])
    warning_rows.extend(
        _shell_cut_warning_rows(
            traced_curves,
            snapped_seeds,
            patch_rows,
            component_rows=component_rows,
            shell_summary=shell_summary,
            overlap_count=int(len(overlap)),
        )
    )
    qc = {
        "shell_summary": [shell_summary],
        "cut_curves": [_surface_cut_curve_row(curve) for curve in traced_curves],
        "seeds": [_surface_patch_seed_row(seed) for seed in snapped_seeds],
        "patch_summary": patch_rows,
        "cut_components": component_rows,
        "cut_edges": cut_edge_rows,
        "warnings": warning_rows,
    }

    if output_qc_dir is not None:
        output_qc_path = Path(output_qc_dir)
        _write_shell_cut_qc_tables(qc, output_qc_path)
        _write_shell_cut_debug_geometry(
            output_qc_path,
            shell,
            traced_curves,
            snapped_seeds,
        )

    error_messages = [
        str(row.get("message", ""))
        for row in qc["warnings"]
        if row.get("severity") == "error"
    ]
    if error_messages:
        raise ValueError("Shell-cut failed: " + "; ".join(error_messages))
    return meshes, qc


def _surface_label_name(label: int) -> str:
    names = {
        BOUNDARY_OUTER: "outer",
        BOUNDARY_INNER: "inner",
        BOUNDARY_LATERAL: "lateral",
        BOUNDARY_UNKNOWN: "unknown",
    }
    return names.get(int(label), f"label_{int(label)}")


def _label_mesh_name(label: int) -> Optional[str]:
    if int(label) == BOUNDARY_OUTER:
        return "outer"
    if int(label) == BOUNDARY_INNER:
        return "inner"
    if int(label) == BOUNDARY_LATERAL:
        return "lateral"
    return None


def _scaled_points(points: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    spacing = np.asarray(spacing, dtype=float).reshape(1, 3)
    return points * spacing


def _polyline_cumulative_distances(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.empty(0, dtype=float)
    distances = np.zeros(len(points), dtype=float)
    if len(points) > 1:
        distances[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    return distances


def _polyline_fraction_positions(points: np.ndarray) -> np.ndarray:
    distances = _polyline_cumulative_distances(points)
    if len(distances) == 0:
        return distances
    total = float(distances[-1])
    if total <= 0:
        return np.linspace(0.0, 1.0, len(distances))
    return distances / total


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros_like(vector, dtype=float)
    return vector / norm


def _arc_tangent(points: np.ndarray, start: bool) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return np.zeros(3, dtype=float)
    if start:
        return _unit_vector(points[1] - points[0])
    return _unit_vector(points[-1] - points[-2])


def _arc_curvature_profile(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return np.empty(0, dtype=float)
    values: List[float] = []
    for index in range(1, len(points) - 1):
        left = points[index] - points[index - 1]
        right = points[index + 1] - points[index]
        left_unit = _unit_vector(left)
        right_unit = _unit_vector(right)
        if not np.any(left_unit) or not np.any(right_unit):
            values.append(0.0)
            continue
        dot = float(np.clip(np.dot(left_unit, right_unit), -1.0, 1.0))
        angle = math.acos(dot)
        local_length = max(1e-6, 0.5 * (float(np.linalg.norm(left)) + float(np.linalg.norm(right))))
        values.append(angle / local_length)
    return np.asarray(values, dtype=float)


def _make_arc_segment(
    boundary: BoundarySlice,
    label: int,
    points: np.ndarray,
    local_index: int,
    spacing: np.ndarray,
) -> ArcSegment:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    scoring_points = _scaled_points(points, spacing)
    length = _polyline_length(scoring_points, closed=False)
    name = _surface_label_name(label)
    return ArcSegment(
        arc_id=f"s{boundary.slice_index}_c{boundary.contour_id}_{name}_{local_index}",
        slice_index=int(boundary.slice_index),
        contour_id=int(boundary.contour_id),
        label=int(label),
        points=points,
        scoring_points=scoring_points,
        is_closed=_is_closed_polyline(points),
        source=str(boundary.source),
        confidence=float(boundary.confidence),
        boundary_source=str(boundary.source),
        length=float(length),
        centroid=np.asarray(scoring_points.mean(axis=0), dtype=float) if len(scoring_points) else np.zeros(3),
        start_point=np.asarray(scoring_points[0], dtype=float) if len(scoring_points) else np.zeros(3),
        end_point=np.asarray(scoring_points[-1], dtype=float) if len(scoring_points) else np.zeros(3),
        tangent_start=_arc_tangent(scoring_points, start=True),
        tangent_end=_arc_tangent(scoring_points, start=False),
        curvature=_arc_curvature_profile(scoring_points),
    )


def build_labeled_arcs(
    boundaries: Sequence[BoundarySlice],
    spacing: np.ndarray,
) -> Dict[int, List[ArcSegment]]:
    arcs_by_slice: Dict[int, List[ArcSegment]] = {}
    for boundary in sorted(boundaries, key=lambda item: item.slice_index):
        arcs: List[ArcSegment] = []
        if len(boundary.outer_arc) >= 2:
            arcs.append(_make_arc_segment(boundary, BOUNDARY_OUTER, boundary.outer_arc, 0, spacing))
        if len(boundary.inner_arc) >= 2:
            arcs.append(_make_arc_segment(boundary, BOUNDARY_INNER, boundary.inner_arc, 0, spacing))
        for lateral_index, lateral_arc in enumerate(boundary.lateral_arcs):
            if len(lateral_arc) >= 2:
                arcs.append(
                    _make_arc_segment(
                        boundary,
                        BOUNDARY_LATERAL,
                        lateral_arc,
                        lateral_index,
                        spacing,
                    )
                )
        arcs_by_slice[int(boundary.slice_index)] = arcs
    return arcs_by_slice


def _arc_range_length(arc: ArcSegment, start_fraction: float, end_fraction: float) -> float:
    if arc.length <= 0:
        return 0.0
    spans = _split_fraction_range(start_fraction, end_fraction, arc.is_closed)
    return float(sum(max(0.0, end - start) for start, end in spans) * arc.length)


def _fraction_to_index(points: np.ndarray, fraction: float) -> int:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return 0
    fractions = _polyline_fraction_positions(points)
    fraction = max(0.0, min(1.0, float(fraction)))
    index = int(np.searchsorted(fractions, fraction, side="left"))
    return max(0, min(index, len(points) - 1))


def _split_fraction_range(
    start_fraction: float,
    end_fraction: float,
    is_closed: bool,
) -> List[Tuple[float, float]]:
    start = float(start_fraction)
    end = float(end_fraction)
    if not is_closed:
        start = max(0.0, min(1.0, start))
        end = max(0.0, min(1.0, end))
        if end < start:
            start, end = end, start
        return [(start, end)] if end - start > 1e-9 else []

    span = end - start
    if span >= 1.0:
        return [(0.0, 1.0)]
    start_mod = start % 1.0
    end_mod = end % 1.0
    if end <= start or end_mod < start_mod:
        ranges = []
        if start_mod < 1.0:
            ranges.append((start_mod, 1.0))
        if end_mod > 0.0:
            ranges.append((0.0, end_mod))
        return ranges
    return [(start_mod, end_mod)] if end_mod - start_mod > 1e-9 else []


def _merge_fraction_ranges(ranges: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    valid = sorted((max(0.0, start), min(1.0, end)) for start, end in ranges if end - start > 1e-9)
    if not valid:
        return []
    merged: List[Tuple[float, float]] = [valid[0]]
    for start, end in valid[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1e-6:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


class ArcUsageMap:
    def __init__(self, arcs: Sequence[ArcSegment]):
        self.arc_by_id = {arc.arc_id: arc for arc in arcs}
        self.used_ranges_by_arc: Dict[str, List[Tuple[float, float]]] = {
            arc.arc_id: [] for arc in arcs
        }

    def add_used_range(self, arc_id: str, start_fraction: float, end_fraction: float) -> None:
        arc = self.arc_by_id[arc_id]
        self.used_ranges_by_arc.setdefault(arc_id, []).extend(
            _split_fraction_range(start_fraction, end_fraction, arc.is_closed)
        )
        self.used_ranges_by_arc[arc_id] = _merge_fraction_ranges(self.used_ranges_by_arc[arc_id])

    def overlap_fraction(self, arc_id: str, start_fraction: float, end_fraction: float) -> float:
        arc = self.arc_by_id[arc_id]
        candidate_ranges = _split_fraction_range(start_fraction, end_fraction, arc.is_closed)
        candidate_length = sum(end - start for start, end in candidate_ranges)
        if candidate_length <= 1e-9:
            return 0.0
        used_ranges = self.used_ranges_by_arc.get(arc_id, [])
        overlap = 0.0
        for start, end in candidate_ranges:
            for used_start, used_end in used_ranges:
                overlap += max(0.0, min(end, used_end) - max(start, used_start))
        return float(overlap / candidate_length)

    def coverage_fraction(self, arc_id: str) -> float:
        ranges = self.used_ranges_by_arc.get(arc_id, [])
        return float(sum(end - start for start, end in ranges))

    def unmatched_ranges(self, arc_id: str) -> List[Tuple[float, float]]:
        used = _merge_fraction_ranges(self.used_ranges_by_arc.get(arc_id, []))
        if not used:
            return [(0.0, 1.0)]
        output: List[Tuple[float, float]] = []
        cursor = 0.0
        for start, end in used:
            if start > cursor + 1e-6:
                output.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < 1.0 - 1e-6:
            output.append((cursor, 1.0))
        return output


def _interpolate_polyline(points: np.ndarray, fraction: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.zeros(3, dtype=float)
    if len(points) == 1:
        return points[0].copy()
    fractions = _polyline_fraction_positions(points)
    fraction = max(0.0, min(1.0, float(fraction)))
    index = int(np.searchsorted(fractions, fraction, side="right") - 1)
    index = max(0, min(index, len(points) - 2))
    left_fraction = float(fractions[index])
    right_fraction = float(fractions[index + 1])
    if right_fraction <= left_fraction:
        return points[index].copy()
    t = (fraction - left_fraction) / (right_fraction - left_fraction)
    return (1.0 - t) * points[index] + t * points[index + 1]


def _extract_open_polyline_range(
    points: np.ndarray,
    start_fraction: float,
    end_fraction: float,
) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) == 0:
        return points
    start = max(0.0, min(1.0, float(start_fraction)))
    end = max(0.0, min(1.0, float(end_fraction)))
    if end < start:
        start, end = end, start
    if abs(end - start) <= 1e-9:
        point = _interpolate_polyline(points, start)
        return np.vstack([point, point])

    fractions = _polyline_fraction_positions(points)
    keep = (fractions > start + 1e-9) & (fractions < end - 1e-9)
    selected = [_interpolate_polyline(points, start)]
    selected.extend(points[keep])
    selected.append(_interpolate_polyline(points, end))
    return np.asarray(selected, dtype=float)


def extract_polyline_range(
    points: np.ndarray,
    start_fraction: float,
    end_fraction: float,
    closed: bool = False,
) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) == 0:
        return points
    if not closed:
        return _extract_open_polyline_range(points, start_fraction, end_fraction)

    unique = _normalize_contour(points)
    if len(unique) < 2:
        return unique
    closed_points = np.vstack([unique, unique[:1]])
    start = float(start_fraction)
    end = float(end_fraction)
    if end < start:
        end += 1.0
    if end - start >= 1.0:
        return closed_points

    start_mod = start % 1.0
    end_mod = end % 1.0
    if end > 1.0 or end_mod < start_mod:
        first = _extract_open_polyline_range(closed_points, start_mod, 1.0)
        second = _extract_open_polyline_range(closed_points, 0.0, end_mod)
        return np.vstack([first[:-1], second])
    return _extract_open_polyline_range(closed_points, start_mod, end_mod)


def _normalization_radius(spacing: np.ndarray, slice_gap: int = 1) -> float:
    spacing = np.asarray(spacing, dtype=float).reshape(-1)
    if len(spacing) < 3 or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        return 1.0
    xy_spacing = float(max(spacing[1], spacing[2]))
    z_spacing = float(spacing[0]) * max(1, int(slice_gap))
    return max(z_spacing, xy_spacing, 1e-6)


def _sample_points_with_fractions(
    points: np.ndarray,
    sample_count: int,
    closed: bool = False,
    duplicate_closed: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if closed:
        unique = _normalize_contour(points)
        if len(unique) < 2:
            return unique, np.zeros(len(unique), dtype=float)
        loop = np.vstack([unique, unique[:1]])
        count = max(3, int(sample_count))
        sampled = resample_polyline(loop, count + 1)[:-1]
        fractions = np.linspace(0.0, 1.0, count, endpoint=False)
        if duplicate_closed:
            sampled = np.vstack([sampled, sampled])
            fractions = np.r_[fractions, fractions + 1.0]
        return sampled, fractions

    count = max(2, int(sample_count))
    sampled = resample_polyline(points, count)
    fractions = np.linspace(0.0, 1.0, len(sampled))
    return sampled, fractions


def _sample_count_for_arc(arc: ArcSegment, base: int = 48, maximum: int = 160) -> int:
    if arc.length <= 0:
        return base
    return max(16, min(maximum, int(round(base * max(1.0, arc.length / 48.0)))))


def _subsequence_dtw(
    query: np.ndarray,
    reference: np.ndarray,
    query_fractions: np.ndarray,
    reference_fractions: np.ndarray,
    normalization: float,
    min_reference_span_fraction: float = 0.15,
) -> Dict[str, object]:
    query = np.asarray(query, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if len(query) < 2 or len(reference) < 2:
        return {"valid": False, "reason": "too_few_points"}

    distances = np.linalg.norm(query[:, None, :] - reference[None, :, :], axis=2)
    distances = distances / max(float(normalization), 1e-6)
    rows, cols = distances.shape
    cost = np.full((rows, cols), np.inf, dtype=float)
    previous = np.full((rows, cols, 2), -1, dtype=np.int32)
    cost[0, :] = distances[0, :]

    for row in range(1, rows):
        for col in range(cols):
            choices = [(cost[row - 1, col], row - 1, col)]
            if col > 0:
                choices.append((cost[row, col - 1], row, col - 1))
                choices.append((cost[row - 1, col - 1], row - 1, col - 1))
            best_cost, prev_row, prev_col = min(choices, key=lambda item: item[0])
            cost[row, col] = distances[row, col] + best_cost
            previous[row, col] = [prev_row, prev_col]

    end_col = int(np.argmin(cost[-1]))
    if not np.isfinite(cost[-1, end_col]):
        return {"valid": False, "reason": "no_finite_path"}

    path_rows: List[int] = []
    path_cols: List[int] = []
    row = rows - 1
    col = end_col
    while row >= 0 and col >= 0:
        path_rows.append(row)
        path_cols.append(col)
        if row == 0:
            break
        prev_row, prev_col = previous[row, col]
        if prev_row < 0 or prev_col < 0:
            break
        row, col = int(prev_row), int(prev_col)

    path_rows = list(reversed(path_rows))
    path_cols = list(reversed(path_cols))
    if len(path_rows) < 2:
        return {"valid": False, "reason": "short_path"}

    ref_start_fraction = float(reference_fractions[min(path_cols)])
    ref_end_fraction = float(reference_fractions[max(path_cols)])
    ref_span = abs(ref_end_fraction - ref_start_fraction)
    if ref_span < min_reference_span_fraction:
        return {
            "valid": False,
            "reason": "reference_span_too_short",
            "reference_span": ref_span,
        }

    unique_reference_steps = len(np.unique(path_cols))
    progress_ratio = unique_reference_steps / float(max(1, len(path_cols)))
    if progress_ratio < 0.25:
        return {
            "valid": False,
            "reason": "dtw_collapse",
            "reference_progress_ratio": progress_ratio,
        }

    path_distances = distances[np.asarray(path_rows, dtype=int), np.asarray(path_cols, dtype=int)]
    return {
        "valid": True,
        "query_indices": np.asarray(path_rows, dtype=np.int32),
        "reference_indices": np.asarray(path_cols, dtype=np.int32),
        "query_fraction_start": float(query_fractions[min(path_rows)]),
        "query_fraction_end": float(query_fractions[max(path_rows)]),
        "reference_fraction_start": ref_start_fraction,
        "reference_fraction_end": ref_end_fraction,
        "mean_distance": float(np.mean(path_distances)),
        "max_distance": float(np.max(path_distances)),
        "cost": float(cost[-1, end_col] / max(1, len(path_rows))),
    }


def generate_arc_match_candidates(
    arcs_left: Sequence[ArcSegment],
    arcs_right: Sequence[ArcSegment],
    spacing: np.ndarray,
) -> List[Tuple[ArcSegment, ArcSegment]]:
    candidates: List[Tuple[ArcSegment, ArcSegment]] = []
    for left in arcs_left:
        for right in arcs_right:
            if left.label != right.label:
                continue
            if len(left.points) < 2 or len(right.points) < 2:
                continue
            centroid_distance = float(np.linalg.norm(left.centroid - right.centroid))
            scale = max(40.0, 2.5 * max(1.0, min(left.length, right.length)))
            if centroid_distance > scale + _normalization_radius(spacing):
                continue
            candidates.append((left, right))
    return candidates


def _curve_cost_components(
    left: ArcSegment,
    right: ArcSegment,
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
    mean_distance: float,
    max_distance: float,
    normalization: float,
    direction: int,
) -> Tuple[float, float, float, float, float]:
    left_points = extract_polyline_range(left.scoring_points, left_start, left_end, left.is_closed)
    right_points = extract_polyline_range(right.scoring_points, right_start, right_end, right.is_closed)
    if direction < 0:
        right_points = right_points[::-1]
    if len(left_points) == 0 or len(right_points) == 0:
        return float("inf"), float("inf"), float("inf"), float("inf"), float("inf")

    endpoint_cost = (
        float(np.linalg.norm(left_points[0] - right_points[0]))
        + float(np.linalg.norm(left_points[-1] - right_points[-1]))
    ) / (2.0 * max(normalization, 1e-6))
    left_start_tangent = _arc_tangent(left_points, start=True)
    right_start_tangent = _arc_tangent(right_points, start=True)
    left_end_tangent = _arc_tangent(left_points, start=False)
    right_end_tangent = _arc_tangent(right_points, start=False)
    tangent_cost = 0.5 * (
        1.0 - float(np.clip(np.dot(left_start_tangent, right_start_tangent), -1.0, 1.0))
        + 1.0 - float(np.clip(np.dot(left_end_tangent, right_end_tangent), -1.0, 1.0))
    )
    left_length = max(1e-6, _polyline_length(left_points))
    right_length = max(1e-6, _polyline_length(right_points))
    length_cost = abs(math.log(left_length / right_length))
    left_curvature = _arc_curvature_profile(left_points)
    right_curvature = _arc_curvature_profile(right_points)
    if len(left_curvature) and len(right_curvature):
        curvature_cost = abs(float(np.mean(left_curvature)) - float(np.mean(right_curvature)))
    else:
        curvature_cost = 0.0
    return mean_distance, endpoint_cost, tangent_cost, length_cost, curvature_cost


def _match_cost(
    curve_cost: float,
    endpoint_cost: float,
    tangent_cost: float,
    length_cost: float,
    curvature_cost: float,
) -> float:
    return float(
        0.45 * curve_cost
        + 0.20 * endpoint_cost
        + 0.15 * tangent_cost
        + 0.15 * length_cost
        + 0.05 * curvature_cost
    )


def _classify_arc_match(cost: float, coverage_prev: float, coverage_next: float, left: ArcSegment, right: ArcSegment) -> str:
    if cost <= 2.5 and coverage_prev >= 0.85 and coverage_next >= 0.85:
        return "normal_match"
    if left.is_closed != right.is_closed and cost <= 4.0:
        return "open_to_closed" if right.is_closed else "closed_to_open"
    if cost <= 4.0 and (coverage_prev >= 0.70 or coverage_next >= 0.70):
        return "partial_match"
    return "rejected"


def score_arc_match(left: ArcSegment, right: ArcSegment, spacing: np.ndarray) -> ArcMatch:
    slice_gap = abs(int(right.slice_index) - int(left.slice_index))
    normalization = _normalization_radius(spacing, slice_gap=slice_gap)

    left_count = min(96, max(24, _sample_count_for_arc(left, base=32, maximum=96)))
    right_count = min(160, max(24, _sample_count_for_arc(right, base=32, maximum=160)))
    left_sample, left_fractions = _sample_points_with_fractions(
        left.scoring_points, left_count, closed=left.is_closed
    )
    right_sample, right_fractions = _sample_points_with_fractions(
        right.scoring_points,
        right_count,
        closed=right.is_closed,
        duplicate_closed=right.is_closed and not left.is_closed,
    )

    scored: List[ArcMatch] = []
    for direction, sample, fractions in (
        (1, right_sample, right_fractions),
        (-1, right_sample[::-1], right_fractions[::-1]),
    ):
        direct_count = min(len(left_sample), len(sample), 64)
        if direct_count >= 2:
            left_direct = resample_polyline(left_sample, direct_count)
            right_direct = resample_polyline(sample, direct_count)
            distances = np.linalg.norm(left_direct - right_direct, axis=1) / max(normalization, 1e-6)
            right_start = float(min(fractions[0], fractions[-1]))
            right_end = float(max(fractions[0], fractions[-1]))
            components = _curve_cost_components(
                left,
                right,
                0.0,
                1.0,
                right_start,
                right_end,
                float(np.mean(distances)),
                float(np.max(distances)),
                normalization,
                direction,
            )
            cost = _match_cost(*components)
            coverage_prev = 1.0
            coverage_next = min(1.0, max(0.0, right_end - right_start))
            event_type = _classify_arc_match(cost, coverage_prev, coverage_next, left, right)
            scored.append(
                ArcMatch(
                    prev_arc_id=left.arc_id,
                    next_arc_id=right.arc_id,
                    label=left.label,
                    prev_sample_indices=np.arange(direct_count, dtype=np.int32),
                    next_sample_indices=np.arange(direct_count, dtype=np.int32),
                    prev_fraction_start=0.0,
                    prev_fraction_end=1.0,
                    next_fraction_start=right_start,
                    next_fraction_end=right_end,
                    direction=direction,
                    event_type=event_type,
                    cost=cost,
                    confidence=1.0 / (1.0 + max(0.0, cost)),
                    coverage_prev=coverage_prev,
                    coverage_next=coverage_next,
                    mean_distance=components[0],
                    max_distance=float(np.max(distances)),
                    length_ratio=math.exp(components[3]) if np.isfinite(components[3]) else float("inf"),
                    reason="direct_full_arc_match",
                )
            )

        if left.length <= right.length:
            query = left_sample
            query_fractions = left_fractions
            reference = sample
            reference_fractions = fractions
            partial = _subsequence_dtw(query, reference, query_fractions, reference_fractions, normalization)
            if partial.get("valid"):
                right_start = float(min(partial["reference_fraction_start"], partial["reference_fraction_end"]))
                right_end = float(max(partial["reference_fraction_start"], partial["reference_fraction_end"]))
                components = _curve_cost_components(
                    left,
                    right,
                    0.0,
                    1.0,
                    right_start,
                    right_end,
                    float(partial["mean_distance"]),
                    float(partial["max_distance"]),
                    normalization,
                    direction,
                )
                cost = _match_cost(*components)
                coverage_prev = 1.0
                coverage_next = min(1.0, max(0.0, right_end - right_start))
                event_type = _classify_arc_match(cost, coverage_prev, coverage_next, left, right)
                if event_type == "normal_match":
                    right_start, right_end = 0.0, 1.0
                    coverage_next = 1.0
                scored.append(
                    ArcMatch(
                        prev_arc_id=left.arc_id,
                        next_arc_id=right.arc_id,
                        label=left.label,
                        prev_sample_indices=np.asarray(partial["query_indices"], dtype=np.int32),
                        next_sample_indices=np.asarray(partial["reference_indices"], dtype=np.int32),
                        prev_fraction_start=0.0,
                        prev_fraction_end=1.0,
                        next_fraction_start=right_start,
                        next_fraction_end=right_end,
                        direction=direction,
                        event_type=event_type,
                        cost=cost,
                        confidence=1.0 / (1.0 + max(0.0, cost)),
                        coverage_prev=coverage_prev,
                        coverage_next=coverage_next,
                        mean_distance=components[0],
                        max_distance=float(partial["max_distance"]),
                        length_ratio=math.exp(components[3]) if np.isfinite(components[3]) else float("inf"),
                        reason="subsequence_dtw_left_full",
                    )
                )
        else:
            partial = _subsequence_dtw(sample, left_sample, fractions, left_fractions, normalization)
            if partial.get("valid"):
                left_start = float(min(partial["reference_fraction_start"], partial["reference_fraction_end"]))
                left_end = float(max(partial["reference_fraction_start"], partial["reference_fraction_end"]))
                right_start = float(min(fractions[0], fractions[-1]))
                right_end = float(max(fractions[0], fractions[-1]))
                components = _curve_cost_components(
                    left,
                    right,
                    left_start,
                    left_end,
                    right_start,
                    right_end,
                    float(partial["mean_distance"]),
                    float(partial["max_distance"]),
                    normalization,
                    direction,
                )
                cost = _match_cost(*components)
                coverage_prev = min(1.0, max(0.0, left_end - left_start))
                coverage_next = min(1.0, max(0.0, right_end - right_start))
                event_type = _classify_arc_match(cost, coverage_prev, coverage_next, left, right)
                if event_type == "normal_match":
                    left_start, left_end = 0.0, 1.0
                    right_start, right_end = 0.0, 1.0
                    coverage_prev = 1.0
                    coverage_next = 1.0
                scored.append(
                    ArcMatch(
                        prev_arc_id=left.arc_id,
                        next_arc_id=right.arc_id,
                        label=left.label,
                        prev_sample_indices=np.asarray(partial["reference_indices"], dtype=np.int32),
                        next_sample_indices=np.asarray(partial["query_indices"], dtype=np.int32),
                        prev_fraction_start=left_start,
                        prev_fraction_end=left_end,
                        next_fraction_start=right_start,
                        next_fraction_end=right_end,
                        direction=direction,
                        event_type=event_type,
                        cost=cost,
                        confidence=1.0 / (1.0 + max(0.0, cost)),
                        coverage_prev=coverage_prev,
                        coverage_next=coverage_next,
                        mean_distance=components[0],
                        max_distance=float(partial["max_distance"]),
                        length_ratio=math.exp(components[3]) if np.isfinite(components[3]) else float("inf"),
                        reason="subsequence_dtw_right_full",
                    )
                )

    if not scored:
        return ArcMatch(
            prev_arc_id=left.arc_id,
            next_arc_id=right.arc_id,
            label=left.label,
            prev_sample_indices=np.empty(0, dtype=np.int32),
            next_sample_indices=np.empty(0, dtype=np.int32),
            prev_fraction_start=0.0,
            prev_fraction_end=0.0,
            next_fraction_start=0.0,
            next_fraction_end=0.0,
            direction=1,
            event_type="rejected",
            cost=float("inf"),
            confidence=0.0,
            coverage_prev=0.0,
            coverage_next=0.0,
            mean_distance=float("inf"),
            max_distance=float("inf"),
            length_ratio=float("inf"),
            reason="no_valid_match",
        )

    return min(scored, key=lambda item: item.cost)


def solve_arc_assignment(
    arcs_left: Sequence[ArcSegment],
    arcs_right: Sequence[ArcSegment],
    matches: Sequence[ArcMatch],
    max_interval_overlap: float = 0.20,
    ambiguous_relative_margin: float = 0.10,
) -> Tuple[List[ArcMatch], List[TopologyEvent]]:
    all_arcs = list(arcs_left) + list(arcs_right)
    usage = ArcUsageMap(all_arcs)
    events: List[TopologyEvent] = []
    accepted: List[ArcMatch] = []
    viable = [
        match
        for match in matches
        if match.event_type != "rejected" and np.isfinite(match.cost) and match.confidence >= 0.10
    ]
    viable.sort(key=lambda item: item.cost)

    for index, match in enumerate(viable):
        left_overlap = usage.overlap_fraction(
            match.prev_arc_id, match.prev_fraction_start, match.prev_fraction_end
        )
        right_overlap = usage.overlap_fraction(
            match.next_arc_id, match.next_fraction_start, match.next_fraction_end
        )
        if left_overlap > max_interval_overlap or right_overlap > max_interval_overlap:
            continue

        ambiguous = False
        for other in viable[index + 1 :]:
            if other.prev_arc_id != match.prev_arc_id and other.next_arc_id != match.next_arc_id:
                continue
            relative_margin = (other.cost - match.cost) / max(match.cost, 1e-6)
            if relative_margin < ambiguous_relative_margin:
                ambiguous = True
                break
        if ambiguous:
            continue

        match.accepted = True
        accepted.append(match)
        usage.add_used_range(match.prev_arc_id, match.prev_fraction_start, match.prev_fraction_end)
        usage.add_used_range(match.next_arc_id, match.next_fraction_start, match.next_fraction_end)

    return accepted, events


def _arc_range_event(
    arc: ArcSegment,
    slice_left: int,
    slice_right: int,
    event_type: str,
    start: float,
    end: float,
    coverage_prev: float,
    coverage_next: float,
    reason: str,
) -> TopologyEvent:
    return TopologyEvent(
        slice_left=int(slice_left),
        slice_right=int(slice_right),
        arc_id=arc.arc_id,
        label=int(arc.label),
        event_type=event_type,
        severity="review",
        start_fraction=float(start),
        end_fraction=float(end),
        start_index=_fraction_to_index(arc.points, start),
        end_index=_fraction_to_index(arc.points, end),
        coverage_prev=float(coverage_prev),
        coverage_next=float(coverage_next),
        cost=None,
        reason=reason,
    )


def emit_unmatched_range_events(
    usage: ArcUsageMap,
    arcs_left: Sequence[ArcSegment],
    arcs_right: Sequence[ArcSegment],
    slice_left: int,
    slice_right: int,
) -> Tuple[List[TopologyEvent], List[Dict]]:
    events: List[TopologyEvent] = []
    open_edges: List[Dict] = []
    for arc in arcs_left:
        coverage = usage.coverage_fraction(arc.arc_id)
        for start, end in usage.unmatched_ranges(arc.arc_id):
            if end - start <= 1e-6:
                continue
            events.append(
                _arc_range_event(
                    arc,
                    slice_left,
                    slice_right,
                    "death",
                    start,
                    end,
                    coverage,
                    0.0,
                    "left arc range has no reliable next-slice match",
                )
            )
            open_edges.append(_open_edge_row_from_arc(arc, slice_left, slice_right, start, end, "unmatched_arc_range", "death"))
    for arc in arcs_right:
        coverage = usage.coverage_fraction(arc.arc_id)
        for start, end in usage.unmatched_ranges(arc.arc_id):
            if end - start <= 1e-6:
                continue
            events.append(
                _arc_range_event(
                    arc,
                    slice_left,
                    slice_right,
                    "birth",
                    start,
                    end,
                    0.0,
                    coverage,
                    "right arc range has no reliable previous-slice match",
                )
            )
            open_edges.append(_open_edge_row_from_arc(arc, slice_left, slice_right, start, end, "partial_match_remainder", "birth"))
    return events, open_edges


def _open_edge_row_from_arc(
    arc: ArcSegment,
    slice_left: int,
    slice_right: int,
    start: float,
    end: float,
    source: str,
    reason: str,
) -> Dict:
    points = extract_polyline_range(arc.points, start, end, closed=arc.is_closed)
    if len(points) == 0:
        start_point = end_point = np.zeros(3, dtype=float)
    else:
        start_point = points[0]
        end_point = points[-1]
    return {
        "mesh_label": _surface_label_name(arc.label),
        "slice_left": int(slice_left),
        "slice_right": int(slice_right),
        "match_id": "",
        "edge_start_x": float(start_point[0]),
        "edge_start_y": float(start_point[1]),
        "edge_start_z": float(start_point[2]),
        "edge_end_x": float(end_point[0]),
        "edge_end_y": float(end_point[1]),
        "edge_end_z": float(end_point[2]),
        "source": source,
        "reason": reason,
    }


class _MeshBuilder:
    def __init__(self, name: str):
        self.name = name
        self.vertices: List[np.ndarray] = []
        self.faces: List[np.ndarray] = []
        self.offset = 0

    def add_patch(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        if len(vertices) == 0 or len(faces) == 0:
            return
        self.vertices.append(np.asarray(vertices, dtype=float))
        self.faces.append(np.asarray(faces, dtype=np.int32) + self.offset)
        self.offset += len(vertices)

    def has_faces(self) -> bool:
        return bool(self.faces)

    def to_mesh(self) -> SurfaceMesh:
        vertices = np.vstack(self.vertices)
        faces = np.vstack(self.faces)
        return _compact_surface_mesh(self.name, vertices, faces)


def _build_arc_strip(left_points: np.ndarray, right_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    left_points = np.asarray(left_points, dtype=float).reshape(-1, 3)
    right_points = np.asarray(right_points, dtype=float).reshape(-1, 3)
    if len(left_points) < 2 or len(right_points) < 2:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int32)
    vertices = np.vstack([left_points, right_points])
    left_indices = list(range(len(left_points)))
    right_indices = list(range(len(left_points), len(left_points) + len(right_points)))
    faces = _strip_faces_between_polylines(left_indices, left_points, right_indices, right_points)
    return vertices, np.asarray(faces, dtype=np.int32)


def _triangle_area(points: np.ndarray) -> float:
    return float(0.5 * np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])))


def _triangle_aspect(points: np.ndarray) -> float:
    edges = [
        float(np.linalg.norm(points[0] - points[1])),
        float(np.linalg.norm(points[1] - points[2])),
        float(np.linalg.norm(points[2] - points[0])),
    ]
    longest = max(edges)
    area = _triangle_area(points)
    if area <= 1e-12:
        return float("inf")
    shortest_altitude = 2.0 * area / max(longest, 1e-12)
    return float(longest / max(shortest_altitude, 1e-12))


def filter_mesh_faces_by_quality(
    vertices: np.ndarray,
    faces: np.ndarray,
    spacing: np.ndarray,
    mesh_label: str,
    slice_left: int,
    slice_right: int,
    match_id: str,
) -> Tuple[np.ndarray, List[Dict], List[Dict]]:
    if len(faces) == 0:
        return faces, [], []
    spacing = np.asarray(spacing, dtype=float).reshape(-1)
    spacing_scale = float(np.nanmax(spacing[:3])) if len(spacing) >= 3 else 1.0
    max_edge_limit = max(50.0, 5.0 * max(1.0, spacing_scale))
    kept: List[np.ndarray] = []
    quality_rows: List[Dict] = []
    open_edges: List[Dict] = []

    for face_index, face in enumerate(np.asarray(faces, dtype=np.int32)):
        points = vertices[face]
        edge_lengths = [
            float(np.linalg.norm(points[0] - points[1])),
            float(np.linalg.norm(points[1] - points[2])),
            float(np.linalg.norm(points[2] - points[0])),
        ]
        max_edge = max(edge_lengths)
        area = _triangle_area(points)
        aspect = _triangle_aspect(points)
        status = "kept"
        reason = ""
        if area <= 1e-9:
            status = "rejected"
            reason = "zero_area"
        elif aspect > 50.0:
            status = "rejected"
            reason = "aspect_ratio"
        elif max_edge > max_edge_limit:
            status = "warn"
            reason = "long_edge"

        quality_rows.append(
            {
                "mesh_label": mesh_label,
                "slice_left": int(slice_left),
                "slice_right": int(slice_right),
                "match_id": match_id,
                "face_local_id": int(face_index),
                "max_edge": max_edge,
                "aspect_ratio": aspect,
                "area": area,
                "status": status,
                "reason": reason,
            }
        )
        if status == "rejected":
            open_edges.append(
                {
                    "mesh_label": mesh_label,
                    "slice_left": int(slice_left),
                    "slice_right": int(slice_right),
                    "match_id": match_id,
                    "edge_start_x": float(points[0, 0]),
                    "edge_start_y": float(points[0, 1]),
                    "edge_start_z": float(points[0, 2]),
                    "edge_end_x": float(points[1, 0]),
                    "edge_end_y": float(points[1, 1]),
                    "edge_end_z": float(points[1, 2]),
                    "source": "rejected_bad_face",
                    "reason": reason,
                }
            )
            continue
        kept.append(face)
    return np.asarray(kept, dtype=np.int32), quality_rows, open_edges


def _match_to_row(match: ArcMatch, slice_left: int, slice_right: int) -> Dict:
    return {
        "slice_left": int(slice_left),
        "slice_right": int(slice_right),
        "prev_arc_id": match.prev_arc_id,
        "next_arc_id": match.next_arc_id,
        "label": _surface_label_name(match.label),
        "event_type": match.event_type,
        "confidence": float(match.confidence),
        "cost": float(match.cost),
        "coverage_prev": float(match.coverage_prev),
        "coverage_next": float(match.coverage_next),
        "prev_fraction_start": float(match.prev_fraction_start),
        "prev_fraction_end": float(match.prev_fraction_end),
        "next_fraction_start": float(match.next_fraction_start),
        "next_fraction_end": float(match.next_fraction_end),
        "mean_distance": float(match.mean_distance),
        "max_distance": float(match.max_distance),
        "length_ratio": float(match.length_ratio),
        "direction": int(match.direction),
        "accepted": bool(match.accepted),
        "reason": match.reason,
    }


def _event_to_row(event: TopologyEvent) -> Dict:
    return {
        "slice_left": int(event.slice_left),
        "slice_right": int(event.slice_right),
        "arc_id": event.arc_id,
        "label": _surface_label_name(event.label),
        "event_type": event.event_type,
        "severity": event.severity,
        "start_fraction": float(event.start_fraction),
        "end_fraction": float(event.end_fraction),
        "start_index": int(event.start_index),
        "end_index": int(event.end_index),
        "coverage_prev": float(event.coverage_prev),
        "coverage_next": float(event.coverage_next),
        "cost": "" if event.cost is None else float(event.cost),
        "reason": event.reason,
    }


def _write_arc_graph_qc_tables(qc: Dict[str, List[Dict]], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_rows(
        output_dir / "arc_matches.csv",
        qc.get("arc_matches", []),
        fieldnames=[
            "slice_left",
            "slice_right",
            "prev_arc_id",
            "next_arc_id",
            "label",
            "event_type",
            "confidence",
            "cost",
            "coverage_prev",
            "coverage_next",
            "prev_fraction_start",
            "prev_fraction_end",
            "next_fraction_start",
            "next_fraction_end",
            "mean_distance",
            "max_distance",
            "length_ratio",
            "direction",
            "accepted",
            "reason",
        ],
    )
    _write_csv_rows(
        output_dir / "topology_events.csv",
        qc.get("topology_events", []),
        fieldnames=[
            "slice_left",
            "slice_right",
            "arc_id",
            "label",
            "event_type",
            "severity",
            "start_fraction",
            "end_fraction",
            "start_index",
            "end_index",
            "coverage_prev",
            "coverage_next",
            "cost",
            "reason",
        ],
    )
    _write_csv_rows(
        output_dir / "face_quality.csv",
        qc.get("face_quality", []),
        fieldnames=[
            "mesh_label",
            "slice_left",
            "slice_right",
            "match_id",
            "face_local_id",
            "max_edge",
            "aspect_ratio",
            "area",
            "status",
            "reason",
        ],
    )
    _write_csv_rows(
        output_dir / "open_edges.csv",
        qc.get("open_edges", []),
        fieldnames=[
            "mesh_label",
            "slice_left",
            "slice_right",
            "match_id",
            "edge_start_x",
            "edge_start_y",
            "edge_start_z",
            "edge_end_x",
            "edge_end_y",
            "edge_end_z",
            "source",
            "reason",
        ],
    )


def arc_graph_surface_patches(
    boundaries: Sequence[BoundarySlice],
    contours: Sequence[Contour2D],
    resample_points: int = 80,
    spacing: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, SurfaceMesh], Dict[str, List[Dict]]]:
    del contours, resample_points  # Current v0.1 builds from propagated boundary arcs.
    spacing = np.asarray(spacing if spacing is not None else np.ones(3), dtype=float)
    arcs_by_slice = build_labeled_arcs(boundaries, spacing)
    all_arcs = [arc for arcs in arcs_by_slice.values() for arc in arcs]
    arc_by_id = {arc.arc_id: arc for arc in all_arcs}
    mesh_builders = {
        BOUNDARY_OUTER: _MeshBuilder("outer"),
        BOUNDARY_INNER: _MeshBuilder("inner"),
        BOUNDARY_LATERAL: _MeshBuilder("lateral"),
    }
    all_match_rows: List[Dict] = []
    topology_events: List[TopologyEvent] = []
    face_quality_rows: List[Dict] = []
    open_edge_rows: List[Dict] = []

    slice_ids = sorted(arcs_by_slice)
    for slice_left, slice_right in zip(slice_ids[:-1], slice_ids[1:]):
        left_arcs = arcs_by_slice.get(slice_left, [])
        right_arcs = arcs_by_slice.get(slice_right, [])
        if slice_right - slice_left != 1:
            for arc in left_arcs + right_arcs:
                open_edge_rows.append(
                    _open_edge_row_from_arc(
                        arc,
                        slice_left,
                        slice_right,
                        0.0,
                        1.0,
                        "gap_between_slices",
                        "non-adjacent boundary slices",
                    )
                )
            continue

        candidates = generate_arc_match_candidates(left_arcs, right_arcs, spacing)
        scored = [score_arc_match(left, right, spacing) for left, right in candidates]
        accepted, events = solve_arc_assignment(left_arcs, right_arcs, scored)
        topology_events.extend(events)
        usage = ArcUsageMap(left_arcs + right_arcs)
        for match in accepted:
            usage.add_used_range(match.prev_arc_id, match.prev_fraction_start, match.prev_fraction_end)
            usage.add_used_range(match.next_arc_id, match.next_fraction_start, match.next_fraction_end)
        range_events, range_edges = emit_unmatched_range_events(
            usage,
            left_arcs,
            right_arcs,
            slice_left,
            slice_right,
        )
        topology_events.extend(range_events)
        open_edge_rows.extend(range_edges)
        all_match_rows.extend(_match_to_row(match, slice_left, slice_right) for match in scored)

        for match in accepted:
            mesh_name = _label_mesh_name(match.label)
            if mesh_name is None:
                continue
            left_arc = arc_by_id[match.prev_arc_id]
            right_arc = arc_by_id[match.next_arc_id]
            left_points = extract_polyline_range(
                left_arc.points,
                match.prev_fraction_start,
                match.prev_fraction_end,
                closed=left_arc.is_closed,
            )
            right_points = extract_polyline_range(
                right_arc.points,
                match.next_fraction_start,
                match.next_fraction_end,
                closed=right_arc.is_closed,
            )
            if match.direction < 0:
                right_points = right_points[::-1]
            vertices, faces = _build_arc_strip(left_points, right_points)
            match_id = f"{match.prev_arc_id}->{match.next_arc_id}"
            faces, quality_rows, rejected_open_edges = filter_mesh_faces_by_quality(
                vertices,
                faces,
                spacing,
                mesh_name,
                slice_left,
                slice_right,
                match_id,
            )
            face_quality_rows.extend(quality_rows)
            open_edge_rows.extend(rejected_open_edges)
            mesh_builders[match.label].add_patch(vertices, faces)

    meshes: Dict[str, SurfaceMesh] = {}
    for label, builder in mesh_builders.items():
        if builder.has_faces():
            mesh = builder.to_mesh()
            if len(mesh.faces) > 0:
                meshes[builder.name] = mesh

    if "outer" not in meshes or "inner" not in meshes:
        missing = ", ".join(name for name in ("outer", "inner") if name not in meshes)
        raise ValueError(f"Arc-graph surface build produced no {missing} mesh")

    qc = {
        "arc_matches": all_match_rows,
        "topology_events": [_event_to_row(event) for event in topology_events],
        "face_quality": face_quality_rows,
        "open_edges": open_edge_rows,
    }
    return meshes, qc


def contour_shell_surface_patches(
    boundaries: Sequence[BoundarySlice],
    contours: Sequence[Contour2D],
    resample_points: int = 80,
    max_bridge_distance: float = 30.0,
) -> Dict[str, SurfaceMesh]:
    """Build outer/inner/lateral patches on the real mask contour shell.

    This keeps the final surfaces on the extracted region boundary instead of
    spanning large topology jumps with free-floating planes.
    """

    ring_points = min(240, max(96, int(resample_points) * 2))
    contour_by_key = {(contour.slice_index, contour.contour_id): contour for contour in contours}
    contours_by_index = contours_by_slice(contours)
    endpoint_slices = {
        min(contour.slice_index for contour in contours),
        max(contour.slice_index for contour in contours),
    }

    entries: List[Tuple[BoundarySlice, np.ndarray, np.ndarray]] = []
    previous_ring: Optional[np.ndarray] = None
    for boundary in sorted(boundaries, key=lambda item: item.slice_index):
        contour = contour_by_key.get((boundary.slice_index, boundary.contour_id))
        if contour is None:
            candidates = contours_by_index.get(boundary.slice_index, [])
            if not candidates:
                continue
            contour = max(candidates, key=lambda item: item.area)

        start_target = _boundary_ring_start_target(boundary, previous_ring)
        ring = _resampled_contour_ring(contour.points, ring_points, start_target, previous_ring)
        labels = _label_mask_shell_ring(boundary, ring)
        entries.append((boundary, ring, labels))
        previous_ring = ring

    if len(entries) < 2:
        raise ValueError("Need at least two mask contour rings to build contour-shell surfaces")

    base_vertices = np.vstack([entry[1] for entry in entries]).astype(float)
    vertex_labels = np.concatenate([entry[2] for entry in entries]).astype(np.uint8)
    extra_vertices: List[np.ndarray] = []
    extra_labels: List[int] = []

    faces_by_label: Dict[int, List[List[int]]] = {
        BOUNDARY_OUTER: [],
        BOUNDARY_INNER: [],
        BOUNDARY_LATERAL: [],
    }

    def add_vertex(point: np.ndarray, label: int) -> int:
        extra_vertices.append(np.asarray(point, dtype=float))
        extra_labels.append(int(label))
        return len(base_vertices) + len(extra_vertices) - 1

    def vertex_label(index: int) -> int:
        if index < len(vertex_labels):
            return int(vertex_labels[index])
        return int(extra_labels[index - len(vertex_labels)])

    def add_face(face: Sequence[int], label: Optional[int] = None) -> None:
        if label is None:
            label = _majority_surface_label([vertex_label(index) for index in face])
        if label in faces_by_label:
            faces_by_label[int(label)].append([int(index) for index in face])

    def add_stable_triangle(face: Sequence[int], prefer_real_surface: bool = False) -> None:
        labels = [vertex_label(index) for index in face]
        label = (
            _real_surface_preferred_label(labels)
            if prefer_real_surface
            else _majority_surface_label(labels)
        )
        if label not in faces_by_label:
            return
        add_face(face, label)

    def add_bridge_faces(
        left_indices: Sequence[int],
        right_indices: Sequence[int],
        prefer_real_surface: bool,
    ) -> None:
        for point_index in range(ring_points):
            next_point = (point_index + 1) % ring_points
            a = int(left_indices[point_index])
            b = int(left_indices[next_point])
            c = int(right_indices[point_index])
            d = int(right_indices[next_point])
            add_stable_triangle([a, c, b], prefer_real_surface=prefer_real_surface)
            add_stable_triangle([b, c, d], prefer_real_surface=prefer_real_surface)

    def add_intermediate_ring(
        left_ring: np.ndarray,
        right_ring: np.ndarray,
        left_labels: np.ndarray,
        right_labels: np.ndarray,
        t: float,
    ) -> List[int]:
        ring = (1.0 - t) * left_ring + t * right_ring
        labels = np.where(t < 0.5, left_labels, right_labels)
        return [add_vertex(point, int(label)) for point, label in zip(ring, labels)]

    def endpoint_cap_label(boundary: BoundarySlice) -> Optional[int]:
        if boundary.slice_index not in endpoint_slices:
            return None
        mode = normalize_surface_mode(boundary.surface_mode)
        if mode == SURFACE_MODE_OUTER_ONLY:
            return BOUNDARY_OUTER
        if mode == SURFACE_MODE_INNER_ONLY:
            return BOUNDARY_INNER
        return None

    def add_endpoint_cap(
        entry: Tuple[BoundarySlice, np.ndarray, np.ndarray],
        base: int,
        reverse: bool,
    ) -> None:
        boundary, ring, _labels = entry
        label = endpoint_cap_label(boundary)
        if label is None:
            return
        center_index = add_vertex(np.asarray(ring, dtype=float).mean(axis=0), label)
        for point_index in range(ring_points):
            next_point = (point_index + 1) % ring_points
            a = base + point_index
            b = base + next_point
            if reverse:
                add_face([center_index, b, a], label)
            else:
                add_face([center_index, a, b], label)

    for entry_index, (left, right) in enumerate(zip(entries[:-1], entries[1:])):
        left_boundary, left_ring, left_labels = left
        right_boundary, right_ring, right_labels = right
        if right_boundary.slice_index - left_boundary.slice_index > 1:
            continue
        base = entry_index * ring_points
        next_base = (entry_index + 1) * ring_points
        left_indices = list(range(base, base + ring_points))
        right_indices = list(range(next_base, next_base + ring_points))
        max_distance = float(np.linalg.norm(left_ring - right_ring, axis=1).max())
        mean_distance = float(np.linalg.norm(left_ring - right_ring, axis=1).mean())
        centroid_distance = float(np.linalg.norm(left_ring.mean(axis=0) - right_ring.mean(axis=0)))
        prefer_real_surface = (
            mean_distance > 30.0
            and centroid_distance > 25.0
            and np.any(left_labels != right_labels)
        )
        bridge_steps = max(1, int(math.ceil(max_distance / max(1.0, max_bridge_distance))))
        bridge_rings: List[List[int]] = [left_indices]
        for step in range(1, bridge_steps):
            bridge_rings.append(
                add_intermediate_ring(
                    left_ring,
                    right_ring,
                    left_labels,
                    right_labels,
                    step / float(bridge_steps),
                )
            )
        bridge_rings.append(right_indices)
        for current, following in zip(bridge_rings[:-1], bridge_rings[1:]):
            add_bridge_faces(current, following, prefer_real_surface=prefer_real_surface)

    add_endpoint_cap(entries[0], 0, reverse=True)
    last_base = (len(entries) - 1) * ring_points
    add_endpoint_cap(entries[-1], last_base, reverse=False)

    vertices = (
        np.vstack([base_vertices, np.vstack(extra_vertices)]).astype(float)
        if extra_vertices
        else base_vertices
    )
    meshes: Dict[str, SurfaceMesh] = {}
    for label, name in (
        (BOUNDARY_OUTER, "outer"),
        (BOUNDARY_INNER, "inner"),
        (BOUNDARY_LATERAL, "lateral"),
    ):
        faces = faces_by_label[label]
        if faces:
            meshes[name] = _compact_surface_mesh(name, vertices, np.asarray(faces, dtype=np.int32))

    if "outer" not in meshes or "inner" not in meshes:
        missing = ", ".join(name for name in ("outer", "inner") if name not in meshes)
        raise ValueError(f"Contour-shell surface build produced no {missing} mesh")
    return meshes


def _boundary_ring_start_target(
    boundary: BoundarySlice,
    previous_ring: Optional[np.ndarray],
) -> np.ndarray:
    if previous_ring is not None and len(previous_ring) > 0:
        return previous_ring[0]
    for curve in (boundary.outer_arc, boundary.inner_arc):
        if len(curve) > 0:
            return np.asarray(curve[0], dtype=float)
    return np.zeros(3, dtype=float)


def _resampled_contour_ring(
    contour_points: np.ndarray,
    ring_points: int,
    start_target: np.ndarray,
    previous_ring: Optional[np.ndarray],
) -> np.ndarray:
    points = _normalize_contour(contour_points)
    if len(points) < 3:
        raise ValueError("Mask contour needs at least three points")

    candidates = [
        _sample_contour_ring(points, ring_points, start_target),
        _sample_contour_ring(points[::-1], ring_points, start_target),
    ]
    if previous_ring is None:
        return candidates[0]
    scores = [_mean_curve_distance(previous_ring, candidate) for candidate in candidates]
    return candidates[int(np.argmin(scores))]


def _sample_contour_ring(points: np.ndarray, ring_points: int, start_target: np.ndarray) -> np.ndarray:
    closed = _align_closed_polyline_start(_closed_contour_points(points), start_target)
    return resample_polyline(closed, ring_points + 1)[:-1]


def _label_mask_shell_ring(boundary: BoundarySlice, ring: np.ndarray) -> np.ndarray:
    mode = normalize_surface_mode(boundary.surface_mode)
    if mode == SURFACE_MODE_OUTER_ONLY:
        return np.full(len(ring), BOUNDARY_OUTER, dtype=np.uint8)
    if mode == SURFACE_MODE_INNER_ONLY:
        return np.full(len(ring), BOUNDARY_INNER, dtype=np.uint8)

    candidates: List[Tuple[int, np.ndarray]] = []
    if len(boundary.outer_arc) > 0:
        candidates.append((BOUNDARY_OUTER, _surface_label_points([boundary.outer_arc])))
    if len(boundary.inner_arc) > 0:
        candidates.append((BOUNDARY_INNER, _surface_label_points([boundary.inner_arc])))
    lateral_points = _surface_label_points(boundary.lateral_arcs)
    if len(lateral_points) > 0:
        candidates.append((BOUNDARY_LATERAL, lateral_points))
    if not candidates:
        return np.full(len(ring), BOUNDARY_LATERAL, dtype=np.uint8)

    distances = np.full((len(candidates), len(ring)), np.inf, dtype=float)
    for candidate_index, (_label, points) in enumerate(candidates):
        if len(points) == 0:
            continue
        tree = cKDTree(np.asarray(points, dtype=float))
        distances[candidate_index], _ = tree.query(ring, k=1)
    best = np.argmin(distances, axis=0)
    return np.asarray([candidates[index][0] for index in best], dtype=np.uint8)


def _surface_label_points(arcs: Sequence[np.ndarray]) -> np.ndarray:
    points: List[np.ndarray] = []
    for arc in arcs:
        arc = np.asarray(arc, dtype=float)
        if len(arc) < 2:
            continue
        sample_count = max(8, min(160, int(math.ceil(_polyline_length(arc) / 2.0))))
        points.append(resample_polyline(arc, sample_count))
    if not points:
        return np.empty((0, 3), dtype=float)
    return np.vstack(points)


def _majority_surface_label(labels: Sequence[int]) -> int:
    counts = {
        BOUNDARY_OUTER: 0,
        BOUNDARY_INNER: 0,
        BOUNDARY_LATERAL: 0,
    }
    for label in labels:
        if int(label) in counts:
            counts[int(label)] += 1
    return max(counts, key=lambda label: (counts[label], label == BOUNDARY_LATERAL))


def _real_surface_preferred_label(labels: Sequence[int]) -> int:
    counts = {
        BOUNDARY_OUTER: 0,
        BOUNDARY_INNER: 0,
        BOUNDARY_LATERAL: 0,
    }
    for label in labels:
        if int(label) in counts:
            counts[int(label)] += 1

    real_counts = {
        label: count
        for label, count in counts.items()
        if label != BOUNDARY_LATERAL and count > 0
    }
    if real_counts:
        return max(real_counts, key=lambda label: (real_counts[label], label == BOUNDARY_OUTER))
    return BOUNDARY_LATERAL


def _compact_surface_mesh(name: str, vertices: np.ndarray, faces: np.ndarray) -> SurfaceMesh:
    used = np.unique(faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return SurfaceMesh(
        name=name,
        vertices=vertices[used],
        faces=remap[faces].astype(np.int32),
    )


def _paint_points(volume: np.ndarray, points: np.ndarray, value: int) -> None:
    if len(points) == 0:
        return
    rounded = np.rint(points).astype(int)
    shape = np.asarray(volume.shape)
    valid = np.all((rounded >= 0) & (rounded < shape), axis=1)
    rounded = rounded[valid]
    if len(rounded) == 0:
        return
    volume[rounded[:, 0], rounded[:, 1], rounded[:, 2]] = value


def make_boundary_label_volume(
    mask: np.ndarray,
    boundaries: Sequence[BoundarySlice],
    dilation_iterations: int = 1,
) -> np.ndarray:
    labels = np.zeros(mask.shape, dtype=np.uint8)
    region = _as_bool_mask(mask)
    labels[region] = BOUNDARY_REGION

    outer = np.zeros(mask.shape, dtype=bool)
    inner = np.zeros(mask.shape, dtype=bool)
    lateral = np.zeros(mask.shape, dtype=bool)
    for boundary in boundaries:
        _paint_points(outer, boundary.outer_arc, True)
        _paint_points(inner, boundary.inner_arc, True)
        for arc in boundary.lateral_arcs:
            _paint_points(lateral, arc, True)

    if dilation_iterations > 0:
        structure = ndimage.generate_binary_structure(3, 1)
        outer = ndimage.binary_dilation(outer, structure=structure, iterations=dilation_iterations)
        inner = ndimage.binary_dilation(inner, structure=structure, iterations=dilation_iterations)
        lateral = ndimage.binary_dilation(lateral, structure=structure, iterations=dilation_iterations)

    labels[region & lateral] = BOUNDARY_LATERAL
    labels[region & outer] = BOUNDARY_OUTER
    labels[region & inner] = BOUNDARY_INNER
    return labels


def _distance_depth(mask: np.ndarray, labels: np.ndarray) -> np.ndarray:
    region = _as_bool_mask(mask)
    outer = labels == BOUNDARY_OUTER
    inner = labels == BOUNDARY_INNER
    if not np.any(outer) or not np.any(inner):
        raise ValueError("Need both outer and inner boundary labels to compute depth")

    distance_outer = ndimage.distance_transform_edt(~outer)
    distance_inner = ndimage.distance_transform_edt(~inner)
    denom = distance_outer + distance_inner
    depth = np.full(mask.shape, np.nan, dtype=np.float32)
    valid = region & (denom > 0)
    depth[valid] = (distance_outer[valid] / denom[valid]).astype(np.float32)
    depth[outer & region] = 0.0
    depth[inner & region] = 1.0
    return depth


def _solve_laplace_depth(
    mask: np.ndarray,
    labels: np.ndarray,
    max_iterations: int = 2000,
    tolerance: float = 1e-5,
) -> np.ndarray:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    region = _as_bool_mask(mask)
    outer = region & (labels == BOUNDARY_OUTER)
    inner = region & (labels == BOUNDARY_INNER)
    fixed = outer | inner
    unknown = region & ~fixed

    depth = np.full(mask.shape, np.nan, dtype=np.float32)
    depth[outer] = 0.0
    depth[inner] = 1.0

    coords = np.argwhere(unknown)
    if len(coords) == 0:
        return depth

    index = np.full(mask.shape, -1, dtype=np.int64)
    index[unknown] = np.arange(len(coords), dtype=np.int64)

    rows: List[int] = []
    cols: List[int] = []
    values: List[float] = []
    rhs = np.zeros(len(coords), dtype=np.float64)
    shape = np.asarray(mask.shape)
    offsets = np.asarray(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ],
        dtype=int,
    )

    for row_index, coord in enumerate(coords):
        diagonal = 0.0
        for offset in offsets:
            neighbor = coord + offset
            if np.any(neighbor < 0) or np.any(neighbor >= shape):
                continue
            n_tuple = tuple(neighbor)
            if not region[n_tuple]:
                # Outside the region is treated as a side wall: no fixed depth
                # value leaks in from the lateral boundary.
                continue

            diagonal += 1.0
            if outer[n_tuple]:
                rhs[row_index] += 0.0
            elif inner[n_tuple]:
                rhs[row_index] += 1.0
            else:
                neighbor_index = index[n_tuple]
                if neighbor_index >= 0:
                    rows.append(row_index)
                    cols.append(int(neighbor_index))
                    values.append(-1.0)

        if diagonal == 0.0:
            diagonal = 1.0
            rhs[row_index] = 0.5
        rows.append(row_index)
        cols.append(row_index)
        values.append(diagonal)

    matrix = sp.csr_matrix((values, (rows, cols)), shape=(len(coords), len(coords)))
    try:
        solution, info = spla.cg(matrix, rhs, maxiter=max_iterations, rtol=tolerance)
    except TypeError:
        solution, info = spla.cg(matrix, rhs, maxiter=max_iterations, tol=tolerance)
    if info != 0:
        warnings.warn(
            f"Laplace solver did not fully converge (info={info}); using best available result.",
            RuntimeWarning,
        )

    depth[unknown] = np.clip(solution, 0.0, 1.0).astype(np.float32)
    return depth


def compute_laminar_depth(
    mask: np.ndarray,
    labels: np.ndarray,
    method: str = "auto",
    max_laplace_voxels: int = 250_000,
    laplace_iterations: int = 2000,
    laplace_tolerance: float = 1e-5,
) -> np.ndarray:
    """Compute laminar depth where outer=0 and inner=1.

    `laplace` treats non-outer/non-inner mask boundaries as side walls by not
    assigning them fixed values. For very large masks, `auto` falls back to a
    distance-ratio field so the pipeline still finishes.
    """

    region_voxels = int(np.count_nonzero(mask))
    method = method.lower()
    if method not in ("auto", "laplace", "distance"):
        raise ValueError("method must be auto, laplace, or distance")

    if method == "distance":
        return _distance_depth(mask, labels)

    if method == "auto" and region_voxels > max_laplace_voxels:
        return _distance_depth(mask, labels)

    try:
        return _solve_laplace_depth(
            mask,
            labels,
            max_iterations=laplace_iterations,
            tolerance=laplace_tolerance,
        )
    except Exception:
        if method == "laplace":
            raise
        warnings.warn("Laplace depth failed; falling back to distance-ratio depth.", RuntimeWarning)
        return _distance_depth(mask, labels)


def _surface_only_build_requested(depth_method: str) -> bool:
    text = str(depth_method).strip().lower().replace("_", " ").replace("-", " ")
    return text in ("surface only", "surfaces only", "surface", "surfaces")


def compute_layer_normals(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a normalized vector field pointing from outer to inner."""

    filled = np.nan_to_num(depth, nan=0.0)
    gradients = np.stack(np.gradient(filled), axis=-1).astype(np.float32)
    norms = np.linalg.norm(gradients, axis=-1)
    normals = np.zeros(depth.shape + (3,), dtype=np.float32)
    valid = _as_bool_mask(mask) & (norms > 1e-8)
    normals[valid] = gradients[valid] / norms[valid, None]
    return normals


def _distance_to_label(labels: np.ndarray, label_value: int) -> np.ndarray:
    source = labels == label_value
    if not np.any(source):
        return np.full(labels.shape, np.nan, dtype=np.float32)
    return ndimage.distance_transform_edt(~source).astype(np.float32)


def _detect_xyz_columns(columns: Iterable[str], prefix: str = "") -> Tuple[str, str, str]:
    columns = list(columns)
    candidates = [
        (f"{prefix}x", f"{prefix}y", f"{prefix}z"),
        (f"{prefix}X", f"{prefix}Y", f"{prefix}Z"),
        ("soma_x", "soma_y", "soma_z"),
        ("x", "y", "z"),
        ("X", "Y", "Z"),
    ]
    for trio in candidates:
        if all(name in columns for name in trio):
            return trio
    raise ValueError(f"Could not detect x/y/z columns from: {columns}")


def _sample_nearest(volume: np.ndarray, points: np.ndarray) -> np.ndarray:
    rounded = np.rint(points).astype(int)
    shape = np.asarray(volume.shape)
    valid = np.all((rounded >= 0) & (rounded < shape), axis=1)
    out = np.full(len(points), np.nan, dtype=float)
    if np.any(valid):
        coords = rounded[valid]
        out[valid] = volume[coords[:, 0], coords[:, 1], coords[:, 2]]
    return out


def write_cell_depth_table(
    cell_csv: str | Path,
    output_csv: str | Path,
    depth: np.ndarray,
    normals: np.ndarray,
    labels: np.ndarray,
) -> List[Dict]:
    with Path(cell_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        source_columns = list(reader.fieldnames or [])

    if not rows:
        _write_csv_rows(output_csv, [], fieldnames=source_columns)
        return []

    x_col, y_col, z_col = _detect_xyz_columns(source_columns)
    points = np.asarray(
        [[float(row[x_col]), float(row[y_col]), float(row[z_col])] for row in rows],
        dtype=float,
    )
    distance_outer = _distance_to_label(labels, BOUNDARY_OUTER)
    distance_inner = _distance_to_label(labels, BOUNDARY_INNER)

    laminar_depth = _sample_nearest(depth, points)
    distance_to_outer = _sample_nearest(distance_outer, points)
    distance_to_inner = _sample_nearest(distance_inner, points)
    for row, value in zip(rows, laminar_depth):
        row["laminar_depth"] = value
        row["depth_confidence"] = 1.0 if np.isfinite(value) else 0.0
    for row, value in zip(rows, distance_to_outer):
        row["distance_to_outer"] = value
    for row, value in zip(rows, distance_to_inner):
        row["distance_to_inner"] = value
    for dim, name in enumerate(("layer_normal_x", "layer_normal_y", "layer_normal_z")):
        for row, value in zip(rows, _sample_nearest(normals[..., dim], points)):
            row[name] = value

    fieldnames = source_columns + [
        "laminar_depth",
        "depth_confidence",
        "distance_to_outer",
        "distance_to_inner",
        "layer_normal_x",
        "layer_normal_y",
        "layer_normal_z",
    ]
    _write_csv_rows(output_csv, rows, fieldnames=fieldnames)
    return rows


def _read_swc_array(path: str | Path) -> np.ndarray:
    rows: List[List[float]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            rows.append([float(part) for part in parts[:7]])
    if not rows:
        return np.empty((0, 7), dtype=float)
    return np.asarray(rows, dtype=float)


def summarize_dendrite_depth(
    swc_paths: Sequence[str | Path],
    depth: np.ndarray,
    normals: np.ndarray,
    output_csv: str | Path,
    dendrite_types: Sequence[int] = (3, 4),
) -> List[Dict]:
    rows: List[Dict] = []
    for swc_path in swc_paths:
        swc = _read_swc_array(swc_path)
        if len(swc) == 0:
            continue
        point_by_id = {int(row[0]): row for row in swc}
        soma_rows = swc[swc[:, 1] == 1]
        soma = soma_rows[0, 2:5] if len(soma_rows) else swc[0, 2:5]

        total_length = 0.0
        weighted_depth = 0.0
        superficial_length = 0.0
        deep_length = 0.0
        depth_values: List[float] = []
        polarity_num = 0.0
        polarity_den = 0.0

        for row in swc:
            point_type = int(row[1])
            parent_id = int(row[6])
            if point_type not in dendrite_types or parent_id < 0 or parent_id not in point_by_id:
                continue
            parent = point_by_id[parent_id]
            if int(parent[1]) not in dendrite_types and int(parent[1]) != 1:
                continue

            p0 = parent[2:5]
            p1 = row[2:5]
            segment = p1 - p0
            length = float(np.linalg.norm(segment))
            if length <= 0:
                continue
            midpoint = (p0 + p1) * 0.5
            sampled_depth = float(_sample_nearest(depth, midpoint[None, :])[0])
            if not np.isfinite(sampled_depth):
                continue

            total_length += length
            weighted_depth += sampled_depth * length
            depth_values.append(sampled_depth)
            if sampled_depth < 1.0 / 3.0:
                superficial_length += length
            if sampled_depth > 2.0 / 3.0:
                deep_length += length

            local_normal = np.array(
                [
                    _sample_nearest(normals[..., 0], midpoint[None, :])[0],
                    _sample_nearest(normals[..., 1], midpoint[None, :])[0],
                    _sample_nearest(normals[..., 2], midpoint[None, :])[0],
                ],
                dtype=float,
            )
            normal_norm = np.linalg.norm(local_normal)
            soma_vector = p1 - soma
            soma_vector_norm = np.linalg.norm(soma_vector)
            if normal_norm > 0 and soma_vector_norm > 0:
                polarity_num += float(np.dot(soma_vector / soma_vector_norm, local_normal / normal_norm)) * length
                polarity_den += length

        mean_depth = weighted_depth / total_length if total_length > 0 else np.nan
        rows.append(
            {
                "cell_id": Path(swc_path).stem,
                "swc_path": str(swc_path),
                "dendrite_total_length": total_length,
                "mean_dendrite_depth": mean_depth,
                "superficial_dendrite_fraction": superficial_length / total_length
                if total_length > 0
                else np.nan,
                "deep_dendrite_fraction": deep_length / total_length if total_length > 0 else np.nan,
                "dendrite_depth_span": float(np.nanmax(depth_values) - np.nanmin(depth_values))
                if depth_values
                else np.nan,
                "dendrite_polarity_along_layer_normal": polarity_num / polarity_den
                if polarity_den > 0
                else np.nan,
            }
        )

    fieldnames = [
        "cell_id",
        "swc_path",
        "dendrite_total_length",
        "mean_dendrite_depth",
        "superficial_dendrite_fraction",
        "deep_dendrite_fraction",
        "dendrite_depth_span",
        "dendrite_polarity_along_layer_normal",
    ]
    _write_csv_rows(output_csv, rows, fieldnames=fieldnames)
    return rows


def write_qc_tables(boundaries: Sequence[BoundarySlice], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = [boundary.to_summary_row() for boundary in boundaries]
    _write_csv_rows(output_dir / "qc_surface_distance.csv", summary_rows)
    jump_rows = surface_jump_diagnostics(boundaries)
    if jump_rows:
        _write_csv_rows(
            output_dir / "surface_jump_diagnostics.csv",
            jump_rows,
            fieldnames=[
                "arc",
                "left_slice",
                "right_slice",
                "slice_gap",
                "left_mode",
                "right_mode",
                "left_source",
                "right_source",
                "mean_point_jump",
                "max_point_jump",
                "centroid_jump",
                "mean_reversed_jump",
                "left_first_last_distance",
                "right_first_last_distance",
                "flags",
            ],
        )

    benign_flags = {
        "no_lateral_boundary",
        SURFACE_MODE_OUTER_ONLY,
        SURFACE_MODE_INNER_ONLY,
    }

    def has_actionable_flags(row: Dict) -> bool:
        flags = [
            flag.strip()
            for flag in str(row.get("flags") or "").split(";")
            if flag.strip()
        ]
        return any(flag not in benign_flags for flag in flags)

    uncertain_rows = [
        row
        for row in summary_rows
        if float(row.get("confidence") or 0.0) < 0.6 or has_actionable_flags(row)
    ]
    _write_csv_rows(output_dir / "qc_uncertain_slices.csv", uncertain_rows)


def surface_jump_diagnostics(boundaries: Sequence[BoundarySlice]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for arc_name in ("outer", "inner"):
        entries = []
        for boundary in sorted(boundaries, key=lambda item: item.slice_index):
            curve = boundary.outer_arc if arc_name == "outer" else boundary.inner_arc
            if len(curve) >= 2:
                entries.append((boundary, np.asarray(curve, dtype=float)))

        for (left_boundary, left_curve), (right_boundary, right_curve) in zip(entries[:-1], entries[1:]):
            count = min(len(left_curve), len(right_curve))
            if count < 2:
                continue
            left_curve = resample_polyline(left_curve, count)
            right_curve = resample_polyline(right_curve, count)
            point_distances = np.linalg.norm(left_curve - right_curve, axis=1)
            reversed_distances = np.linalg.norm(left_curve - right_curve[::-1], axis=1)
            centroid_jump = float(np.linalg.norm(left_curve.mean(axis=0) - right_curve.mean(axis=0)))
            slice_gap = int(right_boundary.slice_index - left_boundary.slice_index)
            flags = []
            has_large_jump = float(point_distances.mean()) > 30.0 and centroid_jump > 25.0
            if has_large_jump:
                flags.append("large_curve_jump")
            if slice_gap > 1:
                flags.append("slice_gap")
            if normalize_surface_mode(left_boundary.surface_mode) != normalize_surface_mode(right_boundary.surface_mode):
                flags.append("mode_change")
            if _is_closed_polyline(left_curve) != _is_closed_polyline(right_curve):
                flags.append("open_closed_change")
            if flags:
                flags.append("transition_review")
            if has_large_jump and left_boundary.source == "manual" and right_boundary.source == "manual":
                flags.append("manual_topology_jump")
            elif has_large_jump:
                flags.append("auto_transition")

            rows.append(
                {
                    "arc": arc_name,
                    "left_slice": int(left_boundary.slice_index),
                    "right_slice": int(right_boundary.slice_index),
                    "slice_gap": slice_gap,
                    "left_mode": normalize_surface_mode(left_boundary.surface_mode),
                    "right_mode": normalize_surface_mode(right_boundary.surface_mode),
                    "left_source": left_boundary.source,
                    "right_source": right_boundary.source,
                    "mean_point_jump": round(float(point_distances.mean()), 4),
                    "max_point_jump": round(float(point_distances.max()), 4),
                    "centroid_jump": round(centroid_jump, 4),
                    "mean_reversed_jump": round(float(reversed_distances.mean()), 4),
                    "left_first_last_distance": round(float(np.linalg.norm(left_curve[0] - left_curve[-1])), 4),
                    "right_first_last_distance": round(float(np.linalg.norm(right_curve[0] - right_curve[-1])), 4),
                    "flags": ";".join(flags),
                }
            )
    return rows


def _image_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        high = low + 1.0
    return np.clip((image - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


def _draw_plane_polyline(draw, points: np.ndarray, color: Tuple[int, int, int], width: int = 1) -> None:
    if len(points) < 2:
        return
    xy = [(float(x), float(y)) for x, y in np.asarray(points, dtype=float)]
    draw.line(xy, fill=color, width=width)


def write_qc_slice_overlays(
    output_dir: str | Path,
    boundaries: Sequence[BoundarySlice],
    mask: np.ndarray,
    template: Optional[np.ndarray] = None,
    slice_axis: int | str = 0,
    every: int = 10,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slice_axis = _slice_axis_to_int(slice_axis)

    selected = [
        boundary
        for index, boundary in enumerate(sorted(boundaries, key=lambda item: item.slice_index))
        if index % max(1, every) == 0 or boundary.source == "manual" or boundary.flags
    ]

    for boundary in selected:
        from PIL import Image, ImageDraw

        image = (
            _take_slice(template, boundary.slice_index, slice_axis)
            if template is not None
            else _take_slice(mask, boundary.slice_index, slice_axis)
        )
        base = _image_to_uint8(image)
        rgb = np.repeat(base[:, :, None], 3, axis=2)
        canvas = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(canvas)

        mask_slice = _take_slice(mask, boundary.slice_index, slice_axis)
        for contour in _find_mask_contours(mask_slice):
            if len(contour) >= 2:
                closed = np.vstack([contour, contour[0]])
                _draw_plane_polyline(draw, closed, (255, 255, 255), width=1)

        for arc, color, label in (
            (boundary.outer_arc, (31, 119, 180), "outer"),
            (boundary.inner_arc, (214, 39, 40), "inner"),
        ):
            plane = _volume_to_plane_points(arc, slice_axis)
            _draw_plane_polyline(draw, plane, color, width=3)
        for arc in boundary.lateral_arcs:
            plane = _volume_to_plane_points(arc, slice_axis)
            _draw_plane_polyline(draw, plane, (127, 127, 127), width=2)

        title = (
            f"slice {boundary.slice_index} | {boundary.source} | "
            f"confidence {boundary.confidence:.2f}"
        )
        if boundary.flags:
            title += " | " + ";".join(boundary.flags[:3])
        draw.rectangle((4, 4, min(canvas.width - 1, 10 + len(title) * 7), 22), fill=(0, 0, 0))
        draw.text((8, 7), title, fill=(255, 255, 255))
        canvas.save(output_dir / f"slice_{boundary.slice_index:04d}.png")


def write_manual_landmark_template(
    contours: Sequence[Contour2D],
    output_csv: str | Path,
    every: int = 10,
) -> List[Dict]:
    """Write an editable CSV template for endpoint-style annotation."""

    fieldnames = [
        "slice_index",
        "contour_id",
        "surface_mode",
        "outer_start_index",
        "outer_start_x",
        "outer_start_y",
        "outer_start_z",
        "outer_end_index",
        "outer_end_x",
        "outer_end_y",
        "outer_end_z",
        "outer_path",
        "inner_start_index",
        "inner_start_x",
        "inner_start_y",
        "inner_start_z",
        "inner_end_index",
        "inner_end_x",
        "inner_end_y",
        "inner_end_z",
        "inner_path",
        "note",
    ]
    rows = []
    for index, contour in enumerate(sorted(contours, key=lambda item: item.slice_index)):
        if index % max(1, every) != 0:
            continue
        rows.append(
            {
                "slice_index": contour.slice_index,
                "contour_id": contour.contour_id,
                "surface_mode": SURFACE_MODE_NORMAL,
                "outer_start_index": "",
                "outer_start_x": "",
                "outer_start_y": "",
                "outer_start_z": "",
                "outer_end_index": "",
                "outer_end_x": "",
                "outer_end_y": "",
                "outer_end_z": "",
                "outer_path": "auto",
                "inner_start_index": "",
                "inner_start_x": "",
                "inner_start_y": "",
                "inner_start_z": "",
                "inner_end_index": "",
                "inner_end_x": "",
                "inner_end_y": "",
                "inner_end_z": "",
                "inner_path": "auto",
                "note": "",
            }
        )
    _write_csv_rows(output_csv, rows, fieldnames=fieldnames)
    return rows


def write_contour_index_table(contours: Sequence[Contour2D], output_csv: str | Path) -> List[Dict]:
    fieldnames = ["slice_index", "contour_id", "area", "length", "point_count"]
    rows = [
        {
            "slice_index": contour.slice_index,
            "contour_id": contour.contour_id,
            "area": contour.area,
            "length": contour.length,
            "point_count": len(contour.points),
        }
        for contour in contours
    ]
    _write_csv_rows(output_csv, rows, fieldnames=fieldnames)
    return rows


def write_contour_points_table(contours: Sequence[Contour2D], output_csv: str | Path) -> List[Dict]:
    fieldnames = ["slice_index", "contour_id", "point_index", "x", "y", "z"]
    rows = []
    for contour in contours:
        for point_index, point in enumerate(_normalize_contour(contour.points)):
            rows.append(
                {
                    "slice_index": contour.slice_index,
                    "contour_id": contour.contour_id,
                    "point_index": point_index,
                    "x": point[0],
                    "y": point[1],
                    "z": point[2],
                }
            )
    _write_csv_rows(output_csv, rows, fieldnames=fieldnames)
    return rows


def prepare_laminar_project(
    mask_path: str | Path,
    output_dir: str | Path,
    slice_axis: int | str = 0,
    min_area: float = 20.0,
    largest_only: bool = True,
    manual_every: int = 10,
) -> List[Contour2D]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_volume = load_volume(mask_path)
    mask = _as_bool_mask(mask_volume.data)
    contours = extract_slice_contours(
        mask,
        slice_axis=slice_axis,
        min_area=min_area,
        largest_only=largest_only,
    )
    save_contours_json(contours, output_dir / "contours.json")
    write_contour_index_table(contours, output_dir / "contour_index.csv")
    write_contour_points_table(contours, output_dir / "contour_points.csv")
    write_manual_landmark_template(contours, output_dir / "manual_landmarks_template.csv", manual_every)
    return contours


def _write_surfaces(
    mask: np.ndarray,
    boundaries: Sequence[BoundarySlice],
    contours: Sequence[Contour2D],
    output_dir: Path,
    resample_points: int,
    surface_method: str,
    slice_axis: int | str,
    spacing: Optional[np.ndarray] = None,
    shell_backend: str = "voxel",
    cut_curve_json: Optional[str | Path] = None,
) -> Dict[str, SurfaceMesh]:
    surface_dir = output_dir / "surfaces"
    surface_dir.mkdir(parents=True, exist_ok=True)
    surface_method = normalize_surface_build_method(surface_method)
    shell_backend = normalize_shell_backend(shell_backend)

    if surface_method == SURFACE_BUILD_FAST_LOFT:
        meshes = _loft_surface_patches(boundaries, resample_points=resample_points)
    elif surface_method == SURFACE_BUILD_CONTOUR_SHELL:
        meshes = contour_shell_surface_patches(
            boundaries,
            contours,
            resample_points=resample_points,
        )
    elif surface_method == SURFACE_BUILD_ARC_GRAPH:
        meshes, qc = arc_graph_surface_patches(
            boundaries,
            contours,
            resample_points=resample_points,
            spacing=spacing,
        )
        _write_arc_graph_qc_tables(qc, output_dir / "qc")
    elif surface_method == SURFACE_BUILD_SHELL_CUT:
        cut_curves: Optional[List[SurfaceCutCurve]] = None
        patch_seeds: Optional[List[SurfacePatchSeed]] = None
        input_warnings: List[Dict] = []
        if cut_curve_json is not None:
            cut_curves, patch_seeds, input_warnings = read_shell_cut_json_with_qc(
                cut_curve_json,
                contours=contours,
            )
        meshes, _qc = shell_cut_surface_patches(
            mask,
            boundaries,
            cut_curves=cut_curves,
            patch_seeds=patch_seeds if patch_seeds else None,
            spacing=spacing,
            shell_backend=shell_backend,
            output_qc_dir=output_dir / "qc",
            input_warnings=input_warnings,
        )
    else:
        meshes = _mask_constrained_surface_patches_with_fallback(
            mask,
            boundaries,
            contours,
            resample_points=resample_points,
            slice_axis=slice_axis,
        )

    for name, mesh in meshes.items():
        if name == "lateral":
            write_ply(surface_dir / "target_lateral_boundary.ply", mesh)
            write_obj(surface_dir / "target_lateral_boundary.obj", mesh)
        else:
            write_ply(surface_dir / f"target_{name}_surface.ply", mesh)
            write_obj(surface_dir / f"target_{name}_surface.obj", mesh)
    return meshes


def _write_named_surface_outputs(
    meshes: Dict[str, SurfaceMesh],
    output_dir: Path,
) -> Dict[str, Path]:
    surface_dir = output_dir / "surfaces"
    surface_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}
    used_names: set[str] = set()
    for index, (name, mesh) in enumerate(meshes.items(), start=1):
        safe_name = _unique_surface_name(name, used_names, fallback=f"surface_{index}")
        mesh.name = safe_name
        ply_path = surface_dir / f"target_{safe_name}_surface.ply"
        obj_path = surface_dir / f"target_{safe_name}_surface.obj"
        write_ply(ply_path, mesh)
        write_obj(obj_path, mesh)
        outputs[f"{safe_name}_surface"] = ply_path
        outputs[f"{safe_name}_surface_obj"] = obj_path
    return outputs


def _mask_constrained_surface_patches_with_fallback(
    mask: np.ndarray,
    boundaries: Sequence[BoundarySlice],
    contours: Sequence[Contour2D],
    resample_points: int,
    slice_axis: int | str,
) -> Dict[str, SurfaceMesh]:
    try:
        return mask_constrained_surface_patches(
            mask,
            boundaries,
            resample_points=resample_points,
            slice_axis=slice_axis,
        )
    except Exception as exc:
        warnings.warn(
            f"Voxel surface build failed; falling back to contour-shell surfaces: {exc}",
            RuntimeWarning,
        )
        try:
            return contour_shell_surface_patches(
                boundaries,
                contours,
                resample_points=resample_points,
            )
        except Exception as contour_exc:
            warnings.warn(
                f"Contour-shell surface build failed; falling back to curve lofting: {contour_exc}",
                RuntimeWarning,
            )
            return _loft_surface_patches(boundaries, resample_points=resample_points)


def _loft_surface_patches(
    boundaries: Sequence[BoundarySlice],
    resample_points: int,
) -> Dict[str, SurfaceMesh]:
    meshes: Dict[str, SurfaceMesh] = {}
    for name in ("outer", "inner"):
        meshes[name] = loft_surface(boundaries, name, resample_points=resample_points)

    lateral_meshes: List[SurfaceMesh] = []
    for lateral_index in (0, 1):
        try:
            lateral_meshes.append(
                loft_surface(
                    boundaries,
                    "lateral",
                    resample_points=max(8, resample_points // 3),
                    lateral_index=lateral_index,
                )
            )
        except ValueError:
            continue
    if lateral_meshes:
        vertices = []
        faces = []
        offset = 0
        for mesh in lateral_meshes:
            vertices.append(mesh.vertices)
            faces.append(mesh.faces + offset)
            offset += len(mesh.vertices)
        meshes["lateral"] = SurfaceMesh(
            name="lateral",
            vertices=np.vstack(vertices),
            faces=np.vstack(faces),
        )
    return meshes


def run_3d_shell_patch_pipeline(
    mask_path: str | Path,
    cut_curve_json: str | Path,
    output_dir: str | Path,
    shell_backend: str = SHELL_BACKEND_MARCHING_CUBES,
    max_surface_quads: int = 1_500_000,
) -> Dict[str, Path]:
    """Build user-named shell patches from direct 3D closed-curve annotations."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_dir = output_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    mask_volume = load_volume(mask_path)
    mask = _as_bool_mask(mask_volume.data)
    spacing = _voxel_spacing_from_volume(mask_volume)
    curves, patch_seeds, input_warnings = read_shell_cut_json_with_qc(
        cut_curve_json,
        contours=None,
    )
    meshes, _qc = named_shell_cut_surface_patches(
        mask,
        curves,
        patch_seeds,
        spacing=spacing,
        shell_backend=shell_backend,
        output_qc_dir=qc_dir,
        max_surface_quads=max_surface_quads,
        input_warnings=input_warnings,
    )
    surface_outputs = _write_named_surface_outputs(meshes, output_dir)
    config = {
        "mask_path": str(mask_path),
        "cut_curve_json": str(cut_curve_json),
        "surface_method": "manual_3d_shell_patch",
        "shell_backend": normalize_shell_backend(shell_backend),
        "spacing": [float(value) for value in spacing],
        "surface_names": list(meshes.keys()),
        "surface_outputs": {key: str(value) for key, value in surface_outputs.items()},
    }
    with (output_dir / "project_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    outputs: Dict[str, Path] = {
        "output_dir": output_dir,
        "annotations": Path(cut_curve_json),
        "qc": qc_dir,
        "project_config": output_dir / "project_config.json",
    }
    outputs.update(surface_outputs)
    return outputs


def run_laminar_boundary_pipeline(
    mask_path: str | Path,
    manual_csv: str | Path,
    output_dir: str | Path,
    template_path: Optional[str | Path] = None,
    cell_csv: Optional[str | Path] = None,
    swc_paths: Optional[Sequence[str | Path]] = None,
    slice_axis: int | str = 0,
    min_area: float = 20.0,
    largest_only: bool = True,
    resample_points: int = 80,
    surface_method: str = SURFACE_BUILD_CONTOUR_SHELL,
    shell_backend: str = "voxel",
    cut_curve_json: Optional[str | Path] = None,
    depth_method: str = "auto",
    max_laplace_voxels: int = 250_000,
    boundary_dilation: int = 1,
    qc_every: int = 10,
    volume_format: str = "nrrd",
) -> Dict[str, Path]:
    """Run the full MVP pipeline from mask and manual landmarks."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    surface_method = normalize_surface_build_method(surface_method)
    shell_backend = normalize_shell_backend(shell_backend)
    volumes_dir = output_dir / "volumes"
    tables_dir = output_dir / "tables"
    qc_dir = output_dir / "qc"
    volumes_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    surface_registry_path = qc_dir / "surface_method_registry.csv"
    write_surface_method_registry(surface_registry_path)

    mask_volume = load_volume(mask_path)
    mask = _as_bool_mask(mask_volume.data)
    spacing = _voxel_spacing_from_volume(mask_volume)
    contours = extract_slice_contours(
        mask,
        slice_axis=slice_axis,
        min_area=min_area,
        largest_only=largest_only,
    )
    if not contours:
        raise ValueError("No usable contours were extracted from the mask")
    save_contours_json(contours, output_dir / "contours.json")
    write_contour_index_table(contours, output_dir / "contour_index.csv")
    write_contour_points_table(contours, output_dir / "contour_points.csv")

    manual_rows = read_manual_landmarks(manual_csv)
    manual_boundaries = build_manual_boundaries(contours, manual_rows, resample_points)
    validate_endpoint_annotations(contours, manual_boundaries)
    boundaries = propagate_boundaries(contours, manual_boundaries, resample_points=resample_points)
    save_boundaries_json(boundaries, output_dir / "boundary_annotations.json")
    write_boundary_summary(boundaries, tables_dir / "boundary_summary.csv")
    target_mask = _mask_for_annotated_components(mask, boundaries)
    effective_cut_curve_json = _resolve_shell_cut_json_path(
        output_dir,
        surface_method,
        cut_curve_json,
    )

    _write_surfaces(
        target_mask,
        boundaries,
        contours,
        output_dir,
        resample_points=resample_points,
        surface_method=surface_method,
        slice_axis=slice_axis,
        spacing=spacing,
        shell_backend=shell_backend,
        cut_curve_json=effective_cut_curve_json,
    )
    surface_outputs = {
        "outer_surface": output_dir / "surfaces" / "target_outer_surface.ply",
        "inner_surface": output_dir / "surfaces" / "target_inner_surface.ply",
        "lateral_surface": output_dir / "surfaces" / "target_lateral_boundary.ply",
    }

    if _surface_only_build_requested(depth_method):
        write_qc_tables(boundaries, qc_dir)
        config = {
            "mask_path": str(mask_path),
            "manual_csv": str(manual_csv),
            "template_path": str(template_path) if template_path else None,
            "cell_csv": str(cell_csv) if cell_csv else None,
            "swc_paths": [str(path) for path in swc_paths] if swc_paths else [],
            "slice_axis": _slice_axis_to_int(slice_axis),
            "resample_points": resample_points,
            "surface_method": surface_method,
            "surface_method_category": surface_build_method_category(surface_method),
            "legacy_surface_method": surface_method in SURFACE_BUILD_LEGACY_METHODS,
            "surface_method_registry": surface_method_registry(),
            "depth_method": "surfaces only",
            "mask_component_selection": "annotation_connected",
            "target_mask_voxels": int(np.count_nonzero(target_mask)),
            "volume_format": volume_format,
            "max_laplace_voxels": max_laplace_voxels,
            "depth_outputs": "skipped",
            "boundary_label_values": {
                "background": BOUNDARY_BACKGROUND,
                "region": BOUNDARY_REGION,
                "outer": BOUNDARY_OUTER,
                "inner": BOUNDARY_INNER,
                "lateral": BOUNDARY_LATERAL,
            },
        }
        if surface_method == SURFACE_BUILD_ARC_GRAPH:
            config["arc_graph"] = {
                "normal_threshold": 2.5,
                "partial_threshold": 4.0,
                "min_confidence": 0.10,
                "max_interval_overlap": 0.20,
                "ambiguous_relative_margin": 0.10,
                "min_reference_span_fraction": 0.15,
                "spacing": [float(value) for value in spacing],
            }
        if surface_method == SURFACE_BUILD_SHELL_CUT:
            config["shell_cut"] = {
                "shell_backend": shell_backend,
                "cut_curve_json": str(effective_cut_curve_json) if effective_cut_curve_json else None,
                "cut_curve_source": shell_cut_json_source(effective_cut_curve_json)
                if effective_cut_curve_json
                else "annotation_derived",
                "spacing": [float(value) for value in spacing],
            }
        with (output_dir / "project_config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        return {
            "output_dir": output_dir,
            "contours": output_dir / "contours.json",
            "boundaries": output_dir / "boundary_annotations.json",
            "boundary_summary": tables_dir / "boundary_summary.csv",
            "outer_surface": surface_outputs["outer_surface"],
            "inner_surface": surface_outputs["inner_surface"],
            "lateral_surface": surface_outputs["lateral_surface"],
            "qc": qc_dir,
            "surface_method_registry": surface_registry_path,
        }

    labels = make_boundary_label_volume(target_mask, boundaries, dilation_iterations=boundary_dilation)
    depth = compute_laminar_depth(
        target_mask,
        labels,
        method=depth_method,
        max_laplace_voxels=max_laplace_voxels,
    )
    normals = compute_layer_normals(depth, target_mask)

    volume_format = volume_format.lower().lstrip(".")
    if volume_format not in ("nrrd", "npy", "nii", "nii.gz"):
        raise ValueError("volume_format must be nrrd, npy, nii, or nii.gz")

    def volume_path(name: str) -> Path:
        return volumes_dir / f"{name}.{volume_format}"

    save_volume(volume_path("target_mask"), target_mask.astype(np.uint8), reference=mask_volume)
    save_volume(volume_path("boundary_label_volume"), labels, reference=mask_volume)
    save_volume(volume_path("laminar_depth"), depth, reference=mask_volume)
    save_volume(volume_path("layer_normal_x"), normals[..., 0], reference=mask_volume)
    save_volume(volume_path("layer_normal_y"), normals[..., 1], reference=mask_volume)
    save_volume(volume_path("layer_normal_z"), normals[..., 2], reference=mask_volume)

    if cell_csv is not None:
        write_cell_depth_table(
            cell_csv,
            tables_dir / "cell_laminar_depth.csv",
            depth,
            normals,
            labels,
        )

    if swc_paths:
        summarize_dendrite_depth(
            swc_paths,
            depth,
            normals,
            tables_dir / "dendrite_laminar_depth.csv",
        )

    template = load_volume(template_path).data if template_path else None
    write_qc_tables(boundaries, qc_dir)
    write_qc_slice_overlays(
        qc_dir / "qc_slice_overlay",
        boundaries,
        mask=target_mask,
        template=template,
        slice_axis=slice_axis,
        every=qc_every,
    )

    config = {
        "mask_path": str(mask_path),
        "manual_csv": str(manual_csv),
        "template_path": str(template_path) if template_path else None,
        "cell_csv": str(cell_csv) if cell_csv else None,
        "swc_paths": [str(path) for path in swc_paths] if swc_paths else [],
        "slice_axis": _slice_axis_to_int(slice_axis),
        "resample_points": resample_points,
        "surface_method": surface_method,
        "surface_method_category": surface_build_method_category(surface_method),
        "legacy_surface_method": surface_method in SURFACE_BUILD_LEGACY_METHODS,
        "surface_method_registry": surface_method_registry(),
        "mask_component_selection": "annotation_connected",
        "target_mask_voxels": int(np.count_nonzero(target_mask)),
        "depth_method": depth_method,
        "volume_format": volume_format,
        "max_laplace_voxels": max_laplace_voxels,
        "boundary_label_values": {
            "background": BOUNDARY_BACKGROUND,
            "region": BOUNDARY_REGION,
            "outer": BOUNDARY_OUTER,
            "inner": BOUNDARY_INNER,
            "lateral": BOUNDARY_LATERAL,
        },
    }
    if surface_method == SURFACE_BUILD_ARC_GRAPH:
        config["arc_graph"] = {
            "normal_threshold": 2.5,
            "partial_threshold": 4.0,
            "min_confidence": 0.10,
            "max_interval_overlap": 0.20,
            "ambiguous_relative_margin": 0.10,
            "min_reference_span_fraction": 0.15,
            "spacing": [float(value) for value in spacing],
        }
    if surface_method == SURFACE_BUILD_SHELL_CUT:
        config["shell_cut"] = {
            "shell_backend": shell_backend,
            "cut_curve_json": str(effective_cut_curve_json) if effective_cut_curve_json else None,
            "cut_curve_source": shell_cut_json_source(effective_cut_curve_json)
            if effective_cut_curve_json
            else "annotation_derived",
            "spacing": [float(value) for value in spacing],
        }
    with (output_dir / "project_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    return {
        "output_dir": output_dir,
        "contours": output_dir / "contours.json",
        "boundaries": output_dir / "boundary_annotations.json",
        "boundary_summary": tables_dir / "boundary_summary.csv",
        "outer_surface": surface_outputs["outer_surface"],
        "inner_surface": surface_outputs["inner_surface"],
        "lateral_surface": surface_outputs["lateral_surface"],
        "laminar_depth": volume_path("laminar_depth"),
        "boundary_labels": volume_path("boundary_label_volume"),
        "qc": qc_dir,
        "surface_method_registry": surface_registry_path,
    }


def run_laminar_depth_pipeline(
    mask_path: str | Path,
    boundaries_json: str | Path,
    output_dir: str | Path,
    template_path: Optional[str | Path] = None,
    cell_csv: Optional[str | Path] = None,
    swc_paths: Optional[Sequence[str | Path]] = None,
    slice_axis: int | str = 0,
    depth_method: str = "auto",
    max_laplace_voxels: int = 250_000,
    boundary_dilation: int = 1,
    qc_every: int = 10,
    volume_format: str = "nrrd",
) -> Dict[str, Path]:
    """Compute laminar depth from a saved boundary_annotations.json file."""

    output_dir = Path(output_dir)
    volumes_dir = output_dir / "volumes"
    tables_dir = output_dir / "tables"
    qc_dir = output_dir / "qc"
    volumes_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    mask_volume = load_volume(mask_path)
    mask = _as_bool_mask(mask_volume.data)
    boundaries = read_boundaries_json(boundaries_json)
    if not boundaries:
        raise ValueError("No boundaries found in boundary_annotations.json")

    labels = make_boundary_label_volume(mask, boundaries, dilation_iterations=boundary_dilation)
    depth = compute_laminar_depth(
        mask,
        labels,
        method=depth_method,
        max_laplace_voxels=max_laplace_voxels,
    )
    normals = compute_layer_normals(depth, mask)

    volume_format = volume_format.lower().lstrip(".")
    if volume_format not in ("nrrd", "npy", "nii", "nii.gz"):
        raise ValueError("volume_format must be nrrd, npy, nii, or nii.gz")

    def volume_path(name: str) -> Path:
        return volumes_dir / f"{name}.{volume_format}"

    save_volume(volume_path("target_mask"), mask.astype(np.uint8), reference=mask_volume)
    save_volume(volume_path("boundary_label_volume"), labels, reference=mask_volume)
    save_volume(volume_path("laminar_depth"), depth, reference=mask_volume)
    save_volume(volume_path("layer_normal_x"), normals[..., 0], reference=mask_volume)
    save_volume(volume_path("layer_normal_y"), normals[..., 1], reference=mask_volume)
    save_volume(volume_path("layer_normal_z"), normals[..., 2], reference=mask_volume)

    if cell_csv is not None:
        write_cell_depth_table(
            cell_csv,
            tables_dir / "cell_laminar_depth.csv",
            depth,
            normals,
            labels,
        )

    if swc_paths:
        summarize_dendrite_depth(
            swc_paths,
            depth,
            normals,
            tables_dir / "dendrite_laminar_depth.csv",
        )

    template = load_volume(template_path).data if template_path else None
    write_qc_tables(boundaries, qc_dir)
    write_qc_slice_overlays(
        qc_dir / "qc_slice_overlay",
        boundaries,
        mask=mask,
        template=template,
        slice_axis=slice_axis,
        every=qc_every,
    )

    config = {
        "mask_path": str(mask_path),
        "boundaries_json": str(boundaries_json),
        "template_path": str(template_path) if template_path else None,
        "cell_csv": str(cell_csv) if cell_csv else None,
        "swc_paths": [str(path) for path in swc_paths] if swc_paths else [],
        "slice_axis": _slice_axis_to_int(slice_axis),
        "depth_method": depth_method,
        "volume_format": volume_format,
        "max_laplace_voxels": max_laplace_voxels,
        "boundary_label_values": {
            "background": BOUNDARY_BACKGROUND,
            "region": BOUNDARY_REGION,
            "outer": BOUNDARY_OUTER,
            "inner": BOUNDARY_INNER,
            "lateral": BOUNDARY_LATERAL,
        },
    }
    with (output_dir / "depth_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    return {
        "output_dir": output_dir,
        "laminar_depth": volume_path("laminar_depth"),
        "boundary_labels": volume_path("boundary_label_volume"),
        "qc": qc_dir,
    }


def make_demo_mask(shape: Tuple[int, int, int] = (18, 64, 72)) -> np.ndarray:
    """Create a small curved slab-like mask for smoke tests and demos."""

    z_count, y_count, x_count = shape
    yy, xx = np.mgrid[0:y_count, 0:x_count]
    mask = np.zeros(shape, dtype=np.uint8)
    for z in range(z_count):
        center_y = y_count * 0.50 + math.sin(z / max(1, z_count - 1) * math.pi) * 5.0
        center_x = x_count * 0.50 + math.cos(z / max(1, z_count - 1) * math.pi) * 4.0
        radius_y = 18.0 - abs(z - z_count / 2.0) * 0.18
        radius_x = 24.0 - abs(z - z_count / 2.0) * 0.15
        ellipse = ((yy - center_y) / radius_y) ** 2 + ((xx - center_x) / radius_x) ** 2 <= 1.0
        mask[z, ellipse] = 1
    return mask


def _demo_landmark_row(contour: Contour2D) -> Dict[str, str]:
    points = _normalize_contour(contour.points)
    plane = _volume_to_plane_points(points, slice_axis=0)
    x_min, y_min = plane.min(axis=0)
    x_max, y_max = plane.max(axis=0)
    center_x = float((x_min + x_max) * 0.5)
    span_x = float(x_max - x_min)

    outer_start_target = np.array([center_x - span_x * 0.32, y_min + 1.0])
    outer_end_target = np.array([center_x + span_x * 0.32, y_min + 1.0])
    inner_start_target = np.array([center_x - span_x * 0.32, y_max - 1.0])
    inner_end_target = np.array([center_x + span_x * 0.32, y_max - 1.0])

    outer_start = int(np.argmin(np.linalg.norm(plane - outer_start_target, axis=1)))
    outer_end = int(np.argmin(np.linalg.norm(plane - outer_end_target, axis=1)))
    inner_start = int(np.argmin(np.linalg.norm(plane - inner_start_target, axis=1)))
    inner_end = int(np.argmin(np.linalg.norm(plane - inner_end_target, axis=1)))

    def choose_by_mean_y(start: int, end: int, prefer_top: bool) -> str:
        forward = _path_from_indices(points, _path_indices(len(points), start, end, 1))
        backward = _path_from_indices(points, _path_indices(len(points), start, end, -1))
        forward_y = _volume_to_plane_points(forward, 0)[:, 1].mean()
        backward_y = _volume_to_plane_points(backward, 0)[:, 1].mean()
        if prefer_top:
            return "1" if forward_y < backward_y else "-1"
        return "1" if forward_y > backward_y else "-1"

    outer_path = choose_by_mean_y(outer_start, outer_end, prefer_top=True)
    inner_path = choose_by_mean_y(inner_start, inner_end, prefer_top=False)

    return {
        "slice_index": str(contour.slice_index),
        "contour_id": str(contour.contour_id),
        "outer_start_index": str(outer_start),
        "outer_end_index": str(outer_end),
        "outer_path": outer_path,
        "inner_start_index": str(inner_start),
        "inner_end_index": str(inner_end),
        "inner_path": inner_path,
    }


def write_demo_project(output_dir: str | Path) -> Tuple[Path, Path]:
    """Write a complete tiny demo input pair: mask.nrrd and manual_landmarks.csv."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask = make_demo_mask()
    mask_path = output_dir / "demo_mask.nrrd"
    nrrd.write(str(mask_path), mask)

    contours = extract_slice_contours(mask, slice_axis=0, min_area=20.0, largest_only=True)
    selected_slices = [contours[0], contours[len(contours) // 2], contours[-1]]
    rows = [_demo_landmark_row(contour) for contour in selected_slices]
    manual_csv = output_dir / "demo_manual_landmarks.csv"
    with manual_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return mask_path, manual_csv
