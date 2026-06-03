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
import json
import math
import os
import pickle
import re
import sys
import warnings
from collections import defaultdict
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

SURFACE_MODE_NORMAL = "normal"
SURFACE_MODE_OUTER_ONLY = "outer_only"
SURFACE_MODE_INNER_ONLY = "inner_only"
SURFACE_MODES = {
    SURFACE_MODE_NORMAL,
    SURFACE_MODE_OUTER_ONLY,
    SURFACE_MODE_INNER_ONLY,
}


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


def load_volume(path: str | Path) -> VolumeData:
    """Load a 3D volume from NRRD, NumPy, or NIfTI if nibabel is installed."""

    path = _resolve_existing_input_path(path)
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
    return np.take(volume, indices=slice_index, axis=slice_axis)


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
            if left_mode == right_mode and left_mode != SURFACE_MODE_NORMAL:
                for slice_index in range(left.slice_index + 1, right.slice_index):
                    candidates = by_slice.get(slice_index, [])
                    if not candidates:
                        continue
                    contour = max(candidates, key=lambda item: item.area)
                    output[slice_index] = make_whole_contour_boundary(
                        contour,
                        surface_mode=left_mode,
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

    return [output[key] for key in sorted(output)]


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
    curves: List[np.ndarray] = []
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
        curves.append(resample_polyline(curve, resample_points))

    if len(curves) < 2:
        raise ValueError(f"Need at least two curves to build {arc_name} surface")
    for index, curve in enumerate(curves):
        if not _is_closed_polyline(curve):
            continue
        target = None
        if index > 0:
            target = curves[index - 1][0]
        elif len(curves) > 1:
            target = curves[index + 1][0]
        if target is not None:
            curves[index] = resample_polyline(
                _align_closed_polyline_start(curve, target),
                resample_points,
            )

    vertices = np.vstack(curves).astype(float)
    faces: List[List[int]] = []
    n_curves = len(curves)
    for curve_index in range(n_curves - 1):
        base = curve_index * resample_points
        next_base = (curve_index + 1) * resample_points
        for point_index in range(resample_points - 1):
            a = base + point_index
            b = base + point_index + 1
            c = next_base + point_index
            d = next_base + point_index + 1
            faces.append([a, c, b])
            faces.append([b, c, d])

    return SurfaceMesh(name=arc_name, vertices=vertices, faces=np.asarray(faces, dtype=np.int32))


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
    boundaries: Sequence[BoundarySlice],
    output_dir: Path,
    resample_points: int,
) -> Dict[str, SurfaceMesh]:
    surface_dir = output_dir / "surfaces"
    surface_dir.mkdir(parents=True, exist_ok=True)
    meshes: Dict[str, SurfaceMesh] = {}
    for name in ("outer", "inner"):
        mesh = loft_surface(boundaries, name, resample_points=resample_points)
        meshes[name] = mesh
        write_ply(surface_dir / f"target_{name}_surface.ply", mesh)
        write_obj(surface_dir / f"target_{name}_surface.obj", mesh)

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
        lateral = SurfaceMesh(
            name="lateral",
            vertices=np.vstack(vertices),
            faces=np.vstack(faces),
        )
        meshes["lateral"] = lateral
        write_ply(surface_dir / "target_lateral_boundary.ply", lateral)
        write_obj(surface_dir / "target_lateral_boundary.obj", lateral)
    return meshes


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
    depth_method: str = "auto",
    max_laplace_voxels: int = 250_000,
    boundary_dilation: int = 1,
    qc_every: int = 10,
    volume_format: str = "nrrd",
) -> Dict[str, Path]:
    """Run the full MVP pipeline from mask and manual landmarks."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    volumes_dir = output_dir / "volumes"
    tables_dir = output_dir / "tables"
    qc_dir = output_dir / "qc"
    volumes_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    mask_volume = load_volume(mask_path)
    mask = _as_bool_mask(mask_volume.data)
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
    boundaries = propagate_boundaries(contours, manual_boundaries, resample_points=resample_points)
    save_boundaries_json(boundaries, output_dir / "boundary_annotations.json")
    write_boundary_summary(boundaries, tables_dir / "boundary_summary.csv")

    _write_surfaces(boundaries, output_dir, resample_points=resample_points)
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
            "depth_method": "surfaces only",
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
        }

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
        "manual_csv": str(manual_csv),
        "template_path": str(template_path) if template_path else None,
        "cell_csv": str(cell_csv) if cell_csv else None,
        "swc_paths": [str(path) for path in swc_paths] if swc_paths else [],
        "slice_axis": _slice_axis_to_int(slice_axis),
        "resample_points": resample_points,
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
    selected_slices = [contours[2], contours[len(contours) // 2], contours[-3]]
    rows = [_demo_landmark_row(contour) for contour in selected_slices]
    manual_csv = output_dir / "demo_manual_landmarks.csv"
    with manual_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return mask_path, manual_csv
