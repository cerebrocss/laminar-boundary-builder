"""OpenGL 3D shell annotation view for the desktop app."""

from __future__ import annotations

import heapq
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from OpenGL.GL import (
    GL_BACK,
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_CULL_FACE,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_FALSE,
    GL_FLOAT,
    GL_LEQUAL,
    GL_LINES,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POINT_SMOOTH,
    GL_POINTS,
    GL_SRC_ALPHA,
    GL_TRIANGLES,
    GL_TRUE,
    glBlendFunc,
    glClear,
    glClearColor,
    glCullFace,
    glDepthFunc,
    glDepthMask,
    glDisable,
    glDrawArrays,
    glEnable,
    glLineWidth,
    glPointSize,
    glViewport,
)
from PyQt5.QtCore import QPointF, Qt, pyqtSignal
from PyQt5.QtGui import (
    QColor,
    QMatrix4x4,
    QOpenGLBuffer,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QPainter,
    QPen,
    QPolygonF,
    QSurfaceFormat,
    QVector3D,
    QVector4D,
)
from PyQt5.QtWidgets import QOpenGLWidget, QSizePolicy


class ShellGLCanvas(QOpenGLWidget):
    """GPU-backed 3D shell view with the same annotation API as the old preview."""

    annotation_changed = pyqtSignal()
    build_ready_changed = pyqtSignal(bool)

    MAX_DISPLAY_FACES = 180000
    MAX_HOVER_FACES = 30000
    MAX_EDGE_SEGMENTS = 90000
    MAX_CURVE_PATH_VISITED = 120000
    CLICK_MOVE_TOLERANCE_SQ = 81.0
    ANNOTATION_POINT_HIT_RADIUS_PX = 22.0
    ANNOTATION_POINT_PATCH_GUARD_RADIUS_PX = 30.0
    ANNOTATION_LINE_COLOR = "#ff3b30"
    ANNOTATION_POINT_COLOR = "#ffe033"
    ANNOTATION_DEPTH_BIAS = 0.0025
    PICK_VISIBLE_DEPTH_TOLERANCE = 0.006
    DRAG_INSIDE_CURVE_MARGIN_PX = 12.0
    MODEL_BASE_SCALE = 0.84
    PROJECTION_NEAR = -10.0
    PROJECTION_FAR = 10.0

    def __init__(self, parent=None):
        super().__init__(parent)
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        # Keep the framebuffer simple; annotations are now regular 3D draws that
        # use the same depth test as the shell.
        fmt.setSamples(0)
        self.setFormat(fmt)
        self.setMinimumSize(420, 420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.shell_mesh = None
        self.surface_name = "surface"
        self.surface_names: List[str] = []
        self.active_surface_index = 0
        self.annotation_mode = "curve"
        self.closed_curves: List[Dict[str, object]] = []
        self.active_curve_vertices: List[int] = []
        self.selected_patches: List[Dict[str, object]] = []
        self.hover_face: Optional[int] = None
        self.message = "Load a mask to start 3D annotation"

        self.rotation_yaw = -0.55
        self.rotation_pitch = 0.38
        self.preview_zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._drag_pos: Optional[QPointF] = None
        self._press_pos: Optional[QPointF] = None
        self._drag_mode = "rotate"
        self._drag_moved = False
        self._drag_point_ref: Optional[Tuple[str, int, int]] = None
        self._last_drag_reject_reason = ""
        self._last_hover_at = 0.0

        self._gl_ready = False
        self._surface_program: Optional[QOpenGLShaderProgram] = None
        self._line_program: Optional[QOpenGLShaderProgram] = None
        self._vbo: Optional[QOpenGLBuffer] = None
        self._edge_vbo: Optional[QOpenGLBuffer] = None
        self._annotation_line_vbo: Optional[QOpenGLBuffer] = None
        self._annotation_point_vbo: Optional[QOpenGLBuffer] = None
        self._mesh_dirty = False
        self._vertex_count = 0
        self._edge_vertex_count = 0
        self._interleaved = np.empty((0, 9), dtype=np.float32)
        self._edge_vertices = np.empty((0, 3), dtype=np.float32)

        self._vertices = np.empty((0, 3), dtype=np.float32)
        self._faces = np.empty((0, 3), dtype=np.int64)
        self._center = np.zeros(3, dtype=np.float32)
        self._radius = 1.0
        self._normalized_vertices = np.empty((0, 3), dtype=np.float32)
        self._display_vertex_normals = np.empty((0, 3), dtype=np.float32)
        self._display_face_ids = np.empty((0,), dtype=np.int64)
        self._display_triangles = np.empty((0, 3), dtype=np.int64)
        self._display_centers = np.empty((0, 3), dtype=np.float32)
        self._display_center_face_ids = np.empty((0,), dtype=np.int64)
        self._curve_edge_graph: Dict[int, List[Tuple[int, float]]] = {}
        self._curve_edge_set: set[Tuple[int, int]] = set()
        self._curve_graph_vertices = np.empty((0, 3), dtype=np.float32)
        self._curve_path_cache: Dict[Tuple[int, ...], List[int]] = {}
        self._screen_cache_key = None
        self._screen_cache = None
        self._front_hit_cache: Dict[Tuple[int, int, int, int], Optional[Tuple[int, float]]] = {}
        self._visible_vertex_cache: Dict[int, bool] = {}
        self._pick_bbox_cache_key = None
        self._pick_bbox_points = np.empty((0, 3, 2), dtype=np.float32)
        self._pick_bbox_mins = np.empty((0, 2), dtype=np.float32)
        self._pick_bbox_maxs = np.empty((0, 2), dtype=np.float32)
        self._viewport_width = 1
        self._viewport_height = 1

    def _clear_pick_caches(self) -> None:
        self._front_hit_cache = {}
        self._visible_vertex_cache = {}
        self._pick_bbox_cache_key = None
        self._pick_bbox_points = np.empty((0, 3, 2), dtype=np.float32)
        self._pick_bbox_mins = np.empty((0, 2), dtype=np.float32)
        self._pick_bbox_maxs = np.empty((0, 2), dtype=np.float32)

    def set_reference_contours(self, _contours, _slice_axis: int = 0) -> None:
        return

    def set_boundaries(self, _boundaries, current_boundary=None, slice_axis: int = 0, message: str = "") -> None:
        if self.shell_mesh is None and message:
            self.message = message
            self.update()

    def set_shell_mesh(self, shell_mesh) -> None:
        self.shell_mesh = shell_mesh
        self.closed_curves = []
        self.active_curve_vertices = []
        self.selected_patches = []
        self.surface_names = []
        self.active_surface_index = 0
        self.surface_name = "surface"
        self.hover_face = None
        self._drag_point_ref = None
        self.annotation_mode = "curve"
        self.rotation_yaw = -0.55
        self.rotation_pitch = 0.38
        self.preview_zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._screen_cache_key = None
        self._screen_cache = None
        self._clear_pick_caches()
        if shell_mesh is None:
            self._clear_mesh_buffers()
            self.message = "3D shell is not available"
        else:
            self._prepare_mesh_arrays(shell_mesh)
            self.message = "3D: click the shaded shell to draw a cut curve"
        self._emit_3d_state()
        self.update()

    def set_surface_name(self, text: str) -> None:
        self._sync_surface_queue()
        name = str(text or "").strip() or self._default_surface_name(self.active_surface_index)
        if self.surface_names:
            self.surface_names[self.active_surface_index] = name
        self.surface_name = name
        self._emit_3d_state()

    def set_active_surface_index(self, index: int) -> None:
        self._sync_surface_queue()
        if not self.surface_names:
            self.active_surface_index = 0
            self.surface_name = "surface"
            self._emit_3d_state()
            return
        self.active_surface_index = max(0, min(int(index), len(self.surface_names) - 1))
        self.surface_name = self.surface_names[self.active_surface_index]
        self._emit_3d_state()

    def set_curve_mode(self) -> None:
        if len(self.active_curve_vertices) >= 3:
            self.close_active_curve_to_start()
            return
        self.annotation_mode = "curve"
        self.message = "3D: click the shaded shell to draw a cut curve"
        self.update()

    def set_patch_mode(self) -> None:
        if not self.closed_curves:
            self.message = "Close at least one cut curve before selecting a surface"
            self.update()
            return
        self.annotation_mode = "patch"
        self.message = "3D: hover the shaded surface patch, then click to keep it"
        self.update()

    def clear_3d_annotations(self) -> None:
        self.closed_curves = []
        self.active_curve_vertices = []
        self.selected_patches = []
        self.surface_names = []
        self.active_surface_index = 0
        self.surface_name = "surface"
        self.hover_face = None
        self._drag_point_ref = None
        self.annotation_mode = "curve"
        self.message = "3D: click the shaded shell to draw a cut curve"
        self._emit_3d_state()
        self.update()

    def can_build_3d_surfaces(self) -> bool:
        if self.shell_mesh is None or not self.closed_curves:
            return False
        self._sync_surface_queue()
        seeded = {
            int(patch.get("surface_index", 0))
            for patch in self.selected_patches
            if 0 <= int(patch.get("surface_index", 0)) < len(self.surface_names)
        }
        return bool(seeded) and all(index in seeded for index in range(len(self.surface_names)))

    def annotation_counts(self) -> tuple[int, int, int]:
        return (len(self.closed_curves), len(self.active_curve_vertices), len(self.selected_patches))

    @staticmethod
    def _default_surface_name(index: int) -> str:
        return f"surface_{int(index) + 1}"

    def _sync_surface_queue(self) -> None:
        while len(self.surface_names) < len(self.closed_curves):
            if not self.surface_names and self.surface_name.strip() and self.surface_name != "surface":
                self.surface_names.append(self.surface_name.strip())
            else:
                self.surface_names.append(self._default_surface_name(len(self.surface_names)))
        if len(self.surface_names) > len(self.closed_curves):
            self.surface_names = self.surface_names[: len(self.closed_curves)]
        if not self.surface_names:
            self.active_surface_index = 0
            self.selected_patches = []
            return
        self.selected_patches = [
            patch
            for patch in self.selected_patches
            if 0 <= int(patch.get("surface_index", 0)) < len(self.surface_names)
        ]
        self.active_surface_index = max(0, min(int(self.active_surface_index), len(self.surface_names) - 1))
        self.surface_name = self.surface_names[self.active_surface_index]

    def surface_queue(self) -> List[Dict[str, object]]:
        self._sync_surface_queue()
        seed_counts = {index: 0 for index in range(len(self.surface_names))}
        for patch in self.selected_patches:
            index = int(patch.get("surface_index", 0))
            if index in seed_counts:
                seed_counts[index] += 1
        return [
            {
                "index": index,
                "name": name,
                "seed_count": seed_counts.get(index, 0),
                "active": index == self.active_surface_index,
            }
            for index, name in enumerate(self.surface_names)
        ]

    def undo_3d_action(self) -> bool:
        if self.active_curve_vertices:
            self.active_curve_vertices.pop()
            self.annotation_mode = "curve"
            self._emit_3d_state()
            self.update()
            return True
        if self.selected_patches:
            self.selected_patches.pop()
            self._emit_3d_state()
            self.update()
            return True
        if self.closed_curves:
            curve = self.closed_curves.pop()
            self._sync_surface_queue()
            vertices = [int(value) for value in curve.get("vertices", [])]
            if len(vertices) > 1 and vertices[0] == vertices[-1]:
                vertices = vertices[:-1]
            self.active_curve_vertices = vertices
            self.annotation_mode = "curve"
            self._emit_3d_state()
            self.update()
            return True
        return False

    def annotation_payload(self, mask_path: Optional[Path] = None) -> Dict[str, object]:
        if self.shell_mesh is None:
            raise ValueError("No 3D shell is loaded")
        self._sync_surface_queue()
        vertices = np.asarray(self.shell_mesh.vertices, dtype=float)
        curves = []
        for index, curve in enumerate(self.closed_curves, start=1):
            vertex_ids = [int(value) for value in curve.get("vertices", [])]
            if len(vertex_ids) < 4:
                continue
            curves.append(
                {
                    "curve_id": str(curve.get("curve_id") or f"cut_curve_{index}"),
                    "label_left": "selected",
                    "label_right": "unselected",
                    "source": "manual_3d",
                    "control_points": vertices[vertex_ids].round(4).tolist(),
                }
            )
        patches = []
        for patch in self.selected_patches:
            surface_index = int(patch.get("surface_index", 0))
            if 0 <= surface_index < len(self.surface_names):
                surface_label = self.surface_names[surface_index]
            else:
                surface_label = str(patch.get("patch_label") or "surface")
            patches.append(
                {
                    "patch_label": surface_label,
                    "surface_index": surface_index,
                    "source": "manual_3d",
                    "face_id": int(patch.get("face_id", -1)),
                    "seed_point": np.asarray(patch.get("seed_point"), dtype=float).round(4).tolist(),
                }
            )
        return {
            "schema": "laminar_boundary_builder.surface_3d_annotations.v1",
            "annotation_type": "manual_3d_shell_patch",
            "mask_path": str(mask_path) if mask_path else None,
            "cut_curves": curves,
            "selected_patches": patches,
        }

    @staticmethod
    def _loaded_curve_vertex_ids(points: object, vertices: np.ndarray) -> List[int]:
        loaded = np.asarray(points, dtype=float)
        if loaded.ndim != 2 or loaded.shape[1] != 3:
            raise ValueError("3D cut curve control_points must be an N x 3 array")
        snapped: List[int] = []
        for point in loaded:
            distances = np.linalg.norm(vertices - point.reshape(1, 3), axis=1)
            vertex_id = int(np.argmin(distances))
            if not snapped or snapped[-1] != vertex_id:
                snapped.append(vertex_id)
        if len(snapped) >= 2 and snapped[0] != snapped[-1]:
            snapped.append(snapped[0])
        return snapped

    def _loaded_seed_face_id(self, seed_point: object, fallback_face_id: int) -> Optional[int]:
        if self.shell_mesh is None or len(self._faces) == 0:
            return None
        vertices = np.asarray(self.shell_mesh.vertices, dtype=float)
        faces = np.asarray(self._faces, dtype=int)
        if seed_point is None:
            return int(fallback_face_id) if 0 <= int(fallback_face_id) < len(faces) else None
        seed = np.asarray(seed_point, dtype=float).reshape(-1)
        if seed.shape[0] != 3:
            return int(fallback_face_id) if 0 <= int(fallback_face_id) < len(faces) else None
        centers: List[np.ndarray] = []
        face_ids: List[int] = []
        for face_id, face in enumerate(faces):
            valid = [int(value) for value in face if 0 <= int(value) < len(vertices)]
            if not valid:
                continue
            centers.append(vertices[np.asarray(valid, dtype=int)].mean(axis=0))
            face_ids.append(int(face_id))
        if not centers:
            return None
        center_array = np.vstack(centers)
        index = int(np.linalg.norm(center_array - seed.reshape(1, 3), axis=1).argmin())
        return int(face_ids[index])

    def load_annotation_payload(self, payload: Dict[str, object]) -> Dict[str, int]:
        if self.shell_mesh is None:
            raise ValueError("Load the target 3D shell before loading saved 3D annotations")
        if payload.get("schema") != "laminar_boundary_builder.surface_3d_annotations.v1":
            raise ValueError("Expected surface_3d_annotations JSON from the 3D surface workflow")
        vertices = np.asarray(self.shell_mesh.vertices, dtype=float)
        if len(vertices) == 0:
            raise ValueError("Current 3D shell has no vertices")

        curve_items = payload.get("cut_curves", payload.get("curves", []))
        if not isinstance(curve_items, list) or not curve_items:
            raise ValueError("3D annotation JSON has no cut curves")

        closed_curves: List[Dict[str, object]] = []
        skipped_curves = 0
        for index, item in enumerate(curve_items):
            if not isinstance(item, dict):
                skipped_curves += 1
                continue
            points = item.get("control_points", item.get("points"))
            if points is None:
                skipped_curves += 1
                continue
            try:
                vertex_ids = self._loaded_curve_vertex_ids(points, vertices)
            except Exception:
                skipped_curves += 1
                continue
            if len(vertex_ids) < 4 or len(set(vertex_ids[:-1])) < 3:
                skipped_curves += 1
                continue
            closed_curves.append(
                {
                    "curve_id": str(item.get("curve_id") or f"cut_curve_{index + 1}"),
                    "vertices": vertex_ids,
                }
            )

        if not closed_curves:
            raise ValueError("No usable 3D cut curves were loaded")

        patch_items = payload.get("selected_patches", payload.get("seeds", payload.get("patch_seeds", [])))
        patch_items = patch_items if isinstance(patch_items, list) else []
        surface_names = [self._default_surface_name(index) for index in range(len(closed_curves))]
        patch_records: List[Tuple[int, int, np.ndarray]] = []
        skipped_patches = 0
        for patch_index, item in enumerate(patch_items):
            if not isinstance(item, dict):
                skipped_patches += 1
                continue
            try:
                surface_index = int(item.get("surface_index", patch_index))
            except Exception:
                surface_index = patch_index
            if surface_index < 0 or surface_index >= len(surface_names):
                skipped_patches += 1
                continue
            label = str(item.get("patch_label") or "").strip()
            if label:
                surface_names[surface_index] = label
            try:
                fallback_face_id = int(item.get("face_id", -1))
            except Exception:
                fallback_face_id = -1
            face_id = self._loaded_seed_face_id(item.get("seed_point"), fallback_face_id)
            if face_id is None:
                skipped_patches += 1
                continue
            face = np.asarray([value for value in self._faces[int(face_id)] if int(value) >= 0], dtype=int)
            if len(face) == 0:
                skipped_patches += 1
                continue
            seed_point = vertices[face].mean(axis=0)
            patch_records.append((surface_index, int(face_id), seed_point))

        self.closed_curves = closed_curves
        self.active_curve_vertices = []
        self.surface_names = surface_names
        self.active_surface_index = 0
        self.surface_name = surface_names[0]
        self.selected_patches = [
            {
                "patch_label": surface_names[surface_index],
                "surface_index": int(surface_index),
                "face_id": int(face_id),
                "seed_point": seed_point,
            }
            for surface_index, face_id, seed_point in patch_records
        ]
        self.hover_face = None
        self.annotation_mode = "patch" if self.closed_curves else "curve"
        self.message = (
            f"Loaded {len(self.closed_curves)} 3D curve(s) and "
            f"{len(self.selected_patches)} seed patch(es). Drag points to refine, or build again."
        )
        self._emit_3d_state()
        self.update()
        return {
            "curves": len(self.closed_curves),
            "patches": len(self.selected_patches),
            "skipped_curves": skipped_curves,
            "skipped_patches": skipped_patches,
        }

    def _emit_3d_state(self) -> None:
        self.annotation_changed.emit()
        self.build_ready_changed.emit(self.can_build_3d_surfaces())

    def _clear_mesh_buffers(self) -> None:
        self._interleaved = np.empty((0, 9), dtype=np.float32)
        self._edge_vertices = np.empty((0, 3), dtype=np.float32)
        self._vertex_count = 0
        self._edge_vertex_count = 0
        self._vertices = np.empty((0, 3), dtype=np.float32)
        self._faces = np.empty((0, 3), dtype=np.int64)
        self._normalized_vertices = np.empty((0, 3), dtype=np.float32)
        self._display_vertex_normals = np.empty((0, 3), dtype=np.float32)
        self._display_face_ids = np.empty((0,), dtype=np.int64)
        self._display_triangles = np.empty((0, 3), dtype=np.int64)
        self._display_centers = np.empty((0, 3), dtype=np.float32)
        self._display_center_face_ids = np.empty((0,), dtype=np.int64)
        self._curve_edge_graph = {}
        self._curve_edge_set = set()
        self._curve_graph_vertices = np.empty((0, 3), dtype=np.float32)
        self._curve_path_cache = {}
        self._clear_pick_caches()
        self._mesh_dirty = True

    def _prepare_mesh_arrays(self, shell_mesh) -> None:
        vertices = np.asarray(shell_mesh.vertices, dtype=np.float32)
        faces = np.asarray(shell_mesh.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or len(faces) == 0:
            self._clear_mesh_buffers()
            return

        self._vertices = vertices
        self._faces = faces
        self._center = ((vertices.min(axis=0) + vertices.max(axis=0)) * 0.5).astype(np.float32)
        radius = float(np.linalg.norm(vertices - self._center.reshape(1, 3), axis=1).max())
        self._radius = max(radius, 1.0)
        self._normalized_vertices = ((vertices - self._center.reshape(1, 3)) / self._radius).astype(np.float32)

        if len(faces) > self.MAX_DISPLAY_FACES:
            face_ids = np.linspace(0, len(faces) - 1, self.MAX_DISPLAY_FACES, dtype=np.int64)
        else:
            face_ids = np.arange(len(faces), dtype=np.int64)
        triangles, triangle_face_ids = self._triangulate_faces(faces, face_ids)
        if len(triangles) == 0:
            self._clear_mesh_buffers()
            return

        positions = self._normalized_vertices[triangles.reshape(-1)]
        tri_points = positions.reshape(-1, 3, 3)
        face_normals = np.cross(tri_points[:, 1] - tri_points[:, 0], tri_points[:, 2] - tri_points[:, 0])
        centers = tri_points.mean(axis=1)
        inward = np.sum(face_normals * centers, axis=1) < 0
        face_normals[inward] *= -1.0
        lengths = np.linalg.norm(face_normals, axis=1)
        lengths[lengths == 0] = 1.0
        face_normals = (face_normals / lengths.reshape(-1, 1)).astype(np.float32)
        vertex_normals = self._smooth_vertex_normals(triangles, face_normals)
        repeated_normals = vertex_normals[triangles.reshape(-1)]

        colors = self._face_ids_to_colors(triangle_face_ids)
        repeated_colors = np.repeat(colors, 3, axis=0)
        self._interleaved = np.column_stack((positions, repeated_normals, repeated_colors)).astype(np.float32)
        self._edge_vertices = self._triangle_edge_vertices(triangles, vertex_normals)
        self._vertex_count = len(self._interleaved)
        self._edge_vertex_count = len(self._edge_vertices)
        self._display_face_ids = np.asarray(triangle_face_ids, dtype=np.int64)
        self._display_triangles = np.asarray(triangles, dtype=np.int64)
        self._display_centers = self._normalized_vertices[triangles].mean(axis=1).astype(np.float32)
        self._display_center_face_ids = np.asarray(triangle_face_ids, dtype=np.int64)
        self._display_vertex_normals = vertex_normals
        spacing = np.asarray(getattr(shell_mesh, "spacing", (1.0, 1.0, 1.0)), dtype=np.float32)
        if spacing.shape != (3,):
            spacing = np.ones(3, dtype=np.float32)
        self._curve_graph_vertices = (self._vertices * spacing.reshape(1, 3)).astype(np.float32)
        self._curve_edge_graph, self._curve_edge_set = self._build_curve_edge_graph(faces)
        self._curve_path_cache = {}
        self._screen_cache_key = None
        self._screen_cache = None
        self._clear_pick_caches()
        self._mesh_dirty = True
        if self._gl_ready:
            self.makeCurrent()
            self._upload_mesh()
            self.doneCurrent()

    @staticmethod
    def _triangulate_faces(faces: np.ndarray, face_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        triangles: List[List[int]] = []
        triangle_face_ids: List[int] = []
        for face_id in face_ids:
            face = [int(value) for value in faces[int(face_id)] if int(value) >= 0]
            if len(face) < 3:
                continue
            if len(face) == 3:
                triangles.append(face)
                triangle_face_ids.append(int(face_id))
                continue
            for index in range(1, len(face) - 1):
                triangles.append([face[0], face[index], face[index + 1]])
                triangle_face_ids.append(int(face_id))
        if not triangles:
            return np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.int64)
        return np.asarray(triangles, dtype=np.int64), np.asarray(triangle_face_ids, dtype=np.int64)

    def _smooth_vertex_normals(self, triangles: np.ndarray, face_normals: np.ndarray) -> np.ndarray:
        normals = np.zeros_like(self._normalized_vertices, dtype=np.float32)
        if len(triangles) == 0:
            return normals
        np.add.at(normals, triangles[:, 0], face_normals)
        np.add.at(normals, triangles[:, 1], face_normals)
        np.add.at(normals, triangles[:, 2], face_normals)
        lengths = np.linalg.norm(normals, axis=1)
        empty = lengths <= 1e-8
        if np.any(empty):
            fallback = np.asarray(self._normalized_vertices[empty], dtype=np.float32)
            fallback_lengths = np.linalg.norm(fallback, axis=1)
            fallback_lengths[fallback_lengths <= 1e-8] = 1.0
            normals[empty] = fallback / fallback_lengths.reshape(-1, 1)
            lengths = np.linalg.norm(normals, axis=1)
        lengths[lengths <= 1e-8] = 1.0
        return (normals / lengths.reshape(-1, 1)).astype(np.float32)

    def _triangle_edge_vertices(self, triangles: np.ndarray, vertex_normals: np.ndarray) -> np.ndarray:
        if len(triangles) == 0:
            return np.empty((0, 3), dtype=np.float32)
        edges = np.concatenate(
            (
                triangles[:, [0, 1]],
                triangles[:, [1, 2]],
                triangles[:, [2, 0]],
            ),
            axis=0,
        )
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)
        if len(edges) > self.MAX_EDGE_SEGMENTS:
            edge_ids = np.linspace(0, len(edges) - 1, self.MAX_EDGE_SEGMENTS, dtype=np.int64)
            edges = edges[edge_ids]
        edge_vertices = self._normalized_vertices[edges.reshape(-1)]
        edge_normals = vertex_normals[edges.reshape(-1)]
        return (edge_vertices + edge_normals * 0.0015).astype(np.float32)

    @staticmethod
    def _edge_key(left: int, right: int) -> Tuple[int, int]:
        left = int(left)
        right = int(right)
        return (left, right) if left < right else (right, left)

    def _build_curve_edge_graph(self, faces: np.ndarray) -> tuple[Dict[int, List[Tuple[int, float]]], set[Tuple[int, int]]]:
        graph: Dict[int, List[Tuple[int, float]]] = {}
        edge_set: set[Tuple[int, int]] = set()
        vertices = self._curve_graph_vertices
        if len(vertices) == 0:
            return graph, edge_set
        for face in np.asarray(faces, dtype=np.int64):
            face_vertices = [int(value) for value in face if int(value) >= 0]
            if len(face_vertices) < 2:
                continue
            for left, right in zip(face_vertices, face_vertices[1:] + face_vertices[:1]):
                if left == right:
                    continue
                key = self._edge_key(left, right)
                if key in edge_set:
                    continue
                edge_set.add(key)
                weight = float(np.linalg.norm(vertices[left] - vertices[right]))
                if not math.isfinite(weight) or weight <= 0:
                    weight = 1.0
                graph.setdefault(left, []).append((right, weight))
                graph.setdefault(right, []).append((left, weight))
        return graph, edge_set

    @staticmethod
    def _face_ids_to_colors(face_ids: np.ndarray) -> np.ndarray:
        codes = np.asarray(face_ids, dtype=np.uint32) + np.uint32(1)
        red = (codes & np.uint32(255)).astype(np.float32) / 255.0
        green = ((codes >> np.uint32(8)) & np.uint32(255)).astype(np.float32) / 255.0
        blue = ((codes >> np.uint32(16)) & np.uint32(255)).astype(np.float32) / 255.0
        return np.column_stack((red, green, blue)).astype(np.float32)

    def initializeGL(self) -> None:
        try:
            glClearColor(0.045, 0.075, 0.07, 1.0)
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LEQUAL)
            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)
            self._surface_program = self._create_surface_program()
            self._line_program = self._create_line_program()
            self._vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
            self._vbo.create()
            self._edge_vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
            self._edge_vbo.create()
            self._annotation_line_vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
            self._annotation_line_vbo.create()
            self._annotation_point_vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
            self._annotation_point_vbo.create()
            self._gl_ready = True
            self._upload_mesh()
        except Exception as exc:
            self._gl_ready = False
            self.message = f"OpenGL view failed to initialize: {exc}"

    def resizeGL(self, width: int, height: int) -> None:
        self._viewport_width = max(1, int(width))
        self._viewport_height = max(1, int(height))
        glViewport(0, 0, self._viewport_width, self._viewport_height)
        self._screen_cache_key = None
        self._screen_cache = None
        self._clear_pick_caches()

    def paintGL(self) -> None:
        try:
            self._paint_gl()
        except Exception as exc:
            self.message = f"3D render failed: {exc}"
            glClearColor(0.045, 0.075, 0.07, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            try:
                self._draw_overlay()
            except Exception:
                pass

    def _paint_gl(self) -> None:
        glClearColor(0.045, 0.075, 0.07, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self._gl_ready and self._vertex_count and self._surface_program is not None:
            self._draw_surface()
            self._draw_edges()
            self._draw_annotations()
        self._draw_overlay()

    def _create_surface_program(self) -> QOpenGLShaderProgram:
        program = QOpenGLShaderProgram(self)
        vertex_shader = """
            attribute vec3 position;
            attribute vec3 normal;
            uniform mat4 mvp;
            varying vec3 v_normal;
            void main() {
                gl_Position = mvp * vec4(position, 1.0);
                v_normal = normalize(normal);
            }
        """
        fragment_shader = """
            varying vec3 v_normal;
            uniform vec3 base_color;
            uniform vec3 light_dir;
            void main() {
                vec3 n = normalize(v_normal);
                float diffuse = max(dot(n, normalize(light_dir)), 0.0);
                float back = max(dot(n, normalize(vec3(-0.45, -0.20, 0.70))), 0.0);
                float rim = pow(1.0 - abs(n.z), 1.35) * 0.10;
                float shade = 0.46 + diffuse * 0.46 + back * 0.10 + rim;
                vec3 color = base_color * shade + vec3(rim * 0.18);
                gl_FragColor = vec4(color, 1.0);
            }
        """
        if not program.addShaderFromSourceCode(QOpenGLShader.Vertex, vertex_shader):
            raise RuntimeError(program.log())
        if not program.addShaderFromSourceCode(QOpenGLShader.Fragment, fragment_shader):
            raise RuntimeError(program.log())
        if not program.link():
            raise RuntimeError(program.log())
        return program

    def _create_line_program(self) -> QOpenGLShaderProgram:
        program = QOpenGLShaderProgram(self)
        vertex_shader = """
            attribute vec3 position;
            uniform mat4 mvp;
            uniform float depth_bias;
            void main() {
                vec4 clip = mvp * vec4(position, 1.0);
                clip.z -= depth_bias * clip.w;
                gl_Position = clip;
            }
        """
        fragment_shader = """
            uniform vec4 line_color;
            void main() {
                gl_FragColor = line_color;
            }
        """
        if not program.addShaderFromSourceCode(QOpenGLShader.Vertex, vertex_shader):
            raise RuntimeError(program.log())
        if not program.addShaderFromSourceCode(QOpenGLShader.Fragment, fragment_shader):
            raise RuntimeError(program.log())
        if not program.link():
            raise RuntimeError(program.log())
        return program

    def _upload_mesh(self) -> None:
        if not self._gl_ready or self._vbo is None or not self._vbo.isCreated():
            return
        self._vbo.bind()
        if self._interleaved.size:
            data = np.ascontiguousarray(self._interleaved, dtype=np.float32)
            self._vbo.allocate(data.tobytes(), int(data.nbytes))
        else:
            self._vbo.allocate(b"", 0)
        self._vbo.release()
        if self._edge_vbo is not None and self._edge_vbo.isCreated():
            self._edge_vbo.bind()
            if self._edge_vertices.size:
                edge_data = np.ascontiguousarray(self._edge_vertices, dtype=np.float32)
                self._edge_vbo.allocate(edge_data.tobytes(), int(edge_data.nbytes))
            else:
                self._edge_vbo.allocate(b"", 0)
            self._edge_vbo.release()
        self._mesh_dirty = False

    def _draw_surface(self) -> None:
        if self._mesh_dirty:
            self._upload_mesh()
        if self._vbo is None or self._surface_program is None:
            return
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        program = self._surface_program
        program.bind()
        program.setUniformValue("mvp", self._mvp_matrix())
        program.setUniformValue("base_color", QVector3D(0.70, 0.88, 0.82))
        program.setUniformValue("light_dir", QVector3D(-0.35, 0.65, 0.72))
        self._vbo.bind()
        stride = 9 * 4
        program.enableAttributeArray("position")
        program.setAttributeBuffer("position", GL_FLOAT, 0, 3, stride)
        program.enableAttributeArray("normal")
        program.setAttributeBuffer("normal", GL_FLOAT, 3 * 4, 3, stride)
        glDrawArrays(GL_TRIANGLES, 0, int(self._vertex_count))
        self._vbo.release()
        program.disableAttributeArray("position")
        program.disableAttributeArray("normal")
        program.release()

    def _draw_edges(self) -> None:
        if not self._edge_vertex_count or self._edge_vbo is None or self._line_program is None:
            return
        if self._mesh_dirty:
            self._upload_mesh()
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glDepthMask(GL_FALSE)
        glLineWidth(1.25)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        try:
            program = self._line_program
            program.bind()
            program.setUniformValue("mvp", self._mvp_matrix())
            program.setUniformValue("depth_bias", 0.0)
            program.setUniformValue("line_color", QVector4D(0.015, 0.075, 0.065, 0.58))
            self._edge_vbo.bind()
            program.enableAttributeArray("position")
            program.setAttributeBuffer("position", GL_FLOAT, 0, 3, 3 * 4)
            glDrawArrays(GL_LINES, 0, int(self._edge_vertex_count))
            self._edge_vbo.release()
            program.disableAttributeArray("position")
            program.release()
        finally:
            glDepthMask(GL_TRUE)
            glDisable(GL_BLEND)

    def _draw_annotations(self) -> None:
        if (
            self._line_program is None
            or self._annotation_line_vbo is None
            or self._annotation_point_vbo is None
            or len(self._normalized_vertices) == 0
        ):
            return
        line_vertices = self._annotation_line_vertices()
        marker_vertices, active_tip = self._annotation_marker_vertices()
        if line_vertices.size == 0 and marker_vertices.size == 0:
            return

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glDepthMask(GL_FALSE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)
        mvp = self._mvp_matrix()
        scale = self._framebuffer_scale()
        try:
            if line_vertices.size:
                self._draw_annotation_line_pass(line_vertices, mvp, QVector4D(0.01, 0.03, 0.025, 0.95), 5.2 * scale)
                self._draw_annotation_line_pass(line_vertices, mvp, QVector4D(1.0, 0.23, 0.19, 1.0), 3.0 * scale)
            if marker_vertices.size:
                self._draw_annotation_point_pass(marker_vertices, mvp, QVector4D(0.01, 0.03, 0.025, 0.95), 17.0 * scale)
                self._draw_annotation_point_pass(marker_vertices, mvp, QVector4D(1.0, 1.0, 1.0, 1.0), 12.5 * scale)
                self._draw_annotation_point_pass(marker_vertices, mvp, QVector4D(1.0, 0.88, 0.20, 1.0), 9.5 * scale)
            if active_tip.size:
                self._draw_annotation_point_pass(active_tip, mvp, QVector4D(1.0, 0.42, 0.24, 1.0), 4.8 * scale)
        finally:
            glDepthMask(GL_TRUE)
            glDisable(GL_BLEND)
            glDisable(GL_POINT_SMOOTH)

    def _draw_annotation_line_pass(
        self,
        vertices: np.ndarray,
        mvp: QMatrix4x4,
        color: QVector4D,
        width: float,
    ) -> None:
        if self._annotation_line_vbo is None or self._line_program is None:
            return
        data = np.ascontiguousarray(vertices, dtype=np.float32)
        glLineWidth(max(1.0, float(width)))
        program = self._line_program
        program.bind()
        program.setUniformValue("mvp", mvp)
        program.setUniformValue("depth_bias", float(self.ANNOTATION_DEPTH_BIAS))
        program.setUniformValue("line_color", color)
        self._annotation_line_vbo.bind()
        self._annotation_line_vbo.allocate(data.tobytes(), int(data.nbytes))
        program.enableAttributeArray("position")
        program.setAttributeBuffer("position", GL_FLOAT, 0, 3, 3 * 4)
        glDrawArrays(GL_LINES, 0, int(len(data)))
        self._annotation_line_vbo.release()
        program.disableAttributeArray("position")
        program.release()

    def _draw_annotation_point_pass(
        self,
        vertices: np.ndarray,
        mvp: QMatrix4x4,
        color: QVector4D,
        point_size: float,
    ) -> None:
        if self._annotation_point_vbo is None or self._line_program is None:
            return
        data = np.ascontiguousarray(vertices, dtype=np.float32)
        glPointSize(max(1.0, float(point_size)))
        program = self._line_program
        program.bind()
        program.setUniformValue("mvp", mvp)
        program.setUniformValue("depth_bias", float(self.ANNOTATION_DEPTH_BIAS))
        program.setUniformValue("line_color", color)
        self._annotation_point_vbo.bind()
        self._annotation_point_vbo.allocate(data.tobytes(), int(data.nbytes))
        program.enableAttributeArray("position")
        program.setAttributeBuffer("position", GL_FLOAT, 0, 3, 3 * 4)
        glDrawArrays(GL_POINTS, 0, int(len(data)))
        self._annotation_point_vbo.release()
        program.disableAttributeArray("position")
        program.release()

    def _annotation_line_vertices(self) -> np.ndarray:
        paths: List[List[int]] = []
        for curve in self.closed_curves:
            paths.append(self._curve_vertex_path([int(value) for value in curve.get("vertices", [])], closed=True))
        if self.active_curve_vertices:
            paths.append(self._curve_vertex_path(list(self.active_curve_vertices), closed=False))
        line_ids: List[int] = []
        for path in paths:
            for left, right in zip(path[:-1], path[1:]):
                if left == right:
                    continue
                line_ids.extend((int(left), int(right)))
        if not line_ids:
            return np.empty((0, 3), dtype=np.float32)
        return self._normalized_vertices[np.asarray(line_ids, dtype=np.int64)].astype(np.float32)

    def _annotation_marker_vertices(self) -> Tuple[np.ndarray, np.ndarray]:
        marker_ids: List[int] = []
        for curve in self.closed_curves:
            marker_ids.extend(self._marker_vertex_ids([int(value) for value in curve.get("vertices", [])]))
        if self.active_curve_vertices:
            marker_ids.extend(self._marker_vertex_ids(list(self.active_curve_vertices)))
        markers = (
            self._normalized_vertices[np.asarray(marker_ids, dtype=np.int64)].astype(np.float32)
            if marker_ids
            else np.empty((0, 3), dtype=np.float32)
        )
        active_tip = np.empty((0, 3), dtype=np.float32)
        if self.active_curve_vertices:
            tip_id = int(self.active_curve_vertices[-1])
            if 0 <= tip_id < len(self._normalized_vertices):
                active_tip = self._normalized_vertices[np.asarray([tip_id], dtype=np.int64)].astype(np.float32)
        return markers, active_tip

    def _framebuffer_scale(self) -> float:
        width_scale = self._viewport_width / float(max(1, self.width()))
        height_scale = self._viewport_height / float(max(1, self.height()))
        return max(1.0, min(max(width_scale, height_scale), 3.0))

    def _marker_vertex_ids(self, vertex_ids: List[int]) -> List[int]:
        marker_ids = [int(value) for value in vertex_ids if 0 <= int(value) < len(self._normalized_vertices)]
        if len(marker_ids) > 1 and marker_ids[0] == marker_ids[-1]:
            marker_ids = marker_ids[:-1]
        return marker_ids

    def _mvp_matrix(self) -> QMatrix4x4:
        width = max(1, self.width())
        height = max(1, self.height())
        aspect = width / height
        projection = QMatrix4x4()
        projection.ortho(-aspect, aspect, -1.0, 1.0, self.PROJECTION_NEAR, self.PROJECTION_FAR)
        model = QMatrix4x4()
        pan_world_x = float(self.pan_x) * aspect / (width * 0.5)
        pan_world_y = -float(self.pan_y) / (height * 0.5)
        model.translate(pan_world_x, pan_world_y, 0.0)
        model.scale(self.MODEL_BASE_SCALE * float(self.preview_zoom))
        model.rotate(math.degrees(float(self.rotation_pitch)), 1.0, 0.0, 0.0)
        model.rotate(math.degrees(float(self.rotation_yaw)), 0.0, 1.0, 0.0)
        return projection * model

    def _screen_cache_data(self):
        key = (
            len(self._normalized_vertices),
            self.width(),
            self.height(),
            round(float(self.rotation_yaw), 5),
            round(float(self.rotation_pitch), 5),
            round(float(self.preview_zoom), 5),
            round(float(self.pan_x), 3),
            round(float(self.pan_y), 3),
        )
        if key == self._screen_cache_key and self._screen_cache is not None:
            return self._screen_cache
        if len(self._normalized_vertices) == 0:
            self._screen_cache_key = key
            self._screen_cache = (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
            return self._screen_cache
        screen, front_depth, window_depth = self._project_points(self._normalized_vertices)
        self._screen_cache_key = key
        self._screen_cache = (screen, front_depth, window_depth)
        self._clear_pick_caches()
        return self._screen_cache

    def _project_points(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        coords = np.asarray(points, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) == 0:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        matrix = np.asarray(self._mvp_matrix().copyDataTo(), dtype=np.float32).reshape(4, 4)
        homogeneous = np.column_stack((coords, np.ones(len(coords), dtype=np.float32)))
        clip = homogeneous @ matrix.T
        w = clip[:, 3]
        safe_w = np.where(np.abs(w) > 1e-8, w, 1.0)
        ndc = clip[:, :3] / safe_w.reshape(-1, 1)
        width = float(max(1, self.width()))
        height = float(max(1, self.height()))
        screen = np.column_stack(
            (
                (ndc[:, 0] * 0.5 + 0.5) * width,
                (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height,
            )
        ).astype(np.float32)
        window_depth = np.clip(ndc[:, 2] * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)
        front_depth = (-window_depth).astype(np.float32)
        return screen, front_depth, window_depth

    def _nearest_active_vertex_index(self, pos, max_distance: Optional[float] = None) -> Optional[int]:
        if max_distance is None:
            max_distance = self.ANNOTATION_POINT_HIT_RADIUS_PX
        if not self.active_curve_vertices:
            return None
        screen, front_depth, _window_depth = self._screen_cache_data()
        ids = np.asarray(self.active_curve_vertices, dtype=np.int64)
        click = np.asarray([pos.x(), pos.y()], dtype=np.float32)
        distances = np.linalg.norm(screen[ids] - click.reshape(1, 2), axis=1)
        for index in np.argsort(distances):
            if float(distances[int(index)]) > max_distance:
                break
            vertex_id = int(ids[int(index)])
            if self._vertex_is_visible(vertex_id, screen, front_depth):
                return int(index)
        return None

    def _nearest_annotation_point_ref(
        self,
        pos,
        max_distance: Optional[float] = None,
    ) -> Optional[Tuple[str, int, int]]:
        if max_distance is None:
            max_distance = self.ANNOTATION_POINT_HIT_RADIUS_PX
        screen, front_depth, _window_depth = self._screen_cache_data()
        if len(screen) == 0:
            return None
        click = np.asarray([pos.x(), pos.y()], dtype=np.float32)
        candidates: List[Tuple[float, float, str, int, int, int]] = []

        def consider(kind: str, curve_index: int, point_index: int, vertex_id: int) -> None:
            if vertex_id < 0 or vertex_id >= len(screen):
                return
            distance = float(np.linalg.norm(screen[vertex_id] - click))
            vertex_depth = float(front_depth[vertex_id])
            if distance > float(max_distance):
                return
            candidates.append((distance, -vertex_depth, kind, int(curve_index), int(point_index), int(vertex_id)))

        for point_index, vertex_id in enumerate(self.active_curve_vertices):
            consider("active", -1, point_index, int(vertex_id))

        for curve_index, curve in enumerate(self.closed_curves):
            vertices = [int(value) for value in curve.get("vertices", [])]
            if len(vertices) > 1 and vertices[0] == vertices[-1]:
                vertices = vertices[:-1]
            for point_index, vertex_id in enumerate(vertices):
                consider("closed", curve_index, point_index, int(vertex_id))

        for _distance, _negative_depth, kind, curve_index, point_index, vertex_id in sorted(candidates):
            if self._vertex_is_visible(vertex_id, screen, front_depth):
                return (kind, int(curve_index), int(point_index))
        return None

    def _front_hit_at_screen_point(
        self,
        screen_point: np.ndarray,
        max_triangles: Optional[int] = None,
        margin: float = 2.0,
    ) -> Optional[Tuple[int, float]]:
        if len(self._display_triangles) == 0:
            return None
        screen, front_depth, _window_depth = self._screen_cache_data()
        click = np.asarray(screen_point, dtype=np.float32).reshape(2)
        cache_key = (
            int(round(float(click[0]))),
            int(round(float(click[1]))),
            int(max_triangles) if max_triangles is not None else -1,
            int(round(float(margin) * 10.0)),
        )
        if cache_key in self._front_hit_cache:
            return self._front_hit_cache[cache_key]

        candidate_triangle_ids, candidate_points = self._pick_bbox_candidate_data(click, margin=margin)
        if len(candidate_triangle_ids) == 0:
            self._front_hit_cache[cache_key] = None
            return None
        if max_triangles is not None and len(candidate_triangle_ids) > max_triangles:
            step = int(math.ceil(len(candidate_triangle_ids) / float(max_triangles)))
            candidate_triangle_ids = candidate_triangle_ids[::step]
            candidate_points = candidate_points[::step]

        triangles = self._display_triangles
        face_ids = self._display_face_ids
        points = candidate_points
        inside_bounds = (
            (points[:, :, 0].min(axis=1) <= click[0] + margin)
            & (points[:, :, 0].max(axis=1) >= click[0] - margin)
            & (points[:, :, 1].min(axis=1) <= click[1] + margin)
            & (points[:, :, 1].max(axis=1) >= click[1] - margin)
        )
        candidates = candidate_triangle_ids[np.flatnonzero(inside_bounds)]
        if len(candidates) == 0:
            self._front_hit_cache[cache_key] = None
            return None

        candidate_points = self._pick_bbox_points[candidates]
        p0 = candidate_points[:, 0, :]
        p1 = candidate_points[:, 1, :]
        p2 = candidate_points[:, 2, :]
        den = (
            (p1[:, 1] - p2[:, 1]) * (p0[:, 0] - p2[:, 0])
            + (p2[:, 0] - p1[:, 0]) * (p0[:, 1] - p2[:, 1])
        )
        valid = np.abs(den) > 1e-6
        if not np.any(valid):
            self._front_hit_cache[cache_key] = None
            return None

        candidate_ids = candidates[valid]
        p0 = p0[valid]
        p1 = p1[valid]
        p2 = p2[valid]
        den = den[valid]
        bary0 = (
            (p1[:, 1] - p2[:, 1]) * (click[0] - p2[:, 0])
            + (p2[:, 0] - p1[:, 0]) * (click[1] - p2[:, 1])
        ) / den
        bary1 = (
            (p2[:, 1] - p0[:, 1]) * (click[0] - p2[:, 0])
            + (p0[:, 0] - p2[:, 0]) * (click[1] - p2[:, 1])
        ) / den
        bary2 = 1.0 - bary0 - bary1
        inside = (bary0 >= -0.02) & (bary1 >= -0.02) & (bary2 >= -0.02)
        if not np.any(inside):
            self._front_hit_cache[cache_key] = None
            return None

        hit_ids = candidate_ids[inside]
        hit_triangles = triangles[hit_ids]
        hit_depth = (
            bary0[inside] * front_depth[hit_triangles[:, 0]]
            + bary1[inside] * front_depth[hit_triangles[:, 1]]
            + bary2[inside] * front_depth[hit_triangles[:, 2]]
        )
        best = int(np.argmax(hit_depth))
        result = (int(face_ids[int(hit_ids[best])]), float(hit_depth[best]))
        self._front_hit_cache[cache_key] = result
        return result

    def _pick_bbox_candidate_data(self, click: np.ndarray, margin: float) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_pick_bbox_cache(margin=margin)
        if len(self._pick_bbox_points) == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0, 3, 2), dtype=np.float32)
        point = np.asarray(click, dtype=np.float32).reshape(2)
        inside_bounds = (
            (self._pick_bbox_mins[:, 0] <= point[0] + float(margin))
            & (self._pick_bbox_maxs[:, 0] >= point[0] - float(margin))
            & (self._pick_bbox_mins[:, 1] <= point[1] + float(margin))
            & (self._pick_bbox_maxs[:, 1] >= point[1] - float(margin))
        )
        candidate_ids = np.flatnonzero(inside_bounds)
        return candidate_ids, self._pick_bbox_points[candidate_ids]

    def _ensure_pick_bbox_cache(self, margin: float = 2.0) -> None:
        screen, _front_depth, _window_depth = self._screen_cache_data()
        key = (
            self._screen_cache_key,
            len(self._display_triangles),
            int(round(float(margin) * 10.0)),
        )
        if self._pick_bbox_cache_key == key:
            return
        self._pick_bbox_cache_key = key
        self._pick_bbox_points = np.empty((0, 3, 2), dtype=np.float32)
        self._pick_bbox_mins = np.empty((0, 2), dtype=np.float32)
        self._pick_bbox_maxs = np.empty((0, 2), dtype=np.float32)
        if len(self._display_triangles) == 0:
            return

        points = screen[self._display_triangles]
        self._pick_bbox_points = points.astype(np.float32, copy=False)
        self._pick_bbox_mins = (points.min(axis=1) - float(margin)).astype(np.float32)
        self._pick_bbox_maxs = (points.max(axis=1) + float(margin)).astype(np.float32)

    def _screen_depth_is_visible(self, screen_point: np.ndarray, front_depth: float) -> bool:
        hit = self._front_hit_at_screen_point(screen_point)
        if hit is None:
            return False
        _face_id, visible_depth = hit
        return float(front_depth) >= float(visible_depth) - self.PICK_VISIBLE_DEPTH_TOLERANCE

    def _vertex_is_visible(self, vertex_id: int, screen: np.ndarray, front_depth: np.ndarray) -> bool:
        vertex_id = int(vertex_id)
        cached = self._visible_vertex_cache.get(vertex_id)
        if cached is not None:
            return bool(cached)
        if vertex_id < 0 or vertex_id >= len(screen):
            self._visible_vertex_cache[vertex_id] = False
            return False
        visible = self._screen_depth_is_visible(screen[vertex_id], float(front_depth[vertex_id]))
        self._visible_vertex_cache[vertex_id] = bool(visible)
        return bool(visible)

    def _nearest_display_face_center_id(self, pos, max_distance: float = 36.0) -> Optional[int]:
        if len(self._display_centers) == 0:
            return None
        centers, front_depth, _window_depth = self._project_points(self._display_centers)
        if len(centers) > self.MAX_HOVER_FACES:
            step = int(math.ceil(len(centers) / self.MAX_HOVER_FACES))
            sample_ids = np.arange(0, len(centers), step, dtype=np.int64)
            centers_to_search = centers[sample_ids]
            face_ids_to_search = self._display_center_face_ids[sample_ids]
            depth_to_search = front_depth[sample_ids]
        else:
            centers_to_search = centers
            face_ids_to_search = self._display_center_face_ids
            depth_to_search = front_depth
        click = np.asarray([pos.x(), pos.y()], dtype=np.float32)
        distances = np.linalg.norm(centers_to_search - click.reshape(1, 2), axis=1)
        near = np.flatnonzero(distances <= max_distance)
        if len(near) == 0:
            return None
        visible = [
            int(index)
            for index in near
            if self._screen_depth_is_visible(
                centers_to_search[int(index)],
                float(depth_to_search[int(index)]),
            )
        ]
        if not visible:
            return None
        visible_array = np.asarray(visible, dtype=np.int64)
        best = visible_array[np.lexsort((distances[visible_array], -depth_to_search[visible_array]))][0]
        return int(face_ids_to_search[int(best)])

    def _pick_display_face_id(
        self,
        pos,
        max_distance: float = 36.0,
        max_triangles: Optional[int] = None,
    ) -> Optional[int]:
        click = np.asarray([pos.x(), pos.y()], dtype=np.float32)
        hit = self._front_hit_at_screen_point(click, max_triangles=max_triangles)
        if hit is None:
            return self._nearest_display_face_center_id(pos, max_distance=max_distance)
        face_id, _depth = hit
        return int(face_id)

    def _nearest_face_vertex_id(self, face_id: int, pos) -> Optional[int]:
        if len(self._faces) == 0 or face_id < 0 or face_id >= len(self._faces):
            return None
        screen, _front_depth, _window_depth = self._screen_cache_data()
        face_vertices = np.asarray([value for value in self._faces[int(face_id)] if int(value) >= 0], dtype=np.int64)
        if len(face_vertices) == 0:
            return None
        click = np.asarray([pos.x(), pos.y()], dtype=np.float32)
        distances = np.linalg.norm(screen[face_vertices] - click.reshape(1, 2), axis=1)
        return int(face_vertices[int(np.argmin(distances))])

    def _closed_curve_screen_path(self, vertex_ids: List[int]) -> np.ndarray:
        path_ids = self._curve_vertex_path(vertex_ids, closed=True)
        if len(path_ids) < 3:
            return np.empty((0, 2), dtype=np.float32)
        screen, _front_depth, _window_depth = self._screen_cache_data()
        valid_ids = [int(value) for value in path_ids if 0 <= int(value) < len(screen)]
        if len(valid_ids) < 3:
            return np.empty((0, 2), dtype=np.float32)
        return np.asarray(screen[valid_ids], dtype=np.float32)

    @staticmethod
    def _point_segment_distance_2d(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        segment = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 1e-9:
            return float(np.linalg.norm(np.asarray(point, dtype=float) - np.asarray(start, dtype=float)))
        t = float(np.dot(np.asarray(point, dtype=float) - np.asarray(start, dtype=float), segment) / length_sq)
        t = max(0.0, min(1.0, t))
        closest = np.asarray(start, dtype=float) + segment * t
        return float(np.linalg.norm(np.asarray(point, dtype=float) - closest))

    @classmethod
    def _point_polyline_distance_2d(cls, point: np.ndarray, polyline: np.ndarray, closed: bool = True) -> float:
        points = np.asarray(polyline, dtype=float).reshape(-1, 2)
        if len(points) < 2:
            return math.inf
        pairs = list(zip(points[:-1], points[1:]))
        if closed and not np.allclose(points[0], points[-1]):
            pairs.append((points[-1], points[0]))
        if not pairs:
            return math.inf
        return min(cls._point_segment_distance_2d(point, start, end) for start, end in pairs)

    @staticmethod
    def _point_in_polygon_2d(point: np.ndarray, polygon: np.ndarray) -> bool:
        points = np.asarray(polygon, dtype=float).reshape(-1, 2)
        if len(points) >= 2 and np.allclose(points[0], points[-1]):
            points = points[:-1]
        if len(points) < 3:
            return False
        x, y = float(point[0]), float(point[1])
        inside = False
        previous = points[-1]
        for current in points:
            x0, y0 = float(previous[0]), float(previous[1])
            x1, y1 = float(current[0]), float(current[1])
            crosses = (y0 > y) != (y1 > y)
            if crosses:
                denom = y1 - y0
                if abs(denom) <= 1e-9:
                    previous = current
                    continue
                edge_x = (x1 - x0) * (y - y0) / denom + x0
                if x < edge_x:
                    inside = not inside
            previous = current
        return inside

    def _curve_path_sets(self, vertex_ids: List[int], closed: bool = True) -> Tuple[set[int], set[Tuple[int, int]]]:
        path = self._curve_vertex_path(vertex_ids, closed=closed)
        vertices = {int(value) for value in path}
        edges = {
            self._edge_key(left, right)
            for left, right in zip(path[:-1], path[1:])
            if int(left) != int(right)
        }
        return vertices, edges

    def _closed_curve_reuses_path(self, vertex_ids: List[int]) -> bool:
        path = self._curve_vertex_path(vertex_ids, closed=True)
        if len(path) < 4:
            return False
        open_path = path[:-1] if path[0] == path[-1] else path
        if len(set(open_path)) != len(open_path):
            return True
        edges = [
            self._edge_key(left, right)
            for left, right in zip(path[:-1], path[1:])
            if int(left) != int(right)
        ]
        return len(set(edges)) != len(edges)

    def _curve_intersects_closed_curves(
        self,
        vertex_ids: List[int],
        ignored_curve_index: Optional[int] = None,
    ) -> bool:
        candidate_vertices, candidate_edges = self._curve_path_sets(vertex_ids, closed=True)
        if not candidate_vertices:
            return False
        for curve_index, curve in enumerate(self.closed_curves):
            if ignored_curve_index is not None and curve_index == ignored_curve_index:
                continue
            existing_ids = [int(value) for value in curve.get("vertices", [])]
            existing_vertices, existing_edges = self._curve_path_sets(existing_ids, closed=True)
            if candidate_edges & existing_edges:
                return True
            if candidate_vertices & existing_vertices:
                return True
        return False

    def _active_segment_hits_closed_curve(self, vertex_id: int) -> bool:
        if not self.active_curve_vertices:
            return False
        segment = self._curve_segment_path(int(self.active_curve_vertices[-1]), int(vertex_id))
        segment_vertices = {int(value) for value in segment}
        segment_edges = {
            self._edge_key(left, right)
            for left, right in zip(segment[:-1], segment[1:])
            if int(left) != int(right)
        }
        for curve in self.closed_curves:
            existing_ids = [int(value) for value in curve.get("vertices", [])]
            existing_vertices, existing_edges = self._curve_path_sets(existing_ids, closed=True)
            if segment_edges & existing_edges:
                return True
            if segment_vertices & existing_vertices:
                return True
        return False

    def _vertex_hits_closed_curve(self, vertex_id: int) -> bool:
        vertex_id = int(vertex_id)
        for curve in self.closed_curves:
            existing_ids = [int(value) for value in curve.get("vertices", [])]
            existing_vertices, _existing_edges = self._curve_path_sets(existing_ids, closed=True)
            if vertex_id in existing_vertices:
                return True
        return False

    def _closed_curve_vertices_after_drag(
        self,
        point_ref: Tuple[str, int, int],
        vertex_id: int,
    ) -> Optional[List[int]]:
        kind, curve_index, point_index = point_ref
        if kind != "closed" or curve_index < 0 or curve_index >= len(self.closed_curves):
            return None
        vertices = [int(value) for value in self.closed_curves[curve_index].get("vertices", [])]
        if len(vertices) < 2:
            return None
        closes_to_first = vertices[0] == vertices[-1]
        editable_count = len(vertices) - 1 if closes_to_first else len(vertices)
        if point_index < 0 or point_index >= editable_count:
            return None
        vertices[point_index] = int(vertex_id)
        if closes_to_first and point_index == 0:
            vertices[-1] = int(vertex_id)
        elif not closes_to_first and vertices[0] != vertices[-1]:
            vertices.append(vertices[0])
        return vertices

    def _drag_target_is_inside_closed_curve(
        self,
        point_ref: Tuple[str, int, int],
        vertex_id: int,
    ) -> bool:
        screen, _front_depth, _window_depth = self._screen_cache_data()
        if vertex_id < 0 or vertex_id >= len(screen):
            return False
        target = np.asarray(screen[int(vertex_id)], dtype=np.float32)
        for curve in self.closed_curves:
            vertices = [int(value) for value in curve.get("vertices", [])]
            if len(set(vertices[:-1] if len(vertices) > 1 and vertices[0] == vertices[-1] else vertices)) < 3:
                continue
            curve_path = self._closed_curve_screen_path(vertices)
            if len(curve_path) < 3:
                continue
            boundary_distance = self._point_polyline_distance_2d(target, curve_path, closed=True)
            if boundary_distance <= self.DRAG_INSIDE_CURVE_MARGIN_PX:
                continue
            if self._point_in_polygon_2d(target, curve_path):
                return True
        return False

    def _drag_target_keeps_curve_shape(
        self,
        point_ref: Tuple[str, int, int],
        vertex_id: int,
    ) -> bool:
        replacement = self._closed_curve_vertices_after_drag(point_ref, vertex_id)
        if replacement is None:
            return True
        if len(set(replacement[:-1])) < 3:
            self._last_drag_reject_reason = "A closed curve needs at least three different points"
            return False
        if self._closed_curve_reuses_path(replacement):
            self._last_drag_reject_reason = "Drag rejected: it would make the closed curve cross itself"
            return False
        curve_index = int(point_ref[1])
        if self._curve_intersects_closed_curves(replacement, ignored_curve_index=curve_index):
            self._last_drag_reject_reason = "Drag rejected: closed curves cannot intersect"
            return False
        return True

    def _drag_target_is_allowed(self, point_ref: Tuple[str, int, int], vertex_id: int) -> bool:
        self._last_drag_reject_reason = ""
        if self._drag_target_is_inside_closed_curve(point_ref, vertex_id):
            self._last_drag_reject_reason = "Drag rejected: keep control points on the closed curve boundary"
            return False
        return self._drag_target_keeps_curve_shape(point_ref, vertex_id)

    def _move_dragged_point(self, pos) -> bool:
        if self._drag_point_ref is None:
            return False
        face_id = self._pick_display_face_id(pos, max_distance=52.0)
        vertex_id = self._nearest_face_vertex_id(face_id, pos) if face_id is not None else None
        if vertex_id is None:
            return False
        if not self._drag_target_is_allowed(self._drag_point_ref, int(vertex_id)):
            if self._last_drag_reject_reason:
                self.message = self._last_drag_reject_reason
                self.update()
            return False
        if not self._set_annotation_point_vertex(self._drag_point_ref, int(vertex_id)):
            return False
        self._curve_path_cache = {}
        self.message = "Moved point to the nearest shell vertex"
        self._emit_3d_state()
        self.update()
        return True

    def _set_annotation_point_vertex(self, point_ref: Tuple[str, int, int], vertex_id: int) -> bool:
        kind, curve_index, point_index = point_ref
        vertex_id = int(vertex_id)
        if kind == "active":
            if point_index < 0 or point_index >= len(self.active_curve_vertices):
                return False
            if int(self.active_curve_vertices[point_index]) == vertex_id:
                return False
            self.active_curve_vertices[point_index] = vertex_id
            return True

        if kind != "closed" or curve_index < 0 or curve_index >= len(self.closed_curves):
            return False
        curve = self.closed_curves[curve_index]
        vertices = [int(value) for value in curve.get("vertices", [])]
        if len(vertices) < 2:
            return False
        closes_to_first = vertices[0] == vertices[-1]
        editable_count = len(vertices) - 1 if closes_to_first else len(vertices)
        if point_index < 0 or point_index >= editable_count:
            return False
        if vertices[point_index] == vertex_id and (point_index != 0 or not closes_to_first or vertices[-1] == vertex_id):
            return False
        vertices[point_index] = vertex_id
        if closes_to_first and point_index == 0:
            vertices[-1] = vertex_id
        curve["vertices"] = vertices
        return True

    def _close_active_curve(self, vertex_index: int) -> None:
        vertices = self.active_curve_vertices[int(vertex_index) :] + [self.active_curve_vertices[int(vertex_index)]]
        if len(set(vertices[:-1])) < 3:
            self.message = "A closed curve needs at least three different points"
            return
        if self._closed_curve_reuses_path(vertices):
            self.message = "Closed curve rejected: it crosses itself"
            return
        if self._curve_intersects_closed_curves(vertices):
            self.message = "Closed curve rejected: it intersects an existing closed curve"
            return
        self.closed_curves.append(
            {
                "curve_id": f"cut_curve_{len(self.closed_curves) + 1}",
                "vertices": list(vertices),
            }
        )
        self._sync_surface_queue()
        self.active_surface_index = len(self.closed_curves) - 1
        self.surface_name = self.surface_names[self.active_surface_index]
        self.active_curve_vertices = []
        self.annotation_mode = "patch"
        self.message = f"Closed curve saved for '{self.surface_name}'. Click its seed patch."

    def close_active_curve_to_start(self) -> bool:
        if len(self.active_curve_vertices) < 3:
            self.annotation_mode = "curve"
            self.message = "A closed curve needs at least three points"
            self.update()
            return False
        curve_count = len(self.closed_curves)
        self._close_active_curve(0)
        self._emit_3d_state()
        self.update()
        return len(self.closed_curves) > curve_count

    def _add_selected_patch(self, face_id: int) -> None:
        if self.shell_mesh is None or len(self._faces) == 0:
            return
        if any(int(patch.get("face_id", -1)) == int(face_id) for patch in self.selected_patches):
            self.message = "That patch seed is already selected"
            return
        face = np.asarray([value for value in self._faces[int(face_id)] if int(value) >= 0], dtype=np.int64)
        if len(face) == 0:
            return
        self._sync_surface_queue()
        if not self.surface_names:
            self.message = "Close a curve before selecting a surface seed"
            return
        seed_point = np.asarray(self.shell_mesh.vertices, dtype=float)[face].mean(axis=0)
        label = self.surface_names[self.active_surface_index]
        self.selected_patches.append(
            {
                "patch_label": label,
                "surface_index": int(self.active_surface_index),
                "face_id": int(face_id),
                "seed_point": seed_point,
            }
        )
        self.message = f"Selected patch for surface '{label}'"

    def _handle_shell_click(self, pos) -> None:
        if self.shell_mesh is None:
            return
        if self.annotation_mode == "patch" and self.closed_curves:
            face_id = self._pick_display_face_id(pos, max_distance=44.0)
            if face_id is not None:
                self._add_selected_patch(face_id)
                self._emit_3d_state()
                self.update()
                return
        active_index = self._nearest_active_vertex_index(pos)
        if active_index is not None and len(self.active_curve_vertices) >= 3:
            self._close_active_curve(active_index)
            self._emit_3d_state()
            self.update()
            return
        face_id = self._pick_display_face_id(pos, max_distance=48.0)
        vertex_id = self._nearest_face_vertex_id(face_id, pos) if face_id is not None else None
        if vertex_id is None:
            self.message = "Click closer to the visible 3D shell"
            self.update()
            return
        if (
            (not self.active_curve_vertices and self._vertex_hits_closed_curve(int(vertex_id)))
            or self._active_segment_hits_closed_curve(int(vertex_id))
        ):
            self.message = "Point rejected: closed curves cannot intersect"
            self.update()
            return
        self.annotation_mode = "curve"
        self.active_curve_vertices.append(int(vertex_id))
        self.message = "Click more points, or click an active point to close the curve"
        self._emit_3d_state()
        self.update()

    def _draw_overlay(self) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        selected_faces = {int(patch["face_id"]) for patch in self.selected_patches}
        for face_id in selected_faces:
            self._draw_face_overlay(painter, face_id, QColor(245, 200, 76, 95), QColor("#f5c84c"), 1.5)
        if self.hover_face is not None:
            self._draw_face_overlay(painter, int(self.hover_face), QColor(88, 204, 156, 95), QColor("#ffffff"), 1.6)

        curve_count, point_count, patch_count = self.annotation_counts()
        mode_text = "Draw curve" if self.annotation_mode == "curve" else "Select surface"
        self._draw_message(
            painter,
            f"{mode_text}: {curve_count} curves · {point_count} points · {patch_count} seeds",
        )
        painter.end()

    def _draw_face_overlay(self, painter: QPainter, face_id: int, fill: QColor, outline: QColor, width: float) -> None:
        if len(self._faces) == 0 or face_id < 0 or face_id >= len(self._faces):
            return
        face_vertices = [int(value) for value in self._faces[int(face_id)] if int(value) >= 0]
        if len(face_vertices) < 3:
            return
        points = self._normalized_vertices[np.asarray(face_vertices, dtype=np.int64)]
        screen, _front_depth, _window_depth = self._project_points(points)
        polygon = QPolygonF([QPointF(float(point[0]), float(point[1])) for point in screen])
        painter.setPen(QPen(outline, width))
        painter.setBrush(fill)
        painter.drawPolygon(polygon)

    def _curve_vertex_path(self, vertex_ids: List[int], closed: bool) -> List[int]:
        if not vertex_ids:
            return []
        cleaned: List[int] = []
        for vertex_id in vertex_ids:
            value = int(vertex_id)
            if value < 0 or value >= len(self._normalized_vertices):
                continue
            if not cleaned or cleaned[-1] != value:
                cleaned.append(value)
        if len(cleaned) < 2:
            return cleaned
        if closed and cleaned[0] != cleaned[-1]:
            cleaned.append(cleaned[0])

        key = tuple(cleaned)
        cached = self._curve_path_cache.get(key)
        if cached is not None:
            return list(cached)

        path: List[int] = []
        for left, right in zip(cleaned[:-1], cleaned[1:]):
            segment = self._curve_segment_path(left, right)
            if path and segment and path[-1] == segment[0]:
                path.extend(segment[1:])
            else:
                path.extend(segment)
        self._curve_path_cache[key] = list(path)
        return path

    def _curve_segment_path(self, start: int, goal: int) -> List[int]:
        start = int(start)
        goal = int(goal)
        if start == goal:
            return [start]
        if self._edge_key(start, goal) in self._curve_edge_set:
            return [start, goal]
        if len(self._curve_graph_vertices) == 0 or not self._curve_edge_graph:
            return [start, goal]

        vertices = self._curve_graph_vertices

        def heuristic(vertex_id: int) -> float:
            return float(np.linalg.norm(vertices[int(vertex_id)] - vertices[goal]))

        heap: List[Tuple[float, float, int]] = [(heuristic(start), 0.0, start)]
        best_cost: Dict[int, float] = {start: 0.0}
        parent: Dict[int, int] = {}
        visited: set[int] = set()

        while heap and len(visited) < self.MAX_CURVE_PATH_VISITED:
            _estimate, cost, vertex_id = heapq.heappop(heap)
            if vertex_id in visited:
                continue
            if vertex_id == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            visited.add(vertex_id)
            for neighbor, weight in self._curve_edge_graph.get(vertex_id, []):
                if neighbor in visited:
                    continue
                next_cost = cost + float(weight)
                if next_cost >= best_cost.get(neighbor, math.inf):
                    continue
                best_cost[neighbor] = next_cost
                parent[neighbor] = vertex_id
                heapq.heappush(heap, (next_cost + heuristic(neighbor), next_cost, neighbor))
        return [start, goal]

    def _draw_message(self, painter: QPainter, text: str) -> None:
        metrics = painter.fontMetrics()
        rect_width = min(max(300, metrics.horizontalAdvance(text) + 24), max(120, self.width() - 20))
        rect = painter.boundingRect(10, 12, rect_width, 30, Qt.AlignVCenter, text)
        rect = rect.adjusted(0, 0, 20, 8)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(14, 22, 20, 220))
        painter.drawRoundedRect(rect, 8, 8)
        painter.setPen(QColor("#eef8f4"))
        painter.drawText(rect.adjusted(10, 0, -6, 0), Qt.AlignVCenter, text)

    @staticmethod
    def _wrap_rotation_angle(angle: float) -> float:
        return (angle + math.pi) % (math.pi * 2.0) - math.pi

    def mousePressEvent(self, event) -> None:
        if self.shell_mesh is None:
            return
        if event.button() not in (Qt.LeftButton, Qt.RightButton):
            return
        self.setFocus(Qt.MouseFocusReason)
        self._drag_pos = QPointF(event.pos())
        self._press_pos = QPointF(event.pos())
        self._drag_moved = False
        self._drag_point_ref = None
        self._last_drag_reject_reason = ""
        if event.button() == Qt.RightButton or event.modifiers() & Qt.ShiftModifier:
            self._drag_mode = "pan"
        else:
            point_ref = self._nearest_annotation_point_ref(event.pos())
            if point_ref is not None:
                self._drag_mode = "drag_point"
                self._drag_point_ref = point_ref
                self.setCursor(Qt.CrossCursor)
            else:
                self._drag_mode = "pick"
                self.setCursor(Qt.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self.shell_mesh is None:
            return
        if self._drag_pos is None:
            now = time.monotonic()
            if self._nearest_annotation_point_ref(event.pos()) is not None:
                self.hover_face = None
                self.setCursor(Qt.CrossCursor)
                self.update()
                return
            self.unsetCursor()
            if self.annotation_mode == "patch" and now - self._last_hover_at > 0.035:
                self.hover_face = self._pick_display_face_id(
                    event.pos(),
                    max_distance=38.0,
                    max_triangles=self.MAX_HOVER_FACES,
                )
                self._last_hover_at = now
                self.update()
            return
        dx = event.pos().x() - self._drag_pos.x()
        dy = event.pos().y() - self._drag_pos.y()
        self._drag_pos = QPointF(event.pos())
        if self._press_pos is not None:
            total_dx = event.pos().x() - self._press_pos.x()
            total_dy = event.pos().y() - self._press_pos.y()
            if total_dx * total_dx + total_dy * total_dy > self.CLICK_MOVE_TOLERANCE_SQ:
                self._drag_moved = True
                if self._drag_mode == "pick":
                    self._drag_mode = "rotate"
        if self._drag_mode == "drag_point":
            if self._drag_moved:
                self._move_dragged_point(event.pos())
            event.accept()
            return
        if self._drag_mode == "pan":
            self.pan_x += dx
            self.pan_y += dy
        elif self._drag_mode == "rotate":
            self.rotation_yaw = self._wrap_rotation_angle(self.rotation_yaw + dx * 0.01)
            self.rotation_pitch = max(-1.45, min(1.45, self.rotation_pitch + dy * 0.01))
            self._screen_cache_key = None
            self._screen_cache = None
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self.shell_mesh is None:
            return
        if event.button() not in (Qt.LeftButton, Qt.RightButton):
            return
        moved_distance_sq = 0.0
        if self._press_pos is not None:
            total_dx = event.pos().x() - self._press_pos.x()
            total_dy = event.pos().y() - self._press_pos.y()
            moved_distance_sq = total_dx * total_dx + total_dy * total_dy
        if self._drag_mode == "drag_point":
            if moved_distance_sq > self.CLICK_MOVE_TOLERANCE_SQ:
                self._move_dragged_point(event.pos())
            elif (
                self._drag_point_ref is not None
                and self._drag_point_ref[0] == "active"
                and len(self.active_curve_vertices) >= 3
            ):
                self._close_active_curve(self._drag_point_ref[2])
                self._emit_3d_state()
                self.update()
        elif event.button() == Qt.LeftButton and moved_distance_sq <= self.CLICK_MOVE_TOLERANCE_SQ:
            if (
                self.annotation_mode == "patch"
                and self._nearest_annotation_point_ref(
                    event.pos(),
                    max_distance=self.ANNOTATION_POINT_PATCH_GUARD_RADIUS_PX,
                )
                is not None
            ):
                self.message = "Click and drag the nearby point, or click farther inside the patch to select a seed"
                self.update()
            else:
                self._handle_shell_click(event.pos())
        self._drag_pos = None
        self._press_pos = None
        self._drag_point_ref = None
        self.unsetCursor()
        event.accept()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.12 if delta > 0 else 1 / 1.12
        self.preview_zoom = max(0.25, min(9.0, self.preview_zoom * factor))
        self._screen_cache_key = None
        self._screen_cache = None
        self.update()
        event.accept()
