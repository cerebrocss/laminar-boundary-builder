"""OpenGL 3D shell annotation view for the desktop app."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from OpenGL.GL import (
    GL_BACK,
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_CULL_FACE,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_FLOAT,
    GL_LEQUAL,
    GL_LINES,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_SRC_ALPHA,
    GL_TRIANGLES,
    glBlendFunc,
    glClear,
    glClearColor,
    glCullFace,
    glDepthFunc,
    glDisable,
    glDrawArrays,
    glEnable,
    glLineWidth,
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
    MAX_EDGE_SEGMENTS = 32000
    CLICK_MOVE_TOLERANCE_SQ = 81.0

    def __init__(self, parent=None):
        super().__init__(parent)
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)
        self.setFormat(fmt)
        self.setMinimumSize(420, 420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.shell_mesh = None
        self.surface_name = "surface"
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
        self._last_hover_at = 0.0

        self._gl_ready = False
        self._surface_program: Optional[QOpenGLShaderProgram] = None
        self._line_program: Optional[QOpenGLShaderProgram] = None
        self._vbo: Optional[QOpenGLBuffer] = None
        self._edge_vbo: Optional[QOpenGLBuffer] = None
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
        self._screen_cache_key = None
        self._screen_cache = None

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
        self.hover_face = None
        self.annotation_mode = "curve"
        self.rotation_yaw = -0.55
        self.rotation_pitch = 0.38
        self.preview_zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._screen_cache_key = None
        self._screen_cache = None
        if shell_mesh is None:
            self._clear_mesh_buffers()
            self.message = "3D shell is not available"
        else:
            self._prepare_mesh_arrays(shell_mesh)
            self.message = "3D: click the shaded shell to draw a cut curve"
        self._emit_3d_state()
        self.update()

    def set_surface_name(self, text: str) -> None:
        self.surface_name = str(text or "").strip() or "surface"

    def set_curve_mode(self) -> None:
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
        self.hover_face = None
        self.annotation_mode = "curve"
        self.message = "3D: click the shaded shell to draw a cut curve"
        self._emit_3d_state()
        self.update()

    def can_build_3d_surfaces(self) -> bool:
        return bool(self.shell_mesh is not None and self.closed_curves and self.selected_patches)

    def annotation_counts(self) -> tuple[int, int, int]:
        return (len(self.closed_curves), len(self.active_curve_vertices), len(self.selected_patches))

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
            patches.append(
                {
                    "patch_label": str(patch.get("patch_label") or "surface"),
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
        self._screen_cache_key = None
        self._screen_cache = None
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
            self._gl_ready = True
            self._upload_mesh()
        except Exception as exc:
            self._gl_ready = False
            self.message = f"OpenGL view failed to initialize: {exc}"

    def resizeGL(self, width: int, height: int) -> None:
        glViewport(0, 0, max(1, width), max(1, height))
        self._screen_cache_key = None
        self._screen_cache = None

    def paintGL(self) -> None:
        glClearColor(0.045, 0.075, 0.07, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self._gl_ready and self._vertex_count and self._surface_program is not None:
            self._draw_surface()
            self._draw_edges()
        self._draw_overlay()

    def _create_surface_program(self) -> QOpenGLShaderProgram:
        program = QOpenGLShaderProgram(self)
        vertex_shader = """
            attribute vec3 position;
            attribute vec3 normal;
            attribute vec3 pick_color;
            uniform mat4 mvp;
            varying vec3 v_normal;
            varying float v_soft_depth;
            void main() {
                gl_Position = mvp * vec4(position, 1.0);
                v_normal = normalize(normal);
                v_soft_depth = gl_Position.z;
            }
        """
        fragment_shader = """
            varying vec3 v_normal;
            varying float v_soft_depth;
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
            void main() {
                gl_Position = mvp * vec4(position, 1.0);
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
        glLineWidth(1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        program = self._line_program
        program.bind()
        program.setUniformValue("mvp", self._mvp_matrix())
        program.setUniformValue("line_color", QVector4D(0.03, 0.10, 0.09, 0.26))
        self._edge_vbo.bind()
        program.enableAttributeArray("position")
        program.setAttributeBuffer("position", GL_FLOAT, 0, 3, 3 * 4)
        glDrawArrays(GL_LINES, 0, int(self._edge_vertex_count))
        self._edge_vbo.release()
        program.disableAttributeArray("position")
        program.release()
        glDisable(GL_BLEND)

    def _mvp_matrix(self) -> QMatrix4x4:
        width = max(1, self.width())
        height = max(1, self.height())
        aspect = width / height
        projection = QMatrix4x4()
        projection.ortho(-aspect, aspect, -1.0, 1.0, -10.0, 10.0)
        model = QMatrix4x4()
        pan_world_x = float(self.pan_x) * aspect / (width * 0.5)
        pan_world_y = -float(self.pan_y) / (height * 0.5)
        model.translate(pan_world_x, pan_world_y, 0.0)
        model.scale(0.84 * float(self.preview_zoom))
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
            )
            return self._screen_cache
        rotated = self._rotated_points(self._normalized_vertices)
        scale = min(max(1, self.width()), max(1, self.height())) * 0.42 * self.preview_zoom
        screen = np.column_stack(
            (
                self.width() * 0.5 + self.pan_x + rotated[:, 0] * scale,
                self.height() * 0.5 + self.pan_y - rotated[:, 1] * scale,
            )
        ).astype(np.float32)
        self._screen_cache_key = key
        self._screen_cache = (screen, rotated[:, 2].astype(np.float32))
        return self._screen_cache

    def _rotated_points(self, points: np.ndarray) -> np.ndarray:
        coords = np.asarray(points, dtype=np.float32)
        yaw_cos = math.cos(self.rotation_yaw)
        yaw_sin = math.sin(self.rotation_yaw)
        pitch_cos = math.cos(self.rotation_pitch)
        pitch_sin = math.sin(self.rotation_pitch)
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        xz = x * yaw_cos + z * yaw_sin
        zz = -x * yaw_sin + z * yaw_cos
        yz = y * pitch_cos - zz * pitch_sin
        depth = y * pitch_sin + zz * pitch_cos
        return np.column_stack((xz, yz, depth)).astype(np.float32)

    def _nearest_active_vertex_index(self, pos, max_distance: float = 15.0) -> Optional[int]:
        if not self.active_curve_vertices:
            return None
        screen, _depth = self._screen_cache_data()
        ids = np.asarray(self.active_curve_vertices, dtype=np.int64)
        click = np.asarray([pos.x(), pos.y()], dtype=np.float32)
        distances = np.linalg.norm(screen[ids] - click.reshape(1, 2), axis=1)
        index = int(np.argmin(distances))
        return index if float(distances[index]) <= max_distance else None

    def _nearest_display_face_center_id(self, pos, max_distance: float = 36.0) -> Optional[int]:
        if len(self._display_centers) == 0:
            return None
        rotated = self._rotated_points(self._display_centers)
        scale = min(max(1, self.width()), max(1, self.height())) * 0.42 * self.preview_zoom
        centers = np.column_stack(
            (
                self.width() * 0.5 + self.pan_x + rotated[:, 0] * scale,
                self.height() * 0.5 + self.pan_y - rotated[:, 1] * scale,
            )
        ).astype(np.float32)
        if len(centers) > self.MAX_HOVER_FACES:
            step = int(math.ceil(len(centers) / self.MAX_HOVER_FACES))
            sample_ids = np.arange(0, len(centers), step, dtype=np.int64)
            centers_to_search = centers[sample_ids]
            face_ids_to_search = self._display_center_face_ids[sample_ids]
            depth_to_search = rotated[sample_ids, 2]
        else:
            centers_to_search = centers
            face_ids_to_search = self._display_center_face_ids
            depth_to_search = rotated[:, 2]
        click = np.asarray([pos.x(), pos.y()], dtype=np.float32)
        distances = np.linalg.norm(centers_to_search - click.reshape(1, 2), axis=1)
        near = np.flatnonzero(distances <= max_distance)
        if len(near) == 0:
            return None
        best = near[np.lexsort((distances[near], -depth_to_search[near]))][0]
        return int(face_ids_to_search[int(best)])

    def _pick_display_face_id(
        self,
        pos,
        max_distance: float = 36.0,
        max_triangles: Optional[int] = None,
    ) -> Optional[int]:
        if len(self._display_triangles) == 0:
            return None
        screen, depth = self._screen_cache_data()
        triangles = self._display_triangles
        face_ids = self._display_face_ids
        if max_triangles is not None and len(triangles) > max_triangles:
            step = int(math.ceil(len(triangles) / float(max_triangles)))
            sample_ids = np.arange(0, len(triangles), step, dtype=np.int64)
            triangles = triangles[sample_ids]
            face_ids = face_ids[sample_ids]

        click = np.asarray([pos.x(), pos.y()], dtype=np.float32)
        points = screen[triangles]
        margin = 2.0
        inside_bounds = (
            (points[:, :, 0].min(axis=1) <= click[0] + margin)
            & (points[:, :, 0].max(axis=1) >= click[0] - margin)
            & (points[:, :, 1].min(axis=1) <= click[1] + margin)
            & (points[:, :, 1].max(axis=1) >= click[1] - margin)
        )
        candidates = np.flatnonzero(inside_bounds)
        if len(candidates) == 0:
            return self._nearest_display_face_center_id(pos, max_distance=max_distance)

        p0 = points[candidates, 0, :]
        p1 = points[candidates, 1, :]
        p2 = points[candidates, 2, :]
        den = (
            (p1[:, 1] - p2[:, 1]) * (p0[:, 0] - p2[:, 0])
            + (p2[:, 0] - p1[:, 0]) * (p0[:, 1] - p2[:, 1])
        )
        valid = np.abs(den) > 1e-6
        if not np.any(valid):
            return self._nearest_display_face_center_id(pos, max_distance=max_distance)
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
            return self._nearest_display_face_center_id(pos, max_distance=max_distance)

        hit_ids = candidate_ids[inside]
        hit_triangles = triangles[hit_ids]
        hit_depth = depth[hit_triangles].mean(axis=1)
        best = int(hit_ids[int(np.argmax(hit_depth))])
        return int(face_ids[best])

    def _nearest_face_vertex_id(self, face_id: int, pos) -> Optional[int]:
        if len(self._faces) == 0 or face_id < 0 or face_id >= len(self._faces):
            return None
        screen, _depth = self._screen_cache_data()
        face_vertices = np.asarray([value for value in self._faces[int(face_id)] if int(value) >= 0], dtype=np.int64)
        if len(face_vertices) == 0:
            return None
        click = np.asarray([pos.x(), pos.y()], dtype=np.float32)
        distances = np.linalg.norm(screen[face_vertices] - click.reshape(1, 2), axis=1)
        return int(face_vertices[int(np.argmin(distances))])

    def _close_active_curve(self, vertex_index: int) -> None:
        vertices = self.active_curve_vertices[int(vertex_index) :] + [self.active_curve_vertices[int(vertex_index)]]
        if len(set(vertices[:-1])) < 3:
            self.message = "A closed curve needs at least three different points"
            return
        self.closed_curves.append(
            {
                "curve_id": f"cut_curve_{len(self.closed_curves) + 1}",
                "vertices": list(vertices),
            }
        )
        self.active_curve_vertices = []
        self.annotation_mode = "patch"
        self.message = "Closed curve saved. Hover and click the surface patch to keep."

    def _add_selected_patch(self, face_id: int) -> None:
        if self.shell_mesh is None or len(self._faces) == 0:
            return
        if any(int(patch.get("face_id", -1)) == int(face_id) for patch in self.selected_patches):
            self.message = "That patch seed is already selected"
            return
        face = np.asarray([value for value in self._faces[int(face_id)] if int(value) >= 0], dtype=np.int64)
        if len(face) == 0:
            return
        seed_point = np.asarray(self.shell_mesh.vertices, dtype=float)[face].mean(axis=0)
        label = self.surface_name or "surface"
        self.selected_patches.append(
            {
                "patch_label": label,
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
        self.annotation_mode = "curve"
        self.active_curve_vertices.append(int(vertex_id))
        self.message = "Click more points, or click an active point to close the curve"
        self._emit_3d_state()
        self.update()

    def _draw_overlay(self) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        screen, _depth = self._screen_cache_data()
        selected_faces = {int(patch["face_id"]) for patch in self.selected_patches}
        for face_id in selected_faces:
            self._draw_face_overlay(painter, face_id, QColor(245, 200, 76, 95), QColor("#f5c84c"), 1.5)
        if self.hover_face is not None:
            self._draw_face_overlay(painter, int(self.hover_face), QColor(88, 204, 156, 95), QColor("#ffffff"), 1.6)

        for index, curve in enumerate(self.closed_curves):
            color = QColor("#f06a5a") if index % 2 == 0 else QColor("#63a0ff")
            self._draw_curve(painter, [int(value) for value in curve.get("vertices", [])], screen, color, True)
        self._draw_curve(painter, [int(value) for value in self.active_curve_vertices], screen, QColor("#f0e95a"), False)
        curve_count, point_count, patch_count = self.annotation_counts()
        mode_text = "Draw curve" if self.annotation_mode == "curve" else "Select surface"
        self._draw_message(
            painter,
            f"{mode_text}: {curve_count} curve(s), {point_count} active point(s), {patch_count} selected patch(es)",
        )
        painter.end()

    def _draw_face_overlay(self, painter: QPainter, face_id: int, fill: QColor, outline: QColor, width: float) -> None:
        if len(self._faces) == 0 or face_id < 0 or face_id >= len(self._faces):
            return
        screen, _depth = self._screen_cache_data()
        face_vertices = [int(value) for value in self._faces[int(face_id)] if int(value) >= 0]
        if len(face_vertices) < 3:
            return
        polygon = QPolygonF([QPointF(float(screen[i, 0]), float(screen[i, 1])) for i in face_vertices])
        painter.setPen(QPen(outline, width))
        painter.setBrush(fill)
        painter.drawPolygon(polygon)

    def _draw_curve(self, painter: QPainter, vertex_ids: List[int], screen: np.ndarray, color: QColor, closed: bool) -> None:
        if not vertex_ids or len(screen) == 0:
            return
        points = [QPointF(float(screen[int(value), 0]), float(screen[int(value), 1])) for value in vertex_ids]
        line_width = 4.6 if closed else 4.0
        shadow = QPen(QColor(1, 8, 7, 210), line_width + 4.0)
        shadow.setStyle(Qt.SolidLine if closed else Qt.DashLine)
        painter.setPen(shadow)
        for left, right in zip(points[:-1], points[1:]):
            painter.drawLine(left, right)

        pen = QPen(color, line_width)
        pen.setStyle(Qt.SolidLine if closed else Qt.DashLine)
        painter.setPen(pen)
        for left, right in zip(points[:-1], points[1:]):
            painter.drawLine(left, right)
        self._draw_curve_points(painter, points, color, active=not closed)

    def _draw_curve_points(self, painter: QPainter, points: List[QPointF], color: QColor, active: bool) -> None:
        radius = 9.0 if active else 7.5
        for index, point in enumerate(points):
            painter.setPen(QPen(QColor(1, 8, 7, 230), 2.5))
            painter.setBrush(QColor(1, 8, 7, 180))
            painter.drawEllipse(point, radius + 4.5, radius + 4.5)

            painter.setPen(QPen(QColor("#ffffff"), 3.0))
            painter.setBrush(color)
            painter.drawEllipse(point, radius, radius)

            if active and index == len(points) - 1:
                painter.setPen(QPen(QColor("#ffffff"), 2.0))
                painter.setBrush(QColor("#ff6a3d"))
                painter.drawEllipse(point, radius * 0.48, radius * 0.48)

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
        if event.button() == Qt.RightButton or event.modifiers() & Qt.ShiftModifier:
            self._drag_mode = "pan"
        else:
            self._drag_mode = "pick"
        self.setCursor(Qt.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self.shell_mesh is None:
            return
        if self._drag_pos is None:
            now = time.monotonic()
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
        if event.button() == Qt.LeftButton and moved_distance_sq <= self.CLICK_MOVE_TOLERANCE_SQ:
            self._handle_shell_click(event.pos())
        self._drag_pos = None
        self._press_pos = None
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
