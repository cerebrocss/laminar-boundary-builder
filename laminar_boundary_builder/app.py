"""PyQt front end for the Laminar Boundary Builder."""

from __future__ import annotations

import contextlib
import csv
import gc
import glob
import io
import json
import math
import os
import shutil
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PyQt5.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, QUrl, QObject, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QDesktopServices, QFont, QImage, QKeySequence, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressDialog,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyleFactory,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

APP_STYLESHEET = """
QMainWindow, QWidget {
    background: #edf2f1;
    color: #263232;
    font-size: 13px;
}
QFrame#header {
    background: #fbfcfc;
    border: 1px solid #d7e0dd;
    border-radius: 8px;
}
QLabel#title {
    color: #172323;
    font-size: 22px;
    font-weight: 700;
}
QLabel#status {
    border-radius: 11px;
    padding: 4px 12px;
    font-weight: 650;
}
QLabel#status[state="ready"] {
    background: #dff1e9;
    color: #205247;
}
QLabel#status[state="running"] {
    background: #e7edf9;
    color: #2c4f84;
}
QLabel#status[state="failed"] {
    background: #fae1dc;
    color: #8b2e24;
}
QLabel#flowCaption {
    color: #60716e;
    font-size: 12px;
    font-weight: 650;
}
QLabel#flowStep {
    border: 1px solid #c4d4d0;
    border-radius: 13px;
    padding: 5px 12px;
    font-weight: 750;
}
QLabel#flowStep[state="active"] {
    background: #2f6e62;
    border-color: #2f6e62;
    color: #ffffff;
}
QLabel#flowStep[state="done"] {
    background: #dfeee9;
    border-color: #b8d0c9;
    color: #285f55;
}
QLabel#flowStep[state="pending"] {
    background: #ffffff;
    border-color: #c4d4d0;
    color: #516864;
}
QLabel#flowStep[state="test"] {
    background: #f5f7f7;
    border-color: #d2ddda;
    color: #667773;
}
QLabel#flowArrow {
    color: #7f9691;
    font-weight: 750;
}
QFrame#hint {
    background: #f4f8f7;
    border: 1px solid #d4e2de;
    border-radius: 8px;
}
QLabel#hintText {
    color: #35534e;
    font-weight: 650;
}
QLabel#progressText {
    background: #eef7f4;
    border: 1px solid #cddfda;
    border-radius: 7px;
    color: #2e504a;
    font-weight: 650;
    padding: 7px 9px;
}
QLabel#nextPointText {
    background: #f7fbfa;
    border: 1px solid #cddfda;
    border-radius: 7px;
    color: #24413c;
    font-weight: 750;
    padding: 7px 9px;
}
QWidget#parameterHelpRow {
    background: transparent;
}
QToolButton#parameterHelpButton {
    background: transparent;
    border: 1px solid #b7c9c5;
    border-radius: 9px;
    color: #5a6f6b;
    font-weight: 800;
    min-width: 18px;
    max-width: 18px;
    min-height: 18px;
    max-height: 18px;
    padding: 0;
}
QToolButton#parameterHelpButton:hover {
    background: #ffffff;
    border-color: #7f9f98;
    color: #2f6e62;
}
QFrame#parameterHelpPopup {
    background: transparent;
    border: 0;
}
QFrame#parameterHelpCard {
    background: #fbfcfc;
    border: 1px solid #aec8c2;
    border-radius: 8px;
}
QLabel#parameterHelpTitle {
    color: #1f3431;
    font-weight: 750;
}
QLabel#parameterHelpBody {
    color: #344847;
    line-height: 135%;
}
QPushButton#settingsPeek {
    border: 1px solid #b7c9c5;
    border-radius: 8px;
    background: #fbfcfc;
    color: #2f6e62;
    font-size: 18px;
    font-weight: 800;
    min-width: 24px;
    max-width: 24px;
    padding: 8px 0;
}
QPushButton#settingsPeek:hover {
    background: #edf7f4;
    border-color: #80a79f;
}
QWidget:disabled, QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {
    color: #7f8f8b;
}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {
    background: #f2f5f4;
    border-color: #d3ddda;
}
QGroupBox {
    background: #fbfcfc;
    border: 1px solid #d7e0dd;
    border-radius: 8px;
    margin-top: 12px;
    padding: 14px 12px 12px 12px;
    font-weight: 700;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 5px;
    color: #2e4744;
}
QLabel {
    color: #344847;
    background: transparent;
}
QCheckBox {
    background: transparent;
}
QLineEdit, QComboBox, QSpinBox, QTextEdit {
    background: #ffffff;
    border: 1px solid #b9cbc7;
    border-radius: 7px;
    padding: 5px 9px;
    min-height: 24px;
    selection-background-color: #5e8f83;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QTextEdit:focus {
    border: 1px solid #32786b;
    background: #ffffff;
}
QComboBox {
    padding-right: 34px;
}
QComboBox::drop-down {
    width: 0;
    border: 0;
}
QComboBox::drop-down:hover {
    background: transparent;
}
QComboBox::down-arrow {
    image: none;
    width: 0px;
    height: 0px;
    border: 0;
}
QComboBox QAbstractItemView {
    background: #ffffff;
    border: 1px solid #b9cbc7;
    border-radius: 7px;
    padding: 4px;
    selection-background-color: #dcece8;
    selection-color: #1e3330;
    outline: 0;
}
QSpinBox {
    padding-right: 9px;
}
QTextEdit {
    font-family: Menlo, Consolas, monospace;
}
QPushButton {
    border: 1px solid #b9c8c4;
    border-radius: 6px;
    padding: 6px 12px;
    min-height: 20px;
    background: #f6f8f7;
    color: #263635;
    font-weight: 600;
}
QPushButton:hover {
    background: #edf4f2;
    border-color: #8fb0a8;
}
QPushButton:pressed {
    background: #dfe9e6;
}
QPushButton[role="primary"] {
    background: #2f6e62;
    border-color: #2f6e62;
    color: white;
}
QPushButton[role="primary"]:hover {
    background: #285f55;
}
QPushButton[role="secondary"] {
    background: #eef4f2;
    color: #2f504a;
}
QPushButton[role="danger"] {
    background: #fbefec;
    border-color: #e2b8af;
    color: #84382e;
}
QPushButton[role="browse"] {
    padding-left: 0;
    padding-right: 0;
    background: #eef4f2;
    color: #1f3532;
    font-weight: 800;
}
QTabWidget::pane {
    border: 0;
}
QTabBar::tab {
    background: #dde8e5;
    color: #465b58;
    border: 1px solid #c9d6d3;
    border-bottom: 0;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
    padding: 8px 18px;
    margin-right: 4px;
    font-weight: 650;
}
QTabBar::tab:selected {
    background: #fbfcfc;
    color: #1f3532;
}
QScrollArea {
    border: 1px solid #d7e0dd;
    border-radius: 8px;
    background: #fbfcfc;
}
QScrollArea#sideScroll {
    border: 0;
    background: transparent;
}
QScrollBar:vertical {
    background: #eef5f3;
    width: 9px;
    margin: 0;
    border: 0;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #b7cbc5;
    border-radius: 5px;
    min-height: 36px;
    margin: 2px;
}
QScrollBar::handle:vertical:hover {
    background: #8fb0a8;
}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0;
    border: 0;
    background: transparent;
}
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
    background: transparent;
}
QScrollBar:horizontal {
    background: #eef5f3;
    height: 9px;
    margin: 0;
    border: 0;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background: #b7cbc5;
    border-radius: 5px;
    min-width: 36px;
    margin: 2px;
}
QScrollBar::handle:horizontal:hover {
    background: #8fb0a8;
}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
    width: 0;
    border: 0;
    background: transparent;
}
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {
    background: transparent;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #d4e0dd;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #2f6e62;
    border: 2px solid #edf7f4;
    width: 15px;
    margin: -6px 0;
    border-radius: 9px;
}
QCheckBox {
    spacing: 8px;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border: 1px solid #b7c9c5;
    border-radius: 4px;
    background: #ffffff;
}
QCheckBox::indicator:hover {
    border-color: #6f9b92;
    background: #f2f8f6;
}
QCheckBox::indicator:checked {
    background: #2f6e62;
    border-color: #2f6e62;
}
"""

_CORE = None
_NP = None
TEMP_MASK_PREFIX = "laminar_boundary_mask_"
TEMP_MASK_MARKER = ".laminar_boundary_builder_temp.json"
UNMARKED_TEMP_MAX_AGE_SECONDS = 24 * 60 * 60
LOG_CACHE_DIR = Path.home() / "Library" / "Caches" / "Laminar Boundary Builder" / "logs"
LOG_FILE_PREFIX = "run_"
LOG_MAX_FILES = 20
ANNOTATION_CONTOUR_CACHE_LIMIT = 96

ANNOTATE_HELP = {
    "region": (
        "Brain region",
        "Purpose: choose the Allen atlas region to extract as a mask. You can type an acronym such as ENT or a numeric ID such as 909.\n"
        "Effect: this controls which voxels become the annotation mask.\n"
        "Recommended: use the exact region acronym when you know it. Keep child regions on for parent regions.",
    ),
    "hemisphere": (
        "Hemisphere",
        "Purpose: restrict the extracted atlas mask to all, left, or right hemisphere.\n"
        "Effect: left or right reduces the mask to one side only.\n"
        "Recommended: use all unless you only want one hemisphere.",
    ),
    "include_children": (
        "Include child regions",
        "Purpose: include all subregions under the selected Allen region.\n"
        "Effect: parent regions such as ENT become a complete mask instead of only one direct atlas ID.\n"
        "Recommended: keep this on for most region-level extraction.",
    ),
    "custom_atlas_enabled": (
        "Use a custom Allen atlas file",
        "Purpose: switch from the built-in Allen annotation_10.nrrd to your own atlas file.\n"
        "Effect: extraction uses the selected file instead of the bundled atlas.\n"
        "Recommended: leave this off unless you need a different atlas resolution or orientation.",
    ),
    "custom_atlas": (
        "Custom atlas",
        "Purpose: choose another Allen annotation volume or cached atlas file.\n"
        "Effect: region masks are extracted from this file, so its orientation and IDs must match your workflow.\n"
        "Recommended: leave empty and use the built-in atlas for normal use.",
    ),
    "mask": (
        "Mask",
        "Purpose: load an existing target mask instead of extracting one from the atlas.\n"
        "Effect: if Brain region is empty, this file is used directly for annotation.\n"
        "Recommended: leave empty when extracting from Allen; choose a binary region mask when you already have one.",
    ),
    "template": (
        "Template image",
        "Purpose: show an image volume behind the mask while picking landmarks.\n"
        "Effect: it only changes the visual background, not the saved landmarks or mask.\n"
        "Recommended: optional. Use it when anatomy is easier to see on the template than on the binary mask.",
    ),
    "output": (
        "Output folder",
        "Purpose: choose where the manual landmark CSV will be saved.\n"
        "Effect: Save CSV for Build writes manual_landmarks_interactive.csv here and prepares the Build step.\n"
        "Recommended: choose a project-specific folder so the CSV is easy to find later.",
    ),
    "previous_csv": (
        "Previous manual CSV",
        "Purpose: reload a saved manual_landmarks_interactive.csv into the annotation workspace.\n"
        "Effect: accepted slices, points, and arc path choices are restored so you can edit only the bad slices.\n"
        "Recommended: load the same mask first, then load the previous CSV and revise flagged slices.",
    ),
    "slice_axis": (
        "Slice axis",
        "Purpose: choose which volume axis is treated as the stack of slices.\n"
        "Effect: this changes which 2D plane you annotate and how landmarks map back into 3D.\n"
        "Recommended: use coronal, sagittal, or horizontal to match your sections. Use 0, 1, or 2 only when you know the raw array axis.",
    ),
    "min_area": (
        "Min contour area",
        "Purpose: ignore tiny contours created by noise or small fragments.\n"
        "Effect: larger values remove more small pieces; smaller values keep more pieces.\n"
        "Recommended: start with 20. Increase it if many tiny contours appear.",
    ),
    "keep_all": (
        "Keep all contours per slice",
        "Purpose: keep every contour found on a slice instead of only the largest one.\n"
        "Effect: this helps when a region is split into multiple visible pieces, but it can make contour selection busier.\n"
        "Recommended: leave off for simple masks. Turn on if the target region has separate pieces on the same slice.",
    ),
    "slice": (
        "Slice",
        "Purpose: choose the current slice to inspect and annotate.\n"
        "Effect: moving it changes the displayed slice but does not save points by itself.\n"
        "Recommended: annotate several representative slices across the region range, not only one slice.",
    ),
    "contour": (
        "Contour",
        "Purpose: choose which detected contour on the current slice receives your four landmarks.\n"
        "Effect: landmarks are saved against this contour ID.\n"
        "Recommended: use contour 0 when only one contour is present. Switch if the highlighted contour is not your target boundary.",
    ),
    "outer_path": (
        "Outer arc path",
        "Purpose: choose how the outer boundary arc is traced between outer_start and outer_end.\n"
        "Effect: forward or backward forces one direction around the contour; auto lets the app choose.\n"
        "Recommended: keep auto unless the previewed boundary follows the wrong side.",
    ),
    "inner_path": (
        "Inner arc path",
        "Purpose: choose how the inner boundary arc is traced between inner_start and inner_end.\n"
        "Effect: forward or backward forces one direction around the contour; auto lets the app choose.\n"
        "Recommended: keep auto unless the previewed boundary follows the wrong side.",
    ),
}

BUILD_HELP = {
    "mask": (
        "Mask",
        "Purpose: the same target mask used during annotation.\n"
        "Effect: surfaces, labels, and depth volumes are built inside this mask.\n"
        "Recommended: use the mask that was prepared by the Annotate step.",
    ),
    "manual_csv": (
        "Manual CSV",
        "Purpose: the landmark CSV saved from the Annotate step.\n"
        "Effect: these landmark rows define the outer and inner boundary curves used to build surfaces.\n"
        "Recommended: use manual_landmarks_interactive.csv created by Save CSV for Build.",
    ),
    "boundaries_json": (
        "Boundary JSON",
        "Purpose: saved propagated outer, inner, and lateral curves from Extract Surfaces.\n"
        "Effect: Compute Laminar Depth Volume can reuse this file later without repeating annotation.\n"
        "Recommended: keep the default boundary_annotations.json from the surface build folder.",
    ),
    "output": (
        "Output folder",
        "Purpose: choose where Build writes surfaces, label volumes, depth volumes, QC images, and tables.\n"
        "Effect: existing files with the same names may be overwritten.\n"
        "Recommended: use a new build folder inside the same project output directory.",
    ),
    "template": (
        "Template image",
        "Purpose: optional image volume for QC previews.\n"
        "Effect: it helps inspect outputs visually but does not change the computed surfaces.\n"
        "Recommended: optional. Use the same template image as Annotate when available.",
    ),
    "cell_csv": (
        "Cell CSV",
        "Purpose: optional soma coordinate table for assigning laminar depth to cells.\n"
        "Effect: if provided, Build writes cell-level depth measurements.\n"
        "Recommended: leave empty unless you have soma coordinates in the same volume space.",
    ),
    "swc_glob": (
        "SWC glob",
        "Purpose: optional file pattern for neuron morphology files.\n"
        "Effect: matching SWC files can be sampled against the generated laminar depth volume.\n"
        "Recommended: leave empty unless you need morphology-depth measurements.",
    ),
    "slice_axis": (
        "Slice axis",
        "Purpose: match the slice axis used when the manual CSV was created.\n"
        "Effect: a mismatch makes landmark coordinates and generated surfaces wrong.\n"
        "Recommended: keep the value copied from Annotate.",
    ),
    "min_area": (
        "Min contour area",
        "Purpose: filter small mask fragments during surface building.\n"
        "Effect: higher values ignore more small contours; lower values keep more fragments.\n"
        "Recommended: start with 20 and match the Annotate setting.",
    ),
    "resample_points": (
        "Resample points",
        "Purpose: set how many points each manual boundary curve is resampled to.\n"
        "Effect: higher values make smoother surfaces but take more work and can overfit noisy landmarks.\n"
        "Recommended: 80 for normal builds; 48 is enough for quick tests.",
    ),
    "depth_method": (
        "Depth method",
        "Purpose: choose how laminar depth is computed between outer and inner boundaries.\n"
        "Effect: surfaces only skips slow volume/depth outputs; laplace is smoother but slower; distance is faster.\n"
        "Recommended: surfaces only for the first automatic build; auto only when depth volume is needed.",
    ),
    "volume_format": (
        "Volume format",
        "Purpose: choose the file format for generated volume outputs.\n"
        "Effect: nrrd is convenient for this app; npy is Python-friendly; nii.gz is useful for many neuroimaging tools.\n"
        "Recommended: nrrd unless another tool requires a different format.",
    ),
    "max_laplace_voxels": (
        "Max Laplace voxels",
        "Purpose: limit how large a mask can be before auto switches away from Laplace depth.\n"
        "Effect: higher values try Laplace on larger masks but can be much slower and use more memory.\n"
        "Recommended: keep 250000 unless you know your machine can handle more.",
    ),
    "boundary_dilation": (
        "Boundary dilation",
        "Purpose: thicken generated boundary labels before depth calculation.\n"
        "Effect: larger values make boundary constraints stronger, but too large can blur thin regions.\n"
        "Recommended: 1 for most masks.",
    ),
    "qc_every": (
        "QC interval",
        "Purpose: control how often QC slice images are written.\n"
        "Effect: smaller values create more QC images; larger values create fewer files.\n"
        "Recommended: 10 for normal builds, 4 to 5 when debugging.",
    ),
    "keep_all": (
        "Keep all contours per slice",
        "Purpose: keep multiple contours when rebuilding surfaces.\n"
        "Effect: helps split masks but may include unwanted fragments if the mask is noisy.\n"
        "Recommended: match the Annotate setting.",
    ),
}

DEMO_HELP = {
    "output": (
        "Output folder",
        "Purpose: choose where the synthetic demo writes its test outputs.\n"
        "Effect: this does not use your real data; it is only for checking that the pipeline runs.\n"
        "Recommended: use a temporary demo folder.",
    ),
    "resample_points": (
        "Resample points",
        "Purpose: set the curve sampling density for the synthetic demo.\n"
        "Effect: higher values are smoother but slower.\n"
        "Recommended: keep 48 for quick demo testing.",
    ),
    "depth_method": (
        "Depth method",
        "Purpose: choose the demo depth algorithm.\n"
        "Effect: auto mirrors the normal pipeline behavior; laplace and distance force one method.\n"
        "Recommended: auto.",
    ),
}


def _core():
    global _CORE
    if _CORE is None:
        from . import core as core_module

        _CORE = core_module
    return _CORE


def _numpy():
    global _NP
    if _NP is None:
        import numpy as numpy_module

        _NP = numpy_module
    return _NP


def _process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _write_temp_mask_marker(directory: str | Path) -> None:
    marker_path = Path(directory) / TEMP_MASK_MARKER
    payload = {
        "app": "Laminar Boundary Builder",
        "pid": os.getpid(),
        "created_at": time.time(),
    }
    try:
        marker_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


def cleanup_orphan_temporary_masks() -> int:
    """Remove stale mask temp dirs left by crashed app processes."""

    temp_root = Path(tempfile.gettempdir())
    removed = 0
    now = time.time()
    for temp_dir in temp_root.glob(f"{TEMP_MASK_PREFIX}*"):
        if not temp_dir.is_dir():
            continue

        marker_path = temp_dir / TEMP_MASK_MARKER
        should_remove = False
        if marker_path.exists():
            try:
                payload = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            pid = int(payload.get("pid") or 0)
            should_remove = not _process_is_running(pid)
        else:
            try:
                age_seconds = now - temp_dir.stat().st_mtime
            except OSError:
                age_seconds = 0
            should_remove = age_seconds >= UNMARKED_TEMP_MAX_AGE_SECONDS

        if should_remove:
            try:
                shutil.rmtree(temp_dir)
                removed += 1
            except OSError:
                pass
    return removed


def cleanup_old_log_files() -> None:
    try:
        files = sorted(
            LOG_CACHE_DIR.glob(f"{LOG_FILE_PREFIX}*.log"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return

    for log_path in files[LOG_MAX_FILES:]:
        try:
            log_path.unlink()
        except OSError:
            pass


@dataclass
class TaskResult:
    title: str
    message: str
    output_dir: Optional[Path] = None
    payload: object = None


class StreamBuffer(io.StringIO):
    def __init__(self, callback: Callable[[str], None]):
        super().__init__()
        self.callback = callback

    def write(self, text: str) -> int:
        if text:
            self.callback(text)
        return len(text)


class Worker(QObject):
    log = pyqtSignal(str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn: Callable[[], TaskResult]):
        super().__init__()
        self.fn = fn

    def run(self) -> None:
        stdout = StreamBuffer(self.log.emit)
        stderr = StreamBuffer(self.log.emit)
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = self.fn()
            self.finished.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class PathRow(QWidget):
    def __init__(
        self,
        placeholder: str = "",
        select_file: bool = True,
        save_file: bool = False,
        file_filter: str = "All files (*)",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.select_file = select_file
        self.save_file = save_file
        self.file_filter = file_filter

        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.setMinimumWidth(190)
        self.edit.setMinimumHeight(28)
        self.button = QPushButton("...")
        self.button.setProperty("role", "browse")
        self.button.setFixedWidth(44)
        self.button.setMinimumHeight(28)
        self.button.clicked.connect(self.choose)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)

    def text(self) -> str:
        return self.edit.text().strip()

    def set_text(self, value: str | Path) -> None:
        self.edit.setText(str(value))

    def choose(self) -> None:
        current = self.text()
        start_dir = str(Path(current).expanduser().parent) if current else str(Path.home())
        if self.save_file:
            path, _ = QFileDialog.getSaveFileName(self, "Choose file", current or start_dir, self.file_filter)
        elif self.select_file:
            path, _ = QFileDialog.getOpenFileName(self, "Choose file", current or start_dir, self.file_filter)
        else:
            path = QFileDialog.getExistingDirectory(self, "Choose folder", current or start_dir)
        if path:
            self.set_text(path)


class CleanComboBox(QComboBox):
    """A normal combo box with a cleaner painted closed state."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumHeight(28)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        focused = self.hasFocus()
        hovered = self.underMouse()
        border = QColor("#32786b" if focused else "#abc0bc")
        button_bg = QColor("#e8f2ef" if hovered else "#f1f7f5")

        painter.setBrush(QColor("#ffffff"))
        painter.setPen(QPen(border, 1.0))
        painter.drawRoundedRect(rect, 7, 7)

        button_rect = QRectF(rect.right() - 30, rect.top(), 30, rect.height())
        painter.setPen(Qt.NoPen)
        painter.setBrush(button_bg)
        painter.drawRect(button_rect.adjusted(0, 1, 0, -1))
        painter.setPen(QPen(QColor("#d2e0dc"), 1.0))
        painter.drawLine(button_rect.topLeft(), button_rect.bottomLeft())

        text_rect = rect.adjusted(9, 0, -39, 0)
        text = painter.fontMetrics().elidedText(self.currentText(), Qt.ElideRight, int(text_rect.width()))
        painter.setPen(QColor("#233534" if self.currentText() else "#8a9895"))
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, text)

        center = button_rect.center()
        arrow = QPolygonF(
            [
                QPointF(center.x() - 4.5, center.y() - 2.0),
                QPointF(center.x() + 4.5, center.y() - 2.0),
                QPointF(center.x(), center.y() + 3.2),
            ]
        )
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#486762"))
        painter.drawPolygon(arrow)


class SliceCanvas(QWidget):
    landmark_changed = pyqtSignal(str)

    LANDMARK_ORDER = ("outer_start", "outer_end", "inner_start", "inner_end")

    COLORS = {
        "outer_start": QColor("#1f77b4"),
        "outer_end": QColor("#1f77b4"),
        "inner_start": QColor("#d62728"),
        "inner_end": QColor("#d62728"),
    }

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(560, 420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.image = None
        self.contours = []
        self.selected_contour_index = 0
        self.landmarks: Dict[str, int] = {}
        self.mode = "outer_start"
        self.outer_path_choice = "auto"
        self.inner_path_choice = "auto"
        self.preview_resample_points = 80
        self.picking_enabled = False
        self.progress_text = ""
        self.slice_axis = 0
        self._scale = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self.zoom_factor = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.show_overlays = True
        self._drag_landmark: Optional[str] = None
        self._drag_pan = False
        self._drag_moved = False
        self._drag_start_pos: Optional[QPointF] = None
        self._pan_start_x = 0.0
        self._pan_start_y = 0.0

    def set_scene(
        self,
        image,
        contours,
        landmarks: Optional[Dict[str, int]] = None,
        selected_contour_index: int = 0,
        slice_axis: int = 0,
    ) -> None:
        np = _numpy()
        self.image = np.asarray(image)
        self.contours = contours
        self.landmarks = dict(landmarks or {})
        self.selected_contour_index = min(max(0, selected_contour_index), max(0, len(contours) - 1))
        self.slice_axis = slice_axis
        self.update()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.update()

    def set_picking_enabled(self, enabled: bool) -> None:
        self.picking_enabled = enabled
        self.update()

    def set_path_choices(self, outer_path: str, inner_path: str) -> None:
        self.outer_path_choice = outer_path
        self.inner_path_choice = inner_path
        self.update()

    def set_progress_text(self, text: str) -> None:
        self.progress_text = text
        self.update()

    def selected_contour(self):
        if not self.contours:
            return None
        return self.contours[self.selected_contour_index]

    def _image_to_qimage(self) -> Optional[QImage]:
        if self.image is None:
            return None
        np = _numpy()
        image = np.asarray(self.image, dtype=float)
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            scaled = np.zeros(image.shape, dtype=np.uint8)
        else:
            lo, hi = np.percentile(finite, [1, 99])
            if hi <= lo:
                hi = lo + 1.0
            scaled = np.clip((image - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        qimage = QImage(
            scaled.data,
            scaled.shape[1],
            scaled.shape[0],
            scaled.strides[0],
            QImage.Format_Grayscale8,
        )
        return qimage.copy()

    def _update_transform(self) -> None:
        if self.image is None:
            self._scale = 1.0
            self._offset_x = 0.0
            self._offset_y = 0.0
            return
        height, width = self.image.shape[:2]
        base_scale = min(self.width() / max(1, width), self.height() / max(1, height))
        self._scale = base_scale * self.zoom_factor
        draw_width = width * self._scale
        draw_height = height * self._scale
        self._offset_x = (self.width() - draw_width) * 0.5 + self._pan_x
        self._offset_y = (self.height() - draw_height) * 0.5 + self._pan_y

    def set_zoom(self, value: float, anchor: Optional[QPointF] = None) -> None:
        if self.image is None:
            return
        self._update_transform()
        anchor = anchor or QPointF(self.width() * 0.5, self.height() * 0.5)
        anchor_plane = self.screen_to_plane(anchor)
        self.zoom_factor = max(1.0, min(16.0, float(value)))
        self._update_transform()
        new_anchor = self.plane_to_screen(anchor_plane)
        self._pan_x += anchor.x() - new_anchor.x()
        self._pan_y += anchor.y() - new_anchor.y()
        self._update_transform()
        self.update()

    def zoom_in(self) -> None:
        self.set_zoom(self.zoom_factor * 1.25)

    def zoom_out(self) -> None:
        self.set_zoom(self.zoom_factor / 1.25)

    def reset_zoom(self) -> None:
        self.zoom_factor = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()

    def toggle_overlays(self) -> None:
        self.show_overlays = not self.show_overlays
        self.update()

    def plane_to_screen(self, point) -> QPointF:
        return QPointF(
            self._offset_x + float(point[0]) * self._scale,
            self._offset_y + float(point[1]) * self._scale,
        )

    def screen_to_plane(self, pos):
        np = _numpy()
        return np.array(
            [
                (pos.x() - self._offset_x) / self._scale,
                (pos.y() - self._offset_y) / self._scale,
            ],
            dtype=float,
        )

    def _selected_plane_points(self):
        contour = self.selected_contour()
        if contour is None:
            return None
        core = _core()
        points = core._normalize_contour(contour.points)
        if len(points) == 0:
            return None
        return core._volume_to_plane_points(points, self.slice_axis)

    def _arc_indices_for_landmarks(
        self,
        start_name: str,
        end_name: str,
        path_choice: str,
    ) -> tuple[Optional[object], set[int]]:
        contour = self.selected_contour()
        if contour is None or start_name not in self.landmarks or end_name not in self.landmarks:
            return None, set()
        core = _core()
        points = core._normalize_contour(contour.points)
        if len(points) < 2:
            return None, set()
        try:
            _, indices, _, _ = core._choose_arc(
                points,
                self.landmarks[start_name],
                self.landmarks[end_name],
                choice=path_choice,
            )
        except Exception:
            return None, set()
        endpoints = {
            int(self.landmarks[start_name]) % len(points),
            int(self.landmarks[end_name]) % len(points),
        }
        return indices, endpoints

    def _excluded_arc_indices_for_landmark(self, name: str) -> set[int]:
        if name.startswith("inner"):
            indices, endpoints = self._arc_indices_for_landmarks(
                "outer_start",
                "outer_end",
                self.outer_path_choice,
            )
        elif name.startswith("outer"):
            indices, endpoints = self._arc_indices_for_landmarks(
                "inner_start",
                "inner_end",
                self.inner_path_choice,
            )
        else:
            return set()
        if indices is None:
            return set()
        return {int(index) for index in indices} - endpoints

    def _nearest_contour_index_for_pos(self, pos, landmark_name: Optional[str] = None) -> Optional[int]:
        np = _numpy()
        plane = self._selected_plane_points()
        if plane is None or len(plane) == 0:
            return None
        click = self.screen_to_plane(pos)
        distances = np.linalg.norm(plane - click[None, :], axis=1)
        if landmark_name is not None:
            excluded = [
                index
                for index in self._excluded_arc_indices_for_landmark(landmark_name)
                if 0 <= index < len(distances)
            ]
            if excluded and len(excluded) < len(distances):
                distances[excluded] = np.inf
            if not np.isfinite(distances).any():
                return None
        return int(np.argmin(distances))

    def _nearest_landmark_for_pos(self, pos, max_distance: float = 14.0) -> Optional[str]:
        np = _numpy()
        plane = self._selected_plane_points()
        if plane is None:
            return None
        nearest_name = None
        nearest_distance = float("inf")
        for name, index in self.landmarks.items():
            if index < 0 or index >= len(plane):
                continue
            screen = self.plane_to_screen(plane[index])
            distance = float(np.linalg.norm([screen.x() - pos.x(), screen.y() - pos.y()]))
            if distance < nearest_distance:
                nearest_name = name
                nearest_distance = distance
        if nearest_name is not None and nearest_distance <= max_distance:
            return nearest_name
        return None

    def _set_landmark_from_pos(self, name: str, pos) -> bool:
        index = self._nearest_contour_index_for_pos(pos, name)
        if index is None:
            return False
        self.landmarks[name] = index
        self.update()
        return True

    def preview_boundary(self):
        contour = self.selected_contour()
        if contour is None:
            return None
        if not all(name in self.landmarks for name in self.LANDMARK_ORDER):
            return None
        row = {
            "slice_index": str(contour.slice_index),
            "contour_id": str(contour.contour_id),
            "outer_start_index": str(self.landmarks["outer_start"]),
            "outer_end_index": str(self.landmarks["outer_end"]),
            "outer_path": self.outer_path_choice,
            "inner_start_index": str(self.landmarks["inner_start"]),
            "inner_end_index": str(self.landmarks["inner_end"]),
            "inner_path": self.inner_path_choice,
        }
        try:
            return _core().make_boundary_from_landmark_row(
                contour,
                row,
                resample_points=self.preview_resample_points,
            )
        except Exception:
            return None

    def _preview_arc(self, points, start_name: str, end_name: str, path_choice: str):
        if start_name not in self.landmarks or end_name not in self.landmarks:
            return None
        try:
            arc, _, _, _ = _core()._choose_arc(
                points,
                self.landmarks[start_name],
                self.landmarks[end_name],
                choice=path_choice,
            )
            return arc
        except Exception:
            return None

    def _draw_plane_polyline(self, painter: QPainter, plane_points, color: QColor, width: float) -> None:
        if plane_points is None or len(plane_points) < 2:
            return
        painter.setPen(QPen(color, width))
        for p0, p1 in zip(plane_points[:-1], plane_points[1:]):
            painter.drawLine(self.plane_to_screen(p0), self.plane_to_screen(p1))

    def _draw_boundary_preview(self, painter: QPainter, points) -> Optional[object]:
        core = _core()
        outer_arc = self._preview_arc(
            points,
            "outer_start",
            "outer_end",
            self.outer_path_choice,
        )
        inner_arc = self._preview_arc(
            points,
            "inner_start",
            "inner_end",
            self.inner_path_choice,
        )
        boundary = self.preview_boundary()

        if boundary is not None:
            outer_plane = core._volume_to_plane_points(boundary.outer_arc, self.slice_axis)
            inner_plane = core._volume_to_plane_points(boundary.inner_arc, self.slice_axis)
            if len(outer_plane) >= 2 and len(inner_plane) >= 2:
                band = QPolygonF()
                for point in outer_plane:
                    band.append(self.plane_to_screen(point))
                for point in reversed(inner_plane):
                    band.append(self.plane_to_screen(point))
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(64, 170, 205, 42))
                painter.drawPolygon(band)
            for lateral_arc in boundary.lateral_arcs:
                lateral_plane = core._volume_to_plane_points(lateral_arc, self.slice_axis)
                self._draw_plane_polyline(painter, lateral_plane, QColor("#dcc05a"), 3.0)

        if boundary is not None:
            outer_arc = boundary.outer_arc
            inner_arc = boundary.inner_arc

        if outer_arc is not None:
            outer_plane = core._volume_to_plane_points(outer_arc, self.slice_axis)
            self._draw_plane_polyline(painter, outer_plane, QColor("#39a8ff"), 5.0)
        if inner_arc is not None:
            inner_plane = core._volume_to_plane_points(inner_arc, self.slice_axis)
            self._draw_plane_polyline(painter, inner_plane, QColor("#f15b5b"), 5.0)
        return boundary

    def _draw_boundary_preview_label(self, painter: QPainter, boundary) -> None:
        if boundary is None:
            return
        text = (
            f"Preview: outer {len(boundary.outer_arc)} pts, "
            f"inner {len(boundary.inner_arc)} pts, "
            f"gap {boundary.min_outer_inner_distance:.1f}"
        )
        metrics = painter.fontMetrics()
        rect_width = min(max(340, metrics.horizontalAdvance(text) + 24), max(120, self.width() - 20))
        label_rect = QRectF(10, self.height() - 42, rect_width, 30)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(18, 24, 23, 210))
        painter.drawRoundedRect(label_rect, 8, 8)
        painter.setPen(QColor("#e8eeee"))
        painter.drawText(label_rect.adjusted(10, 0, 0, 0), Qt.AlignVCenter, text)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#121817"))
        if self.image is None:
            painter.setPen(QColor("#d4dddd"))
            painter.drawText(self.rect(), Qt.AlignCenter, "No mask loaded")
            return

        self._update_transform()
        qimage = self._image_to_qimage()
        if qimage is not None:
            height, width = self.image.shape[:2]
            target = QRectF(
                self._offset_x,
                self._offset_y,
                width * self._scale,
                height * self._scale,
            )
            painter.drawImage(target, qimage)

        if self.contours:
            core = _core()
        for idx, contour in enumerate(self.contours):
            plane = core._volume_to_plane_points(core._normalize_contour(contour.points), self.slice_axis)
            if len(plane) < 2:
                continue
            color = QColor("#f5c84c") if idx == self.selected_contour_index else QColor("#90a09c")
            width = 2.6 if idx == self.selected_contour_index else 1.1
            painter.setPen(QPen(color, width))
            for p0, p1 in zip(plane[:-1], plane[1:]):
                painter.drawLine(self.plane_to_screen(p0), self.plane_to_screen(p1))
            painter.drawLine(self.plane_to_screen(plane[-1]), self.plane_to_screen(plane[0]))

        contour = self.selected_contour()
        if contour is not None:
            core = _core()
            points = core._normalize_contour(contour.points)
            plane = core._volume_to_plane_points(points, self.slice_axis)
            boundary_preview = self._draw_boundary_preview(painter, points)
            for name, index in self.landmarks.items():
                if index < 0 or index >= len(plane):
                    continue
                color = self.COLORS.get(name, QColor("#ffffff"))
                pos = self.plane_to_screen(plane[index])
                painter.setBrush(color)
                painter.setPen(QPen(QColor("#111111"), 1.2))
                painter.drawEllipse(pos, 6.0, 6.0)
                painter.setPen(color)
                label = name.replace("_", " ")
                painter.drawText(pos + QPointF(8, -8), label)
            if self.show_overlays:
                self._draw_boundary_preview_label(painter, boundary_preview)

        if not self.show_overlays:
            hint = "H show info"
            metrics = painter.fontMetrics()
            hint_rect = QRectF(10, 10, metrics.horizontalAdvance(hint) + 18, 26)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(18, 24, 23, 120))
            painter.drawRoundedRect(hint_rect, 7, 7)
            painter.setPen(QColor("#cbd9d6"))
            painter.drawText(hint_rect.adjusted(8, 0, 0, 0), Qt.AlignVCenter, hint)
            return

        label = f"Next point: {self.mode.replace('_', ' ')}"
        if not self.picking_enabled:
            label = "Load mask to start point picking"
        elif all(name in self.landmarks for name in self.LANDMARK_ORDER):
            label = "All four points set. Enter = accept + next"
        label_rect = QRectF(10, 10, max(260, len(label) * 8), 30)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(18, 24, 23, 205))
        painter.drawRoundedRect(label_rect, 8, 8)
        painter.setPen(QColor("#e8eeee"))
        painter.drawText(label_rect.adjusted(10, 0, 0, 0), Qt.AlignVCenter, label)

        if self.picking_enabled:
            keys = "Drag background pan   S no inner   H info   N suggest   O/I flip   Wheel/+/- zoom   0 reset   X undo"
            metrics = painter.fontMetrics()
            keys_width = min(max(560, metrics.horizontalAdvance(keys) + 24), max(120, self.width() - 20))
            keys_rect = QRectF(10, 46, keys_width, 26)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(18, 24, 23, 135))
            painter.drawRoundedRect(keys_rect, 8, 8)
            painter.setPen(QColor("#cbd9d6"))
            painter.drawText(keys_rect.adjusted(10, 0, 0, 0), Qt.AlignVCenter, keys)

        if self.progress_text:
            lines = [line.strip() for line in self.progress_text.splitlines() if line.strip()]
            if lines:
                metrics = painter.fontMetrics()
                text_width = max(metrics.horizontalAdvance(line) for line in lines)
                rect_width = min(max(380, text_width + 24), max(120, self.width() - 20))
                rect_height = 18 + len(lines) * 20
                progress_rect = QRectF(10, 82 if self.picking_enabled else 46, rect_width, rect_height)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(18, 24, 23, 150))
                painter.drawRoundedRect(progress_rect, 8, 8)
                painter.setPen(QColor("#e8eeee"))
                y = progress_rect.top() + 21
                for line in lines:
                    painter.drawText(QPointF(progress_rect.left() + 10, y), line)
                    y += 20

    def mousePressEvent(self, event) -> None:
        if event.button() not in (Qt.LeftButton, Qt.RightButton):
            return
        if self.image is None:
            return
        self.setFocus(Qt.MouseFocusReason)
        self._update_transform()

        pan_requested = event.button() == Qt.RightButton or event.modifiers() & Qt.ShiftModifier
        if pan_requested:
            self._drag_pan = True
            self._drag_start_pos = QPointF(event.pos())
            self._pan_start_x = self._pan_x
            self._pan_start_y = self._pan_y
            self.setCursor(Qt.ClosedHandCursor)
            return

        if not self.picking_enabled:
            self._drag_pan = True
            self._drag_start_pos = QPointF(event.pos())
            self._pan_start_x = self._pan_x
            self._pan_start_y = self._pan_y
            self.setCursor(Qt.ClosedHandCursor)
            return

        if self.mode in self.LANDMARK_ORDER and self.mode not in self.landmarks:
            if self._set_landmark_from_pos(self.mode, event.pos()):
                self._drag_landmark = self.mode
                self._drag_moved = False
                self._drag_start_pos = QPointF(event.pos())
            return

        nearest_landmark = self._nearest_landmark_for_pos(event.pos())
        if nearest_landmark is not None:
            self._drag_landmark = nearest_landmark
            self._drag_moved = False
            self._drag_start_pos = QPointF(event.pos())
            return

        if self.mode not in self.LANDMARK_ORDER:
            self._drag_pan = True
            self._drag_start_pos = QPointF(event.pos())
            self._pan_start_x = self._pan_x
            self._pan_start_y = self._pan_y
            self.setCursor(Qt.ClosedHandCursor)
            return
        if self._set_landmark_from_pos(self.mode, event.pos()):
            self._drag_landmark = self.mode
            self._drag_moved = False
            self._drag_start_pos = QPointF(event.pos())

    def mouseMoveEvent(self, event) -> None:
        if self._drag_pan:
            if self._drag_start_pos is None:
                return
            dx = event.pos().x() - self._drag_start_pos.x()
            dy = event.pos().y() - self._drag_start_pos.y()
            self._pan_x = self._pan_start_x + dx
            self._pan_y = self._pan_start_y + dy
            self.update()
            event.accept()
            return

        if self._drag_landmark is None:
            return
        if not event.buttons() & Qt.LeftButton:
            return
        if self._drag_start_pos is not None:
            dx = event.pos().x() - self._drag_start_pos.x()
            dy = event.pos().y() - self._drag_start_pos.y()
            self._drag_moved = self._drag_moved or (dx * dx + dy * dy > 9.0)
        self._set_landmark_from_pos(self._drag_landmark, event.pos())

    def mouseReleaseEvent(self, event) -> None:
        if event.button() not in (Qt.LeftButton, Qt.RightButton):
            return
        if self._drag_pan:
            self._drag_pan = False
            self._drag_start_pos = None
            self.unsetCursor()
            self.update()
            return

        if self._drag_landmark is None:
            return
        name = self._drag_landmark
        self._drag_landmark = None
        self._drag_start_pos = None
        self.landmark_changed.emit(name)
        self.update()

    def wheelEvent(self, event) -> None:
        if self.image is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.2 if delta > 0 else 1 / 1.2
        self.set_zoom(self.zoom_factor * factor, QPointF(event.pos()))
        event.accept()


class SurfacePreviewCanvas(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(330, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)
        self.boundaries = []
        self.current_boundary = None
        self.reference_contours = []
        self.slice_axis = 0
        self.message = "No surface preview yet"
        self.rotation_yaw = -0.55
        self.rotation_pitch = 0.38
        self.preview_zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._drag_pos: Optional[QPointF] = None
        self._drag_mode = "rotate"

    def set_boundaries(
        self,
        boundaries,
        current_boundary=None,
        slice_axis: int = 0,
        message: str = "",
    ) -> None:
        self.boundaries = list(boundaries or [])
        self.current_boundary = current_boundary
        self.slice_axis = slice_axis
        self.message = message or "No surface preview yet"
        self.update()

    def set_reference_contours(self, contours, slice_axis: int = 0) -> None:
        np = _numpy()
        self.reference_contours = [
            np.asarray(contour, dtype=float)
            for contour in (contours or [])
            if len(contour) >= 2
        ]
        self.slice_axis = slice_axis
        self.update()

    def _combined_boundaries(self):
        boundaries = list(self.boundaries)
        if self.current_boundary is not None:
            current_slice = self.current_boundary.slice_index
            boundaries = [item for item in boundaries if item.slice_index != current_slice]
            boundaries.append(self.current_boundary)
        return sorted(boundaries, key=lambda item: item.slice_index)

    def _all_points(self, boundaries):
        np = _numpy()
        point_sets = []
        for boundary in boundaries:
            point_sets.append(boundary.outer_arc)
            point_sets.append(boundary.inner_arc)
            point_sets.extend(boundary.lateral_arcs)
        point_sets.extend(self.reference_contours)
        point_sets = [np.asarray(points, dtype=float) for points in point_sets if len(points)]
        if not point_sets:
            return np.empty((0, 3), dtype=float)
        return np.vstack(point_sets)

    def _world_points(self, points):
        np = _numpy()
        core = _core()
        points = np.asarray(points, dtype=float)
        plane = core._volume_to_plane_points(points, self.slice_axis)
        depth = points[:, self.slice_axis]
        return np.column_stack((plane[:, 0], plane[:, 1], depth))

    def _project_raw(self, points, center):
        np = _numpy()
        coords = self._world_points(points) - center
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
        return np.column_stack((xz, yz))

    def _projection_transform(self, points):
        np = _numpy()
        world = self._world_points(points)
        center = (world.min(axis=0) + world.max(axis=0)) * 0.5
        radius = float(np.linalg.norm(world - center[None, :], axis=1).max())
        radius = max(radius, 1.0)
        margin = 34.0
        scale = self.preview_zoom * min(
            max(1.0, self.width() - margin * 2.0),
            max(1.0, self.height() - margin * 2.0),
        ) / (2.0 * radius)
        offset = np.array(
            [
                self.width() * 0.5 + self.pan_x,
                self.height() * 0.5 + self.pan_y,
            ],
            dtype=float,
        )
        return center, scale, offset

    def _screen_points(self, points, transform):
        center, scale, offset = transform
        projected = self._project_raw(points, center)
        screen = offset + projected * scale
        return [QPointF(float(point[0]), float(point[1])) for point in screen]

    def reset_view(self) -> None:
        self.rotation_yaw = -0.55
        self.rotation_pitch = 0.38
        self.preview_zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    @staticmethod
    def _wrap_rotation_angle(angle: float) -> float:
        return (angle + math.pi) % (math.pi * 2.0) - math.pi

    def _draw_volume_polyline(
        self,
        painter: QPainter,
        points,
        transform,
        color: QColor,
        width: float,
        line_style=Qt.SolidLine,
    ) -> None:
        screen_points = self._screen_points(points, transform)
        if len(screen_points) < 2:
            return
        pen = QPen(color, width)
        pen.setStyle(line_style)
        painter.setPen(pen)
        for p0, p1 in zip(screen_points[:-1], screen_points[1:]):
            painter.drawLine(p0, p1)

    def _draw_surface_connectors(
        self,
        painter: QPainter,
        boundaries,
        arc_name: str,
        transform,
        color: QColor,
    ) -> None:
        if len(boundaries) < 2:
            return
        painter.setPen(QPen(color, 1.0))
        for first, second in zip(boundaries[:-1], boundaries[1:]):
            first_arc = first.outer_arc if arc_name == "outer" else first.inner_arc
            second_arc = second.outer_arc if arc_name == "outer" else second.inner_arc
            count = min(len(first_arc), len(second_arc))
            if count < 2:
                continue
            step = max(1, count // 10)
            first_screen = self._screen_points(first_arc[:count], transform)
            second_screen = self._screen_points(second_arc[:count], transform)
            for index in range(0, count, step):
                painter.drawLine(first_screen[index], second_screen[index])

    def _reference_connection_limits(self):
        np = _numpy()
        centers = []
        radii = []
        for contour in self.reference_contours:
            world = self._world_points(contour)
            if len(world) == 0:
                centers.append(None)
                radii.append(0.0)
                continue
            center = world.mean(axis=0)
            centers.append(center)
            radii.append(float(np.linalg.norm(world - center[None, :], axis=1).max()))
        steps = [
            float(np.linalg.norm(second - first))
            for first, second in zip(centers[:-1], centers[1:])
            if first is not None and second is not None
        ]
        if not steps:
            return centers, radii, 0.0
        typical_step = float(np.median(np.asarray(steps, dtype=float)))
        return centers, radii, max(12.0, typical_step * 2.5)

    def _reference_contours_should_connect(
        self,
        first_center,
        second_center,
        first_radius: float,
        second_radius: float,
        max_step: float,
    ) -> bool:
        if first_center is None or second_center is None:
            return False
        np = _numpy()
        center_step = float(np.linalg.norm(second_center - first_center))
        local_size_limit = max(12.0, (first_radius + second_radius) * 2.5)
        return center_step <= max_step and center_step <= local_size_limit

    def _draw_reference_structure(self, painter: QPainter, transform) -> None:
        if not self.reference_contours:
            return
        outline_color = QColor(150, 178, 170, 95)
        connector_color = QColor(150, 178, 170, 45)
        for contour in self.reference_contours:
            self._draw_volume_polyline(
                painter,
                contour,
                transform,
                outline_color,
                1.0,
            )

        if len(self.reference_contours) < 2:
            return
        centers, radii, max_step = self._reference_connection_limits()
        painter.setPen(QPen(connector_color, 0.8))
        for index, (first, second) in enumerate(zip(self.reference_contours[:-1], self.reference_contours[1:])):
            if not self._reference_contours_should_connect(
                centers[index],
                centers[index + 1],
                radii[index],
                radii[index + 1],
                max_step,
            ):
                continue
            count = min(len(first), len(second))
            if count < 2:
                continue
            step = max(1, count // 12)
            first_screen = self._screen_points(first[:count], transform)
            second_screen = self._screen_points(second[:count], transform)
            for index in range(0, count, step):
                painter.drawLine(first_screen[index], second_screen[index])

    def _draw_message(self, painter: QPainter, text: str, top: float = 12.0) -> None:
        metrics = painter.fontMetrics()
        rect_width = min(max(260, metrics.horizontalAdvance(text) + 24), max(120, self.width() - 20))
        label_rect = QRectF(10, top, rect_width, 30)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(18, 24, 23, 215))
        painter.drawRoundedRect(label_rect, 8, 8)
        painter.setPen(QColor("#e8eeee"))
        painter.drawText(label_rect.adjusted(10, 0, 0, 0), Qt.AlignVCenter, text)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#101817"))

        boundaries = self._combined_boundaries()
        if not boundaries and not self.reference_contours:
            painter.setPen(QColor("#d4dddd"))
            painter.drawText(self.rect(), Qt.AlignCenter, self.message)
            return

        points = self._all_points(boundaries)
        if len(points) == 0:
            painter.setPen(QColor("#d4dddd"))
            painter.drawText(self.rect(), Qt.AlignCenter, self.message)
            return

        transform = self._projection_transform(points)
        self._draw_reference_structure(painter, transform)
        self._draw_surface_connectors(
            painter,
            boundaries,
            "outer",
            transform,
            QColor(70, 170, 235, 90),
        )
        self._draw_surface_connectors(
            painter,
            boundaries,
            "inner",
            transform,
            QColor(235, 90, 90, 90),
        )

        current_slice = self.current_boundary.slice_index if self.current_boundary is not None else None
        for boundary in boundaries:
            is_current = boundary.slice_index == current_slice
            style = Qt.DashLine if is_current else Qt.SolidLine
            self._draw_volume_polyline(
                painter,
                boundary.outer_arc,
                transform,
                QColor("#6ec7ff") if is_current else QColor("#2f8fd2"),
                3.2 if is_current else 2.0,
                style,
            )
            self._draw_volume_polyline(
                painter,
                boundary.inner_arc,
                transform,
                QColor("#ff8585") if is_current else QColor("#d64a4a"),
                3.2 if is_current else 2.0,
                style,
            )
            for lateral_arc in boundary.lateral_arcs:
                self._draw_volume_polyline(
                    painter,
                    lateral_arc,
                    transform,
                    QColor(220, 192, 90, 150 if is_current else 95),
                    1.5,
                    style,
                )

        self._draw_message(painter, self.message)

    def mousePressEvent(self, event) -> None:
        if event.button() not in (Qt.LeftButton, Qt.RightButton):
            return
        self.setFocus(Qt.MouseFocusReason)
        self._drag_pos = QPointF(event.pos())
        self._drag_mode = "pan" if event.button() == Qt.RightButton or event.modifiers() & Qt.ShiftModifier else "rotate"
        self.setCursor(Qt.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_pos is None:
            return
        dx = event.pos().x() - self._drag_pos.x()
        dy = event.pos().y() - self._drag_pos.y()
        self._drag_pos = QPointF(event.pos())
        if self._drag_mode == "pan":
            self.pan_x += dx
            self.pan_y += dy
        else:
            self.rotation_yaw = self._wrap_rotation_angle(self.rotation_yaw + dx * 0.01)
            self.rotation_pitch = self._wrap_rotation_angle(self.rotation_pitch + dy * 0.01)
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() not in (Qt.LeftButton, Qt.RightButton):
            return
        self._drag_pos = None
        self.unsetCursor()
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.reset_view()
            event.accept()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1 / 1.15
        self.preview_zoom = max(0.25, min(8.0, self.preview_zoom * factor))
        self.update()
        event.accept()


class LaminarBoundaryWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.thread: Optional[QThread] = None
        self.worker: Optional[Worker] = None
        self.progress_dialog: Optional[QProgressDialog] = None
        self.temporary_mask_dir: Optional[tempfile.TemporaryDirectory] = None
        self.annotation_mask_path: Optional[Path] = None
        self.annotation_mask_is_temporary = False
        self.active_help_popup: Optional[QFrame] = None
        self.active_help_button: Optional[QToolButton] = None
        self.setWindowTitle("Laminar Boundary Builder")
        self.resize(1180, 860)
        self.setMinimumSize(1040, 760)
        self.setStyleSheet(APP_STYLESHEET)

        self.status_label = QLabel()
        self.status_label.setObjectName("status")
        self._set_status("Ready", "ready")
        self.log_file_path = self._prepare_log_file()
        autosave_name = self.log_file_path.stem.replace(LOG_FILE_PREFIX.rstrip("_"), "manual_landmarks_autosave", 1)
        self.annotation_autosave_path = self.log_file_path.with_name(f"{autosave_name}.csv")
        self._make_menu_bar()
        self.append_log(
            "Laminar Boundary Builder log started "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"log_file: {self.log_file_path}\n\n"
        )

        self.tabs = QTabWidget()
        self.tabs.addTab(self._make_annotate_tab(), "1 Annotate")
        self.tabs.addTab(self._make_build_tab(), "2 Build")
        self.tabs.addTab(self._make_demo_tab(), "Demo test")

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        header = self._make_header()
        self.tabs.currentChanged.connect(self._update_flow_state)
        self._update_flow_state(self.tabs.currentIndex())
        layout.addWidget(header)
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(body)
        self._make_annotation_shortcuts()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def _make_header(self) -> QWidget:
        box = QFrame()
        box.setObjectName("header")
        box.setFrameShape(QFrame.StyledPanel)
        title = QLabel("Laminar Boundary Builder")
        title.setObjectName("title")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(9)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(self.status_label)

        flow_row = QHBoxLayout()
        flow_row.setContentsMargins(0, 0, 0, 0)
        flow_row.setSpacing(8)
        flow_caption = QLabel("Workflow")
        flow_caption.setObjectName("flowCaption")
        self.flow_annotate = self._make_flow_step("1 Annotate")
        self.flow_build = self._make_flow_step("2 Build")
        self.flow_demo = self._make_flow_step("Demo test")
        arrow = QLabel("->")
        arrow.setObjectName("flowArrow")
        flow_row.addWidget(flow_caption)
        flow_row.addSpacing(4)
        flow_row.addWidget(self.flow_annotate)
        flow_row.addWidget(arrow)
        flow_row.addWidget(self.flow_build)
        flow_row.addStretch(1)
        flow_row.addWidget(self.flow_demo)

        layout.addLayout(title_row)
        layout.addLayout(flow_row)
        return box

    def _make_flow_step(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("flowStep")
        label.setProperty("state", "pending")
        return label

    def _update_flow_state(self, index: int) -> None:
        if not hasattr(self, "flow_annotate"):
            return
        states = {
            0: ("active", "pending", "test"),
            1: ("done", "active", "test"),
            2: ("done", "done", "active"),
        }.get(index, ("pending", "pending", "test"))
        for label, state in zip((self.flow_annotate, self.flow_build, self.flow_demo), states):
            label.setProperty("state", state)
            label.style().unpolish(label)
            label.style().polish(label)

    def _set_status(self, text: str, state: str) -> None:
        self.status_label.setText(text)
        self.status_label.setProperty("state", state)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _prepare_log_file(self) -> Path:
        fallback_dir = Path(tempfile.gettempdir()) / "laminar_boundary_builder_logs"
        for log_dir in (LOG_CACHE_DIR, fallback_dir):
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                cleanup_old_log_files()
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                log_path = log_dir / f"{LOG_FILE_PREFIX}{timestamp}_{os.getpid()}.log"
                log_path.write_text("", encoding="utf-8")
                return log_path
            except OSError:
                continue
        return Path(tempfile.gettempdir()) / f"laminar_boundary_builder_{os.getpid()}.log"

    def _make_menu_bar(self) -> None:
        log_menu = self.menuBar().addMenu("Log")
        view_action = log_menu.addAction("View Current Log")
        view_action.triggered.connect(lambda _checked=False: self.show_log_dialog())
        folder_action = log_menu.addAction("Show Log Folder")
        folder_action.triggered.connect(lambda _checked=False: self.show_log_folder())
        clear_action = log_menu.addAction("Clear Current Log")
        clear_action.triggered.connect(lambda _checked=False: self.clear_current_log())

    def _read_current_log(self) -> str:
        try:
            return self.log_file_path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def show_log_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Current Log")
        dialog.resize(900, 620)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        path_label = QLabel(f"Current log file:\n{self.log_file_path}")
        path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(path_label)

        log_view = QTextEdit()
        log_view.setReadOnly(True)
        log_view.setFont(QFont("Menlo", 11))
        log_view.setPlainText(self._read_current_log() or "No log has been written yet.")
        layout.addWidget(log_view, 1)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        refresh_button = self._make_button("Refresh", "secondary")
        folder_button = self._make_button("Show Folder", "secondary")
        clear_button = self._make_button("Clear Log", "danger")
        close_button = self._make_button("Close", "primary")
        button_row.addWidget(refresh_button)
        button_row.addWidget(folder_button)
        button_row.addWidget(clear_button)
        button_row.addStretch(1)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

        refresh_button.clicked.connect(
            lambda: log_view.setPlainText(self._read_current_log() or "No log has been written yet.")
        )
        folder_button.clicked.connect(self.show_log_folder)

        def clear_and_refresh() -> None:
            self.clear_current_log(show_message=False)
            log_view.setPlainText(self._read_current_log() or "No log has been written yet.")

        clear_button.clicked.connect(clear_and_refresh)
        close_button.clicked.connect(dialog.accept)
        dialog.exec_()

    def show_log_folder(self) -> None:
        try:
            self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.log_file_path.parent)))

    def clear_current_log(self, show_message: bool = True) -> None:
        try:
            self.log_file_path.write_text("", encoding="utf-8")
            self.append_log(
                "Laminar Boundary Builder log cleared "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"log_file: {self.log_file_path}\n\n"
            )
        except OSError as exc:
            if show_message:
                QMessageBox.warning(self, "Could not clear log", str(exc))
            return
        if show_message:
            QMessageBox.information(self, "Log cleared", "Current log file was cleared.")

    def _make_button(self, text: str, role: str = "secondary") -> QPushButton:
        button = QPushButton(text)
        button.setProperty("role", role)
        return button

    def _make_annotation_shortcuts(self) -> None:
        self.undo_shortcut = QShortcut(QKeySequence(Qt.Key_X), self)
        self.undo_shortcut.setContext(Qt.ApplicationShortcut)
        self.undo_shortcut.activated.connect(self.undo_annotation_point)

        self.accept_shortcut = QShortcut(QKeySequence(Qt.Key_Return), self)
        self.accept_shortcut.setContext(Qt.ApplicationShortcut)
        self.accept_shortcut.activated.connect(self.accept_annotation_slice_and_advance)

        self.enter_shortcut = QShortcut(QKeySequence(Qt.Key_Enter), self)
        self.enter_shortcut.setContext(Qt.ApplicationShortcut)
        self.enter_shortcut.activated.connect(self.accept_annotation_slice_and_advance)

        self.exit_picking_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self.exit_picking_shortcut.setContext(Qt.ApplicationShortcut)
        self.exit_picking_shortcut.activated.connect(self.exit_annotation_picking_mode)

        self.previous_annotation_shortcuts = []
        for key in (Qt.Key_Left, Qt.Key_Up):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ApplicationShortcut)
            shortcut.activated.connect(lambda direction=-1: self.navigate_annotation_history(direction))
            self.previous_annotation_shortcuts.append(shortcut)

        self.next_annotation_shortcuts = []
        for key in (Qt.Key_Right, Qt.Key_Down):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ApplicationShortcut)
            shortcut.activated.connect(lambda direction=1: self.navigate_annotation_history(direction))
            self.next_annotation_shortcuts.append(shortcut)

        self.suggest_next_slice_shortcut = QShortcut(QKeySequence(Qt.Key_N), self)
        self.suggest_next_slice_shortcut.setContext(Qt.ApplicationShortcut)
        self.suggest_next_slice_shortcut.activated.connect(self.suggest_next_annotation_slice)

        self.skip_annotation_slice_shortcut = QShortcut(QKeySequence(Qt.Key_S), self)
        self.skip_annotation_slice_shortcut.setContext(Qt.ApplicationShortcut)
        self.skip_annotation_slice_shortcut.activated.connect(self.skip_current_annotation_slice)

        self.flip_outer_path_shortcut = QShortcut(QKeySequence(Qt.Key_O), self)
        self.flip_outer_path_shortcut.setContext(Qt.ApplicationShortcut)
        self.flip_outer_path_shortcut.activated.connect(lambda: self.flip_annotation_arc("outer"))
        self.flip_inner_path_shortcut = QShortcut(QKeySequence(Qt.Key_I), self)
        self.flip_inner_path_shortcut.setContext(Qt.ApplicationShortcut)
        self.flip_inner_path_shortcut.activated.connect(lambda: self.flip_annotation_arc("inner"))

        self.annotation_overlay_shortcut = QShortcut(QKeySequence(Qt.Key_H), self)
        self.annotation_overlay_shortcut.setContext(Qt.ApplicationShortcut)
        self.annotation_overlay_shortcut.activated.connect(lambda: self.slice_canvas.toggle_overlays())

        self.zoom_in_shortcut = QShortcut(QKeySequence(Qt.Key_Plus), self)
        self.zoom_in_shortcut.setContext(Qt.ApplicationShortcut)
        self.zoom_in_shortcut.activated.connect(lambda: self.slice_canvas.zoom_in())
        self.zoom_in_equal_shortcut = QShortcut(QKeySequence(Qt.Key_Equal), self)
        self.zoom_in_equal_shortcut.setContext(Qt.ApplicationShortcut)
        self.zoom_in_equal_shortcut.activated.connect(lambda: self.slice_canvas.zoom_in())
        self.zoom_out_shortcut = QShortcut(QKeySequence(Qt.Key_Minus), self)
        self.zoom_out_shortcut.setContext(Qt.ApplicationShortcut)
        self.zoom_out_shortcut.activated.connect(lambda: self.slice_canvas.zoom_out())
        self.zoom_reset_shortcut = QShortcut(QKeySequence(Qt.Key_0), self)
        self.zoom_reset_shortcut.setContext(Qt.ApplicationShortcut)
        self.zoom_reset_shortcut.activated.connect(lambda: self.slice_canvas.reset_zoom())

    def _make_hint(self, text: str) -> QFrame:
        box = QFrame()
        box.setObjectName("hint")
        layout = QHBoxLayout(box)
        layout.setContentsMargins(12, 8, 12, 8)
        label = QLabel(text)
        label.setObjectName("hintText")
        label.setWordWrap(True)
        layout.addWidget(label)
        return box

    def _tune_form(self, form: QFormLayout) -> None:
        form.setContentsMargins(10, 10, 10, 10)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

    def _add_help_row(self, form: QFormLayout, label: str, field: QWidget, title: str, body: str) -> QWidget:
        row = self._with_help(field, title, body)
        form.addRow(label, row)
        return row

    def _with_help(self, field: QWidget, title: str, body: str) -> QWidget:
        row = QWidget()
        row.setObjectName("parameterHelpRow")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(field, 1)
        layout.addWidget(self._make_help_button(title, body), 0, Qt.AlignVCenter)
        return row

    def _make_help_button(self, title: str, body: str) -> QToolButton:
        button = QToolButton()
        button.setObjectName("parameterHelpButton")
        button.setText("i")
        button.setCursor(Qt.PointingHandCursor)
        button.setToolTip("Show parameter help")
        button.clicked.connect(lambda _checked=False, btn=button: self.toggle_help_popup(btn, title, body))
        return button

    def toggle_help_popup(self, button: QToolButton, title: str, body: str) -> None:
        if self.active_help_popup is not None and self.active_help_popup.isVisible():
            if self.active_help_button is button:
                self._hide_help_popup()
                return
            self._hide_help_popup()

        popup = QFrame(self, Qt.Tool | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        popup.setObjectName("parameterHelpPopup")
        popup.setAttribute(Qt.WA_DeleteOnClose)
        popup.setAttribute(Qt.WA_TranslucentBackground)
        popup.setAutoFillBackground(False)
        popup.setFixedWidth(340)

        layout = QVBoxLayout(popup)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        card = QFrame()
        card.setObjectName("parameterHelpCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 10, 12, 10)
        card_layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setObjectName("parameterHelpTitle")
        title_label.setWordWrap(True)
        body_label = QLabel(body)
        body_label.setObjectName("parameterHelpBody")
        body_label.setWordWrap(True)
        body_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        card_layout.addWidget(title_label)
        card_layout.addWidget(body_label)
        layout.addWidget(card)

        popup.adjustSize()
        popup.move(self._help_popup_position(button, popup))
        popup.destroyed.connect(lambda _obj=None: self._clear_help_popup_reference(popup))
        self.active_help_popup = popup
        self.active_help_button = button
        popup.show()

    def _help_popup_position(self, button: QToolButton, popup: QFrame) -> QPoint:
        position = button.mapToGlobal(QPoint(button.width() + 8, -8))
        screen = QApplication.screenAt(position) or QApplication.primaryScreen()
        if screen is None:
            return position

        available = screen.availableGeometry()
        x = min(max(position.x(), available.left() + 8), available.right() - popup.width() - 8)
        y = min(max(position.y(), available.top() + 8), available.bottom() - popup.height() - 8)
        return QPoint(x, y)

    def _hide_help_popup(self) -> None:
        popup = self.active_help_popup
        self.active_help_popup = None
        self.active_help_button = None
        if popup is not None:
            popup.close()

    def _clear_help_popup_reference(self, popup: QFrame) -> None:
        if self.active_help_popup is popup:
            self.active_help_popup = None
            self.active_help_button = None

    def _widget_contains(self, root: QWidget, widget: Optional[QObject]) -> bool:
        current = widget
        while current is not None:
            if current is root:
                return True
            current = current.parent()
        return False

    def eventFilter(self, watched: QObject, event) -> bool:
        if event.type() == QEvent.ApplicationDeactivate:
            self._hide_help_popup()
        elif event.type() == QEvent.MouseButtonPress and self.active_help_popup is not None:
            if self.active_help_popup.isVisible():
                clicked_widget = watched if isinstance(watched, QWidget) else QApplication.widgetAt(event.globalPos())
                if self.active_help_button is not None and clicked_widget is self.active_help_button:
                    return False
                if self._widget_contains(self.active_help_popup, clicked_widget):
                    return False
                self._hide_help_popup()
        return super().eventFilter(watched, event)

    def _axis_combo(self) -> QComboBox:
        combo = CleanComboBox()
        combo.addItems(["coronal", "sagittal", "horizontal", "0", "1", "2"])
        combo.setMinimumHeight(28)
        return combo

    def _min_area_spin(self) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(0, 100000)
        spin.setValue(20)
        spin.setButtonSymbols(QSpinBox.NoButtons)
        spin.setMinimumHeight(28)
        return spin

    def _make_annotate_tab(self) -> QWidget:
        tab = QWidget()
        page_layout = QVBoxLayout(tab)
        page_layout.setContentsMargins(0, 8, 0, 0)
        page_layout.setSpacing(10)
        page_layout.addWidget(
            self._make_hint(
                "Step 1: extract or load a mask, then click four points in order: "
                "outer_start, outer_end, inner_start, inner_end. X undo, Enter accept + next, Esc edit settings."
            )
        )

        content = QWidget()
        layout = QHBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        controls = QGroupBox("1. Annotate Boundaries")
        controls.setMinimumWidth(420)
        controls.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        form = QFormLayout(controls)
        self._tune_form(form)

        self.annotate_atlas = PathRow(
            "Built-in Allen annotation_10.nrrd, or choose another atlas",
            file_filter="Atlas files (*.pkl *.nrrd *.nhdr *.npy *.npz);;All files (*)",
        )
        self.annotate_custom_atlas = QCheckBox("Use a custom Allen atlas file")
        self.annotate_custom_atlas.toggled.connect(self._update_custom_atlas_visibility)
        self.annotate_region = QLineEdit()
        self.annotate_region.setPlaceholderText("Brain region, for example ENT or 909")
        self.annotate_region.setMinimumHeight(28)
        self.annotate_hemisphere = CleanComboBox()
        self.annotate_hemisphere.addItems(["all", "left", "right"])
        self.annotate_hemisphere.setMinimumHeight(28)
        self.annotate_include_children = QCheckBox("Include child regions")
        self.annotate_include_children.setChecked(True)
        self.annotate_mask = PathRow("Optional existing target mask")
        self.annotate_template = PathRow("Optional template image volume")
        self.annotate_output = PathRow("Output folder for saved manual CSV", select_file=False)
        self.annotate_previous_csv = PathRow(
            "Optional previous manual_landmarks_interactive.csv",
            file_filter="CSV files (*.csv);;All files (*)",
        )
        self.annotate_slice_axis = self._axis_combo()
        self.annotate_min_area = self._min_area_spin()
        self.annotate_keep_all = QCheckBox("Keep all contours per slice")

        self.annotate_slice = QSpinBox()
        self.annotate_slice.setRange(0, 0)
        self.annotate_slice.setButtonSymbols(QSpinBox.NoButtons)
        self.annotate_slice.setMinimumHeight(28)
        self.annotate_slice.valueChanged.connect(self.refresh_annotation_slice)
        self.annotate_slider = QSlider(Qt.Horizontal)
        self.annotate_slider.setRange(0, 0)
        self.annotate_slider.valueChanged.connect(self.annotate_slice.setValue)
        self.annotate_slice.valueChanged.connect(self.annotate_slider.setValue)
        self.annotate_contour = CleanComboBox()
        self.annotate_contour.setMinimumHeight(28)
        self.annotate_contour.currentIndexChanged.connect(self.change_annotation_contour)

        self.annotate_outer_path = CleanComboBox()
        self.annotate_outer_path.addItems(["auto", "forward", "backward"])
        self.annotate_outer_path.setMinimumHeight(28)
        self.annotate_inner_path = CleanComboBox()
        self.annotate_inner_path.addItems(["auto", "forward", "backward"])
        self.annotate_inner_path.setMinimumHeight(28)
        self.annotate_outer_path.currentIndexChanged.connect(self.on_annotation_path_choice_changed)
        self.annotate_inner_path.currentIndexChanged.connect(self.on_annotation_path_choice_changed)

        self.load_button = self._make_button("Extract / Load / Start Picking", "primary")
        self.load_button.clicked.connect(self.load_annotation_data)
        self.load_previous_csv_button = self._make_button("Load Previous CSV", "secondary")
        self.load_previous_csv_button.setToolTip("Load saved manual landmarks into the annotation workspace.")
        self.load_previous_csv_button.clicked.connect(self.load_previous_annotation_csv)
        self.save_slice_button = self._make_button("Accept Slice + Next", "primary")
        self.save_slice_button.setToolTip("Accept landmarks on this slice and move to the next region slice.")
        self.save_slice_button.clicked.connect(self.accept_annotation_slice_and_advance)
        self.suggest_slice_button = self._make_button("Suggest Slice Set (N)", "secondary")
        self.suggest_slice_button.setToolTip("Build a stable suggested slice set and jump to its first unaccepted slice.")
        self.suggest_slice_button.clicked.connect(self.suggest_next_annotation_slice)
        self.skip_slice_button = self._make_button("Skip / No Inner (S)", "secondary")
        self.skip_slice_button.setToolTip("Skip this slice when it has no real inner surface.")
        self.skip_slice_button.clicked.connect(self.skip_current_annotation_slice)
        self.flip_outer_path_button = self._make_button("Flip Outer (O)", "secondary")
        self.flip_outer_path_button.setToolTip("Use the other contour side between outer_start and outer_end.")
        self.flip_outer_path_button.clicked.connect(lambda _checked=False: self.flip_annotation_arc("outer"))
        self.flip_inner_path_button = self._make_button("Flip Inner (I)", "secondary")
        self.flip_inner_path_button.setToolTip("Use the other contour side between inner_start and inner_end.")
        self.flip_inner_path_button.clicked.connect(lambda _checked=False: self.flip_annotation_arc("inner"))
        self.clear_slice_button = self._make_button("Clear Current Slice", "danger")
        self.clear_slice_button.setToolTip("Clear landmarks on the current slice.")
        self.clear_slice_button.clicked.connect(self.clear_annotation_slice)
        self.export_button = self._make_button("Save CSV for Build", "secondary")
        self.export_button.setToolTip("Save accepted landmarks as a CSV and switch to the Build step.")
        self.export_button.clicked.connect(self.export_annotation_csv)
        action_row = QWidget()
        action_layout = QVBoxLayout(action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(6)
        flip_row = QWidget()
        flip_layout = QHBoxLayout(flip_row)
        flip_layout.setContentsMargins(0, 0, 0, 0)
        flip_layout.setSpacing(6)
        flip_layout.addWidget(self.flip_outer_path_button)
        flip_layout.addWidget(self.flip_inner_path_button)
        action_layout.addWidget(self.save_slice_button)
        action_layout.addWidget(self.suggest_slice_button)
        action_layout.addWidget(self.skip_slice_button)
        action_layout.addWidget(flip_row)
        action_layout.addWidget(self.clear_slice_button)
        action_layout.addWidget(self.export_button)

        self.annotate_status = QLabel("No mask loaded")
        self.annotate_status.setWordWrap(True)
        self.next_point_label = QLabel("Next point: outer_start")
        self.next_point_label.setObjectName("nextPointText")
        self.next_point_label.setWordWrap(True)
        self.annotate_progress = QLabel("Load a mask to see slice count.")
        self.annotate_progress.setObjectName("progressText")
        self.annotate_progress.setWordWrap(True)

        self._add_help_row(form, "Brain region", self.annotate_region, *ANNOTATE_HELP["region"])
        self._add_help_row(form, "Hemisphere", self.annotate_hemisphere, *ANNOTATE_HELP["hemisphere"])
        self._add_help_row(form, "", self.annotate_include_children, *ANNOTATE_HELP["include_children"])
        self._add_help_row(form, "", self.annotate_custom_atlas, *ANNOTATE_HELP["custom_atlas_enabled"])
        self.annotate_atlas_row = self._add_help_row(
            form, "Custom atlas", self.annotate_atlas, *ANNOTATE_HELP["custom_atlas"]
        )
        self.annotate_atlas_label = form.labelForField(self.annotate_atlas_row)
        self._update_custom_atlas_visibility(False)
        self._add_help_row(form, "Mask", self.annotate_mask, *ANNOTATE_HELP["mask"])
        self._add_help_row(form, "Template image", self.annotate_template, *ANNOTATE_HELP["template"])
        self._add_help_row(form, "Output folder", self.annotate_output, *ANNOTATE_HELP["output"])
        self._add_help_row(form, "Previous CSV", self.annotate_previous_csv, *ANNOTATE_HELP["previous_csv"])
        form.addRow("", self.load_previous_csv_button)
        self._add_help_row(form, "Slice axis", self.annotate_slice_axis, *ANNOTATE_HELP["slice_axis"])
        self._add_help_row(form, "Min contour area", self.annotate_min_area, *ANNOTATE_HELP["min_area"])
        self._add_help_row(form, "", self.annotate_keep_all, *ANNOTATE_HELP["keep_all"])
        form.addRow("", self.load_button)
        self._add_help_row(form, "Slice", self.annotate_slice, *ANNOTATE_HELP["slice"])
        form.addRow("", self.annotate_slider)
        self._add_help_row(form, "Contour", self.annotate_contour, *ANNOTATE_HELP["contour"])
        self._add_help_row(form, "Outer arc path", self.annotate_outer_path, *ANNOTATE_HELP["outer_path"])
        self._add_help_row(form, "Inner arc path", self.annotate_inner_path, *ANNOTATE_HELP["inner_path"])
        form.addRow("Next", self.next_point_label)
        form.addRow("Actions", action_row)
        form.addRow("Progress", self.annotate_progress)
        form.addRow("Status", self.annotate_status)

        self.slice_canvas = SliceCanvas()
        self.slice_canvas.landmark_changed.connect(self.on_landmark_changed)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.slice_canvas)

        self.surface_preview_canvas = SurfacePreviewCanvas()
        self.surface_preview_canvas.hide()
        self.annotation_preview_splitter = QSplitter(Qt.Horizontal)
        self.annotation_preview_splitter.setChildrenCollapsible(False)
        self.annotation_preview_splitter.addWidget(scroll)
        self.annotation_preview_splitter.addWidget(self.surface_preview_canvas)
        self.annotation_preview_splitter.setStretchFactor(0, 3)
        self.annotation_preview_splitter.setStretchFactor(1, 2)
        self.annotation_preview_splitter.setSizes([740, 430])

        self.annotate_settings_button = QPushButton("›")
        self.annotate_settings_button.setObjectName("settingsPeek")
        self.annotate_settings_button.setToolTip("Show or hide annotation settings. Esc exits point-picking mode.")
        self.annotate_settings_button.clicked.connect(self.toggle_annotation_settings_panel)
        self.annotate_settings_button.hide()

        self.annotate_controls_scroll = QScrollArea()
        self.annotate_controls_scroll.setObjectName("sideScroll")
        self.annotate_controls_scroll.setWidgetResizable(True)
        self.annotate_controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.annotate_controls_scroll.setWidget(controls)
        self.annotate_controls_scroll.setMinimumWidth(450)
        self.annotate_controls_scroll.setMaximumWidth(480)

        layout.addWidget(self.annotate_settings_button)
        layout.addWidget(self.annotate_controls_scroll)
        layout.addWidget(self.annotation_preview_splitter, 1)
        page_layout.addWidget(content, 1)

        self.annotation_mask_data = None
        self.annotation_template_data = None
        self.annotation_slice_axis_int = 0
        self.annotation_rows: Dict[int, Dict[str, str]] = {}
        self.annotation_landmarks_by_slice: Dict[int, Dict[str, int]] = {}
        self.annotation_path_choices_by_slice: Dict[int, Dict[str, str]] = {}
        self.annotation_skipped_slices = set()
        self.annotation_contours_by_slice: Dict[int, list] = {}
        self.annotation_region_slices: List[int] = []
        self.annotation_target_slices: List[int] = []
        self.annotation_reference_contours = []
        self.annotation_slice_counts = None
        self.annotation_picking_active = False
        self.annotation_settings_expanded = True
        return tab

    def _annotation_shortcuts_active(self) -> bool:
        return self.tabs.currentIndex() == 0 and self.annotation_picking_active

    def _next_annotation_point(self) -> Optional[str]:
        for name in SliceCanvas.LANDMARK_ORDER:
            if name not in self.slice_canvas.landmarks:
                return name
        return None

    def _set_next_annotation_mode(self) -> None:
        next_point = self._next_annotation_point()
        if next_point:
            self.slice_canvas.set_mode(next_point)
            self.next_point_label.setText(
                f"Next point: {next_point}. Click on the contour. X = undo."
            )
        else:
            self.slice_canvas.set_mode("")
            self.next_point_label.setText(
                "All four points are set. Press Enter or click Accept Slice + Next."
            )
        self.slice_canvas.set_picking_enabled(self.annotation_picking_active)

    def enter_annotation_picking_mode(self) -> None:
        self.annotation_picking_active = True
        self.annotation_settings_expanded = False
        self.annotate_settings_button.show()
        self._set_annotation_parameter_widgets_enabled(False)
        self._apply_annotation_settings_panel_state()
        self._set_next_annotation_mode()
        self.slice_canvas.setFocus(Qt.OtherFocusReason)

    def exit_annotation_picking_mode(self) -> None:
        if self.tabs.currentIndex() != 0:
            return
        if not self.annotation_picking_active:
            return
        self.annotation_picking_active = False
        self.annotation_settings_expanded = True
        self.annotate_settings_button.hide()
        self._set_annotation_parameter_widgets_enabled(True)
        self._apply_annotation_settings_panel_state()
        self._set_next_annotation_mode()
        self.annotate_status.setText("Point picking paused. Edit settings, then Load Mask / Start Picking again.")

    def toggle_annotation_settings_panel(self) -> None:
        if not self.annotation_picking_active:
            return
        self.annotation_settings_expanded = not self.annotation_settings_expanded
        self._apply_annotation_settings_panel_state()

    def _apply_annotation_settings_panel_state(self) -> None:
        expanded = self.annotation_settings_expanded or not self.annotation_picking_active
        self.annotate_controls_scroll.setVisible(expanded)
        self.annotate_settings_button.setText("‹" if expanded else "›")
        if hasattr(self, "surface_preview_canvas"):
            self.surface_preview_canvas.setVisible(self.annotation_picking_active)

    def _update_custom_atlas_visibility(self, checked: bool) -> None:
        if hasattr(self, "annotate_atlas_row"):
            self.annotate_atlas_row.setVisible(checked)
        if hasattr(self, "annotate_atlas_label"):
            self.annotate_atlas_label.setVisible(checked)

    def _set_annotation_parameter_widgets_enabled(self, enabled: bool) -> None:
        for widget in (
            self.annotate_custom_atlas,
            self.annotate_atlas,
            self.annotate_region,
            self.annotate_hemisphere,
            self.annotate_include_children,
            self.annotate_mask,
            self.annotate_template,
            self.annotate_output,
            self.annotate_previous_csv,
            self.annotate_slice_axis,
            self.annotate_min_area,
            self.annotate_keep_all,
            self.load_button,
            self.load_previous_csv_button,
        ):
            widget.setEnabled(enabled)

    def _path_choices_for_slice(self, slice_index: int) -> Dict[str, str]:
        choices = {"outer_path": "auto", "inner_path": "auto"}
        row = self.annotation_rows.get(slice_index)
        if row is not None:
            choices["outer_path"] = self._normalize_annotation_path_choice(row.get("outer_path"))
            choices["inner_path"] = self._normalize_annotation_path_choice(row.get("inner_path"))
        stored = self.annotation_path_choices_by_slice.get(slice_index)
        if stored is not None:
            choices["outer_path"] = self._normalize_annotation_path_choice(
                stored.get("outer_path") or choices["outer_path"]
            )
            choices["inner_path"] = self._normalize_annotation_path_choice(
                stored.get("inner_path") or choices["inner_path"]
            )
        return choices

    @staticmethod
    def _normalize_annotation_path_choice(text: Optional[str]) -> str:
        value = str(text or "").strip().lower()
        if value in ("forward", "cw", "clockwise", "+", "+1", "1"):
            return "forward"
        if value in ("backward", "ccw", "counterclockwise", "-", "-1"):
            return "backward"
        return "auto"

    def _set_combo_text_without_signal(self, combo: QComboBox, text: str) -> None:
        normalized = self._normalize_annotation_path_choice(text)
        value = normalized if combo.findText(normalized) >= 0 else "auto"
        combo.blockSignals(True)
        combo.setCurrentText(value)
        combo.blockSignals(False)

    def _set_annotation_path_widgets(self, outer_path: str, inner_path: str) -> None:
        self._set_combo_text_without_signal(self.annotate_outer_path, outer_path)
        self._set_combo_text_without_signal(self.annotate_inner_path, inner_path)

    def _remember_current_path_choices(self) -> None:
        if self.annotation_mask_data is None:
            return
        slice_index = int(self.annotate_slice.value())
        self.annotation_path_choices_by_slice[slice_index] = {
            "outer_path": self.annotate_outer_path.currentText(),
            "inner_path": self.annotate_inner_path.currentText(),
        }

    def _sync_current_path_choices_to_row(self, note: str = "path_edit") -> Optional[Path]:
        slice_index = int(self.annotate_slice.value())
        row = self.annotation_rows.get(slice_index)
        if row is None:
            return None
        row["outer_path"] = self.annotate_outer_path.currentText()
        row["inner_path"] = self.annotate_inner_path.currentText()
        row["note"] = note
        return self._autosave_annotation_rows()

    def on_annotation_path_choice_changed(self, *_args) -> None:
        if not hasattr(self, "slice_canvas"):
            return
        if self.annotation_mask_data is not None:
            self._remember_current_path_choices()
            self._sync_current_path_choices_to_row()
        self.refresh_annotation_preview()

    def _effective_annotation_arc_direction(self, arc_name: str) -> Optional[int]:
        contour = self.slice_canvas.selected_contour()
        if contour is None:
            return None
        landmarks = self.slice_canvas.landmarks
        start_name = f"{arc_name}_start"
        end_name = f"{arc_name}_end"
        if start_name not in landmarks or end_name not in landmarks:
            return None
        path_choice = (
            self.annotate_outer_path.currentText()
            if arc_name == "outer"
            else self.annotate_inner_path.currentText()
        )
        try:
            _, _, direction, _ = _core()._choose_arc(
                contour.points,
                landmarks[start_name],
                landmarks[end_name],
                choice=path_choice,
            )
            return int(direction)
        except Exception:
            return None

    def flip_annotation_arc(self, arc_name: str) -> None:
        if not self._annotation_shortcuts_active():
            return
        if arc_name not in ("outer", "inner"):
            return
        start_name = f"{arc_name}_start"
        end_name = f"{arc_name}_end"
        if start_name not in self.slice_canvas.landmarks or end_name not in self.slice_canvas.landmarks:
            self.annotate_status.setText(
                f"Set {start_name} and {end_name} before flipping the {arc_name} path."
            )
            return

        direction = self._effective_annotation_arc_direction(arc_name)
        if direction is None:
            self.annotate_status.setText(f"Could not flip the {arc_name} path on this slice.")
            return
        new_choice = "backward" if direction > 0 else "forward"
        combo = self.annotate_outer_path if arc_name == "outer" else self.annotate_inner_path
        combo.blockSignals(True)
        combo.setCurrentText(new_choice)
        combo.blockSignals(False)
        self._remember_current_path_choices()
        autosave_path = self._sync_current_path_choices_to_row()
        self.refresh_annotation_preview()

        slice_index = int(self.annotate_slice.value())
        saved = f" Autosaved: {autosave_path}" if autosave_path is not None else ""
        self.annotate_status.setText(
            f"Slice {slice_index}: {arc_name} path flipped to {new_choice}.{saved}"
        )
        self.slice_canvas.setFocus(Qt.OtherFocusReason)

    def _closest_region_slice(self, target: float) -> Optional[int]:
        slices = self._target_annotation_slices()
        if not slices:
            return None
        return min(
            slices,
            key=lambda slice_index: (abs(slice_index - target), slice_index),
        )

    def _target_annotation_slices(self) -> List[int]:
        source = self.annotation_target_slices or self.annotation_region_slices
        return [slice_index for slice_index in source if slice_index not in self.annotation_skipped_slices]

    def _viable_annotation_slices(self) -> List[int]:
        region_slices = list(self.annotation_region_slices)
        if not region_slices:
            return []
        np = _numpy()
        counts = self.annotation_slice_counts
        if counts is None:
            return region_slices
        region_counts = np.asarray([int(counts[slice_index]) for slice_index in region_slices], dtype=float)
        if region_counts.size == 0:
            return []
        max_count = float(region_counts.max())
        if max_count <= 0:
            return []
        min_count = max(float(self.annotate_min_area.value()) * 2.0, max_count * 0.025)
        viable = [
            slice_index
            for slice_index, count in zip(region_slices, region_counts)
            if count >= min_count
        ]
        return viable or region_slices

    def _pick_evenly_spaced_slices(self, slices: List[int], count: int) -> List[int]:
        if count >= len(slices):
            return list(slices)
        if count <= 1:
            return [slices[len(slices) // 2]]
        np = _numpy()
        positions = np.linspace(0, len(slices) - 1, count)
        picked = []
        for position in positions:
            slice_index = slices[int(round(float(position)))]
            if slice_index not in picked:
                picked.append(slice_index)
        return picked

    @staticmethod
    def _slice_ranges(slices: List[int]) -> List[tuple[int, int]]:
        ranges: List[tuple[int, int]] = []
        start = None
        previous = None
        for slice_index in sorted(set(slices)):
            if start is None:
                start = previous = slice_index
            elif previous is not None and slice_index == previous + 1:
                previous = slice_index
            else:
                ranges.append((int(start), int(previous)))
                start = previous = slice_index
        if start is not None and previous is not None:
            ranges.append((int(start), int(previous)))
        return ranges

    @staticmethod
    def _review_targets_from_ranges(slices: List[int]) -> List[int]:
        targets = set()
        for start, end in LaminarBoundaryWindow._slice_ranges(slices):
            middle = (start + end) // 2
            targets.update((start, middle, end))
        return sorted(targets)

    @staticmethod
    def _read_qc_review_slices(output_dir: Path) -> List[int]:
        qc_path = output_dir / "qc" / "qc_uncertain_slices.csv"
        if not qc_path.exists():
            return []
        review_slices = []
        try:
            with qc_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    slice_text = str(row.get("slice_index", "")).strip()
                    if not slice_text:
                        continue
                    review_slices.append(int(float(slice_text)))
        except (OSError, ValueError):
            return []
        return sorted(set(review_slices))

    def _current_qc_review_slices(self) -> List[int]:
        output_text = ""
        if hasattr(self, "build_output"):
            output_text = self.build_output.text().strip()
        if output_text:
            return self._read_qc_review_slices(Path(output_text).expanduser())
        annotate_output = self.annotate_output.text().strip() if hasattr(self, "annotate_output") else ""
        if annotate_output:
            return self._read_qc_review_slices(Path(annotate_output).expanduser() / "build")
        return []

    def _build_suggested_annotation_set(self) -> List[int]:
        viable = self._viable_annotation_slices()
        if not viable:
            return []
        np = _numpy()
        target_count = min(len(viable), max(6, min(18, (len(viable) + 23) // 24)))
        picked = set(self._pick_evenly_spaced_slices(viable, target_count))

        counts = self.annotation_slice_counts
        if counts is not None and len(viable) >= 3:
            viable_counts = np.asarray([int(counts[slice_index]) for slice_index in viable], dtype=float)
            changes = np.abs(np.diff(viable_counts))
            change_slots = max(1, target_count // 4)
            for index in np.argsort(changes)[::-1][:change_slots]:
                left = viable[int(index)]
                right = viable[int(index) + 1]
                middle = self._closest_slice_from_list(viable, (left + right) * 0.5)
                if middle is not None:
                    picked.add(middle)

        qc_review_slices = [
            slice_index for slice_index in self._current_qc_review_slices() if slice_index in viable
        ]
        picked.update(self._review_targets_from_ranges(qc_review_slices))

        return sorted(picked)

    def _closest_slice_from_list(self, slices: List[int], target: float) -> Optional[int]:
        if not slices:
            return None
        return min(slices, key=lambda slice_index: (abs(slice_index - target), slice_index))

    def _suggest_annotation_slice(self) -> tuple[Optional[int], str]:
        target_slices = self._target_annotation_slices()
        if not target_slices:
            return None, "No annotation target slice is available yet."

        accepted = sorted(slice_index for slice_index in self.annotation_rows if slice_index in target_slices)
        first_slice = target_slices[0]
        last_slice = target_slices[-1]
        if not accepted:
            return first_slice, "Start the suggested set from the first stable slice."

        if first_slice not in self.annotation_rows:
            return first_slice, "Add the first stable slice in the suggested set."
        if last_slice not in self.annotation_rows:
            return last_slice, "Add the last stable slice in the suggested set."

        best_target = None
        best_gap = -1
        best_pair = None
        for left, right in zip(accepted[:-1], accepted[1:]):
            between = [slice_index for slice_index in target_slices if left < slice_index < right]
            candidates = [slice_index for slice_index in between if slice_index not in self.annotation_rows]
            if not candidates:
                continue
            gap = len(between) + 1
            middle = (left + right) * 0.5
            target = min(candidates, key=lambda slice_index: (abs(slice_index - middle), slice_index))
            if gap > best_gap:
                best_gap = gap
                best_target = target
                best_pair = (left, right)

        if best_target is not None and best_pair is not None:
            left, right = best_pair
            return best_target, f"Fill the widest unchecked gap between accepted slices {left} and {right}."

        missing = [slice_index for slice_index in target_slices if slice_index not in self.annotation_rows]
        if missing:
            current = int(self.annotate_slice.value())
            target = min(missing, key=lambda slice_index: (abs(slice_index - current), slice_index))
            return target, "Fill the nearest unaccepted suggested slice."

        return None, "All suggested slices are already accepted."

    def suggest_next_annotation_slice(self) -> None:
        if not self._annotation_shortcuts_active():
            return
        self.annotation_target_slices = self._build_suggested_annotation_set()
        target, reason = self._suggest_annotation_slice()
        if target is None:
            self.annotate_status.setText(reason)
            self._update_annotation_progress()
            return
        self.annotate_slice.setValue(target)
        self.slice_canvas.setFocus(Qt.OtherFocusReason)
        self.annotate_status.setText(
            f"Suggested set active: {len(self.annotation_target_slices)} slices. "
            f"Now showing slice {target}. {reason}"
        )
        self._update_annotation_progress()

    def skip_current_annotation_slice(self) -> None:
        if not self._annotation_shortcuts_active():
            return
        if self.annotation_mask_data is None:
            return
        slice_index = int(self.annotate_slice.value())
        self.annotation_skipped_slices.add(slice_index)
        self.annotation_rows.pop(slice_index, None)
        self.annotation_landmarks_by_slice.pop(slice_index, None)
        self.annotation_path_choices_by_slice.pop(slice_index, None)
        self.slice_canvas.landmarks = {}
        self._set_annotation_path_widgets("auto", "auto")
        self._set_next_annotation_mode()
        self.refresh_annotation_preview()
        self._update_annotation_status()
        autosave_path = self._autosave_annotation_rows()
        saved = f" Autosaved: {autosave_path}" if autosave_path is not None else ""
        self.append_log(f"Skipped slice {slice_index}: no inner surface.{saved}\n")

        if self._annotation_target_set_complete():
            self.finish_annotation_and_run_build()
            return
        self.annotate_status.setText(f"Skipped slice {slice_index}: no real inner surface.{saved}")
        self._go_to_next_annotation_slice()

    def undo_annotation_point(self) -> None:
        if not self._annotation_shortcuts_active():
            return
        for name in reversed(SliceCanvas.LANDMARK_ORDER):
            if name in self.slice_canvas.landmarks:
                self.slice_canvas.landmarks.pop(name, None)
                slice_index = int(self.annotate_slice.value())
                self.annotation_landmarks_by_slice[slice_index] = dict(self.slice_canvas.landmarks)
                self._set_next_annotation_mode()
                self._update_annotation_status()
                self.slice_canvas.update()
                return

    def navigate_annotation_history(self, direction: int) -> None:
        if not self._annotation_shortcuts_active():
            return
        accepted_slices = sorted(self.annotation_rows)
        if not accepted_slices:
            self.annotate_status.setText("No accepted slices yet.")
            return

        current = int(self.annotate_slice.value())
        if direction < 0:
            candidates = [slice_index for slice_index in accepted_slices if slice_index < current]
            target = candidates[-1] if candidates else accepted_slices[-1]
        else:
            candidates = [slice_index for slice_index in accepted_slices if slice_index > current]
            target = candidates[0] if candidates else accepted_slices[0]

        self.annotate_slice.setValue(target)
        self.slice_canvas.setFocus(Qt.OtherFocusReason)
        self.annotate_status.setText(f"Showing accepted slice {target}. Click a point to move it.")

    def accept_annotation_slice_and_advance(self) -> None:
        if not self._annotation_shortcuts_active():
            return
        if self.accept_annotation_slice(show_success=False):
            if self._annotation_target_set_complete():
                self.finish_annotation_and_run_build()
                return
            self._go_to_next_annotation_slice()

    def _annotation_target_set_complete(self) -> bool:
        if not self.annotation_target_slices:
            return False
        target_slices = self._target_annotation_slices()
        if not target_slices:
            return False
        return all(slice_index in self.annotation_rows for slice_index in target_slices)

    def _auto_build_output_dir(self) -> Path:
        output_text = self.annotate_output.text().strip()
        if output_text:
            return Path(output_text).expanduser()
        if self.annotation_mask_path is not None and not self.annotation_mask_is_temporary:
            return Path(self.annotation_mask_path).expanduser().parent / "laminar_boundary_builder_output"
        return Path.home() / "Desktop" / "laminar_boundary_builder_output"

    def finish_annotation_and_run_build(self) -> None:
        target_slices = self._target_annotation_slices()
        missing = [slice_index for slice_index in target_slices if slice_index not in self.annotation_rows]
        if missing:
            self.annotate_slice.setValue(missing[0])
            self.annotate_status.setText(f"Still missing suggested slice {missing[0]}.")
            return
        try:
            output_dir = self._auto_build_output_dir()
            if not self.annotate_output.text().strip():
                self.annotate_output.set_text(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = output_dir / "manual_landmarks_interactive.csv"
            self._write_annotation_rows_csv(csv_path)
            self._sync_build_from_annotation(csv_path, output_dir)
            self.annotation_picking_active = False
            self.annotation_settings_expanded = True
            self.annotate_settings_button.hide()
            self._set_annotation_parameter_widgets_enabled(True)
            self._apply_annotation_settings_panel_state()
            self.append_log(
                f"Suggested annotation set complete: {len(target_slices)} slices.\n"
                f"Saved manual landmarks: {csv_path}\n"
                "Auto-starting build.\n"
            )
            self.run_build()
        except Exception as exc:
            QMessageBox.critical(self, "Auto build failed", str(exc))

    def _go_to_next_annotation_slice(self) -> None:
        current = int(self.annotate_slice.value())
        target_slices = self._target_annotation_slices()
        next_slice = None
        for slice_index in target_slices:
            if slice_index > current:
                next_slice = slice_index
                break
        if next_slice is None:
            missing = [slice_index for slice_index in target_slices if slice_index not in self.annotation_rows]
            if missing:
                next_slice = missing[0]
            elif self.annotation_target_slices:
                self.annotate_status.setText("Reached the last suggested slice. Press Enter to build.")
                return
            else:
                self.annotate_status.setText("Reached the last region slice. Save CSV when you are done.")
                return

        self.annotate_slice.setValue(next_slice)
        self.slice_canvas.setFocus(Qt.OtherFocusReason)

    def _extract_contours_for_annotation_slice(self, slice_index: int) -> list:
        if self.annotation_mask_data is None:
            return []
        core = _core()
        np = _numpy()
        mask_2d = core._take_slice(
            self.annotation_mask_data,
            slice_index,
            self.annotation_slice_axis_int,
        )
        mask_2d = np.asarray(mask_2d) > 0
        raw_contours = core._find_mask_contours(mask_2d)
        contours = []
        for raw in raw_contours:
            area = core._polygon_area(raw)
            if area < float(self.annotate_min_area.value()):
                continue
            points = core._plane_to_volume_points(raw, slice_index, self.annotation_slice_axis_int)
            contours.append(
                core.Contour2D(
                    slice_index=slice_index,
                    contour_id=len(contours),
                    points=points,
                    area=area,
                    length=core._polyline_length(points, closed=True),
                )
            )
        if contours and not self.annotate_keep_all.isChecked():
            biggest = max(contours, key=lambda contour: contour.area)
            biggest.contour_id = 0
            return [biggest]
        return contours

    def _build_annotation_reference_contours(
        self,
        max_slices: int = 56,
        points_per_contour: int = 96,
    ) -> list:
        if self.annotation_mask_data is None or not self.annotation_region_slices:
            return []
        core = _core()
        np = _numpy()
        region_slices = list(self.annotation_region_slices)
        if len(region_slices) > max_slices:
            positions = np.linspace(0, len(region_slices) - 1, max_slices)
            sample_slices = []
            for position in positions:
                slice_index = region_slices[int(round(float(position)))]
                if slice_index not in sample_slices:
                    sample_slices.append(slice_index)
        else:
            sample_slices = region_slices

        reference_contours = []
        min_area = max(1.0, float(self.annotate_min_area.value()))
        for slice_index in sample_slices:
            mask_2d = core._take_slice(
                self.annotation_mask_data,
                slice_index,
                self.annotation_slice_axis_int,
            )
            raw_contours = core._find_mask_contours(np.asarray(mask_2d) > 0)
            candidates = [
                raw for raw in raw_contours if core._polygon_area(raw) >= min_area
            ]
            if not candidates:
                continue
            raw = max(candidates, key=core._polygon_area)
            points = core._plane_to_volume_points(raw, slice_index, self.annotation_slice_axis_int)
            points = core._normalize_contour(points)
            if len(points) < 2:
                continue
            closed = np.vstack([points, points[:1]])
            if len(closed) > points_per_contour:
                closed = core.resample_polyline(closed, points_per_contour + 1)
            reference_contours.append(closed)
        return reference_contours

    def _refresh_annotation_reference_contours(self) -> None:
        self.annotation_reference_contours = self._build_annotation_reference_contours()
        if hasattr(self, "surface_preview_canvas"):
            self.surface_preview_canvas.set_reference_contours(
                self.annotation_reference_contours,
                self.annotation_slice_axis_int,
            )

    def _uses_atlas_extraction(self) -> bool:
        custom_atlas = self.annotate_custom_atlas.isChecked() and self.annotate_atlas.text()
        return bool(custom_atlas or self.annotate_region.text().strip())

    def _cleanup_temporary_mask(self) -> None:
        if self.temporary_mask_dir is not None:
            self.temporary_mask_dir.cleanup()
            self.temporary_mask_dir = None

    def _load_annotation_template(self, mask_shape) -> Optional[object]:
        if not self.annotate_template.text():
            return None

        core = _core()
        template_path = self._require_path("Template image", self.annotate_template.text())
        try:
            template = core.load_volume(template_path).data
        except Exception as exc:
            raise RuntimeError(f"Could not read Template image:\n{template_path}\n\n{exc}") from exc
        if template.shape == mask_shape:
            return template

        QMessageBox.warning(
            self,
            "Template ignored",
            "Template shape does not match the mask, so only the mask will be shown.",
        )
        return None

    def _finish_annotation_load(
        self,
        mask_data,
        mask_path: str | Path,
        template_data,
        temporary: bool,
    ) -> None:
        np = _numpy()
        core = _core()
        mask_array = np.asarray(mask_data)
        if mask_array.dtype == np.bool_:
            self.annotation_mask_data = mask_array
        else:
            self.annotation_mask_data = mask_array > 0
        self.annotation_template_data = template_data
        self.annotation_slice_axis_int = core._slice_axis_to_int(self.annotate_slice_axis.currentText())
        self.annotation_mask_path = Path(mask_path).expanduser()
        self.annotation_mask_is_temporary = temporary
        self.annotate_mask.set_text(self.annotation_mask_path)

        max_slice = self.annotation_mask_data.shape[self.annotation_slice_axis_int] - 1
        self.annotate_slice.setRange(0, max_slice)
        self.annotate_slider.setRange(0, max_slice)
        slice_counts = self.annotation_mask_data.sum(
            axis=tuple(axis for axis in range(3) if axis != self.annotation_slice_axis_int)
        )
        self.annotation_slice_counts = np.asarray(slice_counts, dtype=np.int64)
        nonzero = np.flatnonzero(slice_counts)
        self.annotation_region_slices = [int(index) for index in nonzero]
        self.annotation_target_slices = []
        if len(nonzero):
            self.annotate_slice.setValue(int(nonzero[len(nonzero) // 2]))
        else:
            self.annotate_slice.setValue(0)
        self.annotation_landmarks_by_slice.clear()
        self.annotation_rows.clear()
        self.annotation_path_choices_by_slice.clear()
        self.annotation_skipped_slices.clear()
        self.annotation_contours_by_slice.clear()
        self._refresh_annotation_reference_contours()
        self._set_annotation_path_widgets("auto", "auto")
        self.refresh_annotation_slice()
        self.enter_annotation_picking_mode()

    def _update_progress_dialog(self, text: str) -> None:
        if self.progress_dialog is None:
            return
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            self.progress_dialog.setLabelText(lines[-1])

    def _close_progress_dialog(self) -> None:
        if self.progress_dialog is not None:
            self.progress_dialog.close()
            self.progress_dialog = None

    def start_annotation_mask_extraction(self) -> None:
        if self.thread is not None:
            QMessageBox.warning(self, "Task running", "Please wait for the current task to finish.")
            return

        core = _core()
        atlas_text = self.annotate_atlas.text() if self.annotate_custom_atlas.isChecked() else ""
        atlas_path = core.resolve_annotation_path(atlas_text or None)
        region_text = self.annotate_region.text().strip()
        if not region_text:
            raise ValueError("Brain region is required when extracting from Allen atlas.")

        self._cleanup_temporary_mask()
        self.temporary_mask_dir = tempfile.TemporaryDirectory(prefix=TEMP_MASK_PREFIX)
        _write_temp_mask_marker(self.temporary_mask_dir.name)
        temp_mask_path = Path(self.temporary_mask_dir.name) / "extracted_region_mask.npy"
        template_path = self.annotate_template.text() or None
        include_children = self.annotate_include_children.isChecked()
        hemisphere = self.annotate_hemisphere.currentText()

        def task() -> TaskResult:
            core = _core()
            extraction = core.extract_region_mask_from_annotation(
                annotation_path=atlas_path,
                region_text=region_text,
                output_path=temp_mask_path,
                template_path=template_path,
                include_children=include_children,
                hemisphere=hemisphere,
                progress=print,
            )
            lines = [
                "Mask extraction finished.",
                f"atlas: {atlas_path}",
                f"temporary_mask: {extraction.mask_path}",
                f"region: {extraction.region_label}",
                f"region_ids: {len(extraction.region_ids)}",
                f"voxels: {extraction.voxel_count}",
            ]
            lines.extend(f"warning: {warning}" for warning in extraction.warnings)
            return TaskResult("Mask extraction finished", "\n".join(lines), payload=extraction)

        self.append_log("\n--- Mask extraction started ---\n")
        self._set_status("Running: extracting mask", "running")
        self.progress_dialog = QProgressDialog("Starting mask extraction...", "", 0, 0, self)
        self.progress_dialog.setWindowTitle("Extracting mask")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.show()

        self.thread = QThread()
        self.worker = Worker(task)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.log.connect(self._update_progress_dialog)
        self.worker.finished.connect(self.annotation_mask_extraction_finished)
        self.worker.failed.connect(self.annotation_mask_extraction_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.clear_thread)
        self.thread.start()

    def annotation_mask_extraction_finished(self, result: TaskResult) -> None:
        self._close_progress_dialog()
        self._set_status("Ready", "ready")
        self.append_log(f"\n{result.message}\n")
        extraction = result.payload
        try:
            self._finish_annotation_load(
                mask_data=extraction.mask,
                mask_path=extraction.mask_path,
                template_data=extraction.template,
                temporary=True,
            )
            warning_text = "\n".join(extraction.warnings)
            if warning_text:
                QMessageBox.warning(self, "Mask extracted with warning", warning_text)
            self.append_log(f"Loaded temporary annotation mask: {extraction.mask_path}\n")
        except Exception as exc:
            self._set_status("Failed", "failed")
            QMessageBox.critical(self, "Load failed", str(exc))

    def annotation_mask_extraction_failed(self, trace: str) -> None:
        self._close_progress_dialog()
        self._set_status("Failed", "failed")
        self._cleanup_temporary_mask()
        self.append_log("\n" + trace)
        QMessageBox.critical(self, "Mask extraction failed", trace.splitlines()[-1] if trace else "Unknown error")

    def load_annotation_data(self) -> None:
        try:
            core = _core()
            if self._uses_atlas_extraction():
                self.start_annotation_mask_extraction()
                return

            mask_path = self._require_path("Mask", self.annotate_mask.text())
            try:
                mask_data = core.load_volume(mask_path).data
            except Exception as exc:
                raise RuntimeError(f"Could not read Mask:\n{mask_path}\n\n{exc}") from exc
            old_temp_dir = self.temporary_mask_dir
            old_temp_path = self.annotation_mask_path if self.annotation_mask_is_temporary else None
            template_data = self._load_annotation_template(mask_data.shape)
            self._finish_annotation_load(mask_data, mask_path, template_data, temporary=False)
            if old_temp_dir is not None and Path(mask_path).expanduser() != old_temp_path:
                old_temp_dir.cleanup()
                if old_temp_dir is self.temporary_mask_dir:
                    self.temporary_mask_dir = None
            self.append_log(f"Loaded annotation mask: {mask_path}\n")
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))

    def _annotation_image_for_slice(self, slice_index: int):
        core = _core()
        np = _numpy()
        if self.annotation_template_data is not None:
            return core._take_slice(self.annotation_template_data, slice_index, self.annotation_slice_axis_int)
        mask_slice = core._take_slice(self.annotation_mask_data, slice_index, self.annotation_slice_axis_int)
        return np.asarray(mask_slice, dtype=np.uint8) * 255

    def _prune_annotation_contour_cache(self, current_slice: int) -> None:
        if len(self.annotation_contours_by_slice) <= ANNOTATION_CONTOUR_CACHE_LIMIT:
            return
        protected = set(self.annotation_rows)
        protected.add(current_slice)
        removable = [
            slice_index
            for slice_index in self.annotation_contours_by_slice
            if slice_index not in protected
        ]
        removable.sort(key=lambda item: abs(item - current_slice), reverse=True)
        overflow = len(self.annotation_contours_by_slice) - ANNOTATION_CONTOUR_CACHE_LIMIT
        for slice_index in removable[:overflow]:
            self.annotation_contours_by_slice.pop(slice_index, None)

    def refresh_annotation_slice(self) -> None:
        if self.annotation_mask_data is None:
            return
        slice_index = int(self.annotate_slice.value())
        contours = self.annotation_contours_by_slice.get(slice_index)
        if contours is None:
            contours = self._extract_contours_for_annotation_slice(slice_index)
            self.annotation_contours_by_slice[slice_index] = contours
            self._prune_annotation_contour_cache(slice_index)

        previous_index = max(0, self.annotate_contour.currentIndex())
        selected_index = previous_index
        accepted_row = self.annotation_rows.get(slice_index)
        if accepted_row is not None:
            try:
                contour_id = int(float(accepted_row.get("contour_id", "")))
                for index, contour in enumerate(contours):
                    if contour.contour_id == contour_id:
                        selected_index = index
                        break
            except (TypeError, ValueError):
                selected_index = previous_index
        self.annotate_contour.blockSignals(True)
        self.annotate_contour.clear()
        core = _core()
        for index, contour in enumerate(contours):
            self.annotate_contour.addItem(
                f"{index}: area {contour.area:.0f}, points {len(core._normalize_contour(contour.points))}"
            )
        if contours:
            self.annotate_contour.setCurrentIndex(min(selected_index, len(contours) - 1))
        self.annotate_contour.blockSignals(False)

        landmarks = self.annotation_landmarks_by_slice.get(slice_index, {})
        path_choices = self._path_choices_for_slice(slice_index)
        self._set_annotation_path_widgets(path_choices["outer_path"], path_choices["inner_path"])
        image = self._annotation_image_for_slice(slice_index)
        self.slice_canvas.set_scene(
            image=image,
            contours=contours,
            landmarks=landmarks,
            selected_contour_index=max(0, self.annotate_contour.currentIndex()),
            slice_axis=self.annotation_slice_axis_int,
        )
        self._set_next_annotation_mode()
        self._update_annotation_status()
        self.refresh_annotation_preview()

    def change_annotation_contour(self) -> None:
        if self.annotation_mask_data is None:
            return
        self.refresh_annotation_slice()

    def _annotation_row_from_landmarks(
        self,
        contour,
        landmarks: Dict[str, int],
        note: str = "interactive",
        outer_path: Optional[str] = None,
        inner_path: Optional[str] = None,
    ) -> Dict[str, str]:
        core = _core()
        points = core._normalize_contour(contour.points)
        outer_choice = self._normalize_annotation_path_choice(
            outer_path if outer_path is not None else self.annotate_outer_path.currentText()
        )
        inner_choice = self._normalize_annotation_path_choice(
            inner_path if inner_path is not None else self.annotate_inner_path.currentText()
        )
        return {
            "slice_index": str(contour.slice_index),
            "contour_id": str(contour.contour_id),
            "outer_start_index": str(landmarks["outer_start"]),
            "outer_start_x": f"{points[landmarks['outer_start'], 0]:.4f}",
            "outer_start_y": f"{points[landmarks['outer_start'], 1]:.4f}",
            "outer_start_z": f"{points[landmarks['outer_start'], 2]:.4f}",
            "outer_end_index": str(landmarks["outer_end"]),
            "outer_end_x": f"{points[landmarks['outer_end'], 0]:.4f}",
            "outer_end_y": f"{points[landmarks['outer_end'], 1]:.4f}",
            "outer_end_z": f"{points[landmarks['outer_end'], 2]:.4f}",
            "outer_path": outer_choice,
            "inner_start_index": str(landmarks["inner_start"]),
            "inner_start_x": f"{points[landmarks['inner_start'], 0]:.4f}",
            "inner_start_y": f"{points[landmarks['inner_start'], 1]:.4f}",
            "inner_start_z": f"{points[landmarks['inner_start'], 2]:.4f}",
            "inner_end_index": str(landmarks["inner_end"]),
            "inner_end_x": f"{points[landmarks['inner_end'], 0]:.4f}",
            "inner_end_y": f"{points[landmarks['inner_end'], 1]:.4f}",
            "inner_end_z": f"{points[landmarks['inner_end'], 2]:.4f}",
            "inner_path": inner_choice,
            "note": note,
        }

    def _annotation_csv_fieldnames(self) -> List[str]:
        return [
            "slice_index",
            "contour_id",
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

    def _write_annotation_rows_csv(self, csv_path: Path) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._annotation_csv_fieldnames())
            writer.writeheader()
            for slice_index in sorted(self.annotation_rows):
                writer.writerow(self.annotation_rows[slice_index])

    def _autosave_annotation_rows(self) -> Optional[Path]:
        if not self.annotation_rows:
            return None
        saved_path = None
        try:
            self._write_annotation_rows_csv(self.annotation_autosave_path)
            saved_path = self.annotation_autosave_path
        except Exception as exc:
            self.append_log(f"Autosave failed: {exc}\n")

        output_text = self.annotate_output.text().strip()
        if output_text:
            try:
                output_path = Path(output_text).expanduser() / "manual_landmarks_autosave.csv"
                self._write_annotation_rows_csv(output_path)
                saved_path = output_path
            except Exception as exc:
                self.append_log(f"Output autosave failed: {exc}\n")
        return saved_path

    def _landmark_index_from_loaded_row(self, row: Dict[str, str], points, name: str) -> int:
        np = _numpy()
        core = _core()
        for key in (f"{name}_index", name):
            text = str(row.get(key, "")).strip()
            if not text:
                continue
            try:
                index = int(float(text))
            except ValueError:
                continue
            if 0 <= index < len(points):
                return index

        coord_keys = (f"{name}_x", f"{name}_y", f"{name}_z")
        if all(str(row.get(key, "")).strip() for key in coord_keys):
            point = np.asarray([float(row[key]) for key in coord_keys], dtype=float)
            return core._nearest_index(points, point)

        raise ValueError(f"Missing landmark {name}")

    def _loaded_annotation_row(self, row: Dict[str, str]) -> tuple[int, Dict[str, str], Dict[str, int]]:
        core = _core()
        slice_index = int(float(row["slice_index"]))
        if self.annotation_mask_data is None:
            raise ValueError("Load a mask before loading previous manual CSV.")
        max_slice = self.annotation_mask_data.shape[self.annotation_slice_axis_int] - 1
        if slice_index < 0 or slice_index > max_slice:
            raise ValueError(f"Slice {slice_index} is outside the loaded mask range 0-{max_slice}.")

        contours = self.annotation_contours_by_slice.get(slice_index)
        if contours is None:
            contours = self._extract_contours_for_annotation_slice(slice_index)
            self.annotation_contours_by_slice[slice_index] = contours
            self._prune_annotation_contour_cache(slice_index)
        contour = core._select_contour_for_row(row, contours)
        points = core._normalize_contour(contour.points)
        landmarks = {
            name: self._landmark_index_from_loaded_row(row, points, name)
            for name in SliceCanvas.LANDMARK_ORDER
        }
        loaded_row = self._annotation_row_from_landmarks(
            contour,
            landmarks,
            note=row.get("note") or "loaded",
            outer_path=row.get("outer_path"),
            inner_path=row.get("inner_path"),
        )
        return slice_index, loaded_row, landmarks

    def _annotation_boundary_from_row(self, row: Dict[str, str]):
        core = _core()
        slice_index = int(float(row["slice_index"]))
        contours = self.annotation_contours_by_slice.get(slice_index)
        if contours is None:
            contours = self._extract_contours_for_annotation_slice(slice_index)
            self.annotation_contours_by_slice[slice_index] = contours
        contour = core._select_contour_for_row(row, contours)
        return core.make_boundary_from_landmark_row(contour, row, resample_points=64)

    def _current_annotation_boundary(self):
        contour = self.slice_canvas.selected_contour()
        if contour is None:
            return None
        landmarks = dict(self.slice_canvas.landmarks)
        if not all(name in landmarks for name in SliceCanvas.LANDMARK_ORDER):
            return None
        row = self._annotation_row_from_landmarks(contour, landmarks, note="live")
        return _core().make_boundary_from_landmark_row(contour, row, resample_points=64)

    def refresh_annotation_preview(self, *_args) -> None:
        if not hasattr(self, "slice_canvas"):
            return
        self.slice_canvas.set_path_choices(
            self.annotate_outer_path.currentText(),
            self.annotate_inner_path.currentText(),
        )
        if not hasattr(self, "surface_preview_canvas"):
            return
        if self.annotation_mask_data is None:
            self.surface_preview_canvas.set_reference_contours(
                [],
                self.annotation_slice_axis_int,
            )
            self.surface_preview_canvas.set_boundaries(
                [],
                slice_axis=self.annotation_slice_axis_int,
                message="No surface preview yet",
            )
            return

        current_slice = int(self.annotate_slice.value())
        current_landmarks = self.annotation_landmarks_by_slice.get(current_slice, self.slice_canvas.landmarks)
        current_complete = all(name in current_landmarks for name in SliceCanvas.LANDMARK_ORDER)
        boundaries = []
        skipped = 0
        for row in self.annotation_rows.values():
            row_slice = int(float(row.get("slice_index", -1)))
            if row_slice == current_slice and not current_complete:
                continue
            try:
                boundaries.append(self._annotation_boundary_from_row(row))
            except Exception:
                skipped += 1

        current_boundary = None
        try:
            current_boundary = self._current_annotation_boundary()
        except Exception:
            current_boundary = None

        shown_slices = {boundary.slice_index for boundary in boundaries}
        if current_boundary is not None:
            shown_slices.add(current_boundary.slice_index)
        shown_count = len(shown_slices)
        if shown_count == 0:
            message = "Target region reference"
        else:
            message = f"Surface preview: {shown_count} slice"
            if shown_count != 1:
                message += "s"
            if current_boundary is not None:
                message += " including current"
        if skipped:
            message += f", {skipped} skipped"
        if self.annotation_skipped_slices:
            message += f", {len(self.annotation_skipped_slices)} no-inner"
        self.surface_preview_canvas.set_boundaries(
            boundaries,
            current_boundary=current_boundary,
            slice_axis=self.annotation_slice_axis_int,
            message=message,
        )

    def on_landmark_changed(self, mode: str) -> None:
        slice_index = int(self.annotate_slice.value())
        self.annotation_landmarks_by_slice[slice_index] = dict(self.slice_canvas.landmarks)
        landmarks = self.annotation_landmarks_by_slice[slice_index]
        is_complete = all(name in landmarks for name in SliceCanvas.LANDMARK_ORDER)
        if slice_index in self.annotation_rows and is_complete:
            contour = self.slice_canvas.selected_contour()
            if contour is not None:
                self.annotation_rows[slice_index] = self._annotation_row_from_landmarks(
                    contour,
                    landmarks,
                    note="interactive_edit",
                )
                autosave_path = self._autosave_annotation_rows()
                if autosave_path is not None:
                    self.append_log(f"Updated accepted slice {slice_index}; autosaved: {autosave_path}\n")
        self._set_next_annotation_mode()
        self._update_annotation_status()
        self._refresh_annotation_reference_contours()
        self.refresh_annotation_preview()

    def _update_annotation_status(self) -> None:
        if self.annotation_mask_data is None:
            self.annotate_status.setText("No mask loaded")
            self.annotate_progress.setText("Load a mask to see slice count.")
            self.slice_canvas.set_progress_text("")
            return
        slice_index = int(self.annotate_slice.value())
        landmarks = self.annotation_landmarks_by_slice.get(slice_index, self.slice_canvas.landmarks)
        missing = [
            name
            for name in ("outer_start", "outer_end", "inner_start", "inner_end")
            if name not in landmarks
        ]
        accepted_count = len(self.annotation_rows)
        contour_text = "no contour" if not self.slice_canvas.contours else f"contour {self.slice_canvas.selected_contour_index}"
        if slice_index in self.annotation_skipped_slices:
            self.annotate_status.setText(
                f"Slice {slice_index} is skipped because it has no real inner surface. "
                f"Clear Current Slice if you want to annotate it."
            )
            self._update_annotation_progress()
            return
        if missing:
            prefix = "Editing accepted slice. " if slice_index in self.annotation_rows else ""
            self.annotate_status.setText(
                f"{prefix}Slice {slice_index}, {contour_text}. Missing: {', '.join(missing)}. "
                f"Accepted slices: {accepted_count}."
            )
        else:
            self.annotate_status.setText(
                f"Slice {slice_index}, {contour_text}. All four points are set. "
                f"Press Enter or click Accept Slice + Next. Accepted slices: {accepted_count}."
            )
        self._update_annotation_progress()

    def _recommended_annotation_count(self) -> int:
        if self.annotation_target_slices:
            return len(self._target_annotation_slices())
        total = len(self.annotation_region_slices)
        if total == 0:
            return 0
        return min(total, max(3, (total + 7) // 8))

    def _update_annotation_progress(self) -> None:
        region_total = len(self.annotation_region_slices)
        target_slices = self._target_annotation_slices()
        target_total = len(target_slices)
        accepted = len([slice_index for slice_index in self.annotation_rows if slice_index in target_slices])
        skipped = len(self.annotation_skipped_slices)
        if region_total == 0:
            self.annotate_progress.setText("Region slices: 0. No annotation target yet.")
            self.slice_canvas.set_progress_text("Progress: no region slice found.")
            return
        first_slice = self.annotation_region_slices[0]
        last_slice = self.annotation_region_slices[-1]
        target = self._recommended_annotation_count()
        remaining = max(0, target - accepted)
        current_slice = int(self.annotate_slice.value())
        current_position = 0
        for index, slice_index in enumerate(target_slices, start=1):
            if slice_index >= current_slice:
                current_position = index
                break
        if current_position == 0:
            current_position = target_total
        mode_label = "Suggested set" if self.annotation_target_slices else "Region slices"
        self.annotate_progress.setText(
            f"Region slices: {region_total} ({first_slice}-{last_slice}). "
            f"{mode_label}: {target_total}. "
            f"Accepted target slices: {accepted}/{target}. Remaining: {remaining}. "
            f"Skipped no-inner: {skipped}."
        )
        self.slice_canvas.set_progress_text(
            f"Accepted target slices: {accepted}/{target}, remaining {remaining}\n"
            f"Current target slice: {current_position}/{target_total} (slice {current_slice})\n"
            f"Region range: {first_slice}-{last_slice}, skipped no-inner {skipped}"
        )

    def accept_annotation_slice(self, show_success: bool = True) -> bool:
        if self.annotation_mask_data is None:
            QMessageBox.warning(self, "No mask", "Load a mask first.")
            return False
        contour = self.slice_canvas.selected_contour()
        if contour is None:
            QMessageBox.warning(self, "No contour", "No usable contour is available on this slice.")
            return False
        slice_index = int(self.annotate_slice.value())
        landmarks = dict(self.slice_canvas.landmarks)
        missing = [
            name
            for name in ("outer_start", "outer_end", "inner_start", "inner_end")
            if name not in landmarks
        ]
        if missing:
            QMessageBox.warning(self, "Missing points", "Please set: " + ", ".join(missing))
            return False

        row = self._annotation_row_from_landmarks(contour, landmarks)
        self.annotation_rows[slice_index] = row
        self.annotation_landmarks_by_slice[slice_index] = landmarks
        self.annotation_skipped_slices.discard(slice_index)
        self.annotation_path_choices_by_slice[slice_index] = {
            "outer_path": row["outer_path"],
            "inner_path": row["inner_path"],
        }
        self._refresh_annotation_reference_contours()
        self._update_annotation_status()
        self.refresh_annotation_preview()
        self.append_log(f"Accepted manual landmarks for slice {slice_index}\n")
        autosave_path = self._autosave_annotation_rows()
        if autosave_path is not None:
            self.append_log(f"Autosaved manual landmarks: {autosave_path}\n")
        if show_success:
            self.slice_canvas.setFocus(Qt.OtherFocusReason)
        return True

    def clear_annotation_slice(self) -> None:
        slice_index = int(self.annotate_slice.value())
        self.annotation_landmarks_by_slice.pop(slice_index, None)
        self.annotation_rows.pop(slice_index, None)
        self.annotation_path_choices_by_slice.pop(slice_index, None)
        self.annotation_skipped_slices.discard(slice_index)
        self._set_annotation_path_widgets("auto", "auto")
        self.slice_canvas.landmarks = {}
        self._set_next_annotation_mode()
        self.slice_canvas.update()
        self._refresh_annotation_reference_contours()
        self._update_annotation_status()
        self.refresh_annotation_preview()
        self._autosave_annotation_rows()

    def load_previous_annotation_csv(self) -> None:
        if self.annotation_mask_data is None:
            QMessageBox.warning(
                self,
                "Load mask first",
                "Load the same target mask before loading a previous manual CSV.",
            )
            return

        try:
            csv_path = self._require_path("Previous CSV", self.annotate_previous_csv.text())
            rows = _core().read_manual_landmarks(csv_path)
            loaded_rows: Dict[int, Dict[str, str]] = {}
            loaded_landmarks: Dict[int, Dict[str, int]] = {}
            loaded_paths: Dict[int, Dict[str, str]] = {}
            errors = []
            for row in rows:
                try:
                    slice_index, loaded_row, landmarks = self._loaded_annotation_row(row)
                except Exception as exc:
                    errors.append(f"slice {row.get('slice_index', '?')}: {exc}")
                    continue
                loaded_rows[slice_index] = loaded_row
                loaded_landmarks[slice_index] = landmarks
                loaded_paths[slice_index] = {
                    "outer_path": loaded_row["outer_path"],
                    "inner_path": loaded_row["inner_path"],
                }

            if not loaded_rows:
                detail = "\n".join(errors[:8])
                raise ValueError("No usable annotation rows were loaded." + (f"\n{detail}" if detail else ""))

            self.annotation_rows = loaded_rows
            self.annotation_landmarks_by_slice = loaded_landmarks
            self.annotation_path_choices_by_slice = loaded_paths
            self.annotation_skipped_slices.clear()
            self.annotation_target_slices = sorted(loaded_rows)
            self.annotate_previous_csv.set_text(csv_path)
            if not self.annotate_output.text().strip():
                self.annotate_output.set_text(csv_path.parent)
            output_dir = Path(self.annotate_output.text()).expanduser()
            self._sync_build_from_annotation(csv_path, output_dir)
            first_slice = self.annotation_target_slices[0]
            current_slice = int(self.annotate_slice.value())
            target = current_slice if current_slice in loaded_rows else first_slice
            self.annotate_slice.setValue(target)
            self.refresh_annotation_slice()
            self.enter_annotation_picking_mode()
            autosave_path = self._autosave_annotation_rows()

            message = f"Loaded {len(loaded_rows)} annotation slices from:\n{csv_path}"
            if errors:
                message += f"\n\nSkipped {len(errors)} rows. See Log > View Current Log for details."
                self.append_log("Previous CSV load skipped rows:\n" + "\n".join(errors) + "\n")
            if autosave_path is not None:
                message += f"\n\nAutosaved editable copy:\n{autosave_path}"
            self.annotate_status.setText(
                f"Loaded {len(loaded_rows)} previous annotation slices. "
                "Use arrows/history or the slice control to edit selected slices."
            )
            self.append_log(f"Loaded previous manual CSV: {csv_path}\n")
            QMessageBox.information(self, "Previous CSV loaded", message)
        except Exception as exc:
            QMessageBox.critical(self, "Load previous CSV failed", str(exc))

    def export_annotation_csv(self) -> None:
        if not self.annotation_rows:
            QMessageBox.warning(self, "No accepted slices", "Accept at least one slice before saving.")
            return
        try:
            output_dir = self._require_path("Output folder", self.annotate_output.text())
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = output_dir / "manual_landmarks_interactive.csv"
            self._write_annotation_rows_csv(csv_path)
            self._sync_build_from_annotation(csv_path, output_dir)
            QMessageBox.information(
                self,
                "Manual CSV saved",
                f"Saved:\n{csv_path}\n\nBuild is ready with the same mask and settings.",
            )
            self.append_log(f"Saved interactive manual CSV: {csv_path}\n")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _sync_build_from_annotation(self, csv_path: Path, annotation_output_dir: Path) -> None:
        self.build_manual.set_text(csv_path)
        self.build_mask.set_text(self.annotation_mask_path or self.annotate_mask.text())
        self.build_template.set_text(self.annotate_template.text())
        self.build_output.set_text(annotation_output_dir / "build")
        self.build_boundaries.set_text(annotation_output_dir / "build" / "boundary_annotations.json")
        self.build_slice_axis.setCurrentText(self.annotate_slice_axis.currentText())
        self.build_min_area.setValue(self.annotate_min_area.value())
        self.build_keep_all.setChecked(self.annotate_keep_all.isChecked())
        self.depth_method.setCurrentText("surfaces only")
        self.tabs.setCurrentIndex(1)

    def _make_prepare_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(12)

        settings = QGroupBox("Prepare Annotation Template")

        self.prepare_mask = PathRow("Target mask volume (.nrrd/.npy/.npz)")
        self.prepare_output = PathRow("Output folder for contour template", select_file=False)
        self.prepare_slice_axis = self._axis_combo()
        self.prepare_min_area = self._min_area_spin()
        self.prepare_manual_every = QSpinBox()
        self.prepare_manual_every.setRange(1, 500)
        self.prepare_manual_every.setValue(8)
        self.prepare_manual_every.setButtonSymbols(QSpinBox.NoButtons)
        self.prepare_manual_every.setMinimumHeight(28)
        self.prepare_keep_all = QCheckBox("Keep all contours per slice")

        form = QFormLayout(settings)
        self._tune_form(form)
        form.addRow("Mask", self.prepare_mask)
        form.addRow("Output folder", self.prepare_output)
        form.addRow("Slice axis", self.prepare_slice_axis)
        form.addRow("Min contour area", self.prepare_min_area)
        form.addRow("Manual row interval", self.prepare_manual_every)
        form.addRow("", self.prepare_keep_all)

        run_button = self._make_button("Prepare Annotation Template", "primary")
        run_button.clicked.connect(self.run_prepare)
        run_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        layout.addWidget(settings)
        layout.addWidget(run_button)
        layout.addStretch(1)
        return tab

    def _make_build_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(
            self._make_hint("Step 2: extract surfaces first. Compute laminar depth volume later if needed.")
        )

        settings = QGroupBox("2. Extract Surfaces / Optional Depth Volume")

        self.build_mask = PathRow("Target mask volume (.nrrd/.npy/.npz)")
        self.build_manual = PathRow("manual_landmarks_template.csv", file_filter="CSV files (*.csv);;All files (*)")
        self.build_boundaries = PathRow("boundary_annotations.json", file_filter="JSON files (*.json);;All files (*)")
        self.build_output = PathRow("Output folder for surfaces, volumes, tables, and QC", select_file=False)
        self.build_template = PathRow("Optional template image volume")
        self.build_cell_csv = PathRow("Optional soma coordinate CSV", file_filter="CSV files (*.csv);;All files (*)")
        self.build_swc_glob = QLineEdit()
        self.build_swc_glob.setPlaceholderText("Optional SWC glob, for example data/local/swc/*.swc")
        self.build_slice_axis = self._axis_combo()
        self.build_min_area = self._min_area_spin()

        self.resample_points = QSpinBox()
        self.resample_points.setRange(8, 1000)
        self.resample_points.setValue(80)
        self.resample_points.setButtonSymbols(QSpinBox.NoButtons)
        self.resample_points.setMinimumHeight(28)
        self.depth_method = CleanComboBox()
        self.depth_method.addItems(["surfaces only", "auto", "laplace", "distance"])
        self.depth_method.setMinimumHeight(28)
        self.volume_format = CleanComboBox()
        self.volume_format.addItems(["nrrd", "npy", "nii.gz"])
        self.volume_format.setMinimumHeight(28)
        self.max_laplace_voxels = QSpinBox()
        self.max_laplace_voxels.setRange(1000, 100000000)
        self.max_laplace_voxels.setSingleStep(50000)
        self.max_laplace_voxels.setValue(250000)
        self.max_laplace_voxels.setButtonSymbols(QSpinBox.NoButtons)
        self.max_laplace_voxels.setMinimumHeight(28)
        self.boundary_dilation = QSpinBox()
        self.boundary_dilation.setRange(0, 10)
        self.boundary_dilation.setValue(1)
        self.boundary_dilation.setButtonSymbols(QSpinBox.NoButtons)
        self.boundary_dilation.setMinimumHeight(28)
        self.qc_every = QSpinBox()
        self.qc_every.setRange(1, 500)
        self.qc_every.setValue(10)
        self.qc_every.setButtonSymbols(QSpinBox.NoButtons)
        self.qc_every.setMinimumHeight(28)
        self.build_keep_all = QCheckBox("Keep all contours per slice")

        form = QFormLayout(settings)
        self._tune_form(form)
        self._add_help_row(form, "Mask", self.build_mask, *BUILD_HELP["mask"])
        self._add_help_row(form, "Manual CSV", self.build_manual, *BUILD_HELP["manual_csv"])
        self._add_help_row(form, "Boundary JSON", self.build_boundaries, *BUILD_HELP["boundaries_json"])
        self._add_help_row(form, "Output folder", self.build_output, *BUILD_HELP["output"])
        self._add_help_row(form, "Template image", self.build_template, *BUILD_HELP["template"])
        self._add_help_row(form, "Cell CSV", self.build_cell_csv, *BUILD_HELP["cell_csv"])
        self._add_help_row(form, "SWC glob", self.build_swc_glob, *BUILD_HELP["swc_glob"])
        self._add_help_row(form, "Slice axis", self.build_slice_axis, *BUILD_HELP["slice_axis"])
        self._add_help_row(form, "Min contour area", self.build_min_area, *BUILD_HELP["min_area"])
        self._add_help_row(form, "Resample points", self.resample_points, *BUILD_HELP["resample_points"])
        self._add_help_row(form, "Depth method", self.depth_method, *BUILD_HELP["depth_method"])
        self._add_help_row(form, "Volume format", self.volume_format, *BUILD_HELP["volume_format"])
        self._add_help_row(form, "Max Laplace voxels", self.max_laplace_voxels, *BUILD_HELP["max_laplace_voxels"])
        self._add_help_row(form, "Boundary dilation", self.boundary_dilation, *BUILD_HELP["boundary_dilation"])
        self._add_help_row(form, "QC interval", self.qc_every, *BUILD_HELP["qc_every"])
        self._add_help_row(form, "", self.build_keep_all, *BUILD_HELP["keep_all"])

        surface_button = self._make_button("Extract Surfaces", "primary")
        surface_button.clicked.connect(self.run_surface_build)
        depth_button = self._make_button("Compute Laminar Depth Volume", "secondary")
        depth_button.clicked.connect(self.run_depth_build)
        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(8)
        button_row.addWidget(surface_button)
        button_row.addWidget(depth_button)
        button_row.addStretch(1)

        layout.addWidget(settings)
        layout.addLayout(button_row)
        layout.addStretch(1)
        return tab

    def _make_demo_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self._make_hint("Demo is only a setup test with synthetic data."))

        settings = QGroupBox("Demo Test")

        self.demo_output = PathRow("Demo output folder", select_file=False)
        self.demo_resample_points = QSpinBox()
        self.demo_resample_points.setRange(8, 1000)
        self.demo_resample_points.setValue(48)
        self.demo_resample_points.setButtonSymbols(QSpinBox.NoButtons)
        self.demo_resample_points.setMinimumHeight(28)
        self.demo_depth_method = CleanComboBox()
        self.demo_depth_method.addItems(["auto", "laplace", "distance"])
        self.demo_depth_method.setMinimumHeight(28)

        form = QFormLayout(settings)
        self._tune_form(form)
        self._add_help_row(form, "Output folder", self.demo_output, *DEMO_HELP["output"])
        self._add_help_row(form, "Resample points", self.demo_resample_points, *DEMO_HELP["resample_points"])
        self._add_help_row(form, "Depth method", self.demo_depth_method, *DEMO_HELP["depth_method"])

        run_button = self._make_button("Run Demo Test", "primary")
        run_button.clicked.connect(self.run_demo)
        run_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        layout.addWidget(settings)
        layout.addWidget(run_button)
        layout.addStretch(1)
        return tab

    def _require_path(self, label: str, value: str) -> Path:
        if not value:
            raise ValueError(f"{label} is required.")
        return Path(value).expanduser()

    def append_log(self, text: str) -> None:
        if not text:
            return
        try:
            with self.log_file_path.open("a", encoding="utf-8") as handle:
                handle.write(text)
        except OSError:
            pass

    def start_task(self, label: str, fn: Callable[[], TaskResult]) -> None:
        if self.thread is not None:
            QMessageBox.warning(self, "Task running", "Please wait for the current task to finish.")
            return

        self.append_log(f"\n--- {label} started ---\n")
        self._set_status(f"Running: {label}", "running")
        self.thread = QThread()
        self.worker = Worker(fn)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.finished.connect(self.task_finished)
        self.worker.failed.connect(self.task_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.clear_thread)
        self.thread.start()

    def clear_thread(self) -> None:
        self.thread = None
        self.worker = None

    def force_stop_current_task(self) -> None:
        if self.thread is None:
            return
        self.append_log("\n--- task force-stopped by user ---\n")
        self.thread.requestInterruption()
        self.thread.terminate()
        if not self.thread.wait(3000):
            self.append_log("Task thread did not stop within 3 seconds; forcing process exit.\n")
            os._exit(0)
        self.thread = None
        self.worker = None
        self._set_status("Ready", "ready")

    def task_finished(self, result: TaskResult) -> None:
        self._set_status("Ready", "ready")
        self.append_log(f"\n{result.message}\n")
        self._show_task_result(result)

    def _show_task_result(self, result: TaskResult) -> None:
        summary = result.message.splitlines()[0] if result.message else result.title
        if result.output_dir:
            summary = f"{summary}\n\nOutput folder:\n{result.output_dir}"

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle(result.title)
        box.setText(summary)
        box.setInformativeText("Full file list was written to Log > View Current Log.")
        box.setDetailedText(result.message)
        box.exec_()

    def task_failed(self, trace: str) -> None:
        self._set_status("Failed", "failed")
        self.append_log("\n" + trace)
        QMessageBox.critical(self, "Run failed", trace.splitlines()[-1] if trace else "Unknown error")

    def run_prepare(self) -> None:
        def task() -> TaskResult:
            output_dir = self._require_path("Output folder", self.prepare_output.text())
            core = _core()
            contours = core.prepare_laminar_project(
                mask_path=self._require_path("Mask", self.prepare_mask.text()),
                output_dir=output_dir,
                slice_axis=self.prepare_slice_axis.currentText(),
                min_area=float(self.prepare_min_area.value()),
                largest_only=not self.prepare_keep_all.isChecked(),
                manual_every=int(self.prepare_manual_every.value()),
            )
            return TaskResult(
                title="Prepare finished",
                message=(
                    f"Prepared {len(contours)} slice contours.\n"
                    f"Edit: {output_dir / 'manual_landmarks_template.csv'}"
                ),
                output_dir=output_dir,
            )

        self.start_task("prepare", task)

    def run_surface_build(self) -> None:
        self.depth_method.setCurrentText("surfaces only")
        self.run_build()

    def run_build(self) -> None:
        def task() -> TaskResult:
            output_dir = self._require_path("Output folder", self.build_output.text())
            swc_glob = self.build_swc_glob.text().strip()
            swc_paths = sorted(glob.glob(swc_glob)) if swc_glob else []
            core = _core()
            outputs = core.run_laminar_boundary_pipeline(
                mask_path=self._require_path("Mask", self.build_mask.text()),
                manual_csv=self._require_path("Manual CSV", self.build_manual.text()),
                output_dir=output_dir,
                template_path=self.build_template.text() or None,
                cell_csv=self.build_cell_csv.text() or None,
                swc_paths=swc_paths,
                slice_axis=self.build_slice_axis.currentText(),
                min_area=float(self.build_min_area.value()),
                largest_only=not self.build_keep_all.isChecked(),
                resample_points=int(self.resample_points.value()),
                depth_method=self.depth_method.currentText(),
                max_laplace_voxels=int(self.max_laplace_voxels.value()),
                boundary_dilation=int(self.boundary_dilation.value()),
                qc_every=int(self.qc_every.value()),
                volume_format=self.volume_format.currentText(),
            )
            qc_summary = self._surface_qc_summary(output_dir)
            title = "Build needs QC review" if qc_summary.startswith("QC review needed") else "Build finished"
            lines = [title + "."]
            lines.extend(f"{key}: {value}" for key, value in outputs.items())
            lines.extend(("", qc_summary))
            return TaskResult(title, "\n".join(lines), output_dir=output_dir)

        self.start_task("build", task)

    def _build_boundaries_path(self) -> Path:
        text = self.build_boundaries.text().strip()
        if text:
            return Path(text).expanduser()
        output_dir = self._require_path("Output folder", self.build_output.text())
        return output_dir / "boundary_annotations.json"

    def _surface_qc_summary(self, output_dir: Path) -> str:
        review_slices = self._read_qc_review_slices(output_dir)
        if not review_slices:
            return "QC: no uncertain propagated slices."
        ranges = ", ".join(
            str(start) if start == end else f"{start}-{end}"
            for start, end in self._slice_ranges(review_slices)
        )
        targets = ", ".join(str(value) for value in self._review_targets_from_ranges(review_slices))

        manual_bad = []
        benign_flags = {"no_lateral_boundary"}
        summary_path = output_dir / "tables" / "boundary_summary.csv"
        try:
            with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    flags = [
                        flag.strip()
                        for flag in str(row.get("flags") or "").split(";")
                        if flag.strip()
                    ]
                    has_actionable_flags = any(flag not in benign_flags for flag in flags)
                    if row.get("source") == "manual" and has_actionable_flags:
                        manual_bad.append(str(int(float(row["slice_index"]))))
        except (OSError, ValueError, KeyError):
            manual_bad = []

        lines = [
            f"QC review needed: {len(review_slices)} uncertain propagated slices.",
            f"Uncertain ranges: {ranges}",
            f"Suggested re-annotation targets: {targets}",
        ]
        if manual_bad:
            lines.append(
                "Manual slices with QC flags: "
                + ", ".join(manual_bad)
                + ". Re-check these before trusting the surface."
            )
        return "\n".join(lines)

    def run_depth_build(self) -> None:
        def task() -> TaskResult:
            output_dir = self._require_path("Output folder", self.build_output.text())
            swc_glob = self.build_swc_glob.text().strip()
            swc_paths = sorted(glob.glob(swc_glob)) if swc_glob else []
            core = _core()
            depth_method = self.depth_method.currentText()
            if depth_method == "surfaces only":
                depth_method = "auto"
            outputs = core.run_laminar_depth_pipeline(
                mask_path=self._require_path("Mask", self.build_mask.text()),
                boundaries_json=self._build_boundaries_path(),
                output_dir=output_dir,
                template_path=self.build_template.text() or None,
                cell_csv=self.build_cell_csv.text() or None,
                swc_paths=swc_paths,
                slice_axis=self.build_slice_axis.currentText(),
                depth_method=depth_method,
                max_laplace_voxels=int(self.max_laplace_voxels.value()),
                boundary_dilation=int(self.boundary_dilation.value()),
                qc_every=int(self.qc_every.value()),
                volume_format=self.volume_format.currentText(),
            )
            lines = ["Depth volume finished."]
            lines.extend(f"{key}: {value}" for key, value in outputs.items())
            return TaskResult("Depth volume finished", "\n".join(lines), output_dir=output_dir)

        self.start_task("depth", task)

    def run_demo(self) -> None:
        def task() -> TaskResult:
            output_dir = self._require_path("Output folder", self.demo_output.text())
            demo_input_dir = output_dir / "demo_input"
            core = _core()
            mask_path, manual_csv = core.write_demo_project(demo_input_dir)
            build_dir = output_dir / "demo_build"
            outputs = core.run_laminar_boundary_pipeline(
                mask_path=mask_path,
                manual_csv=manual_csv,
                output_dir=build_dir,
                slice_axis=0,
                min_area=20.0,
                resample_points=int(self.demo_resample_points.value()),
                depth_method=self.demo_depth_method.currentText(),
                qc_every=4,
            )
            lines = ["Demo finished.", f"demo_mask: {mask_path}", f"demo_manual_csv: {manual_csv}"]
            lines.extend(f"{key}: {value}" for key, value in outputs.items())
            return TaskResult("Demo finished", "\n".join(lines), output_dir=build_dir)

        self.start_task("demo", task)

    def closeEvent(self, event) -> None:
        if self.thread is not None:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Task running")
            box.setText("A build or extraction task is still running.")
            box.setInformativeText("Stop the task and close the app?")
            keep_button = box.addButton("Keep Running", QMessageBox.RejectRole)
            stop_button = box.addButton("Stop Task And Close", QMessageBox.DestructiveRole)
            box.setDefaultButton(keep_button)
            box.exec_()
            if box.clickedButton() is not stop_button:
                event.ignore()
                return
            self.force_stop_current_task()

        self._hide_help_popup()
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self.annotation_mask_data = None
        self.annotation_template_data = None
        self.annotation_contours_by_slice.clear()
        self.annotation_landmarks_by_slice.clear()
        self.annotation_rows.clear()
        self.annotation_path_choices_by_slice.clear()
        self.annotation_skipped_slices.clear()
        self.annotation_reference_contours = []
        self.annotation_mask_path = None
        self.annotation_mask_is_temporary = False
        self._cleanup_temporary_mask()
        gc.collect()
        super().closeEvent(event)


def main() -> int:
    cleanup_orphan_temporary_masks()
    app = QApplication([])
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setApplicationName("Laminar Boundary Builder")
    window = LaminarBoundaryWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
