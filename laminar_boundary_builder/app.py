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
import re
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
    QLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
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

try:
    from .gl_view import ShellGLCanvas
except Exception:
    ShellGLCanvas = None

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
QFrame#hint {
    background: #f4f8f7;
    border: 1px solid #d4e2de;
    border-radius: 8px;
}
QLabel#hintText {
    color: #35534e;
    font-weight: 650;
}
QLabel#sectionNote {
    color: #5c706c;
    font-size: 12px;
    line-height: 130%;
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
QLabel#readinessText {
    background: #f8fbfa;
    border: 1px solid #ccdcd8;
    border-radius: 7px;
    color: #304946;
    padding: 7px 9px;
}
QLabel#readinessText[state="ready"] {
    background: #edf8f1;
    border-color: #a7cdb7;
    color: #24563c;
}
QLabel#readinessText[state="warning"] {
    background: #fff8e5;
    border-color: #dfc77b;
    color: #6c5219;
}
QLabel#readinessText[state="missing"] {
    background: #fff6f3;
    border-color: #e2b8af;
    color: #7c342b;
}
QLabel#buildReadyText {
    background: #fff8e5;
    border: 1px solid #dfc77b;
    border-radius: 7px;
    color: #6c5219;
    padding: 7px 9px;
}
QLabel#buildReadyText[state="ready"] {
    background: #edf8f1;
    border-color: #a7cdb7;
    color: #24563c;
}
QLabel#buildReadyText[state="warning"] {
    background: #fff8e5;
    border-color: #dfc77b;
    color: #6c5219;
}
QLabel#buildReadyText[state="missing"] {
    background: #fff6f3;
    border-color: #e2b8af;
    color: #7c342b;
}
QLabel#buildResultText {
    background: #f8fbfa;
    border: 1px solid #ccdcd8;
    border-radius: 7px;
    color: #304946;
    padding: 8px 10px;
}
QLabel#buildResultText[state="running"] {
    background: #eef4ff;
    border-color: #b9cbea;
    color: #294e82;
}
QLabel#buildResultText[state="ready"] {
    background: #edf8f1;
    border-color: #a7cdb7;
    color: #24563c;
}
QLabel#buildResultText[state="failed"] {
    background: #fff6f3;
    border-color: #e2b8af;
    color: #7c342b;
}
QFrame#buildProgressCard {
    background: #f7fbfa;
    border: 1px solid #ccdcd8;
    border-radius: 7px;
    padding: 8px 10px;
}
QFrame#buildProgressCard[state="running"] {
    background: #eef7f4;
    border-color: #adcfc6;
}
QFrame#buildProgressCard[state="ready"] {
    background: #edf8f1;
    border-color: #a7cdb7;
}
QFrame#buildProgressCard[state="failed"] {
    background: #fff6f3;
    border-color: #e2b8af;
}
QLabel#buildProgressStage {
    color: #213d38;
    font-weight: 750;
}
QLabel#buildProgressPercent {
    color: #2f6e62;
    font-weight: 800;
}
QLabel#buildProgressDetail {
    color: #54706b;
    font-size: 12px;
}
QFrame#surfaceQueuePanel {
    background: transparent;
    border: 0;
}
QPushButton#surfaceQueueItem {
    background: #f7fbfa;
    border: 1px solid #cddfda;
    border-radius: 7px;
    color: #24413c;
    font-weight: 700;
    padding: 7px 9px;
    text-align: left;
}
QPushButton#surfaceQueueItem[active="true"] {
    background: #e5f5ef;
    border-color: #7fb8a7;
    color: #123e35;
}
QPushButton#surfaceQueueItem[seeded="false"] {
    border-color: #dfc77b;
}
QLabel#surfaceQueueEmpty {
    background: #f8fbfa;
    border: 1px dashed #cddfda;
    border-radius: 7px;
    color: #60736f;
    padding: 7px 9px;
}
QProgressBar#buildProgressBar {
    background: #dbe8e5;
    border: 0;
    border-radius: 4px;
    min-height: 8px;
    max-height: 8px;
}
QProgressBar#buildProgressBar::chunk {
    background: #2f6e62;
    border-radius: 4px;
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
QPushButton:disabled, QPushButton[role="primary"]:disabled, QPushButton[role="secondary"]:disabled {
    background: #e5ebe9;
    border-color: #cbd6d3;
    color: #879894;
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
QScrollArea#buildScroll {
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
        "Purpose: choose where shell-cut annotations and build inputs will be saved.\n"
        "Effect: Save Shell Cut And Review Build writes shell_cut_annotations.json plus a derived build CSV.\n"
        "Recommended: choose a project-specific folder so the annotation files are easy to find later.",
    ),
    "previous_csv": (
        "Previous shell-cut JSON",
        "Purpose: reload a saved shell_cut_annotations.json into the annotation workspace.\n"
        "Effect: accepted shell-cut slices and points are restored so you can edit only the bad slices.\n"
        "Recommended: load the same mask first, then load the previous JSON and revise flagged slices.",
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
        "Purpose: choose which detected contour on the current slice receives your shell-cut points.\n"
        "Effect: landmarks are saved against this contour ID.\n"
        "Recommended: use contour 0 when only one contour is present. Switch if the highlighted contour is not your target boundary.",
    ),
    "pick_mode": (
        "Shell cut points",
        "Purpose: clicks edit shell-cut boundary points only.\n"
        "Effect: Shell Cut Boundary stores outer_cut_A/B and inner_cut_A/B in shell_cut_annotations.json.\n"
        "Recommended: annotate sparse smooth regions and add more slices near sudden shape changes.",
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
        "Derived build CSV",
        "Purpose: an internal CSV derived from shell-cut annotations.\n"
        "Effect: the current build pipeline uses these derived rows to prepare propagated boundary curves.\n"
        "Recommended: use the file created by Save Shell Cut And Review Build.",
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
        "Purpose: match the slice axis used when shell-cut annotations were created.\n"
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
    "surface_method": (
        "Surface method",
        "Purpose: choose how the surface mesh is reconstructed from the mask and landmarks.\n"
        "Effect: Shell cut builds the full mask shell first, then cuts outer/inner/lateral patches from it. Arc graph is an experimental local-arc stitcher. Contour shell and Fast loft are legacy contour stitchers. Mask constrained is the older voxel-shell labeler.\n"
        "Recommended: Shell cut for topology-sensitive review; Contour shell for comparing old results.",
    ),
    "shell_backend": (
        "Shell backend",
        "Purpose: choose the shell geometry used by Shell cut.\n"
        "Effect: Voxel is the blocky debug baseline. Marching cubes builds a smoother shell before snapping cut curves and flood filling patches.\n"
        "Recommended: Voxel for debugging; Marching cubes for smoother review once the cut curves look correct.",
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


@dataclass
class AnnotationLoadData:
    mask_data: object
    mask_path: Path
    template_data: object
    temporary: bool
    slice_axis_int: int
    slice_counts: object
    region_slices: List[int]
    shell_mesh: object
    warnings: List[str]


@dataclass
class AnnotationPreviewCacheData:
    request_id: int
    reference_contours: list
    contours_by_slice: Dict[int, list]


@dataclass
class ReviewRepairRequest:
    mask_path: Path
    manual_csv: Path
    build_output_dir: Path
    queue_slices: List[int]


@dataclass
class PreviousProjectPaths:
    project_dir: Path
    mask_path: Path
    manual_csv: Path
    build_output_dir: Path


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
    progress = pyqtSignal(object)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn: Callable, accepts_progress: bool = False, task_label: str = ""):
        super().__init__()
        self.fn = fn
        self.accepts_progress = bool(accepts_progress)
        self.task_label = str(task_label)

    def _emit_progress(self, value: int, stage: str, detail: str = "") -> None:
        self.progress.emit(
            {
                "label": self.task_label,
                "value": int(value),
                "stage": str(stage),
                "detail": str(detail or ""),
            }
        )

    def run(self) -> None:
        stdout = StreamBuffer(self.log.emit)
        stderr = StreamBuffer(self.log.emit)
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                if self.accepts_progress:
                    result = self.fn(self._emit_progress)
                else:
                    result = self.fn()
            self.finished.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class BuildProgressPanel(QFrame):
    def __init__(self, idle_detail: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.idle_detail = idle_detail
        self.setObjectName("buildProgressCard")

        self.stage_label = QLabel("Idle")
        self.stage_label.setObjectName("buildProgressStage")
        self.stage_label.setWordWrap(True)
        self.percent_label = QLabel("0%")
        self.percent_label.setObjectName("buildProgressPercent")
        self.percent_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("buildProgressBar")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(False)
        self.detail_label = QLabel(idle_detail)
        self.detail_label.setObjectName("buildProgressDetail")
        self.detail_label.setWordWrap(True)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        header_layout.addWidget(self.stage_label, 1)
        header_layout.addWidget(self.percent_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(header)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.detail_label)
        self.reset()

    def _set_state(self, state: str) -> None:
        self.setProperty("state", state)
        self.style().unpolish(self)
        self.style().polish(self)

    def reset(self) -> None:
        self._set_state("idle")
        self.stage_label.setText("No build running")
        self.percent_label.setText("0%")
        self.progress_bar.setValue(0)
        self.detail_label.setText(self.idle_detail)

    def start(self, stage: str, detail: str) -> None:
        self.update_progress(0, stage, detail, state="running")

    def update_progress(self, value: int, stage: str, detail: str = "", state: str = "running") -> None:
        value = max(0, min(100, int(value)))
        self._set_state(state)
        self.stage_label.setText(stage or "Working")
        self.percent_label.setText(f"{value}%")
        self.progress_bar.setValue(value)
        self.detail_label.setText(detail or "Working...")

    def finish(self, stage: str, detail: str) -> None:
        self.update_progress(100, stage, detail, state="ready")

    def fail(self, detail: str) -> None:
        self.update_progress(self.progress_bar.value(), "Build failed", detail, state="failed")


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
    SHELL_CUT_ORDER = (
        "outer_cut_A",
        "outer_cut_B",
        "inner_cut_A",
        "inner_cut_B",
    )

    COLORS = {
        "outer_start": QColor("#1f77b4"),
        "outer_end": QColor("#1f77b4"),
        "inner_start": QColor("#d62728"),
        "inner_end": QColor("#d62728"),
        "outer_cut_A": QColor("#e64b3c"),
        "outer_cut_B": QColor("#e64b3c"),
        "inner_cut_A": QColor("#3973e6"),
        "inner_cut_B": QColor("#3973e6"),
    }

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(560, 420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.image = None
        self._qimage_cache: Optional[QImage] = None
        self.contours = []
        self.selected_contour_index = 0
        self.landmarks: Dict[str, int] = {}
        self.mode = "outer_cut_A"
        self.picking_mode_kind = "shell_cut"
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
        self.show_shortcuts = False
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
        self._qimage_cache = None
        self.contours = contours
        self.landmarks = dict(landmarks or {})
        self.selected_contour_index = min(max(0, selected_contour_index), max(0, len(contours) - 1))
        self.slice_axis = slice_axis
        self.update()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.update()

    def set_picking_mode_kind(self, mode: str) -> None:
        self.picking_mode_kind = "shell_cut" if str(mode).lower() == "shell_cut" else "boundary"
        self.update()

    def active_landmark_order(self) -> tuple[str, ...]:
        if self.picking_mode_kind == "shell_cut":
            return self.SHELL_CUT_ORDER
        return self.LANDMARK_ORDER

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
        if self._qimage_cache is not None:
            return self._qimage_cache
        np = _numpy()
        image = np.asarray(self.image)
        if image.dtype == np.uint8:
            scaled = np.ascontiguousarray(image)
        else:
            image = image.astype(float, copy=False)
            finite = image[np.isfinite(image)]
            if finite.size == 0:
                scaled = np.zeros(image.shape, dtype=np.uint8)
            else:
                lo, hi = np.percentile(finite, [1, 99])
                if hi <= lo:
                    hi = lo + 1.0
                scaled = np.clip((image - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
            scaled = np.ascontiguousarray(scaled)
        qimage = QImage(
            scaled.data,
            scaled.shape[1],
            scaled.shape[0],
            scaled.strides[0],
            QImage.Format_Grayscale8,
        )
        self._qimage_cache = qimage.copy()
        return self._qimage_cache

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

    def toggle_shortcut_help(self) -> None:
        self.show_shortcuts = not self.show_shortcuts
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
        if self.picking_mode_kind == "shell_cut":
            if name.startswith("inner_cut"):
                indices, endpoints = self._arc_indices_for_landmarks(
                    "outer_cut_A",
                    "outer_cut_B",
                    self.outer_path_choice,
                )
            elif name.startswith("outer_cut"):
                indices, endpoints = self._arc_indices_for_landmarks(
                    "inner_cut_A",
                    "inner_cut_B",
                    self.inner_path_choice,
                )
            else:
                return set()
        elif name.startswith("inner"):
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
        if self.picking_mode_kind == "shell_cut":
            return None
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

    def _shell_cut_preview_arcs(self, points) -> Dict[str, object]:
        if not all(name in self.landmarks for name in self.SHELL_CUT_ORDER):
            arcs = {}
            outer = self._preview_arc(points, "outer_cut_A", "outer_cut_B", self.outer_path_choice)
            inner = self._preview_arc(points, "inner_cut_A", "inner_cut_B", self.inner_path_choice)
            if outer is not None:
                arcs["outer"] = outer
            if inner is not None:
                arcs["inner"] = inner
            return arcs

        core = _core()
        try:
            (
                outer_arc,
                _outer_indices,
                _outer_direction,
                _outer_score,
                inner_arc,
                _inner_indices,
                _inner_direction,
                _inner_score,
            ) = core._choose_outer_inner_arcs(
                points,
                self.landmarks["outer_cut_A"],
                self.landmarks["outer_cut_B"],
                self.landmarks["inner_cut_A"],
                self.landmarks["inner_cut_B"],
                outer_choice=self.outer_path_choice,
                inner_choice=self.inner_path_choice,
            )
            return {"outer": outer_arc, "inner": inner_arc}
        except Exception:
            return {}

    def shell_cut_overlap_excess(self) -> int:
        contour = self.selected_contour()
        if contour is None or not all(name in self.landmarks for name in self.SHELL_CUT_ORDER):
            return 0
        core = _core()
        points = core._normalize_contour(contour.points)
        if len(points) < 2:
            return 0
        try:
            (
                _outer_arc,
                outer_indices,
                _outer_direction,
                _outer_score,
                _inner_arc,
                inner_indices,
                _inner_direction,
                _inner_score,
            ) = core._choose_outer_inner_arcs(
                points,
                self.landmarks["outer_cut_A"],
                self.landmarks["outer_cut_B"],
                self.landmarks["inner_cut_A"],
                self.landmarks["inner_cut_B"],
                outer_choice=self.outer_path_choice,
                inner_choice=self.inner_path_choice,
            )
            allowed_overlap = core._shared_endpoint_count(
                self.landmarks["outer_cut_A"],
                self.landmarks["outer_cut_B"],
                self.landmarks["inner_cut_A"],
                self.landmarks["inner_cut_B"],
                len(points),
            )
            return int(core._arc_overlap_excess(
                len(points),
                outer_indices,
                inner_indices,
                allowed_overlap,
            ))
        except Exception:
            return 0

    def _draw_plane_polyline(
        self,
        painter: QPainter,
        plane_points,
        color: QColor,
        width: float,
        style=Qt.SolidLine,
    ) -> None:
        if plane_points is None or len(plane_points) < 2:
            return
        painter.setPen(QPen(color, width, style))
        for p0, p1 in zip(plane_points[:-1], plane_points[1:]):
            painter.drawLine(self.plane_to_screen(p0), self.plane_to_screen(p1))

    def _draw_boundary_preview(self, painter: QPainter, points) -> Optional[object]:
        if self.picking_mode_kind == "shell_cut":
            return None
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

    def _draw_shell_cut_preview(self, painter: QPainter, points) -> None:
        core = _core()
        arcs = self._shell_cut_preview_arcs(points)
        for arc_name, color in (
            ("outer", QColor("#e64b3c")),
            ("inner", QColor("#3973e6")),
        ):
            arc = arcs.get(arc_name)
            if arc is None:
                continue
            arc_plane = core._volume_to_plane_points(arc, self.slice_axis)
            self._draw_plane_polyline(
                painter,
                arc_plane,
                color,
                3.5,
                style=Qt.DashLine,
            )

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
            if self.picking_mode_kind == "shell_cut":
                self._draw_shell_cut_preview(painter, points)
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

        label = f"Next point: {self.mode.replace('_', ' ')}"
        if not self.picking_enabled:
            label = "Load mask to start point picking"
        elif all(name in self.landmarks for name in self.active_landmark_order()):
            if self.picking_mode_kind == "shell_cut":
                label = "All shell cut endpoints set. Enter = accept + next"
            else:
                label = "All four points set. Enter = accept + next"
        label_rect = QRectF(10, 10, max(260, len(label) * 8), 30)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(18, 24, 23, 205))
        painter.drawRoundedRect(label_rect, 8, 8)
        painter.setPen(QColor("#e8eeee"))
        painter.drawText(label_rect.adjusted(10, 0, 0, 0), Qt.AlignVCenter, label)

        if self.picking_enabled and self.show_shortcuts:
            if self.picking_mode_kind == "shell_cut":
                keys = "Drag pan   S outer only   A inner only   K skip   H shortcuts   N suggest   Wheel/+/- zoom   0 reset   X undo"
            else:
                keys = "Drag pan   S outer only   A inner only   H shortcuts   N suggest   O/I flip   Wheel/+/- zoom   0 reset   X undo"
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
                progress_top = 82 if self.picking_enabled and self.show_shortcuts else 46
                progress_rect = QRectF(10, progress_top, rect_width, rect_height)
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

        active_order = self.active_landmark_order()
        if self.mode in active_order and self.mode not in self.landmarks:
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

        if self.mode not in active_order:
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
    annotation_changed = pyqtSignal()
    build_ready_changed = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(330, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.boundaries = []
        self.current_boundary = None
        self.reference_contours = []
        self.shell_mesh = None
        self.closed_curves: List[Dict[str, object]] = []
        self.active_curve_vertices: List[int] = []
        self.selected_patches: List[Dict[str, object]] = []
        self.surface_name = "surface"
        self.surface_names: List[str] = []
        self.active_surface_index = 0
        self.annotation_mode = "curve"
        self.hover_face: Optional[int] = None
        self.slice_axis = 0
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
        self._projection_cache_key = None
        self._projection_cache = None
        self._draw_face_ids = None
        self._draw_vertex_ids = None

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
            QColor(230, 75, 60, 90),
        )
        self._draw_surface_connectors(
            painter,
            boundaries,
            "inner",
            transform,
            QColor(57, 115, 230, 90),
        )

        current_slice = self.current_boundary.slice_index if self.current_boundary is not None else None
        for boundary in boundaries:
            is_current = boundary.slice_index == current_slice
            style = Qt.DashLine if is_current else Qt.SolidLine
            self._draw_volume_polyline(
                painter,
                boundary.outer_arc,
                transform,
                QColor("#ff756a") if is_current else QColor("#e64b3c"),
                3.2 if is_current else 2.0,
                style,
            )
            self._draw_volume_polyline(
                painter,
                boundary.inner_arc,
                transform,
                QColor("#7ba2ff") if is_current else QColor("#3973e6"),
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

    def set_shell_mesh(self, shell_mesh) -> None:
        self.shell_mesh = shell_mesh
        self.closed_curves = []
        self.active_curve_vertices = []
        self.selected_patches = []
        self.surface_names = []
        self.active_surface_index = 0
        self.surface_name = "surface"
        self.hover_face = None
        self.annotation_mode = "curve"
        self.preview_zoom = 1.15
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._projection_cache_key = None
        self._projection_cache = None
        self._draw_face_ids = None
        self._draw_vertex_ids = None
        self.message = "3D: click shell points to draw a cut curve" if shell_mesh is not None else "3D shell is not available"
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
        self.message = "3D: click shell points to draw a cut curve"
        self.update()

    def set_patch_mode(self) -> None:
        if not self.closed_curves:
            self.message = "Close at least one cut curve before selecting a surface"
            self.update()
            return
        self.annotation_mode = "patch"
        self.message = "3D: hover a patch, then click it to save under the current name"
        self.update()

    def clear_3d_annotations(self) -> None:
        self.closed_curves = []
        self.active_curve_vertices = []
        self.selected_patches = []
        self.surface_names = []
        self.active_surface_index = 0
        self.surface_name = "surface"
        self.hover_face = None
        self.annotation_mode = "curve"
        self.message = "3D: click shell points to draw a cut curve"
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
        return (
            len(self.closed_curves),
            len(self.active_curve_vertices),
            len(self.selected_patches),
        )

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
        np = _numpy()
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

    def _emit_3d_state(self) -> None:
        self.annotation_changed.emit()
        self.build_ready_changed.emit(self.can_build_3d_surfaces())

    def _3d_vertices(self):
        if self.shell_mesh is None:
            return _numpy().empty((0, 3), dtype=float)
        return _numpy().asarray(self.shell_mesh.vertices, dtype=float)

    def _3d_rotated_points(self, points, center):
        np = _numpy()
        coords = np.asarray(points, dtype=float) - center
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
        return np.column_stack((xz, yz, depth))

    def _3d_projection(self):
        np = _numpy()
        vertices = self._3d_vertices()
        if len(vertices) == 0:
            return None
        face_count = len(getattr(self.shell_mesh, "faces", [])) if self.shell_mesh is not None else 0
        key = (
            id(self.shell_mesh),
            len(vertices),
            face_count,
            self.width(),
            self.height(),
            round(float(self.rotation_yaw), 5),
            round(float(self.rotation_pitch), 5),
            round(float(self.preview_zoom), 5),
            round(float(self.pan_x), 3),
            round(float(self.pan_y), 3),
        )
        if self._projection_cache_key == key and self._projection_cache is not None:
            return self._projection_cache
        center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
        radius = float(np.linalg.norm(vertices - center.reshape(1, 3), axis=1).max())
        radius = max(radius, 1.0)
        margin = 34.0
        scale = self.preview_zoom * min(
            max(1.0, self.width() - margin * 2.0),
            max(1.0, self.height() - margin * 2.0),
        ) / (2.0 * radius)
        offset = np.asarray([self.width() * 0.5 + self.pan_x, self.height() * 0.5 + self.pan_y], dtype=float)
        rotated = self._3d_rotated_points(vertices, center)
        screen = offset.reshape(1, 2) + rotated[:, :2] * scale
        self._projection_cache_key = key
        self._projection_cache = (screen, rotated[:, 2])
        return self._projection_cache

    def _sample_face_ids_for_display(self, face_depth, max_faces: int = 12000):
        np = _numpy()
        if len(face_depth) == 0:
            return np.empty((0,), dtype=int)
        draw_order = np.argsort(face_depth)
        if len(draw_order) > max_faces:
            step = int(math.ceil(len(draw_order) / max_faces))
            draw_order = draw_order[::step]
        return np.asarray(draw_order, dtype=int)

    def _nearest_vertex_id(self, pos, max_distance: float = 16.0) -> Optional[int]:
        np = _numpy()
        projection = self._3d_projection()
        if projection is None:
            return None
        screen, _depth = projection
        click = np.asarray([pos.x(), pos.y()], dtype=float)
        candidates = self._draw_vertex_ids
        if candidates is not None and len(candidates):
            candidates = np.asarray(candidates, dtype=int)
        else:
            candidates = np.arange(len(screen), dtype=int)
        distances = np.linalg.norm(screen[candidates] - click.reshape(1, 2), axis=1)
        index = int(np.argmin(distances))
        return int(candidates[index]) if float(distances[index]) <= max_distance else None

    def _nearest_active_vertex_index(self, pos, max_distance: float = 22.0) -> Optional[int]:
        if not self.active_curve_vertices:
            return None
        np = _numpy()
        projection = self._3d_projection()
        if projection is None:
            return None
        screen, _depth = projection
        ids = np.asarray(self.active_curve_vertices, dtype=int)
        click = np.asarray([pos.x(), pos.y()], dtype=float)
        distances = np.linalg.norm(screen[ids] - click.reshape(1, 2), axis=1)
        index = int(np.argmin(distances))
        return index if float(distances[index]) <= max_distance else None

    def _nearest_face_id(self, pos, max_distance: float = 24.0) -> Optional[int]:
        np = _numpy()
        if self.shell_mesh is None:
            return None
        projection = self._3d_projection()
        if projection is None:
            return None
        screen, _depth = projection
        faces = np.asarray(self.shell_mesh.faces, dtype=int)
        if len(faces) == 0:
            return None
        candidates = self._draw_face_ids
        if candidates is not None and len(candidates):
            candidates = np.asarray(candidates, dtype=int)
        else:
            face_depth = _depth[faces].mean(axis=1)
            candidates = self._sample_face_ids_for_display(face_depth)
        if len(candidates) == 0:
            return None
        centers = screen[faces[candidates]].mean(axis=1)
        click = np.asarray([pos.x(), pos.y()], dtype=float)
        distances = np.linalg.norm(centers - click.reshape(1, 2), axis=1)
        index = int(np.argmin(distances))
        return int(candidates[index]) if float(distances[index]) <= max_distance else None

    def _nearest_face_vertex_id(self, face_id: int, pos) -> Optional[int]:
        np = _numpy()
        if self.shell_mesh is None:
            return None
        projection = self._3d_projection()
        if projection is None:
            return None
        screen, _depth = projection
        faces = np.asarray(self.shell_mesh.faces, dtype=int)
        if face_id < 0 or face_id >= len(faces):
            return None
        face_vertices = np.asarray(faces[int(face_id)], dtype=int)
        click = np.asarray([pos.x(), pos.y()], dtype=float)
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
        if self.shell_mesh is None:
            return
        if any(int(patch.get("face_id", -1)) == int(face_id) for patch in self.selected_patches):
            self.message = "That patch seed is already selected"
            return
        self._sync_surface_queue()
        if not self.surface_names:
            self.message = "Close a curve before selecting a surface seed"
            return
        np = _numpy()
        face = np.asarray(self.shell_mesh.faces[int(face_id)], dtype=int)
        seed_point = np.asarray(self.shell_mesh.vertices[face], dtype=float).mean(axis=0)
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
            face_id = self._nearest_face_id(pos, max_distance=30.0)
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
        vertex_id = self._nearest_vertex_id(pos)
        if vertex_id is None:
            face_id = self._nearest_face_id(pos, max_distance=42.0)
            if face_id is not None:
                vertex_id = self._nearest_face_vertex_id(face_id, pos)
        if vertex_id is None:
            self.message = "Click closer to the visible 3D shell"
            self.update()
            return
        self.annotation_mode = "curve"
        self.active_curve_vertices.append(int(vertex_id))
        self.message = "Click more points, or click an active point to close the curve"
        self._emit_3d_state()
        self.update()

    def _draw_3d_curve(self, painter: QPainter, vertex_ids: List[int], screen, color: QColor, closed: bool) -> None:
        if not vertex_ids:
            return
        points = [QPointF(float(screen[int(value), 0]), float(screen[int(value), 1])) for value in vertex_ids]
        pen = QPen(color, 3.0 if closed else 2.4)
        pen.setStyle(Qt.SolidLine if closed else Qt.DashLine)
        painter.setPen(pen)
        for left, right in zip(points[:-1], points[1:]):
            painter.drawLine(left, right)
        painter.setPen(QPen(QColor("#101817"), 1.0))
        painter.setBrush(color)
        for point in points:
            painter.drawEllipse(point, 4.5, 4.5)

    def paintEvent(self, event) -> None:
        if self.shell_mesh is None:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor("#101817"))
            painter.setPen(QColor("#d4dddd"))
            painter.drawText(self.rect(), Qt.AlignCenter, self.message)
            return
        np = _numpy()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#101817"))
        projection = self._3d_projection()
        if projection is None:
            painter.setPen(QColor("#d4dddd"))
            painter.drawText(self.rect(), Qt.AlignCenter, self.message)
            return
        screen, depth = projection
        faces = np.asarray(self.shell_mesh.faces, dtype=int)
        face_depth = depth[faces].mean(axis=1)
        draw_order = self._sample_face_ids_for_display(face_depth)
        self._draw_face_ids = draw_order
        self._draw_vertex_ids = np.unique(faces[draw_order].reshape(-1)) if len(draw_order) else None
        selected_faces = {int(patch["face_id"]) for patch in self.selected_patches}
        if len(face_depth):
            depth_min = float(face_depth.min())
            depth_span = max(1.0, float(face_depth.max() - depth_min))
        else:
            depth_min = 0.0
            depth_span = 1.0
        for face_id in draw_order:
            face_id = int(face_id)
            face = faces[face_id]
            polygon = QPolygonF([QPointF(float(screen[i, 0]), float(screen[i, 1])) for i in face])
            if face_id == self.hover_face:
                painter.setPen(QPen(QColor("#ffffff"), 1.2))
                painter.setBrush(QColor(85, 180, 145, 165))
            elif face_id in selected_faces:
                painter.setPen(QPen(QColor("#f5c84c"), 1.1))
                painter.setBrush(QColor(245, 200, 76, 145))
            else:
                normalized_depth = (float(face_depth[face_id]) - depth_min) / depth_span
                shade = int(92 + normalized_depth * 110)
                painter.setPen(QPen(QColor(190, 230, 220, 45), 0.5))
                painter.setBrush(QColor(shade, min(255, shade + 34), min(255, shade + 25), 138))
            painter.drawPolygon(polygon)

        if self._draw_vertex_ids is not None and len(self._draw_vertex_ids):
            vertex_ids = np.asarray(self._draw_vertex_ids, dtype=int)
            max_points = 7000
            if len(vertex_ids) > max_points:
                vertex_ids = vertex_ids[:: int(math.ceil(len(vertex_ids) / max_points))]
            painter.setPen(QPen(QColor(232, 255, 247, 150), 1.8))
            for vertex_id in vertex_ids:
                point = screen[int(vertex_id)]
                painter.drawPoint(QPointF(float(point[0]), float(point[1])))

        for index, curve in enumerate(self.closed_curves):
            color = QColor("#f06a5a") if index % 2 == 0 else QColor("#63a0ff")
            self._draw_3d_curve(
                painter,
                [int(value) for value in curve.get("vertices", [])],
                screen,
                color,
                closed=True,
            )
        self._draw_3d_curve(
            painter,
            [int(value) for value in self.active_curve_vertices],
            screen,
            QColor("#e8f06a"),
            closed=False,
        )
        curve_count, point_count, patch_count = self.annotation_counts()
        mode_text = "Draw curve" if self.annotation_mode == "curve" else "Select surface"
        self._draw_message(
            painter,
            f"{mode_text}: {curve_count} curves · {point_count} points · {patch_count} seeds",
        )

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
            self.hover_face = self._nearest_face_id(event.pos(), max_distance=26.0)
            self.update()
            return
        dx = event.pos().x() - self._drag_pos.x()
        dy = event.pos().y() - self._drag_pos.y()
        self._drag_pos = QPointF(event.pos())
        if self._press_pos is not None:
            total_dx = event.pos().x() - self._press_pos.x()
            total_dy = event.pos().y() - self._press_pos.y()
            if total_dx * total_dx + total_dy * total_dy > 16.0:
                self._drag_moved = True
                if self._drag_mode == "pick":
                    self._drag_mode = "rotate"
        if self._drag_mode == "pan":
            self.pan_x += dx
            self.pan_y += dy
        elif self._drag_mode == "rotate":
            self.rotation_yaw = self._wrap_rotation_angle(self.rotation_yaw + dx * 0.01)
            self.rotation_pitch = self._wrap_rotation_angle(self.rotation_pitch + dy * 0.01)
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self.shell_mesh is None:
            return
        if event.button() not in (Qt.LeftButton, Qt.RightButton):
            return
        if self._drag_mode == "pick" and not self._drag_moved:
            self._handle_shell_click(event.pos())
        self._drag_pos = None
        self._press_pos = None
        self.unsetCursor()
        event.accept()


class LaminarBoundaryWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.thread: Optional[QThread] = None
        self.worker: Optional[Worker] = None
        self.current_task_label: Optional[str] = None
        self.annotation_preview_thread: Optional[QThread] = None
        self.annotation_preview_worker: Optional[Worker] = None
        self.annotation_preview_request_id = 0
        self.pending_review_request: Optional[ReviewRepairRequest] = None
        self.pending_previous_csv_load: Optional[Tuple[Path, Path, Path]] = None
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
        shell_cut_autosave_name = self.log_file_path.stem.replace(
            LOG_FILE_PREFIX.rstrip("_"),
            "shell_cut_annotations_autosave",
            1,
        )
        self.shell_cut_autosave_path = self.log_file_path.with_name(f"{shell_cut_autosave_name}.json")
        self._make_menu_bar()
        self.append_log(
            "Laminar Boundary Builder log started "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"log_file: {self.log_file_path}\n\n"
        )

        self.tabs = QTabWidget()
        self.tabs.addTab(self._make_annotate_tab(), "Annotate")
        self.tabs.addTab(self._make_build_tab(), "Build")

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        header = self._make_header()
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
        self.open_previous_project_button = self._make_button("Open Previous Project", "secondary")
        self.open_previous_project_button.setToolTip(
            "Open a saved laminar_boundary_builder_output folder and restore its mask, latest CSV, and build."
        )
        self.open_previous_project_button.clicked.connect(self.open_previous_project)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(0)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(self.open_previous_project_button)
        title_row.addSpacing(8)
        title_row.addWidget(self.status_label)

        layout.addLayout(title_row)
        return box

    def _set_status(self, text: str, state: str) -> None:
        self.status_label.setText(text)
        self.status_label.setProperty("state", state)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _set_label_state(self, label: QLabel, state: str) -> None:
        label.setProperty("state", state)
        label.style().unpolish(label)
        label.style().polish(label)

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
        tools_menu = self.menuBar().addMenu("Tools")
        open_project_action = tools_menu.addAction("Open Previous Project")
        open_project_action.triggered.connect(lambda _checked=False: self.open_previous_project())
        demo_action = tools_menu.addAction("Run Demo Test")
        demo_action.triggered.connect(lambda _checked=False: self.show_demo_dialog())

        log_menu = self.menuBar().addMenu("Log")
        view_action = log_menu.addAction("View Current Log")
        view_action.triggered.connect(lambda _checked=False: self.show_log_dialog())
        folder_action = log_menu.addAction("Show Log Folder")
        folder_action.triggered.connect(lambda _checked=False: self.show_log_folder())
        clear_action = log_menu.addAction("Clear Current Log")
        clear_action.triggered.connect(lambda _checked=False: self.clear_current_log())

    def show_demo_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Demo Test")
        dialog.resize(900, 360)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(self._make_demo_tab(), 1)

        close_row = QHBoxLayout()
        close_row.setContentsMargins(0, 0, 0, 0)
        close_row.addStretch(1)
        close_button = self._make_button("Close", "secondary")
        close_button.clicked.connect(dialog.accept)
        close_row.addWidget(close_button)
        layout.addLayout(close_row)

        dialog.exec_()

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

    def _default_previous_project_dir(self) -> Path:
        for text in (
            self.build_output.text().strip() if hasattr(self, "build_output") else "",
            self.annotate_output.text().strip() if hasattr(self, "annotate_output") else "",
        ):
            if not text:
                continue
            path = Path(text).expanduser()
            if path.name.lower() == "build" or path.name.lower().startswith("build_review_round"):
                path = path.parent
            if path.exists():
                return path

        desktop_project = Path.home() / "Desktop" / "laminar_boundary_builder_output"
        if desktop_project.exists():
            return desktop_project
        return Path.home() / "Desktop"

    @staticmethod
    def _previous_project_round_key(path: Path) -> tuple[int, float]:
        match = re.search(r"round(\d+)", path.stem)
        round_index = int(match.group(1)) if match else 0
        try:
            modified = path.stat().st_mtime
        except OSError:
            modified = 0.0
        return round_index, modified

    @staticmethod
    def _is_supported_volume_path(path: Path) -> bool:
        return "".join(path.suffixes).lower() in {
            ".npy",
            ".npz",
            ".nrrd",
            ".nhdr",
            ".nii",
            ".nii.gz",
        }

    @classmethod
    def _normalize_previous_project_selection(cls, selected_dir: Path) -> tuple[Path, Optional[Path]]:
        selected_dir = Path(selected_dir).expanduser()
        name = selected_dir.name.lower()
        if name == "build" or name.startswith("build_review_round"):
            return selected_dir.parent, selected_dir
        return selected_dir, None

    @classmethod
    def _find_previous_project_mask(cls, project_dir: Path) -> Optional[Path]:
        direct_names = ["target_mask.npy", "target_mask.npz", "target_mask.nrrd", "target_mask.nhdr"]
        direct_paths = [project_dir / "inputs" / name for name in direct_names]
        direct_paths.extend(project_dir / name for name in direct_names)
        direct_paths.extend(project_dir / "build" / "volumes" / name for name in direct_names)
        for path in direct_paths:
            if path.exists():
                return path

        candidates: List[Path] = []
        for folder in (project_dir / "inputs", project_dir):
            if not folder.exists():
                continue
            for path in folder.iterdir():
                if path.is_file() and "mask" in path.name.lower() and cls._is_supported_volume_path(path):
                    candidates.append(path)
        if not candidates:
            return None
        candidates.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0.0)
        return candidates[-1]

    @classmethod
    def _find_previous_project_manual_csv(cls, project_dir: Path) -> Optional[Path]:
        review_csvs = [
            path
            for path in project_dir.glob("manual_landmarks_review_round*.csv")
            if path.is_file()
        ]
        if review_csvs:
            return sorted(review_csvs, key=cls._previous_project_round_key)[-1]

        for name in ("manual_landmarks_interactive.csv", "manual_landmarks_autosave.csv"):
            path = project_dir / name
            if path.exists():
                return path

        candidates = [
            path
            for path in project_dir.glob("manual_landmarks*.csv")
            if path.is_file() and "template" not in path.stem.lower()
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0.0)
        return candidates[-1]

    @classmethod
    def _find_previous_project_build_dir(
        cls,
        project_dir: Path,
        preferred_build_dir: Optional[Path],
    ) -> Path:
        if preferred_build_dir is not None and preferred_build_dir.exists():
            return preferred_build_dir

        review_dirs = [
            path
            for path in project_dir.glob("build_review_round*")
            if path.is_dir()
        ]
        if review_dirs:
            return sorted(review_dirs, key=cls._previous_project_round_key)[-1]
        return project_dir / "build"

    @classmethod
    def _resolve_previous_project_paths(cls, selected_dir: Path) -> PreviousProjectPaths:
        project_dir, preferred_build_dir = cls._normalize_previous_project_selection(selected_dir)
        if not project_dir.exists():
            raise ValueError(f"Folder does not exist:\n{project_dir}")

        mask_path = cls._find_previous_project_mask(project_dir)
        if mask_path is None:
            raise ValueError(
                "Could not find a saved target mask.\n\n"
                "Expected something like inputs/target_mask.npy inside the project folder."
            )

        manual_csv = cls._find_previous_project_manual_csv(project_dir)
        if manual_csv is None:
            raise ValueError(
                "Could not find a saved manual landmark CSV.\n\n"
                "Expected manual_landmarks_interactive.csv or manual_landmarks_review_roundN.csv."
            )

        build_output_dir = cls._find_previous_project_build_dir(project_dir, preferred_build_dir)
        return PreviousProjectPaths(
            project_dir=project_dir,
            mask_path=mask_path,
            manual_csv=manual_csv,
            build_output_dir=build_output_dir,
        )

    def open_previous_project(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Open previous project",
            str(self._default_previous_project_dir()),
        )
        if not selected:
            return
        try:
            paths = self._resolve_previous_project_paths(Path(selected))
            self._restore_previous_project(paths)
        except Exception as exc:
            self._show_exception_dialog("Open previous project failed", exc)

    def _restore_previous_project(self, paths: PreviousProjectPaths) -> None:
        self.annotate_output.set_text(paths.project_dir)
        self.annotate_mask.set_text(paths.mask_path)
        self.annotate_previous_csv.set_text(paths.manual_csv)
        self.build_mask.set_text(paths.mask_path)
        self.build_manual.set_text(paths.manual_csv)
        self.build_output.set_text(paths.build_output_dir)
        self.build_boundaries.set_text(paths.build_output_dir / "boundary_annotations.json")
        self.depth_method.setCurrentText("surfaces only")
        self._update_annotation_readiness()
        self._update_build_readiness()
        self.append_log(
            "Opened previous project.\n"
            f"project_dir: {paths.project_dir}\n"
            f"mask: {paths.mask_path}\n"
            f"manual_csv: {paths.manual_csv}\n"
            f"build_output: {paths.build_output_dir}\n"
        )

        if paths.build_output_dir.exists():
            self.review_build_qc_slices()
            return

        self.pending_previous_csv_load = (paths.mask_path, paths.manual_csv, paths.project_dir)
        self.tabs.setCurrentIndex(0)
        if self.annotation_mask_data is not None and self.annotation_mask_path is not None:
            try:
                if self._same_path(self.annotation_mask_path, paths.mask_path):
                    self._finish_pending_previous_csv_after_mask_load(paths.mask_path)
                    return
            except OSError:
                pass
        self.start_annotation_mask_load(paths.mask_path)

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

        self.skip_annotation_slice_shortcut = QShortcut(QKeySequence(Qt.Key_K), self)
        self.skip_annotation_slice_shortcut.setContext(Qt.ApplicationShortcut)
        self.skip_annotation_slice_shortcut.activated.connect(self.skip_current_annotation_slice)

        self.outer_only_annotation_slice_shortcut = QShortcut(QKeySequence(Qt.Key_S), self)
        self.outer_only_annotation_slice_shortcut.setContext(Qt.ApplicationShortcut)
        self.outer_only_annotation_slice_shortcut.activated.connect(
            lambda: self.mark_current_annotation_cap("outer_only")
        )
        self.inner_only_annotation_slice_shortcut = QShortcut(QKeySequence(Qt.Key_A), self)
        self.inner_only_annotation_slice_shortcut.setContext(Qt.ApplicationShortcut)
        self.inner_only_annotation_slice_shortcut.activated.connect(
            lambda: self.mark_current_annotation_cap("inner_only")
        )

        self.flip_outer_path_shortcut = QShortcut(QKeySequence(Qt.Key_O), self)
        self.flip_outer_path_shortcut.setContext(Qt.ApplicationShortcut)
        self.flip_outer_path_shortcut.activated.connect(lambda: self.flip_annotation_arc("outer"))
        self.flip_inner_path_shortcut = QShortcut(QKeySequence(Qt.Key_I), self)
        self.flip_inner_path_shortcut.setContext(Qt.ApplicationShortcut)
        self.flip_inner_path_shortcut.activated.connect(lambda: self.flip_annotation_arc("inner"))

        self.annotation_shortcuts_overlay = QShortcut(QKeySequence(Qt.Key_H), self)
        self.annotation_shortcuts_overlay.setContext(Qt.ApplicationShortcut)
        self.annotation_shortcuts_overlay.activated.connect(lambda: self.slice_canvas.toggle_shortcut_help())

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

    def _make_note_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionNote")
        label.setWordWrap(True)
        return label

    def _tune_form(self, form: QFormLayout) -> None:
        form.setContentsMargins(10, 10, 10, 10)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

    def _make_form_section(
        self,
        title: str,
        checkable: bool = False,
        checked: bool = True,
    ) -> tuple[QGroupBox, QFormLayout]:
        section = QGroupBox(title)
        section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        if checkable:
            section.setCheckable(True)
            section.setChecked(checked)
        form = QFormLayout(section)
        self._tune_form(form)
        if checkable:
            section.toggled.connect(
                lambda visible, current_section=section, current_form=form: self._set_collapsible_form_visible(
                    current_section,
                    current_form,
                    visible,
                )
            )
        return section, form

    def _set_form_widgets_visible(self, form: QFormLayout, visible: bool) -> None:
        for row in range(form.rowCount()):
            for role in (QFormLayout.LabelRole, QFormLayout.FieldRole, QFormLayout.SpanningRole):
                item = form.itemAt(row, role)
                widget = item.widget() if item is not None else None
                if widget is not None:
                    widget.setVisible(visible)

    def _set_collapsible_form_visible(self, section: QGroupBox, form: QFormLayout, visible: bool) -> None:
        self._set_form_widgets_visible(form, visible)
        section.setMaximumHeight(16777215 if visible else 48)

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
        page_layout.addWidget(self._make_annotation_hint())

        content = QWidget()
        layout = QHBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        controls = self._make_annotation_control_panel()
        self.annotation_preview_splitter = self._make_annotation_preview_area()
        self.annotate_settings_button = self._make_annotation_settings_button()
        self.annotate_controls_scroll = self._make_annotation_controls_scroll(controls)

        layout.addWidget(self.annotate_settings_button)
        layout.addWidget(self.annotate_controls_scroll)
        layout.addWidget(self.annotation_preview_splitter, 1)
        page_layout.addWidget(content, 1)

        self._init_annotation_state()
        return tab

    def _make_annotation_hint(self) -> QFrame:
        return self._make_hint(
            "Choose one input source, then pick points directly on the 3D shell. "
            "Close a curve, select the surface patch, then build."
        )

    def _make_annotation_control_panel(self) -> QWidget:
        controls = QWidget()
        controls.setMinimumWidth(420)
        controls.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        self._create_annotation_input_fields()
        action_row = self._create_annotation_action_buttons()
        self._create_annotation_status_labels()

        self._add_annotation_source_section(controls_layout)
        self._add_annotation_reference_section(controls_layout)
        self._add_annotation_advanced_section(controls_layout)
        self._add_annotation_picking_section(controls_layout, action_row)
        controls_layout.addStretch(1)
        return controls

    def _create_annotation_input_fields(self) -> None:
        self.annotate_atlas = PathRow(
            "Built-in Allen annotation_10.nrrd, or choose another atlas",
            file_filter="Atlas files (*.pkl *.nrrd *.nhdr *.npy *.npz);;All files (*)",
        )
        self.annotate_custom_atlas = QCheckBox("Use a custom Allen atlas file")
        self.annotate_custom_atlas.toggled.connect(self._update_custom_atlas_visibility)
        self.annotate_region = QLineEdit()
        self.annotate_region.setPlaceholderText("Brain region, for example ENT or 909")
        self.annotate_region.setText("ENT")
        self.annotate_region.setMinimumHeight(28)
        self.annotate_hemisphere = CleanComboBox()
        self.annotate_hemisphere.addItems(["all", "left", "right"])
        self.annotate_hemisphere.setMinimumHeight(28)
        self.annotate_include_children = QCheckBox("Include child regions")
        self.annotate_include_children.setChecked(True)
        self.annotate_mask = PathRow("Optional existing target mask")
        self.annotate_template = PathRow("Optional template image volume")
        self.annotate_output = PathRow("Output folder for shell-cut annotations", select_file=False)
        self.annotate_previous_csv = PathRow(
            "Optional previous shell_cut_annotations.json",
            file_filter="JSON files (*.json);;All files (*)",
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
        self.annotate_surface_queue = CleanComboBox()
        self.annotate_surface_queue.setMinimumHeight(28)
        self.annotate_surface_queue.hide()
        self.annotate_surface_queue.currentIndexChanged.connect(self.on_3d_surface_queue_changed)
        self.annotate_surface_queue_panel = QFrame()
        self.annotate_surface_queue_panel.setObjectName("surfaceQueuePanel")
        self.annotate_surface_queue_layout = QVBoxLayout(self.annotate_surface_queue_panel)
        self.annotate_surface_queue_layout.setContentsMargins(0, 0, 0, 0)
        self.annotate_surface_queue_layout.setSpacing(5)
        self.surface_queue_buttons: List[QPushButton] = []
        self.annotate_surface_name = QLineEdit()
        self.annotate_surface_name.setPlaceholderText("Selected surface name, for example layer_outer")
        self.annotate_surface_name.setText("surface_1")
        self.annotate_surface_name.setMinimumHeight(28)
        self.annotate_surface_name.textChanged.connect(self.on_3d_surface_name_changed)
        self.annotate_pick_mode = CleanComboBox()
        self.annotate_pick_mode.addItems(["Shell Cut Boundary"])
        self.annotate_pick_mode.setMinimumHeight(28)
        self.annotate_pick_mode.currentIndexChanged.connect(self.on_annotation_pick_mode_changed)

        self.annotate_outer_path = CleanComboBox()
        self.annotate_outer_path.addItems(["auto", "forward", "backward"])
        self.annotate_outer_path.setMinimumHeight(28)
        self.annotate_inner_path = CleanComboBox()
        self.annotate_inner_path.addItems(["auto", "forward", "backward"])
        self.annotate_inner_path.setMinimumHeight(28)
        self.annotate_outer_path.currentIndexChanged.connect(self.on_annotation_path_choice_changed)
        self.annotate_inner_path.currentIndexChanged.connect(self.on_annotation_path_choice_changed)
        self._connect_annotation_readiness_signals()

    def _connect_annotation_readiness_signals(self) -> None:
        self.annotate_region.textChanged.connect(self._update_annotation_readiness)
        self.annotate_atlas.edit.textChanged.connect(self._update_annotation_readiness)
        self.annotate_mask.edit.textChanged.connect(self._update_annotation_readiness)
        self.annotate_template.edit.textChanged.connect(self._update_annotation_readiness)
        self.annotate_output.edit.textChanged.connect(self._update_annotation_readiness)
        self.annotate_custom_atlas.toggled.connect(self._update_annotation_readiness)
        self.annotate_hemisphere.currentIndexChanged.connect(self._update_annotation_readiness)
        self.annotate_include_children.toggled.connect(self._update_annotation_readiness)

    def _create_annotation_action_buttons(self) -> QWidget:
        self.load_button = self._make_button("Load / Reload Source And Start Picking", "primary")
        self.load_button.setToolTip(
            "Load the selected brain region or mask. You can change region/hemisphere and click again to rebuild the 3D shell."
        )
        self.load_button.clicked.connect(self.load_annotation_data)
        self.load_previous_csv_button = self._make_button("Load Previous Shell Cut JSON", "secondary")
        self.load_previous_csv_button.setToolTip("Load saved shell-cut annotations into the annotation workspace.")
        self.load_previous_csv_button.clicked.connect(self.load_previous_annotation_csv)
        self.draw_curve_button = self._make_button("Draw New Closed Curve", "secondary")
        self.draw_curve_button.setToolTip("Start or continue a 3D shell cut curve.")
        self.draw_curve_button.clicked.connect(self.set_3d_curve_mode)
        self.select_patch_button = self._make_button("Select Surface Patch", "secondary")
        self.select_patch_button.setToolTip("After at least one closed curve exists, click a surface patch to save.")
        self.select_patch_button.clicked.connect(self.set_3d_patch_mode)
        self.undo_3d_button = self._make_button("Undo Last 3D Step (X)", "secondary")
        self.undo_3d_button.setToolTip("Remove the last point, selected patch, or closed curve.")
        self.undo_3d_button.clicked.connect(self.undo_3d_annotation_action)
        self.clear_3d_button = self._make_button("Clear 3D Annotation", "danger")
        self.clear_3d_button.setToolTip("Clear all 3D cut curves and selected surfaces.")
        self.clear_3d_button.clicked.connect(self.clear_3d_annotations)
        self.build_3d_button = self._make_button("Build Queued 3D Surfaces", "primary")
        self.build_3d_button.setToolTip(
            "Build all queued surfaces. Each closed curve creates one queue item; choose the item, name it, then select its seed patch."
        )
        self.build_3d_button.setEnabled(False)
        self.build_3d_button.clicked.connect(self.run_3d_surface_build)
        self.save_slice_button = self._make_button("Accept Slice + Next", "primary")
        self.save_slice_button.setToolTip("Accept landmarks on this slice and move to the next smart target slice.")
        self.save_slice_button.clicked.connect(self.accept_annotation_slice_and_advance)
        self.suggest_slice_button = self._make_button("Refresh Smart Slice Set (N)", "secondary")
        self.suggest_slice_button.setToolTip(
            "Build a smart slice set that is denser near tails, shape changes, and QC review ranges."
        )
        self.suggest_slice_button.clicked.connect(self.suggest_next_annotation_slice)
        self.outer_only_slice_button = self._make_button("Outer Only / No Inner (S)", "secondary")
        self.outer_only_slice_button.setToolTip(
            "Mark the whole current contour as outer surface, with no inner surface on this slice."
        )
        self.outer_only_slice_button.clicked.connect(
            lambda _checked=False: self.mark_current_annotation_cap("outer_only")
        )
        self.inner_only_slice_button = self._make_button("Inner Only / No Outer (A)", "secondary")
        self.inner_only_slice_button.setToolTip(
            "Mark the whole current contour as inner surface, with no outer surface on this slice."
        )
        self.inner_only_slice_button.clicked.connect(
            lambda _checked=False: self.mark_current_annotation_cap("inner_only")
        )
        self.skip_slice_button = self._make_button("Skip Slice (K)", "secondary")
        self.skip_slice_button.setToolTip("Ignore this slice entirely when the contour is not usable.")
        self.skip_slice_button.clicked.connect(self.skip_current_annotation_slice)
        self.flip_outer_path_button = self._make_button("Flip Outer (O)", "secondary")
        self.flip_outer_path_button.setToolTip("Use the other contour side between the outer cut A/B points.")
        self.flip_outer_path_button.clicked.connect(lambda _checked=False: self.flip_annotation_arc("outer"))
        self.flip_inner_path_button = self._make_button("Flip Inner (I)", "secondary")
        self.flip_inner_path_button.setToolTip("Use the other contour side between the inner cut A/B points.")
        self.flip_inner_path_button.clicked.connect(lambda _checked=False: self.flip_annotation_arc("inner"))
        self.clear_slice_button = self._make_button("Clear Current Slice", "danger")
        self.clear_slice_button.setToolTip("Clear landmarks on the current slice.")
        self.clear_slice_button.clicked.connect(self.clear_annotation_slice)
        self.review_ok_button = self._make_button("Looks OK + Next", "secondary")
        self.review_ok_button.setToolTip("Mark this review slice as checked and move to the next review target.")
        self.review_ok_button.clicked.connect(self.mark_review_slice_ok)
        self.save_review_button = self._make_button("Save Review CSV", "secondary")
        self.save_review_button.setToolTip("Save the current repaired landmarks as a review-round CSV.")
        self.save_review_button.clicked.connect(self.save_review_repair_csv)
        self.rebuild_review_button = self._make_button("Rebuild Review Round", "primary")
        self.rebuild_review_button.setToolTip("Save review landmarks and rebuild into a new review output folder.")
        self.rebuild_review_button.clicked.connect(self.rebuild_review_round)
        self.export_button = self._make_button("Save Shell Cut And Review Build", "secondary")
        self.export_button.setToolTip("Save accepted shell-cut annotations and switch to the Build settings.")
        self.export_button.clicked.connect(self.export_annotation_csv)
        self.export_button.hide()
        self.export_mask_button = self._make_button("Export Current Mask", "secondary")
        self.export_mask_button.setToolTip("Save the current temporary extracted mask as a permanent file.")
        self.export_mask_button.clicked.connect(self.export_current_annotation_mask)
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
        single_surface_row = QWidget()
        single_surface_layout = QHBoxLayout(single_surface_row)
        single_surface_layout.setContentsMargins(0, 0, 0, 0)
        single_surface_layout.setSpacing(6)
        single_surface_layout.addWidget(self.outer_only_slice_button)
        single_surface_layout.addWidget(self.inner_only_slice_button)
        action_layout.addWidget(self.draw_curve_button)
        action_layout.addWidget(self.select_patch_button)
        action_layout.addWidget(self.undo_3d_button)
        action_layout.addWidget(self.clear_3d_button)
        action_layout.addWidget(self.review_ok_button)
        action_layout.addWidget(self.save_review_button)
        action_layout.addWidget(self.rebuild_review_button)
        action_layout.addWidget(self.export_mask_button)
        action_layout.addWidget(self.build_3d_button)
        self._apply_review_mode_ui(False)
        return action_row

    def _create_annotation_status_labels(self) -> None:
        self.annotation_readiness = QLabel()
        self.annotation_readiness.setObjectName("readinessText")
        self.annotation_readiness.setWordWrap(True)
        self.annotate_status = QLabel("No mask loaded")
        self.annotate_status.setWordWrap(True)
        self.next_point_label = QLabel("Next point: outer_cut_A")
        self.next_point_label.setObjectName("nextPointText")
        self.next_point_label.setWordWrap(True)
        self.annotate_progress = QLabel("Load a mask to see slice count.")
        self.annotate_progress.setObjectName("progressText")
        self.annotate_progress.setWordWrap(True)
        self.annotate_build_progress = BuildProgressPanel("Surface build progress appears here after you click Build.")
        self.annotate_build_progress.hide()
        self.annotation_review_slice = CleanComboBox()
        self.annotation_review_slice.addItem("No accepted slices yet", None)
        self.annotation_review_slice.activated.connect(self.jump_to_annotation_review_slice)

    def _add_annotation_source_section(self, controls_layout: QVBoxLayout) -> None:
        source_box, source_form = self._make_form_section(
            "1. Source",
            checkable=True,
            checked=True,
        )
        self.annotation_source_box = source_box
        source_form.addRow(
            "",
            self._make_note_label(
                "Default: extract ENT from the whole Allen atlas into a temporary mask. Choose Mask only when you already have one."
            ),
        )
        self._add_help_row(source_form, "Brain region", self.annotate_region, *ANNOTATE_HELP["region"])
        self._add_help_row(source_form, "Hemisphere", self.annotate_hemisphere, *ANNOTATE_HELP["hemisphere"])
        self._add_help_row(source_form, "", self.annotate_include_children, *ANNOTATE_HELP["include_children"])
        self._add_help_row(source_form, "", self.annotate_custom_atlas, *ANNOTATE_HELP["custom_atlas_enabled"])
        self.annotate_atlas_row = self._add_help_row(
            source_form, "Custom atlas", self.annotate_atlas, *ANNOTATE_HELP["custom_atlas"]
        )
        self.annotate_atlas_label = source_form.labelForField(self.annotate_atlas_row)
        self._update_custom_atlas_visibility(False)
        self._add_help_row(source_form, "Mask", self.annotate_mask, *ANNOTATE_HELP["mask"])
        source_form.addRow("Checklist", self.annotation_readiness)
        source_form.addRow("", self.load_button)
        controls_layout.addWidget(source_box)
        self._update_annotation_source_title()

    def _same_path(self, first: str | Path, second: str | Path) -> bool:
        return Path(first).expanduser().resolve() == Path(second).expanduser().resolve()

    def _mask_text_is_current_temporary_mask(self) -> bool:
        if not self.annotation_mask_is_temporary or self.annotation_mask_path is None:
            return False
        mask_text = self.annotate_mask.text().strip()
        if not mask_text:
            return False
        try:
            return self._same_path(mask_text, self.annotation_mask_path)
        except OSError:
            return False

    def _annotation_mask_source_name(self, mask_path: Path) -> str:
        if self.annotation_mask_is_temporary and self.annotation_mask_path is not None:
            try:
                if self._same_path(mask_path, self.annotation_mask_path):
                    return "temporary extracted mask"
            except OSError:
                pass
        return "existing mask"

    def _update_annotation_source_title(self) -> None:
        if not hasattr(self, "annotation_source_box"):
            return
        source = self.annotate_region.text().strip() or "region"
        mask_text = self.annotate_mask.text().strip()
        if mask_text and not self._mask_text_is_current_temporary_mask():
            source = Path(mask_text).expanduser().name or "mask"
        hemisphere = self.annotate_hemisphere.currentText().strip() or "all"
        parts = [source, hemisphere]
        if hasattr(self, "surface_preview_canvas") and self.surface_preview_canvas.shell_mesh is not None:
            shell_mesh = self.surface_preview_canvas.shell_mesh
            vertex_count = len(getattr(shell_mesh, "vertices", []))
            face_count = len(getattr(shell_mesh, "faces", []))
            if vertex_count and face_count:
                parts.append(f"{vertex_count:,} pts")
        self.annotation_source_box.setTitle("1. Source: " + " · ".join(parts))

    def _add_annotation_reference_section(self, controls_layout: QVBoxLayout) -> None:
        save_box, save_form = self._make_form_section(
            "2. Files & Recovery",
            checkable=True,
            checked=False,
        )
        self.annotation_reference_box = save_box
        self._add_help_row(save_form, "Template image", self.annotate_template, *ANNOTATE_HELP["template"])
        self._add_help_row(save_form, "Output folder", self.annotate_output, *ANNOTATE_HELP["output"])
        self._add_help_row(save_form, "Previous shell-cut JSON", self.annotate_previous_csv, *ANNOTATE_HELP["previous_csv"])
        save_form.addRow("", self.load_previous_csv_button)
        controls_layout.addWidget(save_box)

    def _add_annotation_advanced_section(self, controls_layout: QVBoxLayout) -> None:
        advanced_box, advanced_form = self._make_form_section(
            "Advanced Annotation Settings",
            checkable=True,
            checked=False,
        )
        self._add_help_row(advanced_form, "Slice axis", self.annotate_slice_axis, *ANNOTATE_HELP["slice_axis"])
        self._add_help_row(advanced_form, "Min contour area", self.annotate_min_area, *ANNOTATE_HELP["min_area"])
        self._add_help_row(advanced_form, "", self.annotate_keep_all, *ANNOTATE_HELP["keep_all"])
        self._set_collapsible_form_visible(advanced_box, advanced_form, False)
        controls_layout.addWidget(advanced_box)

    def _add_annotation_picking_section(self, controls_layout: QVBoxLayout, action_row: QWidget) -> None:
        picking_box, picking_form = self._make_form_section("3. 3D Pick And Build")
        self.annotation_picking_box = picking_box
        picking_form.addRow("Surfaces", self.annotate_surface_queue_panel)
        picking_form.addRow("Selected name", self.annotate_surface_name)
        picking_form.addRow("Next", self.next_point_label)
        picking_form.addRow("Actions", action_row)
        picking_form.addRow("Build", self.annotate_build_progress)
        picking_form.addRow("Shell", self.annotate_progress)
        controls_layout.addWidget(picking_box)

    def _make_annotation_preview_area(self) -> QSplitter:
        self.slice_canvas = SliceCanvas()
        self.slice_canvas.landmark_changed.connect(self.on_landmark_changed)
        self.slice_canvas.hide()

        self.surface_preview_canvas = ShellGLCanvas() if ShellGLCanvas is not None else SurfacePreviewCanvas()
        self.surface_preview_canvas.annotation_changed.connect(self.on_3d_annotation_changed)
        self.surface_preview_canvas.build_ready_changed.connect(self.on_3d_build_ready_changed)
        self.annotation_preview_splitter = QSplitter(Qt.Horizontal)
        self.annotation_preview_splitter.setChildrenCollapsible(False)
        self.annotation_preview_splitter.addWidget(self.surface_preview_canvas)
        self.annotation_preview_splitter.setStretchFactor(0, 1)
        return self.annotation_preview_splitter

    def _make_annotation_settings_button(self) -> QPushButton:
        self.annotate_settings_button = QPushButton("›")
        self.annotate_settings_button.setObjectName("settingsPeek")
        self.annotate_settings_button.setToolTip("Show or hide annotation settings. Esc exits point-picking mode.")
        self.annotate_settings_button.clicked.connect(self.toggle_annotation_settings_panel)
        self.annotate_settings_button.hide()
        return self.annotate_settings_button

    def _make_annotation_controls_scroll(self, controls: QWidget) -> QScrollArea:
        self.annotate_controls_scroll = QScrollArea()
        self.annotate_controls_scroll.setObjectName("sideScroll")
        self.annotate_controls_scroll.setWidgetResizable(True)
        self.annotate_controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.annotate_controls_scroll.setWidget(controls)
        self.annotate_controls_scroll.setMinimumWidth(450)
        self.annotate_controls_scroll.setMaximumWidth(480)
        return self.annotate_controls_scroll

    def _init_annotation_state(self) -> None:
        self.annotation_mask_data = None
        self.annotation_template_data = None
        self.annotation_slice_axis_int = 0
        self.annotation_rows: Dict[int, Dict[str, str]] = {}
        self.annotation_landmarks_by_slice: Dict[int, Dict[str, int]] = {}
        self.shell_cut_rows_by_slice: Dict[int, Dict] = {}
        self.shell_cut_landmarks_by_slice: Dict[int, Dict[str, int]] = {}
        self.annotation_path_choices_by_slice: Dict[int, Dict[str, str]] = {}
        self.annotation_skipped_slices = set()
        self.annotation_contours_by_slice: Dict[int, list] = {}
        self.annotation_boundary_cache: Dict[int, tuple[tuple[str, ...], object]] = {}
        self.annotation_region_slices: List[int] = []
        self.annotation_target_slices: List[int] = []
        self.annotation_reference_contours = []
        self.annotation_slice_counts = None
        self.annotation_extraction_signature = None
        self.pending_annotation_extraction_signature = None
        self.annotation_picking_active = False
        self.annotation_pick_mode_kind = "shell_cut"
        self.annotation_settings_expanded = True
        self.review_mode_active = False
        self.review_queue_slices: List[int] = []
        self.review_checked_slices = set()
        self.review_source_build_dir: Optional[Path] = None
        self.review_round_csv_path: Optional[Path] = None
        self._update_annotation_readiness()
        self._update_annotation_review_slice_choices()
        self._apply_review_mode_ui(False)
        if hasattr(self, "surface_preview_canvas"):
            self.on_3d_annotation_changed()

    def _resolved_existing_input_path(self, text: str) -> Optional[Path]:
        raw = str(text or "").strip()
        if not raw:
            return None
        try:
            path = _core()._resolve_existing_input_path(raw)
        except Exception:
            path = Path(raw).expanduser()
        return path if path.exists() else None

    def _update_annotation_readiness(self, *_args) -> None:
        if not hasattr(self, "annotation_readiness"):
            return

        region = self.annotate_region.text().strip()
        mask_text = self.annotate_mask.text()
        template_text = self.annotate_template.text()
        output_text = self.annotate_output.text().strip()
        custom_atlas_checked = self.annotate_custom_atlas.isChecked()
        custom_atlas_text = self.annotate_atlas.text()

        lines = []
        has_source = False
        has_warning = False
        mask_is_current_temp = self._mask_text_is_current_temporary_mask()
        mask_path = self._resolved_existing_input_path(mask_text) if mask_text else None
        uses_atlas = bool(region) and (not mask_text or mask_is_current_temp)
        if uses_atlas:
            if custom_atlas_checked:
                atlas_path = self._resolved_existing_input_path(custom_atlas_text)
                if atlas_path is None:
                    has_warning = True
                    lines.append("Input source: Brain region set, but custom atlas file is missing.")
                else:
                    has_source = True
                    lines.append(f"Input source: Brain region from custom atlas ({atlas_path.name}).")
            else:
                try:
                    atlas_path = _core().resolve_annotation_path(None)
                    has_source = True
                    lines.append(f"Input source: Brain region from built-in atlas ({Path(atlas_path).name}).")
                except Exception:
                    has_warning = True
                    lines.append("Input source: Brain region set, but built-in atlas was not found.")
            if mask_is_current_temp:
                lines.append("Mask: temporary extraction cache; it will be replaced when loading again.")
        elif mask_text:
            has_source = mask_path is not None
            if mask_path is None:
                has_warning = True
                lines.append("Input source: Mask path is set, but the file was not found.")
            else:
                source_name = self._annotation_mask_source_name(mask_path)
                lines.append(f"Input source: {source_name} ({mask_path.name}).")
                if region:
                    lines.append("Brain region: ignored while a non-temporary Mask path is set.")
        else:
            lines.append("Input source: choose Brain region or an existing Mask.")

        if template_text:
            template_path = self._resolved_existing_input_path(template_text)
            if template_path is None:
                has_warning = True
                lines.append("Template: file was not found; it will not load.")
            else:
                lines.append(f"Template: ready ({template_path.name}).")
        else:
            lines.append("Template: optional.")

        if output_text:
            lines.append("Output folder: set.")
        else:
            lines.append("Output folder: optional now; the app can choose one when saving.")

        self.annotation_readiness.setText("\n".join(lines))
        state = "ready" if has_source and not has_warning else "warning" if has_source else "missing"
        self._set_label_state(self.annotation_readiness, state)
        self.load_button.setEnabled(has_source)
        self._update_annotation_source_title()

    def _update_annotation_review_slice_choices(self) -> None:
        if not hasattr(self, "annotation_review_slice"):
            return
        current_slice = int(self.annotate_slice.value()) if hasattr(self, "annotate_slice") else None
        self.annotation_review_slice.blockSignals(True)
        self.annotation_review_slice.clear()

        rows = []
        for slice_index in sorted(self.annotation_rows):
            row = self.annotation_rows[slice_index]
            try:
                mode = self._surface_mode_label(row.get("surface_mode", "normal"))
            except Exception:
                mode = "accepted"
            rows.append((slice_index, f"Accepted slice {slice_index} - {mode}"))
        for slice_index in sorted(self.shell_cut_rows_by_slice):
            rows.append((slice_index, f"Shell cut slice {slice_index}"))
        for slice_index in sorted(self.annotation_skipped_slices):
            if slice_index not in self.annotation_rows:
                rows.append((slice_index, f"Skipped slice {slice_index}"))

        if not rows:
            self.annotation_review_slice.addItem("No accepted slices yet", None)
        else:
            for slice_index, label in sorted(rows):
                self.annotation_review_slice.addItem(label, slice_index)
            if current_slice is not None:
                for index in range(self.annotation_review_slice.count()):
                    if self.annotation_review_slice.itemData(index) == current_slice:
                        self.annotation_review_slice.setCurrentIndex(index)
                        break
        self.annotation_review_slice.blockSignals(False)

    def jump_to_annotation_review_slice(self, index: int) -> None:
        if index < 0:
            return
        slice_index = self.annotation_review_slice.itemData(index)
        if slice_index is None or self.annotation_mask_data is None:
            return
        self.annotate_slice.setValue(int(slice_index))
        self.slice_canvas.setFocus(Qt.OtherFocusReason)
        self.annotate_status.setText(f"Reviewing slice {slice_index}. Edit points or clear it if needed.")

    def _annotation_shortcuts_active(self) -> bool:
        return self.tabs.currentIndex() == 0 and self.annotation_picking_active

    def _annotation_pick_mode_kind_from_ui(self) -> str:
        return "shell_cut"

    def _active_annotation_landmark_order(self) -> tuple[str, ...]:
        if self.annotation_pick_mode_kind == "shell_cut":
            return SliceCanvas.SHELL_CUT_ORDER
        return SliceCanvas.LANDMARK_ORDER

    def _next_annotation_point(self) -> Optional[str]:
        for name in self._active_annotation_landmark_order():
            if name not in self.slice_canvas.landmarks:
                return name
        return None

    def _set_next_annotation_mode(self) -> None:
        if hasattr(self, "surface_preview_canvas") and self.surface_preview_canvas.shell_mesh is not None:
            self.on_3d_annotation_changed()
            return
        if hasattr(self, "slice_canvas"):
            self.slice_canvas.set_picking_mode_kind(self.annotation_pick_mode_kind)
        next_point = self._next_annotation_point()
        if next_point:
            self.slice_canvas.set_mode(next_point)
            self.next_point_label.setText(
                f"Next point: {next_point}. Click on the contour. X = undo."
            )
        else:
            self.slice_canvas.set_mode("")
            if self.annotation_pick_mode_kind == "shell_cut":
                self.next_point_label.setText(
                    "All four shell-cut endpoints are set. Press Enter or click Accept Slice + Next."
                )
            else:
                self.next_point_label.setText(
                    "All four points are set. Press Enter or click Accept Slice + Next."
                )
        self.slice_canvas.set_picking_enabled(self.annotation_picking_active)

    def _store_current_canvas_landmarks(self) -> None:
        if self.annotation_mask_data is None or not hasattr(self, "slice_canvas"):
            return
        slice_index = int(self.annotate_slice.value())
        if self.annotation_pick_mode_kind == "shell_cut":
            self.shell_cut_landmarks_by_slice[slice_index] = dict(self.slice_canvas.landmarks)
        else:
            self.annotation_landmarks_by_slice[slice_index] = dict(self.slice_canvas.landmarks)

    def on_annotation_pick_mode_changed(self, *_args) -> None:
        if not hasattr(self, "slice_canvas"):
            return
        self._store_current_canvas_landmarks()
        self.annotation_pick_mode_kind = self._annotation_pick_mode_kind_from_ui()
        self.annotate_outer_path.setEnabled(True)
        self.annotate_inner_path.setEnabled(True)
        self.outer_only_slice_button.setEnabled(True)
        self.inner_only_slice_button.setEnabled(True)
        self.flip_outer_path_button.setEnabled(True)
        self.flip_inner_path_button.setEnabled(True)
        self.refresh_annotation_slice()

    def enter_annotation_picking_mode(self) -> None:
        self.annotation_picking_active = True
        is_3d = hasattr(self, "surface_preview_canvas") and self.surface_preview_canvas.shell_mesh is not None
        self.annotation_settings_expanded = True if is_3d else False
        self.annotate_settings_button.setVisible(not is_3d)
        self._set_annotation_parameter_widgets_enabled(is_3d)
        if is_3d:
            self._update_annotation_source_title()
            if hasattr(self, "annotation_source_box"):
                self.annotation_source_box.setChecked(False)
            if hasattr(self, "annotation_reference_box"):
                self.annotation_reference_box.setChecked(False)
        self._apply_annotation_settings_panel_state()
        self._set_next_annotation_mode()
        if hasattr(self, "surface_preview_canvas") and self.surface_preview_canvas.shell_mesh is not None:
            self.surface_preview_canvas.setFocus(Qt.OtherFocusReason)
        else:
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
        self.annotate_status.setText(
            "Point picking paused. Edit settings, then Load / Reload Source And Start Picking again."
        )

    def toggle_annotation_settings_panel(self) -> None:
        if not self.annotation_picking_active:
            return
        self.annotation_settings_expanded = not self.annotation_settings_expanded
        self._apply_annotation_settings_panel_state()

    def _apply_annotation_settings_panel_state(self) -> None:
        is_3d = hasattr(self, "surface_preview_canvas") and self.surface_preview_canvas.shell_mesh is not None
        expanded = is_3d or self.annotation_settings_expanded or not self.annotation_picking_active
        self.annotate_controls_scroll.setVisible(expanded)
        self.annotate_settings_button.setText("‹" if expanded else "›")
        if hasattr(self, "surface_preview_canvas"):
            self.surface_preview_canvas.setVisible(self.annotation_picking_active)

    def _has_3d_annotation_shell(self) -> bool:
        return bool(
            hasattr(self, "surface_preview_canvas")
            and self.surface_preview_canvas.shell_mesh is not None
        )

    def on_3d_surface_name_changed(self, text: str) -> None:
        if hasattr(self, "surface_preview_canvas"):
            self.surface_preview_canvas.set_surface_name(text)

    def on_3d_surface_queue_changed(self, index: int) -> None:
        if not hasattr(self, "surface_preview_canvas"):
            return
        surface_index = self.annotate_surface_queue.itemData(index)
        if surface_index is None:
            return
        self.surface_preview_canvas.set_active_surface_index(int(surface_index))
        self.surface_preview_canvas.setFocus(Qt.OtherFocusReason)

    def on_3d_build_ready_changed(self, ready: bool) -> None:
        if hasattr(self, "build_3d_button"):
            self.build_3d_button.setEnabled(bool(ready))

    def _clear_surface_queue_rows(self) -> None:
        if not hasattr(self, "annotate_surface_queue_layout"):
            return
        while self.annotate_surface_queue_layout.count():
            item = self.annotate_surface_queue_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.surface_queue_buttons = []

    def _activate_3d_surface_from_queue_row(self, surface_index: int) -> None:
        if not hasattr(self, "surface_preview_canvas"):
            return
        self.surface_preview_canvas.set_active_surface_index(surface_index)
        self.surface_preview_canvas.setFocus(Qt.OtherFocusReason)

    def _rebuild_surface_queue_rows(self, queue: List[Dict[str, object]], active_index: int) -> None:
        if not hasattr(self, "annotate_surface_queue_layout"):
            return
        self._clear_surface_queue_rows()
        if not queue:
            empty = QLabel("Close a curve to add the first surface.")
            empty.setObjectName("surfaceQueueEmpty")
            empty.setWordWrap(True)
            self.annotate_surface_queue_layout.addWidget(empty)
            return

        for item in queue:
            index = int(item["index"])
            name = str(item["name"])
            seed_count = int(item["seed_count"])
            active = index == active_index
            seed_text = "seed selected" if seed_count else "needs seed"
            marker = "●" if active else "○"
            button = QPushButton(f"{marker} {index + 1}. {name} · {seed_text}")
            button.setObjectName("surfaceQueueItem")
            button.setCheckable(True)
            button.setChecked(active)
            button.setProperty("active", "true" if active else "false")
            button.setProperty("seeded", "true" if seed_count else "false")
            button.setToolTip("Click to make this the current surface for naming and seed selection.")
            button.clicked.connect(
                lambda _checked=False, current_index=index: self._activate_3d_surface_from_queue_row(current_index)
            )
            self.annotate_surface_queue_layout.addWidget(button)
            self.surface_queue_buttons.append(button)

    def _sync_3d_surface_queue_controls(self) -> List[Dict[str, object]]:
        if not hasattr(self, "surface_preview_canvas") or not hasattr(self, "annotate_surface_queue"):
            return []
        queue = self.surface_preview_canvas.surface_queue()
        active_index = int(getattr(self.surface_preview_canvas, "active_surface_index", 0))

        self.annotate_surface_queue.blockSignals(True)
        self.annotate_surface_queue.clear()
        if not queue:
            self.annotate_surface_queue.addItem("No closed curve yet", None)
        else:
            for item in queue:
                index = int(item["index"])
                name = str(item["name"])
                seed_count = int(item["seed_count"])
                self.annotate_surface_queue.addItem(
                    f"{index + 1}. {name}  ({seed_count} seed{'s' if seed_count != 1 else ''})",
                    index,
                )
            self.annotate_surface_queue.setCurrentIndex(max(0, min(active_index, self.annotate_surface_queue.count() - 1)))
        self.annotate_surface_queue.blockSignals(False)
        self._rebuild_surface_queue_rows(queue, active_index)

        active_name = str(queue[active_index]["name"]) if queue else "surface_1"
        self.annotate_surface_name.blockSignals(True)
        self.annotate_surface_name.setText(active_name)
        self.annotate_surface_name.setEnabled(bool(queue))
        self.annotate_surface_queue.setEnabled(bool(queue))
        if hasattr(self, "annotate_surface_queue_panel"):
            self.annotate_surface_queue_panel.setEnabled(bool(queue))
        self.annotate_surface_name.blockSignals(False)
        return queue

    def on_3d_annotation_changed(self) -> None:
        if not hasattr(self, "surface_preview_canvas") or not hasattr(self, "next_point_label"):
            return
        curve_count, point_count, patch_count = self.surface_preview_canvas.annotation_counts()
        queue = self._sync_3d_surface_queue_controls()
        if self.surface_preview_canvas.shell_mesh is None:
            text = "3D shell is not ready."
        elif point_count:
            close_hint = "Click an active point or press Close Current Curve to close."
            text = f"Drawing: {point_count} point(s). {close_hint} X = undo."
        elif curve_count and self.surface_preview_canvas.can_build_3d_surfaces():
            text = "Ready: every surface has a seed. Check names, then build."
        elif queue:
            missing = [str(int(item["index"]) + 1) for item in queue if int(item["seed_count"]) == 0]
            text = f"Next: select seed patch for surface {', '.join(missing)}."
        elif curve_count:
            text = "Next: select a seed patch for the selected surface."
        else:
            text = "Next: draw a closed curve on the shell."
        self.next_point_label.setText(text)
        if hasattr(self, "draw_curve_button"):
            can_close = point_count >= 3
            self.draw_curve_button.setText("Close Current Curve" if can_close else "Draw New Closed Curve")
            self.draw_curve_button.setToolTip(
                "Close the current curve by connecting the last point back to the first point."
                if can_close
                else "Start or continue a 3D shell cut curve."
            )
        self.on_3d_build_ready_changed(self.surface_preview_canvas.can_build_3d_surfaces())
        if hasattr(self, "annotate_progress"):
            self._update_annotation_progress()

    def set_3d_curve_mode(self) -> None:
        if hasattr(self, "surface_preview_canvas"):
            self.surface_preview_canvas.set_curve_mode()
            self.surface_preview_canvas.setFocus(Qt.OtherFocusReason)
            self.on_3d_annotation_changed()

    def set_3d_patch_mode(self) -> None:
        if hasattr(self, "surface_preview_canvas"):
            self.surface_preview_canvas.set_patch_mode()
            self.surface_preview_canvas.setFocus(Qt.OtherFocusReason)
            self.on_3d_annotation_changed()

    def undo_3d_annotation_action(self) -> None:
        if hasattr(self, "surface_preview_canvas") and self.surface_preview_canvas.undo_3d_action():
            self.surface_preview_canvas.setFocus(Qt.OtherFocusReason)
            self.on_3d_annotation_changed()
            return
        self.annotate_status.setText("Nothing to undo in the 3D annotation.")

    def clear_3d_annotations(self) -> None:
        if not hasattr(self, "surface_preview_canvas"):
            return
        self.surface_preview_canvas.clear_3d_annotations()
        self.annotate_status.setText("3D annotation cleared.")
        self.surface_preview_canvas.setFocus(Qt.OtherFocusReason)

    def _surface_3d_annotations_payload(self) -> Dict[str, object]:
        if not hasattr(self, "surface_preview_canvas"):
            raise ValueError("3D annotation canvas is not ready")
        return self.surface_preview_canvas.annotation_payload(mask_path=self.annotation_mask_path)

    def _write_surface_3d_annotations_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._surface_3d_annotations_payload()
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def run_3d_surface_build(self) -> None:
        if self.annotation_mask_path is None:
            QMessageBox.warning(self, "No mask", "Load a mask before building 3D surfaces.")
            return
        if not hasattr(self, "surface_preview_canvas") or not self.surface_preview_canvas.can_build_3d_surfaces():
            QMessageBox.warning(
                self,
                "3D annotation incomplete",
                "Close at least one cut curve, then select one seed patch for every queued surface.",
            )
            return

        try:
            core = _core()
            annotation_output_dir = self._auto_build_output_dir()
            if not self.annotate_output.text().strip():
                self.annotate_output.set_text(annotation_output_dir)
            surface_queue = self.surface_preview_canvas.surface_queue()
            surface_names = [str(item["name"]) for item in surface_queue]
            build_dir = annotation_output_dir / "build_3d"
            surface_slugs = [
                core.safe_surface_name(name, fallback=f"surface_{index + 1}")
                for index, name in enumerate(surface_names)
            ]
            annotation_suffix = "_".join(surface_slugs[:3])
            if len(surface_slugs) > 3:
                annotation_suffix = f"{annotation_suffix}_and_{len(surface_slugs) - 3}_more"
            annotations_path = build_dir / f"surface_3d_annotations_{annotation_suffix or 'surface'}.json"
            self._write_surface_3d_annotations_json(annotations_path)
            self.build_mask.set_text(self.annotation_mask_path)
            self.build_output.set_text(build_dir)
            self.surface_method.setCurrentText("Shell cut")
            self.depth_method.setCurrentText("surfaces only")
            shell_backend = (
                getattr(self.surface_preview_canvas.shell_mesh, "backend", None)
                or self.shell_backend.currentText()
            )
            shell_backend_text = str(shell_backend)
            if shell_backend_text.startswith("voxel_preview"):
                shell_backend = core.SHELL_BACKEND_VOXEL
            elif shell_backend_text.startswith("marching_cubes_preview"):
                shell_backend = core.SHELL_BACKEND_MARCHING_CUBES
        except Exception as exc:
            self._show_exception_dialog("Save 3D annotation failed", exc)
            return

        def task(progress: Callable[[int, str, str], None]) -> TaskResult:
            core = _core()
            outputs = core.run_3d_shell_patch_pipeline(
                mask_path=self.annotation_mask_path,
                cut_curve_json=annotations_path,
                output_dir=build_dir,
                shell_backend=shell_backend,
                progress_callback=progress,
            )
            lines = ["3D surface build finished."]
            lines.extend(f"{key}: {value}" for key, value in outputs.items())
            return TaskResult(
                "Build finished",
                "\n".join(lines),
                output_dir=build_dir,
                payload={
                    "clear_3d_annotations": True,
                    "surface_names": surface_names,
                    "annotations_path": annotations_path,
                },
            )

        self.start_task("build", task, accepts_progress=True)

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
        if enabled:
            self._update_annotation_readiness()

    def _apply_review_mode_ui(self, active: bool) -> None:
        for name in ("review_ok_button", "save_review_button", "rebuild_review_button"):
            if hasattr(self, name):
                getattr(self, name).setVisible(active)
        if hasattr(self, "export_button"):
            self.export_button.hide()

    def _path_choices_for_slice(self, slice_index: int) -> Dict[str, str]:
        choices = {"outer_path": "auto", "inner_path": "auto"}
        row = (
            self.shell_cut_rows_by_slice.get(slice_index)
            if self.annotation_pick_mode_kind == "shell_cut"
            else self.annotation_rows.get(slice_index)
        )
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

    @staticmethod
    def _surface_mode_label(surface_mode: str) -> str:
        mode = _core().normalize_surface_mode(surface_mode)
        if mode == "outer_only":
            return "outer-only / no inner"
        if mode == "inner_only":
            return "inner-only / no outer"
        return "normal"

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
        if self.annotation_pick_mode_kind == "shell_cut":
            row = self.shell_cut_rows_by_slice.get(slice_index)
            if row is None:
                return None
            outer_path = self.annotate_outer_path.currentText()
            inner_path = self.annotate_inner_path.currentText()
            row["outer_path"] = outer_path
            row["inner_path"] = inner_path
            row["note"] = note
            annotation_row = self.annotation_rows.get(slice_index)
            if annotation_row is not None:
                annotation_row["outer_path"] = outer_path
                annotation_row["inner_path"] = inner_path
                annotation_row["note"] = f"derived_from_shell_cut_{note}"
                self.annotation_boundary_cache.pop(slice_index, None)
            self.annotation_path_choices_by_slice[slice_index] = {
                "outer_path": outer_path,
                "inner_path": inner_path,
            }
            shell_path = self._autosave_shell_cut_annotations()
            self._autosave_annotation_rows()
            return shell_path

        row = self.annotation_rows.get(slice_index)
        if row is None:
            return None
        row["outer_path"] = self.annotate_outer_path.currentText()
        row["inner_path"] = self.annotate_inner_path.currentText()
        row["note"] = note
        self.annotation_boundary_cache.pop(slice_index, None)
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
        if self.annotation_pick_mode_kind == "shell_cut":
            start_name = f"{arc_name}_cut_A"
            end_name = f"{arc_name}_cut_B"
        else:
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
        if self.annotation_pick_mode_kind == "shell_cut":
            start_name = f"{arc_name}_cut_A"
            end_name = f"{arc_name}_cut_B"
        else:
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
            f"Slice {slice_index}: {arc_name} contour side flipped to {new_choice}.{saved}"
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

    def _slice_shape_features(self, slices: List[int]) -> Dict[int, Dict[str, float]]:
        if self.annotation_mask_data is None:
            return {}
        np = _numpy()
        core = _core()
        features: Dict[int, Dict[str, float]] = {}
        for slice_index in slices:
            try:
                plane = core._take_slice(
                    self.annotation_mask_data,
                    int(slice_index),
                    self.annotation_slice_axis_int,
                )
            except Exception:
                continue
            mask = np.asarray(plane, dtype=bool)
            coords = np.argwhere(mask)
            if coords.size == 0:
                continue
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0)
            centroid = coords.mean(axis=0)
            padded = np.pad(mask, 1, mode="constant", constant_values=False)
            center = padded[1:-1, 1:-1]
            interior = (
                center
                & padded[:-2, 1:-1]
                & padded[2:, 1:-1]
                & padded[1:-1, :-2]
                & padded[1:-1, 2:]
            )
            boundary = center & ~interior
            features[int(slice_index)] = {
                "area": float(coords.shape[0]),
                "row": float(centroid[0]),
                "col": float(centroid[1]),
                "height": float(maxs[0] - mins[0] + 1),
                "width": float(maxs[1] - mins[1] + 1),
                "perimeter": float(np.count_nonzero(boundary)),
            }
        return features

    @staticmethod
    def _add_nearby_slices(picked: set, slices: List[int], center_slice: int, radius: int = 1) -> None:
        if not slices:
            return
        nearest = min(slices, key=lambda value: (abs(value - center_slice), value))
        center_index = slices.index(nearest)
        for offset in range(-radius, radius + 1):
            index = center_index + offset
            if 0 <= index < len(slices):
                picked.add(int(slices[index]))

    def _tail_anchor_slices(
        self,
        slices: List[int],
        features: Dict[int, Dict[str, float]],
    ) -> List[int]:
        if not slices or not features:
            return []
        max_area = max((feature["area"] for feature in features.values()), default=0.0)
        if max_area <= 0:
            return []
        anchors = {int(slices[0]), int(slices[-1])}
        for fraction in (0.025, 0.05, 0.10, 0.25):
            minimum = max_area * fraction
            left = next(
                (slice_index for slice_index in slices if features.get(slice_index, {}).get("area", 0.0) >= minimum),
                None,
            )
            right = next(
                (
                    slice_index
                    for slice_index in reversed(slices)
                    if features.get(slice_index, {}).get("area", 0.0) >= minimum
                ),
                None,
            )
            if left is not None:
                anchors.add(int(left))
            if right is not None:
                anchors.add(int(right))
        return sorted(anchors)

    @staticmethod
    def _shape_change_scores(
        slices: List[int],
        features: Dict[int, Dict[str, float]],
    ) -> List[tuple[float, int, int]]:
        scores: List[tuple[float, int, int]] = []
        for left, right in zip(slices[:-1], slices[1:]):
            left_feature = features.get(left)
            right_feature = features.get(right)
            if left_feature is None or right_feature is None:
                continue
            area_change = abs(math.log((right_feature["area"] + 1.0) / (left_feature["area"] + 1.0)))
            perimeter_change = abs(
                math.log((right_feature["perimeter"] + 1.0) / (left_feature["perimeter"] + 1.0))
            )
            width_change = abs(math.log((right_feature["width"] + 1.0) / (left_feature["width"] + 1.0)))
            height_change = abs(math.log((right_feature["height"] + 1.0) / (left_feature["height"] + 1.0)))
            centroid_scale = max(8.0, math.sqrt(max(left_feature["area"], right_feature["area"])))
            centroid_shift = math.hypot(
                right_feature["row"] - left_feature["row"],
                right_feature["col"] - left_feature["col"],
            ) / centroid_scale
            score = 1.4 * area_change + centroid_shift + 0.8 * (width_change + height_change) + 0.7 * perimeter_change
            scores.append((float(score), int(left), int(right)))
        return sorted(scores, reverse=True)

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

    @staticmethod
    def _read_surface_jump_slices(output_dir: Path) -> List[int]:
        jump_path = output_dir / "qc" / "surface_jump_diagnostics.csv"
        if not jump_path.exists():
            return []
        review_slices = []
        try:
            with jump_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    if not str(row.get("flags") or "").strip():
                        continue
                    review_slices.append(int(float(row["left_slice"])))
                    review_slices.append(int(float(row["right_slice"])))
        except (OSError, ValueError, KeyError):
            return []
        return sorted(set(review_slices))

    @staticmethod
    def _read_surface_jump_details(output_dir: Path) -> List[str]:
        jump_path = output_dir / "qc" / "surface_jump_diagnostics.csv"
        if not jump_path.exists():
            return []
        details = []
        try:
            with jump_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    flags = [
                        flag.strip()
                        for flag in str(row.get("flags") or "").split(";")
                        if flag.strip()
                    ]
                    if not flags:
                        continue
                    label = (
                        f"{row.get('arc', 'surface')} "
                        f"{int(float(row['left_slice']))}-{int(float(row['right_slice']))}"
                    )
                    if "manual_topology_jump" in flags:
                        label += " manual"
                    elif "auto_transition" in flags:
                        label += " auto"
                    details.append(label)
        except (OSError, ValueError, KeyError):
            return []
        return details

    @classmethod
    def _read_build_review_slices(cls, output_dir: Path) -> List[int]:
        review_slices = set(cls._read_qc_review_slices(output_dir))
        review_slices.update(cls._read_surface_jump_slices(output_dir))
        return sorted(review_slices)

    def _current_qc_review_slices(self) -> List[int]:
        output_text = ""
        if hasattr(self, "build_output"):
            output_text = self.build_output.text().strip()
        if output_text:
            return self._read_build_review_slices(Path(output_text).expanduser())
        annotate_output = self.annotate_output.text().strip() if hasattr(self, "annotate_output") else ""
        if annotate_output:
            return self._read_build_review_slices(Path(annotate_output).expanduser() / "build")
        return []

    @staticmethod
    def _manual_qc_flag_slices(output_dir: Path) -> List[int]:
        benign_flags = {"no_lateral_boundary"}
        summary_path = output_dir / "tables" / "boundary_summary.csv"
        flagged = []
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
                        flagged.append(int(float(row["slice_index"])))
        except (OSError, ValueError, KeyError):
            return []
        return sorted(set(flagged))

    def _review_queue_from_build(self, output_dir: Path) -> List[int]:
        review_slices = self._read_build_review_slices(output_dir)
        queue = set(self._review_targets_from_ranges(review_slices))
        queue.update(self._manual_qc_flag_slices(output_dir))
        return sorted(queue or review_slices)

    @staticmethod
    def _manual_slices_from_csv(manual_csv: Path) -> List[int]:
        slices = []
        try:
            for row in _core().read_manual_landmarks(manual_csv):
                slices.append(int(float(row["slice_index"])))
        except (OSError, ValueError, KeyError):
            return []
        return sorted(set(slices))

    @staticmethod
    def _annotation_output_dir_from_build_dir(build_dir: Path) -> Path:
        name = build_dir.name.lower()
        if name == "build" or name.startswith("build_review_round"):
            return build_dir.parent
        return build_dir

    @staticmethod
    def _next_review_round_paths(annotation_output_dir: Path) -> tuple[Path, Path, int]:
        round_index = 2
        while True:
            csv_path = annotation_output_dir / f"manual_landmarks_review_round{round_index}.csv"
            build_dir = annotation_output_dir / f"build_review_round{round_index}"
            if not csv_path.exists() and not build_dir.exists():
                return csv_path, build_dir, round_index
            round_index += 1

    def _build_suggested_annotation_set(self) -> List[int]:
        viable = self._viable_annotation_slices()
        if not viable:
            return []
        target_count = min(len(viable), max(8, min(72, (len(viable) + 7) // 8)))
        picked = set(self._pick_evenly_spaced_slices(viable, target_count))
        if self.annotation_region_slices:
            picked.add(int(self.annotation_region_slices[0]))
            picked.add(int(self.annotation_region_slices[-1]))

        features = self._slice_shape_features(viable)
        for slice_index in self._tail_anchor_slices(viable, features):
            self._add_nearby_slices(picked, viable, slice_index, radius=1)

        shape_scores = self._shape_change_scores(viable, features)
        change_slots = min(max(4, target_count // 4), max(0, len(shape_scores)))
        for _score, left, right in shape_scores[:change_slots]:
            self._add_nearby_slices(picked, viable, left, radius=1)
            self._add_nearby_slices(picked, viable, right, radius=1)

        qc_review_slices = [
            slice_index for slice_index in self._current_qc_review_slices() if slice_index in viable
        ]
        for slice_index in self._review_targets_from_ranges(qc_review_slices):
            self._add_nearby_slices(picked, viable, slice_index, radius=1)

        return sorted(picked)

    def _suggest_annotation_slice(self) -> tuple[Optional[int], str]:
        target_slices = self._target_annotation_slices()
        if not target_slices:
            return None, "No annotation target slice is available yet."

        accepted = sorted(slice_index for slice_index in self.shell_cut_rows_by_slice if slice_index in target_slices)
        first_slice = target_slices[0]
        last_slice = target_slices[-1]
        if not accepted:
            return first_slice, "Start the suggested set from the first region slice."

        if first_slice not in self.shell_cut_rows_by_slice:
            return first_slice, "Add the first region slice in the suggested set."
        if last_slice not in self.shell_cut_rows_by_slice:
            return last_slice, "Add the last region slice in the suggested set."

        best_target = None
        best_gap = -1
        best_pair = None
        for left, right in zip(accepted[:-1], accepted[1:]):
            between = [slice_index for slice_index in target_slices if left < slice_index < right]
            candidates = [slice_index for slice_index in between if slice_index not in self.shell_cut_rows_by_slice]
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

        missing = [slice_index for slice_index in target_slices if slice_index not in self.shell_cut_rows_by_slice]
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
            f"Smart suggested set active: {len(self.annotation_target_slices)} slices. "
            f"Now showing slice {target}. {reason}"
        )
        self._update_annotation_progress()
        self.start_annotation_preview_cache_warmup(target)

    def skip_current_annotation_slice(self) -> None:
        if not self._annotation_shortcuts_active():
            return
        if self.annotation_mask_data is None:
            return
        slice_index = int(self.annotate_slice.value())
        self.annotation_skipped_slices.add(slice_index)
        if self.annotation_pick_mode_kind == "shell_cut":
            self.shell_cut_landmarks_by_slice.pop(slice_index, None)
            self.shell_cut_rows_by_slice.pop(slice_index, None)
        self.annotation_rows.pop(slice_index, None)
        self.annotation_landmarks_by_slice.pop(slice_index, None)
        self.annotation_path_choices_by_slice.pop(slice_index, None)
        self.annotation_boundary_cache.pop(slice_index, None)
        self.slice_canvas.landmarks = {}
        self._set_annotation_path_widgets("auto", "auto")
        self._set_next_annotation_mode()
        self.refresh_annotation_preview()
        self._update_annotation_status()
        if self.annotation_pick_mode_kind == "shell_cut":
            self._autosave_shell_cut_annotations()
        autosave_path = self._autosave_annotation_rows()
        saved = f" Autosaved: {autosave_path}" if autosave_path is not None else ""
        self.append_log(f"Skipped slice {slice_index}: unusable contour.{saved}\n")

        if self._annotation_target_set_complete():
            self.finish_annotation_and_run_build()
            return
        self.annotate_status.setText(f"Skipped slice {slice_index}: unusable contour.{saved}")
        self._go_to_next_annotation_slice()

    def mark_current_annotation_cap(self, surface_mode: str) -> None:
        if not self._annotation_shortcuts_active():
            return
        if self.annotation_mask_data is None:
            return
        contour = self.slice_canvas.selected_contour()
        if contour is None:
            QMessageBox.warning(self, "No contour", "No usable contour is available on this slice.")
            return

        mode = _core().normalize_surface_mode(surface_mode)
        if mode == "normal":
            return
        slice_index = int(self.annotate_slice.value())
        row = self._annotation_row_from_cap(contour, mode, note=f"{mode}_interactive")
        if self.annotation_pick_mode_kind == "shell_cut":
            shell_row = self._shell_cut_row_from_cap(contour, mode, note=f"{mode}_manual_2d")
            self.shell_cut_rows_by_slice[slice_index] = shell_row
            self.shell_cut_landmarks_by_slice.pop(slice_index, None)
        self.annotation_rows[slice_index] = row
        self.annotation_landmarks_by_slice.pop(slice_index, None)
        self.annotation_path_choices_by_slice[slice_index] = {
            "outer_path": row["outer_path"],
            "inner_path": row["inner_path"],
        }
        self.annotation_skipped_slices.discard(slice_index)
        self.annotation_boundary_cache.pop(slice_index, None)
        self.slice_canvas.landmarks = {}
        self._set_annotation_path_widgets("auto", "auto")
        self._set_next_annotation_mode()
        self.refresh_annotation_preview()
        self._update_annotation_status()
        shell_autosave_path = (
            self._autosave_shell_cut_annotations()
            if self.annotation_pick_mode_kind == "shell_cut"
            else None
        )
        autosave_path = self._autosave_annotation_rows()
        saved = f" Autosaved: {autosave_path}" if autosave_path is not None else ""
        if shell_autosave_path is not None:
            saved += f" Shell-cut autosaved: {shell_autosave_path}"
        label = self._surface_mode_label(mode)
        self.append_log(f"Marked slice {slice_index} as {label}.{saved}\n")

        if self._annotation_target_set_complete():
            self.finish_annotation_and_run_build()
            return
        self.annotate_status.setText(f"Marked slice {slice_index} as {label}.{saved}")
        self._go_to_next_annotation_slice()

    def undo_annotation_point(self) -> None:
        if hasattr(self, "surface_preview_canvas") and self.surface_preview_canvas.shell_mesh is not None:
            self.undo_3d_annotation_action()
            return
        if not self._annotation_shortcuts_active():
            return
        order = SliceCanvas.SHELL_CUT_ORDER if self.annotation_pick_mode_kind == "shell_cut" else SliceCanvas.LANDMARK_ORDER
        for name in reversed(order):
            if name in self.slice_canvas.landmarks:
                self.slice_canvas.landmarks.pop(name, None)
                slice_index = int(self.annotate_slice.value())
                if self.annotation_pick_mode_kind == "shell_cut":
                    self.shell_cut_landmarks_by_slice[slice_index] = dict(self.slice_canvas.landmarks)
                else:
                    self.annotation_landmarks_by_slice[slice_index] = dict(self.slice_canvas.landmarks)
                self._set_next_annotation_mode()
                self._update_annotation_status()
                self.slice_canvas.update()
                return

    def navigate_annotation_history(self, direction: int) -> None:
        if not self._annotation_shortcuts_active():
            return
        accepted_slices = (
            sorted(self.shell_cut_rows_by_slice)
            if self.annotation_pick_mode_kind == "shell_cut"
            else sorted(self.annotation_rows)
        )
        if not accepted_slices:
            if self.annotation_pick_mode_kind == "shell_cut":
                self.annotate_status.setText("No accepted shell-cut slices yet.")
            else:
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
        if self.annotation_pick_mode_kind == "shell_cut":
            self.annotate_status.setText(f"Showing accepted shell-cut slice {target}. Click a point to move it.")
        else:
            self.annotate_status.setText(f"Showing accepted slice {target}. Click a point to move it.")

    def accept_annotation_slice_and_advance(self) -> None:
        if hasattr(self, "surface_preview_canvas") and self.surface_preview_canvas.shell_mesh is not None:
            if self.surface_preview_canvas.can_build_3d_surfaces():
                self.run_3d_surface_build()
            else:
                self.annotate_status.setText(
                    "Close at least one 3D cut curve and select a surface patch before building."
                )
            return
        if not self._annotation_shortcuts_active():
            return
        if self.accept_annotation_slice(show_success=False):
            if self.annotation_pick_mode_kind == "shell_cut":
                if self._annotation_target_set_complete():
                    self.finish_annotation_and_run_build()
                    return
                self._go_to_next_annotation_slice()
                return
            if self.review_mode_active:
                self.review_checked_slices.add(int(self.annotate_slice.value()))
                self._update_annotation_progress()
                self._advance_review_queue()
                return
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
        return all(slice_index in self.shell_cut_rows_by_slice for slice_index in target_slices)

    def _auto_build_output_dir(self) -> Path:
        output_text = self.annotate_output.text().strip()
        if output_text:
            return Path(output_text).expanduser()
        if self.annotation_mask_path is not None and not self.annotation_mask_is_temporary:
            return Path(self.annotation_mask_path).expanduser().parent / "laminar_boundary_builder_output"
        return Path.home() / "Desktop" / "laminar_boundary_builder_output"

    def finish_annotation_and_run_build(self) -> None:
        target_slices = self._target_annotation_slices()
        missing = [slice_index for slice_index in target_slices if slice_index not in self.shell_cut_rows_by_slice]
        if missing:
            self.annotate_slice.setValue(missing[0])
            self.annotate_status.setText(f"Still missing suggested slice {missing[0]}.")
            return
        try:
            output_dir = self._auto_build_output_dir()
            if not self.annotate_output.text().strip():
                self.annotate_output.set_text(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self._sync_derived_annotation_rows_from_shell_cut()
            csv_path = output_dir / "manual_landmarks_interactive.csv"
            self._write_annotation_rows_csv(csv_path)
            shell_cut_path = output_dir / "shell_cut_annotations.json"
            self._write_shell_cut_annotations_json(shell_cut_path)
            self._sync_build_from_annotation(csv_path, output_dir)
            self.annotation_picking_active = False
            self.annotation_settings_expanded = True
            self.annotate_settings_button.hide()
            self._set_annotation_parameter_widgets_enabled(True)
            self._apply_annotation_settings_panel_state()
            self.tabs.setCurrentIndex(1)
            self.append_log(
                f"Suggested annotation set complete: {len(target_slices)} slices.\n"
                f"Saved shell-cut annotations: {shell_cut_path}\n"
                f"Saved derived build CSV: {csv_path}\n"
                "Build settings are ready for review.\n"
            )
            self.append_log(f"Saved shell-cut annotations: {shell_cut_path}\n")
            reply = QMessageBox.question(
                self,
                "Annotation saved",
                "Suggested annotation set is complete and the Build step is ready.\n\n"
                "Run surface extraction now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.run_surface_build()
        except Exception as exc:
            self._show_exception_dialog("Auto build failed", exc)

    def _go_to_next_annotation_slice(self) -> None:
        current = int(self.annotate_slice.value())
        target_slices = self._target_annotation_slices()
        next_slice = None
        for slice_index in target_slices:
            if slice_index > current:
                next_slice = slice_index
                break
        if next_slice is None:
            missing = [slice_index for slice_index in target_slices if slice_index not in self.shell_cut_rows_by_slice]
            if missing:
                next_slice = missing[0]
            elif self.annotation_target_slices:
                self.annotate_status.setText("Reached the last suggested slice. Press Enter to build.")
                return
            else:
                self.annotate_status.setText("Reached the last region slice. Save shell-cut annotations when you are done.")
                return

        self.annotate_slice.setValue(next_slice)
        self.slice_canvas.setFocus(Qt.OtherFocusReason)
        self.start_annotation_preview_cache_warmup(next_slice)

    @staticmethod
    def _extract_annotation_contours_from_mask(
        mask_data,
        slice_index: int,
        slice_axis_int: int,
        min_area: float,
        keep_all: bool,
    ) -> list:
        if mask_data is None:
            return []
        core = _core()
        mask_2d = core._take_slice(
            mask_data,
            slice_index,
            slice_axis_int,
        )
        raw_contours = core._find_mask_contours(mask_2d)
        contours = []
        for raw in raw_contours:
            area = core._polygon_area(raw)
            if area < float(min_area):
                continue
            points = core._plane_to_volume_points(raw, slice_index, slice_axis_int)
            contours.append(
                core.Contour2D(
                    slice_index=slice_index,
                    contour_id=len(contours),
                    points=points,
                    area=area,
                    length=core._polyline_length(points, closed=True),
                )
            )
        if contours and not keep_all:
            biggest = max(contours, key=lambda contour: contour.area)
            biggest.contour_id = 0
            return [biggest]
        return contours

    def _extract_contours_for_annotation_slice(self, slice_index: int) -> list:
        return self._extract_annotation_contours_from_mask(
            self.annotation_mask_data,
            slice_index,
            self.annotation_slice_axis_int,
            float(self.annotate_min_area.value()),
            self.annotate_keep_all.isChecked(),
        )

    @staticmethod
    def _build_annotation_reference_contours_from_mask(
        mask_data,
        region_slices: List[int],
        slice_axis_int: int,
        min_area: float,
        max_slices: int = 56,
        points_per_contour: int = 96,
    ) -> list:
        if mask_data is None or not region_slices:
            return []
        core = _core()
        np = _numpy()
        region_slices = list(region_slices)
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
        min_area = max(1.0, float(min_area))
        for slice_index in sample_slices:
            contours = LaminarBoundaryWindow._extract_annotation_contours_from_mask(
                mask_data,
                slice_index,
                slice_axis_int,
                min_area,
                keep_all=False,
            )
            if not contours:
                continue
            points = core._normalize_contour(contours[0].points)
            if len(points) < 2:
                continue
            closed = np.vstack([points, points[:1]])
            if len(closed) > points_per_contour:
                closed = core.resample_polyline(closed, points_per_contour + 1)
            reference_contours.append(closed)
        return reference_contours

    def _build_annotation_reference_contours(
        self,
        max_slices: int = 56,
        points_per_contour: int = 96,
    ) -> list:
        return self._build_annotation_reference_contours_from_mask(
            self.annotation_mask_data,
            list(self.annotation_region_slices),
            self.annotation_slice_axis_int,
            float(self.annotate_min_area.value()),
            max_slices=max_slices,
            points_per_contour=points_per_contour,
        )

    def _refresh_annotation_reference_contours(self) -> None:
        self.annotation_reference_contours = self._build_annotation_reference_contours()
        if hasattr(self, "surface_preview_canvas"):
            self.surface_preview_canvas.set_reference_contours(
                self.annotation_reference_contours,
                self.annotation_slice_axis_int,
            )

    def _annotation_warmup_slices(self, current_slice: int) -> List[int]:
        warmup_slices = [int(current_slice)]
        for slice_index in self._build_suggested_annotation_set():
            if slice_index not in warmup_slices:
                warmup_slices.append(slice_index)
        return warmup_slices[:24]

    def start_annotation_preview_cache_warmup(self, current_slice: int) -> None:
        if self.annotation_mask_data is None or self.annotation_preview_thread is not None:
            return
        if self._has_3d_annotation_shell():
            return

        self.annotation_preview_request_id += 1
        request_id = self.annotation_preview_request_id
        mask_data = self.annotation_mask_data
        region_slices = list(self.annotation_region_slices)
        slice_axis_int = int(self.annotation_slice_axis_int)
        min_area = float(self.annotate_min_area.value())
        keep_all = self.annotate_keep_all.isChecked()
        cached_slices = set(self.annotation_contours_by_slice)
        warmup_slices = [
            slice_index
            for slice_index in self._annotation_warmup_slices(current_slice)
            if slice_index not in cached_slices
        ]
        existing_reference = list(self.annotation_reference_contours)
        needs_reference = not existing_reference and bool(region_slices)
        if not needs_reference and not warmup_slices:
            return

        def task() -> TaskResult:
            if needs_reference:
                reference_contours = self._build_annotation_reference_contours_from_mask(
                    mask_data,
                    region_slices,
                    slice_axis_int,
                    min_area,
                    max_slices=56,
                    points_per_contour=96,
                )
            else:
                reference_contours = existing_reference
            contours_by_slice = {}
            for slice_index in warmup_slices:
                contours_by_slice[slice_index] = self._extract_annotation_contours_from_mask(
                    mask_data,
                    slice_index,
                    slice_axis_int,
                    min_area,
                    keep_all,
                )
            return TaskResult(
                "Annotation preview cache ready",
                (
                    f"Prepared target reference from {len(reference_contours)} slices and "
                    f"cached {len(contours_by_slice)} annotation slices."
                ),
                payload=AnnotationPreviewCacheData(
                    request_id=request_id,
                    reference_contours=reference_contours,
                    contours_by_slice=contours_by_slice,
                ),
            )

        if hasattr(self, "surface_preview_canvas") and not self.annotation_reference_contours:
            self.surface_preview_canvas.set_boundaries(
                [],
                slice_axis=self.annotation_slice_axis_int,
                message="Preparing target region reference...",
            )
        self.annotation_preview_thread = QThread()
        self.annotation_preview_worker = Worker(task)
        self.annotation_preview_worker.moveToThread(self.annotation_preview_thread)
        self.annotation_preview_thread.started.connect(self.annotation_preview_worker.run)
        self.annotation_preview_worker.finished.connect(self.annotation_preview_cache_finished)
        self.annotation_preview_worker.failed.connect(self.annotation_preview_cache_failed)
        self.annotation_preview_worker.finished.connect(self.annotation_preview_thread.quit)
        self.annotation_preview_worker.failed.connect(self.annotation_preview_thread.quit)
        self.annotation_preview_thread.finished.connect(self.annotation_preview_thread.deleteLater)
        self.annotation_preview_thread.finished.connect(self.clear_annotation_preview_thread)
        self.annotation_preview_thread.start()

    def annotation_preview_cache_finished(self, result: TaskResult) -> None:
        payload = result.payload
        if not isinstance(payload, AnnotationPreviewCacheData):
            return
        if payload.request_id != self.annotation_preview_request_id:
            return
        for slice_index, contours in payload.contours_by_slice.items():
            self.annotation_contours_by_slice.setdefault(int(slice_index), contours)
        if self.annotation_mask_data is not None:
            self._prune_annotation_contour_cache(int(self.annotate_slice.value()))
        self.annotation_reference_contours = list(payload.reference_contours)
        if hasattr(self, "surface_preview_canvas"):
            self.surface_preview_canvas.set_reference_contours(
                self.annotation_reference_contours,
                self.annotation_slice_axis_int,
            )
        self.refresh_annotation_preview()
        self.append_log(result.message + "\n")

    def annotation_preview_cache_failed(self, trace: str) -> None:
        self.append_log("\nPreview cache failed:\n" + trace)

    def clear_annotation_preview_thread(self) -> None:
        self.annotation_preview_thread = None
        self.annotation_preview_worker = None

    def _uses_atlas_extraction(self) -> bool:
        region = self.annotate_region.text().strip()
        mask_text = self.annotate_mask.text().strip()
        return bool(region) and (not mask_text or self._mask_text_is_current_temporary_mask())

    def _current_annotation_extraction_signature(self, atlas_path: Optional[Path] = None):
        core = _core()
        if atlas_path is None:
            atlas_text = self.annotate_atlas.text() if self.annotate_custom_atlas.isChecked() else ""
            atlas_path = core.resolve_annotation_path(atlas_text or None)
        template_text = self.annotate_template.text().strip()
        return (
            str(Path(atlas_path).expanduser().resolve()),
            self.annotate_region.text().strip(),
            bool(self.annotate_include_children.isChecked()),
            self.annotate_hemisphere.currentText(),
            str(Path(template_text).expanduser().resolve()) if template_text else "",
        )

    def _can_reuse_temporary_annotation_mask(self) -> bool:
        if self.annotation_extraction_signature is None:
            return False
        if not self._mask_text_is_current_temporary_mask():
            return False
        if self.annotation_mask_path is None or not Path(self.annotation_mask_path).exists():
            return False
        try:
            return self.annotation_extraction_signature == self._current_annotation_extraction_signature()
        except Exception:
            return False

    def _cleanup_temporary_mask(self) -> None:
        if self.temporary_mask_dir is not None:
            self.temporary_mask_dir.cleanup()
            self.temporary_mask_dir = None

    @staticmethod
    def _preview_shell_mask(prepared_mask, max_dim: int = 260):
        np = _numpy()
        mask = np.asarray(prepared_mask, dtype=bool)
        if mask.ndim != 3 or max(mask.shape) <= max_dim:
            return mask, np.ones(3, dtype=float)
        stride = int(math.ceil(max(mask.shape) / float(max_dim)))
        stride = max(1, stride)
        pad_width = [(0, (-dim) % stride) for dim in mask.shape]
        padded = np.pad(mask, pad_width, mode="constant", constant_values=False)
        reduced = padded.reshape(
            padded.shape[0] // stride,
            stride,
            padded.shape[1] // stride,
            stride,
            padded.shape[2] // stride,
            stride,
        ).max(axis=(1, 3, 5))
        return reduced, np.asarray([stride, stride, stride], dtype=float)

    def _prepare_annotation_load_data(
        self,
        mask_data,
        mask_path: str | Path,
        template_data,
        temporary: bool,
        slice_axis_text: str,
        warnings: Optional[List[str]] = None,
        progress: Optional[Callable[[str], None]] = None,
    ) -> AnnotationLoadData:
        emit = progress or (lambda _message: None)
        np = _numpy()
        core = _core()

        emit("Preparing mask array...")
        mask_array = np.asarray(mask_data)
        if mask_array.dtype == np.bool_:
            prepared_mask = mask_array
        else:
            prepared_mask = mask_array > 0

        slice_axis_int = core._slice_axis_to_int(slice_axis_text)
        emit("Counting non-empty slices...")
        slice_counts = prepared_mask.sum(
            axis=tuple(axis for axis in range(3) if axis != slice_axis_int)
        )
        slice_counts = np.asarray(slice_counts, dtype=np.int64)
        region_slices = [int(index) for index in np.flatnonzero(slice_counts)]
        load_warnings = list(warnings or [])
        shell_mesh = None
        shell_errors = []
        preview_mask, preview_scale = self._preview_shell_mask(prepared_mask)
        for backend in (core.SHELL_BACKEND_MARCHING_CUBES, core.SHELL_BACKEND_VOXEL):
            try:
                scale_label = (
                    f", preview downsample x{int(preview_scale[0])}"
                    if int(preview_scale[0]) > 1
                    else ""
                )
                shell_label = (
                    "smooth triangle shell"
                    if backend == core.SHELL_BACKEND_MARCHING_CUBES
                    else "voxel shell"
                )
                emit(f"Building fast 3D shell preview ({shell_label}{scale_label})...")
                shell_mesh = core.build_shell_mesh(
                    preview_mask,
                    shell_backend=backend,
                    max_surface_quads=1_500_000,
                )
                if int(preview_scale[0]) > 1:
                    shell_mesh.vertices = (
                        np.asarray(shell_mesh.vertices, dtype=float)
                        * preview_scale.reshape(1, 3)
                    )
                    shell_mesh.backend = f"{backend}_preview_x{int(preview_scale[0])}"
                break
            except Exception as exc:
                shell_errors.append(f"{backend}: {exc}")
        if shell_mesh is None and shell_errors:
            load_warnings.append("3D shell preview failed. " + " | ".join(shell_errors))
        emit(f"Mask ready with {len(region_slices)} non-empty slices.")
        return AnnotationLoadData(
            mask_data=prepared_mask,
            mask_path=Path(mask_path).expanduser(),
            template_data=template_data,
            temporary=temporary,
            slice_axis_int=slice_axis_int,
            slice_counts=slice_counts,
            region_slices=region_slices,
            shell_mesh=shell_mesh,
            warnings=load_warnings,
        )

    def _finish_annotation_load(
        self,
        load_data: AnnotationLoadData,
    ) -> None:
        self.annotation_mask_data = load_data.mask_data
        self.annotation_template_data = load_data.template_data
        self.annotation_slice_axis_int = load_data.slice_axis_int
        self.annotation_mask_path = load_data.mask_path
        self.annotation_mask_is_temporary = load_data.temporary
        self.annotate_mask.set_text(self.annotation_mask_path)
        self.annotation_pick_mode_kind = "shell_cut"
        if hasattr(self, "annotate_pick_mode"):
            self.annotate_pick_mode.setCurrentText("Shell Cut Boundary")

        max_slice = self.annotation_mask_data.shape[self.annotation_slice_axis_int] - 1
        self.annotation_slice_counts = load_data.slice_counts
        self.annotation_region_slices = list(load_data.region_slices)
        self.annotation_landmarks_by_slice.clear()
        self.annotation_rows.clear()
        self.shell_cut_landmarks_by_slice.clear()
        self.shell_cut_rows_by_slice.clear()
        self.annotation_path_choices_by_slice.clear()
        self.annotation_skipped_slices.clear()
        self.annotation_contours_by_slice.clear()
        self.annotation_boundary_cache.clear()
        self.annotation_reference_contours = []
        self.review_mode_active = False
        self.review_queue_slices = []
        self.review_checked_slices = set()
        self.review_source_build_dir = None
        self.review_round_csv_path = None
        self._apply_review_mode_ui(False)
        self.annotation_target_slices = self._build_suggested_annotation_set()
        if self.annotation_target_slices:
            initial_slice = int(self.annotation_target_slices[0])
        elif self.annotation_region_slices:
            middle = self.annotation_region_slices[len(self.annotation_region_slices) // 2]
            initial_slice = int(middle)
        else:
            initial_slice = 0
        self.annotation_preview_request_id += 1
        self.annotate_slice.blockSignals(True)
        self.annotate_slider.blockSignals(True)
        self.annotate_slice.setRange(0, max_slice)
        self.annotate_slider.setRange(0, max_slice)
        self.annotate_slice.setValue(initial_slice)
        self.annotate_slider.setValue(initial_slice)
        self.annotate_slider.blockSignals(False)
        self.annotate_slice.blockSignals(False)
        if hasattr(self, "surface_preview_canvas"):
            self.surface_preview_canvas.set_reference_contours([], self.annotation_slice_axis_int)
            self.surface_preview_canvas.set_shell_mesh(load_data.shell_mesh)
        self._set_annotation_path_widgets("auto", "auto")
        if load_data.shell_mesh is None:
            self.refresh_annotation_slice()
        else:
            self._update_annotation_status()
        self._update_annotation_readiness()
        self.enter_annotation_picking_mode()
        self.start_annotation_preview_cache_warmup(initial_slice)

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
        slice_axis_text = self.annotate_slice_axis.currentText()
        self.pending_annotation_extraction_signature = self._current_annotation_extraction_signature(atlas_path)

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
            load_data = self._prepare_annotation_load_data(
                mask_data=extraction.mask,
                mask_path=extraction.mask_path,
                template_data=extraction.template,
                temporary=True,
                slice_axis_text=slice_axis_text,
                warnings=extraction.warnings,
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
            return TaskResult("Mask extraction finished", "\n".join(lines), payload=(extraction, load_data))

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
        extraction, load_data = result.payload
        try:
            self._finish_annotation_load(load_data)
            self.annotation_extraction_signature = self.pending_annotation_extraction_signature
            self.pending_annotation_extraction_signature = None
            warning_text = "\n".join(load_data.warnings)
            if warning_text:
                QMessageBox.warning(self, "Mask extracted with warning", warning_text)
            self.append_log(f"Loaded temporary annotation mask: {extraction.mask_path}\n")
        except Exception as exc:
            self._set_status("Failed", "failed")
            self._show_exception_dialog("Load failed", exc)

    def annotation_mask_extraction_failed(self, trace: str) -> None:
        self._close_progress_dialog()
        self._set_status("Failed", "failed")
        self._cleanup_temporary_mask()
        self.pending_annotation_extraction_signature = None
        self.append_log("\n" + trace)
        self._show_error_dialog("Mask extraction failed", trace)

    def start_annotation_mask_load(self, mask_path: Path, temporary: bool = False) -> None:
        if self.thread is not None:
            QMessageBox.warning(self, "Task running", "Please wait for the current task to finish.")
            return

        template_text = self.annotate_template.text()
        template_path = self._require_path("Template image", template_text) if template_text else None
        slice_axis_text = self.annotate_slice_axis.currentText()
        old_temp_dir = self.temporary_mask_dir
        old_temp_path = self.annotation_mask_path if self.annotation_mask_is_temporary else None

        def task() -> TaskResult:
            core = _core()
            warnings: List[str] = []
            print("Loading mask volume...")
            try:
                mask_data = core.load_volume(mask_path).data
            except Exception as exc:
                raise RuntimeError(f"Could not read Mask:\n{mask_path}\n\n{exc}") from exc

            template_data = None
            if template_path is not None:
                print("Loading template image...")
                try:
                    template = core.load_volume(template_path).data
                except Exception as exc:
                    raise RuntimeError(f"Could not read Template image:\n{template_path}\n\n{exc}") from exc
                if template.shape == mask_data.shape:
                    template_data = template
                else:
                    warnings.append(
                        "Template shape does not match the mask, so only the mask will be shown."
                    )

            load_data = self._prepare_annotation_load_data(
                mask_data=mask_data,
                mask_path=mask_path,
                template_data=template_data,
                temporary=temporary,
                slice_axis_text=slice_axis_text,
                warnings=warnings,
                progress=print,
            )
            message = (
                "Annotation mask loaded.\n"
                f"mask: {mask_path}\n"
                f"non_empty_slices: {len(load_data.region_slices)}"
            )
            return TaskResult("Annotation mask loaded", message, payload=load_data)

        self.append_log("\n--- Annotation mask load started ---\n")
        self._set_status("Running: loading mask", "running")
        self.progress_dialog = QProgressDialog("Loading annotation mask...", "", 0, 0, self)
        self.progress_dialog.setWindowTitle("Loading mask")
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
        self.worker.finished.connect(
            lambda result: self.annotation_mask_load_finished(result, old_temp_dir, old_temp_path)
        )
        self.worker.failed.connect(self.annotation_mask_load_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.clear_thread)
        self.thread.start()

    def annotation_mask_load_finished(
        self,
        result: TaskResult,
        old_temp_dir: Optional[tempfile.TemporaryDirectory],
        old_temp_path: Optional[Path],
    ) -> None:
        self._close_progress_dialog()
        self._set_status("Ready", "ready")
        load_data = result.payload
        try:
            self._finish_annotation_load(load_data)
            if not load_data.temporary:
                self.annotation_extraction_signature = None
            if old_temp_dir is not None and load_data.mask_path != old_temp_path:
                old_temp_dir.cleanup()
                if old_temp_dir is self.temporary_mask_dir:
                    self.temporary_mask_dir = None
            warning_text = "\n".join(load_data.warnings)
            if warning_text:
                QMessageBox.warning(self, "Mask loaded with warning", warning_text)
            self.append_log(f"\n{result.message}\n")
            self._finish_pending_review_after_mask_load(load_data)
            self._finish_pending_previous_csv_after_mask_load(load_data.mask_path)
        except Exception as exc:
            self._set_status("Failed", "failed")
            self._show_exception_dialog("Load failed", exc)

    def annotation_mask_load_failed(self, trace: str) -> None:
        self._close_progress_dialog()
        self._set_status("Failed", "failed")
        self.append_log("\n" + trace)
        self._show_error_dialog("Load failed", trace)

    def load_annotation_data(self) -> None:
        try:
            if self._reload_would_clear_3d_annotations():
                reply = QMessageBox.question(
                    self,
                    "Reload source?",
                    "Reloading the source will clear the current 3D curves and selected patches.\n\n"
                    "Continue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return
            if self._can_reuse_temporary_annotation_mask():
                self.append_log("\n--- Reusing current extracted mask ---\n")
                self.start_annotation_mask_load(Path(self.annotation_mask_path), temporary=True)
                return
            if self._uses_atlas_extraction():
                self.start_annotation_mask_extraction()
                return
            if not self.annotate_mask.text().strip():
                QMessageBox.warning(
                    self,
                    "Choose input source",
                    "Choose a Brain region or select an existing Mask before starting.",
                )
                return

            mask_path = self._require_path("Mask", self.annotate_mask.text())
            self.start_annotation_mask_load(mask_path)
        except Exception as exc:
            self._show_exception_dialog("Load failed", exc)

    def _reload_would_clear_3d_annotations(self) -> bool:
        if not self.annotation_picking_active:
            return False
        if not hasattr(self, "surface_preview_canvas") or self.surface_preview_canvas.shell_mesh is None:
            return False
        curve_count, point_count, patch_count = self.surface_preview_canvas.annotation_counts()
        return bool(curve_count or point_count or patch_count)

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
        protected = set(self.annotation_rows) | set(self.shell_cut_rows_by_slice)
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
        if self.annotation_pick_mode_kind == "shell_cut":
            accepted_row = self.shell_cut_rows_by_slice.get(slice_index)
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

        if self.annotation_pick_mode_kind == "shell_cut":
            landmarks = self.shell_cut_landmarks_by_slice.get(slice_index, {})
        else:
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
            "surface_mode": "normal",
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

    def _boundary_landmarks_from_shell_cut(self, shell_landmarks: Dict[str, int]) -> Dict[str, int]:
        return {
            "outer_start": int(shell_landmarks["outer_cut_A"]),
            "outer_end": int(shell_landmarks["outer_cut_B"]),
            "inner_start": int(shell_landmarks["inner_cut_A"]),
            "inner_end": int(shell_landmarks["inner_cut_B"]),
        }

    def _annotation_row_from_shell_cut_landmarks(
        self,
        contour,
        shell_landmarks: Dict[str, int],
        note: str = "derived_from_shell_cut",
        outer_path: Optional[str] = None,
        inner_path: Optional[str] = None,
    ) -> Dict[str, str]:
        return self._annotation_row_from_landmarks(
            contour,
            self._boundary_landmarks_from_shell_cut(shell_landmarks),
            note=note,
            outer_path=outer_path if outer_path is not None else self.annotate_outer_path.currentText(),
            inner_path=inner_path if inner_path is not None else self.annotate_inner_path.currentText(),
        )

    def _annotation_row_from_cap(
        self,
        contour,
        surface_mode: str,
        note: Optional[str] = None,
    ) -> Dict[str, str]:
        mode = _core().normalize_surface_mode(surface_mode)
        row = {field: "" for field in self._annotation_csv_fieldnames()}
        row.update(
            {
                "slice_index": str(contour.slice_index),
                "contour_id": str(contour.contour_id),
                "surface_mode": mode,
                "outer_path": "whole" if mode == "outer_only" else "",
                "inner_path": "whole" if mode == "inner_only" else "",
                "note": note or mode,
            }
        )
        return row

    def _annotation_csv_fieldnames(self) -> List[str]:
        return [
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

    def _write_annotation_rows_csv(self, csv_path: Path) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._annotation_csv_fieldnames())
            writer.writeheader()
            for slice_index in sorted(self.annotation_rows):
                writer.writerow(self.annotation_rows[slice_index])

    def _shell_cut_row_from_landmarks(
        self,
        contour,
        landmarks: Dict[str, int],
        note: str = "manual_2d",
    ) -> Dict:
        core = _core()
        points = core._normalize_contour(contour.points)
        point_rows = {}
        for name in SliceCanvas.SHELL_CUT_ORDER:
            index = int(landmarks[name])
            point = points[index]
            point_rows[name] = {
                "index": index,
                "point": [float(point[0]), float(point[1]), float(point[2])],
            }
        return {
            "slice_index": int(contour.slice_index),
            "contour_id": int(contour.contour_id),
            "source": "manual_2d",
            "note": note,
            "outer_path": self._normalize_annotation_path_choice(self.annotate_outer_path.currentText()),
            "inner_path": self._normalize_annotation_path_choice(self.annotate_inner_path.currentText()),
            "points": point_rows,
        }

    def _shell_cut_row_from_cap(
        self,
        contour,
        surface_mode: str,
        note: str = "manual_2d_cap",
    ) -> Dict:
        mode = _core().normalize_surface_mode(surface_mode)
        return {
            "slice_index": int(contour.slice_index),
            "contour_id": int(contour.contour_id),
            "source": "manual_2d",
            "surface_mode": mode,
            "note": note,
            "outer_path": "whole" if mode == "outer_only" else "",
            "inner_path": "whole" if mode == "inner_only" else "",
            "points": {},
        }

    def _sync_derived_annotation_rows_from_shell_cut(self) -> None:
        self.annotation_rows.clear()
        self.annotation_landmarks_by_slice.clear()
        self.annotation_path_choices_by_slice.clear()
        self.annotation_boundary_cache.clear()
        core = _core()
        for slice_index in sorted(self.shell_cut_rows_by_slice):
            row = self.shell_cut_rows_by_slice[slice_index]
            contours = self.annotation_contours_by_slice.get(slice_index)
            if contours is None:
                contours = self._extract_contours_for_annotation_slice(slice_index)
                self.annotation_contours_by_slice[slice_index] = contours
                self._prune_annotation_contour_cache(slice_index)
            contour = core._select_contour_for_row(row, contours)
            surface_mode = core.normalize_surface_mode(row.get("surface_mode"))
            if surface_mode != "normal":
                self.annotation_rows[slice_index] = self._annotation_row_from_cap(
                    contour,
                    surface_mode,
                    note=row.get("note") or "derived_from_shell_cut_cap",
                )
                self.annotation_landmarks_by_slice.pop(slice_index, None)
                self.annotation_path_choices_by_slice[slice_index] = {
                    "outer_path": row.get("outer_path", ""),
                    "inner_path": row.get("inner_path", ""),
                }
                continue
            points = row.get("points", {})
            shell_landmarks = {
                name: int(points[name]["index"])
                for name in SliceCanvas.SHELL_CUT_ORDER
                if name in points and isinstance(points[name], dict) and points[name].get("index") is not None
            }
            if not all(name in shell_landmarks for name in SliceCanvas.SHELL_CUT_ORDER):
                continue
            boundary_landmarks = self._boundary_landmarks_from_shell_cut(shell_landmarks)
            outer_path = self._normalize_annotation_path_choice(row.get("outer_path"))
            inner_path = self._normalize_annotation_path_choice(row.get("inner_path"))
            self.annotation_rows[slice_index] = self._annotation_row_from_shell_cut_landmarks(
                contour,
                shell_landmarks,
                note="derived_from_shell_cut",
                outer_path=outer_path,
                inner_path=inner_path,
            )
            self.annotation_landmarks_by_slice[slice_index] = boundary_landmarks
            self.annotation_path_choices_by_slice[slice_index] = {
                "outer_path": outer_path,
                "inner_path": inner_path,
            }

    def _shell_cut_annotations_payload(self) -> Dict:
        return {
            "schema": "laminar_boundary_builder.shell_cut_annotations.v1",
            "annotation_type": "manual_2d_shell_cut_boundary",
            "slice_axis": int(self.annotation_slice_axis_int),
            "mask_path": str(self.annotation_mask_path) if self.annotation_mask_path else None,
            "rows": [
                self.shell_cut_rows_by_slice[slice_index]
                for slice_index in sorted(self.shell_cut_rows_by_slice)
            ],
        }

    def _write_shell_cut_annotations_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._shell_cut_annotations_payload()
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _autosave_shell_cut_annotations(self) -> Optional[Path]:
        if not self.shell_cut_rows_by_slice:
            return None
        saved_path = None
        try:
            self._write_shell_cut_annotations_json(self.shell_cut_autosave_path)
            saved_path = self.shell_cut_autosave_path
        except Exception as exc:
            self.append_log(f"Shell-cut autosave failed: {exc}\n")

        output_text = self.annotate_output.text().strip()
        if output_text:
            try:
                output_path = Path(output_text).expanduser() / "shell_cut_annotations.json"
                self._write_shell_cut_annotations_json(output_path)
                saved_path = output_path
            except Exception as exc:
                self.append_log(f"Output shell-cut autosave failed: {exc}\n")
        return saved_path

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
        surface_mode = core.normalize_surface_mode(row.get("surface_mode"))
        if surface_mode != "normal":
            loaded_row = self._annotation_row_from_cap(
                contour,
                surface_mode,
                note=row.get("note") or "loaded",
            )
            return slice_index, loaded_row, {}

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
        signature = tuple(str(row.get(key, "")) for key in self._annotation_csv_fieldnames())
        cached = self.annotation_boundary_cache.get(slice_index)
        if cached is not None and cached[0] == signature:
            return cached[1]
        contours = self.annotation_contours_by_slice.get(slice_index)
        if contours is None:
            contours = self._extract_contours_for_annotation_slice(slice_index)
            self.annotation_contours_by_slice[slice_index] = contours
        contour = core._select_contour_for_row(row, contours)
        boundary = core.make_boundary_from_landmark_row(contour, row, resample_points=64)
        self.annotation_boundary_cache[slice_index] = (signature, boundary)
        return boundary

    def _current_annotation_boundary(self):
        contour = self.slice_canvas.selected_contour()
        if contour is None:
            return None
        landmarks = dict(self.slice_canvas.landmarks)
        if self.annotation_pick_mode_kind == "shell_cut":
            if not all(name in landmarks for name in SliceCanvas.SHELL_CUT_ORDER):
                return None
            row = self._annotation_row_from_shell_cut_landmarks(contour, landmarks, note="live_shell_cut")
            return _core().make_boundary_from_landmark_row(contour, row, resample_points=64)
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
        if self._has_3d_annotation_shell():
            self.surface_preview_canvas.update()
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
        current_boundary = None
        for row in self.annotation_rows.values():
            row_slice = int(float(row.get("slice_index", -1)))
            row_mode = _core().normalize_surface_mode(row.get("surface_mode"))
            if row_slice == current_slice and row_mode == "normal" and not current_complete:
                continue
            try:
                boundary = self._annotation_boundary_from_row(row)
                boundaries.append(boundary)
                if row_slice == current_slice:
                    current_boundary = boundary
            except Exception:
                skipped += 1

        if current_boundary is None:
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
            message += f", {len(self.annotation_skipped_slices)} ignored"
        self.surface_preview_canvas.set_boundaries(
            boundaries,
            current_boundary=current_boundary,
            slice_axis=self.annotation_slice_axis_int,
            message=message,
        )

    def on_landmark_changed(self, mode: str) -> None:
        slice_index = int(self.annotate_slice.value())
        if self.annotation_pick_mode_kind == "shell_cut":
            self.shell_cut_landmarks_by_slice[slice_index] = dict(self.slice_canvas.landmarks)
            landmarks = self.shell_cut_landmarks_by_slice[slice_index]
            existing_row = self.shell_cut_rows_by_slice.get(slice_index)
            if (
                existing_row is not None
                and _core().normalize_surface_mode(existing_row.get("surface_mode")) != "normal"
            ):
                self.shell_cut_rows_by_slice.pop(slice_index, None)
                self.annotation_rows.pop(slice_index, None)
                self.annotation_path_choices_by_slice.pop(slice_index, None)
                self.annotation_boundary_cache.pop(slice_index, None)
            is_complete = all(name in landmarks for name in SliceCanvas.SHELL_CUT_ORDER)
            if slice_index in self.shell_cut_rows_by_slice and is_complete:
                contour = self.slice_canvas.selected_contour()
                if contour is not None:
                    row = self._shell_cut_row_from_landmarks(
                        contour,
                        landmarks,
                        note="manual_2d_edit",
                    )
                    outer_path = self._normalize_annotation_path_choice(row.get("outer_path"))
                    inner_path = self._normalize_annotation_path_choice(row.get("inner_path"))
                    self.shell_cut_rows_by_slice[slice_index] = row
                    self.annotation_rows[slice_index] = self._annotation_row_from_shell_cut_landmarks(
                        contour,
                        landmarks,
                        note="derived_from_shell_cut_edit",
                        outer_path=outer_path,
                        inner_path=inner_path,
                    )
                    self.annotation_landmarks_by_slice[slice_index] = self._boundary_landmarks_from_shell_cut(landmarks)
                    self.annotation_path_choices_by_slice[slice_index] = {
                        "outer_path": outer_path,
                        "inner_path": inner_path,
                    }
                    self.annotation_boundary_cache.pop(slice_index, None)
                    autosave_path = self._autosave_shell_cut_annotations()
                    if autosave_path is not None:
                        self.append_log(
                            f"Updated shell-cut slice {slice_index}; autosaved: {autosave_path}\n"
                        )
                    self._autosave_annotation_rows()
            self._set_next_annotation_mode()
            self._update_annotation_status()
            self.refresh_annotation_preview()
            return

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
                self.annotation_boundary_cache.pop(slice_index, None)
                autosave_path = self._autosave_annotation_rows()
                if autosave_path is not None:
                    self.append_log(f"Updated accepted slice {slice_index}; autosaved: {autosave_path}\n")
        self._set_next_annotation_mode()
        self._update_annotation_status()
        self.refresh_annotation_preview()

    def _update_annotation_status(self) -> None:
        if self.annotation_mask_data is None:
            self.annotate_status.setText("No mask loaded")
            self.annotate_progress.setText("Load a mask to see slice count.")
            self.slice_canvas.set_progress_text("")
            self._update_annotation_review_slice_choices()
            return
        if self._has_3d_annotation_shell():
            curve_count, point_count, patch_count = self.surface_preview_canvas.annotation_counts()
            if point_count:
                status = f"3D shell ready. Drawing curve with {point_count} point(s). Click an active point to close."
            elif curve_count and patch_count:
                status = (
                    f"3D shell ready. {curve_count} closed curve(s), "
                    f"{patch_count} selected surface patch(es). Build is ready."
                )
            elif curve_count:
                status = f"3D shell ready. {curve_count} closed curve(s). Hover and click a surface patch to keep."
            else:
                status = "3D shell ready. Click the shaded shell surface to draw the first closed curve."
            self.annotate_status.setText(status)
            self._update_annotation_progress()
            self._update_annotation_review_slice_choices()
            self.on_3d_annotation_changed()
            return
        slice_index = int(self.annotate_slice.value())
        if self.annotation_pick_mode_kind == "shell_cut":
            landmarks = self.shell_cut_landmarks_by_slice.get(slice_index, self.slice_canvas.landmarks)
            missing = [name for name in SliceCanvas.SHELL_CUT_ORDER if name not in landmarks]
            accepted_count = len(self.shell_cut_rows_by_slice)
            contour_text = "no contour" if not self.slice_canvas.contours else f"contour {self.slice_canvas.selected_contour_index}"
            row = self.shell_cut_rows_by_slice.get(slice_index)
            if slice_index in self.annotation_skipped_slices:
                self.annotate_status.setText(
                    f"Shell Cut Boundary, slice {slice_index} is skipped because its contour is not usable. "
                    "Clear Current Slice if you want to annotate it."
                )
            elif row is not None and _core().normalize_surface_mode(row.get("surface_mode")) != "normal":
                label = self._surface_mode_label(row.get("surface_mode", "normal"))
                self.annotate_status.setText(
                    f"Shell Cut Boundary, slice {slice_index}, {contour_text}. Marked as {label}. "
                    "Clear Current Slice if you want four-point shell-cut annotation. "
                    f"Accepted shell-cut slices: {accepted_count}."
                )
            elif missing:
                prefix = "Editing accepted shell-cut slice. " if slice_index in self.shell_cut_rows_by_slice else ""
                self.annotate_status.setText(
                    f"{prefix}Shell Cut Boundary, slice {slice_index}, {contour_text}. "
                    f"Missing: {', '.join(missing)}. Accepted shell-cut slices: {accepted_count}."
                )
            elif self.slice_canvas.shell_cut_overlap_excess() > 0:
                self.annotate_status.setText(
                    f"Shell Cut Boundary, slice {slice_index}, {contour_text}. "
                    "Outer and inner arcs overlap. Move the inner points outside the outer arc, "
                    "or use Flip Outer / Flip Inner. "
                    f"Accepted shell-cut slices: {accepted_count}."
                )
            else:
                self.annotate_status.setText(
                    f"Shell Cut Boundary, slice {slice_index}, {contour_text}. "
                    f"All four endpoints are set. Press Enter or click Accept Slice + Next. "
                    f"Accepted shell-cut slices: {accepted_count}."
                )
            self._update_annotation_progress()
            self._update_annotation_review_slice_choices()
            return

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
                f"Slice {slice_index} is skipped because its contour is not usable. "
                f"Clear Current Slice if you want to annotate it."
            )
            self._update_annotation_progress()
            self._update_annotation_review_slice_choices()
            return
        row = self.annotation_rows.get(slice_index)
        if row is not None and _core().normalize_surface_mode(row.get("surface_mode")) != "normal":
            label = self._surface_mode_label(row.get("surface_mode", "normal"))
            self.annotate_status.setText(
                f"Slice {slice_index}, {contour_text}. Marked as {label}. "
                f"Clear Current Slice if you want four-point annotation. Accepted slices: {accepted_count}."
            )
            self._update_annotation_progress()
            self._update_annotation_review_slice_choices()
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
        self._update_annotation_review_slice_choices()

    def _recommended_annotation_count(self) -> int:
        if self.annotation_target_slices:
            return len(self._target_annotation_slices())
        total = len(self.annotation_region_slices)
        if total == 0:
            return 0
        return min(total, max(3, (total + 7) // 8))

    def _update_annotation_progress(self) -> None:
        region_total = len(self.annotation_region_slices)
        if self._has_3d_annotation_shell():
            curve_count, point_count, patch_count = self.surface_preview_canvas.annotation_counts()
            shell_mesh = self.surface_preview_canvas.shell_mesh
            vertex_count = len(getattr(shell_mesh, "vertices", []))
            face_count = len(getattr(shell_mesh, "faces", []))
            backend = getattr(shell_mesh, "backend", "voxel")
            if region_total:
                first_slice = self.annotation_region_slices[0]
                last_slice = self.annotation_region_slices[-1]
                range_text = f"{region_total} slices ({first_slice}-{last_slice})"
            else:
                range_text = "0 slices"
            self.annotate_progress.setText(
                f"{vertex_count:,} pts · {face_count:,} faces · {range_text} · {backend}"
            )
            self._update_annotation_source_title()
            self.slice_canvas.set_progress_text("")
            return
        target_slices = self._target_annotation_slices()
        target_total = len(target_slices)
        accepted = len([slice_index for slice_index in self.shell_cut_rows_by_slice if slice_index in target_slices])
        if region_total == 0:
            self.annotate_progress.setText("Region slices: 0. No annotation target yet.")
            self.slice_canvas.set_progress_text("Progress: no region slice found.")
            return
        first_slice = self.annotation_region_slices[0]
        last_slice = self.annotation_region_slices[-1]
        if self.review_mode_active:
            current_slice = int(self.annotate_slice.value())
            queue_total = len(self.review_queue_slices)
            checked = len([slice_index for slice_index in self.review_checked_slices if slice_index in self.review_queue_slices])
            current_position = 0
            for index, slice_index in enumerate(self.review_queue_slices, start=1):
                if slice_index >= current_slice:
                    current_position = index
                    break
            if current_position == 0 and queue_total:
                current_position = queue_total
            remaining = max(0, queue_total - checked)
            self.annotate_progress.setText(
                f"Review & Repair: {checked}/{queue_total} checked. "
                f"Remaining: {remaining}. Region range: {first_slice}-{last_slice}."
            )
            self.slice_canvas.set_progress_text(
                f"Review queue: {checked}/{queue_total} checked, remaining {remaining}\n"
                f"Current review target: {current_position}/{queue_total} (slice {current_slice})\n"
                f"Source build: {self.review_source_build_dir or 'current build'}"
            )
            return
        target = self._recommended_annotation_count()
        all_accepted = len(self.shell_cut_rows_by_slice)
        shell_cut_count = len(self.shell_cut_rows_by_slice)
        if self.annotation_target_slices:
            remaining = max(0, target_total - accepted)
            extra_slices = max(0, all_accepted - accepted)
            progress_line = (
                f"Smart suggested slices: {accepted}/{target_total}. "
                f"Remaining: {remaining}. Extra manual slices: {extra_slices}."
            )
            canvas_line = (
                f"Smart suggested slices: {accepted}/{target_total}, remaining {remaining}\n"
                f"Extra manual slices: {extra_slices}"
            )
        else:
            remaining = max(0, target - all_accepted)
            extra_slices = max(0, all_accepted - target)
            progress_line = (
                f"Accepted slices: {all_accepted}. Recommended minimum: {target}. "
                f"Remaining to recommendation: {remaining}. Extra anchors: {extra_slices}."
            )
            canvas_line = (
                f"Accepted slices: {all_accepted}; recommended minimum {target}, remaining {remaining}\n"
                f"Extra anchors: {extra_slices}"
            )
        current_slice = int(self.annotate_slice.value())
        current_position = 0
        for index, slice_index in enumerate(target_slices, start=1):
            if slice_index >= current_slice:
                current_position = index
                break
        if current_position == 0:
            current_position = target_total
        self.annotate_progress.setText(
            f"Region slices: {region_total} ({first_slice}-{last_slice}). "
            f"{progress_line} "
            f"Shell-cut slices: {shell_cut_count}."
        )
        self.slice_canvas.set_progress_text(
            f"{canvas_line}\n"
            f"Current target slice: {current_position}/{target_total} (slice {current_slice})\n"
            f"Region range: {first_slice}-{last_slice}, shell-cut {shell_cut_count}"
        )

    def accept_annotation_slice(self, show_success: bool = True) -> bool:
        if self.annotation_mask_data is None:
            QMessageBox.warning(self, "No mask", "Load a mask first.")
            return False
        if self.annotation_pick_mode_kind == "shell_cut":
            return self.accept_shell_cut_annotation_slice(show_success=show_success)
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
        self.annotation_boundary_cache.pop(slice_index, None)
        self._update_annotation_status()
        self.refresh_annotation_preview()
        self.append_log(f"Accepted manual landmarks for slice {slice_index}\n")
        autosave_path = self._autosave_annotation_rows()
        if autosave_path is not None:
            self.append_log(f"Autosaved manual landmarks: {autosave_path}\n")
        if show_success:
            self.slice_canvas.setFocus(Qt.OtherFocusReason)
        return True

    def accept_shell_cut_annotation_slice(self, show_success: bool = True) -> bool:
        if self.annotation_mask_data is None:
            QMessageBox.warning(self, "No mask", "Load a mask first.")
            return False
        contour = self.slice_canvas.selected_contour()
        if contour is None:
            QMessageBox.warning(self, "No contour", "No usable contour is available on this slice.")
            return False
        slice_index = int(self.annotate_slice.value())
        landmarks = dict(self.slice_canvas.landmarks)
        existing_row = self.shell_cut_rows_by_slice.get(slice_index)
        if (
            existing_row is not None
            and _core().normalize_surface_mode(existing_row.get("surface_mode")) != "normal"
        ):
            self._update_annotation_status()
            self.refresh_annotation_preview()
            return True
        missing = [name for name in SliceCanvas.SHELL_CUT_ORDER if name not in landmarks]
        if missing:
            QMessageBox.warning(self, "Missing shell-cut points", "Please set: " + ", ".join(missing))
            return False
        overlap_excess = self.slice_canvas.shell_cut_overlap_excess()
        if overlap_excess > 0:
            QMessageBox.warning(
                self,
                "Overlapping cut arcs",
                (
                    "Outer and inner cut arcs overlap on this contour.\n\n"
                    "Move the inner points outside the outer arc, or use Flip Outer / Flip Inner "
                    "so the two arcs only share endpoints."
                ),
            )
            return False

        row = self._shell_cut_row_from_landmarks(contour, landmarks)
        outer_path = self._normalize_annotation_path_choice(row.get("outer_path"))
        inner_path = self._normalize_annotation_path_choice(row.get("inner_path"))
        self.shell_cut_rows_by_slice[slice_index] = row
        self.shell_cut_landmarks_by_slice[slice_index] = landmarks
        self.annotation_rows[slice_index] = self._annotation_row_from_shell_cut_landmarks(
            contour,
            landmarks,
            outer_path=outer_path,
            inner_path=inner_path,
        )
        self.annotation_landmarks_by_slice[slice_index] = self._boundary_landmarks_from_shell_cut(landmarks)
        self.annotation_path_choices_by_slice[slice_index] = {
            "outer_path": outer_path,
            "inner_path": inner_path,
        }
        self.annotation_boundary_cache.pop(slice_index, None)
        self.annotation_skipped_slices.discard(slice_index)
        self._update_annotation_status()
        self.refresh_annotation_preview()
        self.append_log(f"Accepted shell-cut boundary points for slice {slice_index}\n")
        autosave_path = self._autosave_shell_cut_annotations()
        if autosave_path is not None:
            self.append_log(f"Autosaved shell-cut annotations: {autosave_path}\n")
        csv_autosave_path = self._autosave_annotation_rows()
        if csv_autosave_path is not None:
            self.append_log(f"Autosaved derived build CSV: {csv_autosave_path}\n")
        if show_success:
            self.slice_canvas.setFocus(Qt.OtherFocusReason)
        return True

    def clear_annotation_slice(self) -> None:
        slice_index = int(self.annotate_slice.value())
        if self.annotation_pick_mode_kind == "shell_cut":
            self.shell_cut_landmarks_by_slice.pop(slice_index, None)
            self.shell_cut_rows_by_slice.pop(slice_index, None)
            self.annotation_landmarks_by_slice.pop(slice_index, None)
            self.annotation_rows.pop(slice_index, None)
            self.annotation_path_choices_by_slice.pop(slice_index, None)
            self.annotation_boundary_cache.pop(slice_index, None)
            self.annotation_skipped_slices.discard(slice_index)
            self.slice_canvas.landmarks = {}
            self._set_next_annotation_mode()
            self.slice_canvas.update()
            self._update_annotation_status()
            self.refresh_annotation_preview()
            self._autosave_shell_cut_annotations()
            self._autosave_annotation_rows()
            return
        self.annotation_landmarks_by_slice.pop(slice_index, None)
        self.annotation_rows.pop(slice_index, None)
        self.annotation_path_choices_by_slice.pop(slice_index, None)
        self.annotation_boundary_cache.pop(slice_index, None)
        self.annotation_skipped_slices.discard(slice_index)
        self._set_annotation_path_widgets("auto", "auto")
        self.slice_canvas.landmarks = {}
        self._set_next_annotation_mode()
        self.slice_canvas.update()
        self._update_annotation_status()
        self.refresh_annotation_preview()
        self._autosave_annotation_rows()

    def _load_annotation_rows_from_csv(
        self,
        csv_path: Path,
        output_dir: Optional[Path] = None,
        show_message: bool = True,
        sync_build: bool = True,
    ) -> tuple[int, List[str]]:
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
        self.annotation_boundary_cache.clear()
        self.annotation_target_slices = sorted(loaded_rows)
        self.annotate_previous_csv.set_text(csv_path)
        if output_dir is None:
            output_dir = Path(self.annotate_output.text()).expanduser() if self.annotate_output.text().strip() else csv_path.parent
        if not self.annotate_output.text().strip():
            self.annotate_output.set_text(output_dir)
        if sync_build:
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
        if show_message:
            QMessageBox.information(self, "Previous CSV loaded", message)
        return len(loaded_rows), errors

    def load_previous_annotation_csv(self) -> None:
        if self.annotation_mask_data is None:
            QMessageBox.warning(
                self,
                "Load mask first",
                "Load the same target mask before loading previous shell-cut annotations.",
            )
            return

        try:
            json_path = self._require_path("Previous shell-cut JSON", self.annotate_previous_csv.text())
            self._load_shell_cut_rows_from_json(json_path, show_message=True)
        except Exception as exc:
            self._show_exception_dialog("Load previous shell-cut JSON failed", exc)

    def _load_shell_cut_rows_from_json(
        self,
        json_path: Path,
        output_dir: Optional[Path] = None,
        show_message: bool = True,
        sync_build: bool = True,
    ) -> tuple[int, List[str]]:
        with Path(json_path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("schema") != "laminar_boundary_builder.shell_cut_annotations.v1":
            raise ValueError("Expected shell_cut_annotations.json from the Shell Cut Boundary workflow.")
        rows = data.get("rows", [])
        if not rows:
            raise ValueError("No shell-cut rows found in JSON.")

        loaded_rows: Dict[int, Dict] = {}
        loaded_landmarks: Dict[int, Dict[str, int]] = {}
        errors: List[str] = []
        for row in rows:
            try:
                slice_index = int(row["slice_index"])
                contours = self.annotation_contours_by_slice.get(slice_index)
                if contours is None:
                    contours = self._extract_contours_for_annotation_slice(slice_index)
                    self.annotation_contours_by_slice[slice_index] = contours
                    self._prune_annotation_contour_cache(slice_index)
                _core()._select_contour_for_row(row, contours)
                surface_mode = _core().normalize_surface_mode(row.get("surface_mode"))
                row["surface_mode"] = surface_mode
                if surface_mode != "normal":
                    row["points"] = {}
                    row["outer_path"] = "whole" if surface_mode == "outer_only" else ""
                    row["inner_path"] = "whole" if surface_mode == "inner_only" else ""
                    loaded_rows[slice_index] = row
                    continue
                points = row.get("points", {})
                landmarks = {
                    name: int(points[name]["index"])
                    for name in SliceCanvas.SHELL_CUT_ORDER
                    if name in points and isinstance(points[name], dict) and points[name].get("index") is not None
                }
                missing = [name for name in SliceCanvas.SHELL_CUT_ORDER if name not in landmarks]
                if missing:
                    raise ValueError("missing " + ", ".join(missing))
                row["outer_path"] = self._normalize_annotation_path_choice(row.get("outer_path"))
                row["inner_path"] = self._normalize_annotation_path_choice(row.get("inner_path"))
            except Exception as exc:
                errors.append(f"slice {row.get('slice_index', '?')}: {exc}")
                continue
            loaded_rows[slice_index] = row
            loaded_landmarks[slice_index] = landmarks

        if not loaded_rows:
            detail = "\n".join(errors[:8])
            raise ValueError("No usable shell-cut rows were loaded." + (f"\n{detail}" if detail else ""))

        self.shell_cut_rows_by_slice = loaded_rows
        self.shell_cut_landmarks_by_slice = loaded_landmarks
        self.annotation_skipped_slices.clear()
        self.annotation_target_slices = sorted(loaded_rows)
        self.annotate_previous_csv.set_text(json_path)
        if output_dir is None:
            output_dir = Path(self.annotate_output.text()).expanduser() if self.annotate_output.text().strip() else json_path.parent
        if not self.annotate_output.text().strip():
            self.annotate_output.set_text(output_dir)
        self._sync_derived_annotation_rows_from_shell_cut()

        if sync_build:
            csv_path = output_dir / "manual_landmarks_interactive.csv"
            self._write_annotation_rows_csv(csv_path)
            self._sync_build_from_annotation(csv_path, output_dir)

        first_slice = self.annotation_target_slices[0]
        current_slice = int(self.annotate_slice.value())
        target = current_slice if current_slice in loaded_rows else first_slice
        self.annotate_slice.setValue(target)
        self.refresh_annotation_slice()
        self.enter_annotation_picking_mode()
        autosave_path = self._autosave_shell_cut_annotations()
        self._autosave_annotation_rows()

        message = f"Loaded {len(loaded_rows)} shell-cut slice(s) from:\n{json_path}"
        if errors:
            message += f"\n\nSkipped {len(errors)} row(s). See Log > View Current Log for details."
            self.append_log("Shell-cut JSON load skipped rows:\n" + "\n".join(errors) + "\n")
        if autosave_path is not None:
            message += f"\n\nAutosaved editable copy:\n{autosave_path}"
        self.annotate_status.setText(
            f"Loaded {len(loaded_rows)} shell-cut slice(s). Edit points or continue marking slices."
        )
        self.append_log(f"Loaded previous shell-cut JSON: {json_path}\n")
        if show_message:
            QMessageBox.information(self, "Shell-cut JSON loaded", message)
        return len(loaded_rows), errors

    def export_annotation_csv(self) -> None:
        if not self.shell_cut_rows_by_slice:
            QMessageBox.warning(self, "No accepted slices", "Accept at least one slice before saving.")
            return
        try:
            output_dir = self._require_path("Output folder", self.annotate_output.text())
            output_dir.mkdir(parents=True, exist_ok=True)
            self._sync_derived_annotation_rows_from_shell_cut()
            csv_path = output_dir / "manual_landmarks_interactive.csv"
            self._write_annotation_rows_csv(csv_path)
            shell_cut_path = output_dir / "shell_cut_annotations.json"
            self._write_shell_cut_annotations_json(shell_cut_path)
            self._sync_build_from_annotation(csv_path, output_dir)
            self.tabs.setCurrentIndex(1)
            QMessageBox.information(
                self,
                "Annotations saved",
                f"Saved shell-cut annotations:\n{shell_cut_path}\n\n"
                f"Saved derived build CSV:\n{csv_path}\n\n"
                "Review the Build settings.",
            )
            self.append_log(f"Saved shell-cut annotations: {shell_cut_path}\n")
            self.append_log(f"Saved derived build CSV: {csv_path}\n")
        except Exception as exc:
            self._show_exception_dialog("Save failed", exc)

    def export_current_annotation_mask(self) -> None:
        if self.annotation_mask_data is None or self.annotation_mask_path is None:
            QMessageBox.warning(self, "No mask", "Load or extract a mask first.")
            return

        output_text = self.annotate_output.text().strip()
        output_dir = Path(output_text).expanduser() if output_text else Path.home() / "Desktop"
        region = "".join(
            character if character.isalnum() or character in ("_", "-") else "_"
            for character in (self.annotate_region.text().strip() or "region")
        ).strip("_")
        hemisphere = self.annotate_hemisphere.currentText().strip() or "all"
        default_path = output_dir / f"{region}_{hemisphere}_extracted_mask.npy"
        path_text, _ = QFileDialog.getSaveFileName(
            self,
            "Export current mask",
            str(default_path),
            "Mask volumes (*.npy *.nrrd *.nhdr);;All files (*)",
        )
        if not path_text:
            return

        export_path = Path(path_text).expanduser()
        if not export_path.suffix:
            export_path = export_path.with_suffix(".npy")
        old_mask_path = self.annotation_mask_path
        if self._same_path(old_mask_path, export_path):
            QMessageBox.information(
                self,
                "Choose another path",
                "Choose a different file path so the mask is saved outside the temporary cache.",
            )
            return
        try:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            if old_mask_path.exists():
                shutil.copy2(old_mask_path, export_path)
            else:
                core = _core()
                np = _numpy()
                core.save_volume(export_path, np.asarray(self.annotation_mask_data, dtype=np.uint8))
            self.annotation_mask_path = export_path
            self.annotation_mask_is_temporary = False
            self.annotate_mask.set_text(export_path)
            if not self.build_mask.text().strip() or self._same_path(self.build_mask.text(), old_mask_path):
                self.build_mask.set_text(export_path)
            self._update_annotation_readiness()
            self.append_log(f"Exported current mask: {export_path}\n")
            QMessageBox.information(self, "Mask exported", f"Saved:\n{export_path}")
        except Exception as exc:
            self._show_exception_dialog("Export mask failed", exc)

    def _ensure_persistent_annotation_mask(self, annotation_output_dir: Path) -> Optional[Path]:
        mask_path = self.annotation_mask_path
        if mask_path is None:
            current_mask = self._resolved_existing_input_path(self.annotate_mask.text())
            return current_mask
        if not self.annotation_mask_is_temporary:
            return mask_path

        output_dir = Path(annotation_output_dir).expanduser()
        export_path = output_dir / "inputs" / "target_mask.npy"
        export_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if mask_path.exists() and not self._same_path(mask_path, export_path):
                shutil.copy2(mask_path, export_path)
            elif not self._same_path(mask_path, export_path):
                core = _core()
                np = _numpy()
                core.save_volume(export_path, np.asarray(self.annotation_mask_data, dtype=np.uint8))
        except Exception as exc:
            raise RuntimeError(f"Could not persist temporary mask to:\n{export_path}\n\n{exc}") from exc

        self.annotation_mask_path = export_path
        self.annotation_mask_is_temporary = False
        self.annotate_mask.set_text(export_path)
        self.append_log(f"Persisted temporary mask for review/build: {export_path}\n")
        return export_path

    def _sync_build_from_annotation(self, csv_path: Path, annotation_output_dir: Path) -> None:
        mask_path = self._ensure_persistent_annotation_mask(annotation_output_dir)
        self.build_manual.set_text(csv_path)
        self.build_mask.set_text(mask_path or self.annotate_mask.text())
        self.build_template.set_text(self.annotate_template.text())
        self.build_output.set_text(annotation_output_dir / "build")
        self.build_boundaries.set_text(annotation_output_dir / "build" / "boundary_annotations.json")
        self.build_slice_axis.setCurrentText(self.annotate_slice_axis.currentText())
        self.build_min_area.setValue(self.annotate_min_area.value())
        self.build_keep_all.setChecked(self.annotate_keep_all.isChecked())
        if self.shell_cut_rows_by_slice:
            self.surface_method.setCurrentText("Shell cut")
        self.depth_method.setCurrentText("surfaces only")
        self.tabs.setCurrentIndex(1)

    def _validate_build_inputs(self, needs_boundaries: bool = False) -> bool:
        missing = []
        if not self.build_mask.text().strip():
            missing.append("Mask")
        if not self.build_manual.text().strip() and not needs_boundaries:
            missing.append("Derived build CSV")
        if not self.build_output.text().strip():
            missing.append("Output folder")
        if missing:
            QMessageBox.warning(
                self,
                "Build input missing",
                "Please fill: " + ", ".join(missing) + ".",
            )
            return False

        path_checks = [
            ("Mask", self.build_mask.text()),
        ]
        if not needs_boundaries:
            path_checks.append(("Derived build CSV", self.build_manual.text()))
        if self.build_template.text().strip():
            path_checks.append(("Template image", self.build_template.text()))
        if self.build_cell_csv.text().strip():
            path_checks.append(("Cell CSV", self.build_cell_csv.text()))

        for label, text in path_checks:
            if self._resolved_existing_input_path(text) is None:
                QMessageBox.warning(
                    self,
                    f"{label} not found",
                    f"Choose an existing {label} file:\n{text}",
                )
                return False

        if needs_boundaries:
            boundaries_path = self._build_boundaries_path()
            if not boundaries_path.exists():
                QMessageBox.warning(
                    self,
                    "Boundary JSON not found",
                    "Run Extract Surfaces first, or choose an existing boundary_annotations.json.",
                )
                return False
        return True

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
        page_layout = QVBoxLayout(tab)
        page_layout.setContentsMargins(0, 8, 0, 0)
        page_layout.setSpacing(10)
        page_layout.addWidget(self._make_build_hint())

        self._create_build_input_fields()

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)
        content_layout.setSizeConstraint(QLayout.SetMinAndMaxSize)
        content_layout.addWidget(self._make_build_required_section())
        content_layout.addWidget(self._make_build_optional_section())
        content_layout.addWidget(self._make_build_advanced_section())
        content_layout.addLayout(self._make_build_button_row())
        content_layout.addWidget(self._make_build_result_section())
        content_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setObjectName("buildScroll")
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        page_layout.addWidget(scroll, 1)
        return tab

    def _make_build_hint(self) -> QFrame:
        return self._make_hint(
            "Check Mask, derived build CSV, and Output folder. Extract surfaces first; depth unlocks after Boundary JSON exists."
        )

    def _create_build_input_fields(self) -> None:
        self.build_mask = PathRow("Target mask volume (.nrrd/.npy/.npz)")
        self.build_manual = PathRow(
            "manual_landmarks_template.csv",
            file_filter="CSV files (*.csv);;All files (*)",
        )
        self.build_boundaries = PathRow(
            "boundary_annotations.json",
            file_filter="JSON files (*.json);;All files (*)",
        )
        self.build_output = PathRow("Output folder for surfaces, volumes, tables, and QC", select_file=False)
        self.build_template = PathRow("Optional template image volume")
        self.build_cell_csv = PathRow(
            "Optional soma coordinate CSV",
            file_filter="CSV files (*.csv);;All files (*)",
        )
        self.build_swc_glob = QLineEdit()
        self.build_swc_glob.setPlaceholderText("Optional SWC glob, for example data/local/swc/*.swc")
        self.build_slice_axis = self._axis_combo()
        self.build_min_area = self._min_area_spin()

        self.resample_points = QSpinBox()
        self.resample_points.setRange(8, 1000)
        self.resample_points.setValue(80)
        self.resample_points.setButtonSymbols(QSpinBox.NoButtons)
        self.resample_points.setMinimumHeight(28)
        self.surface_method = CleanComboBox()
        self.surface_method.addItems(["Shell cut", "Contour shell", "Arc graph", "Mask constrained", "Fast loft"])
        self.surface_method.setCurrentText("Contour shell")
        self.surface_method.setMinimumHeight(28)
        self.shell_backend = CleanComboBox()
        self.shell_backend.addItems(["Voxel", "Marching cubes"])
        self.shell_backend.setCurrentText("Voxel")
        self.shell_backend.setMinimumHeight(28)
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
        self.build_readiness = QLabel()
        self.build_readiness.setObjectName("buildReadyText")
        self.build_readiness.setWordWrap(True)
        self._connect_build_readiness_signals()

    def _make_build_required_section(self) -> QGroupBox:
        required_box, required_form = self._make_form_section("Required Build Inputs")
        required_form.addRow("Status", self.build_readiness)
        self._add_help_row(required_form, "Mask", self.build_mask, *BUILD_HELP["mask"])
        self._add_help_row(required_form, "Derived build CSV", self.build_manual, *BUILD_HELP["manual_csv"])
        self._add_help_row(required_form, "Output folder", self.build_output, *BUILD_HELP["output"])
        return required_box

    def _connect_build_readiness_signals(self) -> None:
        for path_row in (
            self.build_mask,
            self.build_manual,
            self.build_boundaries,
            self.build_output,
            self.build_template,
            self.build_cell_csv,
        ):
            path_row.edit.textChanged.connect(self._update_build_readiness)
        self.build_swc_glob.textChanged.connect(self._update_build_readiness)

    def _update_build_readiness(self, *_args) -> None:
        if not hasattr(self, "build_readiness"):
            return

        missing = []
        warnings = []
        mask_text = self.build_mask.text().strip()
        manual_text = self.build_manual.text().strip()
        output_text = self.build_output.text().strip()

        if not mask_text:
            missing.append("Mask")
        elif self._resolved_existing_input_path(mask_text) is None:
            missing.append("Mask file")

        if not manual_text:
            missing.append("Derived build CSV")
        elif self._resolved_existing_input_path(manual_text) is None:
            missing.append("Derived build CSV file")

        if not output_text:
            missing.append("Output folder")

        for label, text in (
            ("Template image", self.build_template.text().strip()),
            ("Cell CSV", self.build_cell_csv.text().strip()),
        ):
            if text and self._resolved_existing_input_path(text) is None:
                warnings.append(f"{label} file is missing.")

        surface_ready = not missing
        boundary_text = self.build_boundaries.text().strip()
        if boundary_text:
            boundary_path = Path(boundary_text).expanduser()
        elif output_text:
            boundary_path = Path(output_text).expanduser() / "boundary_annotations.json"
        else:
            boundary_path = None
        depth_ready = surface_ready and boundary_path is not None and boundary_path.exists()

        lines = []
        if not surface_ready:
            lines.append("Surface build: fill " + ", ".join(missing) + ".")
        elif depth_ready:
            lines.append("Surface build and depth volume are ready.")
        else:
            lines.append("Surface build is ready. Depth unlocks after Boundary JSON exists.")
        lines.extend(warnings)

        self.build_readiness.setText("\n".join(lines))
        state = "ready" if surface_ready and not warnings else "missing" if missing else "warning"
        self._set_label_state(self.build_readiness, state)

        if hasattr(self, "surface_button"):
            self.surface_button.setEnabled(surface_ready)
        if hasattr(self, "depth_button"):
            self.depth_button.setEnabled(depth_ready)
        if hasattr(self, "review_qc_button"):
            review_ready = surface_ready and bool(output_text) and Path(output_text).expanduser().exists()
            self.review_qc_button.setEnabled(review_ready)

    def _make_build_optional_section(self) -> QGroupBox:
        optional_box, optional_form = self._make_form_section(
            "Optional Measurements",
            checkable=True,
            checked=False,
        )
        self._add_help_row(optional_form, "Template image", self.build_template, *BUILD_HELP["template"])
        self._add_help_row(optional_form, "Cell CSV", self.build_cell_csv, *BUILD_HELP["cell_csv"])
        self._add_help_row(optional_form, "SWC glob", self.build_swc_glob, *BUILD_HELP["swc_glob"])
        self._set_collapsible_form_visible(optional_box, optional_form, False)
        return optional_box

    def _make_build_advanced_section(self) -> QGroupBox:
        advanced_box, advanced_form = self._make_form_section(
            "Advanced Build Settings",
            checkable=True,
            checked=False,
        )
        self._add_help_row(advanced_form, "Boundary JSON", self.build_boundaries, *BUILD_HELP["boundaries_json"])
        self._add_help_row(advanced_form, "Slice axis", self.build_slice_axis, *BUILD_HELP["slice_axis"])
        self._add_help_row(advanced_form, "Min contour area", self.build_min_area, *BUILD_HELP["min_area"])
        self._add_help_row(advanced_form, "Resample points", self.resample_points, *BUILD_HELP["resample_points"])
        self._add_help_row(advanced_form, "Surface method", self.surface_method, *BUILD_HELP["surface_method"])
        self._add_help_row(advanced_form, "Shell backend", self.shell_backend, *BUILD_HELP["shell_backend"])
        self._add_help_row(advanced_form, "Depth method", self.depth_method, *BUILD_HELP["depth_method"])
        self._add_help_row(advanced_form, "Volume format", self.volume_format, *BUILD_HELP["volume_format"])
        self._add_help_row(advanced_form, "Max Laplace voxels", self.max_laplace_voxels, *BUILD_HELP["max_laplace_voxels"])
        self._add_help_row(advanced_form, "Boundary dilation", self.boundary_dilation, *BUILD_HELP["boundary_dilation"])
        self._add_help_row(advanced_form, "QC interval", self.qc_every, *BUILD_HELP["qc_every"])
        self._add_help_row(advanced_form, "", self.build_keep_all, *BUILD_HELP["keep_all"])
        self._set_collapsible_form_visible(advanced_box, advanced_form, False)
        return advanced_box

    def _make_build_button_row(self) -> QHBoxLayout:
        self.surface_button = self._make_button("Extract Surfaces", "primary")
        self.surface_button.clicked.connect(self.run_surface_build)
        self.depth_button = self._make_button("Compute Laminar Depth Volume", "secondary")
        self.depth_button.clicked.connect(self.run_depth_build)
        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(8)
        button_row.addWidget(self.surface_button)
        button_row.addWidget(self.depth_button)
        button_row.addStretch(1)
        self._update_build_readiness()
        return button_row

    def _make_build_result_section(self) -> QGroupBox:
        result_box, result_form = self._make_form_section("Build Result")
        self.build_result_label = QLabel("No build has run yet.")
        self.build_result_label.setObjectName("buildResultText")
        self.build_result_label.setWordWrap(True)
        self._set_label_state(self.build_result_label, "idle")
        self.build_progress = BuildProgressPanel("Build progress appears here while surfaces or depth volumes are running.")

        self.open_build_output_button = self._make_button("Open Output Folder", "secondary")
        self.open_build_output_button.clicked.connect(self.open_build_output_folder)
        self.open_build_output_button.setEnabled(False)
        self.review_qc_button = self._make_button("Review And Repair Build", "secondary")
        self.review_qc_button.clicked.connect(self.review_build_qc_slices)
        self.review_qc_button.setEnabled(False)

        result_buttons = QWidget()
        button_layout = QHBoxLayout(result_buttons)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)
        button_layout.addWidget(self.open_build_output_button)
        button_layout.addWidget(self.review_qc_button)
        button_layout.addStretch(1)

        result_form.addRow("Progress", self.build_progress)
        result_form.addRow("Summary", self.build_result_label)
        result_form.addRow("", result_buttons)
        self._update_build_readiness()
        return result_box

    def open_build_output_folder(self) -> None:
        output_text = self.build_output.text().strip()
        if not output_text:
            QMessageBox.information(self, "No output folder", "Choose or run a Build output folder first.")
            return
        output_dir = Path(output_text).expanduser()
        if not output_dir.exists():
            QMessageBox.information(
                self,
                "Output folder missing",
                f"Folder does not exist yet:\n{output_dir}",
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_dir)))

    def review_build_qc_slices(self) -> None:
        output_text = self.build_output.text().strip()
        if not output_text:
            QMessageBox.information(self, "No build output", "Run a build before starting review.")
            return
        build_output_dir = Path(output_text).expanduser()
        if not build_output_dir.exists():
            QMessageBox.information(
                self,
                "Build output missing",
                f"Folder does not exist yet:\n{build_output_dir}",
            )
            return

        mask_path = self._resolved_existing_input_path(self.build_mask.text())
        manual_csv = self._resolved_existing_input_path(self.build_manual.text())
        if mask_path is None or manual_csv is None:
            QMessageBox.warning(
                self,
                "Review inputs missing",
                "Build review needs an existing Mask and derived build CSV.",
            )
            return

        queue_slices = self._review_queue_from_build(build_output_dir)
        if queue_slices:
            self.append_log(
                "Review & Repair queue uses QC targets: "
                + ", ".join(str(value) for value in queue_slices)
                + "\n"
            )
        else:
            queue_slices = self._manual_slices_from_csv(manual_csv)
            if queue_slices:
                self.append_log(
                    "Review & Repair found no QC targets; reviewing derived build CSV slices: "
                    + ", ".join(str(value) for value in queue_slices)
                    + "\n"
                )
        if not queue_slices:
            QMessageBox.information(
                self,
                "No review slices",
                "No QC targets or manual annotation slices were found for this build.",
            )
            return

        request = ReviewRepairRequest(
            mask_path=mask_path,
            manual_csv=manual_csv,
            build_output_dir=build_output_dir,
            queue_slices=queue_slices,
        )
        if self.annotation_mask_data is not None and self.annotation_mask_path is not None:
            try:
                if self._same_path(self.annotation_mask_path, mask_path):
                    self._finish_review_repair_start(request)
                    return
            except OSError:
                pass

        self.pending_review_request = request
        self.annotate_mask.set_text(mask_path)
        self.tabs.setCurrentIndex(0)
        self.start_annotation_mask_load(mask_path)

    def _finish_review_repair_start(self, request: ReviewRepairRequest) -> None:
        annotation_output_dir = self._annotation_output_dir_from_build_dir(request.build_output_dir)
        loaded_count, errors = self._load_annotation_rows_from_csv(
            request.manual_csv,
            output_dir=annotation_output_dir,
            show_message=False,
            sync_build=False,
        )
        self.review_mode_active = True
        self.review_source_build_dir = request.build_output_dir
        self.review_queue_slices = [
            slice_index for slice_index in request.queue_slices if slice_index in self.annotation_region_slices
        ] or list(request.queue_slices)
        self.review_checked_slices = set()
        self.review_round_csv_path = None
        self.annotation_target_slices = list(self.review_queue_slices)
        self.build_mask.set_text(request.mask_path)
        self.build_manual.set_text(request.manual_csv)
        self.build_output.set_text(request.build_output_dir)
        self.build_boundaries.set_text(request.build_output_dir / "boundary_annotations.json")
        self._apply_review_mode_ui(True)
        self.tabs.setCurrentIndex(0)
        if self.review_queue_slices:
            self.annotate_slice.setValue(self.review_queue_slices[0])
        if not self.annotation_picking_active:
            self.enter_annotation_picking_mode()
        self._update_annotation_progress()
        skipped_text = f" {len(errors)} rows skipped." if errors else ""
        self.annotate_status.setText(
            f"Review & Repair loaded {loaded_count} manual slices.{skipped_text} "
            f"Queue: {len(self.review_queue_slices)} target slices."
        )
        self.append_log(
            "Review & Repair started.\n"
            f"source_build: {request.build_output_dir}\n"
            f"manual_csv: {request.manual_csv}\n"
            f"queue_slices: {', '.join(str(value) for value in self.review_queue_slices)}\n"
        )

    def _advance_review_queue(self) -> None:
        if not self.review_queue_slices:
            self.annotate_status.setText("Review queue is empty.")
            return
        current = int(self.annotate_slice.value())
        for slice_index in self.review_queue_slices:
            if slice_index > current and slice_index not in self.review_checked_slices:
                self.annotate_slice.setValue(slice_index)
                self.slice_canvas.setFocus(Qt.OtherFocusReason)
                return
        for slice_index in self.review_queue_slices:
            if slice_index not in self.review_checked_slices:
                self.annotate_slice.setValue(slice_index)
                self.slice_canvas.setFocus(Qt.OtherFocusReason)
                return
        self.annotate_status.setText(
            "Review queue complete. Save Review CSV or run Rebuild Review Round."
        )
        self._update_annotation_progress()

    def mark_review_slice_ok(self) -> None:
        if not self.review_mode_active:
            return
        slice_index = int(self.annotate_slice.value())
        self.review_checked_slices.add(slice_index)
        self._update_annotation_progress()
        self.append_log(f"Review slice marked OK: {slice_index}\n")
        self._advance_review_queue()

    def save_review_repair_csv(self) -> Optional[Path]:
        if not self.review_mode_active:
            QMessageBox.information(self, "Not in review mode", "Start Review And Repair from a build first.")
            return None
        if not self.annotation_rows:
            QMessageBox.warning(self, "No annotations", "No annotation rows are loaded.")
            return None

        source_build = self.review_source_build_dir or Path(self.build_output.text()).expanduser()
        annotation_output_dir = self._annotation_output_dir_from_build_dir(source_build)
        if self.review_round_csv_path is None:
            csv_path, _build_dir, _round_index = self._next_review_round_paths(annotation_output_dir)
            self.review_round_csv_path = csv_path
        self._write_annotation_rows_csv(self.review_round_csv_path)
        self.annotate_previous_csv.set_text(self.review_round_csv_path)
        self.build_manual.set_text(self.review_round_csv_path)
        self.append_log(f"Saved review repair CSV: {self.review_round_csv_path}\n")
        self.annotate_status.setText(f"Saved review CSV: {self.review_round_csv_path}")
        return self.review_round_csv_path

    def rebuild_review_round(self) -> None:
        csv_path = self.save_review_repair_csv()
        if csv_path is None:
            return
        source_build = self.review_source_build_dir or Path(self.build_output.text()).expanduser()
        annotation_output_dir = self._annotation_output_dir_from_build_dir(source_build)
        _csv_path, build_dir, round_index = self._next_review_round_paths(annotation_output_dir)
        if self.review_round_csv_path is not None:
            stem = self.review_round_csv_path.stem
            match = re.search(r"round(\d+)$", stem)
            if match:
                round_index = int(match.group(1))
                build_dir = annotation_output_dir / f"build_review_round{round_index}"
        self.build_manual.set_text(csv_path)
        self.build_output.set_text(build_dir)
        self.build_boundaries.set_text(build_dir / "boundary_annotations.json")
        self.tabs.setCurrentIndex(1)
        self.append_log(f"Review round {round_index} rebuild started: {build_dir}\n")
        self.run_surface_build()

    def _finish_pending_review_after_mask_load(self, load_data: AnnotationLoadData) -> None:
        request = self.pending_review_request
        if request is None:
            return
        try:
            if not self._same_path(load_data.mask_path, request.mask_path):
                return
        except OSError:
            return
        self.pending_review_request = None
        try:
            self._finish_review_repair_start(request)
        except Exception as exc:
            self._show_exception_dialog("Review setup failed", exc)
            return

    def _finish_pending_previous_csv_after_mask_load(self, loaded_mask_path: Path) -> None:
        request = self.pending_previous_csv_load
        if request is None:
            return
        mask_path, csv_path, output_dir = request
        try:
            if not self._same_path(loaded_mask_path, mask_path):
                return
        except OSError:
            return
        self.pending_previous_csv_load = None
        try:
            loaded_count, errors = self._load_annotation_rows_from_csv(
                csv_path,
                output_dir=output_dir,
                show_message=False,
            )
        except Exception as exc:
            self._show_exception_dialog("Load previous project annotations failed", exc)
            return

        skipped_text = f" {len(errors)} rows skipped." if errors else ""
        self.annotate_status.setText(
            f"Previous project loaded {loaded_count} manual slices.{skipped_text} "
            "Run Extract Surfaces when you are ready."
        )
        QMessageBox.information(
            self,
            "Previous project loaded",
            f"Loaded {loaded_count} manual slices from:\n{csv_path}\n\n"
            "No existing build folder was found, so the Build page is ready for Extract Surfaces.",
        )

    def _update_build_result_from_task(self, result: TaskResult) -> None:
        if not hasattr(self, "build_result_label"):
            return
        if not result.title.lower().startswith(("build", "depth")):
            return

        lines = result.message.splitlines()
        summary_lines = [lines[0] if lines else result.title]
        if result.output_dir:
            summary_lines.append(f"Output folder: {result.output_dir}")

        qc_lines = [
            line
            for line in lines
            if line.startswith(
                (
                    "QC:",
                    "QC review needed",
                    "Uncertain ranges",
                    "Suggested re-annotation",
                    "Surface topology jumps",
                    "Suggested topology review",
                )
            )
        ]
        summary_lines.extend(qc_lines)
        self.build_result_label.setText("\n".join(summary_lines))
        self._set_label_state(self.build_result_label, "ready")
        self.open_build_output_button.setEnabled(bool(result.output_dir))
        self.review_qc_button.setEnabled(any(line.startswith("QC review needed") for line in qc_lines))
        self._update_build_readiness()

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

    def _build_progress_panels(self, label: str) -> List[BuildProgressPanel]:
        panels: List[BuildProgressPanel] = []
        if label in ("build", "depth") and hasattr(self, "build_progress"):
            panels.append(self.build_progress)
        if label == "build" and hasattr(self, "annotate_build_progress"):
            panels.append(self.annotate_build_progress)
        return panels

    def _begin_task_progress(self, label: str) -> None:
        title = "Extracting surfaces" if label == "build" else "Computing laminar depth"
        detail = "Starting build task..."
        for panel in self._build_progress_panels(label):
            panel.show()
            panel.start(title, detail)

    def _update_task_progress(self, event: object) -> None:
        if not isinstance(event, dict):
            return
        label = str(event.get("label") or self.current_task_label or "")
        try:
            value = int(event.get("value", 0))
        except (TypeError, ValueError):
            value = 0
        stage = str(event.get("stage") or "Working")
        detail = str(event.get("detail") or "")
        for panel in self._build_progress_panels(label):
            panel.update_progress(value, stage, detail)

    def _finish_task_progress(self, label: Optional[str], result: TaskResult) -> None:
        if label not in ("build", "depth"):
            return
        title = "Surface build finished" if label == "build" else "Depth volume finished"
        detail = str(result.output_dir) if result.output_dir else "Task finished."
        for panel in self._build_progress_panels(label):
            panel.finish(title, detail)

    def _fail_task_progress(self, label: Optional[str], trace: str) -> None:
        if label not in ("build", "depth"):
            return
        detail = self._error_summary(trace)
        for panel in self._build_progress_panels(label):
            panel.fail(detail)

    def start_task(self, label: str, fn: Callable, accepts_progress: bool = False) -> None:
        if self.thread is not None:
            QMessageBox.warning(self, "Task running", "Please wait for the current task to finish.")
            return

        self.current_task_label = label
        self.append_log(f"\n--- {label} started ---\n")
        self._set_status(f"Running: {label}", "running")
        if hasattr(self, "build_result_label") and label in ("build", "depth"):
            action = "Extracting surfaces" if label == "build" else "Computing laminar depth"
            self.build_result_label.setText(action + "...")
            self._set_label_state(self.build_result_label, "running")
            self.open_build_output_button.setEnabled(False)
            self.review_qc_button.setEnabled(False)
            self._begin_task_progress(label)
        self.thread = QThread()
        self.worker = Worker(fn, accepts_progress=accepts_progress, task_label=label)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.progress.connect(self._update_task_progress)
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
        self.current_task_label = None

    def force_stop_current_task(self) -> None:
        if self.thread is None:
            return
        label = self.current_task_label
        self.append_log("\n--- task force-stopped by user ---\n")
        self.thread.requestInterruption()
        self.thread.terminate()
        if not self.thread.wait(3000):
            self.append_log("Task thread did not stop within 3 seconds; forcing process exit.\n")
            os._exit(0)
        self.thread = None
        self.worker = None
        self._fail_task_progress(label, "Task stopped by user.")
        self.current_task_label = None
        self._set_status("Ready", "ready")

    def task_finished(self, result: TaskResult) -> None:
        label = self.current_task_label
        self._set_status("Ready", "ready")
        self.append_log(f"\n{result.message}\n")
        self._finish_task_progress(label, result)
        self._update_build_result_from_task(result)
        self._apply_successful_task_result(result)
        self._show_task_result(result)

    def _apply_successful_task_result(self, result: TaskResult) -> None:
        payload = result.payload if isinstance(result.payload, dict) else {}
        if not payload.get("clear_3d_annotations"):
            return
        if not hasattr(self, "surface_preview_canvas"):
            return
        surface_names = payload.get("surface_names")
        if isinstance(surface_names, list) and surface_names:
            surface_text = ", ".join(str(name) for name in surface_names)
        else:
            surface_text = str(payload.get("surface_name") or "surface")
        annotations_path = payload.get("annotations_path")
        self.surface_preview_canvas.clear_3d_annotations()
        message = f"Built queued surface(s): {surface_text}. 3D annotation cleared for the next queue."
        if annotations_path:
            message += f" Saved markers: {annotations_path}"
        self.annotate_status.setText(message)
        self.on_3d_annotation_changed()

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

    def _error_summary(self, trace: str) -> str:
        lines = [line.strip() for line in trace.splitlines() if line.strip()]
        for line in reversed(lines):
            if line.startswith(("Traceback ", "File ")):
                continue
            return line
        return "Unknown error"

    def _error_recovery_hint(self, summary: str) -> str:
        text = summary.lower()
        if "allen annotation atlas" in text:
            return "Use a custom atlas file, or choose an existing Mask instead."
        if "input file does not exist" in text or "no such file" in text or "not found" in text:
            return "Choose an existing file and try again."
        if "boundary_annotations.json" in text or "boundary json" in text:
            return "Run Extract Surfaces first, or choose the existing boundary_annotations.json file."
        if "template shape" in text:
            return "The template is only a visual background. Choose a matching template, or leave it empty."
        return "Check the highlighted input and try again."

    def _show_error_dialog(self, title: str, trace: str) -> None:
        summary = self._error_summary(trace)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle(title)
        box.setText(summary)
        box.setInformativeText(
            self._error_recovery_hint(summary)
            + "\n\nDetails were written to Log > View Current Log."
        )
        if trace:
            box.setDetailedText(trace)
        box.exec_()

    def _show_exception_dialog(self, title: str, exc: Exception) -> None:
        trace = traceback.format_exc()
        if "NoneType: None" in trace:
            trace = str(exc)
        self.append_log("\n" + trace)
        self._show_error_dialog(title, trace)

    def task_failed(self, trace: str) -> None:
        label = self.current_task_label
        running_status = self.status_label.text().lower()
        self._set_status("Failed", "failed")
        self.append_log("\n" + trace)
        self._fail_task_progress(label, trace)
        if hasattr(self, "build_result_label") and ("build" in running_status or "depth" in running_status):
            self.build_result_label.setText("Run failed.\n" + self._error_summary(trace))
            self._set_label_state(self.build_result_label, "failed")
            self.open_build_output_button.setEnabled(False)
            self.review_qc_button.setEnabled(False)
        self._show_error_dialog("Run failed", trace)

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
        if not self._validate_build_inputs():
            return

        def task(progress: Callable[[int, str, str], None]) -> TaskResult:
            output_dir = self._require_path("Output folder", self.build_output.text())
            swc_glob = self.build_swc_glob.text().strip()
            swc_paths = sorted(glob.glob(swc_glob)) if swc_glob else []
            core = _core()
            outputs = core.run_laminar_boundary_pipeline(
                mask_path=self._require_path("Mask", self.build_mask.text()),
                manual_csv=self._require_path("Derived build CSV", self.build_manual.text()),
                output_dir=output_dir,
                template_path=self.build_template.text() or None,
                cell_csv=self.build_cell_csv.text() or None,
                swc_paths=swc_paths,
                slice_axis=self.build_slice_axis.currentText(),
                min_area=float(self.build_min_area.value()),
                largest_only=not self.build_keep_all.isChecked(),
                resample_points=int(self.resample_points.value()),
                surface_method=self.surface_method.currentText(),
                shell_backend=self.shell_backend.currentText(),
                depth_method=self.depth_method.currentText(),
                max_laplace_voxels=int(self.max_laplace_voxels.value()),
                boundary_dilation=int(self.boundary_dilation.value()),
                qc_every=int(self.qc_every.value()),
                volume_format=self.volume_format.currentText(),
                progress_callback=progress,
            )
            qc_summary = self._surface_qc_summary(output_dir)
            title = "Build needs QC review" if qc_summary.startswith("QC review needed") else "Build finished"
            lines = [title + "."]
            lines.extend(f"{key}: {value}" for key, value in outputs.items())
            lines.extend(("", qc_summary))
            return TaskResult(title, "\n".join(lines), output_dir=output_dir)

        self.start_task("build", task, accepts_progress=True)

    def _build_boundaries_path(self) -> Path:
        text = self.build_boundaries.text().strip()
        if text:
            return Path(text).expanduser()
        output_dir = self._require_path("Output folder", self.build_output.text())
        return output_dir / "boundary_annotations.json"

    def _surface_qc_summary(self, output_dir: Path) -> str:
        review_slices = self._read_qc_review_slices(output_dir)
        jump_slices = self._read_surface_jump_slices(output_dir)
        if not review_slices and not jump_slices:
            return "QC: no uncertain propagated slices."

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

        lines = []
        if review_slices:
            ranges = ", ".join(
                str(start) if start == end else f"{start}-{end}"
                for start, end in self._slice_ranges(review_slices)
            )
            targets = ", ".join(str(value) for value in self._review_targets_from_ranges(review_slices))
            lines.extend(
                [
                    f"QC review needed: {len(review_slices)} uncertain propagated slices.",
                    f"Uncertain ranges: {ranges}",
                    f"Suggested re-annotation targets: {targets}",
                ]
            )
        else:
            lines.append("QC review needed: surface topology jumps detected.")

        if jump_slices:
            jump_ranges = ", ".join(
                str(start) if start == end else f"{start}-{end}"
                for start, end in self._slice_ranges(jump_slices)
            )
            jump_targets = ", ".join(str(value) for value in self._review_targets_from_ranges(jump_slices))
            jump_details = self._read_surface_jump_details(output_dir)
            lines.extend(
                [
                    f"Surface topology jumps: {jump_ranges}",
                    f"Suggested topology review targets: {jump_targets}",
                ]
            )
            if jump_details:
                shown = "; ".join(jump_details[:6])
                if len(jump_details) > 6:
                    shown += f"; +{len(jump_details) - 6} more"
                lines.append(f"Topology jump details: {shown}")
        if manual_bad:
            lines.append(
                "Manual slices with QC flags: "
                + ", ".join(manual_bad)
                + ". Re-check these before trusting the surface."
            )
        return "\n".join(lines)

    def run_depth_build(self) -> None:
        if not self._validate_build_inputs(needs_boundaries=True):
            return

        def task(progress: Callable[[int, str, str], None]) -> TaskResult:
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
                progress_callback=progress,
            )
            lines = ["Depth volume finished."]
            lines.extend(f"{key}: {value}" for key, value in outputs.items())
            return TaskResult("Depth volume finished", "\n".join(lines), output_dir=output_dir)

        if hasattr(self, "build_result_label"):
            self.build_result_label.setText("Computing laminar depth volume...")
            self.open_build_output_button.setEnabled(False)
            self.review_qc_button.setEnabled(False)
        self.start_task("depth", task, accepts_progress=True)

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

        self.annotation_preview_request_id += 1
        if self.annotation_preview_thread is not None:
            self.annotation_preview_thread.requestInterruption()
            self.annotation_preview_thread.terminate()
            self.annotation_preview_thread.wait(1000)
            self.annotation_preview_thread = None
            self.annotation_preview_worker = None

        self._hide_help_popup()
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self.annotation_mask_data = None
        self.annotation_template_data = None
        self.annotation_contours_by_slice.clear()
        self.annotation_landmarks_by_slice.clear()
        self.shell_cut_landmarks_by_slice.clear()
        self.shell_cut_rows_by_slice.clear()
        self.annotation_rows.clear()
        self.annotation_path_choices_by_slice.clear()
        self.annotation_skipped_slices.clear()
        self.annotation_boundary_cache.clear()
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
