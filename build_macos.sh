#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

APP_NAME="Laminar Boundary Builder"
BUNDLE_ID="org.cerebrocss.laminar-boundary-builder"
ENTRY="$APP_DIR/launch_gui.py"
ICON="$APP_DIR/assets/app.icns"
ONTOLOGY_JSON="$APP_DIR/../../data/local/misc/1.json"
ONTOLOGY_TXT="$APP_DIR/../../data/local/misc/annot.txt"
ANNOTATION_NRRD="$APP_DIR/../../data/local/misc/annotation_10.nrrd"
BUILTIN_MASK_DIR="$APP_DIR/laminar_boundary_builder/data/masks"

python -m pip install -e ".[packaging]"

PYINSTALLER_ARGS=(
  --noconfirm
  --clean
  --windowed
  --name "$APP_NAME"
  --osx-bundle-identifier "$BUNDLE_ID"
  --paths "$APP_DIR"
  --hidden-import scipy.ndimage
  --hidden-import scipy.spatial
  --hidden-import scipy.spatial._ckdtree
  --hidden-import scipy.sparse
  --hidden-import scipy.sparse.linalg
  --exclude-module matplotlib
  --exclude-module pandas
  --exclude-module IPython
  --exclude-module ipykernel
  --exclude-module ipywidgets
  --exclude-module jupyter
  --exclude-module jupyter_client
  --exclude-module jupyter_core
  --exclude-module jupyterlab
  --exclude-module nbformat
  --exclude-module notebook
  --exclude-module tensorboard
  --exclude-module tensorflow
  --exclude-module torch
  --distpath "$APP_DIR/dist"
  --workpath "$APP_DIR/build"
  --specpath "$APP_DIR"
)

if [[ -f "$ONTOLOGY_JSON" ]]; then
  PYINSTALLER_ARGS+=(--add-data "$ONTOLOGY_JSON:data/local/misc")
fi

if [[ -f "$ONTOLOGY_TXT" ]]; then
  PYINSTALLER_ARGS+=(--add-data "$ONTOLOGY_TXT:data/local/misc")
fi

if [[ -f "$ANNOTATION_NRRD" ]]; then
  PYINSTALLER_ARGS+=(--add-data "$ANNOTATION_NRRD:data/local/misc")
fi

if [[ -d "$BUILTIN_MASK_DIR" ]]; then
  PYINSTALLER_ARGS+=(--add-data "$BUILTIN_MASK_DIR:laminar_boundary_builder/data/masks")
fi

if [[ -f "$ICON" ]]; then
  PYINSTALLER_ARGS+=(--icon "$ICON")
else
  echo "No app.icns found at $ICON. Building with the default macOS app icon."
fi

python -m PyInstaller "${PYINSTALLER_ARGS[@]}" "$ENTRY"

if command -v hdiutil >/dev/null 2>&1; then
  hdiutil create \
    -volname "$APP_NAME" \
    -srcfolder "$APP_DIR/dist/$APP_NAME.app" \
    -ov \
    -format UDZO \
    "$APP_DIR/dist/$APP_NAME.dmg"
fi

echo "Build finished:"
echo "  $APP_DIR/dist/$APP_NAME.app"
if [[ -f "$APP_DIR/dist/$APP_NAME.dmg" ]]; then
  echo "  $APP_DIR/dist/$APP_NAME.dmg"
fi
