# Laminar Boundary Builder

Laminar Boundary Builder 是一款 macOS 桌面工具，用于在脑区的 3D 表面上标记 outer / inner 边界，并生成层深度（laminar depth）、层方向（layer normal）和质量检查结果。

[下载最新版 macOS 安装包](https://github.com/cerebrocss/laminar-boundary-builder/releases/latest)

## 你可以用它做什么

- 从内置 Allen atlas 提取指定脑区，也可以加载自己的脑区 mask。
- 直接在 3D shell 上画闭合边界，并选择需要保留的表面。
- 构建 outer / inner surface OBJ。
- 计算从 outer 到 inner 的归一化层深度，其中 outer 为 0，inner 为 1。
- 将层深度赋给细胞坐标或 SWC 形态点，并输出可检查的表格和体积文件。

## 下载与打开

1. 打开[最新版 Release](https://github.com/cerebrocss/laminar-boundary-builder/releases/latest)，下载 `Laminar.Boundary.Builder.dmg`。
2. 双击 DMG，将 `Laminar Boundary Builder.app` 拖到“应用程序”文件夹。
3. 双击应用程序启动。

如果 macOS 第一次打开时提示无法验证开发者，请进入“系统设置 → 隐私与安全性”，找到对应提示并选择“仍要打开”。

## 界面说明

应用中有两个主要页面：

- `Annotate`：加载脑区，在 3D shell 上画边界并构建 surfaces。
- `Build`：使用 outer / inner surfaces 计算层深度，并按需测量细胞或 SWC。

整个操作顺序是：

```text
加载脑区或 mask
        ↓
画 outer / inner 闭合边界
        ↓
选择每条边界围住的目标表面
        ↓
构建 3D surfaces
        ↓
计算 laminar depth
```

## 五分钟图文演示

### 1. 加载脑区或已有 mask

打开 `Annotate` 页面。你可以使用以下任一种输入：

- 在 `Brain region` 中填写 Allen 脑区 acronym 或 ID，例如 `ENT`、`VISp`。
- 在 `Mask` 中选择已有的 `.nrrd`、`.npy` 或 `.npz` 脑区 mask。

选择 `Hemisphere` 后，点击 `Load / Reload Source And Start Picking`。

![Annotate 首页：选择脑区、半球或已有 mask](docs/images/ui-01-start.jpg)

脑区提取和 3D shell 构建过程中会显示当前进度：

![脑区提取进度](docs/images/ui-02-loading-progress.jpg)

### 2. 在 3D shell 上画闭合边界

shell 出现后，紫色按钮表示当前模式。保持 `Draw Curve` 为紫色，在目标边界上依次点击。已有点可以拖动微调。

至少放置三个点后，点击 `Close Current Curve` 闭合边界。

![Draw Curve 模式：在 3D shell 上依次落点](docs/images/ui-03-draw-curve.jpg)

### 3. 选择目标表面

边界闭合后，应用会进入 `Select Surface` 模式。在闭合边界围住的目标一侧点击一次。

这个点相当于告诉应用：“我要保留这一侧的表面。”如果选择了错误的一侧，可以点击 `Undo (X)` 后重新选择。

![Select Surface 模式：红线是闭合边界，点击目标一侧作为 seed patch](docs/images/ui-04-select-surface.jpg)

### 4. 命名并构建 outer / inner surfaces

在左侧为当前 surface 命名：

- `outer`：层深度起点，计算结果为 0。
- `inner`：层深度终点，计算结果为 1。

为 outer 和 inner 分别完成“画闭合边界 → 选择目标表面”。两项都准备好后，点击底部的 `Build 3D Surfaces`。

![一条边界和目标表面准备完成后，Build 3D Surfaces 变为可用](docs/images/ui-05-build-ready.jpg)

### 5. 计算 laminar depth

切换到 `Build` 页面：

1. `Mask` 选择标注时使用的同一个脑区 mask。
2. `Output folder` 选择刚才构建 surfaces 的项目目录。应用会自动查找其中的 `build_3d/project_config.json`。
3. 状态显示 `Depth source: 3D outer/inner surfaces ready.` 后，点击 `Compute Laminar Depth`。

![Build 页面：填写 mask 和项目输出目录](docs/images/ui-06-depth-build.jpg)

普通计算只需要 Mask 和 Output folder。其他输入按需展开：

- `Template image`：用于生成更容易检查的 QC 预览。
- `Cell CSV`：为细胞坐标添加层深度。
- `SWC glob`：为一批 SWC 形态点测量层深度。
- `Details`：查看检测到的文件，或调整计算方法和输出格式。

每个 `i` 按钮都提供当前参数的用途和建议选项：

![参数帮助气泡](docs/images/ui-07-parameter-help.jpg)

## 选择输入数据

### 使用 Allen 脑区

在 `Brain region` 中填写 acronym 或结构 ID。常用设置包括：

- `Hemisphere`：选择 `all`、`left` 或 `right`。
- `Include child regions`：同时包含该脑区下面的子结构，通常建议开启。
- `Use a custom Allen atlas file`：需要使用其他分辨率或方向的 atlas 时开启。

如果希望保存这次提取出的 mask，可在加载完成后点击 `Export Current Mask`。

### 使用自己的 mask

在 `Mask` 中选择文件即可。mask 应满足以下条件：

- 支持 `.nrrd`、`.npy` 或 `.npz`。
- 目标脑区体素为非零值，背景为 0。
- 如果还要使用细胞坐标、SWC 或 template image，它们必须与 mask 位于同一坐标空间。

## 保存、继续和修改项目

开始标注前，为项目选择一个独立的 `Output folder`。应用会在这里保存 surfaces、标注和计算结果。

下次继续工作时可以：

- 点击窗口顶部的 `Open Previous Project` 打开项目。
- 加载同一个脑区或 mask，然后在 `Previous annotation JSON` 中选择 `surface_3d_annotations_*.json`，继续调整曲线和目标表面。

建议每个脑区、半球或实验条件使用单独的项目目录，避免不同结果互相覆盖。

## 输出文件

### 3D surface 结果

构建完成后，项目目录中会出现：

```text
build_3d/
├── project_config.json
├── surfaces/
│   ├── ...outer....obj
│   └── ...inner....obj
└── surface_3d_annotations_*.json
```

- `project_config.json`：记录 depth 计算需要使用的 surface 文件。
- `surfaces/*.obj`：构建出的 3D outer / inner surfaces。
- `surface_3d_annotations_*.json`：可重新加载和修改的曲线、名称与目标表面选择。

### 层深度结果

点击 `Compute Laminar Depth` 后，主要结果包括：

```text
volumes/
├── laminar_depth.nrrd
├── boundary_labels.nrrd
├── layer_normal_x.nrrd
├── layer_normal_y.nrrd
└── layer_normal_z.nrrd

tables/
├── cell_laminar_depth.csv
└── dendrite_laminar_depth.csv

qc/
└── qc_slice_overlay/
```

只有提供 `Cell CSV` 或 SWC 时，应用才会生成相应的测量表格。

## 常用快捷键

| 快捷键 | 作用 |
| --- | --- |
| `X` | 撤销上一步 3D 标注 |
| `Enter` | 当前 surface 准备完成时开始构建 |
| `Esc` | 暂停点选并返回输入设置 |

## 常见问题

### `Select Surface` 无法点击

先用 `Draw Curve` 放置至少三个点并闭合当前边界。只有存在闭合边界后，才能选择目标表面。

### `Build 3D Surfaces` 无法点击

检查左侧 `Next` 提示。每个 surface 都需要：

1. 一条闭合边界；
2. 一个名称；
3. 一次目标表面选择。

### `Compute Laminar Depth` 无法点击

确认以下内容：

- Mask 文件存在，并且与标注时使用的是同一个 mask。
- Output folder 中存在 `build_3d/project_config.json`。
- 项目中同时存在名为 outer 和 inner 的 surfaces。

### 选中了错误的表面

点击 `Undo (X)` 撤销 seed patch，然后在闭合边界的另一侧重新点击。必要时可以拖动边界点后再次选择。

### 细胞或 SWC 的层深度位置不正确

最常见原因是坐标空间不一致。请确认细胞坐标、SWC、mask 和 template image 使用相同的体素方向、原点和分辨率。

## 从源码运行

需要 Python 3.9 或更高版本：

```bash
git clone https://github.com/cerebrocss/laminar-boundary-builder.git
cd laminar-boundary-builder
python -m pip install -e .
laminar-boundary-builder-gui
```

也可以在仓库目录中直接启动：

```bash
python launch_gui.py
```

## 命令行计算层深度

已经有 mask 和 `build_3d/project_config.json` 时，可以直接运行：

```bash
laminar-boundary-builder depth \
  --mask /path/to/region_mask.nrrd \
  --project-config /path/to/project/build_3d/project_config.json \
  --output-dir /path/to/project
```

默认输出 NRRD。需要输出 NIfTI 时安装可选依赖：

```bash
python -m pip install -e ".[nifti]"
```

然后添加：

```bash
--volume-format nii.gz
```

## License

MIT License. See [LICENSE](LICENSE).
