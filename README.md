# Laminar Boundary Builder

这是一个独立工作目录，用来后续做成可安装 app。当前第一版已经有一个可双击打开的 macOS GUI，同时保留命令行入口。核心代码已经复制到本目录里，不依赖仓库外的 `src/neuronvis`。

## 文件结构

```text
apps/laminar_boundary_builder/
├── laminar_boundary_builder/
│   ├── core.py      # contour、边界传播、surface、depth、QC 核心算法
│   ├── cli.py       # prepare/build/demo 命令行入口
│   ├── app.py       # PyQt 桌面 GUI
│   ├── __main__.py  # python -m laminar_boundary_builder
│   └── __init__.py
├── run.py           # 不安装时也能直接运行
├── launch_gui.py    # GUI 打包入口
├── pyproject.toml   # 后续 pip 安装和打包用
├── requirements.txt
└── build_macos.sh   # PyInstaller 打包起点
```

## 本地打开 GUI

在仓库根目录运行：

```bash
python apps/laminar_boundary_builder/launch_gui.py
```

GUI 主界面有两个正式页面：

- `Annotate`：在切片图上直接点 outer / inner 的起止点，并保存人工标注
- `Build`：生成 surface、laminar depth、layer normal、表格和 QC

`Tools > Run Demo Test` 可以用合成数据检查软件能不能正常跑，不是真实分析入口。

## 图上交互标注

推荐现在用这个流程：

1. 打开 `Annotate` 页。
2. 默认 `Brain region` 会填入：

```text
ENT
```

点 `Load Source And Start Picking` 时，软件会从全脑 Allen `annotation_10.nrrd` 里提取 ENT mask，并先保存成临时 `.npy` 缓存文件。这个临时 mask 会在 app 关闭时自动清理，不会变成长期数据。

如果想换别的脑区，直接把 `Brain region` 改成对应 acronym 或 ID，例如：

```text
VISp
```

默认会把子脑区一起提取；`Hemisphere` 可以选 `all` / `left` / `right`。软件会把这个 mask 放进临时目录，关闭软件时自动删除。

如果想长期保存当前提取出来的 mask，加载后点 `Export Current Mask`，手动导出到项目目录。只有这一步会把临时 mask 变成你自己管理的永久文件。

如果源码运行时没有内置 atlas，或者要换成别的 annotation atlas，勾选 `Use a custom Allen atlas file`，再选择 `.nrrd/.pkl` 文件。本机打包版可以带上 `annotation_10.nrrd`；公开源码仓库不包含 atlas 大文件，其他人也可以加载现成 mask，或在本地提供自己的 Allen annotation atlas。

3. `Template image` 可以选择：

```text
data/local/misc/average_template_10.nrrd
```

4. 点 `Load Source And Start Picking`，等待提取进度窗口结束后开始标点。

如果已经有其他现成 mask，直接在 `Mask` 里选择；只要 `Mask` 里有非临时路径，软件会优先使用这个现成 mask，例如：

```text
data/local/laminar_boundary_masks/ENT_left_ml_low_10um_mask.nrrd
```

5. `Output folder` 选择一个标注输出目录。
6. 左侧 `Progress` 会显示 mask 里有多少张有效切片，以及当前建议关键切片标注进度。
7. 进入点选模式后，参数设置会收起到左边；点左侧小箭头可以临时展开查看参数。展开时参数是灰色的，按 `Esc` 才能退出点选模式并重新编辑参数。
8. 每张切片直接按顺序点四个点：

```text
outer_start
outer_end
inner_start
inner_end
```

软件会自动吸附到最近的 contour 点，并在图上标出每个点的名字。

9. 快捷键：

```text
X      撤销当前切片上最后一个点
Enter  接受当前切片并进入下一张有效切片
S      当前整圈 contour 只作为 outer surface；适合 inner 已经结束的封顶切片
A      当前整圈 contour 只作为 inner surface；适合 outer 已经结束的封顶切片
Esc    退出点选模式，重新编辑参数
```

11. 标完几个关键切片后，点 `Save CSV And Review Build`。

软件会自动生成：

```text
manual_landmarks_interactive.csv
```

并自动跳到 `Build` 页，同时把这些内容同步过去：

```text
Mask
Manual CSV
Output folder
Template image
Slice axis
Min contour area
Keep all contours per slice
```

CSV 现在只是后台保存格式，不需要手动编辑，也不需要在 `Build` 页重新挑这些参数。

## 本地直接运行

在仓库根目录运行：

```bash
python apps/laminar_boundary_builder/run.py demo \
  --output-dir /tmp/laminar_boundary_demo
```

也可以跑一个更完整的冒烟检查：

```bash
python apps/laminar_boundary_builder/run.py selfcheck \
  --output-dir /tmp/laminar_boundary_selfcheck
```

## 安装成命令

进入这个文件夹：

```bash
cd apps/laminar_boundary_builder
python -m pip install -e .
```

安装后可以直接运行：

```bash
laminar-boundary-builder demo \
  --output-dir /tmp/laminar_boundary_demo
```

## 正式数据流程

推荐走图上交互流程：

```text
Annotate 里加载 mask 并点 outer/inner
        ↓
Save Manual CSV
        ↓
Build 里选择 mask + Manual CSV
        ↓
生成 surface、laminar depth 和 QC
```

命令行仍然保留一个旧的批量入口，可以提取每张切片的 mask 轮廓，并生成可填写的人工标注模板：

```bash
python apps/laminar_boundary_builder/run.py prepare \
  --mask data/local/ENT_mask.nrrd \
  --output-dir results/laminar_boundary/ENT_left_prepare \
  --slice-axis coronal \
  --manual-every 8
```

如果暂时不用图上交互，也可以打开：

```text
results/laminar_boundary/ENT_left_prepare/manual_landmarks_template.csv
```

每个关键切片手动填写：

```text
outer_start / outer_end
inner_start / inner_end
```

可以填 `*_index`，也可以填 `*_x/*_y/*_z` 坐标。点号和坐标可从同一目录下的 `contour_points.csv` 查。

第二步，用人工关键切片传播到中间切片，并生成 surface、depth field 和 QC：

```bash
python apps/laminar_boundary_builder/run.py build \
  --mask data/local/ENT_mask.nrrd \
  --manual-csv results/laminar_boundary/ENT_left_prepare/manual_landmarks_template.csv \
  --template data/local/misc/average_template_10.nrrd \
  --cell-csv results/ENT_soma_coordinates.csv \
  --output-dir results/laminar_boundary/ENT_left_build \
  --slice-axis coronal
```

## 输出

主要输出在 `--output-dir` 下：

- `surfaces/target_outer_surface.ply`
- `surfaces/target_inner_surface.ply`
- `surfaces/target_lateral_boundary.ply`
- `volumes/laminar_depth.nrrd`
- `volumes/layer_normal_x.nrrd`
- `volumes/layer_normal_y.nrrd`
- `volumes/layer_normal_z.nrrd`
- `tables/boundary_summary.csv`
- `tables/cell_laminar_depth.csv`
- `tables/dendrite_laminar_depth.csv`
- `qc/qc_slice_overlay/`
- `qc/qc_uncertain_slices.csv`

体积文件默认写成 NRRD。如果要输出 NIfTI，先安装：

```bash
python -m pip install ".[nifti]"
```

然后运行时加：

```bash
--volume-format nii.gz
```

## macOS 打包

运行：

```bash
bash apps/laminar_boundary_builder/build_macos.sh
```

成功后会得到：

```text
apps/laminar_boundary_builder/dist/Laminar Boundary Builder.app
apps/laminar_boundary_builder/dist/Laminar Boundary Builder.dmg
```

双击 `.app` 就会打开窗口。正式发给其他电脑前，后续还可以补 Developer ID 签名和 Apple notarization。

后面如果做切片点选编辑器，建议继续放在这个目录里，并复用 `laminar_boundary_builder/core.py`，不要再把算法散回主仓库。

## License

MIT License. See `LICENSE`.
