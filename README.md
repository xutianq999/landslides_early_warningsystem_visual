# YOLOE + Depth Anything V2 推理工作台(Mac / Apple Silicon)

零样本分割 + 相对深度估计的 demo 环境:命令行脚本 + 网页版推理台(Gradio)。
全部零样本:不训练,给提示(类别名)即可推理。已在 MacBook(MPS)实测通过。

**技术选型(已确定)**:分割用 **YOLOE**,深度用 **Depth Anything V2**。
早期对比过的 YOLO26-depth 与 DA3 已弃用,相关文件备份在 `_deprecated/`。

## 快速开始

```bash
cd ~/Desktop/yoloe-demo
python3 -m venv .venv                # 已建好可跳过
.venv/bin/pip install -r requirements.txt

# 启动网页推理台(推荐)
.venv/bin/python app.py
# 浏览器打开 http://127.0.0.1:7860
```

## 网页推理台(app.py)

左栏传原图,右栏选模型 → 点「开始推理」→ 结果图 + 信息显示在右侧。

| 模型选项 | 能力 | 备注 |
|---|---|---|
| YOLOE 零样本分割 | 文本提示任意类别的实例分割 | 类别框逗号分隔,任意概念,可中文 |
| Depth Anything V2 Small | 相对深度,速度快(121 ms) | 日常够用 |
| Depth Anything V2 Base | 相对深度,边界更准 | 精度优先时用 |
| 横向对比 | YOLOE + 两个 DA V2 一起跑,拼图 + 耗时/内存 | |

- 每个模型首次推理时才加载权重(懒加载),之后常驻内存
- 选分割模型时才显示类别输入和置信度滑块
- 深度图统一着色:越红越近,越蓝越远

## 滑坡监测特征提取(features.py)

从监控抓图提取一行特征向量写入 CSV,供 xLSTM 与报警模块使用。
基础 30 项;加 `--roi-auto`(只在自动检测到的沟壑/堆积体区域算特征)后再加 14 项区域掩模特征,共 44 项。
**完整流程与报警逻辑见 [PIPELINE.md](PIPELINE.md);字段含义见 [FEATURES.md](FEATURES.md)。**

```bash
# 单张
.venv/bin/python features.py 图片.jpg --out features.csv
# 批量(按文件名排序,自动与上一帧配准做变化检测)
.venv/bin/python features.py ./snapshots --out features.csv
# 带降雨(本地 CSV: 两列 time,precip_mm / 或在线拉取 Open-Meteo)
.venv/bin/python features.py 图片.jpg --weather-csv rain.csv
.venv/bin/python features.py 图片.jpg --lat 30.1 --lon 104.2

# 只在中央沟壑区域算特征(自动检测,或手动画框)
.venv/bin/python features.py 图片.jpg --roi-auto
.venv/bin/python features.py 图片.jpg --roi 0.28 0.18 0.72 0.98

# 沟壑分割可视化(颜色法 / OpenCV 边界追踪 / 混合)
.venv/bin/python segment_gully.py 图片.jpg -o gully_seg.png

# 三维点云 + 主平面可视化
.venv/bin/python visualize3d.py 图片.jpg -o pc_plane.png

# 特征有效性验证(几何真值 / 光照扰动 / 已知位移 / 动态剔除)
.venv/bin/python validate_features.py 图片.jpg

# 报警分析(变化率 + 加速度 + 噪声门控)
.venv/bin/python alarm.py features.csv --out alarm_result.csv

# 导出 Excel(字段清单 + 原始数据,供挑选特征)
.venv/bin/python export_features_excel.py 图片.jpg
```

**四层数据源**:① 图像(滑坡面积/动态物体)→ ② 深度图 → ③ 点云几何 → ④ 帧间时序。
识别到的 person/car/truck/construction vehicle 区域会**从几何与时序计算中剔除**,防止人为误差。

特征组(全部尺度无关,现场暂无标定):

| 组 | 特征 | 说明 |
|---|---|---|
| 深度统计 | disp_p05/p50/p95/std | 归一化**逆深度(视差)**分位数,值越大越近 |
| 三维几何 | slope_mean/p95, rough_local, curv_mean | 点云局部法向量 → 坡度角、粗糙度、曲率 |
| 三维结构 | plane_tilt, plane_rms | PCA 主平面拟合:倾角、归一化残差(平整度) |
| 三维结构 | bulge_frac | 凸起(朝相机外凸)面积占比,上升是坡脚鼓胀前兆 |
| 深度可信度 | edge_depth_corr | 图像边缘与深度边缘相关性 |
| 分割 | seg_<类>_frac/_max/_n | YOLOE 类别像素占比/最大连通域占比/实例个数 |
| 区域掩模 | gully_* / debris_* 的 area_frac, width_min/max/std, y_top/bottom/extent | 仅 `--roi-auto`:沟壑与堆积体掩模的尺寸与位置(14 项) |
| 变化 | shift_px, shift_resp, diff_mean/p95/frac | 与上一帧的配准位移 + 深度差统计 |
| 降雨 | rain_1h/24h/72h | 抓图时刻前累积降雨(mm) |

**每个字段的详细含义、单位、危险方向见 [FEATURES.md](FEATURES.md)。**

**重要局限(实测结论)**:
- **尺度未标定**:单帧坡度有仿射系统偏差(`d = a/z + b` 里的 `b` 被忽略),越远越明显;但同一相机下偏差恒定,**时间序列的变化量可靠**。位移目前只能用像素单位。
- **YOLOE 零样本认不出"裂缝"**:实测对 crack/debris/裸土等细结构概念检出为 0,只对 rock/tree/grass/deep valley 等实体名词有效。`features.py` 默认类别为 `deep valley,person,car,landslide,truck,construction vehicle`(deep valley/landslide 计面积,person/car/truck/construction vehicle 只计数并用作动态掩码)。裂缝/沟壑这类细结构改由 `segment_gully.py` 的经典 CV 检测(`--roi-auto`),不依赖模型。
- **无标签时建议把 xLSTM 用作"特征向量下一步预测器"**:预测残差(实际 vs 预测的偏离)就是异常分,不需要灾害标签;降雨特征用来解释天气引起的图像变化,降低误报。

## 命令行脚本

```bash
# 1. 零样本实例分割(文本提示,任意类别)
.venv/bin/python demo_image.py 图片.jpg --classes "person,bus,helmet"

# 2. 摄像头实时分割(按 q 退出)
.venv/bin/python demo_camera.py --classes "person,phone,cup"

# 3. 相对深度(Depth Anything V2 Small)
.venv/bin/python demo_depth.py 图片.jpg [--model base]

# 3b. 深度 + 点云/网格导出(.ply,MeshLab/CloudCompare/Blender 可直接打开)
.venv/bin/python demo_depth.py 图片.jpg --ply  --max-depth 15 --fov 60   # 点云
.venv/bin/python demo_depth.py 图片.jpg --mesh --max-depth 15 --fov 60   # 三角网格
```

## 点云 / 网格导出说明

- **原理**:DA V2 相对深度(逆深度)→ 针孔模型反投影:`z = 1/(disp+0.1)` 归一化后乘「场景最远距离」作近似尺度;水平 FOV 定焦距(不知道相机 FOV 就用默认 60°)
- **点云**:离散点,远处因透视必然稀疏(3D 点间距 ∝ 距离,面密度按 1/z² 衰减),这是几何决定的,不是丢点
- **网格**:按深度图的规则网格连三角面 → 表面连续,没有稀疏感;相邻四角深度比超过 2 倍不连面,避免把天空和近景缝成斜膜
- **滤波**:双边滤波压深度噪声同时保边(高斯会把物体边界糊成斜面)
- **注意**:DA V2 无绝对尺度,点云的"米数"是近似值,形状/比例正确,绝对距离不可信;要米制得换 DA V2 metric 权重或加相机标定
- **坐标**:相机在原点、y 轴向上、z 轴向前,导出前平移到质心便于查看器取景
- **文件**:二进制 PLY,每点 `x y z float32 + r g b uint8`,网格额外带三角面;超 150 万点自动抽稀
- 网页台:勾选「导出点云/网格」+ 选格式 + 调最远距离 → 推理后可在底部预览并下载

## 模型对比(Mac MPS 实测,640 级输入,热身后)

| 模型 | 速度 | 深度类型 | 边界质量 | 部署友好度 |
|---|---|---|---|---|
| YOLOE-26s-seg | ~37 ms | —(分割) | 好 | ★★★ ultralytics 一条链导 TensorRT |
| DA V2 Small | 121 ms | 相对 | 好 | ★★ 需走 transformers→ONNX |
| DA V2 Base | ~250 ms | 相对 | 更好 | ★★ 同上 |

弃用原因:YOLO26-depth 边界质量一般且深度能力被 DA V2 取代;DA3 质量最好但官方生态全在 CUDA、依赖链重,边缘部署不友好。

## 概念速查

- **YOLOE 三种提示模式**:`-seg.pt` 文本/视觉提示(可 `set_classes`);`-seg-pf.pt` 无提示(内置 4585 类词表,不可改)
- **零样本原理**:CLIP 把类别名编码成向量,与图像区域特征算相似度,类别成为推理时输入而非训练时固定输出;CLIP 编码只在 `set_classes` 时算一次
- **导出 TensorRT 时类别固化**:换类需重新导出;engine 文件不跨设备,须在 Orin 上导
- **DA V2 输出逆深度**(值越大越近),本工作台已统一转成「近红远蓝」显示

## 首次运行自动下载

| 文件 | 大小 | 用途 |
|---|---|---|
| yoloe-26s-seg.pt | 31 MB | YOLOE 权重(已在本地) |
| mobileclip2_b.ts | 242 MB | CLIP 文本编码器(YOLOE 文本提示需要,已在本地) |
| DA V2 Small / Base | 99 MB / 390 MB | HuggingFace 缓存 |

离线使用前先在有网环境各跑一次。

## 部署到 Orin NX 的步骤

1. Orin 上装 JetPack + TensorRT,`pip install ultralytics`
2. 拷贝权重;YOLOE 先 `set_classes(定好类别)`
3. `model.export(format="engine", half=True)`(engine 必须在 Orin 上导出)
4. DA V2 走 transformers → ONNX → TensorRT 路线

## 文件清单

```
core.py                共享内核:设备/模型懒加载/反投影/PLY 导出(app、features、demo 共用)
app.py                 网页推理台(Gradio)
features.py            滑坡监测特征提取(30 项,ROI 模式 44 项)→ CSV,支持 --roi/--roi-auto
segment_gully.py       沟壑分割:颜色阈值 + OpenCV 逐行边界追踪
segment_color.py       Lab 颜色 K-means 分割主坡体(实验性)
visualize3d.py         点云 + 主平面可视化(含 ROI 与侧视图)
validate_features.py   特征有效性验证(合成真值 / 扰动 / 位移 / 动态剔除)
alarm.py               实时报警:变化率/加速度/噪声门控 → 分级
export_features_excel.py  特征导出 Excel(字段清单 + 原始数据)
PIPELINE.md            四层数据源与报警流程说明
FEATURES.md            每个特征字段的详细说明
demo_image.py          图片零样本分割
demo_camera.py         摄像头实时分割
demo_depth.py          DA V2 深度(+ 点云/网格导出)
requirements.txt       依赖
test.jpg / street.jpg / 3.jpg  测试图(含各类结果输出 *_result / *_depth)
pointclouds/           网页台导出的点云/网格默认目录
_deprecated/           弃用的 YOLO26-depth / DA3 脚本、权重与输出(可整目录删除)
```
