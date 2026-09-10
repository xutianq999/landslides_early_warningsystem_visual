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

## 桌面工作台(studio.py)

交互操作统一走这一个入口,五个标签页覆盖全流程;界面只做编排,计算全部调用命令行同一套模块,
结果必然一致。

```bash
.venv/bin/python studio.py
```

| 标签页 | 做什么 |
|---|---|
| ① 分割 | 沟壑分割 **5 种方法**(darkrun/edge/color/hybrid/grabcut)+ 坡体颜色 K-means;参数全暴露、改完自动重算、可导出掩模/叠加图/**参数 JSON** |
| ② 深度 | DA V2 深度图 + 点云、主平面预览,导出 ply 点云/网格 |
| ③ 特征 | 按设备参数提取特征、预览区域检测、一键入库(带设备号与参数) |
| ④ 抓图 | RTSP 预览一帧、抓一帧并入库、定时抓图(无相机时可填本地图片路径当源) |
| ⑤ 数据 | 数据库概况、最近帧、各设备报警等级,一键启停 API 服务、打开接口文档 |

状态栏常驻显示**当前设备号、生效参数、模型来源与推理设备**,避免"不知道现在用的是哪套参数"。

### 各入口的职责(别再混着用)

| 入口 | 定位 |
|---|---|
| `studio.py` | **交互式操作**:调参、看效果、单张验证、手动抓帧 |
| `app.py` | 浏览器版(可远程访问),功能子集 |
| `features.py` / `alarm.py` | 命令行/批量处理,可脚本化 |
| `capture.py` | 无人值守定时抓图(launchd 常驻) |
| `api.py` | 对外数据接口,供平台拉取 |

## 网页推理台(app.py)

左栏传原图,右栏选模型 → 点「开始推理」→ 结果图 + 信息显示在右侧。

| 模型选项 | 能力 | 备注 |
|---|---|---|
| YOLOE 零样本分割 | 文本提示任意类别的实例分割 | 类别框逗号分隔,任意概念,可中文 |
| Depth Anything V2 Small | 相对深度,速度快(127 ms) | 唯一深度模型 |

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

## 特征有效性验证(validate_features.py)

用**已知真值的合成数据 / 受控扰动**检验特征是否可信,不需要灾害标签。在 `3.jpg`(1440×1171)上实测:

| 测试 | 内容 | 结果 |
|---|---|---|
| T1 几何真值 | 合成 0~75° 已知倾角平面 → 反投影后还原坡度与主平面倾角 | 单调通过,最大绝对误差 **1.7°** → 角度可近似当真实坡度用 |
| T2 扰动稳定性 | 亮度 ±20% / JPEG q60 / 高斯噪声 | 10/10 个几何特征相对变化 **<10%**;极端对比度变化 82%(属预期失效,靠质量门控拦截) |
| T3 已知位移 | 图像平移 5.8 / 14.4 / 30 px | `shift_px` 误差 **≤0.4 px**,坡度仅变 1~2% |
| T4 动态剔除 | 仅在动态物体区域制造深度突变,对比剔除前后 `diff_frac` | 假变化 0.0251 → 0.0000,机制有效 |

T1 是**不依赖 DA V2** 的合成测试:它证明反投影本身能精确还原倾角,真实数据里坡度的系统偏差来自 DA V2 的仿射歧义(`d = a/z + b`),不是我们的几何代码。

> 模型加载为只读本地缓存(不联网),见「模型权重与离线部署」。

## 命令行脚本

```bash
# 1. 零样本实例分割(文本提示,任意类别)
.venv/bin/python demo_image.py 图片.jpg --classes "person,bus,helmet"

# 2. 摄像头实时分割(按 q 退出)
.venv/bin/python demo_camera.py --classes "person,phone,cup"

# 3. 相对深度(Depth Anything V2 Small)
.venv/bin/python demo_depth.py 图片.jpg

# 3b. 深度 + 点云/网格导出(.ply,MeshLab/CloudCompare/Blender 可直接打开)
.venv/bin/python demo_depth.py 图片.jpg --ply  --max-depth 15 --fov 60   # 点云
.venv/bin/python demo_depth.py 图片.jpg --mesh --max-depth 15 --fov 60   # 三角网格
```

## 点云 / 网格导出说明

- **原理**:DA V2 相对深度(逆深度/视差)→ 针孔模型反投影:`z = 1/视差` 直接反演(不加偏移,否则陡坡坡度会饱和),再按「场景最远距离」缩放到近似米制;水平 FOV 定焦距(不知道相机 FOV 就用默认 60°)
- **点云**:离散点,远处因透视必然稀疏(3D 点间距 ∝ 距离,面密度按 1/z² 衰减),这是几何决定的,不是丢点
- **网格**:按深度图的规则网格连三角面 → 表面连续,没有稀疏感;相邻四角深度比超过 2 倍不连面,避免把天空和近景缝成斜膜
- **滤波**:双边滤波压深度噪声同时保边(高斯会把物体边界糊成斜面)
- **注意**:DA V2 无绝对尺度,点云的"米数"是近似值,形状/比例正确,绝对距离不可信;要米制得换 DA V2 metric 权重或加相机标定
- **坐标**:相机在原点、y 轴向上、z 轴向前,导出前平移到质心便于查看器取景
- **文件**:二进制 PLY,每点 `x y z float32 + r g b uint8`,网格额外带三角面;超 150 万点自动抽稀
- 网页台:勾选「导出点云/网格」+ 选格式 + 调最远距离 → 推理后可在底部预览并下载

## 模型对比(Mac MPS 实测,热身后)

| 模型 | 速度 | 深度类型 | 边界质量 | 部署友好度 |
|---|---|---|---|---|
| YOLOE-26s-seg | ~37 ms | —(分割) | 好 | ★★★ ultralytics 一条链导 TensorRT |
| DA V2 Small | 127 ms | 相对 | 好 | ★★ 需走 transformers→ONNX |

DA V2 Base 实测无实质收益(Small 相关性 0.996、成图肉眼无差,却慢 2.25×、大 4×),
已弃用,详见「模型权重与离线部署」。
弃用原因:YOLO26-depth 边界质量一般且深度能力被 DA V2 取代;DA3 质量最好但官方生态全在 CUDA、依赖链重,边缘部署不友好。

## 概念速查

- **YOLOE 三种提示模式**:`-seg.pt` 文本/视觉提示(可 `set_classes`);`-seg-pf.pt` 无提示(内置 4585 类词表,不可改)
- **零样本原理**:CLIP 把类别名编码成向量,与图像区域特征算相似度,类别成为推理时输入而非训练时固定输出;CLIP 编码只在 `set_classes` 时算一次
- **导出 TensorRT 时类别固化**:换类需重新导出;engine 文件不跨设备,须在 Orin 上导
- **DA V2 输出逆深度**(值越大越近),本工作台已统一转成「近红远蓝」显示

> `core.py` 加载 DA V2 时用 `local_files_only=True`,**运行时不联网**:优先读项目内
> `models/`,再回退 HuggingFace 本地缓存;两者都没有会立即报错并提示,不会卡在重试上。

## 数据 API 与数据库(api.py / db.py)

特征与报警落 SQLite,由本机 FastAPI 服务暴露 HTTP 接口供平台拉取。**接口与字段说明见 [API.md](API.md)**。

```bash
cp config.example.json config.json     # 改 device_id / devices 元信息 / 监听地址等
.venv/bin/python db.py --init          # 建表
.venv/bin/python features.py 3.jpg --db --device HIK-01   # 提取特征并入库(带设备号)
.venv/bin/python alarm.py --db --device HIK-01            # 尾部窗口重算报警并写回
.venv/bin/python api.py                        # http://127.0.0.1:8000/docs
```

- **设备号与每设备参数**:每帧归属一个 `device_id`(一路海康 RTSP 流 = 一个点位)。
  CLI 用 `--device` 指定,否则取 `config.json`;不同点位可有不同的类别 / FOV / ROI / 报警阈值,
  优先级为 `命令行 > devices[设备号] > defaults > 内置默认`,每帧的计算参数会存进 `frames.params`。
- 写入**幂等**:自然键 `(device_id, captured_at)`,重跑同一帧只覆盖不重复;`NaN` 存 `NULL`。
- 图片留磁盘(`images/<设备号>/`),库里只存路径;`GET /api/v1/frames/{id}/image` 按路径回传。
- 默认只监听 `127.0.0.1`(仅本机);改成 `0.0.0.0` 对局域网开放前,请确认网络边界安全
  ——本服务**不鉴权**,任何能连上端口的人都能读取全部数据。

## 定时抓图(capture.py)

从海康 RTSP 流按点位定时抓帧 → 提取特征 → 入库 → 刷新报警,整条链路自动跑起来。

```bash
# 常驻:按每设备 interval_min 抓(生产用 launchd 拉起,见 deploy/capture.plist.example)
.venv/bin/python capture.py
.venv/bin/python capture.py --log capture.log

# 抓一轮就退出(适合 launchd/cron)
.venv/bin/python capture.py --once
.venv/bin/python capture.py --once --device HIK-01

# 不连相机也能跑通全链路(测试/补录)
.venv/bin/python capture.py --from-file 3.jpg --device HIK-01
```

在 `config.json` 的 `devices.<设备号>` 里配 `rtsp_url`、`interval_min`、`rtsp_transport`、
`enabled`(可选 `lat`/`lon` 自动带降雨特征)。海康地址形如
`rtsp://admin:密码@ip:554/Streaming/Channels/102`(末尾 `102` 是子码流,分辨率低、省带宽更稳)。

- **每次采样独立建连**:RTSP 长连接会积陈旧缓冲、抓到几秒前的画面;新建连接天然拿最新帧。
- **上一帧状态存 `data/state/<设备号>.npz`**:`diff_*`/`shift_*` 需要上一帧做对比,低频抓图进程是
  短命的,所以状态落盘,保证时序特征跨进程连续(首次采样时序字段为空)。
- 抓帧失败会退避重试(2/4/8 秒)后跳过本次,**不中断循环**;某次失败会让下一次的帧间间隔被拉长。
- 多设备串行处理:一块 MPS 上推理排队,采样间隔建议 ≥2 分钟。

## 两套系统:实时报警 + 分析预警

两件事目标不同,共用一条 RTSP 连接(各自建连会占满相机连接数、带宽翻倍):

| | 实时报警(`realtime.py`) | 分析预警(`capture.py`) |
|---|---|---|
| 抓什么 | **突发**(垮塌、落石、急变) | **趋势**(累积形变、缓慢失稳) |
| 频率 | 抽帧 1 Hz,判定 1 Hz | 每 10 s ~ 几分钟 |
| 比较对象 | **N 分钟前那一帧**(默认 10 min,可配) | 历史序列 |
| 算法 | 剔除动态物体 + 光照归一化 + 灰度/边缘差异 | 分割 + 深度 + 点云几何 → 44 项特征 |
| 落库 | `realtime_metrics`(默认 10 s 落一次) | `frames` |
| 存图 | 平时不存,**报警时存证据**(当前帧+基线帧) | 只存特征,图片进 `images/` |

```bash
# 实时报警(常驻)
.venv/bin/python realtime.py --device HIK-01 --db
.venv/bin/python realtime.py --source /path/video.mp4 --baseline-min 0.1 --no-mask   # 无相机试跑
```

**实时通道的关键设计**(每条都在代码注释里写了原因):

- **基线取自下采样灰度环形缓冲**:10 分钟 ≈ 46 MB(存原图要 3.6 GB)。
- **光照归一化必做**:云影/日落会让大面积灰度变化,不归一化必然误报(实测整体变亮 60 灰阶时,
  不做归一化 change_frac=100%,做了之后 0%)。
- **动态物体剔除**:两帧掩码取并集并外扩;单边遮挡用另一边像素填充(差异恒为 0);
  **双边都遮挡的区域标为无效并从分母剔除**,否则比例会被稀释。
- **只报上升**:速率突增或加速为正才报警。用绝对值会把"事件后的回落"也判成红警。
- **相机被碰动**表现为全图都在变 → 配准位移超阈时标记为"相机异常",不计入滑坡判定。
- **清晰度门控在全分辨率上算**:拉普拉斯方差随尺度变化很大,在下采样图上算会把正常帧误判成失焦。

> 阈值(`change_t1/t2/t3`、`rate_limit`、`accel_limit`)是**保守起点**,现场必须用"正常期"
> 数据标定;`DEFAULTS` 在 `realtime.py` 顶部。

## 模型权重与离线部署

权重**随项目文件夹一起走**。新机器上只需拷贝整个目录 + `pip install -r requirements.txt`
(**不要拷 `.venv`**,它是平台相关的)。

| 位置 | 内容 | 大小 | 说明 |
|---|---|---|---|
| 项目根 `yoloe-26s-seg.pt` | YOLOE 分割权重 | 31 MB | 必需 |
| 项目根 `mobileclip2_b.ts` | CLIP 文本编码器 | 242 MB | YOLOE 文本提示必需 |
| `models/da2-small/` | DA V2 Small | 95 MB | 深度模型(唯一保留,理由见下) |

**为什么只留 Small**:实测(1440×1171,MPS)Small 与 Base 的深度图相关性 **0.996**、
成图肉眼无差,但 Base 慢 **2.25×**(286 vs 127 ms)、体积大 4×,特征差异只是整体尺度平移
(不影响时间序列比较),深度边缘对齐甚至略差(0.256 vs 0.271)。故 Base 不再随包提供。

`models/` 里是**扁平文件**(`config.json`、`model.safetensors`、`preprocessor_config.json`),
没有 HuggingFace 缓存那种符号链接,`cp` / `rsync` / 压缩包都能正常搬运。
`models/` 已加入 `.gitignore`,权重不进 git(clone 后需另行拷贝)。

- **必须在项目根目录运行**(`cd` 进去再执行,见「快速开始」)。
  原因:YOLOE 的 `mobileclip2_b.ts` 由 ultralytics 按当前工作目录解析,换目录会触发联网下载。
- 从 HF 缓存重建 `models/`(解引用拷出,不需要联网):

```bash
mkdir -p models/da2-small
cp -L ~/.cache/huggingface/hub/models--depth-anything--Depth-Anything-V2-Small-hf/snapshots/*/* models/da2-small/
```

## 部署到 Orin NX 的步骤

1. Orin 上装 JetPack + TensorRT,`pip install ultralytics`
2. 拷贝权重;YOLOE 先 `set_classes(定好类别)`
3. `model.export(format="engine", half=True)`(engine 必须在 Orin 上导出)
4. DA V2 走 transformers → ONNX → TensorRT 路线

## 文件清单

```
core.py                共享内核:设备/模型懒加载/反投影/PLY 导出(app、features、demo 共用)
studio.py              桌面工作台入口(PySide6):分割/深度/特征/抓图/数据
gui/                   工作台界面:主窗口、后台任务、图像查看器、各标签页
app.py                 网页推理台(Gradio),选设备号可自动套用该点位参数
features.py            滑坡监测特征提取(30 项,ROI 模式 44 项)→ CSV,支持 --roi/--roi-auto
segment_gully.py       沟壑分割:颜色阈值 + OpenCV 逐行边界追踪
segment_color.py       Lab 颜色 K-means 分割主坡体(实验性)
visualize3d.py         点云 + 主平面可视化(含 ROI 与侧视图)
validate_features.py   特征有效性验证(合成真值 / 扰动 / 位移 / 动态剔除)
alarm.py               实时报警:变化率/加速度/噪声门控 → 分级
export_features_excel.py  特征导出 Excel(字段清单 + 原始数据)
config.py              配置:默认值 < config.json < 环境变量
db.py                  SQLite 存储层:建表 / 幂等写入 / 查询
api.py                 本地只读 REST API(FastAPI),供平台拉取
capture.py             海康 RTSP 定时抓图 → 特征 → 入库 → 刷新报警(分析通道)
framesource.py         共享帧源:常驻连接 + 1fps 抽帧 + 环形缓冲 + 门控 + 选帧
realtime.py            实时报警通道:基线比对 → 变化率/速率/加速度 → 报警存证据
deploy/capture.plist.example  launchd 常驻模板(macOS 开机自启/崩溃拉起)
API.md                 接口文档(端点 / 字段映射 / 数据语义说明)
config.example.json    配置模板(复制为 config.json,后者不入库)
PIPELINE.md            四层数据源与报警流程说明
FEATURES.md            每个特征字段的详细说明
demo_image.py          图片零样本分割
demo_camera.py         摄像头实时分割
demo_depth.py          DA V2 深度(+ 点云/网格导出)
requirements.txt       依赖
test.jpg / street.jpg / 3.jpg  测试图(含各类结果输出 *_result / *_depth)
models/                项目内自包含深度权重(da2-small,gitignore)
pointclouds/           网页台导出的点云/网格默认目录
_deprecated/           弃用的 YOLO26-depth / DA3 脚本、权重与输出(可整目录删除)
```
