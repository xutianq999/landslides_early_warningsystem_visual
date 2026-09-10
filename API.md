# 数据 API(api.py)

边缘端把每帧特征与分级报警写入 SQLite,由本机 FastAPI 服务通过 HTTP 暴露,平台按需拉取。
**当前不鉴权**(内网使用);`config.json` 里 `api_key` 一旦填上,所有 `/api/v1/*` 接口即要求
请求头 `X-API-Key`。

## 启动

```bash
cp config.example.json config.json     # 按现场改 device_id / db_path / 监听地址
.venv/bin/python db.py --init          # 建表(幂等)
.venv/bin/python api.py                # 默认 127.0.0.1:8000
# 或 uvicorn api:app --host 0.0.0.0 --port 8000
```

交互式文档(FastAPI 自动生成,可直接给平台方):`http://<host>:<port>/docs`

> 默认只监听 `127.0.0.1`。平台要远程拉取,需把 `api_host` 改成 `0.0.0.0`
> ——此时**任何能访问该端口的人都能读到数据**,务必先设 `api_key`。

## 配置(config.json,可用环境变量覆盖)

| 键 | 默认 | 环境变量 | 说明 |
|---|---|---|---|
| `device_id` | `site1` | `MONITOR_DEVICE_ID` | 当前生效的设备号(点位标识),多节点汇入平台时用它区分 |
| `devices` | `{}` | — | 设备元信息 `{"<device_id>": {"name","location","rtsp_url"}}`,自动登记进 `devices` 表 |
| `db_path` | `data/monitor.db` | `MONITOR_DB` | SQLite 文件(相对项目根) |
| `images_dir` | `images` | `MONITOR_IMAGES_DIR` | 网页端抓图落盘目录 |
| `api_host` | `127.0.0.1` | `MONITOR_API_HOST` | 监听地址 |
| `api_port` | `8000` | `MONITOR_API_PORT` | 端口 |
| `api_key` | `""`(不鉴权) | `MONITOR_API_KEY` | 非空即启用 `X-API-Key` 校验 |

## 数据怎么进库

| 入口 | 命令 |
|---|---|
| 命令行批量 | `.venv/bin/python features.py ./snapshots --db --device HIK-01` |
| 命令行指定库 | `.venv/bin/python features.py 3.jpg --db /path/monitor.db` |
| 网页台 | 点「提取滑坡特征」自动落盘图片 + 入库(用 config 默认设备号) |
| 报警重算 | `.venv/bin/python alarm.py --db --device HIK-01`(取尾部 500 帧重算,`--limit` 可调) |

写入是**幂等**的:自然键 `(device_id, captured_at)`,重跑同一帧只覆盖不重复。
数据库是事实源;`features.csv` 仍照常产出,作为兼容导出。

## 设备号(device_id)

每帧都归属一个设备号——后续一路海康 RTSP 流对应一个点位,设备号就是它的标识。

- **指定方式**:CLI 用 `--device`(优先级最高),否则取 `config.json` 的 `device_id`。
  报警重算同样支持 `alarm.py --db --device <号>`。
- **元信息**:在 `config.json` 的 `devices` 里登记名称、位置、RTSP 地址,
  首次入库时自动写进 `devices` 表;平台用 `/api/v1/devices` 查询,
  **已登记但还没数据的设备也会列出**(`frames: 0`)。
- **RTSP 地址不对外**:`rtsp_url` 含相机口令,API 只返回 `has_rtsp_url` 布尔值,
  地址本身只供本机抓图模块使用。

> 默认设备号是占位值 `site1`,上线前请改掉;命令行入库时若仍是占位值会打印提醒。

**后续接入海康 RTSP 的做法**(尚未实现):一个点位配一个 `device_id` 和一条 `rtsp_url`,
定时从 `rtsp://…/Streaming/Channels/101` 抓帧存盘(文件名带时间戳),再调
`features.py <图片> --db --device <号>` 入库即可,无需改动存储与接口层。

## 接口一览(前缀 `/api/v1`)

| 方法 | 路径 | 说明 | 主要参数 |
|---|---|---|---|
| GET | `/health` | 探活:schema 版本、库路径、帧数 | — |
| GET | `/devices` | 设备列表:设备号、名称/位置、帧数、首末上报、当前报警等级 | — |
| GET | `/devices/{device_id}` | 单台设备状态 | — |
| GET | `/frames` | 条件查询特征帧 | `device_id`, `since`, `until`, `limit`, `offset`, `order` |
| GET | `/frames/latest` | 最近 N 帧(看板) | `device_id`, `n` |
| GET | `/frames/{id}` | 单帧全字段 | — |
| GET | `/frames/{id}/image` | 该帧抓图(只读项目目录内文件) | — |
| GET | `/series` | 指定字段的时间序列 | `fields`(逗号分隔), `device_id`, `since`, `until`, `limit` |
| GET | `/alarms` | 报警查询 | `device_id`, `since`, `until`, `min_level`, `limit`, `offset` |
| GET | `/alarms/latest` | 每台设备当前等级 | `device_id` |
| GET | `/export.csv` | 特征帧 CSV 导出 | `device_id`, `since`, `until`, `limit` |

时间统一为 ISO8601 字符串(如 `2026-09-10T10:00:00`),直接做字符串比较即可筛选。

## 示例

```bash
curl -s localhost:8000/api/v1/health
curl -s "localhost:8000/api/v1/frames/latest?n=3"
curl -s "localhost:8000/api/v1/series?fields=diff_frac,slope_mean&since=2026-09-10T00:00:00"
curl -s "localhost:8000/api/v1/alarms?min_level=1&limit=20"
curl -s -o frame.jpg "localhost:8000/api/v1/frames/12/image"
# 启用鉴权后:
curl -s -H "X-API-Key: <你的key>" localhost:8000/api/v1/devices
```

## 字段命名

数据库列名把原始特征名里的**空格换成下划线**,便于写 SQL 和放进 URL:

| features.csv / FEATURES.md | API / 数据库 |
|---|---|
| `seg_deep valley_frac` | `seg_deep_valley_frac` |
| `seg_deep valley_max` | `seg_deep_valley_max` |
| `seg_construction vehicle_n` | `seg_construction_vehicle_n` |

其余字段名不变(共 44 个特征列 + `frames.id`/`device_id`/`captured_at`/`image_path`)。
完整字段含义、单位与危险方向见 [FEATURES.md](FEATURES.md)。

## 表结构

**`frames`** — 每帧一行。固定超集:44 个特征列全部建表(含 14 个 ROI 掩模列、
3 个降雨列、8 个分割列),该帧没有的写 `NULL`(不是 `"nan"`)。无法识别的字段收进 `extra` JSON。
唯一键 `(device_id, captured_at)`。

**`alarms`** — 每帧的报警判定:`valid`、`score`、`top_signal`、`top_severity`、`level`(0~3)、
`level_name`、`camera_alarm`,以及 `detail` JSON(各信号单独存 `z`/`rate`/`accel`)。
主键 `(device_id, captured_at)`。

**`devices`** — 设备号 + 元信息(`name`、`location`、`rtsp_url`),首次入库自动登记。

**`meta`** — `schema_version`,用于后续迁移。

## 重要说明(给平台方)

- **坡度不是真实角度**:DA V2 输出的是相对视差,重建结果存在仿射系统偏差(`d = a/z + b` 的
  `b` 未标定),越远越明显;实测单帧坡度绝对值偏大。**请当作时间序列的相对变化量使用**。
- **位移是像素单位**:现场未做尺度标定,`shift_px` 无法换算成米。
- **首帧时序字段为 `NULL`**:`diff_*` / `shift_*` 依赖与上一帧配准,每段第一帧必为空;
  `level` 在预热期内强制为 0。
- **ROI 掩模列可能整列为空**:只有采集端开启 `--roi-auto` 时才有值。
- **报警等级每行依赖历史**:`alarm.py` 是按整段时间序列批量算的(滚动基线 + 持续性判定
  + 预热期),不是逐帧在线产出;新数据用 `alarm.py --db` 重算尾部窗口即可。

## 暂未提供

- 写入类接口(当前只读;数据由边缘端本地写入)。
- 图片上传 / 对象存储。
- 反向推送:如果平台更希望边缘端主动上报,需要平台方先给出接口契约与鉴权方式。
