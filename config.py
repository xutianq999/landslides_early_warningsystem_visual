"""集中配置:默认值 < config.json < 环境变量。

部署到别的机器时复制 config.example.json 为 config.json 按现场改即可;
config.json 不入库(gitignore)。环境变量用于临时覆盖或容器化部署。
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DEFAULTS = {
    "device_id": "site1",          # 当前生效的设备号(点位标识),多节点汇入平台时靠它区分
    "defaults": {},                # 所有设备共用的参数默认值(可被单设备覆盖)
    "devices": {},                 # 每设备配置: {"<device_id>": {name/location/rtsp_url/参数...}}
    "db_path": "data/monitor.db",  # SQLite 文件(相对项目根)
    "images_dir": "images",        # 图片落盘目录(相对项目根)
    "api_host": "127.0.0.1",       # 默认只监听本机;平台要远程拉取再改 0.0.0.0
    "api_port": 8000,
    "api_key": "",                 # 预留:非空则要求请求头 X-API-Key(当前不启用)
}

# 可按设备覆盖的参数(仅作文档与校验;代码用 dc.get(key, 各自内置默认))
DEVICE_PARAM_KEYS = (
    # 采集/推理
    "classes", "conf", "max_depth", "fov", "roi", "roi_auto", "roi_target",
    # 报警(现场必须按点位标定,见 PIPELINE.md)
    "window", "persist", "t1", "t2", "t3",
    # 抓图(供后续 RTSP 采集模块)
    "rtsp_url", "interval_min",
    # 展示用元信息
    "name", "location",
)

_ENV = {
    "device_id": "MONITOR_DEVICE_ID",
    "db_path": "MONITOR_DB",
    "images_dir": "MONITOR_IMAGES_DIR",
    "api_host": "MONITOR_API_HOST",
    "api_port": "MONITOR_API_PORT",
    "api_key": "MONITOR_API_KEY",
}


def _load() -> dict:
    cfg = dict(DEFAULTS)
    f = ROOT / "config.json"
    if f.exists():
        cfg.update(json.loads(f.read_text(encoding="utf-8")))
    for key, env in _ENV.items():
        v = os.environ.get(env)
        if v:
            cfg[key] = int(v) if key == "api_port" else v
    return cfg


CONFIG = _load()


def _under_root(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def db_path(given: str | None = None) -> Path:
    """数据库绝对路径;given 为空时用配置值"""
    return _under_root(given or CONFIG["db_path"])


def images_dir() -> Path:
    return _under_root(CONFIG["images_dir"])


def device_meta(device_id: str | None = None) -> dict:
    """某台设备的元信息(name / location / rtsp_url);没有则空 dict"""
    dev = device_id or CONFIG["device_id"]
    return (CONFIG.get("devices") or {}).get(dev, {})


def for_device(device_id: str | None = None) -> dict:
    """该设备生效的参数:全局 defaults 与设备自身配置合并(设备优先)。

    只包含用户显式写过的键;代码各自用内置默认兜底,避免两处常量重复维护。
    """
    dev = device_id or CONFIG["device_id"]
    merged = dict(CONFIG.get("defaults") or {})
    merged.update((CONFIG.get("devices") or {}).get(dev) or {})
    return merged


def safe_params(device_id: str | None = None) -> dict:
    """对外(API)可见的设备参数:剔除含口令的 rtsp_url"""
    p = for_device(device_id)
    p.pop("rtsp_url", None)
    return p


def is_placeholder_device(device_id: str | None) -> bool:
    """是否还是默认占位设备号(提醒现场改成真实编号)"""
    return (device_id or CONFIG["device_id"]) == DEFAULTS["device_id"]
