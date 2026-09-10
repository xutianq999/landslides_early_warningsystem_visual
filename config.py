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
    "devices": {},                 # 设备元信息: {"<device_id>": {"name":..,"location":..,"rtsp_url":..}}
    "db_path": "data/monitor.db",  # SQLite 文件(相对项目根)
    "images_dir": "images",        # 图片落盘目录(相对项目根)
    "api_host": "127.0.0.1",       # 默认只监听本机;平台要远程拉取再改 0.0.0.0
    "api_port": 8000,
    "api_key": "",                 # 预留:非空则要求请求头 X-API-Key(当前不启用)
}

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


def is_placeholder_device(device_id: str | None) -> bool:
    """是否还是默认占位设备号(提醒现场改成真实编号)"""
    return (device_id or CONFIG["device_id"]) == DEFAULTS["device_id"]
