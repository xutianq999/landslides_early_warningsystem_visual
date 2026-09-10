"""SQLite 存储层:建表、写入(幂等 upsert)、查询。标准库 sqlite3,不引入额外依赖。

数据库是事实源;features.csv 保留为兼容导出。

用法:
    python db.py --init                 # 建表(幂等)
    python db.py --stats                # 看库内概况
    python db.py --init --db data/monitor.db

设计要点:
  - **固定超集 schema**:features.py 的全部特征列(含 ROI 与降雨)一律建表,
    该帧缺哪列就写 NULL,避免表结构随运行模式漂移。
  - **自然键 (device_id, captured_at) 唯一**:重复入库同一帧是 upsert,不产生重复行。
  - **NaN / Inf → NULL**:不再像 CSV 那样写成字面 "nan"。
  - 原始列名里的空格转成下划线(如 seg_deep valley_frac → seg_deep_valley_frac),
    便于写 SQL 和放进 URL 查询参数;映射见 FEATURE_KEYS / to_col()。
"""

import argparse
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import config

SCHEMA_VERSION = "4"
META_KEY = "schema_version"

# features.py 输出的特征列(原始名,顺序即建表顺序)
FEATURE_KEYS = [
    # 图像质量
    "img_brightness", "img_blur",
    # 分割(空格会在入库时转下划线)
    "seg_deep valley_frac", "seg_deep valley_max",
    "seg_landslide_frac", "seg_landslide_max",
    "seg_person_n", "seg_car_n", "seg_truck_n", "seg_construction vehicle_n",
    # 深度
    "disp_p05", "disp_p50", "disp_p95", "disp_std",
    # 三维几何
    "slope_mean", "slope_p95", "rough_local", "curv_mean",
    "plane_tilt", "plane_rms", "bulge_frac", "edge_depth_corr",
    # 帧间时序
    "shift_px", "shift_resp", "diff_mean", "diff_p95", "diff_frac",
    # 降雨
    "rain_1h", "rain_24h", "rain_72h",
    # 区域掩模(仅 --roi-auto)
    "gully_area_frac", "gully_width_min", "gully_width_max", "gully_width_std",
    "gully_y_top", "gully_y_bottom", "gully_y_extent",
    "debris_area_frac", "debris_width_min", "debris_width_max", "debris_width_std",
    "debris_y_top", "debris_y_bottom", "debris_y_extent",
]

# 计数类用 INTEGER,其余 REAL
INT_KEYS = {"seg_person_n", "seg_car_n", "seg_truck_n", "seg_construction vehicle_n"}

_ALARM_COLS = ["valid", "score", "top_signal", "top_severity",
               "level", "level_name", "camera_alarm", "source"]
_ALARM_UPDATE = ", ".join(f'"{c}"=excluded."{c}"' for c in _ALARM_COLS)


def to_col(key: str) -> str:
    """原始特征名 → DB 列名(空格转下划线)"""
    return key.replace(" ", "_")


COLS = [to_col(k) for k in FEATURE_KEYS]
KEY_BY_COL = {to_col(k): k for k in FEATURE_KEYS}
INT_COLS = {to_col(k) for k in INT_KEYS}
_RESERVED = {"time", "captured_at", "image", "image_path", "device_id", "params"}


# ---------------------------------------------------------------- 连接 / 建表

def resolve_db_path(path: str | None = None) -> Path:
    return config.db_path(path)


def connect(path: str | None = None) -> sqlite3.Connection:
    p = resolve_db_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # 读(WAL)不阻塞写,平台查询不影响入库
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _feature_ddl() -> str:
    return ",\n".join(
        f'  "{to_col(k)}" {"INTEGER" if to_col(k) in INT_COLS else "REAL"}'
        for k in FEATURE_KEYS
    )


def init_db(conn: sqlite3.Connection) -> None:
    """建表(幂等),并写入 schema_version"""
    conn.executescript(f"""
CREATE TABLE IF NOT EXISTS frames (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device_id TEXT NOT NULL,
  captured_at TEXT NOT NULL,
  image_path TEXT,
  created_at TEXT NOT NULL,
{_feature_ddl()},
  params TEXT,
  extra TEXT,
  UNIQUE(device_id, captured_at)
);
CREATE INDEX IF NOT EXISTS idx_frames_captured ON frames(captured_at);
CREATE INDEX IF NOT EXISTS idx_frames_device_captured ON frames(device_id, captured_at);

CREATE TABLE IF NOT EXISTS alarms (
  device_id TEXT NOT NULL,
  captured_at TEXT NOT NULL,
  frame_id INTEGER,
  valid INTEGER,
  score REAL,
  top_signal TEXT,
  top_severity REAL,
  level INTEGER,
  level_name TEXT,
  camera_alarm INTEGER,
  detail TEXT,
  source TEXT,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (device_id, captured_at)
);
CREATE INDEX IF NOT EXISTS idx_alarms_level ON alarms(level);
CREATE INDEX IF NOT EXISTS idx_alarms_captured ON alarms(captured_at);

-- 实时通道的指标:与分析通道(frames)分表,因为两套语义不同、列数差很多,
-- 混在一张表里会大面积 NULL。平台看"突发"查这张,看"趋势"查 frames。
CREATE TABLE IF NOT EXISTS realtime_metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device_id TEXT NOT NULL,
  captured_at TEXT NOT NULL,
  base_at TEXT,
  base_age_s REAL,
  change_frac REAL,
  change_mean REAL,
  valid_frac REAL,
  occluded_frac REAL,
  filled_frac REAL,
  accel REAL,
  shift_px REAL,
  shift_resp REAL,
  brightness REAL,
  ok INTEGER,
  level INTEGER,
  level_name TEXT,
  camera_alarm INTEGER,
  reasons TEXT,
  params TEXT,
  created_at TEXT NOT NULL,
  UNIQUE(device_id, captured_at)
);
CREATE INDEX IF NOT EXISTS idx_rt_captured ON realtime_metrics(captured_at);
CREATE INDEX IF NOT EXISTS idx_rt_device_captured ON realtime_metrics(device_id, captured_at);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS devices (
  device_id TEXT PRIMARY KEY,
  name TEXT,
  location TEXT,
  rtsp_url TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
""")
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                 (META_KEY, SCHEMA_VERSION))
    _ensure_columns(conn)
    conn.commit()


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """轻量迁移:给已存在的库补上后加的列(幂等;CREATE IF NOT EXISTS 不会加列)"""
    frames_cols = {r[1] for r in conn.execute("PRAGMA table_info(frames)")}
    if "params" not in frames_cols:
        conn.execute("ALTER TABLE frames ADD COLUMN params TEXT")
    alarm_cols = {r[1] for r in conn.execute("PRAGMA table_info(alarms)")}
    if "source" not in alarm_cols:
        conn.execute("ALTER TABLE alarms ADD COLUMN source TEXT")


# ---------------------------------------------------------------- 值清洗

def _num(v):
    """转 float;NaN/Inf/不可解析 → None(写库即 NULL)"""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _int(v):
    f = _num(v)
    return None if f is None else int(round(f))


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- 设备

def register_device(conn: sqlite3.Connection, device_id: str, name=None,
                    location=None, rtsp_url=None) -> None:
    """登记/更新设备元信息(幂等)。只覆盖传入的非空字段。"""
    init_db(conn)
    now = _iso_now()
    conn.execute("""
        INSERT INTO devices (device_id, name, location, rtsp_url, created_at, updated_at)
        VALUES (?,?,?,?,?,?)
        ON CONFLICT(device_id) DO UPDATE SET
          name      = COALESCE(excluded.name, devices.name),
          location  = COALESCE(excluded.location, devices.location),
          rtsp_url  = COALESCE(excluded.rtsp_url, devices.rtsp_url),
          updated_at= excluded.updated_at
    """, (device_id, name, location, rtsp_url, now, now))
    conn.commit()


def register_device_from_config(conn: sqlite3.Connection, device_id: str | None = None) -> str:
    """按 config.json 的 devices 元信息登记设备(没有元信息也登记一条空记录)"""
    dev = device_id or config.CONFIG["device_id"]
    m = config.device_meta(dev)
    register_device(conn, dev, m.get("name"), m.get("location"), m.get("rtsp_url"))
    return dev


# ---------------------------------------------------------------- 写入

def upsert_frames(conn: sqlite3.Connection, rows: list[dict],
                  device_id: str | None = None) -> tuple[int, set[str]]:
    """写入特征行(幂等)。

    rows: features.py / app.py 产出的行,须含 `time`(或 captured_at);
          图片路径用 `image_path`(优先)或 `image`;其余键按 FEATURE_KEYS 识别,
          无法识别的键收进 extra JSON(前向兼容)。
    返回 (写入行数, 未知字段名集合)。
    """
    init_db(conn)
    dev = device_id or config.CONFIG["device_id"]
    register_device_from_config(conn, dev)   # 顺带登记设备元信息(config.devices)
    cols = ["device_id", "captured_at", "image_path", "created_at", "params", *COLS, "extra"]
    quoted = ",".join(f'"{c}"' for c in cols)
    placeholders = ",".join("?" * len(cols))
    update = ", ".join(f'"{c}"=excluded."{c}"'
                       for c in [*COLS, "image_path", "params", "extra", "created_at"])
    sql = (f"INSERT INTO frames ({quoted}) VALUES ({placeholders}) "
           f"ON CONFLICT(device_id, captured_at) DO UPDATE SET {update}")

    now = _iso_now()
    payload, unknown = [], set()
    for r in rows:
        captured = r.get("time") or r.get("captured_at")
        if not captured:
            raise ValueError("特征行缺少 time 字段,无法入库")
        extra = {k: v for k, v in r.items()
                 if k not in _RESERVED and k not in KEY_BY_COL.values()}
        unknown |= set(extra)
        params = r.get("params")
        values = [dev, str(captured), r.get("image_path") or r.get("image"), now,
                  json.dumps(params, ensure_ascii=False, default=str) if params else None]
        values += [_int(r.get(k)) if to_col(k) in INT_COLS else _num(r.get(k))
                   for k in FEATURE_KEYS]
        values.append(json.dumps(extra, ensure_ascii=False, default=str) if extra else None)
        payload.append(values)

    conn.executemany(sql, payload)
    conn.commit()
    return len(payload), unknown


def frame_id_map(conn: sqlite3.Connection, device_id: str) -> dict[str, int]:
    """captured_at → frame id,用于给报警行关联帧"""
    cur = conn.execute("SELECT id, captured_at FROM frames WHERE device_id=?", (device_id,))
    return {r["captured_at"]: r["id"] for r in cur.fetchall()}


def alarm_records(df, device_id: str) -> list[dict]:
    """把 alarm.analyze 的输出 DataFrame 转成可入库的记录(含逐信号 detail)"""
    z_cols = [c for c in df.columns if c.endswith("_z")]
    recs = []
    for _, r in df.iterrows():
        detail = {}
        for c in z_cols:
            s = c[:-2]
            detail[s] = {"z": _num(r.get(c)),
                         "rate": _num(r.get(f"{s}_rate")),
                         "accel": _num(r.get(f"{s}_accel"))}
        recs.append({
            "device_id": device_id,
            "captured_at": r.get("time"),
            "valid": _int(r.get("valid")),
            "score": _num(r.get("score")),
            "top_signal": r.get("top_signal"),
            "top_severity": _num(r.get("top_severity")),
            "level": _int(r.get("level")),
            "level_name": r.get("level_name"),
            "camera_alarm": _int(r.get("camera_alarm")),
            "detail": detail,
        })
    return recs


def upsert_alarms(conn: sqlite3.Connection, records: list[dict],
                  with_frame_id: bool = True) -> int:
    """写入报警判定(按 device_id+captured_at 幂等覆盖)。

    `source` 区分来源:`analysis`(分析通道,趋势)或 `realtime`(实时通道,突发)。
    两套的等级语义不同,平台查询时应带上 source。
    """
    init_db(conn)
    now = _iso_now()
    ids = {}
    for r in records:
        dev = r.get("device_id") or config.CONFIG["device_id"]
        if with_frame_id and dev not in ids:
            ids[dev] = frame_id_map(conn, dev)
        payload = [
            dev, str(r["captured_at"]),
            r.get("frame_id") or ids.get(dev, {}).get(str(r["captured_at"])),
            _int(r.get("valid")), _num(r.get("score")), r.get("top_signal"),
            _num(r.get("top_severity")), _int(r.get("level")), r.get("level_name"),
            _int(r.get("camera_alarm")), r.get("source") or "analysis",
            json.dumps(r.get("detail"), ensure_ascii=False) if r.get("detail") else None,
            now,
        ]
        conn.execute(
            "INSERT INTO alarms (device_id, captured_at, frame_id, valid, score, top_signal,"
            " top_severity, level, level_name, camera_alarm, source, detail, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"
            f" ON CONFLICT(device_id, captured_at) DO UPDATE SET"
            f" frame_id=excluded.frame_id, {_ALARM_UPDATE}, detail=excluded.detail,"
            " updated_at=excluded.updated_at", payload)
    conn.commit()
    return len(records)


_REALTIME_COLS = ["base_at", "base_age_s", "change_frac", "change_mean", "valid_frac",
                  "occluded_frac", "filled_frac", "accel", "shift_px", "shift_resp",
                  "brightness", "ok", "level", "level_name", "camera_alarm", "reasons", "params"]


def upsert_realtime(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """写入实时通道指标(按 device_id+captured_at 幂等覆盖)"""
    init_db(conn)
    now = _iso_now()
    cols = ["device_id", "captured_at", *_REALTIME_COLS, "created_at"]
    quoted = ",".join(f'"{c}"' for c in cols)
    update = ", ".join(f'"{c}"=excluded."{c}"' for c in [*_REALTIME_COLS, "created_at"])
    sql = (f"INSERT INTO realtime_metrics ({quoted}) VALUES ({','.join('?' * len(cols))}) "
           f"ON CONFLICT(device_id, captured_at) DO UPDATE SET {update}")
    payload = []
    for r in rows:
        captured = r.get("captured_at") or r.get("time")
        if not captured:
            raise ValueError("实时指标缺少 captured_at")
        payload.append([
            r.get("device_id") or config.CONFIG["device_id"], str(captured),
            r.get("base_at"), _num(r.get("base_age_s")),
            _num(r.get("change_frac")), _num(r.get("change_mean")), _num(r.get("valid_frac")),
            _num(r.get("occluded_frac")), _num(r.get("filled_frac")), _num(r.get("accel")),
            _num(r.get("shift_px")), _num(r.get("shift_resp")), _num(r.get("brightness")),
            _int(r.get("ok")), _int(r.get("level")), r.get("level_name"),
            _int(r.get("camera_alarm")),
            json.dumps(r.get("reasons"), ensure_ascii=False) if r.get("reasons") else None,
            json.dumps(r.get("params"), ensure_ascii=False) if r.get("params") else None,
            now,
        ])
    conn.executemany(sql, payload)
    conn.commit()
    return len(payload)


def query_realtime(conn, device_id=None, since=None, until=None, min_level=0,
                   limit=100, offset=0, order="desc") -> list[dict]:
    init_db(conn)
    where, params = ["level >= ?"], [int(min_level)]
    if device_id:
        where.append("device_id = ?")
        params.append(device_id)
    _time_filter(where, params, since, until)
    sql = ("SELECT * FROM realtime_metrics WHERE " + " AND ".join(where)
           + f" ORDER BY captured_at {'DESC' if order == 'desc' else 'ASC'} LIMIT ? OFFSET ?")
    params += [int(limit), int(offset)]
    return _dicts(conn.execute(sql, params))


# ---------------------------------------------------------------- 查询

def _dicts(cur) -> list[dict]:
    return [dict(r) for r in cur.fetchall()]


def _time_filter(where: list[str], params: list, since, until):
    if since:
        where.append("captured_at >= ?")
        params.append(str(since))
    if until:
        where.append("captured_at <= ?")
        params.append(str(until))


def query_frames(conn, device_id=None, since=None, until=None,
                 limit=100, offset=0, order="desc", with_extra=False) -> list[dict]:
    init_db(conn)
    where, params = [], []
    if device_id:
        where.append("device_id = ?")
        params.append(device_id)
    _time_filter(where, params, since, until)
    sql = "SELECT * FROM frames"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" ORDER BY captured_at {'DESC' if order == 'desc' else 'ASC'} LIMIT ? OFFSET ?"
    params += [int(limit), int(offset)]
    rows = _dicts(conn.execute(sql, params))
    if not with_extra:
        for r in rows:
            r.pop("extra", None)
    return rows


def latest_frames(conn, limit=1, device_id=None) -> list[dict]:
    return query_frames(conn, device_id=device_id, limit=limit, order="desc")


def get_frame(conn, frame_id: int) -> dict | None:
    init_db(conn)
    r = conn.execute("SELECT * FROM frames WHERE id=?", (int(frame_id),)).fetchone()
    return dict(r) if r else None


def query_alarms(conn, device_id=None, since=None, until=None, min_level=0,
                 limit=100, offset=0, order="desc", source=None) -> list[dict]:
    """source 过滤:`realtime`(突发)或 `analysis`(趋势);不传则两者都有"""
    init_db(conn)
    where, params = ["level >= ?"], [int(min_level)]
    if device_id:
        where.append("device_id = ?")
        params.append(device_id)
    if source:
        where.append("source = ?")
        params.append(source)
    _time_filter(where, params, since, until)
    sql = ("SELECT * FROM alarms WHERE " + " AND ".join(where)
           + f" ORDER BY captured_at {'DESC' if order == 'desc' else 'ASC'} LIMIT ? OFFSET ?")
    params += [int(limit), int(offset)]
    return _dicts(conn.execute(sql, params))


def latest_alarms(conn, device_id=None) -> list[dict]:
    """每台设备最近一条报警判定"""
    init_db(conn)
    sql = """SELECT a.* FROM alarms a
             JOIN (SELECT device_id, MAX(captured_at) mx FROM alarms
                   {where} GROUP BY device_id) b
             ON a.device_id=b.device_id AND a.captured_at=b.mx"""
    params = []
    where = ""
    if device_id:
        where = "WHERE device_id = ?"
        params.append(device_id)
    return _dicts(conn.execute(sql.format(where=where), params))


def list_devices(conn) -> list[dict]:
    """设备概况:设备号 + 元信息 + 帧数/首末上报 + 当前等级。

    已登记但还没有数据的设备也会列出。**不回传 rtsp_url**(含相机口令),
    只给 has_rtsp_url 布尔值。
    """
    init_db(conn)
    stats = {r["device_id"]: dict(r) for r in conn.execute(
        "SELECT device_id, COUNT(*) AS frames, MIN(captured_at) AS first_seen,"
        " MAX(captured_at) AS last_seen FROM frames GROUP BY device_id")}
    meta = {r["device_id"]: dict(r) for r in conn.execute("SELECT * FROM devices")}
    out = []
    for dev in sorted(set(stats) | set(meta)):
        s, m = stats.get(dev, {}), meta.get(dev, {})
        row = conn.execute(
            "SELECT level, level_name, camera_alarm FROM alarms"
            " WHERE device_id=? ORDER BY captured_at DESC LIMIT 1", (dev,)).fetchone()
        out.append({
            "device_id": dev,
            "name": m.get("name"), "location": m.get("location"),
            "has_rtsp_url": bool(m.get("rtsp_url")),
            "frames": s.get("frames", 0),
            "first_seen": s.get("first_seen"), "last_seen": s.get("last_seen"),
            **(dict(row) if row else {"level": None, "level_name": None, "camera_alarm": None}),
        })
    out.sort(key=lambda d: d["last_seen"] or "", reverse=True)
    return out


def get_device(conn, device_id: str) -> dict | None:
    for d in list_devices(conn):
        if d["device_id"] == device_id:
            return d
    return None


def series(conn, fields: list[str], device_id=None, since=None, until=None,
           limit=None, order="asc") -> list[dict]:
    """时间序列:capped 到已注册的特征列,防注入"""
    init_db(conn)
    cols = [c for c in fields if c in KEY_BY_COL]
    if not cols:
        raise ValueError(f"没有有效字段;可用字段见 db.COLS(如 {COLS[0]})")
    where, params = [], []
    if device_id:
        where.append("device_id = ?")
        params.append(device_id)
    _time_filter(where, params, since, until)
    sel = ", ".join(f'"{c}"' for c in ["captured_at", *cols])
    sql = f"SELECT {sel} FROM frames"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" ORDER BY captured_at {'DESC' if order == 'desc' else 'ASC'}"
    if limit:
        sql += " LIMIT ?"
        params.append(int(limit))
    rows = _dicts(conn.execute(sql, params))
    if limit and order == "desc":
        rows.reverse()
    return [{"time": r.pop("captured_at"), **r} for r in rows]


def load_frames_df(conn, device_id=None, limit=500, since=None):
    """取尾部 N 帧为 pandas DataFrame(供 alarm.analyze 重算);列名沿用原始特征名"""
    import pandas as pd
    where, params = [], []
    if device_id:
        where.append("device_id = ?")
        params.append(device_id)
    if since:
        where.append("captured_at >= ?")
        params.append(str(since))
    w = ("WHERE " + " AND ".join(where)) if where else ""
    inner = (f'SELECT captured_at AS "time", device_id, '
             + ", ".join(f'"{c}"' for c in COLS)
             + f" FROM frames {w} ORDER BY captured_at DESC LIMIT ?")
    sql = f"SELECT * FROM ({inner}) ORDER BY \"time\" ASC"
    df = pd.read_sql_query(sql, conn, params=[*params, int(limit)])
    # DB 列名 → 原始特征名(带空格),让 alarm.SIGNALS 能对上
    return df.rename(columns=KEY_BY_COL)


def stats(conn, path: str | None = None) -> dict:
    init_db(conn)
    n_frames = conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
    n_alarms = conn.execute("SELECT COUNT(*) FROM alarms").fetchone()[0]
    n_alert = conn.execute("SELECT COUNT(*) FROM alarms WHERE level > 0").fetchone()[0]
    span = conn.execute("SELECT MIN(captured_at), MAX(captured_at) FROM frames").fetchone()
    ver = conn.execute("SELECT value FROM meta WHERE key=?", (META_KEY,)).fetchone()
    return {"db_path": str(resolve_db_path(path)),
            "schema_version": ver[0] if ver else None,
            "frames": n_frames, "alarms": n_alarms, "alerts": n_alert,
            "first": span[0], "last": span[1],
            "devices": [d["device_id"] for d in list_devices(conn)]}


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description="SQLite 存储层")
    ap.add_argument("--db", help="数据库路径(默认取 config)")
    ap.add_argument("--init", action="store_true", help="建表(幂等)")
    ap.add_argument("--stats", action="store_true", help="打印库内概况")
    args = ap.parse_args()

    conn = connect(args.db)
    if args.init:
        init_db(conn)
        print(f"已初始化 {resolve_db_path(args.db)}(schema v{SCHEMA_VERSION})")
    if args.stats or not args.init:
        s = stats(conn, args.db)
        print(f"数据库: {s['db_path']}  (schema v{s['schema_version']})")
        print(f"帧数: {s['frames']}  报警记录: {s['alarms']}(其中 level>0: {s['alerts']})")
        print(f"时间范围: {s['first']} ~ {s['last']}")
        print(f"设备: {', '.join(s['devices']) if s['devices'] else '(空)'}")
    conn.close()


if __name__ == "__main__":
    main()
