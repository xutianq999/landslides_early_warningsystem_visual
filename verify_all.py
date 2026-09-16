#!/usr/bin/env python3
"""全链路自检:不需要相机、不需要网络,一条命令跑完整个软件。

    .venv/bin/python verify_all.py            # 全跑
    .venv/bin/python verify_all.py -k api     # 只跑名字含 api 的检查
    .venv/bin/python verify_all.py --keep     # 保留临时目录(排查用)

覆盖范围(分组打印):
  A 模块导入    B 命令入口    C 算法       D 存储(schema v4 + 迁移)
  E 数据接口    F 参数链路     G 图形界面    H 端到端(视频 → 运行时 → 库 → API)

所有写入都落在临时目录(用 config 覆盖),不会碰仓库里的 data/ images/ alarms/。
失败会打印原因并让退出码为 1,可直接接进 CI 或部署前自检。
"""

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")     # 界面检查必须离屏

TMP = None                    # 临时工作目录(建好后覆盖 config)
CHECKS = []                   # [(group, name, fn)]
RESULTS = []                  # [(ok, group, name, detail, seconds)]


def check(group: str, name: str):
    def deco(fn):
        CHECKS.append((group, name, fn))
        return fn
    return deco


def _prepare_workspace() -> str:
    """把 config 的落盘位置全部指到临时目录,保证自检不污染真实数据"""
    import config
    d = tempfile.mkdtemp(prefix="verify_")
    config.CONFIG["db_path"] = os.path.join(d, "monitor.db")
    config.CONFIG["images_dir"] = os.path.join(d, "images")
    os.makedirs(config.CONFIG["images_dir"], exist_ok=True)
    return d


def _image_fixture():
    """自检用的图片:优先用仓库里的真实抓图,没有就合成一张有纹理的"""
    import numpy as np
    from PIL import Image
    for name in ("3.jpg", "test.jpg", "street.jpg"):
        p = os.path.join(ROOT, name)
        if os.path.exists(p):
            return Image.open(p).convert("RGB"), p
    rng = np.random.default_rng(0)
    arr = rng.integers(40, 200, (480, 640, 3), dtype=np.uint8)
    return Image.fromarray(arr), "<synthetic>"


def _step_video(path: str, seconds: int = 12, fps: int = 15, step_frame: int = 42) -> str:
    """合成一段"正常 → 突然垮塌"的视频:纹理要够强,否则清晰度门控会整段拒收"""
    import cv2
    import numpy as np
    rng = np.random.default_rng(3)
    h, w = 480, 640
    base = cv2.GaussianBlur(rng.integers(0, 255, (h, w), dtype=np.uint8), (0, 0), 0.8)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(seconds * fps):
        f = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        if i >= step_frame:
            f[120:400, 100:540] = 20
        vw.write(f)
    vw.release()
    return path


# ================================================================ A 模块导入

MODULES = ("core", "config", "db", "framesource", "realtime", "monitor", "tuning",
           "capture", "features", "alarm", "segment_gully", "segment_color",
           "api", "validate_features", "export_features_excel", "visualize3d",
           "gui.imageview", "gui.session", "gui.workers", "gui.mainwindow",
           "gui.runtime_window", "gui.tabs.segment", "gui.tabs.depth",
           "gui.tabs.features", "gui.tabs.capture", "gui.tabs.data")


@check("A 模块导入", "全部模块可导入")
def _a_import():
    import importlib
    bad = []
    for m in MODULES:
        try:
            importlib.import_module(m)
        except Exception as e:
            bad.append(f"{m}: {e}")
    assert not bad, "导入失败:\n    " + "\n    ".join(bad)
    return f"{len(MODULES)} 个模块"


# ================================================================ B 命令入口

CLI_SCRIPTS = ("alarm.py", "api.py", "capture.py", "features.py", "realtime.py",
               "monitor.py", "db.py", "tuning.py", "export_features_excel.py",
               "validate_features.py", "visualize3d.py", "segment_gully.py",
               "segment_color.py", "demo_image.py", "demo_depth.py", "demo_camera.py")


@check("B 命令入口", "各入口 --help 可执行")
def _b_cli():
    import subprocess
    # 只测真正带 argparse 的入口:studio/runtime/app/demo_camera 会直接开窗口或服务
    with_argparse = []
    for s in CLI_SCRIPTS:
        p = os.path.join(ROOT, s)
        if os.path.exists(p) and "argparse" in open(p, encoding="utf-8").read():
            with_argparse.append(s)
    bad = []
    for s in with_argparse:
        r = subprocess.run([sys.executable, s, "--help"], cwd=ROOT,
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or not r.stdout.strip():
            bad.append(f"{s}: rc={r.returncode} {r.stderr.strip()[:200]}")
    assert not bad, "入口异常:\n    " + "\n    ".join(bad)
    return f"{len(with_argparse)} 个入口: " + ", ".join(x[:-3] for x in with_argparse)


# ================================================================ C 算法

@check("C 算法", "沟壑分割 5 种方法都能出掩模")
def _c_gully_methods():
    import numpy as np
    import cv2
    from PIL import Image
    import segment_gully
    pil, src = _image_fixture()
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    areas = {}
    for m in segment_gully.GULLY_METHODS:
        res = segment_gully.detect_gully(bgr, method=m)
        mask = res["mask"]
        assert mask.shape == bgr.shape[:2], f"{m}: 掩模尺寸不对 {mask.shape}"
        assert mask.dtype == bool, f"{m}: 掩模不是布尔 {mask.dtype}"
        for k in ("debris", "left", "right", "gx", "norm"):
            assert k in res, f"{m}: 缺返回字段 {k}"
        areas[m] = round(float(mask.mean()) * 100, 2)
    empty = [m for m, a in areas.items() if a <= 0]
    assert not empty, f"这些方法没检出任何区域: {empty}(面积% {areas})"
    return f"[{src}] 面积% " + " ".join(f"{m}={a}" for m, a in areas.items())


@check("C 算法", "方法/参数真的进到 detect_gully(不是各写一遍吃默认值)")
def _c_params_reach():
    import numpy as np
    import cv2
    from PIL import Image
    import segment_gully
    pil, _ = _image_fixture()
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    a = segment_gully.detect_gully(bgr, method="darkrun")["mask"]
    b = segment_gully.detect_gully(bgr, method="edge")["mask"]
    assert not np.array_equal(a, b), "换方法掩模却没变 → 方法参数没生效"
    c = segment_gully.detect_gully(bgr, method="darkrun", y_range=(0.05, 0.4))["mask"]
    assert not np.array_equal(a, c), "改 y_range 掩模却没变 → 参数没生效"
    # 单一来源:函数默认值必须等于 GULLY_DEFAULTS(以前 CLI 与函数各写一套)
    d = segment_gully.GULLY_DEFAULTS
    assert segment_gully.detect_gully(bgr)["mask"].shape == a.shape
    assert d["method"] == "darkrun"
    return f"方法切换/参数改动都会改变结果(y_range 默认 {d['y_range']})"


@check("C 算法", "坡体颜色分割 detect_slope")
def _c_slope():
    import numpy as np
    import cv2
    from PIL import Image
    import segment_color
    pil, _ = _image_fixture()
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    res = segment_color.detect_slope(bgr, k=5, spatial=25.0, work_width=320)
    for k in ("mask", "labels", "stats", "cluster"):
        assert k in res, f"缺返回字段 {k}"
    assert res["mask"].shape == bgr.shape[:2]
    assert res["mask"].any(), "坡体掩模为空"
    return f"k={len(res['stats'])} 主簇 #{res['cluster']} 面积 {res['mask'].mean():.1%}"


@check("C 算法", "特征提取:基础 30 项 / ROI 模式 44 项")
def _c_features():
    import numpy as np
    import cv2
    from PIL import Image
    import features as F
    import segment_gully
    import db as dbm
    pil, _ = _image_fixture()
    row, cur = F.features_from_image(pil, classes=["person"], conf=0.15, max_depth=12.0,
                                     fov=58.0, use_seg=False)
    n_base = len(row)
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    mask = segment_gully.detect_gully(bgr, method="darkrun")["mask"]
    row2, _ = F.features_from_image(pil, classes=["person"], conf=0.15, max_depth=12.0,
                                    fov=58.0, use_seg=False, roi_mask=mask)
    n_roi = len(row2)
    assert n_base == 30, f"基础特征应为 30 项,实际 {n_base}"
    assert n_roi == 44, f"ROI 特征应为 44 项,实际 {n_roi}"
    unknown = set(row2) - set(dbm.FEATURE_KEYS)
    assert not unknown, f"有特征不在 db.FEATURE_KEYS 里(会写不进库): {sorted(unknown)[:8]}"
    return f"基础 {n_base} 项 / ROI {n_roi} 项,与 db.FEATURE_KEYS 一致"


@check("C 算法", "深度反投影:视差越大越近")
def _c_backproject():
    import numpy as np
    import core
    disp = np.array([[0.1, 0.5, 0.9]], dtype=np.float32)
    z = core.backproject(disp, fov=60.0)
    zz = z[2] if z.ndim == 3 else z
    vals = np.asarray(zz).reshape(-1)[:3] if np.asarray(zz).size >= 3 else np.asarray(zz).reshape(-1)
    assert np.all(np.diff(vals) < 0), f"深度应随视差单调下降,实际 {vals}"
    return f"视差 0.1/0.5/0.9 → 深度 {np.round(vals, 3)}"


@check("C 算法", "光照归一化能消掉整体亮度变化")
def _c_lighting():
    import numpy as np
    import realtime as RT
    rng = np.random.default_rng(0)
    base = rng.integers(40, 200, (240, 320), dtype=np.uint8)
    brighter = np.clip(base.astype(np.int16) + 60, 0, 255).astype(np.uint8)
    raw = RT.compare(brighter, base, None, None, {"lighting": "none", "diff_thresh": 0.20})
    fix = RT.compare(brighter, base, None, None, {"lighting": "mean_std", "diff_thresh": 0.20})
    assert raw[0] > 0.5, f"不归一化应当大面积判为变化,实际 {raw[0]:.2f}"
    assert fix[0] < 0.05, f"归一化后不该还有大面积变化,实际 {fix[0]:.3f}"
    return f"整体 +60 灰阶:不归一化 {raw[0]:.1%} → 归一化 {fix[0]:.1%}"


@check("C 算法", "配准:已知位移能测出来")
def _c_align():
    import numpy as np
    import cv2
    import realtime as RT
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (240, 320), dtype=np.uint8)       # 纯梯度会让相位相关退化
    M = np.float32([[1, 0, 5], [0, 1, 3]])
    shifted = cv2.warpAffine(img, M, (320, 240))
    d, dx, dy, resp = RT.align(img, shifted)
    assert abs(d - np.hypot(5, 3)) < 2.0, f"位移估计偏差过大: {d:.2f} (期望 ~5.83)"
    assert resp > 0.05, f"响应过低,配准不可信: {resp:.3f}"
    return f"真值 5.83 px → 估计 {d:.2f} px (dx={dx:.1f}, dy={dy:.1f}, resp={resp:.2f})"


@check("C 算法", "动态物体:单边遮挡用另一边填充、双边遮挡从分母剔除")
def _c_occlusion():
    import numpy as np
    import realtime as RT
    shape = (100, 100)
    m = np.zeros(shape, bool)
    m[10:30, 10:30] = True                       # 只有当前帧有 → 单边
    only_cur, only_base, both = RT.occluded_from_masks(m, None, 5, shape)
    assert only_cur[20, 20] and not only_base[20, 20] and not both[20, 20], "单边判定错"
    both_m = m.copy()
    only_cur2, only_base2, both2 = RT.occluded_from_masks(m, both_m, 5, shape)
    assert both2[20, 20], "两帧都遮挡的区域应标为无效"
    cf, cm, vf, of, ff = RT.compare(m.astype(np.uint8) * 255, np.zeros(shape, np.uint8),
                                    m, both_m, {"dilate_px": 5})
    assert of > 0 and vf < 1.0, f"双边遮挡应从分母剔除(occluded={of:.2f}, valid={vf:.2f})"
    return f"单边填充 filled={ff:.2f};双边剔除 occluded={of:.2f} valid={vf:.2f}"


@check("C 算法", "判定规则:阶跃报警、回落不报警、重复帧不放大速率")
def _c_decide():
    import realtime as RT
    p = dict(RT.DEFAULTS)

    def series(vals, dt=1.0):
        return [RT.Metrics(ts=1000 + i * dt, change_frac=v, valid_frac=1.0, ok=True)
                for i, v in enumerate(vals)]

    flat = RT.decide(series([0.002] * 8), p)
    assert flat.level == 0, f"平稳序列不该报警,实际 {flat.level_name}"
    step = RT.decide(series([0.002] * 5 + [0.40, 0.40, 0.40]), p)
    assert step.level == 3, f"阶跃应报红,实际 {step.level_name} {step.reasons}"
    fall = RT.decide(series([0.002] * 5 + [0.40, 0.40, 0.40, 0.001, 0.001, 0.001]), p)
    assert fall.level < 3, f"回落到正常不该还是红警,实际 {fall.level_name} {fall.reasons}"
    # 重复帧(ts 相同)必须按"同一帧"处理,否则 dt→0 让速率爆表
    dup = series([0.002] * 4 + [0.40]) + [RT.Metrics(ts=1004.0, change_frac=0.40,
                                                    valid_frac=1.0, ok=True)]
    dv = RT.decide(dup, p)
    assert abs(dv.signals.get("rate", 0)) < 100, f"重复帧把速率算爆了: {dv.signals.get('rate')}"
    return (f"平稳 {flat.level_name} / 阶跃 {step.level_name} / 回落 {fall.level_name} / "
            f"重复帧速率 {dv.signals.get('rate', 0):.3f}")


@check("C 算法", "报警分级 alarm.analyze")
def _c_alarm():
    import numpy as np
    import pandas as pd
    import alarm
    n = 40
    t = pd.date_range("2026-01-01", periods=n, freq="10min").astype(str)
    flat = pd.DataFrame({"time": t, "diff_frac": np.full(n, 0.5)})
    r0 = alarm.analyze(flat, window=24, persist=2, thresholds=(1.5, 3.0, 5.0))
    ramp = np.concatenate([np.full(n - 6, 0.5), np.linspace(1.0, 40.0, 6)])
    rising = pd.DataFrame({"time": t, "diff_frac": ramp})
    r1 = alarm.analyze(rising, window=24, persist=2, thresholds=(1.5, 3.0, 5.0))
    assert r0.iloc[-1]["level"] == 0, f"平稳序列不该报警: {r0.iloc[-1]['level_name']}"
    assert r1.iloc[-1]["level"] > 0, f"突升序列应报警: {r1.iloc[-1]['level_name']}"
    return f"平稳 {r0.iloc[-1]['level_name']} → 突升 {r1.iloc[-1]['level_name']}"


@check("C 算法", "动态物体剔除:图里有车/人时特征不含动态影响(蒙版可关)")
def _c_mask_dynamic():
    import numpy as np
    from PIL import Image
    import features as F
    rng = np.random.default_rng(4)
    arr = rng.integers(40, 200, (480, 640, 3), dtype=np.uint8)
    pil = Image.fromarray(arr)
    row_on, _ = F.features_from_image(pil, classes=["person"], use_seg=False,
                                      mask_dynamic=True)
    row_off, _ = F.features_from_image(pil, classes=["person"], use_seg=False,
                                       mask_dynamic=False)
    assert "dyn_frac" in row_on or "dyn_frac" in row_off, "缺少动态物体占比特征"
    assert set(row_on) == set(row_off), "开关动态剔除不该改变特征项数"
    return f"dyn_frac {row_on.get('dyn_frac')} / {row_off.get('dyn_frac')}(开/关)"


@check("C 算法", "特征有效性验证脚本(合成真值/扰动/位移/动态剔除)")
def _c_validate_script():
    import subprocess
    img = "3.jpg" if os.path.exists(os.path.join(ROOT, "3.jpg")) else "test.jpg"
    r = subprocess.run([sys.executable, "validate_features.py", img], cwd=ROOT,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"退出码 {r.returncode}: {r.stderr.strip()[-300:]}"
    out = r.stdout
    for key, expect in (("T1 几何真值单调性", "通过"), ("T4 动态剔除机制", "通过")):
        line = next((l for l in out.splitlines() if key in l), None)
        assert line, f"输出里找不到 {key}"
        assert expect in line, f"{key} 未通过: {line.strip()}"
    t2 = next((l for l in out.splitlines() if "T2 合理扰动稳定性" in l), None)
    assert t2 and "失败" not in t2, f"T2 扰动稳定性不合格: {t2}"
    return f"[{img}] T1/T4 通过;{t2.strip()}"


# ================================================================ D 存储

@check("D 存储", "建表幂等 + schema 版本")
def _d_init():
    import sqlite3
    import db as dbm
    p = os.path.join(TMP, "d_init.db")
    conn = dbm.connect(p)
    dbm.init_db(conn)
    dbm.init_db(conn)                                  # 幂等:跑两次不报错
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    for t in ("frames", "alarms", "realtime_metrics", "devices", "meta"):
        assert t in tables, f"缺表 {t}"
    ver = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
    assert ver == dbm.SCHEMA_VERSION, f"版本不一致 {ver} != {dbm.SCHEMA_VERSION}"
    conn.close()
    return f"5 张表,version={ver}"


@check("D 存储", "写入幂等 + NaN 存 NULL + 未知字段进 extra")
def _d_upsert_frames():
    import sqlite3
    import db as dbm
    p = os.path.join(TMP, "d_frames.db")
    conn = dbm.connect(p)
    row = {"time": "2026-01-01T00:00:00", "diff_frac": float("nan"), "slope_mean": 0.5,
           "unknown_field_xyz": 7}
    dbm.upsert_frames(conn, [dict(row)], device_id="T-01")
    dbm.upsert_frames(conn, [dict(row)], device_id="T-01")      # 重跑同一帧
    n = conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
    assert n == 1, f"幂等失败:重跑后 {n} 行"
    got = dbm.query_frames(conn, device_id="T-01", limit=1)[0]
    assert got["diff_frac"] is None, f"NaN 应存 NULL,实际 {got['diff_frac']!r}"
    assert "unknown_field_xyz" in (got.get("extra") or ""), "未知字段没进 extra"
    conn.close()
    return f"1 行,NaN→NULL,未知字段进 extra"


@check("D 存储", "报警/实时指标写入与 source 过滤")
def _d_upsert_alarms():
    import db as dbm
    p = os.path.join(TMP, "d_alarms.db")
    conn = dbm.connect(p)
    base = {"device_id": "T-01", "captured_at": "2026-01-01T00:00:00", "valid": 1, "level": 3,
            "level_name": "红色-紧急", "camera_alarm": 0, "detail": {"reasons": ["变化率突增"]}}
    dbm.upsert_alarms(conn, [{**base, "source": "realtime"}])
    dbm.upsert_alarms(conn, [{**base, "captured_at": "2026-01-01T00:10:00",
                              "level": 1, "level_name": "黄色-关注", "source": "analysis",
                              "detail": {}}])
    rt = dbm.query_alarms(conn, device_id="T-01", source="realtime")
    an = dbm.query_alarms(conn, device_id="T-01", source="analysis")
    all_ = dbm.query_alarms(conn, device_id="T-01")
    assert len(rt) == 1 and rt[0]["level"] == 3, f"realtime 过滤错: {rt}"
    assert len(an) == 1 and an[0]["level"] == 1, f"analysis 过滤错: {an}"
    assert len(all_) == 2, f"不传 source 应返回全部,实际 {len(all_)}"
    dbm.upsert_realtime(conn, [{"device_id": "T-01", "captured_at": "2026-01-01T00:00:00",
                                "change_frac": 0.35, "valid_frac": 1.0, "level": 3,
                                "level_name": "红色-紧急", "reasons": ["变化率突增"]}])
    m = dbm.query_realtime(conn, device_id="T-01", min_level=2)
    assert m and abs(m[0]["change_frac"] - 0.35) < 1e-9, f"实时指标读回不对: {m}"
    conn.close()
    return "source 过滤 realtime/analysis/全部 都正确"


@check("D 存储", "老库加列式迁移(缺 params / source 也能升上来)")
def _d_migration():
    import sqlite3
    import db as dbm
    p = os.path.join(TMP, "d_migrate.db")
    conn = sqlite3.connect(p)
    conn.executescript("""
        CREATE TABLE frames (id INTEGER PRIMARY KEY, device_id TEXT, captured_at TEXT);
        CREATE TABLE alarms (device_id TEXT, captured_at TEXT, level INTEGER);
    """)
    conn.execute("INSERT INTO frames (device_id, captured_at) VALUES ('OLD','2020-01-01')")
    conn.commit()
    dbm.init_db(conn)                                   # 应补齐缺失的列而不是重建表
    fcols = {r[1] for r in conn.execute("PRAGMA table_info(frames)")}
    acols = {r[1] for r in conn.execute("PRAGMA table_info(alarms)")}
    assert "params" in fcols, f"frames.params 没补上: {sorted(fcols)[:8]}"
    assert "source" in acols, f"alarms.source 没补上: {sorted(acols)[:8]}"
    kept = conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
    assert kept == 1, "迁移把老数据弄丢了"
    conn.close()
    return "补列成功且老数据保留"


@check("D 存储", "时间序列 / 概况 / 统计查询")
def _d_queries():
    import db as dbm
    p = os.path.join(TMP, "d_query.db")
    conn = dbm.connect(p)
    for i in range(5):
        dbm.upsert_frames(conn, [{"time": f"2026-01-01T00:{i:02d}:00", "diff_frac": 0.1 * i,
                                  "slope_mean": float(i)}], device_id="T-02")
    s = dbm.series(conn, ["diff_frac"], device_id="T-02", order="asc")
    assert len(s) == 5 and s[0]["diff_frac"] == 0.0, f"series 不对: {s[:2]}"
    st = dbm.stats(conn)
    assert st.get("frames", 0) >= 5, f"stats 不对: {st}"
    devs = dbm.list_devices(conn)
    assert any(d["device_id"] == "T-02" for d in devs), "设备列表里没有 T-02"
    latest = dbm.latest_frames(conn, limit=2, device_id="T-02")
    assert len(latest) == 2, f"latest_frames 不对: {len(latest)}"
    fmap = dbm.frame_id_map(conn, "T-02")
    assert "2026-01-01T00:04:00" in fmap, "frame_id_map 不对"
    df = dbm.load_frames_df(conn, device_id="T-02")
    assert len(df) == 5, f"load_frames_df 行数不对: {len(df)}"
    conn.close()
    return f"series/stats/devices/latest/id_map/df 全部正确({len(devs)} 台设备)"


# ================================================================ E 数据接口

def _seed_db(path: str, device: str = "API-01"):
    """种一批数据:特征帧 + 关联图片 + 报警(两个来源)+ 实时指标"""
    import db as dbm
    import numpy as np
    from PIL import Image
    img_dir = os.path.join(TMP, "images", device)
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, "20260101000000.jpg")
    Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(img_path)
    conn = dbm.connect(path)
    dbm.init_db(conn)
    for i in range(3):
        dbm.upsert_frames(conn, [{"time": f"2026-01-01T00:{i:02d}:00", "diff_frac": 0.05 * i,
                                  "slope_mean": 20.0 + i, "image_path": img_path if i == 0 else None}],
                          device_id=device)
    dbm.upsert_alarms(conn, [
        {"device_id": device, "captured_at": "2026-01-01T00:02:00", "valid": 1, "level": 2,
         "level_name": "橙色-预警", "camera_alarm": 0, "source": "realtime", "detail": {}},
        {"device_id": device, "captured_at": "2026-01-01T00:02:00", "valid": 1, "level": 3,
         "level_name": "红色-紧急", "camera_alarm": 0, "source": "analysis", "detail": {}}])
    dbm.upsert_realtime(conn, [{"device_id": device, "captured_at": "2026-01-01T00:02:00",
                                "change_frac": 0.42, "valid_frac": 0.98, "level": 2,
                                "level_name": "橙色-预警", "reasons": ["变化率突增"]}])
    conn.close()
    return img_path


@check("E 数据接口", "全部端点返回正常")
def _e_api_all():
    import config
    from fastapi.testclient import TestClient
    import api
    dev = "API-01"
    img = _seed_db(config.db_path(), dev)
    c = TestClient(api.app)
    hits = []

    def get(url, **kw):
        r = c.get(url, **kw)
        assert r.status_code == 200, f"GET {url} → {r.status_code} {r.text[:200]}"
        hits.append(url)
        return r

    assert get("/api/v1/health").json() is not None
    get("/api/v1/devices")
    get(f"/api/v1/devices/{dev}")
    fr = get(f"/api/v1/frames?device_id={dev}&limit=10").json()
    assert fr["count"] == 3, f"frames 应 3 条,实际 {fr['count']}"
    assert get(f"/api/v1/frames/latest?device_id={dev}&n=1").json()["count"] == 1
    fid = fr["items"][0]["id"]
    get(f"/api/v1/frames/{fid}")
    r = get(f"/api/v1/frames/{fid}/image")
    assert r.headers["content-type"].startswith("image/"), "图片端点没回图片"
    ser = get(f"/api/v1/series?fields=diff_frac,slope_mean&device_id={dev}").json()
    assert ser["count"] == 3, f"series 应 3 条,实际 {ser['count']}"
    al = get(f"/api/v1/alarms?device_id={dev}").json()
    assert al["count"] == 2, f"alarms 应 2 条,实际 {al['count']}"
    assert get(f"/api/v1/alarms?device_id={dev}&source=realtime").json()["count"] == 1
    get(f"/api/v1/alarms/latest?device_id={dev}")
    assert get(f"/api/v1/realtime?device_id={dev}").json()["count"] == 1
    csv = get(f"/api/v1/export.csv?device_id={dev}")
    assert "diff_frac" in csv.text.splitlines()[0], "CSV 表头不对"
    assert len(csv.text.strip().splitlines()) == 4, "CSV 行数不对(表头+3)"
    return f"{len(hits)} 个端点全部 200(含 CSV 与图片)"


@check("E 数据接口", "错误处理:不存在 → 404,路径穿越 → 403")
def _e_api_errors():
    import config
    import db as dbm
    from fastapi.testclient import TestClient
    import api
    c = TestClient(api.app)
    assert c.get("/api/v1/frames/999999").status_code == 404, "不存在的帧应 404"
    assert c.get("/api/v1/devices/NO-SUCH").status_code == 404, "不存在的设备应 404"
    assert c.get("/api/v1/series?fields=no_such_field").json()["count"] == 0, \
        "未注册字段应被 cap 掉(防注入)"
    conn = dbm.connect(config.db_path())
    dbm.upsert_frames(conn, [{"time": "2026-02-02T00:00:00", "diff_frac": 0.1,
                              "image_path": "/etc/passwd"}], device_id="EVIL")
    fid = conn.execute("SELECT id FROM frames WHERE device_id='EVIL'").fetchone()[0]
    conn.close()
    r = c.get(f"/api/v1/frames/{fid}/image")
    assert r.status_code == 403, f"项目目录外的图片应 403,实际 {r.status_code}"
    r2 = c.get("/api/v1/devices")
    assert "rtsp_url" not in json.dumps(r2.json()), "设备接口泄露了 rtsp_url(含口令)"
    return "404 / 403 / 字段 cap / 不回传 rtsp_url 都正确"


# ================================================================ F 参数链路

@check("F 参数链路", "配置文件导出 → 校验 → 加载 往返一致")
def _f_tuning_roundtrip():
    import tuning
    prof = tuning.build("HIK-01")
    prof["realtime"]["baseline_min"] = 0.25
    prof["segmentation"]["method"] = "hybrid"
    prof["analysis"]["fov"] = 66.0
    p = os.path.join(TMP, "prof.json")
    tuning.save(p, prof)
    back = tuning.load(p)
    assert back["realtime"]["baseline_min"] == 0.25
    assert back["segmentation"]["method"] == "hybrid"
    assert back["analysis"]["fov"] == 66.0
    assert tuning.validate(back) == [], f"合法配置却报错: {tuning.validate(back)}"
    bad = json.loads(json.dumps(back))
    bad["segmentation"]["method"] = "不存在的算法"
    errs = tuning.validate(bad)
    assert errs, "非法分割方法应该被校验拦下"
    bad2 = json.loads(json.dumps(back))
    bad2["realtime"]["baseline_min"] = -5
    assert tuning.validate(bad2), "负数基线时长应该被拦下"
    return f"往返一致;非法方法/负数都能拦下(错误 {len(errs)} 条)"


@check("F 参数链路", "配置文件的值真的进到运行时(实时通道 + 分析覆盖)")
def _f_profile_to_runtime():
    import tuning
    import monitor
    prof = tuning.build("HIK-01")
    prof["realtime"].update({"change_t3": 0.99, "exclude_classes": ["person"]})
    prof["segmentation"]["method"] = "grabcut"
    prof["analysis"]["fov"] = 71.0
    r = monitor.make_runtime("HIK-01", os.path.join(TMP, "f_runtime.db"),
                             profile=prof, interval=7.0, use_mask=False,
                             source=os.path.join(TMP, "nonexistent.mp4"))
    try:
        assert r.rt.params["change_t3"] == 0.99, f"实时阈值没生效: {r.rt.params['change_t3']}"
        assert r.rt.params["exclude_classes"] == ["person"], "剔除类别没生效"
        assert r.analysis_overrides.get("fov") == 71.0, "分析参数没生效"
        assert (r.analysis_overrides.get("gully") or {}).get("method") == "grabcut", \
            f"分割方法没折进 gully 覆盖: {r.analysis_overrides.get('gully')}"
        assert r.interval_normal == 7.0
        st = r.status()
        for k in ("level", "level_name", "change_frac", "ready", "history_s", "frames"):
            assert k in st, f"状态快照缺字段 {k}"
    finally:
        r.stop()
    return "阈值/类别/FOV/分割方法全部落到运行时;状态快照字段齐全"


@check("F 参数链路", "设备参数优先级:命令行 > devices[设备] > defaults")
def _f_priority():
    import config
    import features as F
    d = config.CONFIG.get("defaults") or {}
    dev = config.device_ids()[0]
    dv = (config.CONFIG.get("devices") or {}).get(dev) or {}
    p = F.resolve_params(dev)
    for k, v in d.items():
        if k in dv:
            assert p.get(k) == dv[k], f"{k} 应以 devices[{dev}] 为准: {p.get(k)} != {dv[k]}"
    safe = config.safe_params(dev)
    assert "rtsp_url" not in safe, "safe_params 泄露了 rtsp_url"
    return f"{dev} 取值优先 devices 覆盖 defaults;safe_params 不含口令"


# ================================================================ G 图形界面

@contextlib.contextmanager
def _qt_app():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _pump(app, seconds: float = 20.0, until=None):
    """转事件循环直到 until() 为真或超时"""
    t0 = time.time()
    while time.time() - t0 < seconds:
        app.processEvents()
        if until and until():
            return True
        time.sleep(0.02)
    return bool(until and until())


@check("G 图形界面", "工作台主窗口 + 五个标签页")
def _g_studio():
    from gui.mainwindow import MainWindow
    with _qt_app() as app:
        w = MainWindow()
        w.show()
        titles = [w.tabs.tabText(i) for i in range(w.tabs.count())]
        assert len(titles) == 5, f"标签页数量不对: {titles}"
        assert "分割" in titles[0] and "数据" in titles[4], f"标签页不对: {titles}"
        app.processEvents()
        w.close()
    return " · ".join(titles)


@check("G 图形界面", "分割页:5 种方法可切换、结果随之改变、参数导出正确")
def _g_segment_tab():
    import segment_gully
    import numpy as np
    from gui.mainwindow import MainWindow
    with _qt_app() as app:
        w = MainWindow()
        w.session.load_image(os.path.join(ROOT, "3.jpg")
                             if os.path.exists(os.path.join(ROOT, "3.jpg")) else "3.jpg")
        app.processEvents()
        _pump(app, 20, until=lambda: not w.runner.busy() and w.segment_tab._result is not None)
        assert w.segment_tab._result is not None, "载入图片后没有自动出结果"
        seg = w.segment_tab
        labels = [seg.method_combo.itemText(i) for i in range(seg.method_combo.count())]
        assert seg.method_combo.count() == 5, f"方法数量应 5 个,实际 {seg.method_combo.count()}"
        masks = {}
        for i in range(seg.method_combo.count()):
            seg.method_combo.setCurrentIndex(i)
            _pump(app, 20, until=lambda: not w.runner.busy()
                  and seg._result is not None and seg._result[1] ==
                  seg.method_combo.currentData()[1])
            kind, method, res = seg._result
            masks[method or "kmeans"] = (res["mask"] > 0).mean()
        assert len(set(round(v, 6) for v in masks.values())) > 1, \
            f"5 种方法结果完全一样,说明方法没生效: {masks}"
        p = seg.segmentation_params()
        assert p.get("method") in segment_gully.GULLY_METHODS, f"导出的参数不对: {p}"
        assert set(p) >= {"method", "y_range", "x_left", "x_right"}, f"参数缺项: {sorted(p)}"
        # 参数改动要触发重算
        before = seg._result[2]["mask"].mean()
        seg.sp_y0.setValue(0.05)
        seg.sp_y1.setValue(0.35)
        _pump(app, 20, until=lambda: not w.runner.busy())
        after = seg._result[2]["mask"].mean()
        assert abs(after - before) > 1e-9, "改了 y_range 结果却没变"
        w.close()
    return "5 种方法结果互不相同;参数改动触发重算;导出参数完整"


@check("G 图形界面", "配置载入:分割页与特征页都能吃下配置文件")
def _g_apply_profile():
    import tuning
    from gui.mainwindow import MainWindow
    prof = tuning.build("HIK-01")
    prof["segmentation"]["method"] = "edge"
    prof["segmentation"]["y_range"] = [0.2, 0.6]
    prof["analysis"]["fov"] = 63.0
    with _qt_app() as app:
        w = MainWindow()
        w.segment_tab.apply_profile(prof["segmentation"])
        w.features_tab.apply_analysis(prof["analysis"])
        app.processEvents()
        p = w.segment_tab.segmentation_params()
        assert p["method"] == "edge", f"分割方法没套上: {p['method']}"
        assert abs(p["y_range"][1] - 0.6) < 1e-6, f"y_range 没套上: {p['y_range']}"
        a = w.features_tab.analysis_params()
        assert abs(float(a.get("fov", 0)) - 63.0) < 1e-6, f"FOV 没套上: {a.get('fov')}"
        w.close()
    return "method / y_range / fov 都能从配置文件还原到界面"


@check("G 图形界面", "数据页:库概况能刷新出来")
def _g_data_tab():
    import config
    _seed_db(config.db_path(), "GUI-01")
    from gui.mainwindow import MainWindow
    with _qt_app() as app:
        w = MainWindow()
        w.data_tab.refresh()
        app.processEvents()
        w.close()
    return "数据页刷新无异常"


@check("G 图形界面", "运行时监视台:能构造、能按配置文件解析选项、能渲染快照")
def _g_runtime_window():
    import tuning
    prof = tuning.build("HIK-01")
    p = os.path.join(TMP, "gui_prof.json")
    tuning.save(p, prof)
    from gui.runtime_window import RuntimeWindow
    with _qt_app() as app:
        w = RuntimeWindow()
        w.show()
        assert w.table.rowCount() >= 1, "设备表没有行"
        assert w.table.columnCount() == len(__import__("gui.runtime_window",
                                                       fromlist=["COLS"]).COLS)
        w.cb_profile.addItem("t.json", p)
        w.cb_profile.setCurrentIndex(w.cb_profile.count() - 1)
        opts = w._options()
        assert isinstance(opts.get("profile"), dict), "配置文件没被解析进选项"
        assert opts["profile"]["realtime"]["baseline_min"] > 0
        # 喂一个快照,确认渲染路径(含报警横幅)不报错
        fake = {"device": "HIK-01", "enable_realtime": True, "enable_analysis": True,
                "connected": True, "frames": 10, "errors": 0, "reconnects": 0, "fps": 1.0,
                "src_error": "", "history_s": 30.0, "need_s": 600.0, "ready": False,
                "has_verdict": True, "level": 3, "level_name": "红色-紧急",
                "camera_alarm": False, "reasons": ["变化率突增(+0.2/s 超 0.1)"],
                "change_frac": 0.35, "change_mean": 0.1, "valid_frac": 0.99,
                "occluded_frac": 0.0, "shift_px": 0.4, "base_age_s": 600.0,
                "brightness": 0.4, "rate": 0.2, "accel": 0.1, "interval": 10.0,
                "analysis_at": time.time(), "analysis_diff": 0.28, "analysis_shift": 1.0,
                "analysis_rain": None, "analysis_alarm": "红色-紧急",
                "analysis_alarm_level": 3, "error": ""}
        w.on_snapshot([fake])
        app.processEvents()
        assert "报警" in w.banner.text(), f"报警横幅没出来: {w.banner.text()}"
        assert w.table.item(0, 5).text() == "红色-紧急", f"状态列不对: {w.table.item(0,5).text()}"
        assert "35.00%" in w.table.item(0, 6).text(), f"变化率列不对: {w.table.item(0,6).text()}"
        w.close()
    return f"选项解析 + 快照渲染 + 横幅: {w.banner.text()[:24]}"


# ================================================================ H 端到端

@check("H 端到端", "视频 → 运行时(两套通道)→ 数据库 → API 都能读到")
def _h_end_to_end():
    import config
    import db as dbm
    import monitor
    from fastapi.testclient import TestClient
    import api
    video = _step_video(os.path.join(TMP, "step.mp4"), seconds=8)
    prof = {"realtime": {"baseline_min": 0.1, "baseline_tol_s": 30.0, "tick_s": 1.0,
                         "store_interval": 2.0, "save_alarm_frames": False},
            "analysis": {"classes": "deep valley", "conf": 0.15, "roi_auto": False,
                         "fov": 58.0, "max_depth": 12.0}}
    r = monitor.make_runtime("E2E-01", config.db_path(), profile=prof,
                             interval=4.0, interval_rain=2.0, use_mask=False, source=video)
    r.start()
    t0 = time.time()
    seen = {"realtime": 0, "analysis": 0, "alarm": 0}
    try:
        while time.time() - t0 < 40:
            r.step()
            st = r.status()
            if st["has_verdict"]:
                seen["realtime"] = 1
            if st["level"] > 0:
                seen["alarm"] = 1
            if st["analysis_at"]:
                seen["analysis"] = 1
            if all(seen.values()) and time.time() - t0 > 20:
                break
            time.sleep(0.5)
    finally:
        r.stop()
    assert seen["realtime"], "实时通道一直没出判定(门控/预热有问题?)"
    assert seen["analysis"], "分析通道没跑出结果"
    assert seen["alarm"], "阶跃变化没触发报警"
    conn = dbm.connect(config.db_path())
    n_frames = conn.execute("SELECT COUNT(*) FROM frames WHERE device_id='E2E-01'").fetchone()[0]
    n_rt = conn.execute("SELECT COUNT(*) FROM realtime_metrics "
                        "WHERE device_id='E2E-01'").fetchone()[0]
    n_al = conn.execute("SELECT COUNT(*) FROM alarms WHERE device_id='E2E-01'").fetchone()[0]
    conn.close()
    assert n_frames >= 1, f"分析帧没入库: {n_frames}"
    assert n_rt >= 3, f"实时指标没入库: {n_rt}"
    assert n_al >= 1, f"报警没入库: {n_al}"
    c = TestClient(api.app)
    got_f = c.get("/api/v1/frames?device_id=E2E-01").json()
    got_rt = c.get("/api/v1/realtime?device_id=E2E-01").json()
    got_al = c.get("/api/v1/alarms?device_id=E2E-01").json()
    assert got_f["count"] == n_frames, f"API 读到的帧数不一致: {got_f['count']} != {n_frames}"
    assert got_rt["count"] == n_rt, f"API 读到的实时指标不一致: {got_rt['count']}"
    assert got_al["count"] == n_al, f"API 读到的报警不一致: {got_al['count']}"
    assert got_al["items"][0]["level"] >= 1, "报警记录的等级不对"
    return (f"帧 {n_frames} / 实时指标 {n_rt} / 报警 {n_al} 全部入库且 API 可读"
            f"(最高等级 {got_al['items'][0]['level_name']})")


# ================================================================ 入口

def main() -> int:
    global TMP
    ap = argparse.ArgumentParser(description="全链路自检(不需要相机与网络)")
    ap.add_argument("-k", "--filter", help="只跑名字含该子串的检查")
    ap.add_argument("--keep", action="store_true", help="保留临时目录")
    args = ap.parse_args()

    TMP = _prepare_workspace()
    print(f"临时工作目录: {TMP}\n")
    group = None
    failed = []
    for g, name, fn in CHECKS:
        if args.filter and args.filter not in f"{g}{name}":
            continue
        if g != group:
            print(f"\n{g}")
            print("-" * 72)
            group = g
        t0 = time.time()
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                detail = fn()
            ok, msg = True, str(detail)
        except Exception as e:
            ok = False
            msg = f"{e}"
            if not isinstance(e, AssertionError):
                msg = f"{type(e).__name__}: {e}"
            failed.append((g, name, msg, traceback.format_exc()))
        dt = time.time() - t0
        print(f"  {'✓' if ok else '✗'} {name}  ({dt:.1f}s)")
        if msg:
            for line in str(msg).splitlines()[:6]:
                print(f"      {line}")
        RESULTS.append((ok, g, name, msg, dt))

    n_ok = sum(1 for r in RESULTS if r[0])
    print("\n" + "=" * 72)
    print(f"结果: {n_ok}/{len(RESULTS)} 通过,用时 {sum(r[4] for r in RESULTS):.0f}s")
    if failed:
        print("\n失败项:")
        for g, name, msg, tb in failed:
            print(f"  ✗ [{g}] {name}\n      {msg}")
            if os.environ.get("VERIFY_TRACE"):
                print(tb)
    if not args.keep:
        shutil.rmtree(TMP, ignore_errors=True)
    else:
        print(f"临时目录保留在 {TMP}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
