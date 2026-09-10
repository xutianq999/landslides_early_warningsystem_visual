"""滑坡监测特征提取:从监控抓图 → 一行特征向量(30 项)→ CSV,供 xLSTM 与报警模块使用。

用法:
    # 单张
    .venv/bin/python features.py 图片.jpg --out features.csv
    # 批量(按文件名排序,自动与上一帧配准做变化检测)
    .venv/bin/python features.py ./snapshots --out features.csv
    # 带降雨(本地 CSV: time,precip_mm 两列 / 或在线拉取)
    .venv/bin/python features.py 图片.jpg --weather-csv rain.csv
    .venv/bin/python features.py 图片.jpg --lat 30.1 --lon 104.2

四层数据源(见 PIPELINE.md):
  ① 图像    分割出滑坡区域面积、动态物体计数 → 直接反映坡体表面变化
  ② 深度图  DA V2 相对深度分布 → 距离结构
  ③ 点云    反投影后的三维几何 → 坡度/平整度/鼓胀
  ④ 时序    与上一帧配准后的深度差 + 变化率/加速度(在 alarm.py 里计算)

动态物体剔除:person/car/truck/construction vehicle 的像素掩码会从深度统计、
几何计算和帧间差异中全部排除,避免人和车路过造成的假变化。
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core import DEVICE, backproject, get_model

DEFAULT_CLASSES = "deep valley,person,car,landslide,truck,construction vehicle"
# 动态物体:其掩码区域从几何/深度/变化统计中剔除
DYNAMIC_CLASSES = ("person", "car", "truck", "construction vehicle")
# 需要面积统计的静态类别(frac/max),动态类别只保留计数
STATIC_CLASSES = ("landslide", "deep valley")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(description="滑坡监测特征提取")
    p.add_argument("input", help="图片路径或目录")
    p.add_argument("--out", default="features.csv", help="输出 CSV(追加写入)")
    p.add_argument("--classes", default=DEFAULT_CLASSES,
                   help=f"YOLOE 分割类别,逗号分隔(默认: {DEFAULT_CLASSES})")
    p.add_argument("--conf", type=float, default=0.15, help="分割置信度阈值")
    p.add_argument("--max-depth", type=float, default=10.0, help="点云近似尺度")
    p.add_argument("--fov", type=float, default=60.0, help="相机水平 FOV(度)")
    p.add_argument("--weather-csv", help="降雨 CSV,列: time,precip_mm")
    p.add_argument("--lat", type=float, help="纬度,在线拉取降雨(Open-Meteo)")
    p.add_argument("--lon", type=float, help="经度,在线拉取降雨")
    p.add_argument("--time", help="本帧时间(如 2026-09-09T09:00),默认取文件名/文件时间")
    p.add_argument("--roi", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"),
                   help="归一化 ROI(如 0.28 0.18 0.72 0.98),只在该区域内算特征")
    p.add_argument("--roi-auto", action="store_true",
                   help="自动检测中央沟壑(segment_gully)并作为 ROI")
    p.add_argument("--roi-target", choices=["gully", "debris", "both"], default="gully",
                   help="配合 --roi-auto:用沟壑 / 底部堆积体 / 两者合并作为 ROI")
    p.add_argument("--no-seg", action="store_true", help="跳过分割(也跳过动态掩码剔除)")
    p.add_argument("--db", nargs="?", const="", default=None, metavar="PATH",
                   help="同时写入 SQLite(可选路径;只写 --db 则用 config.json 里的默认路径)")
    p.add_argument("--device", help="设备号/点位标识(默认取 config.json 的 device_id)")
    return p.parse_args()


# ---------------------------------------------------------------- 工具

def _resize_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    """把布尔掩码缩放到目标 (h, w),最近邻"""
    if mask is None:
        return None
    if mask.shape == shape:
        return mask
    m = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m.astype(bool)


def frame_time(path: Path, override: str | None = None) -> datetime:
    """时间戳:命令行 > 文件名里的 YYYYmmddHHMM/YYYY-mm-dd_HH-MM > 文件修改时间"""
    if override:
        return datetime.fromisoformat(override)
    name = path.stem
    for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y-%m-%d_%H-%M-%S", "%Y-%m-%d_%H-%M"):
        try:
            return datetime.strptime(name, fmt)
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime)


# ---------------------------------------------------------------- ① 图像层

def image_quality(pil: Image.Image) -> dict:
    """图像质量:亮度与清晰度。用于报警前的噪声门控(夜间/雨雾/失焦的帧不可信)。"""
    gray = np.array(pil.convert("L"), dtype=np.float32)
    return {
        "img_brightness": float(gray.mean() / 255.0),          # 0~1
        "img_blur": float(cv2.Laplacian(gray, cv2.CV_32F).var()),  # 拉普拉斯方差,越大越清晰
    }


def segment(pil: Image.Image, classes: list[str], conf: float):
    """YOLOE 零样本分割。

    返回 (统计 dict, 每类掩码 dict{类名: 图像尺寸 bool 数组})。
    掩码用于两件事:统计面积/个数,以及剔除动态物体区域。
    """
    model = get_model("yoloe")
    model.set_classes(classes)
    r = model.predict(source=np.array(pil), conf=conf, device=DEVICE, verbose=False)[0]
    W, H = pil.size
    masks: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {c: 0 for c in classes}

    if r.masks is not None and r.boxes is not None:
        names = [model.names[int(i)] for i in r.boxes.cls]
        for name, m in zip(names, r.masks.data.cpu().numpy()):
            if name not in counts:
                continue
            counts[name] += 1
            m8 = _resize_mask(m > 0.5, (H, W))
            masks[name] = masks[name] | m8 if name in masks else m8

    stats = {}
    for name, mk in masks.items():
        n_lbl, _, st, _ = cv2.connectedComponentsWithStats(mk.astype(np.uint8), 8)
        max_frac = float(st[1:, cv2.CC_STAT_AREA].max() / mk.size) if n_lbl > 1 else 0.0
        if name in STATIC_CLASSES:
            stats[f"seg_{name}_frac"] = float(mk.mean())
            stats[f"seg_{name}_max"] = max_frac
        if name in DYNAMIC_CLASSES:
            stats[f"seg_{name}_n"] = counts[name]
    for name in classes:  # 未检出的也补 0,保证列稳定
        if name in STATIC_CLASSES:
            stats.setdefault(f"seg_{name}_frac", 0.0)
            stats.setdefault(f"seg_{name}_max", 0.0)
        if name in DYNAMIC_CLASSES:
            stats.setdefault(f"seg_{name}_n", 0)
    return stats, masks


def dynamic_mask(masks: dict[str, np.ndarray], shape: tuple[int, int]) -> np.ndarray | None:
    """合并所有动态物体掩码;没有则返回 None"""
    total = None
    for name in DYNAMIC_CLASSES:
        if name in masks:
            m = _resize_mask(masks[name], shape)
            total = m if total is None else (total | m)
    return total


# ---------------------------------------------------------------- ② 深度层

def depth_features(disp01: np.ndarray, ignore: np.ndarray | None = None) -> dict:
    """归一化逆深度(视差)统计。值越大越近。ignore 内的像素(动态物体)排除。"""
    d = disp01.ravel()
    if ignore is not None:
        ig = _resize_mask(ignore, disp01.shape).ravel()
        d = d[~ig]
        if d.size < 100:
            d = disp01.ravel()
    q = np.percentile(d, [5, 50, 95])
    return {"disp_p05": q[0], "disp_p50": q[1], "disp_p95": q[2], "disp_std": float(d.std())}


# ---------------------------------------------------------------- ③ 点云几何层

def geometry_features(pil: Image.Image, disp01: np.ndarray, max_depth: float, fov: float,
                      ignore: np.ndarray | None = None) -> dict:
    """反投影点云 → 三维几何特征(坡度/粗糙度/曲率/主平面/鼓胀)。

    注意:坡度绝对值受 DA V2 仿射歧义影响(有系统偏差),但**单调可区分**陡缓;
    监测看时间序列变化。详见 PIPELINE.md 与 validate_features.py 的 T1 测试。
    """
    keys = ("slope_mean", "slope_p95", "rough_local", "curv_mean",
            "plane_tilt", "plane_rms", "bulge_frac", "edge_depth_corr")
    empty = {k: np.nan for k in keys}

    pts, _, valid = backproject(pil, disp01, max_depth=max_depth, fov_deg=fov)
    ig = _resize_mask(ignore, valid.shape)
    if ig is not None:
        valid = valid & ~ig  # 剔除动态物体

    z = pts[..., 2]
    a = (pts[:, 1:] - pts[:, :-1])[:-1, :]
    b = (pts[1:, :] - pts[:-1, :])[:, :-1]
    nrm = np.cross(a, b)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9
    slope = np.degrees(np.arccos(np.clip(np.abs(nrm @ np.array([0.0, 1.0, 0.0])), 0, 1)))

    cell = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    corners = np.stack([z[:-1, :-1], z[:-1, 1:], z[1:, :-1], z[1:, 1:]])
    cell &= corners.max(0) < corners.min(0) * 2.0
    if cell.sum() < 100:
        return empty
    s = slope[cell]

    # 局部粗糙度
    s_map = np.where(cell, slope, np.nan)
    s_filled = np.nan_to_num(s_map, nan=float(np.nanmean(s_map)))
    m1 = cv2.blur(s_filled, (5, 5))
    m2 = cv2.blur(s_filled ** 2, (5, 5))
    rough = float(np.sqrt(np.maximum(m2 - m1 ** 2, 0))[cell].mean())

    # 曲率与凸起/凹陷
    lap = cv2.Laplacian(np.nan_to_num(z, nan=0.0).astype(np.float32), cv2.CV_32F)[:-1, :-1]
    curv = float(np.abs(lap[cell]).mean() / (max_depth + 1e-6))
    bulge = float((lap[cell] > 0).mean())

    # 深度可信度:图像边缘与深度边缘的一致性
    gray = np.array(pil.convert("L").resize((z.shape[1], z.shape[0])), dtype=np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx ** 2 + gy ** 2)[:-1, :-1]
    zf = np.nan_to_num(z, nan=0.0).astype(np.float32)
    zmag = (np.abs(np.diff(zf, axis=1))[:-1, :] + np.abs(np.diff(zf, axis=0))[:, :-1]) / 2
    corr = (float(np.corrcoef(gmag[cell], zmag[cell])[0, 1])
            if gmag[cell].std() > 0 and zmag[cell].std() > 0 else np.nan)

    # 主平面拟合(PCA,法向固定朝上保证符号稳定)
    P = pts[valid]
    if P.shape[0] > 60000:
        P = P[np.random.default_rng(0).choice(P.shape[0], 60000, replace=False)]
    X = P - P.mean(0)
    _, vecs = np.linalg.eigh(X.T @ X / len(P))
    n_plane = vecs[:, 0]
    if n_plane @ np.array([0.0, 1.0, 0.0]) < 0:
        n_plane = -n_plane
    dist = X @ n_plane
    rms = float(np.sqrt((dist ** 2).mean())) + 1e-9
    tilt = float(np.degrees(np.arccos(np.clip(abs(n_plane @ np.array([0.0, 1.0, 0.0])), 0, 1))))

    return {
        "slope_mean": float(s.mean()), "slope_p95": float(np.percentile(s, 95)),
        "rough_local": rough, "curv_mean": curv,
        "plane_tilt": tilt, "plane_rms": float(rms / (max_depth + 1e-6)),
        "bulge_frac": bulge, "edge_depth_corr": corr,
    }


# ---------------------------------------------------------------- ④ 时序层(帧间)

def align_and_diff(prev: tuple[np.ndarray, np.ndarray], cur: tuple[np.ndarray, np.ndarray],
                   ignore: np.ndarray | None = None, scale: float = 1.0) -> dict:
    """配准上一帧到当前帧(相位相关),统计深度差。动态物体区域排除。

    scale: 配准图相对原图的分辨率比例(原图宽/配准图宽),用于把位移换算回原图像素。
    """
    g_prev, d_prev = prev
    g_cur, d_cur = cur
    g_prev = cv2.resize(g_prev, g_cur.shape[::-1])
    d_prev = cv2.resize(d_prev, d_cur.shape[::-1], interpolation=cv2.INTER_LINEAR)

    (sx, sy), resp = cv2.phaseCorrelate(g_prev.astype(np.float32), g_cur.astype(np.float32))
    M = np.float32([[1, 0, -sx], [0, 1, -sy]])
    d_prev_aligned = cv2.warpAffine(d_prev, M, d_cur.shape[::-1], borderMode=cv2.BORDER_REPLICATE)

    diff = np.abs(d_cur - d_prev_aligned)
    ig = _resize_mask(ignore, diff.shape)
    if ig is not None and (~ig).sum() > 100:
        diff = diff[~ig]
    else:
        diff = diff.ravel()
    return {
        "shift_px": float(np.hypot(sx, sy) * scale),  # 换算回原图像素;相机漂移本身即告警
        "shift_resp": float(resp),                    # 配准可靠度,低则该帧变化特征不可信
        "diff_mean": float(diff.mean()),
        "diff_p95": float(np.percentile(diff, 95)),
        "diff_frac": float((diff > 0.10).mean()),     # 深度变化超 10% 的面积占比
    }


def roi_mask_from_args(pil: Image.Image, roi: list[float] | None) -> np.ndarray | None:
    """归一化 ROI (x1,y1,x2,y2) → 图像尺寸的布尔掩码;None 表示全图"""
    if roi is None:
        return None
    W, H = pil.size
    x1, y1, x2, y2 = roi
    m = np.zeros((H, W), bool)
    m[int(y1 * H):int(y2 * H), int(x1 * W):int(x2 * W)] = True
    return m


def mask_shape_features(mask: np.ndarray | None, prefix: str) -> dict:
    """掩码形状特征:面积、x 方向宽度(最小/最大/波动)、y 方向纵深。

    宽度按行统计(逐行左右边界之差),只统计宽度 ≥1% 图宽的有效行(避免边缘毛刺行被当成最小值)。
    y 方向给出顶/底位置和纵深跨度,都按图像高归一化。
    """
    keys = [f"{prefix}_{k}" for k in
            ("area_frac", "width_min", "width_max", "width_std", "y_top", "y_bottom", "y_extent")]
    if mask is None:
        return {k: np.nan for k in keys}
    m = mask > 0
    if not m.any():
        return {k: 0.0 for k in keys}
    W, H = mask.shape[1], mask.shape[0]
    widths = m.sum(1).astype(float)      # 每行宽度(像素)
    valid = widths >= 0.01 * W           # 只统计有效行
    nz = np.where(valid)[0]
    if nz.size == 0:
        nz = np.where(widths > 0)[0]
    w = widths[nz]
    return {
        f"{prefix}_area_frac": float(m.mean()),
        f"{prefix}_width_min": float(w.min()) / W,
        f"{prefix}_width_max": float(w.max()) / W,
        f"{prefix}_width_std": float(w.std()) / W,   # 宽度沿纵深的波动
        f"{prefix}_y_top": float(nz.min()) / H,
        f"{prefix}_y_bottom": float(nz.max()) / H,
        f"{prefix}_y_extent": float(nz.max() - nz.min()) / H,
    }


def features_from_image(pil: Image.Image, classes: list[str] | None = None, conf: float = 0.15,
                        max_depth: float = 10.0, fov: float = 60.0, prev: tuple | None = None,
                        use_seg: bool = True, mask_dynamic: bool = True,
                        roi: list[float] | None = None, roi_mask: np.ndarray | None = None,
                        extra_masks: dict | None = None):
    """一张 PIL 图 → (30 项特征 dict, 供下一帧配准的 (gray, disp))。网页/命令行共用。

    mask_dynamic=False 可关闭动态物体剔除(仅用于验证剔除效果)。
    roi: 归一化 (x1,y1,x2,y2) 矩形;roi_mask: 直接给图像尺寸的布尔掩码(如自动检测的沟壑)。
    两者都给时 roi_mask 优先。
    """
    classes = classes or [c.strip() for c in DEFAULT_CLASSES.split(",") if c.strip()]

    # ① 图像层:先分割,拿到动态物体掩码
    seg_stats, masks = ({}, {})
    ignore = None
    if use_seg:
        seg_stats, masks = segment(pil, classes, conf)
        if mask_dynamic:
            ignore = dynamic_mask(masks, (pil.size[1], pil.size[0]))

    # ROI:区域外的像素全部忽略(几何/深度/时序只在 ROI 内统计)
    roi_m = roi_mask if roi_mask is not None else roi_mask_from_args(pil, roi)
    if roi_m is not None:
        ignore = ~roi_m if ignore is None else (ignore | ~roi_m)

    # ② 深度层
    depth = np.array(get_model("da2s")(pil)["predicted_depth"])
    disp01 = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

    row = {}
    row.update(image_quality(pil))
    row.update(seg_stats)
    for name, m in (extra_masks or {}).items():  # 各掩模(沟壑/堆积体)的面积与最大宽度
        row.update(mask_shape_features(m, name))
    row.update(depth_features(disp01, ignore))
    row.update(geometry_features(pil, disp01, max_depth, fov, ignore))

    # ④ 时序层
    gray = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)
    scale = pil.size[0] / 640.0  # 配准图固定 640 宽,位移需换算回原图像素
    gray = cv2.resize(gray, (640, int(640 * gray.shape[0] / gray.shape[1])))
    cur = (gray, cv2.resize(disp01.astype(np.float32), (gray.shape[1], gray.shape[0])))
    if prev is not None:
        row.update(align_and_diff(prev, cur, ignore, scale))
    else:
        row.update({k: np.nan for k in ("shift_px", "shift_resp", "diff_mean",
                                        "diff_p95", "diff_frac")})
    return row, cur


# ---------------------------------------------------------------- 降雨

def rain_from_csv(path: str, when: datetime) -> dict:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((datetime.fromisoformat(row["time"].strip()), float(row["precip_mm"])))
    rows.sort()
    return {key: sum(v for t, v in rows if when - timedelta(hours=h) < t <= when)
            for h, key in ((1, "rain_1h"), (24, "rain_24h"), (72, "rain_72h"))}


def rain_from_api(lat: float, lon: float, when: datetime) -> dict:
    import json
    import urllib.request
    start = (when - timedelta(hours=72)).strftime("%Y-%m-%d")
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&hourly=precipitation&past_days=5&start_date={start}&end_date={when:%Y-%m-%d}")
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.load(r)["hourly"]
        times = [datetime.fromisoformat(t) for t in data["time"]]
        precs = data["precipitation"]
        return {key: float(sum(p for t, p in zip(times, precs)
                               if when - timedelta(hours=h) < t <= when and p is not None))
                for h, key in ((1, "rain_1h"), (24, "rain_24h"), (72, "rain_72h"))}
    except Exception as e:
        print(f"[warn] 在线降雨获取失败: {e}", file=sys.stderr)
        return {"rain_1h": np.nan, "rain_24h": np.nan, "rain_72h": np.nan}


# ---------------------------------------------------------------- 主流程

def collect_images(input_path: str) -> list[Path]:
    p = Path(input_path)
    if p.is_dir():
        return sorted([f for f in p.iterdir() if f.suffix.lower() in IMG_EXT])
    return [p]


def extract_one(path: Path, args, prev: tuple | None) -> tuple[dict, tuple]:
    pil = Image.open(path).convert("RGB")
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    roi_mask, extra_masks = None, None
    if getattr(args, "roi_auto", False):
        import segment_gully
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        res = segment_gully.detect_gully(bgr)
        target = getattr(args, "roi_target", "gully")
        key = {"gully": "mask", "debris": "debris", "both": "mask_all"}[target]
        roi_mask = res[key] > 0
        extra_masks = {"gully": res["mask"], "debris": res["debris"]}
    row, cur = features_from_image(pil, classes=classes, conf=args.conf,
                                   max_depth=args.max_depth, fov=args.fov, prev=prev,
                                   use_seg=not args.no_seg, roi=args.roi, roi_mask=roi_mask,
                                   extra_masks=extra_masks)
    out = {"time": frame_time(path, args.time).isoformat(timespec="seconds"), "image": path.name}
    out.update(row)
    if args.weather_csv:
        out.update(rain_from_csv(args.weather_csv, datetime.fromisoformat(out["time"])))
    elif args.lat is not None and args.lon is not None:
        out.update(rain_from_api(args.lat, args.lon, datetime.fromisoformat(out["time"])))
    return out, cur


def main():
    args = parse_args()
    images = collect_images(args.input)
    if not images:
        raise SystemExit(f"没有找到图片: {args.input}")

    rows = []
    prev = None
    for i, path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {path.name} ...", flush=True)
        row, prev = extract_one(path, args, prev)
        rows.append(row)

    cols = ["time", "image"]
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)

    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"已写入 {len(rows)} 行 × {len(cols)} 列 → {args.out}")

    if args.db is not None:
        import config
        import db as dbm
        device = args.device or config.CONFIG["device_id"]
        note = "(仍是占位默认值,建议用 --device 或 config.json 指定真实设备号)" \
            if config.is_placeholder_device(device) else ""
        conn = dbm.connect(args.db or None)
        # 入库时带上图片绝对路径(CSV 里只记文件名,保持原样)
        db_rows = [{**r, "image_path": str(p.resolve())} for r, p in zip(rows, images)]
        n, unknown = dbm.upsert_frames(conn, db_rows, device_id=device)
        conn.close()
        print(f"设备号: {device} {note}".rstrip())
        print(f"已入库 {n} 帧 → {dbm.resolve_db_path(args.db or None)}")
        if unknown:
            print(f"  未知字段存入 extra: {', '.join(sorted(unknown))}")


if __name__ == "__main__":
    main()
