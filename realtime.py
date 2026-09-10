"""实时报警通道:与 N 分钟前的基线帧比对,剔除动态物体,算变化率与加速度,按规则报警。

它和分析通道的分工(见 README「两套系统」):
  - 实时通道:**快、轻**,只出变化率/加速度,默认不存图,报警时才存证据;
  - 分析通道:**全、慢**,每 10 s 跑分割+深度+点云几何,产出完整特征行。
  两者共用 framesource.FrameSource(一条 RTSP 连接),不各自建连。

为什么用灰度而不是深度做比对:深度推理 129 ms/帧,1 fps 下要占 13% 的 MPS;而且 DA V2 是
单目模型,光照一变它的深度估计同样会漂。灰度 + 光照归一化在成本上划算得多。

关键实现细节(每条都踩过坑):
  1. **光照归一化**:云影/日落会让大面积灰度变化 → 不归一化必然误报;
  2. **动态物体剔除**:两帧掩码取并集并外扩;单边遮挡用可见那一边的像素填充;
     双边都遮挡的区域标为无效,**并从分母里剔除**(用全图当分母会稀释比例);
  3. **加速度先平滑**:二阶差分对噪声极敏感,必须先做 3 点中值滤波,否则天天误报;
  4. **相机被碰动**表现为全图都在变 → 配准位移超阈时单独标记,不算滑坡。
"""

import logging
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from framesource import FrameSource

LOG = logging.getLogger("realtime")

# 全部可配(将来收敛进 profile);这里给的是保守起点,现场必须用"正常期"数据标定
DEFAULTS = {
    "tick_s": 1.0,               # 抽帧/判定节拍(秒)
    "store_interval": 10.0,      # 落库间隔秒(判定仍按 tick_s;1Hz 存一年是 3000 万行)
    "exclude_classes": ["person", "car", "truck", "construction vehicle"],  # 只剔这些类
    "baseline_min": 10.0,        # 基线回看时长(分钟)——测的是"这 10 分钟内的变化"
    "baseline_tol_s": 90.0,      # 基线与目标时间的容差,超出说明历史还没攒够
    "down_size": (320, 240),     # 比对用的下采样尺寸
    "dilate_px": 5,              # 掩码外扩像素(分割边界不精确,会留一圈残留)
    "lighting": "clahe",         # none | clahe | mean_std | match
    "diff_thresh": 0.20,         # 单像素差异阈值(归一化 0~1)
    "persist": 2,                # 连续超阈次数才算(去抖)
    "change_t1": 0.05,           # 黄
    "change_t2": 0.12,           # 橙
    "change_t3": 0.25,           # 红
    "rate_limit": 0.10,          # 变化率的一阶差分上限(每秒,只报"上升")
    "accel_limit": 0.02,         # 二阶差分上限(每秒²,只报"加速")
    "accel_persist": 1,          # 1 = 直接报(突发定位);调大更保守
    "shift_px_max": 8.0,         # 配准位移超此值 → 判为"相机异常"
    "min_valid_frac": 0.05,      # 有效像素太少就不判定
    "save_alarm_frames": True,
    "alarm_dir": "alarms",
    "alarm_keep": 200,           # 报警证据图最多保留张数(每张含当前+基线,故为对数×2)
}

LEVEL_NAMES = {0: "正常", 1: "黄色-关注", 2: "橙色-预警", 3: "红色-紧急"}


@dataclass
class Metrics:
    ts: float
    change_frac: float = 0.0     # 有效像素中差异超阈的占比 —— 核心变化率
    change_mean: float = 0.0
    valid_frac: float = 0.0      # 有效像素占比(剔除遮挡后)
    occluded_frac: float = 0.0   # 两帧都被遮挡而剔除的占比
    filled_frac: float = 0.0     # 单边遮挡、用另一边像素填充的占比
    shift_px: float = 0.0
    shift_resp: float = 0.0
    brightness: float = 0.0
    ok: bool = True
    base_age_s: float = 0.0


@dataclass
class Verdict:
    level: int = 0
    level_name: str = "正常"
    camera_alarm: bool = False
    reasons: list = field(default_factory=list)
    signals: dict = field(default_factory=dict)


# ---------------------------------------------------------------- 图像处理

def normalize_lighting(gray: np.ndarray, ref: np.ndarray | None, mode: str) -> np.ndarray:
    """把两张图拉到可比的光照条件下。ref 只在对齐/直方图匹配时用。"""
    if mode == "none":
        return gray
    if mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    if mode == "mean_std":
        g = gray.astype(np.float32)
        return np.clip((g - g.mean()) / (g.std() + 1e-6) * 40 + 128, 0, 255).astype(np.uint8)
    if mode == "match" and ref is not None:
        return _hist_match(gray, ref)
    return gray


def _hist_match(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """把 src 的直方图映射到 ref(cv2 没有现成的,这里手写 LUT)"""
    src_hist = cv2.calcHist([src], [0], None, [256], [0, 256]).ravel()
    ref_hist = cv2.calcHist([ref], [0], None, [256], [0, 256]).ravel()
    src_cdf = np.cumsum(src_hist) / max(src_hist.sum(), 1)
    ref_cdf = np.cumsum(ref_hist) / max(ref_hist.sum(), 1)
    lut = np.interp(src_cdf, ref_cdf, np.arange(256)).astype(np.uint8)
    return cv2.LUT(src, lut)


def align(g_prev: np.ndarray, g_cur: np.ndarray) -> tuple[float, float, float, float]:
    """相位相关估计位移 → (shift_px, dx, dy, resp);响应值低说明配准不可信"""
    a = np.float32(g_prev)
    b = np.float32(g_cur)
    (dx, dy), resp = cv2.phaseCorrelate(a, b)
    return float(np.hypot(dx, dy)), float(dx), float(dy), float(resp)


def occluded_from_masks(mask_cur, mask_base, dilate_px: int, shape: tuple[int, int]):
    """返回 (仅当前遮挡, 仅基线遮挡, 两帧都遮挡) 三个布尔图"""
    h, w = shape
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    cur = np.zeros((h, w), np.uint8) if mask_cur is None else mask_cur.astype(np.uint8)
    base = np.zeros((h, w), np.uint8) if mask_base is None else mask_base.astype(np.uint8)
    cur = cv2.dilate(cur, k) > 0
    base = cv2.dilate(base, k) > 0
    return cur & ~base, base & ~cur, cur & base


def compare(cur_gray: np.ndarray, base_gray: np.ndarray, mask_cur=None, mask_base=None,
            params: dict | None = None) -> tuple[float, float, float, float, float]:
    """两帧比对 → (change_frac, change_mean, valid_frac, occluded_frac, filled_frac)

    只处理单边遮挡并填充;两边都被遮挡的区域不参与统计(否则会稀释比例)。
    """
    p = {**DEFAULTS, **(params or {})}
    only_cur, only_base, both = occluded_from_masks(mask_cur, mask_base,
                                                    int(p["dilate_px"]), cur_gray.shape)
    valid = ~both
    n_valid = int(valid.sum())
    if n_valid < 10:
        return 0.0, 0.0, 0.0, float(both.mean()), 0.0

    c = cur_gray.astype(np.float32)
    b = base_gray.astype(np.float32)
    # 单边遮挡:用可见那一边的像素同时替换两帧 → 该处差异恒为 0,不产生假变化
    if only_cur.any():
        b = np.where(only_cur, c, b)
    if only_base.any():
        c = np.where(only_base, b, c)

    c = normalize_lighting(c.astype(np.uint8), b.astype(np.uint8), p["lighting"]).astype(np.float32)
    b = normalize_lighting(b.astype(np.uint8), c.astype(np.uint8), p["lighting"]).astype(np.float32)
    diff = np.abs(c - b)[valid] / 255.0
    return (float((diff > p["diff_thresh"]).mean()), float(diff.mean()),
            float(valid.mean()), float(both.mean()),
            float((only_cur | only_base).mean()))


# ---------------------------------------------------------------- 判定

def _smooth3(vals: list[float]) -> list[float]:
    """3 点中值滤波:加速度前必须先做,否则单帧抖动就能让二阶差分爆表"""
    out = []
    for i in range(len(vals)):
        lo = max(0, i - 1)
        out.append(float(np.median(vals[lo:i + 2])))
    return out


def decide(series: list[Metrics], params: dict | None = None) -> Verdict:
    """按规则判定等级。series 按时间从旧到新,最后一个是当前帧。"""
    p = {**DEFAULTS, **(params or {})}
    if not series:
        return Verdict()
    cur = series[-1]
    v = Verdict(signals={"change_frac": cur.change_frac, "change_mean": cur.change_mean,
                         "valid_frac": cur.valid_frac, "shift_px": cur.shift_px,
                         "base_age_s": cur.base_age_s})

    if not cur.ok:
        v.reasons.append("帧不可信(亮度/清晰度未过门控)")
        return v
    if cur.valid_frac < p["min_valid_frac"]:
        v.reasons.append(f"有效像素不足({cur.valid_frac:.1%})")
        return v
    if cur.shift_px > p["shift_px_max"]:
        v.camera_alarm = True
        v.reasons.append(f"相机异常(位移 {cur.shift_px:.1f} px),不计入滑坡判定")
        return v

    fr = _smooth3([m.change_frac for m in series])
    persist = int(p["persist"])
    for th, lv in ((p["change_t3"], 3), (p["change_t2"], 2), (p["change_t1"], 1)):
        if len(fr) >= persist and all(x >= th for x in fr[-persist:]):
            v.level = lv
            v.reasons.append(f"变化率 {cur.change_frac:.1%} 连续 {persist} 次 ≥ {th:.1%}")
            break

    # 突发判定:**只报上升**(速率变大 / 加速度为正)。
    # 用绝对值会把"变形速率回落"也判成红警——那是事件后的沉降,不是危险;
    # 而阶跃的上升沿一定会让一阶差分出现正尖峰,所以用"速率"抓突发比用二阶差分更直接。
    # 阈值必须用现场"正常期"数据标定:这里的默认值是保守起点,不是现场值。
    if len(fr) >= 2:
        dt = max(series[-1].ts - series[-2].ts, 1e-3)
        rate = (fr[-1] - fr[-2]) / dt
        v.signals["rate"] = rate
        if rate > p["rate_limit"]:
            v.level = 3
            v.reasons.append(f"变化率突增({rate:+.3f}/s 超 {p['rate_limit']})")
    if len(fr) >= 3:
        dts = [max(series[i].ts - series[i - 1].ts, 1e-3) for i in range(1, len(series))]
        acc = (fr[-1] - 2 * fr[-2] + fr[-3]) / (float(np.mean(dts[-2:])) ** 2)
        v.signals["accel"] = acc
        need = int(p["accel_persist"])
        accs = []
        for i in range(2, len(fr)):
            dt = max(dts[i - 1], 1e-3)
            accs.append((fr[i] - 2 * fr[i - 1] + fr[i - 2]) / (dt * dt))
        if len(accs) >= need and all(a > p["accel_limit"] for a in accs[-need:]):
            v.level = 3
            v.reasons.append(f"加速度过快({acc:+.3f}/s² 超 {p['accel_limit']})")
    v.level_name = LEVEL_NAMES[v.level]
    return v


# ---------------------------------------------------------------- 通道

class RealtimeChannel:
    """把帧源的抽帧结果接上比较与判定;可选落库与报警存图。"""

    def __init__(self, source, device_id: str = "site1", params: dict | None = None,
                 persist_fn=None):
        self.source = source
        self.device_id = device_id
        self.params = {**DEFAULTS, **(params or {})}
        self.persist_fn = persist_fn            # fn(metrics, verdict) 由调用方注入(落库)
        self.history: list[Metrics] = []
        self.last: Verdict | None = None

    def tick(self) -> Verdict | None:
        """比对一次(通常每秒调用)。历史不足(还没攒够基线)时返回 None。"""
        p = self.params
        age = float(p["baseline_min"]) * 60.0
        # 预热:缓冲还没覆盖到基线时长的 80% 就还不该判定,否则会拿"1 秒前"当"10 分钟基线"
        if self.source.history_seconds() < age * 0.8:
            return None
        base = self.source.nearest_before(age, require_ok=True,
                                          tol_s=float(p["baseline_tol_s"]))
        cur = self.source.latest(require_ok=False)
        if base is None or cur is None or cur is base:
            return None                        # 预热期:还没有可用的基线帧
        cf, cm, vf, of, ff = compare(cur.gray, base.gray, cur.mask, base.mask, p)
        m = Metrics(ts=cur.ts, change_frac=cf, change_mean=cm, valid_frac=vf,
                    occluded_frac=of, filled_frac=ff, brightness=cur.brightness,
                    ok=cur.ok, base_age_s=abs(cur.ts - base.ts))
        sh, _dx, _dy, resp = align(base.gray, cur.gray)
        m.shift_px, m.shift_resp = sh, resp

        self.history.append(m)
        maxlen = max(8, int(30 * 60 / max(self.source.interval_s, 0.1)))   # 保留 30 分钟
        if len(self.history) > maxlen:
            self.history = self.history[-maxlen:]

        v = decide(self.history, p)
        self.last = v
        if self.persist_fn:
            try:
                self.persist_fn(m, v)
            except Exception as e:              # 落库失败不该中断报警链路
                LOG.warning("实时指标落库失败: %s", e)
        if v.level > 0 and self.params.get("save_alarm_frames"):
            self._save_evidence(cur, base, v)
        return v

    def _save_evidence(self, cur, base, verdict: Verdict):
        """报警时存证据图:当前帧 + 基线帧(只存一张没法复盘变了什么)"""
        d = os.path.join(self.params["alarm_dir"], self.device_id)
        os.makedirs(d, exist_ok=True)
        stamp = time.strftime("%Y%m%d%H%M%S", time.localtime(cur.ts))
        ts, blur, frame = self.source.latest_full() or (None, None, None)
        if frame is not None:
            cv2.imwrite(os.path.join(d, f"{stamp}_cur.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
        # 基线帧只有下采样灰度,放大保存(它本来就是用来复盘结构变化的)
        cv2.imwrite(os.path.join(d, f"{stamp}_base.png"), base.gray)
        LOG.warning("报警证据已存: %s (等级 %d %s, %s)", d, verdict.level,
                    verdict.level_name, "; ".join(verdict.reasons))
        _prune(d, int(self.params["alarm_keep"]))


def _prune(d: str, keep_pairs: int):
    """按数量滚动清理:每对证据是 *_cur.jpg + *_base.png"""
    try:
        files = sorted(f for f in os.listdir(d) if f.endswith("_cur.jpg"))
        for f in files[:-max(1, keep_pairs)]:
            for suffix in ("_cur.jpg", "_base.png"):
                p = os.path.join(d, f.replace("_cur.jpg", suffix))
                if os.path.exists(p):
                    os.remove(p)
    except Exception as e:
        LOG.warning("清理报警证据失败: %s", e)


# ---------------------------------------------------------------- 动态物体掩码

def yoloe_mask_fn(classes, conf: float = 0.15, imgsz: int = 480):
    """返回一个 mask_fn:对 BGR 帧跑 YOLOE,取指定类别的掩码并集。

    imgsz 默认降到 480:实时通道只是"挖洞",不需要精细边界,省算力(1 fps 下约 4~7% 占用)。
    """
    cls = [c.strip() for c in (classes if isinstance(classes, (list, tuple))
                               else str(classes).split(",")) if c.strip()]

    def fn(frame_bgr):
        from core import DEVICE, get_model
        model = get_model("yoloe")
        model.set_classes(cls)
        r = model.predict(source=frame_bgr, conf=conf, imgsz=imgsz, device=DEVICE,
                          verbose=False)[0]
        if r.masks is None or r.boxes is None:
            return None
        h, w = frame_bgr.shape[:2]
        out = np.zeros((h, w), bool)
        for name, m in zip([model.names[int(i)] for i in r.boxes.cls],
                           r.masks.data.cpu().numpy()):
            if name not in cls:
                continue
            mm = cv2.resize(m > 0.5, (w, h), interpolation=cv2.INTER_NEAREST)
            out |= mm.astype(bool)
        return out if out.any() else None

    return fn


# ---------------------------------------------------------------- 入口

def main():
    """常驻实时报警:帧源每秒抽帧,按 10 分钟基线比对,报警时存证据图。

    判定是 1 Hz,但**落库默认 10 秒一次**(等级变化时立即落)——1 Hz 存一年是 3000 万行,
    没必要;曲线看 10 秒粒度足够,报警时刻又不会漏。
    """
    import argparse
    import config
    import db as dbm

    ap = argparse.ArgumentParser(description="实时报警通道(与 N 分钟前基线比对)")
    ap.add_argument("--device", help="设备号(默认取 config)")
    ap.add_argument("--source", help="RTSP 地址(默认取该设备配置;也可填本地视频文件)")
    ap.add_argument("--db", nargs="?", const="", default=None, metavar="PATH",
                    help="写入 SQLite(可选路径;只写 --db 用 config 默认)")
    ap.add_argument("--interval", type=float, default=1.0, help="抽帧间隔秒(默认 1)")
    ap.add_argument("--baseline-min", type=float, default=None, help="基线回看分钟(默认 10)")
    ap.add_argument("--store-interval", type=float, default=10.0, help="落库间隔秒(默认 10)")
    ap.add_argument("--no-mask", action="store_true", help="不做动态物体剔除(无 YOLOE 时)")
    ap.add_argument("--classes", default="person,car,truck,construction vehicle",
                    help="要剔除的类别(逗号分隔,只剔这些)")
    ap.add_argument("--imgsz", type=int, default=480, help="实时通道分割尺寸(默认 480)")
    ap.add_argument("--no-save-frames", action="store_true", help="报警时不存证据图")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = args.device or config.CONFIG["device_id"]
    meta = config.device_meta(device)
    url = args.source or meta.get("rtsp_url")
    if not url:
        raise SystemExit(f"设备 {device} 没有 rtsp_url,也没给 --source")

    params = dict(DEFAULTS)
    if args.baseline_min is not None:
        params["baseline_min"] = args.baseline_min
    params["baseline_tol_s"] = max(5.0, params["baseline_min"] * 60 * 0.1)
    params["save_alarm_frames"] = not args.no_save_frames
    params["alarm_dir"] = str(config.ROOT / "alarms")

    mask_fn = None if args.no_mask else yoloe_mask_fn(args.classes, imgsz=args.imgsz)
    if mask_fn:
        LOG.info("动态物体剔除已启用,类别: %s (imgsz=%d)", args.classes, args.imgsz)
    else:
        LOG.warning("未启用动态物体剔除(--no-mask):人车经过会造成假变化")

    conn = dbm.connect(args.db or None)
    store_s = float(args.store_interval)
    last_store = 0.0
    last_level = -1

    def persist(m: Metrics, v: Verdict):
        nonlocal last_store, last_level
        now = time.time()
        if v.level < 1 and now - last_store < store_s and v.level == last_level:
            return                                    # 常态:按落库间隔采样
        row = {"device_id": device, "captured_at": _iso(m.ts), "base_at": _iso(m.ts - m.base_age_s),
               "base_age_s": m.base_age_s, "change_frac": m.change_frac, "change_mean": m.change_mean,
               "valid_frac": m.valid_frac, "occluded_frac": m.occluded_frac,
               "filled_frac": m.filled_frac, "accel": v.signals.get("accel"),
               "shift_px": m.shift_px, "shift_resp": m.shift_resp, "brightness": m.brightness,
               "ok": int(m.ok), "level": v.level, "level_name": v.level_name,
               "camera_alarm": int(v.camera_alarm), "reasons": v.reasons, "params": params}
        dbm.upsert_realtime(conn, [row])
        if v.level > 0 or v.camera_alarm:
            dbm.upsert_alarms(conn, [{"device_id": device, "captured_at": row["captured_at"],
                                      "valid": int(m.ok), "level": v.level,
                                      "level_name": v.level_name, "camera_alarm": int(v.camera_alarm),
                                      "source": "realtime",
                                      "detail": {"reasons": v.reasons, "signals": v.signals}}])
        last_store, last_level = now, v.level

    src = FrameSource(url, interval_s=args.interval,
                      buffer_min=max(params["baseline_min"] * 1.2, 0.1),
                      down_size=tuple(params["down_size"]), mask_fn=mask_fn)
    ch = RealtimeChannel(src, device_id=device, params=params, persist_fn=persist)
    src.start()
    LOG.info("实时报警启动 | 设备 %s | 基线 %.1f 分钟 | 抽帧 %.1fs | 落库 %.0fs",
             device, params["baseline_min"], args.interval, store_s)
    LOG.info("预热中:需要攒够 %.1f 分钟的基线历史才会开始判定", params["baseline_min"])
    try:
        while True:
            time.sleep(args.interval)
            v = ch.tick()
            if v is None:
                continue
            LOG.info("变化率 %.2f%% | 有效 %.0f%% | 位移 %.1fpx | %s%s",
                     ch.history[-1].change_frac * 100, ch.history[-1].valid_frac * 100,
                     ch.history[-1].shift_px, v.level_name,
                     (" | " + "; ".join(v.reasons)) if v.reasons else "")
    except KeyboardInterrupt:
        LOG.info("收到中断,退出")
    finally:
        src.stop()
        conn.close()


def _iso(ts: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
