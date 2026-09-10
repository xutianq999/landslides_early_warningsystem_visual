"""运行时:一台相机一条连接,同时驱动实时报警与分析通道。

为什么要它:`realtime.py` 和 `capture.py` 各自建连的话,同一台相机会被占两条连接,
带宽也翻倍。这里由一个进程持有帧源,两套系统都从它消费。

两套通道的节奏不同,各自独立调度:
  - 实时通道:**每次抽帧(1 Hz)都判定**,轻量,只出变化率/速率/加速度;
  - 分析通道:按 `--interval` 取一帧跑全链(分割+深度+点云几何)入库;降雨期自动加密。

用法:
    .venv/bin/python monitor.py --device HIK-01                 # 单点位
    .venv/bin/python monitor.py --all                           # 配置里所有启用的点位
    .venv/bin/python monitor.py --device HIK-01 --no-analysis   # 只跑实时报警
    .venv/bin/python monitor.py --device HIK-01 --profile profiles/site1.json
    .venv/bin/python monitor.py --all --log monitor.log

生产建议用 launchd 常驻(见 deploy/capture.plist.example,把脚本换成 monitor.py)。
"""

import argparse
import logging
import os
import signal
import threading
import time
from datetime import datetime

import config
import db as dbm
import realtime as RT
from capture import sample_from_source
from framesource import FrameSource

LOG = logging.getLogger("monitor")


class DeviceRuntime:
    """一个点位:一条帧源连接 + 实时通道 + 分析调度(分析跑在子线程,不阻塞实时判定)"""

    def __init__(self, device: str, db_path: str, rt_params: dict, analysis_overrides: dict,
                 interval_normal: float, interval_rain: float, rain_trigger: float,
                 enable_realtime: bool = True, enable_analysis: bool = True,
                 mask_fn=None, pick_sharpest: bool = True):
        meta = config.device_meta(device)
        url = meta.get("rtsp_url")
        if not url:
            raise SystemExit(f"设备 {device} 没有 rtsp_url,请在 config.json 的 devices 里配置")
        self.device = device
        self.db_path = db_path
        self.rt_params = rt_params
        self.analysis_overrides = analysis_overrides
        self.interval_normal = float(interval_normal)
        self.interval_rain = float(interval_rain)
        self.rain_trigger = float(rain_trigger)
        self.enable_realtime = enable_realtime
        self.enable_analysis = enable_analysis
        self.pick_sharpest = pick_sharpest
        self.interval = self.interval_normal

        self.src = FrameSource(url, interval_s=float(rt_params.get("tick_s", 1.0)),
                               buffer_min=max(float(rt_params["baseline_min"]) * 1.2, 0.2),
                               down_size=tuple(rt_params["down_size"]), mask_fn=mask_fn,
                               name=f"fs-{device}")
        self.rt = RT.RealtimeChannel(self.src, device_id=device, params=rt_params,
                                     persist_fn=self._persist)
        self._busy = threading.Lock()
        self.next_analysis = time.time() + 2.0        # 启动后先等帧源攒几帧
        self.last_level = -1

    # ---- 生命周期
    def start(self):
        self.src.start()
        LOG.info("设备 %s 已启动 | 分析间隔 %.0fs(降雨 %.0fs) | 实时判定 %.1fHz",
                 self.device, self.interval_normal, self.interval_rain,
                 1.0 / max(self.rt_params.get("tick_s", 1.0), 1e-3))

    def stop(self):
        self.src.stop()

    # ---- 每个节拍调用
    def step(self):
        if self.enable_realtime:
            try:
                v = self.rt.tick()
                if v is not None:
                    self.last_level = v.level
            except Exception as e:                     # 单次判定失败不该拖垮循环
                LOG.exception("设备 %s 实时判定异常: %s", self.device, e)
        if self.enable_analysis and time.time() >= self.next_analysis and not self._busy.locked():
            threading.Thread(target=self._analysis, name=f"an-{self.device}",
                             daemon=True).start()

    # ---- 分析(子线程)
    def _analysis(self):
        if not self._busy.acquire(blocking=False):
            return
        try:
            row = sample_from_source(self.src, self.device, datetime.now(),
                                     overrides=self.analysis_overrides,
                                     pick_sharpest=self.pick_sharpest, db_path=self.db_path)
            self.interval = self._next_interval(row)
        except Exception as e:
            LOG.exception("设备 %s 分析异常: %s", self.device, e)
        finally:
            self.next_analysis = time.time() + self.interval
            self._busy.release()

    def _next_interval(self, row) -> float:
        """降雨期加密:拿这一帧的 rain_1h 决定下一次间隔(没配降雨就固定用常规间隔)"""
        rain = (row or {}).get("rain_1h")
        if rain is not None and rain == rain and rain >= self.rain_trigger:
            if self.interval != self.interval_rain:
                LOG.info("设备 %s 降雨 %.1fmm ≥ %.1fmm,分析间隔加密到 %.0fs",
                         self.device, rain, self.rain_trigger, self.interval_rain)
            return self.interval_rain
        if self.interval != self.interval_normal:
            LOG.info("设备 %s 降雨回落,分析间隔恢复 %.0fs", self.device, self.interval_normal)
        return self.interval_normal

    # ---- 落库
    def _persist(self, m, v):
        """实时指标落库(默认按 store_interval 采样,等级变化立即落)"""
        now = time.time()
        if v.level < 1 and v.level == self.last_level and \
                now - getattr(self, "_last_store", 0) < self.rt_params.get("store_interval", 10):
            return
        row = {"device_id": self.device, "captured_at": RT._iso(m.ts),
               "base_at": RT._iso(m.ts - m.base_age_s), "base_age_s": m.base_age_s,
               "change_frac": m.change_frac, "change_mean": m.change_mean,
               "valid_frac": m.valid_frac, "occluded_frac": m.occluded_frac,
               "filled_frac": m.filled_frac, "accel": v.signals.get("accel"),
               "shift_px": m.shift_px, "shift_resp": m.shift_resp,
               "brightness": m.brightness, "ok": int(m.ok), "level": v.level,
               "level_name": v.level_name, "camera_alarm": int(v.camera_alarm),
               "reasons": v.reasons,
               "params": {k: self.rt_params.get(k) for k in
                          ("baseline_min", "lighting", "diff_thresh", "change_t1",
                           "change_t2", "change_t3", "rate_limit", "accel_limit")}}
        conn = dbm.connect(self.db_path)
        try:
            dbm.upsert_realtime(conn, [row])
            if v.level > 0 or v.camera_alarm:
                dbm.upsert_alarms(conn, [{
                    "device_id": self.device, "captured_at": row["captured_at"],
                    "valid": int(m.ok), "level": v.level, "level_name": v.level_name,
                    "camera_alarm": int(v.camera_alarm), "source": "realtime",
                    "detail": {"reasons": v.reasons, "signals": v.signals}}])
        finally:
            conn.close()
        self._last_store = now


def select_devices(args) -> list[str]:
    if args.all:
        return [d for d in config.device_ids()
                if config.for_device(d).get("enabled", True) is not False]
    if not args.device:
        return [config.CONFIG["device_id"]]
    out = []
    for chunk in args.device:
        out += [d.strip() for d in chunk.split(",") if d.strip()]
    return out


def main():
    ap = argparse.ArgumentParser(description="运行时:实时报警 + 分析通道(一条连接)")
    ap.add_argument("--device", action="append", help="设备号,可多次或用逗号分隔")
    ap.add_argument("--all", action="store_true", help="配置里所有启用的点位")
    ap.add_argument("--db", nargs="?", const="", default=None, metavar="PATH")
    ap.add_argument("--profile", help="算法参数配置文件(profiles/*.json)")
    ap.add_argument("--interval", type=float, default=10.0, help="分析间隔秒(默认 10)")
    ap.add_argument("--rain-interval", type=float, default=5.0, help="降雨期分析间隔秒(默认 5)")
    ap.add_argument("--rain-trigger", type=float, default=2.0, help="触发加密的 1h 降雨量 mm")
    ap.add_argument("--imgsz", type=int, default=480, help="实时通道分割尺寸(默认 480)")
    ap.add_argument("--no-mask", action="store_true", help="实时通道不做动态物体剔除")
    ap.add_argument("--no-realtime", action="store_true", help="只跑分析通道")
    ap.add_argument("--no-analysis", action="store_true", help="只跑实时通道")
    ap.add_argument("--no-pick-sharpest", action="store_true", help="分析时不选帧,直接用最新帧")
    ap.add_argument("--log", metavar="FILE", help="同时写日志文件")
    args = ap.parse_args()

    handlers = [logging.StreamHandler()]
    if args.log:
        handlers.append(logging.FileHandler(args.log, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s", force=True)

    devices = select_devices(args)
    if not devices:
        raise SystemExit("没有可运行的设备")

    prof = None
    prof_mod = None
    if args.profile:
        import tuning as prof_mod
        prof = prof_mod.load(args.profile)
        LOG.info("已加载配置文件 %s(version=%s)", args.profile, prof.get("version"))

    db_path = str(dbm.resolve_db_path(args.db or None))
    _c = dbm.connect(db_path)
    dbm.init_db(_c)                   # 主动建表:即使还没数据,API 也能立刻查
    _c.close()
    runtimes = []
    for dev in devices:
        if prof:
            rt_params = prof_mod.realtime_params(prof)
            analysis_over = prof_mod.analysis_overrides(prof)
        else:
            rt_params, analysis_over = dict(RT.DEFAULTS), {}
        rt_params.setdefault("store_interval", 10.0)
        rt_params["alarm_dir"] = str(config.ROOT / "alarms")
        mask_classes = rt_params.get("exclude_classes") \
            or "person,car,truck,construction vehicle"
        mask_fn = None if args.no_mask else RT.yoloe_mask_fn(mask_classes, imgsz=args.imgsz)
        try:
            runtimes.append(DeviceRuntime(
                dev, db_path, rt_params=rt_params, analysis_overrides=analysis_over,
                interval_normal=args.interval, interval_rain=args.rain_interval,
                rain_trigger=args.rain_trigger,
                enable_realtime=not args.no_realtime, enable_analysis=not args.no_analysis,
                mask_fn=mask_fn, pick_sharpest=not args.no_pick_sharpest))
        except SystemExit as e:
            LOG.warning("%s", e)

    if not runtimes:
        raise SystemExit("没有可运行的设备(检查 rtsp_url / enabled)")

    stop = threading.Event()

    def _on_signal(_sig, _frm):
        LOG.info("收到退出信号,正在停止…")
        stop.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    for r in runtimes:
        r.start()
    LOG.info("运行时已启动 | 设备: %s", ", ".join(r.device for r in runtimes))
    if not args.no_realtime:
        LOG.info("实时通道需要攒够 %.1f 分钟基线历史才开始判定(预热期不报警)",
                 max(r.rt_params["baseline_min"] for r in runtimes))
    try:
        while not stop.is_set():
            for r in runtimes:
                r.step()
            stop.wait(0.5)
    finally:
        for r in runtimes:
            r.stop()
        LOG.info("已退出")


if __name__ == "__main__":
    main()
