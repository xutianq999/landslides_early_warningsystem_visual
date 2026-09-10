"""共享帧源:相机只用一条 RTSP 连接,两套系统都从这里取帧。

为什么需要它(而不是各自建连):
  - RTSP 建连要 1~2 秒,**1 fps 抽帧靠"每次重新建连"根本做不到**;
  - 两套系统各自建连会占用相机连接数(海康有上限),带宽也翻倍;
  - 陈旧缓冲、断流重连这些脏活只该在一处处理。

它提供三样东西:
  1. **常驻连接 + 1 fps 抽帧**:用 grab() 丢弃、retrieve() 解码,只解需要的那一帧;
  2. **环形缓冲(下采样灰度)**:给实时通道找"N 分钟前"的基线帧。存下采样灰度而不是
     原图,10 分钟约 46 MB(原图要 3.6 GB);
  3. **近若干帧的原图 + 清晰度**:给分析通道"挑最清晰的一帧"送去推理(深度对模糊很敏感)。

质量门控(亮度/清晰度)在这里做,因为它决定一帧能不能参与对比与判定。
"""

import collections
import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

LOG = logging.getLogger("framesource")

# 质量门控默认值(与 alarm.GATE 口径一致:夜间/雨雾/失焦的帧不可信)
# 注意 blur_min 是**在参考分辨率下**标定的:拉普拉斯方差随分辨率变化很大
# (同一场景,640×480 的方差可能只有 1440×1171 的 1/5),所以要把阈值按像素数缩放,
# 否则换成子码流后所有帧都会被判"失焦",实时通道会静默地一条判定都不做。
GATE_DEFAULTS = {
    "brightness_min": 0.12,
    "brightness_max": 0.95,
    "blur_min": 40.0,                    # 参考分辨率下的阈值
    "blur_ref_pixels": 1440 * 1171,      # 参考分辨率(现有标定值就是在这个尺度上得到的)
}


@dataclass
class Sample:
    """一帧的轻量记录(环形缓冲里存这个,不存原图)"""
    ts: float                 # time.time()
    gray: np.ndarray          # (h, w) uint8 下采样灰度
    brightness: float         # 0~1
    blur: float               # 拉普拉斯方差,越大越清晰
    ok: bool                  # 是否通过质量门控
    mask: np.ndarray | None = None   # (h, w) bool 动态物体掩码(下采样尺寸);没配 mask_fn 时为 None


class FrameSource(threading.Thread):
    """常驻抽帧线程。用法:

        src = FrameSource(url, interval_s=1.0, buffer_min=10)
        src.start()
        ...
        base = src.nearest_before(600)      # 约 10 分钟前的合格帧
        frame = src.sharpest_recent(10)     # 最近 10 帧里最清晰的一张(原图)
        src.stop()
    """

    def __init__(self, url: str, interval_s: float = 1.0, buffer_min: float = 10.0,
                 down_size: tuple[int, int] = (320, 240), transport: str = "tcp",
                 flush: int = 3, gate: dict | None = None, keep_full: int = 8,
                 mask_fn=None, reader_factory=None, name: str = "framesource"):
        super().__init__(name=name, daemon=True)
        self.url = url
        self.interval_s = float(interval_s)
        self.buffer_min = float(buffer_min)
        self.down_size = down_size
        self.transport = transport
        self.flush = max(1, int(flush))
        self.keep_full = max(1, int(keep_full))
        self.gate = {**GATE_DEFAULTS, **(gate or {})}
        # 动态物体掩码:必须在**入缓冲时**算好并随帧存下来——缓冲里只有下采样灰度,
        # 没有原图,事后无法再对基线帧跑检测。mask_fn(frame_bgr) -> bool (H,W) 或 None
        self.mask_fn = mask_fn
        self._reader_factory = reader_factory or self._default_reader

        maxlen = max(8, int(self.buffer_min * 60 / self.interval_s) + 60)
        self._buf: collections.deque[Sample] = collections.deque(maxlen=maxlen)
        self._full: collections.deque[tuple[float, float, np.ndarray]] = collections.deque(
            maxlen=self.keep_full)          # (ts, blur, 原图 BGR)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._reader = None

        # 统计(GUI 状态栏/日志用)
        self.stats = {"ticks": 0, "frames": 0, "reconnects": 0, "errors": 0,
                      "last_error": "", "connected": False, "fps": 0.0}

    # ---------------------------------------------------------------- 读取后端
    @staticmethod
    def _default_reader(url: str):
        """OpenCV 打开 RTSP;每次调用返回一个可 grab/retrieve/release 的对象。

        OpenCV 的 ffmpeg 后端用 "键;值" 形式传参;海康走 UDP 容易丢包花屏,默认 TCP。
        """
        import os
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("无法打开视频流(检查地址/账号/网络)")
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    # ---------------------------------------------------------------- 线程主体
    def run(self):
        backoff = 2.0
        t_prev = time.time()
        while not self._stop.is_set():
            t0 = time.time()
            if self._reader is None:
                try:
                    self._reader = self._reader_factory(self.url)
                    self.stats["connected"] = True
                    self.stats["reconnects"] += 0 if self.stats["frames"] == 0 else 1
                    backoff = 2.0
                    LOG.info("已连接: %s", self.url)
                except Exception as e:
                    self.stats["connected"] = False
                    self.stats["errors"] += 1
                    self.stats["last_error"] = str(e)
                    LOG.warning("连接失败,%.0fs 后重试: %s", backoff, e)
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2, 30.0)
                    continue

            frame = self._grab_one()
            if frame is None:
                self._drop_reader("读帧失败")
                continue

            self._ingest(frame)
            self.stats["ticks"] += 1
            dt = time.time() - t_prev
            t_prev = time.time()
            if dt > 0:
                self.stats["fps"] = round(1.0 / dt, 2) if self.stats["ticks"] % 10 else self.stats["fps"]

            # 按 interval_s 节拍
            sleep = self.interval_s - (time.time() - t0)
            if sleep > 0:
                self._stop.wait(sleep)
        self._drop_reader("停止")

    def _blur_min_effective(self, shape) -> float:
        """按像素数把清晰度阈值缩放到当前分辨率(方差近似与像素数成正比)"""
        ref = float(self.gate.get("blur_ref_pixels") or 0)
        if ref <= 0:
            return float(self.gate["blur_min"])
        h, w = shape
        return float(self.gate["blur_min"]) * (h * w) / ref

    def _grab_one(self):
        """丢弃 flush-1 帧冲刷缓冲,再解一帧——RTSP 刚连上时吐的常是陈旧帧"""
        try:
            for _ in range(self.flush - 1):
                if not self._reader.grab():
                    return None
            if not self._reader.grab():
                return None
            ok, frame = self._reader.retrieve()
            return frame if ok and frame is not None else None
        except Exception as e:
            self.stats["last_error"] = str(e)
            return None

    def _drop_reader(self, why: str):
        if self._reader is not None:
            try:
                self._reader.release()
            except Exception:
                pass
            self._reader = None
        self.stats["connected"] = False
        self.stats["errors"] += 1
        LOG.warning("断开(%s),将重连", why)

    def _ingest(self, frame: np.ndarray):
        ts = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, self.down_size, interpolation=cv2.INTER_AREA)
        brightness = float(small.mean()) / 255.0
        # 清晰度必须在**全分辨率**上算,且阈值要按分辨率缩放:
        # 拉普拉斯方差随分辨率变化很大(同一场景 640×480 的方差可能只有 1440×1171 的 1/5),
        # 直接用参考分辨率标定的 40 会把子码流的正常帧全判成"失焦"。
        blur = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        ok = (self.gate["brightness_min"] <= brightness <= self.gate["brightness_max"]
              and blur >= self._blur_min_effective(gray.shape))
        mask_small = None
        if self.mask_fn is not None:
            try:
                m = self.mask_fn(frame)
                if m is not None:
                    mask_small = cv2.resize(m.astype(np.uint8), self.down_size,
                                            interpolation=cv2.INTER_NEAREST) > 0
            except Exception as e:                       # 掩码失败不该中断抽帧
                LOG.warning("动态物体掩码失败,本帧按无掩码处理: %s", e)
        with self._lock:
            self._buf.append(Sample(ts, small, brightness, blur, ok, mask_small))
            self._full.append((ts, blur, frame.copy()))
        self.stats["frames"] += 1

    # ---------------------------------------------------------------- 消费接口
    def latest(self, require_ok: bool = False) -> Sample | None:
        with self._lock:
            if not self._buf:
                return None
            if not require_ok:
                return self._buf[-1]
            for s in reversed(self._buf):
                if s.ok:
                    return s
        return None

    def latest_full(self) -> tuple[float, np.ndarray] | None:
        """最近一帧原图 (ts, BGR)"""
        with self._lock:
            if not self._full:
                return None
            ts, _blur, img = self._full[-1]
            return ts, img.copy()

    def sharpest_recent(self, n: int | None = None, require_ok: bool = True):
        """最近 n 帧里挑最清晰的一张原图 → (ts, blur, BGR)。

        实现"1 fps 里挑帧送推理":60 秒有 60 个候选,挑拉普拉斯方差最大的那帧,
        深度对运动模糊/雨滴遮挡非常敏感,选帧能直接提升质量、减少误判。
        """
        with self._lock:
            cands = list(self._full)[-(n or self.keep_full):]
        if require_ok:
            with self._lock:
                ok_ts = {s.ts for s in self._buf if s.ok}
            cands = [c for c in cands if c[0] in ok_ts] or cands
        if not cands:
            return None
        ts, blur, img = max(cands, key=lambda c: c[1])
        return ts, blur, img.copy()

    def nearest_before(self, age_s: float, require_ok: bool = True,
                       tol_s: float | None = None) -> Sample | None:
        """取约 age_s 秒之前的那一帧(实时通道的基线)。

        require_ok=True 时只在合格帧里挑——基线帧自己不可信的话对比就没意义。
        tol_s 给定时,若最近候选偏离目标超过该容差则返回 None(说明还没攒够历史)。
        """
        target = time.time() - float(age_s)
        with self._lock:
            samples = [s for s in self._buf if (not require_ok or s.ok)]
        if not samples:
            return None
        best = min(samples, key=lambda s: abs(s.ts - target))
        if tol_s is not None and abs(best.ts - target) > tol_s:
            return None
        return best

    def history_seconds(self) -> float:
        """缓冲覆盖了多长历史(判断是否还在预热)"""
        with self._lock:
            if len(self._buf) < 2:
                return 0.0
            return self._buf[-1].ts - self._buf[0].ts

    def buffer_len(self) -> int:
        with self._lock:
            return len(self._buf)

    def stop(self, timeout: float = 5.0):
        self._stop.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=timeout)
