"""海康 RTSP 定时抓图 → 特征提取 → 入库 → 刷新报警。

每台设备(点位)配一条 rtsp_url,按 interval_min 定时抓一帧,存到
`images/<设备号>/<时间戳>.jpg`,再用该设备的参数提取特征写入 SQLite,
并重算该设备的报警等级。数据随后由 api.py 提供给平台。

用法:
    .venv/bin/python capture.py                      # 常驻,按每设备 interval_min 抓
    .venv/bin/python capture.py --once               # 抓一轮就退出(交给 launchd/cron)
    .venv/bin/python capture.py --once --device HIK-01
    .venv/bin/python capture.py --once --log capture.log
    .venv/bin/python capture.py --from-file 3.jpg --device HIK-01   # 不连相机,处理已有图片
    .venv/bin/python capture.py --once --source rtsp://... --device HIK-01  # 临时换流地址

两个关键设计:
  - **每次采样独立建连**(不维持长连接)。RTSP 长连接会积累陈旧缓冲,抓到的常是几秒前的
    画面;每次新建连接天然拿最新帧,且进程崩溃/重启不会残留坏流。代价是每次约 1~2 秒建连,
    相对分钟级采样间隔可忽略。
  - **上一帧状态持久化**。`diff_*`/`shift_*` 需要上一帧的 (gray, disp),而低频抓图的进程是
    短命的,所以把状态存到 `data/state/<设备号>.npz`,下次采样读回当 prev,保证时序特征连续。
    首次采样没有状态文件,时序字段为空(与 CLI 首帧行为一致)。某次抓帧失败的下一帧会拿更早的
    一帧作对比(间隔被拉长),这是可接受的降级。
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

import config
import db as dbm

LOG = logging.getLogger("capture")


def setup_logging(logfile: str | None = None) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if logfile:
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(message)s", force=True)


# ---------------------------------------------------------------- 设备与状态

def select_devices(only: str | None = None) -> list[str]:
    """要处理的设备:显式指定 > 配置里 enabled 的 > 当前默认设备"""
    if only:
        return [only]
    devs = list((config.CONFIG.get("devices") or {}).keys())
    if not devs:
        return [config.CONFIG["device_id"]]
    keep = [d for d in devs if config.for_device(d).get("enabled", True) is not False]
    return sorted(keep)


def state_path(device: str) -> Path:
    return config.ROOT / "data" / "state" / f"{device}.npz"


def load_prev(device: str) -> tuple | None:
    """读上一帧的 (gray, disp);没有或损坏则 None(该帧时序特征留空)"""
    p = state_path(device)
    if not p.exists():
        return None
    try:
        z = np.load(p)
        return (z["gray"], z["disp"])
    except Exception as e:
        LOG.warning("上一帧状态读取失败(%s): %s —— 本次时序特征留空", p.name, e)
        return None


def save_prev(device: str, cur: tuple) -> None:
    p = state_path(device)
    p.parent.mkdir(parents=True, exist_ok=True)
    gray, disp = cur
    np.savez_compressed(p, gray=gray, disp=disp)


# ---------------------------------------------------------------- 抓帧

def grab_frame(url: str, transport: str = "tcp", flush: int = 5,
               timeout_s: float = 15.0):
    """连一次 RTSP 抓一帧(BGR)。失败抛 RuntimeError。

    先丢弃前几帧冲刷缓冲,取最后读到的有效帧——RTSP 刚连上时往往吐的是陈旧帧。
    """
    import cv2
    # OpenCV 的 ffmpeg 后端用 "键;值" 形式,海康走 UDP 容易丢包花屏,默认 TCP
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport}"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened():
            raise RuntimeError("无法打开视频流(检查地址/账号/网络)")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 只留最新一帧,减少陈旧缓冲
        frame, deadline = None, time.time() + timeout_s
        for _ in range(max(1, flush)):
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
            if time.time() > deadline:
                break
        if frame is None:
            raise RuntimeError("未读到有效帧(流可能已断开)")
        return frame
    finally:
        cap.release()


def grab_with_retry(url: str, transport: str = "tcp", attempts: int = 3,
                    base_delay: float = 2.0):
    """退避重试抓帧;全部失败返回 None(调用方跳过本次,不中断循环)"""
    for i in range(1, attempts + 1):
        try:
            return grab_frame(url, transport)
        except Exception as e:
            LOG.warning("抓帧失败 %d/%d: %s", i, attempts, e)
            if i < attempts:
                time.sleep(base_delay * (2 ** (i - 1)))   # 2s, 4s, 8s
    return None


# ---------------------------------------------------------------- 落盘与处理

def save_snapshot(frame_bgr, device: str, when: datetime) -> Path:
    """BGR 帧 → images/<设备号>/<YYYYmmddHHMMSS>.jpg(先写临时文件再改名,避免半截文件)"""
    import cv2
    d = config.images_dir() / device
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{when:%Y%m%d%H%M%S}.jpg"
    tmp = path.with_name(path.stem + ".tmp.jpg")
    if not cv2.imwrite(str(tmp), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"图片写入失败: {tmp}")
    tmp.replace(path)
    return path


def device_rain(device: str, when: datetime) -> dict | None:
    """配了 lat/lon 就取该时刻的累积降雨(在线 Open-Meteo)"""
    dc = config.for_device(device)
    if dc.get("lat") is None or dc.get("lon") is None:
        return None
    import features as F
    return F.rain_from_api(dc["lat"], dc["lon"], when)


def process_image(path: Path, device: str, when: datetime,
                  rain: dict | None = None) -> dict:
    """已有图片 → 特征 → 入库(用该设备参数)。返回特征行。"""
    import cv2
    import features as F
    import segment_gully

    p = F.resolve_params(device)
    pil = Image.open(path).convert("RGB")

    roi_mask, extra_masks = None, None
    if p["roi_auto"]:
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        res = segment_gully.detect_gully(bgr)
        key = {"gully": "mask", "debris": "debris", "both": "mask_all"}[p["roi_target"]]
        roi_mask = res[key] > 0
        extra_masks = {"gully": res["mask"], "debris": res["debris"]}

    prev = load_prev(device)
    classes = [c.strip() for c in p["classes"].split(",") if c.strip()]
    row, cur = F.features_from_image(pil, classes=classes, conf=p["conf"], prev=prev,
                                     max_depth=p["max_depth"], fov=p["fov"], roi=p["roi"],
                                     roi_mask=roi_mask, extra_masks=extra_masks)
    row["time"] = when.isoformat(timespec="seconds")
    if rain:
        row.update(rain)

    params = {k: p[k] for k in ("classes", "conf", "max_depth", "fov",
                                "roi", "roi_auto", "roi_target")}
    conn = dbm.connect()
    try:
        dbm.upsert_frames(conn,
                          [{**row, "image_path": str(path.resolve()), "params": params}],
                          device_id=device)
    finally:
        conn.close()
    save_prev(device, cur)          # 供下一次采样做帧间对比
    return row


def refresh_alarm(device: str, limit: int = 500) -> dict | None:
    """用该设备的阈值重算尾部窗口的报警等级并写回"""
    import alarm
    dc = config.for_device(device)
    window = int(dc.get("window", 24))
    persist = int(dc.get("persist", 2))
    thresholds = (float(dc.get("t1", 1.5)), float(dc.get("t2", 3.0)), float(dc.get("t3", 5.0)))
    conn = dbm.connect()
    try:
        df = dbm.load_frames_df(conn, device_id=device, limit=limit)
        if len(df) == 0:
            return None
        result = alarm.analyze(df, window, persist, thresholds=thresholds)
        dbm.upsert_alarms(conn, dbm.alarm_records(result, device))
        last = result.iloc[-1]
        return {"level": int(last["level"]), "level_name": last["level_name"],
                "score": float(last["score"])}
    finally:
        conn.close()


def sample_once(device: str, source: str | None = None, from_file: str | None = None,
                do_alarm: bool = True) -> bool:
    """一次完整采样:抓帧/取图 → 落盘 → 特征 → 入库 → 刷新报警。返回是否成功。"""
    when = datetime.now()
    try:
        if from_file:
            src = Path(from_file)
            if not src.exists():
                LOG.error("图片不存在: %s", src)
                return False
            d = config.images_dir() / device
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{when:%Y%m%d%H%M%S}.jpg"
            Image.open(src).convert("RGB").save(path, quality=92)
        else:
            dc = config.for_device(device)
            url = source or dc.get("rtsp_url")
            if not url:
                LOG.error("设备 %s 没有 rtsp_url(也没给 --source);请在 config.json 的 devices 里配置",
                          device)
                return False
            frame = grab_with_retry(url, dc.get("rtsp_transport", "tcp"))
            if frame is None:
                LOG.error("设备 %s 抓帧失败,本次跳过(下次到点再试)", device)
                return False
            path = save_snapshot(frame, device, when)

        row = process_image(path, device, when, rain=device_rain(device, when))
        info = f"{path.parent.name}/{path.name}  diff_frac={row.get('diff_frac')}"
        if do_alarm:
            lv = refresh_alarm(device)
            if lv:
                info += f"  报警={lv['level']}({lv['level_name']})"
        LOG.info("设备 %s 采样完成: %s", device, info)
        return True
    except Exception as e:
        LOG.exception("设备 %s 采样异常: %s", device, e)
        return False


def run_loop(devices: list[str], do_alarm: bool = True,
             source: str | None = None) -> None:
    """常驻循环:按每设备 interval_min 排下一次到期时间,设备间串行处理"""
    next_due = {d: 0.0 for d in devices}
    intervals = {d: max(1.0, float(config.for_device(d).get("interval_min", 5)) * 60)
                 for d in devices}
    LOG.info("常驻抓图启动 | 设备: %s | 间隔(分): %s",
             ", ".join(devices), {d: round(intervals[d] / 60, 1) for d in devices})
    while True:
        now = time.time()
        for d in sorted((d for d in devices if next_due[d] <= now), key=lambda x: next_due[x]):
            sample_once(d, source=source, do_alarm=do_alarm)
            next_due[d] = time.time() + intervals[d]
        sleep = min(next_due.values()) - time.time() if next_due else 60.0
        time.sleep(min(max(sleep, 1.0), 60.0))   # 最多睡 60s,便于响应配置变化


def main():
    ap = argparse.ArgumentParser(description="RTSP 定时抓图 → 特征入库")
    ap.add_argument("--device", help="只处理该设备(默认处理配置里所有启用的设备)")
    ap.add_argument("--once", action="store_true", help="抓一轮就退出(适合 launchd/cron)")
    ap.add_argument("--source", help="临时覆盖 RTSP 地址(可为视频文件或任意流;多设备时慎用)")
    ap.add_argument("--from-file", metavar="IMG", help="跳过抓帧,直接处理已有图片(测试/补录)")
    ap.add_argument("--no-alarm", action="store_true", help="采样后不刷新报警等级")
    ap.add_argument("--log", metavar="FILE", help="同时把日志写入文件")
    args = ap.parse_args()
    setup_logging(args.log)

    devices = select_devices(args.device)
    if not devices:
        sys.exit("没有可处理的设备;检查 config.json 的 devices / enabled")
    do_alarm = not args.no_alarm

    if args.from_file:
        ok = sample_once(devices[0], from_file=args.from_file, do_alarm=do_alarm)
        sys.exit(0 if ok else 1)

    if args.once:
        results = [sample_once(d, source=args.source, do_alarm=do_alarm) for d in devices]
        sys.exit(0 if any(results) else 1)

    run_loop(devices, do_alarm=do_alarm, source=args.source)


if __name__ == "__main__":
    main()
