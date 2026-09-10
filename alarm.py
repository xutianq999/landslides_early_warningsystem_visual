"""滑坡实时报警:读 features.csv 时间序列 → 变化率/加速度 → 分级报警。

思路(详见 PIPELINE.md):
  1. 质量门控:剔除不可信帧(太暗/失焦/配准失败/相机被碰动)
  2. 中值滤波:去掉单帧尖刺噪声
  3. 稳健基线:滚动中位数 + MAD(比均值/标准差抗离群)
  4. 变化率(一阶差分)与加速度(二阶差分),按 MAD 归一化成稳健 z 值
  5. 多信号加权融合 + 持续性判定(连续 N 个采样点超阈值才算报警,防抖)

用法:
    .venv/bin/python alarm.py features.csv [--out alarm_result.csv]
    .venv/bin/python alarm.py features.csv --window 48 --persist 3 --json
    .venv/bin/python alarm.py --db                     # 从 SQLite 取最近 500 帧重算并写回
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

# 监控信号: (列名, 权重, 危险方向)  direction: +1 上升危险 / -1 下降危险 / 0 双向
SIGNALS = [
    ("diff_frac", 3.0, +1),        # 深度变化面积占比 —— 形变核心指标
    ("bulge_frac", 2.5, +1),       # 坡脚鼓胀
    ("seg_landslide_frac", 2.0, +1),  # 滑坡区域面积
    ("rough_local", 1.5, +1),      # 表面破碎化
    ("slope_mean", 1.0, +1),       # 坡度
    ("plane_rms", 1.0, +1),        # 平整度残差
    ("disp_p50", 1.0, 0),          # 深度中位数(双向:整体远近变化)
]

# 质量门控阈值
GATE = {
    "shift_resp_min": 0.20,     # 配准响应太低 → 该帧不可信
    "img_brightness_min": 0.12,  # 太暗(夜间/雨雾)
    "img_brightness_max": 0.95,  # 过曝
    "img_blur_min": 40.0,        # 拉普拉斯方差太小 → 失焦/糊
    "shift_px_max": 8.0,         # 相机位移过大 → 被碰动或漂移
}

LEVELS = [(3.0, 3, "红色-紧急"), (2.0, 2, "橙色-预警"), (1.2, 1, "黄色-关注")]  # 默认阈值,可用 --t1/--t2/--t3 覆盖


def parse_args():
    p = argparse.ArgumentParser(description="滑坡报警:变化率/加速度分析")
    p.add_argument("csv", nargs="?", default="features.csv", help="特征 CSV")
    p.add_argument("--out", default="alarm_result.csv", help="输出 CSV")
    p.add_argument("--window", type=int, default=24, help="稳健基线滚动窗口(采样点数)")
    p.add_argument("--persist", type=int, default=2, help="连续超阈值的采样点数才算报警")
    p.add_argument("--t1", type=float, default=1.5, help="黄色阈值")
    p.add_argument("--t2", type=float, default=3.0, help="橙色阈值")
    p.add_argument("--t3", type=float, default=5.0, help="红色阈值")
    p.add_argument("--json", action="store_true", help="同时打印最新一条的 JSON")
    p.add_argument("--db", nargs="?", const="", default=None, metavar="PATH",
                   help="从 SQLite 读特征并把报警写回(可选路径;只写 --db 用 config 默认)")
    p.add_argument("--device", help="DB 模式的设备 ID(默认取 config)")
    p.add_argument("--limit", type=int, default=500,
                   help="DB 模式:只重算最近 N 帧(报警依赖全序列,取尾部窗口即可)")
    return p.parse_args()


def quality_gate(df: pd.DataFrame) -> pd.Series:
    """返回每帧是否可信(True=可信)"""
    ok = pd.Series(True, index=df.index)
    if "shift_resp" in df:
        ok &= df["shift_resp"].fillna(1.0) >= GATE["shift_resp_min"]
    if "img_brightness" in df:
        b = df["img_brightness"].fillna(0.5)
        ok &= (b >= GATE["img_brightness_min"]) & (b <= GATE["img_brightness_max"])
    if "img_blur" in df:
        ok &= df["img_blur"].fillna(1e9) >= GATE["img_blur_min"]
    if "shift_px" in df:
        ok &= df["shift_px"].fillna(0.0) <= GATE["shift_px_max"]
    return ok


def signal_terms(series: pd.Series, window: int):
    """把一个信号分解成三个异常分量。

    - z_ref : 相对**固定参考期**(前若干采样点的中位数)的持续偏离 → 捕捉慢速蠕变
    - z_rate: 相对滚动基线的变化率 → 捕捉加速变形
    - z_acc : 加速度(变化率的一阶差分) → 捕捉突变

    尺度用 MAD 并设相对下限,防止基线恒定时微小抖动被放大。
    """
    n = len(series)
    ref_n = max(3, min(max(3, window // 2), max(3, n // 5)))
    ref = series.iloc[:ref_n]
    base_ref = float(ref.median())
    scale_ref = float((ref - base_ref).abs().median()) * 1.4826
    floor = max(scale_ref, float(ref.abs().median()) * 0.2, 1e-6)

    base_roll = series.rolling(window, min_periods=max(3, window // 4)).median()
    mad_roll = ((series - base_roll).abs()
                .rolling(window, min_periods=max(3, window // 4)).median()) * 1.4826
    scale = np.maximum(mad_roll.fillna(0.0), floor)

    rate = series.diff()
    accel = rate.diff()
    z_ref = ((series - base_ref) / floor).clip(-10, 10)
    z_rate = (rate / scale).clip(-10, 10)
    z_acc = (accel / scale).clip(-10, 10)
    return z_ref, z_rate, z_acc


def analyze(df: pd.DataFrame, window: int, persist: int,
            thresholds=(2.0, 4.0, 6.0)) -> pd.DataFrame:
    """逐信号算异常度 → 取最严重信号定等级。

    阈值是对"异常度 a"(0~10)的,不是对总分,避免多信号平均把单个强信号稀释掉。
    """
    valid = quality_gate(df)
    out = pd.DataFrame(index=df.index)
    out["time"] = df.get("time", pd.Series(range(len(df)), index=df.index))
    out["valid"] = valid

    severity = pd.DataFrame(index=df.index)  # 每个信号的异常度 0~10
    for col, weight, direction in SIGNALS:
        if col not in df:
            continue
        raw = pd.to_numeric(df[col], errors="coerce")
        smooth = raw.rolling(3, center=True, min_periods=1).median().ffill().bfill()
        z_ref, z_rate, z_acc = signal_terms(smooth, window)

        def danger(v):  # 上升危险取正向,双向取绝对值
            return v.abs() if direction == 0 else v * direction

        # 持续偏离(慢速蠕变) + 变化率(加速变形) + 加速度(突变),各自截顶
        a = (0.5 * danger(z_ref).clip(0, 5)
             + 1.0 * danger(z_rate).clip(0, 5)
             + 0.5 * danger(z_acc).clip(0, 5))
        severity[col] = a.clip(0, 10).fillna(0)
        out[f"{col}_z"] = z_ref.round(3)
        out[f"{col}_rate"] = smooth.diff().round(5)
        out[f"{col}_accel"] = smooth.diff().diff().round(5)

    if severity.empty:
        out["score"] = 0.0
        out["level"] = 0
        out["level_name"] = "正常"
        return out

    # 报告用加权均值(0~10),判定用最严重信号
    weights = pd.Series({c: w for c, w, _ in SIGNALS if c in severity.columns})
    out["score"] = (severity.mul(weights, axis=1).sum(axis=1) / weights.sum()).round(3)
    out["top_signal"] = severity.idxmax(axis=1)
    out["top_severity"] = severity.max(axis=1).round(2)

    level = pd.Series(0, index=df.index)
    for th, lv in zip(thresholds, (1, 2, 3)):
        level = level.where(out["top_severity"] < th, lv)

    # 持续性判定:连续 persist 个点达标才算,否则降一级(去抖)
    hold = level.rolling(persist, min_periods=persist).min().fillna(0).astype(int)
    level = np.minimum(level, hold + 1)
    # 预热期(参考基线未建立)与不可信帧不报警
    warmup = max(3, min(window // 2, len(df) // 5))
    level.iloc[:warmup] = 0
    level = level.where(valid, 0)
    out["level"] = level
    out["level_name"] = level.map({0: "正常", 1: "黄色-关注", 2: "橙色-预警", 3: "红色-紧急"})

    if "shift_px" in df:
        out["camera_alarm"] = (df["shift_px"].fillna(0) > GATE["shift_px_max"])
    return out


def main():
    args = parse_args()

    conn = None
    device = None
    if args.db is not None:
        import config
        import db as dbm
        device = args.device or config.CONFIG["device_id"]
        conn = dbm.connect(args.db or None)
        df = dbm.load_frames_df(conn, device_id=device, limit=args.limit)
        if len(df) == 0:
            sys.exit(f"库里没有设备 {device} 的特征;先跑 features.py --db")
        print(f"数据源: 数据库 {dbm.resolve_db_path(args.db or None)} | 设备 {device} | 最近 {len(df)} 帧")
    else:
        try:
            df = pd.read_csv(args.csv)
        except FileNotFoundError:
            sys.exit(f"找不到 {args.csv};先用 features.py 生成特征")
        if len(df) == 0:
            sys.exit("特征表为空")

    result = analyze(df, args.window, args.persist,
                     thresholds=(args.t1, args.t2, args.t3))
    result.to_csv(args.out, index=False)

    n_alert = int((result["level"] > 0).sum())
    print(f"共 {len(df)} 个采样点,报警 {n_alert} 个 → {args.out}")
    if n_alert:
        last = result[result["level"] > 0].iloc[-1]
        print(f"最近一次报警: {last['time']} | {last['level_name']} | score={last['score']}")
    else:
        print("当前无报警")

    if conn is not None:
        n = dbm.upsert_alarms(conn, dbm.alarm_records(result, device))
        conn.close()
        print(f"报警结果已入库 {n} 帧")

    if args.json:
        last = result.iloc[-1]
        print(json.dumps({"time": str(last["time"]), "score": float(last["score"]),
                          "level": int(last["level"]), "level_name": last["level_name"]},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
