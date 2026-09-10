"""算法参数配置文件(profile):把散落在代码里的阈值收口到一处,可导出 / 分享 / 回滚。

与 config.json 的分工(**别再混在一起**):
  - `config.json` = **环境**:库路径、端口、RTSP 地址(含相机口令)、采样间隔 → 不可外发
  - `profiles/*.json` = **算法**:分割方法与参数、类别、FOV、ROI、报警阈值 → 可外发/存档/回滚

取值优先级:命令行 > profile > config.json 的 devices[设备号] > defaults > 代码内置默认

用法:
    .venv/bin/python tuning.py --from-config HIK-01 -o profiles/HIK-01.json   # 导出
    .venv/bin/python tuning.py --show profiles/HIK-01.json                    # 查看
    .venv/bin/python tuning.py --defaults -o profiles/default.json            # 内置默认
"""

import argparse
import json
from pathlib import Path

import config

PROFILE_VERSION = 1
SECTIONS = ("realtime", "analysis", "segmentation")
_ANALYSIS_KEYS = ("classes", "conf", "max_depth", "fov", "roi", "roi_auto", "roi_target")


def defaults() -> dict:
    """内置默认:直接从各模块取,避免又出现"多处各写一份默认值" """
    import realtime
    import segment_gully
    import features
    rt = {k: v for k, v in realtime.DEFAULTS.items() if k != "alarm_dir"}
    rt["down_size"] = list(rt["down_size"])
    return {
        "version": PROFILE_VERSION,
        "realtime": rt,
        "analysis": {"classes": features.DEFAULT_CLASSES, "conf": 0.15, "max_depth": 10.0,
                     "fov": 60.0, "roi": None, "roi_auto": False, "roi_target": "gully"},
        "segmentation": dict(segment_gully.GULLY_DEFAULTS),
    }


def build(device: str | None = None) -> dict:
    """按当前设备配置生成一份 profile(供 studio 导出)"""
    device = device or config.CONFIG["device_id"]
    prof = defaults()
    dc = config.for_device(device)
    for k in _ANALYSIS_KEYS:
        if k in dc:
            prof["analysis"][k] = dc[k]
    if dc.get("gully"):
        prof["segmentation"].update(dc["gully"])
    prof["device"] = device
    return prof


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def normalize(prof: dict) -> dict:
    """补齐缺省 + 类型修正(JSON 里元组会变成 list)"""
    prof = _merge(defaults(), prof or {})
    rt = prof["realtime"]
    if isinstance(rt.get("down_size"), list):
        rt["down_size"] = tuple(rt["down_size"])
    return prof


def validate(prof: dict) -> list[str]:
    """返回问题列表(空 = 通过)。只做结构性校验,不做现场标定判断。"""
    errs = []
    ver = prof.get("version", PROFILE_VERSION)
    if not isinstance(ver, int) or ver > PROFILE_VERSION:
        errs.append(f"profile version {ver} 比当前支持的 {PROFILE_VERSION} 新,拒绝加载")
    for sec in SECTIONS:
        if sec in prof and not isinstance(prof[sec], dict):
            errs.append(f"`{sec}` 必须是对象")
    seg = prof.get("segmentation") or {}
    m = seg.get("method")
    if m is not None:
        import segment_gully
        if m not in segment_gully.GULLY_METHODS:
            errs.append(f"未知分割方法 `{m}`,可选: {', '.join(segment_gully.GULLY_METHODS)}")
    rt = prof.get("realtime") or {}
    for k, lo, hi in (("change_t1", 0, 1), ("change_t2", 0, 1), ("change_t3", 0, 1),
                      ("diff_thresh", 0, 1)):
        if k in rt and not (lo <= float(rt[k]) <= hi):
            errs.append(f"realtime.{k} 应在 {lo}~{hi} 之间,当前 {rt[k]}")
    for a, b in (("change_t1", "change_t2"), ("change_t2", "change_t3")):
        if a in rt and b in rt and float(rt[a]) >= float(rt[b]):
            errs.append(f"阈值顺序不对:realtime.{a} ({rt[a]}) 应小于 {b} ({rt[b]})")
    return errs


def save(path: str | Path, prof: dict) -> Path:
    p = Path(path)
    if p.parent and str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    prof = {**prof, "version": prof.get("version", PROFILE_VERSION)}
    errs = validate(prof)
    if errs:
        raise ValueError("配置文件校验失败:\n  - " + "\n  - ".join(errs))
    p.write_text(json.dumps(prof, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p


def load(path: str | Path) -> dict:
    prof = json.loads(Path(path).read_text(encoding="utf-8"))
    errs = validate(prof)
    if errs:
        raise ValueError(f"{path} 校验失败:\n  - " + "\n  - ".join(errs))
    return normalize(prof)


# ---------------------------------------------------------------- 给运行时用

def realtime_params(prof: dict) -> dict:
    """实时通道参数(已合并内置默认)"""
    import realtime
    p = {**realtime.DEFAULTS, **(prof.get("realtime") or {})}
    p["down_size"] = tuple(p["down_size"])
    return p


def analysis_overrides(prof: dict) -> dict:
    """传给 capture.process(overrides=...) 的分析参数。

    分割方法与参数从 `segmentation` 段折叠进 `gully` 键——这样 analyze 时真正能生效
    (以前分割参数只在命令行能调,运行时永远吃默认值)。
    """
    a = prof.get("analysis") or {}
    out = {k: a[k] for k in _ANALYSIS_KEYS if k in a}
    seg = prof.get("segmentation")
    if seg:
        out["gully"] = dict(seg)
    return out


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description="算法参数配置文件(profile)")
    ap.add_argument("--from-config", metavar="DEVICE", help="按 config.json 里该设备生成")
    ap.add_argument("--defaults", action="store_true", help="生成内置默认配置")
    ap.add_argument("--show", metavar="FILE", help="打印配置文件(已补默认值)")
    ap.add_argument("-o", "--out", help="输出路径")
    args = ap.parse_args()

    if args.show:
        prof = load(args.show)
        print(json.dumps(prof, ensure_ascii=False, indent=2))
        errs = validate(json.loads(Path(args.show).read_text(encoding="utf-8")))
        print("\n校验:", "通过" if not errs else "; ".join(errs))
        return

    prof = build(args.from_config) if args.from_config else defaults()
    out = args.out or (f"profiles/{args.from_config or 'default'}.json")
    p = save(out, prof)
    print(f"已生成 {p}")
    print(json.dumps(prof, ensure_ascii=False, indent=2)[:800])


if __name__ == "__main__":
    main()
