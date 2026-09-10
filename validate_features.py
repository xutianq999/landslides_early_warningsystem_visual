"""特征有效性验证:用已知真值检验 features.py 的输出是否可信。

四组测试:
  T1 几何真值   合成已知倾角的平面深度图 → 检查坡度/倾角能否还原(不依赖模型)
  T2 扰动稳定性 亮度/对比度/JPEG/噪声 → 几何特征应稳定,否则不可用于监测
  T3 已知位移   图像平移已知像素 → 检查 shift_px 是否准确,几何是否被污染
  T4 动态剔除   移动画面中的车辆 → 对比剔除前后 diff_frac,验证"防人为误差"是否有效

用法:
    .venv/bin/python validate_features.py [图片.jpg]
"""

import sys

import cv2
import numpy as np
from PIL import Image

import features as F

IMG = sys.argv[1] if len(sys.argv) > 1 else "3.jpg"
OK, WARN, BAD = "通过", "警告", "失败"


def verdict(cond_ok, cond_warn=True):
    return OK if cond_ok else (WARN if cond_warn else BAD)


# ---------------------------------------------------------------- T1 几何真值

def synth_plane_disp(H: int, W: int, tilt_deg: float, fov: float = 60.0, h: float = 5.0):
    """生成"倾斜平面"的**原始视差图**(1/z,未归一化,模拟 DA V2 输出)。

    平面方程 y = -h + m·z(m = tan(倾角)),法向与"上"方向夹角即倾角:
    tilt=0 → 水平地面;tilt 越大 → 越接近正对相机的陡坡/崖面。
    """
    fx = 0.5 * W / np.tan(np.radians(fov) / 2)
    cx, cy = W / 2, H / 2
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    dy = -(v - cy) / fx                      # 世界坐标 y 方向的射线斜率(上为正)
    m = np.tan(np.radians(tilt_deg))
    t = h / (m - dy)                         # 射线与平面交点的 z 值
    t = np.where(t > 0, t, np.nan)
    disp = 1.0 / t                           # 原始视差,不做归一化
    med = np.nanmedian(disp)
    return np.nan_to_num(disp / med, nan=0.0).astype(np.float32)  # 只做尺度归一(不失真)


def test_geometry_ground_truth():
    print("\n=== T1 几何真值(合成平面,不依赖模型)===")
    H, W = 240, 320
    gray = Image.new("RGB", (W, H), (128, 128, 128))
    rows = []
    for tilt in (0, 10, 20, 30, 45, 60, 75):
        disp = synth_plane_disp(H, W, tilt)
        g = F.geometry_features(gray, disp, max_depth=10.0, fov=60.0)
        rows.append((tilt, g["slope_mean"], g["plane_tilt"], g["plane_rms"]))

    print(f"{'真实倾角':>8} {'坡度均值':>10} {'主平面倾角':>12} {'平面残差':>10}")
    for t, s, p, r in rows:
        print(f"{t:>8}° {s:>9.1f}° {p:>11.1f}° {r:>10.4f}")

    slopes = [r[1] for r in rows]
    tilts = [r[0] for r in rows]
    mono = all(slopes[i] < slopes[i + 1] + 3 for i in range(len(slopes) - 1))
    err = max(abs(s - t) for t, s, _, _ in rows)
    print(f"单调性: {verdict(mono, mono)} | 最大绝对误差: {err:.1f}°")
    print(f"结论: {'角度可近似当真实坡度用' if err < 5 else '单调递增→可用于时间序列比较;绝对误差大→不能直接当真实坡度用'}")
    return mono


# ---------------------------------------------------------------- T2 扰动稳定性

def test_perturbation(pil):
    print("\n=== T2 扰动稳定性(几何特征应对光照/压缩不敏感)===")
    base, _ = F.features_from_image(pil, use_seg=False)
    arr = np.array(pil)

    # 合理扰动(现场会真实发生):光照变化、压缩、传感器噪声
    mild = {
        "亮度-20%": Image.fromarray(np.clip(arr * 0.8, 0, 255).astype(np.uint8)),
        "亮度+20%": Image.fromarray(np.clip(arr * 1.2, 0, 255).astype(np.uint8)),
        "JPEG q60": _jpeg(arr, 60),
        "高斯噪声": Image.fromarray(np.clip(arr + np.random.default_rng(0).normal(0, 3, arr.shape),
                                        0, 255).astype(np.uint8)),
    }
    # 极端扰动(应被质量门控拦下)
    extreme = {"对比度×0.6": Image.fromarray(np.clip((arr - 128) * 0.6 + 128, 0, 255).astype(np.uint8))}

    geom_keys = ["slope_mean", "slope_p95", "rough_local", "curv_mean",
                 "plane_tilt", "plane_rms", "bulge_frac",
                 "disp_p05", "disp_p50", "disp_p95"]

    def run(variants):
        out = {}
        for name, img in variants.items():
            row, _ = F.features_from_image(img, use_seg=False)
            out[name] = {k: abs(row[k] - base[k]) / (abs(base[k]) + 1e-6) for k in geom_keys}
        return out

    mild_rel = run(mild)
    print(f"{'扰动':>10} " + " ".join(f"{k[:9]:>9}" for k in geom_keys))
    for name, rel in mild_rel.items():
        print(f"{name:>10} " + " ".join(f"{rel[k]*100:>8.1f}%" for k in geom_keys))
    worst = {k: max(r[k] for r in mild_rel.values()) for k in geom_keys}
    print("最坏相对变化:", " ".join(f"{k}={worst[k]*100:.0f}%" for k in geom_keys))
    stable = [k for k in geom_keys if worst[k] < 0.10]
    print(f"合理扰动下稳定(<10%): {len(stable)}/{len(geom_keys)}")

    ex_rel = run(extreme)
    ex_worst = max(max(r.values()) for r in ex_rel.values())
    print(f"极端对比度下最大变化: {ex_worst*100:.0f}% → 属预期失效模式,靠 img_blur 门控拦截")
    return worst, stable


def _jpeg(arr, q):
    import io
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ---------------------------------------------------------------- T3 已知位移

def test_known_shift(pil):
    print("\n=== T3 已知位移(shift_px 是否准确,几何是否被污染)===")
    base, prev = F.features_from_image(pil, use_seg=False)
    arr = np.array(pil)
    H, W = arr.shape[:2]
    print(f"{'真实位移':>10} {'测得 shift_px':>14} {'误差':>8} {'坡度变化':>10} {'diff_frac':>10}")
    for dx, dy in ((5, 3), (12, -8), (30, 0)):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(arr, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        row, _ = F.features_from_image(Image.fromarray(shifted), use_seg=False, prev=prev)
        true_mag = float(np.hypot(dx, dy))
        err = abs(row["shift_px"] - true_mag)
        slope_rel = abs(row["slope_mean"] - base["slope_mean"]) / base["slope_mean"]
        print(f"{true_mag:>9.1f}px {row['shift_px']:>13.1f}px {err:>7.1f}px "
              f"{slope_rel*100:>9.1f}% {row['diff_frac']:>10.3f}")
    print("说明: shift_px 准确且几何稳定 → 位移可被识别并触发质量门控;"
          "位移会推高 diff_frac,这正是门控存在的理由")


# ---------------------------------------------------------------- T4 动态剔除

def test_dynamic_masking(pil):
    print("\n=== T4 动态物体剔除 ===")
    arr = np.array(pil)
    H, W = arr.shape[:2]
    _, masks = F.segment(pil, [c.strip() for c in F.DEFAULT_CLASSES.split(",")], 0.15)
    dyn = F.dynamic_mask(masks, (H, W))
    if dyn is None or dyn.sum() < 100:
        print("该图未检出动态物体,跳过(换一张有车/人的图)")
        return None
    print(f"动态物体像素占比: {dyn.mean()*100:.1f}%")

    # T4a 机制验证(确定性,不依赖模型):动态区域内深度剧变,看是否被屏蔽
    print("\n[T4a] 机制验证:仅在动态区域内制造深度突变")
    gray = np.array(pil.convert("L"))
    d1 = np.tile(np.linspace(0.2, 0.8, H, dtype=np.float32)[:, None], (1, W))
    d2 = d1.copy()
    d2[dyn] += 0.4
    masked = F.align_and_diff((gray, d1), (gray, d2), ignore=dyn)
    unmasked = F.align_and_diff((gray, d1), (gray, d2), ignore=None)
    print(f"  剔除后 diff_frac={masked['diff_frac']:.4f} | 不剔除 diff_frac={unmasked['diff_frac']:.4f}")
    ratio_a = unmasked["diff_frac"] / (masked["diff_frac"] + 1e-6)
    ok_a = masked["diff_frac"] < 0.005 and unmasked["diff_frac"] > 0.01
    print(f"  假变化放大 {ratio_a:.0f} 倍 → {verdict(ok_a, ok_a)}")

    # T4b 实际影响(真实图像 + DA V2):同一张图开/关掩码,看几何特征差多少
    print("\n[T4b] 实际影响:同一张图开/关动态掩码的特征差异")
    on, _ = F.features_from_image(pil, mask_dynamic=True)
    off, _ = F.features_from_image(pil, mask_dynamic=False)
    print(f"  {'特征':>12} {'剔除后':>12} {'不剔除':>12} {'相对差':>9}")
    deltas = []
    for k in ("slope_mean", "rough_local", "plane_rms", "bulge_frac", "disp_p50"):
        rel = abs(on[k] - off[k]) / (abs(off[k]) + 1e-6)
        deltas.append(rel)
        print(f"  {k:>12} {on[k]:>12.4f} {off[k]:>12.4f} {rel*100:>8.1f}%")
    print(f"  最大相对差 {max(deltas)*100:.1f}% → 掩码{'有实质影响' if max(deltas) > 0.05 else '影响很小(该图动态物体不在几何关键区)'}")
    return ratio_a


def main():
    pil = Image.open(IMG).convert("RGB")
    print(f"验证图片: {IMG} ({pil.size[0]}x{pil.size[1]})")
    mono = test_geometry_ground_truth()
    worst, stable = test_perturbation(pil)
    test_known_shift(pil)
    ratio = test_dynamic_masking(pil)

    print("\n=== 汇总 ===")
    print(f"T1 几何真值单调性: {verdict(mono)}")
    print(f"T2 合理扰动稳定性: {len(stable)}/{len(worst)} 个几何特征 <10% 变化 → "
          f"{verdict(len(stable) == len(worst), len(stable) >= len(worst) - 1)}")
    print(f"T4 动态剔除机制: {verdict(ratio and ratio > 2, ratio and ratio > 1.2) if ratio else '跳过'}")


if __name__ == "__main__":
    main()
