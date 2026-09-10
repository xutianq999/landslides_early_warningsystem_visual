"""三维点云分析可视化:点云 + 拟合主平面 + ROI(沟壑),并输出平面参数。

回答"主平面指哪部分":主平面不是某个真实物体,而是对点云做的**最佳拟合平面**
(PCA 最小特征向量方向即法向)。它代表这一片表面的**整体平均朝向**:
  - plane_tilt = 该平面法向与铅垂方向的夹角(整体坡度)
  - plane_rms  = 点到该平面的均方根距离(表面相对平面有多不平整)
只看整幅图时,主平面被整个场景的平均朝向主导;限定到沟壑 ROI 后,它描述的是沟壑局部的走向。

用法:
    .venv/bin/python visualize3d.py 图片.jpg
    .venv/bin/python visualize3d.py 图片.jpg --roi 0.30 0.20 0.70 0.95   # 归一化 x1 y1 x2 y2
    .venv/bin/python visualize3d.py 图片.jpg --max-points 30000 -o pc_plane.png
"""

import argparse

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from core import backproject, get_model  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "PingFang SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    p = argparse.ArgumentParser(description="点云 + 主平面可视化")
    p.add_argument("image", nargs="?", default="3.jpg")
    p.add_argument("-o", "--out", default="pc_plane.png")
    p.add_argument("--roi", nargs="*", default=None,
                   help="归一化 x1 y1 x2 y2,或 'auto'(自动检测沟壑掩模)")
    p.add_argument("--max-depth", type=float, default=10.0)
    p.add_argument("--fov", type=float, default=60.0)
    p.add_argument("--plot-points", type=int, default=25000, help="绘图抽样点数")
    return p.parse_args()


def fit_plane(pts):
    """PCA 拟合平面:返回 (法向, 质心, 残差 rms)"""
    c = pts.mean(0)
    X = pts - c
    _, vecs = np.linalg.eigh(X.T @ X / len(pts))
    n = vecs[:, 0]
    if n @ np.array([0.0, 1.0, 0.0]) < 0:  # 法向固定朝上,便于理解
        n = -n
    rms = float(np.sqrt(((X @ n) ** 2).mean()))
    return n, c, rms


def main():
    args = parse_args()
    pil = Image.open(args.image).convert("RGB")
    W, H = pil.size

    disp = np.array(get_model("da2s")(pil)["predicted_depth"])
    disp01 = (disp - disp.min()) / (disp.max() - disp.min() + 1e-6)
    pts, cols, valid = backproject(pil, disp01, max_depth=args.max_depth, fov_deg=args.fov)
    h, w = valid.shape

    # ROI:auto=自动检测沟壑掩模;否则用归一化矩形
    auto = bool(args.roi) and str(args.roi[0]) == "auto"
    if auto:
        import segment_gully
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        m = segment_gully.detect_gully(bgr)["mask"] > 0
        roi = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        rx1, ry1, rx2, ry2 = 0, 0, 0, 0  # 仅用于矩形绘制
    else:
        roi_vals = [float(v) for v in args.roi] if args.roi else [0.28, 0.18, 0.72, 0.98]
        x1, y1, x2, y2 = roi_vals
        rx1, ry1, rx2, ry2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
        roi = np.zeros((h, w), bool)
        roi[int(ry1 * h / H):int(ry2 * h / H), int(rx1 * w / W):int(rx2 * w / W)] = True
    roi &= valid

    P_all, C_all = pts[valid], cols[valid]
    P_roi = pts[roi]
    n_roi, c_roi, rms_roi = fit_plane(P_roi)
    n_all, c_all, rms_all = fit_plane(P_all)
    tilt_roi = np.degrees(np.arccos(np.clip(abs(n_roi @ np.array([0.0, 1.0, 0.0])), 0, 1)))
    tilt_all = np.degrees(np.arccos(np.clip(abs(n_all @ np.array([0.0, 1.0, 0.0])), 0, 1)))

    print(f"图像 {W}x{H} | ROI = {'自动检测沟壑掩模' if auto else f'矩形 ({rx1},{ry1})-({rx2},{ry2})'}")
    print(f"点云有效点 {len(P_all):,} | ROI 内 {len(P_roi):,} ({len(P_roi)/max(len(P_all),1)*100:.1f}%)")
    print(f"ROI 主平面 : 倾角 {tilt_roi:.1f}°  残差 {rms_roi:.3f}  法向 ({n_roi[0]:.2f},{n_roi[1]:.2f},{n_roi[2]:.2f})")
    print(f"全局主平面 : 倾角 {tilt_all:.1f}°  残差 {rms_all:.3f}(对比用)")
    print("提示: 残差越大说明表面越偏离平面(沟壑越明显)")

    # ---- 绘图 ----
    rng = np.random.default_rng(0)
    n_plot = min(args.plot_points, len(P_all))
    idx = rng.choice(len(P_all), n_plot, replace=False)
    sel = np.zeros(len(P_all), bool)
    sel[idx] = True
    roi_mask_full = roi[valid]
    P, C = P_all[sel], C_all[sel].astype(np.float32) / 255.0
    in_roi = roi_mask_full[sel]

    # 坐标范围按分位数裁剪,避免少数远点把视野撑爆
    lo = np.percentile(P, 2, axis=0)
    hi = np.percentile(P, 98, axis=0)
    keep = np.all((P >= lo) & (P <= hi), axis=1)
    P, C, in_roi = P[keep], C[keep], in_roi[keep]

    fig = plt.figure(figsize=(18, 6), dpi=130)

    # 面板 1:原图 + ROI(自动掩模轮廓 或 矩形框)
    ax0 = fig.add_subplot(1, 3, 1)
    img_show = np.array(pil).copy()
    if auto:
        m_full = cv2.resize(roi.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(m_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_show, cnts, -1, (0, 255, 102), 4)
        ax0.set_title("① ROI(绿线)= 自动检测沟壑掩模", fontsize=12)
    else:
        ax0.add_patch(plt.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1,
                                    fill=False, edgecolor="#00FF66", linewidth=3))
        ax0.set_title(f"① ROI(绿框)= 中央沟壑\n像素 ({rx1},{ry1})-({rx2},{ry2})", fontsize=12)
    ax0.imshow(img_show)
    ax0.axis("off")

    # 面板 2:三维点云 + 主平面
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.scatter(P[~in_roi, 0], P[~in_roi, 2], P[~in_roi, 1],
               c="#9AA0A6", s=1.0, alpha=0.25, linewidths=0)
    ax.scatter(P[in_roi, 0], P[in_roi, 2], P[in_roi, 1],
               c=C[in_roi], s=2.5, alpha=0.95, linewidths=0)

    _, vecs = np.linalg.eigh((P_roi - c_roi).T @ (P_roi - c_roi) / len(P_roi))
    u, v = vecs[:, 1], vecs[:, 2]
    su = np.percentile(np.abs((P_roi - c_roi) @ u), 98) * 1.15
    sv = np.percentile(np.abs((P_roi - c_roi) @ v), 98) * 1.15
    s, t = np.meshgrid(np.linspace(-su, su, 14), np.linspace(-sv, sv, 14))
    surf = c_roi[None, None, :] + s[..., None] * u + t[..., None] * v
    ax.plot_surface(surf[..., 0], surf[..., 2], surf[..., 1],
                    alpha=0.28, color="#FF3B30", linewidth=0, shade=False)

    L = max(su, sv) * 0.8
    ax.quiver(c_roi[0], c_roi[2], c_roi[1], n_roi[0] * L, n_roi[2] * L, n_roi[1] * L,
              color="#FF3B30", linewidth=3, arrow_length_ratio=0.28)

    ax.set_xlabel("X 右 (m)")
    ax.set_ylabel("Z 前 (m)")
    ax.set_zlabel("Y 上 (m)")
    ax.set_title(f"② 点云 + 主平面(红面/红箭头=法向)\nROI 倾角 {tilt_roi:.1f}°,残差 {rms_roi:.3f}", fontsize=12)
    ax.view_init(elev=26, azim=-58)
    ax.set_box_aspect((1, 1, 0.5))
    ax.grid(False)

    # 面板 3:侧视图(沿 X 看),平面投影成一条直线 → 倾角直观
    ax2 = fig.add_subplot(1, 3, 3)
    ax2.scatter(P[~in_roi, 2], P[~in_roi, 1], c="#9AA0A6", s=1.0, alpha=0.25, linewidths=0)
    ax2.scatter(P[in_roi, 2], P[in_roi, 1], c=C[in_roi], s=2.5, alpha=0.95, linewidths=0)
    # 平面在 Z-Y 投影上是一条直线,取平面内 v 方向跨度画线
    tt = np.linspace(-sv * 1.1, sv * 1.1, 50)
    line = c_roi[None, :] + tt[:, None] * v
    ax2.plot(line[:, 2], line[:, 1], color="#FF3B30", linewidth=2.5,
             label=f"主平面(倾角 {tilt_roi:.1f}°)")
    ax2.set_xlabel("Z 前 (m)")
    ax2.set_ylabel("Y 上 (m)")
    ax2.set_title("③ 侧视图(沿 X 看)\n红线的斜率 = 主平面倾角", fontsize=12)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"已保存: {args.out}")


if __name__ == "__main__":
    main()
