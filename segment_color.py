"""颜色分割主坡体:不依赖深度/模型,直接用 Lab 颜色聚类把坡体分出来。

原理:K-means 在 Lab 颜色空间聚类(附加轻微空间权重,避免同一坡面被切成碎块),
再自动挑出"主坡体"簇(面积大、位置居中、颜色偏土黄而非绿/蓝)。

用法:
    .venv/bin/python segment_color.py 图片.jpg [-o color_seg_result.png] [-k 5]
    .venv/bin/python segment_color.py 图片.jpg --cluster 2   # 手动指定主坡体簇号
"""

import argparse

import cv2
import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="颜色分割主坡体")
    p.add_argument("image", nargs="?", default="3.jpg")
    p.add_argument("-o", "--out", default="color_seg_result.png")
    p.add_argument("-k", "--clusters", type=int, default=5, help="聚类数")
    p.add_argument("--cluster", type=int, help="手动指定主坡体簇号(默认自动挑选)")
    p.add_argument("--spatial", type=float, default=25.0, help="空间权重(0=纯颜色)")
    return p.parse_args()


def kmeans_lab(bgr, k, spatial=25.0):
    """Lab 颜色 + 归一化空间坐标 的 K-means。返回 (标签图, 各簇统计)"""
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ys, xs = np.mgrid[0:h, 0:w]
    feats = np.dstack([lab,
                       (xs / w - 0.5) * spatial * 2,
                       (ys / h - 0.5) * spatial * 2]).reshape(-1, 5).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(feats, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(h, w)

    stats = []
    for i in range(k):
        m = labels == i
        area = float(m.mean())
        lab_mean = lab[m].mean(0) if m.any() else np.zeros(3)
        cx = float((xs[m].mean() / w)) if m.any() else 0.5
        cy = float((ys[m].mean() / h)) if m.any() else 0.5
        # Lab → RGB 便于人读
        rgb = cv2.cvtColor(np.uint8([[lab_mean]]), cv2.COLOR_LAB2RGB)[0, 0]
        stats.append({"id": i, "area": area, "lab": lab_mean, "rgb": rgb,
                      "cx": cx, "cy": cy})
    return labels, stats


def pick_main_slope(stats):
    """自动挑主坡体:面积大、位置居中、颜色非绿非蓝(土黄/棕)"""
    best, best_score = None, -1
    for s in stats:
        a = s["lab"][1] - 128          # a>0 偏红(土/岩), a<0 偏绿(植被)
        b = s["lab"][2] - 128          # b>0 偏黄, b<0 偏蓝
        green = a < -6                 # 明显植被
        blue = b < -12                 # 明显蓝(天空/水面/蓝布)
        center = 1 - abs(s["cx"] - 0.5) * 1.5 - max(0, 0.35 - s["cy"]) * 1.2
        score = s["area"] * max(0.0, center) * (0.15 if (green or blue) else 1.0)
        if score > best_score:
            best, best_score = s, score
    return best


def detect_slope(bgr, k=5, spatial=25.0, cluster=None, work_width=480,
                 min_area_frac=0.002, morph_size=15):
    """Lab K-means 分割主坡体,返回 dict(mask, labels, stats, cluster)。

    mask 为全分辨率 uint8(0/255);labels 是降采样工作图的簇标签;
    cluster=None 时自动挑主坡体。可被 GUI / 其他脚本复用(算法不依赖 argparse)。
    """
    h, w = bgr.shape[:2]
    scale = work_width / w
    small = cv2.resize(bgr, (work_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    labels_s, stats = kmeans_lab(small, k, spatial)

    idx = cluster if cluster is not None else (pick_main_slope(stats) or {"id": 0})["id"]
    mask = cv2.resize((labels_s == idx).astype(np.uint8) * 255, (w, h),
                      interpolation=cv2.INTER_NEAREST)
    # 形态学清理:去小洞、平滑边界
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k5, iterations=1)

    # 去掉碎斑:只保留面积 ≥ min_area_frac 的连通域
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area_frac * mask.size:
            keep[lbl == i] = 255
    mask = keep

    # 填内部孔洞(不接触图像边界的背景连通域)
    inv = cv2.bitwise_not(mask)
    n2, lbl2, st2, _ = cv2.connectedComponentsWithStats(inv, 8)
    for i in range(1, n2):
        x, y, ww, hh = (st2[i, cv2.CC_STAT_LEFT], st2[i, cv2.CC_STAT_TOP],
                        st2[i, cv2.CC_STAT_WIDTH], st2[i, cv2.CC_STAT_HEIGHT])
        if not (x == 0 or y == 0 or x + ww >= w or y + hh >= h):
            mask[lbl2 == i] = 255
    return {"mask": mask, "labels": labels_s, "stats": stats, "cluster": int(idx)}


def main():
    args = parse_args()
    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"读不到图片: {args.image}")

    res = detect_slope(bgr, args.clusters, args.spatial, args.cluster)
    mask, labels_s, stats, idx = res["mask"], res["labels"], res["stats"], res["cluster"]

    print(f"聚类数 k={args.clusters},主坡体簇=#{idx}")
    print(f"{'簇':>3} {'面积占比':>9} {'中心(x,y)':>14} {'RGB':>16}")
    for s in stats:
        mark = " ← 主坡体" if s["id"] == idx else ""
        print(f"{s['id']:>3} {s['area']*100:>8.1f}% ({s['cx']:.2f},{s['cy']:.2f})"
              f" ({s['rgb'][0]:>3},{s['rgb'][1]:>3},{s['rgb'][2]:>3}){mark}")
    print(f"主坡体掩码占比: {mask.mean()/255*100:.1f}%")

    # 四联图:原图 / 聚类着色 / 掩码 / 叠加
    rng = np.random.default_rng(0)
    palette = rng.integers(60, 230, (args.clusters, 3), dtype=np.uint8)
    label_vis = palette[labels_s].astype(np.uint8)
    label_vis = cv2.resize(label_vis, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = bgr.copy()
    green = np.zeros_like(bgr)
    green[:, :, 1] = 255
    m3 = mask.astype(bool)
    overlay[m3] = (0.55 * bgr[m3] + 0.45 * green[m3]).astype(np.uint8)

    def put(img, text):
        img = img.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (30, 30, 30), -1)
        cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img

    h = 360
    def rz(img):
        w = int(img.shape[1] * h / img.shape[0])
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    panels = [put(rz(bgr), "Input"),
              put(rz(label_vis), f"K-means k={args.clusters}"),
              put(rz(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)), f"Main slope mask #{idx}"),
              put(rz(overlay), "Overlay")]
    canvas = np.hstack([np.hstack([p, np.full((h, 6, 3), 255, np.uint8)]) for p in panels])
    cv2.imwrite(args.out, canvas)
    print(f"已保存: {args.out}")


if __name__ == "__main__":
    main()
