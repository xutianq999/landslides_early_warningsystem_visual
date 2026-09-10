"""沟壑分割:把中央沟壑从两侧崖面中分出来。

两种方法(可单独用或一起用):
  A. 颜色法   沟壑比崖面更暗、饱和度更低 → 光照归一化后在中央带做阈值 + 连通域
  B. 边界法   沟壑左右壁是强梯度边 → 逐行追踪左右边界曲线,两线之间即为沟壑
  C. 混合     用边界法得到的左右线作空间约束,再在带内用颜色法细化

用法:
    .venv/bin/python segment_gully.py 图片.jpg
    .venv/bin/python segment_gully.py 图片.jpg --method edge -o gully_edge.png
"""

import argparse

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="沟壑分割(颜色 / 边界追踪)")
    p.add_argument("image", nargs="?", default="3.jpg")
    p.add_argument("-o", "--out", default="gully_seg.png")
    p.add_argument("--method", choices=["color", "edge", "hybrid", "grabcut", "darkrun"],
                   default="darkrun")
    p.add_argument("--y-range", nargs=2, type=float, default=[0.16, 0.97],
                   help="沟壑在图像中的纵向范围(归一化)")
    p.add_argument("--x-left", nargs=2, type=float, default=[0.18, 0.52],
                   help="左壁搜索横向范围(归一化)")
    p.add_argument("--x-right", nargs=2, type=float, default=[0.48, 0.78],
                   help="右壁搜索横向范围(归一化)")
    p.add_argument("--max-jump", type=int, default=40, help="边界追踪每行最大跳变(像素)")
    p.add_argument("--fill-window", type=int, default=40,
                   help="凹陷填充窗口(行),0=关闭;填掉沟底浅色斑块造成的缺口")
    p.add_argument("--extend-start", type=float, default=0.70,
                   help="底部延伸起点(归一化 y),从这里向下张开覆盖堆积体")
    p.add_argument("--extend-end", type=float, default=0.99, help="底部延伸终点(归一化 y)")
    p.add_argument("--extend-expand", type=float, default=0.26,
                   help="底部延伸时每侧张开量(占图像宽比例)")
    return p.parse_args()


def normalize_illumination(bgr):
    """CLAHE 只作用于 L 通道,压掉左右光照差,让色差更可比"""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def row_candidates(gx_row, x_lo, x_hi, sign, topk=6):
    """一行里找 sign 方向的强梯度候选点(返回像素 x 列表)"""
    seg = gx_row[x_lo:x_hi]
    if seg.size == 0:
        return []
    idx = np.argsort(seg)[::-1][:topk] if sign > 0 else np.argsort(seg)[:topk]
    return [x_lo + int(i) for i in idx]


def medfilt1(x: np.ndarray, k: int) -> np.ndarray:
    """一维中值滤波(沿行方向平滑边界曲线)"""
    k = max(3, k | 1)
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, k), axis=-1)


def track_boundary(gx, y0, y1, seed_y, seed_x, x_lo, x_hi, sign, max_jump):
    """从种子行向上下追踪一条边界曲线,返回逐行 x 坐标(缺失处用上一行值补齐)"""
    xs = np.full(gx.shape[0], np.nan, np.float32)
    xs[seed_y] = seed_x
    for direction in (-1, 1):
        x_prev = seed_x
        y = seed_y + direction
        while y0 <= y <= y1:
            cands = row_candidates(gx[y], x_lo, x_hi, sign)
            near = [c for c in cands if abs(c - x_prev) <= max_jump]
            if near:
                x_prev = min(near, key=lambda c: abs(c - x_prev))
            xs[y] = x_prev  # 没有候选就沿用上一行(相当于插值)
            y += direction
    # 中值平滑
    valid = ~np.isnan(xs)
    if valid.sum() > 5:
        idx = np.arange(len(xs))
        xs = np.interp(idx, idx[valid], xs[valid])
        xs = medfilt1(xs, max(5, (y1 - y0) // 20))
    return xs


def find_seed(gx, y0, y1, x_lo, x_hi, sign, frac=0.55):
    """在纵向范围的中部行附近找梯度最强的候选作为追踪种子。

    不用全局最强(容易被外侧崖壁/车辆等更强边缘带偏),只用中部若干行的平均值。
    """
    yc = int(y0 + (y1 - y0) * frac)
    band = gx[max(y0, yc - 4):min(y1, yc + 5), x_lo:x_hi].mean(0)
    if band.size == 0:
        return yc, (x_lo + x_hi) // 2
    i = int(np.argmax(band)) if sign > 0 else int(np.argmin(band))
    return yc, x_lo + i


def clean_mask(mask, min_area_frac=0.01):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area_frac * mask.size:
            keep[lbl == i] = 255
    # 填内部洞
    inv = cv2.bitwise_not(keep)
    n2, lbl2, st2, _ = cv2.connectedComponentsWithStats(inv, 8)
    H, W = mask.shape
    for i in range(1, n2):
        x, y, w, h = st2[i, :4]
        if not (x == 0 or y == 0 or x + w >= W or y + h >= H):
            keep[lbl2 == i] = 255
    return keep


def grabcut_refine(bgr, rough_mask, iters=6):
    """用边界追踪得到的粗掩码初始化 GrabCut,让它顺着颜色边界细化轮廓。

    沟壑是深灰蓝砾石,两侧崖面和底部前景是浅土黄——颜色分布可分,GrabCut 能沿
    真实色差边界收边,比"两条竖线+水平底边"贴合得多(尤其是底部斜边)。
    """
    H, W = bgr.shape[:2]
    gc = np.full((H, W), cv2.GC_PR_BGD, np.uint8)
    # 粗掩码内标为"可能前景",外面标为"可能背景";边缘一带保持不确定
    er = cv2.erode((rough_mask > 0).astype(np.uint8), np.ones((15, 15), np.uint8))
    gc[er > 0] = cv2.GC_PR_FGD
    bgm, fgm = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, gc, None, bgm, fgm, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return rough_mask
    out = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # 只保留与粗掩码重叠最大的连通域(防止 GrabCut 跑到别处)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(out, 8)
    best, best_ov = 0, -1
    for i in range(1, n):
        ov = int(((lbl == i) & (rough_mask > 0)).sum())
        if ov > best_ov:
            best, best_ov = i, ov
    if best:
        out = np.where(lbl == best, 255, 0).astype(np.uint8)
    return clean_mask(out)


def envelope_fill(xs, xe, window=40):
    """用上下 ±window 行的最大范围包络,填掉边界的局部凹陷。

    沟底若有浅色斑块(被太阳照亮),逐行暗带会在这里断掉,边界出现缺口。
    取邻域内最宽的范围可以桥接这类缺口,同时保留整体的宽度变化趋势。
    """
    w = max(1, int(window))
    offs = np.arange(-w, w + 1)
    xs_stack = np.stack([np.roll(xs, s) for s in offs])
    xe_stack = np.stack([np.roll(xe, s) for s in offs])
    # 避免 roll 的循环污染两端
    for k, s in enumerate(offs):
        if s > 0:
            xs_stack[k, :s] = np.inf
            xe_stack[k, :s] = -np.inf
        elif s < 0:
            xs_stack[k, s:] = np.inf
            xe_stack[k, s:] = -np.inf
    return xs_stack.min(0), xe_stack.max(0)


def darkrun_mask(bgr, y_range=(0.20, 0.84), x_band=(0.14, 0.86), center_band=(0.30, 0.70),
                 smooth_k=31, min_width_frac=0.05, fill_window=40,
                 extend_start=0.70, extend_end=0.99, extend_expand=0.26):
    """逐行找"最长的暗色连通段"(沟壑在每行就是中间那条暗带)。

    约束:① 暗段中心必须落在画面中部(center_band),避免选中崖面暗斑;
         ② 长度减去偏离画面中心的惩罚作为打分,再对两端点做中值平滑,保证跨行连续;
         ③ envelope_fill 桥接沟底浅色斑块造成的缺口;
         ④ 沟壑口以下按梯形延伸,把底部堆积体一并纳入。
    """
    H, W = bgr.shape[:2]
    norm = normalize_illumination(bgr)
    gray = cv2.GaussianBlur(cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 2)

    y0, y1 = int(y_range[0] * H), int(y_range[1] * H)
    bx0, bx1 = int(x_band[0] * W), int(x_band[1] * W)
    c0, c1 = center_band[0] * W, center_band[1] * W
    cx0 = 0.5 * W

    runs = {}
    for y in range(y0, y1):
        row = gray[y, bx0:bx1]
        thr = cv2.threshold(row.astype(np.uint8), 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[0]
        dark = (row < thr).astype(np.uint8)
        d = np.diff(np.concatenate([[0], dark, [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        best, best_score = None, -1e9
        for s, e in zip(starts, ends):
            center = bx0 + (s + e) / 2
            if not (c0 < center < c1):
                continue
            score = (e - s) - 0.8 * abs(center - cx0)
            if score > best_score:
                best_score, best = score, (bx0 + s, bx0 + e)
        if best:
            runs[y] = best

    mask = np.zeros((H, W), np.uint8)
    if len(runs) < 5:
        return mask, np.zeros((H, W), np.uint8)
    ys = sorted(runs)
    xs = medfilt1(np.array([runs[y][0] for y in ys], float), smooth_k)
    xe = medfilt1(np.array([runs[y][1] for y in ys], float), smooth_k)
    if fill_window > 0:
        xs, xe = envelope_fill(xs, xe, fill_window)

    for y, a, b in zip(ys, xs, xe):
        if b - a > min_width_frac * W:
            mask[y, int(a):int(b)] = 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    gully = clean_mask(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3))

    # ---- 底部堆积体:从沟壑口(extend_start)起向下张开的独立掩模 ----
    debris = np.zeros((H, W), np.uint8)
    if extend_end > 0:
        ys_arr = np.array(ys)
        yb = int(min(H - 1, max(ys_arr[0], extend_start * H)))
        near = int(np.abs(ys_arr - yb).argmin())  # 取起始行附近的实测宽度
        a0, b0 = xs[near], xe[near]
        ye = min(H - 1, int(extend_end * H))
        for y in range(yb + 1, ye + 1):
            t = (y - yb) / max(ye - yb, 1)
            ramp = min(1.0, t / 0.7) ** 0.85  # 先快速张开,再保持
            a = a0 - ramp * extend_expand * W
            b = b0 + ramp * extend_expand * W
            debris[y, max(0, int(a)):min(W, int(b))] = 255
        debris = clean_mask(cv2.morphologyEx(debris, cv2.MORPH_CLOSE, k, iterations=3))
        debris[gully > 0] = 0  # 与沟壑互斥,避免重复统计
    return gully, debris


def detect_gully(bgr, y_range=(0.20, 0.93), x_left=(0.18, 0.52), x_right=(0.48, 0.78),
                 max_jump=40, method="darkrun", fill_window=40,
                 extend_start=0.70, extend_end=0.99, extend_expand=0.26):
    """自动检测中央沟壑。返回 dict(mask, color_mask, edge_mask, left, right, seeds)。

    可被 features.py 复用(把检测到的沟壑当 ROI)。
    """
    H, W = bgr.shape[:2]
    y0, y1 = int(y_range[0] * H), int(y_range[1] * H)
    lx0, lx1 = int(x_left[0] * W), int(x_left[1] * W)
    rx0, rx1 = int(x_right[0] * W), int(x_right[1] * W)

    norm = normalize_illumination(bgr)
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.5)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)

    # B: 逐行边界追踪
    seed_ly, seed_lx = find_seed(gx, y0, y1, lx0, lx1, -1)  # 左壁:亮→暗,负梯度
    seed_ry, seed_rx = find_seed(gx, y0, y1, rx0, rx1, +1)  # 右壁:暗→亮,正梯度
    left = track_boundary(gx, y0, y1, seed_ly, seed_lx, lx0, lx1, -1, max_jump)
    right = track_boundary(gx, y0, y1, seed_ry, seed_rx, rx0, rx1, +1, max_jump)

    edge_mask = np.zeros((H, W), np.uint8)
    for y in range(y0, y1 + 1):
        if np.isnan(left[y]) or np.isnan(right[y]):
            continue
        xa, xb = int(min(left[y], right[y])), int(max(left[y], right[y]))
        if xb - xa > 0.02 * W:
            edge_mask[y, xa:xb] = 255
    edge_mask = clean_mask(edge_mask)

    # A: 颜色法(中央带内自适应阈值:暗 + 低饱和)
    hsv = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1].astype(np.float32), hsv[:, :, 2].astype(np.float32)
    band = np.zeros((H, W), np.uint8)
    band[y0:y1, int(0.15 * W):int(0.85 * W)] = 255
    v_thr = np.percentile(V[band > 0], 35)
    s_thr = np.percentile(S[band > 0], 45)
    color_mask = ((V < v_thr) & (S < s_thr) & (band > 0)).astype(np.uint8) * 255
    color_mask = clean_mask(color_mask)

    # C: 混合
    hybrid = clean_mask(cv2.bitwise_and(edge_mask, cv2.bitwise_or(color_mask, edge_mask)))
    # D: GrabCut 细化(推荐:底部斜边和宽度变化都能贴合)
    grabcut = grabcut_refine(bgr, hybrid)
    # E: 逐行暗带追踪(本场景最稳:直接利用"沟壑比两侧暗")+ 底部堆积体独立掩模
    darkrun, debris = darkrun_mask(bgr, y_range=(y_range[0], min(y_range[1], 0.88)),
                                   fill_window=fill_window, extend_start=extend_start,
                                   extend_end=extend_end, extend_expand=extend_expand)
    mask = {"color": color_mask, "edge": edge_mask, "hybrid": hybrid,
            "grabcut": grabcut, "darkrun": darkrun}[method]
    return {"mask": mask, "debris": debris, "mask_all": cv2.bitwise_or(mask, debris),
            "color_mask": color_mask, "edge_mask": edge_mask,
            "hybrid": hybrid, "grabcut": grabcut, "darkrun": darkrun,
            "left": left, "right": right, "seeds": (seed_lx, seed_ly, seed_rx, seed_ry),
            "gx": gx, "norm": norm}


def main():
    args = parse_args()
    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"读不到图片: {args.image}")
    H, W = bgr.shape[:2]

    res = detect_gully(bgr, tuple(args.y_range), tuple(args.x_left), tuple(args.x_right),
                       args.max_jump, args.method, args.fill_window,
                       args.extend_start, args.extend_end, args.extend_expand)
    color_mask, edge_mask = res["color_mask"], res["edge_mask"]
    chosen, left, right, gx = res["mask"], res["left"], res["right"], res["gx"]
    seed_lx, seed_ly, seed_rx, seed_ry = res["seeds"]
    y0, y1 = int(args.y_range[0] * H), int(args.y_range[1] * H)
    hsv = cv2.cvtColor(res["norm"], cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1].astype(np.float32), hsv[:, :, 2].astype(np.float32)

    def stat(name, m):
        mm = m > 0
        if not mm.any():
            print(f"{name}: 空")
            return
        print(f"{name}: 面积 {mm.mean()*100:.1f}% | 内部亮度 {V[mm].mean():.0f} "
              f"vs 外部 {V[~mm].mean():.0f} (差 {V[~mm].mean()-V[mm].mean():+.0f})")
    print(f"图像 {W}x{H} | 边界种子: 左({seed_lx},{seed_ly}) 右({seed_rx},{seed_ry})")
    stat("颜色法", color_mask)
    stat("边界法", edge_mask)
    stat("沟壑掩模", res["mask"])
    stat("堆积体掩模", res["debris"])
    print(f"最终采用: {args.method}")

    # ---- 可视化:沟壑=绿,堆积体=橙 ----
    overlay = bgr.copy()
    for m, color in ((res["mask"], (0, 255, 0)), (res["debris"], (0, 165, 255))):
        mm = m > 0
        if mm.any():
            tint = np.zeros_like(bgr)
            tint[:] = color
            overlay[mm] = (0.5 * bgr[mm] + 0.5 * tint[mm]).astype(np.uint8)
    # 画边界曲线
    curve = bgr.copy()
    for y in range(y0, y1 + 1):
        if not np.isnan(left[y]):
            cv2.circle(curve, (int(left[y]), y), 2, (0, 0, 255), -1)
        if not np.isnan(right[y]):
            cv2.circle(curve, (int(right[y]), y), 2, (255, 0, 0), -1)
    gx_vis = cv2.normalize(np.abs(gx), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gx_vis = cv2.cvtColor(gx_vis, cv2.COLOR_GRAY2BGR)

    def put(img, text):
        img = img.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (30, 30, 30), -1)
        cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return img

    h = 380
    def rz(img):
        w = int(img.shape[1] * h / img.shape[0])
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    # 两个掩模分别单色显示
    gully_vis = np.zeros_like(bgr)
    gully_vis[res["mask"] > 0] = (0, 255, 0)
    debris_vis = np.zeros_like(bgr)
    debris_vis[res["debris"] > 0] = (0, 165, 255)

    panels = [put(rz(bgr), "Input"),
              put(rz(curve), "Edge tracking (red=left, blue=right)"),
              put(rz(gully_vis), f"Gully mask (green, {res['mask'].mean()/255*100:.1f}%)"),
              put(rz(debris_vis), f"Debris mask (orange, {res['debris'].mean()/255*100:.1f}%)"),
              put(rz(overlay), "Final: gully + debris (two masks)")]
    row1 = np.hstack([np.hstack([p, np.full((h, 6, 3), 255, np.uint8)]) for p in panels[:3]])
    row2 = np.hstack([np.hstack([p, np.full((h, 6, 3), 255, np.uint8)]) for p in panels[3:]])
    w = max(row1.shape[1], row2.shape[1])
    row1 = cv2.copyMakeBorder(row1, 0, 0, 0, w - row1.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
    row2 = cv2.copyMakeBorder(row2, 0, 0, 0, w - row2.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
    canvas = np.vstack([row1, np.full((6, w, 3), 255, np.uint8), row2])
    cv2.imwrite(args.out, canvas)
    print(f"已保存: {args.out}")


if __name__ == "__main__":
    main()
