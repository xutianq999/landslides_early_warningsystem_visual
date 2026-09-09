"""共享推理内核:设备选择、模型懒加载、深度反投影、PLY 导出。

app.py(网页)、features.py(特征提取)、demo_depth.py(命令行)都从这里取,
避免彼此 import 造成循环导入。
"""

import os

import numpy as np
import torch
from PIL import Image

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# 模型懒加载:首次用到某个模型时才加载权重,避免启动慢
_cache = {}


def get_model(name):
    if name in _cache:
        return _cache[name]
    if name == "yoloe":
        from ultralytics import YOLOE
        m = YOLOE("yoloe-26s-seg.pt")
    elif name in ("da2s", "da2b"):
        from transformers import pipeline
        mid = {"da2s": "depth-anything/Depth-Anything-V2-Small-hf",
               "da2b": "depth-anything/Depth-Anything-V2-Base-hf"}[name]
        m = pipeline("depth-estimation", model=mid, device=DEVICE)
    else:
        raise ValueError(name)
    _cache[name] = m
    return m


def backproject(pil, disp01, max_depth=10.0, fov_deg=60.0, max_points=1_500_000):
    """相对深度(0~1,越大越近)→ 相机坐标系点云。

    返回 (pts (h,w,3), cols (h,w,3), valid (h,w))。保留网格结构,网格导出要用拓扑。
    DA V2 无绝对尺度:针孔模型反投影,水平 FOV 定焦距,z 按 1/(disp+0.1) 归一化到 max_depth。
    """
    W, H = pil.size
    # 超点数上限时隔行抽稀。注意全程浮点:量化成 8 位会让远端相邻级被 1/(d+c) 放大成波纹等高线
    step = max(1, int(np.ceil(max(H, W) / np.sqrt(max_points))))
    disp = Image.fromarray(disp01.astype(np.float32), mode="F").resize(pil.size, Image.BILINEAR)
    disp = np.array(disp)[::step, ::step]
    rgb = np.array(pil)[::step, ::step, :3]

    # 分位截断防离群点撑爆范围
    lo, hi = np.percentile(disp, 2), np.percentile(disp, 98)
    disp = np.clip((disp - lo) / (hi - lo + 1e-6), 0, 1)
    # 双边滤波:压深度噪声同时保住物体边界(高斯会把边界糊成斜面,点云上表现为拉伸的尖刺)
    import cv2
    disp = cv2.bilateralFilter(disp, d=7, sigmaColor=0.08, sigmaSpace=7)

    z = 1.0 / (disp + 0.1)  # 逆深度 → 深度;+0.1 限制远端拉伸倍率
    z = z / z.max() * max_depth  # 近似米制:最远 = max_depth
    fx = 0.5 * W / np.tan(np.radians(fov_deg) / 2)
    cx, cy = W / 2, H / 2
    u = np.arange(z.shape[1]) * step - cx
    v = np.arange(z.shape[0]) * step - cy
    x = u[None, :] * z / fx
    y = -v[:, None] * z / fx  # 图像行方向向下,翻正成世界坐标的"上"

    valid = z < max_depth * 0.999  # 贴最远平面的点无意义
    pts = np.stack([x, y, z], axis=-1)
    pts = pts - np.median(pts[valid], axis=0)  # 平移到质心,便于查看器自动取景
    return pts, rgb, valid


def export_pointcloud(pil, disp01, max_depth=10.0, fov_deg=60.0, max_points=1_500_000,
                      out_dir="pointclouds", stem=None):
    """点云导出(离散点)。"""
    pts, cols, valid = backproject(pil, disp01, max_depth, fov_deg, max_points)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{_stem(pil, stem)}_pointcloud.ply")
    write_ply(path, pts[valid], cols[valid])
    return path


def export_mesh(pil, disp01, max_depth=10.0, fov_deg=60.0, max_points=1_500_000,
                out_dir="pointclouds", stem=None, max_ratio=2.0):
    """网格导出(三角面片)。

    深度图本来就是规则网格,直接按相邻像素连三角面 → 表面连续,消除点云"远处稀疏"的观感。
    max_ratio 控制断裂:相邻四角深度比超过该倍数就不连面,避免把天空和近景缝成一张斜膜。
    """
    pts, cols, valid = backproject(pil, disp01, max_depth, fov_deg, max_points)
    h, w = valid.shape
    z = pts[..., 2]

    # 2x2 单元格四角都有效,且深度连续,才生成两个三角面
    cell = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    corners = np.stack([z[:-1, :-1], z[:-1, 1:], z[1:, :-1], z[1:, 1:]])
    zmin, zmax = corners.min(0), corners.max(0)
    cell &= zmax < zmin * max_ratio

    # 顶点压缩:只保留有效点,并建立 网格序号 → 压缩后序号 的映射
    idx = np.full(valid.size, -1, np.int64)
    idx[valid.ravel()] = np.arange(valid.sum())
    ii, jj = np.nonzero(cell)
    v00 = ii * w + jj
    faces = np.concatenate([
        np.stack([idx[v00], idx[v00 + 1], idx[v00 + w + 1]], 1),
        np.stack([idx[v00], idx[v00 + w + 1], idx[v00 + w]], 1),
    ])
    faces = faces[(faces >= 0).all(1)]

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{_stem(pil, stem)}_mesh.ply")
    write_ply(path, pts[valid], cols[valid], faces)
    return path


def _stem(pil, stem):
    if stem is None:
        stem = os.path.splitext(os.path.basename(str(getattr(pil, "filename", None) or "scene")))[0]
    return stem


def write_ply(path, pts, cols, faces=None):
    """二进制 PLY:顶点 x y z float32 + r g b uint8,可选三角面。MeshLab / CloudCompare / Blender 可直接开"""
    n = pts.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    )
    if faces is not None:
        header += (f"element face {len(faces)}\n"
                   "property list uchar int vertex_indices\n")
    header += "end_header\n"

    data = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                              ("r", "u1"), ("g", "u1"), ("b", "u1")])
    data["x"], data["y"], data["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    data["r"], data["g"], data["b"] = cols[:, 0], cols[:, 1], cols[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
        if faces is not None:
            fdata = np.empty(len(faces), dtype=[("n", "u1"), ("v", "<i4", (3,))])
            fdata["n"] = 3
            fdata["v"] = faces
            f.write(fdata.tobytes())
