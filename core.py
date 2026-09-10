"""共享推理内核:设备选择、模型懒加载、深度反投影、PLY 导出。

app.py(网页)、features.py(特征提取)、demo_depth.py(命令行)都从这里取,
避免彼此 import 造成循环导入。
"""

import os

import numpy as np
import torch
from PIL import Image

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

_ROOT = os.path.dirname(os.path.abspath(__file__))

# DA V2 权重:优先项目内 models/(扁平文件,整个文件夹拷走即可部署),
# 缺失时回退 HuggingFace 本地缓存。两条路径都只读本地、不联网。
_DA2 = {
    "da2s": ("models/da2-small", "depth-anything/Depth-Anything-V2-Small-hf"),
    "da2b": ("models/da2-base", "depth-anything/Depth-Anything-V2-Base-hf"),
}

# 模型懒加载:首次用到某个模型时才加载权重,避免启动慢
_cache = {}


def get_model(name):
    if name in _cache:
        return _cache[name]
    if name == "yoloe":
        from ultralytics import YOLOE
        # 绝对路径:否则 ultralytics 按当前工作目录找权重,找不到会尝试联网下载
        m = YOLOE(os.path.join(_ROOT, "yoloe-26s-seg.pt"))
    elif name in ("da2s", "da2b"):
        # 只读本地,不联网。默认 pipeline(model="repo_id") 会先向 HF Hub 发一次版本核对请求,
        # 离线时退避重试 5 次(约 23 s)才回退缓存,纯属白等;local_files_only=True 直接读本地。
        # (YOLOE 的 mobileclip2_b.ts 由 ultralytics 按工作目录解析,故需在项目根目录运行。)
        from transformers import (AutoImageProcessor, AutoModelForDepthEstimation,
                                  pipeline)
        rel, repo = _DA2[name]
        local = os.path.join(_ROOT, rel)
        src = local if os.path.isdir(local) else repo  # 项目内优先,其次 HF 本地缓存
        try:
            model = AutoModelForDepthEstimation.from_pretrained(src, local_files_only=True)
            image_processor = AutoImageProcessor.from_pretrained(src, local_files_only=True)
        except OSError as e:
            raise RuntimeError(
                f"加载 DA V2 权重失败:项目内 {rel}/ 与 HuggingFace 本地缓存都没有 "
                f"{repo},且已设为只读本地(不联网)。"
                f"参考 README「模型权重与离线部署」准备权重。"
            ) from e
        source = f"项目内 {rel}/" if src == local else f"HF 缓存 {repo}"
        print(f"[core] 加载 DA V2 {name} ← {source}")
        m = pipeline("depth-estimation", model=model, image_processor=image_processor,
                     device=DEVICE)
    else:
        raise ValueError(name)
    _cache[name] = m
    return m


def backproject(pil, disp01, max_depth=10.0, fov_deg=60.0, max_points=1_500_000, offset=0.0):
    """相对深度(视差,越大越近)→ 相机坐标系点云。

    返回 (pts (h,w,3), cols (h,w,3), valid (h,w))。保留网格结构,网格导出要用拓扑。

    **关键**:z 必须由原始视差直接反演(z = 1/视差),不能先归一化到 [0,1] 再加偏移。
    归一化等价于给视差加了一个未知平移,会让陡坡的坡度饱和(实测:先归一化再加 0.02 偏移时,
    30° 以上全部塌到 23.8°;直接反演则 30°/45°/60°/75° 精确还原)。
    offset 是可选相机常数(视差单位):DA V2 存在仿射歧义 d = a/z + b,若能标定出 b 可传入。
    """
    W, H = pil.size
    # 超点数上限时隔行抽稀
    step = max(1, int(np.ceil(max(H, W) / np.sqrt(max_points))))
    disp = Image.fromarray(disp01.astype(np.float32), mode="F").resize(pil.size, Image.BILINEAR)
    disp = np.array(disp)[::step, ::step].astype(np.float32)
    rgb = np.array(pil)[::step, ::step, :3]

    # 无效值(0/NaN,如天空或无纹理区)不能当作"最远平面",否则 p2 被拉到 0 会让 z 整体塌缩
    finite = np.isfinite(disp) & (disp > 0)
    if finite.sum() < 100:  # 极端兜底:全图几乎无有效值
        finite = np.isfinite(disp)
    ref = disp[finite]
    lo, hi = np.percentile(ref, 2), np.percentile(ref, 98)
    disp = np.clip(np.where(finite, disp, lo), lo, hi) - offset
    # 双边滤波:压深度噪声同时保住物体边界(高斯会把边界糊成斜面,点云上表现为拉伸的尖刺)
    import cv2
    disp = cv2.bilateralFilter(disp, d=7, sigmaColor=0.08 * (hi - lo + 1e-6), sigmaSpace=7)

    z = 1.0 / np.maximum(disp, 1e-6)  # 视差直接反演;不减去远平面视差(那会压平陡坡)
    z = z / z.max() * max_depth  # 近似米制:最远 = max_depth
    fx = 0.5 * W / np.tan(np.radians(fov_deg) / 2)
    cx, cy = W / 2, H / 2
    u = np.arange(z.shape[1]) * step - cx
    v = np.arange(z.shape[0]) * step - cy
    x = u[None, :] * z / fx
    y = -v[:, None] * z / fx  # 图像行方向向下,翻正成世界坐标的"上"

    valid = finite & (z < max_depth * 0.999)  # 无效区与贴最远平面的点都不参与
    pts = np.stack([x, y, z], axis=-1)
    # 注意:返回的是相机坐标系(未平移)。平移会让近处 z 变负,破坏依赖正值假设的
    # 深度断裂判断(max < min×2);需要居中显示的调用方(点云/网格导出)自行平移。
    return pts, rgb, valid


def export_pointcloud(pil, disp01, max_depth=10.0, fov_deg=60.0, max_points=1_500_000,
                      out_dir="pointclouds", stem=None):
    """点云导出(离散点)。导出前平移到质心,便于查看器自动取景。"""
    pts, cols, valid = backproject(pil, disp01, max_depth, fov_deg, max_points)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{_stem(pil, stem)}_pointcloud.ply")
    pts = pts - np.median(pts[valid], axis=0)
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
    pts = pts - np.median(pts[valid], axis=0)
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
