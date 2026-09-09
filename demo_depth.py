"""单目深度估计演示:Depth Anything V2(零样本,无需训练)。

用法:
    python demo_depth.py <图片路径> [--model small|base] [--ply] [--max-depth 10] [--fov 60]

输出 <图片名>_depth.png(深度热力图,越亮越近);
加 --ply 额外输出 <图片名>_pointcloud.ply(RGB 点云,MeshLab/CloudCompare 可开)。
首次运行自动从 HuggingFace 下载权重(约 100~400MB)。
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import pipeline

MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",   # 最快
    "base": "depth-anything/Depth-Anything-V2-Base-hf",    # 略慢,更细
}


def parse_args():
    parser = argparse.ArgumentParser(description="单目深度估计")
    parser.add_argument("image", nargs="?", default="test.jpg", help="输入图片路径")
    parser.add_argument("--model", default="small", choices=MODELS.keys())
    parser.add_argument("--ply", action="store_true", help="同时导出 RGB 点云 .ply")
    parser.add_argument("--mesh", action="store_true",
                        help="导出三角网格 .ply(表面连续,无点云远景稀疏感)")
    parser.add_argument("--max-depth", type=float, default=10.0,
                        help="场景最远距离/米,作点云近似尺度(默认 10)")
    parser.add_argument("--fov", type=float, default=60.0, help="相机水平 FOV 角度(默认 60)")
    return parser.parse_args()


def main():
    args = parse_args()
    image = Path(args.image)
    if not image.exists():
        raise SystemExit(f"找不到图片: {image}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"加载 {MODELS[args.model]} (device={device}) ...")
    depth_estimator = pipeline("depth-estimation", model=MODELS[args.model], device=device)

    print(f"推理 {image.name} ...")
    import time
    t0 = time.time()
    result = depth_estimator(str(image))
    print(f"耗时 {time.time() - t0:.2f}s")

    depth = np.array(result["predicted_depth"])
    print(f"深度图尺寸: {depth.shape},范围: {depth.min():.2f} ~ {depth.max():.2f}(逆深度,值越大越近)")

    # Depth Anything 输出逆深度:值越大越近。归一化后近处亮/红,远处暗/蓝
    d = (depth - depth.min()) / (depth.max() - depth.min())
    d8 = (d * 255).astype(np.uint8)
    heat = Image.fromarray(d8).resize(Image.open(image).size, Image.BILINEAR)
    heat = Image.fromarray(np.stack([d8 := np.array(heat)] * 3, axis=-1))  # 灰度转3通道
    # 用 matplotlib 的 turbo 伪彩色更好看
    try:
        import matplotlib.cm as cm
        colored = (cm.turbo(np.array(heat)[:, :, 0] / 255.0)[..., :3] * 255).astype(np.uint8)
        heat = Image.fromarray(colored)
    except ImportError:
        pass  # 没装 matplotlib 就输出灰度图

    out = image.with_stem(image.stem + "_depth").with_suffix(".png")
    heat.save(out)
    print(f"深度图已保存: {out} (逆深度:越亮/越红越近,越蓝越远)")

    if args.ply or args.mesh:
        from core import export_mesh, export_pointcloud
        pil = Image.open(image).convert("RGB")
        fn = export_mesh if args.mesh else export_pointcloud
        out_ply = fn(pil, d, max_depth=args.max_depth, fov_deg=args.fov,
                     out_dir=str(image.parent), stem=image.stem)
        print(f"已保存: {out_ply} (最远≈{args.max_depth}m, FOV={args.fov}°, 近似尺度)")


if __name__ == "__main__":
    main()
