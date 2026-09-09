"""YOLOE 零样本分割:单张图片演示。

用法:
    python demo_image.py <图片路径> [--classes 人,车,狗] [--conf 0.25]

首次运行会自动下载 yoloe-26s-seg.pt 权重和 CLIP 文本编码器(约 254MB),
之后离线可用。
"""

import argparse
from pathlib import Path

from ultralytics import YOLOE

# 想换成你的场景,改这一行即可,类别任意、无需训练
DEFAULT_CLASSES = ["person", "car", "dog", "bicycle"]


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOE 零样本实例分割(图片)")
    parser.add_argument("image", nargs="?", default="test.jpg", help="输入图片路径")
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES),
                        help="逗号分隔的类别名,任意中文/英文概念")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--model", default="yoloe-26s-seg.pt", help="YOLOE 检查点")
    return parser.parse_args()


def pick_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"  # Apple GPU
    return "cpu"


def main():
    args = parse_args()
    image = Path(args.image)
    if not image.exists():
        raise SystemExit(f"找不到图片: {image} (可以放一张命名为 test.jpg)")

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    device = pick_device()
    print(f"[1/3] 加载模型 {args.model} (device={device}) ...")
    model = YOLOE(args.model)

    print(f"[2/3] 设置零样本类别: {classes}")
    model.set_classes(classes)  # 文本提示:CLIP 编码一次并缓存

    print(f"[3/3] 推理 {image.name} ...")
    results = model.predict(source=str(image), conf=args.conf, device=device)

    r = results[0]
    n = 0 if r.boxes is None else len(r.boxes)
    print(f"检测到 {n} 个目标")
    if r.boxes is not None:
        names = r.names
        for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            x1, y1, x2, y2 = (int(v) for v in box)
            print(f"  - {names[int(cls_id)]}: conf={float(conf):.2f} box=({x1},{y1},{x2},{y2})")

    out = image.with_stem(image.stem + "_result")
    r.save(filename=str(out))
    print(f"结果图已保存: {out}")


if __name__ == "__main__":
    main()
