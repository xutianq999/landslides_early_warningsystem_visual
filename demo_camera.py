"""YOLOE 零样本分割:摄像头实时演示(Mac)。

用法:
    python demo_camera.py [--classes 人,车,狗] [--conf 0.3]

按 q 退出。
"""

import argparse

import cv2
from ultralytics import YOLOE

DEFAULT_CLASSES = ["person", "cup", "phone", "book"]


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOE 零样本实例分割(摄像头)")
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES),
                        help="逗号分隔的类别名")
    parser.add_argument("--conf", type=float, default=0.3, help="置信度阈值")
    parser.add_argument("--model", default="yoloe-26s-seg.pt", help="YOLOE 检查点")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号")
    return parser.parse_args()


def pick_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = parse_args()
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    device = pick_device()
    print(f"加载 {args.model} (device={device}),零样本类别: {classes}")

    model = YOLOE(args.model)
    model.set_classes(classes)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("无法打开摄像头(系统设置里检查相机权限)")

    frame_count, import_time = 0, __import__("time").time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, conf=args.conf, device=device, verbose=False)
        annotated = results[0].plot()  # 框 + 掩码 + 类别名

        frame_count += 1
        elapsed = __import__("time").time() - import_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLOE zero-shot segmentation (q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
