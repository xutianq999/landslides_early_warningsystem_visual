"""YOLOE / 深度估计 小工具:左侧原图,右侧选模型推理,结果显示在右侧。

启动:
    KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python app.py
然后浏览器打开 http://127.0.0.1:7860
"""

import os

try:
    import pillow_heif
    pillow_heif.register_heif_opener()  # 支持 iPhone HEIC/HEIF 图片拖入
except ImportError:
    pass

import numpy as np
import time
import gradio as gr
from PIL import Image

from core import DEVICE, backproject, export_mesh, export_pointcloud, get_model


def turbo_colormap(gray01: np.ndarray) -> np.ndarray:
    """0~1 灰度 → turbo 伪彩 (H,W,3) uint8"""
    import matplotlib.cm as cm
    return (cm.turbo(gray01)[..., :3] * 255).astype(np.uint8)


def infer_seg(image, classes_text, conf):
    """YOLOE 零样本实例分割"""
    if image is None:
        raise gr.Error("请先在左侧上传图片")
    classes = [c.strip() for c in classes_text.split(",") if c.strip()]
    if not classes:
        raise gr.Error("请填写至少一个类别(逗号分隔)")
    model = get_model("yoloe")
    model.set_classes(classes)
    results = model.predict(source=image, conf=conf, device=DEVICE)
    r = results[0]
    n = 0 if r.boxes is None else len(r.boxes)
    lines = [f"{model.names[int(c)]}: {float(p):.2f}" for c, p in zip(r.boxes.cls, r.boxes.conf)] if n else []
    info = f"检测到 {n} 个目标\n" + "\n".join(lines)
    return r.plot(), info


def _infer_da2(image, model_id, export_pc=False, max_depth=10.0, export_fmt="点云"):
    """Depth Anything V2 通用推理(Small/Base 同接口),可选导出 .ply 点云/网格"""
    pipe = get_model(model_id)
    pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    result = pipe(pil)
    depth = np.array(result["predicted_depth"])
    d01 = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    d8 = (d01 * 255).astype(np.uint8)
    colored = turbo_colormap(np.array(Image.fromarray(d8).resize(pil.size, Image.BILINEAR)) / 255.0)
    size = {"da2s": "Small 24.8M", "da2b": "Base 97M"}[model_id]
    info = f"相对深度(逆深度,越红越近)\nDA V2 {size}, {DEVICE}"

    ply_path = None
    if export_pc:
        try:
            fn = export_mesh if export_fmt.startswith("网格") else export_pointcloud
            ply_path = fn(pil, d01, max_depth=max_depth)
            info += f"\n{export_fmt}已导出:{ply_path}"
        except Exception as e:
            info += f"\n点云导出失败: {e}"
    return colored, info, ply_path


def infer_da2(image, export_pc=False, max_depth=10.0, export_fmt="点云"):
    if image is None:
        raise gr.Error("请先在左侧上传图片")
    return _infer_da2(image, "da2s", export_pc, max_depth, export_fmt)


def infer_da2_base(image, export_pc=False, max_depth=10.0, export_fmt="点云"):
    if image is None:
        raise gr.Error("请先在左侧上传图片")
    return _infer_da2(image, "da2b", export_pc, max_depth, export_fmt)


def rss_mb():
    """当前进程常驻内存 MB(MPS 统一内存,以此近似模型显存占用)"""
    import psutil
    return psutil.Process().memory_info().rss / 1024 / 1024


def infer_compare(image, export_pc=False, max_depth=10.0, export_fmt="点云"):
    """横向对比:原图 + 分割 + 两个深度模型,拼成一张宽图,标注耗时与内存"""
    if image is None:
        raise gr.Error("请先在左侧上传图片")
    import cv2
    panels = []
    infos = []

    def add(img, title, sub=""):
        h = 360
        img8 = img if img.dtype == np.uint8 else (img * 255).astype(np.uint8)
        scale = h / img8.shape[0]
        w = int(img8.shape[1] * scale)
        panel = cv2.resize(img8, (w, h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel, (0, 0), (w, 56), (30, 30, 30), -1)
        cv2.putText(panel, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if sub:
            cv2.putText(panel, sub, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
        panels.append(panel)

    add(image, "Input")
    # (显示名, 模型全称, 参数量/权重, 推理函数)
    entries = [
        ("YOLOE Seg", "yoloe-26s-seg.pt", "15.3M params / 31MB", infer_seg),
        ("DA V2 S", "Depth-Anything-V2-Small", "24.8M params / 99MB", infer_da2),
        ("DA V2 B", "Depth-Anything-V2-Base", "97M params / 390MB", infer_da2_base),
    ]
    for mode, full, size, fn in entries:
        try:
            mem0 = rss_mb()
            t0 = time.time()
            if fn is infer_seg:
                out, info = fn(image, "person,car,bus", 0.25)
                out = out[:, :, ::-1]  # ultralytics plot() 输出 BGR,转 RGB
            else:
                out, info, _ = fn(image)
            ms = (time.time() - t0) * 1000
            delta = rss_mb() - mem0
            add(out, f"{mode} | {size}", f"{full}  {ms:.0f}ms")
            infos.append(f"{mode} [{full}, {size}]: {info.splitlines()[0]}  [{ms:.0f}ms]")
        except Exception as e:
            infos.append(f"{mode}: 失败 {e}")

    gap = 6
    total_w = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)
    canvas = np.full((360, total_w, 3), 255, np.uint8)
    x = 0
    for p in panels:
        canvas[:, x:x + p.shape[1]] = p
        x += p.shape[1] + gap

    ply = None
    if export_pc:
        try:
            # da2s 刚在对比里加载过,这里复用缓存补一份点云/网格
            _, pc_info, ply = _infer_da2(image, "da2s", True, max_depth, export_fmt)
            infos.append(pc_info.splitlines()[-1])
        except Exception as e:
            infos.append(f"点云导出失败: {e}")
    return canvas, "\n".join(infos), ply


MODES = {
    "横向对比(全部模型)": infer_compare,
    "YOLOE 零样本分割": infer_seg,
    "Depth Anything V2 Small": infer_da2,
    "Depth Anything V2 Base": infer_da2_base,
}


def ui_extract_features(image, classes_text, prev, roi_auto, roi_target):
    """网页端特征提取:一张图 → 特征表 + 累积 CSV,并用 gr.State 记住上一帧做变化检测

    roi_auto=True 时先用 segment_gully 自动检测,按 roi_target 选择 ROI:
    沟壑 / 底部堆积体 / 两者合并。沟壑与堆积体是两个独立掩模(互不重叠)。
    """
    if image is None:
        raise gr.Error("请先在左侧上传图片")
    import csv as _csv
    from datetime import datetime
    import features as F

    pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    classes = [c.strip() for c in classes_text.split(",") if c.strip()] or None

    roi_mask, overlay, extra_masks = None, None, None
    if roi_auto:
        import cv2
        import segment_gully
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        res = segment_gully.detect_gully(bgr)
        key = {"沟壑": "mask", "堆积体": "debris", "两者": "mask_all"}[roi_target]
        roi_mask = res[key] > 0
        extra_masks = {"gully": res["mask"], "debris": res["debris"]}
        # 叠加预览:沟壑绿、堆积体橙,选中的区域加亮
        for m, color in ((res["mask"], (0, 255, 0)), (res["debris"], (0, 165, 255))):
            mm = m > 0
            if mm.any():
                tint = np.zeros_like(bgr)
                tint[:] = color
                bgr[mm] = (0.5 * bgr[mm] + 0.5 * tint[mm]).astype(np.uint8)
        sel = res[key] > 0
        edge = cv2.Canny((sel * 255).astype(np.uint8), 50, 150)
        bgr[edge > 0] = (255, 0, 0)
        for y in range(len(res["left"])):
            if not np.isnan(res["left"][y]):
                cv2.circle(bgr, (int(res["left"][y]), y), 2, (0, 0, 255), -1)
            if not np.isnan(res["right"][y]):
                cv2.circle(bgr, (int(res["right"][y]), y), 2, (255, 0, 0), -1)
        overlay = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    row, cur = F.features_from_image(pil, classes=classes, conf=0.15, prev=prev,
                                     roi_mask=roi_mask, extra_masks=extra_masks)
    row = {"time": datetime.now().isoformat(timespec="seconds"), **row}

    out = "features.csv"
    new_file = not os.path.exists(out)
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            w.writeheader()
        w.writerow(row)

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) and v == v else ("nan" if v != v else str(v))
    table = [[k, fmt(v)] for k, v in row.items()]
    return table, out, cur, overlay


def run(mode, image, classes_text, conf, export_pc, max_depth, export_fmt):
    try:
        if mode.startswith("YOLOE 零"):
            img, info = MODES[mode](image, classes_text, conf)
            return img, info, None, None
        img, info, ply = MODES[mode](image, export_pc, max_depth, export_fmt)
        return img, info, ply, ply
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"推理失败: {e}")


with gr.Blocks(title="视觉小工具:零样本分割 + 单目深度") as demo:
    gr.Markdown("# 零样本分割 + 单目深度 推理台\n"
                "左侧放原图并选模型 → 点「开始推理」→ 右侧看结果,底部预览点云。"
                "首次使用某个模型会先加载权重,稍等片刻。")
    with gr.Row():
        with gr.Column(scale=1, min_width=380):
            gr.Markdown("### 输入")
            input_img = gr.Image(label="原图", type="numpy", height=380)
            mode = gr.Radio(list(MODES.keys()), value="横向对比(全部模型)", label="模型")
            classes_text = gr.Textbox(
                value="person,car,dog,bicycle",
                label="类别(仅 YOLOE 分割用,逗号分隔,任意概念)",
                visible=True)
            conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="置信度阈值(仅 YOLOE)")
            export_pc = gr.Checkbox(value=False, label="导出点云/网格 .ply(DA V2 / 对比模式)")
            max_depth = gr.Slider(2, 50, value=10, step=1,
                                  label="场景最远距离/米(近似尺度)")
            export_fmt = gr.Radio(["点云", "网格"], value="点云",
                                  label="导出格式(网格=三角面,表面连续无稀疏感)")
            btn = gr.Button("开始推理", variant="primary")
            btn_feat = gr.Button("提取滑坡特征(不推理,只出特征表)", variant="secondary")
        with gr.Column(scale=1, min_width=380):
            gr.Markdown("### 结果")
            output_img = gr.Image(label="结果图", height=380)
            output_info = gr.Textbox(label="信息", lines=4)
            output_file = gr.File(label="点云下载(.ply)")
    output_3d = gr.Model3D(label="点云预览(拖拽旋转,滚轮缩放)", height=560)

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### 滑坡监测特征\n"
                        "点左侧「提取滑坡特征」→ 这里显示本次特征;连续提取会自动与上一张比对算变化量。"
                        "识别到的行人/车辆区域会从几何与时序特征中自动剔除。"
                        "勾选「自动检测沟壑」后,几何/深度/时序特征只在中央沟壑区域内计算。"
                        "完整流程见 PIPELINE.md,报警分析用 `alarm.py`。")
            feat_df = gr.Dataframe(headers=["特征", "值"], datatype=["str", "str"],
                                   label="本次特征", interactive=False, wrap=True)
        with gr.Column(scale=1):
            feat_classes = gr.Textbox(value="deep valley,person,car,landslide,truck,construction vehicle",
                                      label="分割类别(动态物体区域会自动从几何特征中剔除)")
            feat_roi_auto = gr.Checkbox(value=False, label="自动检测沟壑(只用检测到的区域算特征)")
            feat_roi_target = gr.Radio(["沟壑", "堆积体", "两者"], value="沟壑", label="ROI 目标")
            feat_file = gr.File(label="特征 CSV(累积追加)")
            feat_gully_img = gr.Image(label="区域检测(绿=沟壑,橙=堆积体,红线=左壁,蓝线=右壁)",
                                      height=300, interactive=False)
    prev_state = gr.State(None)

    # 选非分割模型时隐藏分割专属控件
    mode.change(lambda m: (gr.update(visible=m.startswith("YOLOE 分割") or m.startswith("YOLOE 零")),
                           gr.update(visible=m.startswith("YOLOE"))),
                inputs=mode, outputs=[classes_text, conf])
    btn.click(run, inputs=[mode, input_img, classes_text, conf, export_pc, max_depth, export_fmt],
              outputs=[output_img, output_info, output_file, output_3d])
    btn_feat.click(ui_extract_features,
                   inputs=[input_img, feat_classes, prev_state, feat_roi_auto, feat_roi_target],
                   outputs=[feat_df, feat_file, prev_state, feat_gully_img])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
