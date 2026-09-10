"""② 深度 + 点云/主平面:复用 core 的模型与反投影,导出复用 core.export_*。"""

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QMessageBox, QPushButton, QSplitter, QVBoxLayout, QWidget)

from gui.imageview import ImageView


def turbo(gray01: np.ndarray) -> np.ndarray:
    """0~1 → turbo 伪彩 uint8 (H,W,3)"""
    import matplotlib.cm as cm
    return (cm.turbo(np.clip(gray01, 0, 1))[..., :3] * 255).astype(np.uint8)


def fit_plane(pts):
    """PCA 拟合平面 → (法向, 质心, rms)。

    这里复写一小段而不是 from visualize3d import fit_plane:后者在 import 时会执行
    matplotlib.use("Agg"),会把 Qt 的绘图后端顶掉。
    """
    c = pts.mean(0)
    X = pts - c
    _, vecs = np.linalg.eigh(X.T @ X / len(pts))
    n = vecs[:, 0]
    if n @ np.array([0.0, 1.0, 0.0]) < 0:
        n = -n
    return n, c, float(np.sqrt(((X @ n) ** 2).mean()))


def compute(pil, max_depth, fov):
    """后台执行:深度图 + 点云 + 主平面参数"""
    import core
    disp = np.array(core.get_model("da2s")(pil)["predicted_depth"]).astype(np.float32)
    d01 = (disp - disp.min()) / (disp.max() - disp.min() + 1e-6)
    pts, cols, valid = core.backproject(pil, d01, max_depth=max_depth, fov_deg=fov)
    h, w = valid.shape
    roi = np.zeros((h, w), bool)
    roi[int(0.18 * h):int(0.98 * h), int(0.28 * w):int(0.72 * w)] = True
    roi &= valid
    out = {"d01": d01, "tilt": None, "rms": None, "n_valid": int(valid.sum())}
    if roi.sum() > 50:
        n, _c, rms = fit_plane(pts[roi])
        out["tilt"] = float(np.degrees(np.arccos(np.clip(abs(n @ np.array([0.0, 1.0, 0.0])), 0, 1))))
        out["rms"] = rms
        # 抽样点云供 3D 预览
        P, C = pts[valid], cols[valid]
        idx = np.random.default_rng(0).choice(len(P), size=min(20000, len(P)), replace=False)
        out["pts"] = P[idx]
        out["cols"] = C[idx].astype(np.float32) / 255.0
        out["roi_mask"] = roi[valid][idx]
    return out


class DepthTab(QWidget):
    def __init__(self, session, runner, parent=None):
        super().__init__(parent)
        self.session = session
        self.runner = runner
        self._depth01 = None

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_panel())
        split.addWidget(self._build_views())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([380, 1040])
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)

        self.session.image_changed.connect(self._on_image)
        self.session.device_changed.connect(lambda *_: self._sync_device())

    # ---------------------------------------------------------------- 面板
    def _build_panel(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(8, 8, 8, 8)

        self.gbox = QGroupBox("深度参数(默认取当前设备配置,可临时改)")
        f = QFormLayout(self.gbox)
        self.sp_max = QDoubleSpinBox(); self.sp_max.setRange(1, 200); self.sp_max.setDecimals(1)
        self.sp_max.setSingleStep(1.0); self.sp_max.setSuffix(" m")
        f.addRow("场景最远距离", self.sp_max)
        self.sp_fov = QDoubleSpinBox(); self.sp_fov.setRange(10, 180); self.sp_fov.setDecimals(1)
        self.sp_fov.setSuffix(" °")
        f.addRow("相机水平 FOV", self.sp_fov)
        v.addWidget(self.gbox)

        self.chk_pc = QCheckBox("导出点云 .ply(写入 pointclouds/)")
        self.chk_mesh = QCheckBox("导出网格 .ply(三角面,表面连续)")
        v.addWidget(self.chk_pc)
        v.addWidget(self.chk_mesh)

        btn = QPushButton("运行深度估计")
        btn.clicked.connect(self._run)
        v.addWidget(btn)

        self.lbl = QLabel("—")
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet("color:#333;")
        v.addWidget(self.lbl)
        v.addStretch()

        self._sync_device()
        return panel

    def _build_views(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        top = QHBoxLayout()
        top.addWidget(QLabel("深度图(越红越近)"))
        top.addStretch()
        v.addLayout(top)
        self.view = ImageView()
        v.addWidget(self.view, 3)

        v.addWidget(QLabel("点云 + 主平面(中央区域 PCA,仅预览;正式特征值见「③ 特征」)"))
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self.canvas = FigureCanvasQTAgg(Figure(figsize=(6, 2.6), dpi=100))
            self.ax = self.canvas.figure.add_subplot(111, projection="3d")
            self.canvas.figure.tight_layout()
            v.addWidget(self.canvas, 2)
        except Exception as e:                       # matplotlib 不可用也不影响其余功能
            self.canvas = None
            v.addWidget(QLabel(f"(3D 预览不可用: {e})"))
        return w

    # ---------------------------------------------------------------- 逻辑
    def _sync_device(self):
        import features as F
        p = F.resolve_params(self.session.device)
        self.sp_max.setValue(float(p["max_depth"]))
        self.sp_fov.setValue(float(p["fov"]))

    def _on_image(self):
        if self.session.has_image():
            self.view.set_array(self.session.rgb)

    def _run(self):
        if not self.session.has_image():
            QMessageBox.information(self, "无图片", "请先打开一张图片")
            return
        pil = Image.fromarray(self.session.rgb)
        md, fov = self.sp_max.value(), self.sp_fov.value()
        do_pc, do_mesh = self.chk_pc.isChecked(), self.chk_mesh.isChecked()

        def job():
            out = compute(pil, md, fov)
            paths = []
            if do_pc:
                import core
                paths.append(core.export_pointcloud(pil, out["d01"], max_depth=md, fov_deg=fov))
            if do_mesh:
                import core
                paths.append(core.export_mesh(pil, out["d01"], max_depth=md, fov_deg=fov))
            out["paths"] = paths
            return out

        self.runner.run(job, name="深度估计", on_done=self._apply,
                        on_error=lambda m: QMessageBox.critical(self, "失败", m))

    def _apply(self, out):
        self._depth01 = out["d01"]
        self.view.set_array(turbo(out["d01"]))
        txt = [f"有效点 {out['n_valid']:,}",
               f"主平面倾角 {out['tilt']:.1f}°" if out["tilt"] is not None else "ROI 内点太少",
               f"平面残差 {out['rms']:.4f}" if out["rms"] is not None else ""]
        if out.get("paths"):
            txt.append("已导出:" + "、".join(p.split("/")[-1] for p in out["paths"]))
        self.lbl.setText("  ·  ".join(t for t in txt if t))
        if self.canvas is not None and "pts" in out:
            self._plot3d(out)

    def _plot3d(self, out):
        self.ax.clear()
        P, C = out["pts"], out["cols"]
        inside = out["roi_mask"]
        self.ax.scatter(P[~inside, 0], P[~inside, 1], P[~inside, 2], s=1, c="#9AA0A6", alpha=0.35)
        self.ax.scatter(P[inside, 0], P[inside, 1], P[inside, 2], s=2, c=C[inside], alpha=0.9)
        self.ax.set_xlabel("X"); self.ax.set_ylabel("Y"); self.ax.set_zlabel("Z")
        self.ax.set_box_aspect((1, 0.5, 1))
        self.canvas.draw_idle()
