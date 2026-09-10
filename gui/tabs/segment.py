"""① 分割工作台:沟壑分割(5 种方法)+ 坡体颜色分割,参数全暴露、实时预览、可导出。

算法全部复用 segment_gully.detect_gully / segment_color.detect_slope,
不做第二套实现,因此结果与命令行逐像素一致。
"""

import json
from datetime import datetime

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QMenu, QMessageBox, QPushButton,
                               QScrollArea, QSpinBox, QSplitter, QToolButton, QVBoxLayout,
                               QWidget)

import config
import segment_color
import segment_gully
from gui.imageview import ImageView

# 界面标签 → (类别, 方法名)——把 5 种方法全部暴露出来
METHODS = [
    ("gully", "darkrun", "沟壑:逐行暗带(darkrun)"),
    ("gully", "edge", "沟壑:边界追踪(edge)"),
    ("gully", "color", "沟壑:颜色(color)"),
    ("gully", "hybrid", "沟壑:混合(hybrid)"),
    ("gully", "grabcut", "沟壑:GrabCut(grabcut)"),
    ("slope", None, "坡体:颜色 K-means"),
]

VIEWS = [
    ("overlay", "叠加(掩模+边界)"),
    ("mask", "掩模"),
    ("boundary", "左右边界曲线"),
    ("grad", "梯度图"),
]


def compute(bgr, kind, method, params):
    """在后台线程里执行的实际计算(纯函数,便于测试)"""
    if kind == "gully":
        return kind, method, segment_gully.detect_gully(bgr, method=method, **(params or {}))
    return kind, None, segment_color.detect_slope(bgr, **(params or {}))


def render(bgr, kind, method, res, view, y_range=None):
    """结果 → RGB 图(与命令行可视化同样的颜色约定)

    y_range 给定时,边界曲线只画在追踪带内——否则 track_boundary 会把 x 外推到整幅
    图高度,顶部出现误导性的竖直直线(命令行可视化也只画带内)。
    """
    out = bgr.copy()
    if kind == "gully":
        mask, debris = res["mask"], res["debris"]
        h = bgr.shape[0]
        if y_range:
            ys = range(int(y_range[0] * h), int(y_range[1] * h) + 1)
        else:
            ys = range(h)
        if view == "mask":
            out = np.zeros_like(bgr)
            out[debris > 0] = (0, 165, 255)
            out[mask > 0] = (0, 255, 0)
        elif view == "grad":
            g = cv2.normalize(np.abs(res["gx"]), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        elif view == "boundary":
            for y in ys:
                if not np.isnan(res["left"][y]):
                    cv2.circle(out, (int(res["left"][y]), y), 2, (0, 0, 255), -1)
                if not np.isnan(res["right"][y]):
                    cv2.circle(out, (int(res["right"][y]), y), 2, (255, 0, 0), -1)
        else:  # overlay
            for m, color in ((mask, (0, 255, 0)), (debris, (0, 165, 255))):
                mm = m > 0
                if mm.any():
                    tint = np.zeros_like(bgr)
                    tint[:] = color
                    out[mm] = (0.5 * bgr[mm] + 0.5 * tint[mm]).astype(np.uint8)
            edge = cv2.Canny((mask > 0).astype(np.uint8) * 255, 50, 150)
            out[edge > 0] = (255, 0, 0)
            for y in ys:
                if not np.isnan(res["left"][y]):
                    cv2.circle(out, (int(res["left"][y]), y), 1, (0, 0, 255), -1)
                if not np.isnan(res["right"][y]):
                    cv2.circle(out, (int(res["right"][y]), y), 1, (255, 0, 0), -1)
    else:
        mask = res["mask"]
        if view == "mask":
            out = np.zeros_like(bgr)
            out[mask > 0] = (0, 255, 0)
        else:
            mm = mask > 0
            tint = np.zeros_like(bgr)
            tint[:] = (0, 255, 0)
            out[mm] = (0.55 * bgr[mm] + 0.45 * tint[mm]).astype(np.uint8)
            edge = cv2.Canny(mask, 50, 150)
            out[edge > 0] = (255, 0, 0)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


class SegmentTab(QWidget):
    def __init__(self, session, runner, parent=None):
        super().__init__(parent)
        self.session = session
        self.runner = runner
        self._result = None
        self._raw = None                     # 后台返回的原始结果

        split = QSplitter(Qt.Horizontal, self)
        split.addWidget(self._build_params_panel())
        split.addWidget(self._build_view_panel())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([420, 1000])

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)      # 拖滑块时不要每次都算
        self._debounce.timeout.connect(self._recompute)

        self.session.image_changed.connect(self._on_image)
        self._on_method_change()

    # ---------------------------------------------------------------- 参数面板
    def _build_params_panel(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(8, 8, 8, 8)

        self.method_combo = QComboBox()
        for kind, method, label in METHODS:
            self.method_combo.addItem(label, (kind, method))
        self.method_combo.currentIndexChanged.connect(self._on_method_change)
        form = QFormLayout()
        form.addRow("方法", self.method_combo)
        v.addLayout(form)

        # ---- 沟壑参数(全部来自 GULLY_DEFAULTS,与命令行同一套) ----
        self.gbox_gully = QGroupBox("沟壑参数")
        g = QFormLayout(self.gbox_gully)
        d = segment_gully.GULLY_DEFAULTS
        self.sp_y0, self.sp_y1 = self._pair(g, "y 范围(上/下)", d["y_range"], 0.0, 1.0, 0.01)
        self.sp_xl0, self.sp_xl1 = self._pair(g, "左壁搜索 x", d["x_left"], 0.0, 1.0, 0.01)
        self.sp_xr0, self.sp_xr1 = self._pair(g, "右壁搜索 x", d["x_right"], 0.0, 1.0, 0.01)
        self.sp_jump = self._int(g, "边界最大跳变 px", d["max_jump"], 1, 300)
        self.sp_fill = self._int(g, "凹陷填充窗口 行", d["fill_window"], 0, 300)
        self.sp_es = self._dbl(g, "底部延伸起点 y", d["extend_start"], 0.0, 1.0, 0.01)
        self.sp_ee = self._dbl(g, "底部延伸终点 y", d["extend_end"], 0.0, 1.0, 0.01)
        self.sp_ex = self._dbl(g, "底部张开量(占宽)", d["extend_expand"], 0.0, 0.8, 0.01)
        self.sp_y0.setToolTip("沟壑在画面中的纵向范围,只在带内追踪")
        v.addWidget(self.gbox_gully)

        # ---- 坡体参数 ----
        self.gbox_slope = QGroupBox("坡体颜色分割参数")
        s = QFormLayout(self.gbox_slope)
        self.sp_k = self._int(s, "聚类数 k", 5, 2, 12)
        self.sp_spatial = self._dbl(s, "空间权重", 25.0, 0.0, 100.0, 1.0)
        self.sp_cluster = self._int(s, "指定簇号(-1=自动)", -1, -1, 12)
        v.addWidget(self.gbox_slope)

        row = QHBoxLayout()
        btn_recalc = QPushButton("重算")
        btn_recalc.clicked.connect(self._recompute)
        btn_default = QPushButton("恢复默认")
        btn_default.clicked.connect(self._reset_defaults)
        self.btn_export = QToolButton()
        self.btn_export.setText("导出 ▾")
        self.btn_export.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.btn_export)
        for text, key in (("掩模 PNG…", "mask"), ("叠加图 PNG…", "overlay"), ("参数 JSON…", "json")):
            act = QAction(text, self)
            act.triggered.connect(lambda _=False, k=key: self._export(k))
            menu.addAction(act)
        self.btn_export.setMenu(menu)
        row.addWidget(btn_recalc)
        row.addWidget(btn_default)
        row.addWidget(self.btn_export)
        v.addLayout(row)

        self.lbl_pick = QLabel("点击图像可查看该点是否落在掩模内")
        self.lbl_pick.setStyleSheet("color:#666;")
        self.lbl_pick.setWordWrap(True)
        v.addWidget(self.lbl_pick)
        v.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(400)
        return scroll

    def _pair(self, form: QFormLayout, label, value, lo, hi, step):
        a = QDoubleSpinBox()
        a.setRange(lo, hi)
        a.setSingleStep(step)
        a.setDecimals(2)
        a.setValue(float(value[0]))
        b = QDoubleSpinBox()
        b.setRange(lo, hi)
        b.setSingleStep(step)
        b.setDecimals(2)
        b.setValue(float(value[1]))
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(a)
        h.addWidget(b)
        form.addRow(label, box)
        for w in (a, b):
            w.valueChanged.connect(self._touch)
        return a, b

    def _dbl(self, form, label, value, lo, hi, step) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setDecimals(2)
        w.setValue(float(value))
        w.valueChanged.connect(self._touch)
        form.addRow(label, w)
        return w

    def _int(self, form, label, value, lo, hi) -> QSpinBox:
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(int(value))
        w.valueChanged.connect(self._touch)
        form.addRow(label, w)
        return w

    # ---------------------------------------------------------------- 视图面板
    def _build_view_panel(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(8, 8, 8, 8)

        top = QHBoxLayout()
        top.addWidget(QLabel("视图"))
        self.view_combo = QComboBox()
        for key, label in VIEWS:
            self.view_combo.addItem(label, key)
        self.view_combo.currentIndexChanged.connect(lambda *_: self._draw())
        top.addWidget(self.view_combo)
        top.addStretch()
        self.lbl_info = QLabel("—")
        self.lbl_info.setStyleSheet("color:#333;")
        top.addWidget(self.lbl_info)
        v.addLayout(top)

        self.view = ImageView()
        v.addWidget(self.view, 1)
        return panel

    # ---------------------------------------------------------------- 交互
    def _on_image(self):
        if self.session.has_image():
            self.view.set_array(self.session.rgb)
            self._recompute()

    def _on_method_change(self, *_):
        kind, _m = self.method_combo.currentData()
        self.gbox_gully.setEnabled(kind == "gully")
        self.gbox_slope.setEnabled(kind == "slope")
        self._recompute()

    def _touch(self, *_):
        self._debounce.start()

    def _reset_defaults(self):
        d = segment_gully.GULLY_DEFAULTS
        self.sp_y0.setValue(d["y_range"][0]); self.sp_y1.setValue(d["y_range"][1])
        self.sp_xl0.setValue(d["x_left"][0]); self.sp_xl1.setValue(d["x_left"][1])
        self.sp_xr0.setValue(d["x_right"][0]); self.sp_xr1.setValue(d["x_right"][1])
        self.sp_jump.setValue(d["max_jump"]); self.sp_fill.setValue(d["fill_window"])
        self.sp_es.setValue(d["extend_start"]); self.sp_ee.setValue(d["extend_end"])
        self.sp_ex.setValue(d["extend_expand"])
        self.sp_k.setValue(5); self.sp_spatial.setValue(25.0); self.sp_cluster.setValue(-1)
        self._recompute()

    def params(self) -> dict:
        """当前界面上的参数(导出 JSON / 复现用)"""
        kind, _m = self.method_combo.currentData()
        if kind == "gully":
            return {"y_range": (self.sp_y0.value(), self.sp_y1.value()),
                    "x_left": (self.sp_xl0.value(), self.sp_xl1.value()),
                    "x_right": (self.sp_xr0.value(), self.sp_xr1.value()),
                    "max_jump": self.sp_jump.value(),
                    "fill_window": self.sp_fill.value(),
                    "extend_start": self.sp_es.value(),
                    "extend_end": self.sp_ee.value(),
                    "extend_expand": self.sp_ex.value()}
        cluster = self.sp_cluster.value()
        return {"k": self.sp_k.value(), "spatial": self.sp_spatial.value(),
                "cluster": None if cluster < 0 else cluster}

    def _recompute(self):
        if not self.session.has_image() or self.runner.busy():
            return
        kind, method = self.method_combo.currentData()
        self.runner.run(compute, self.session.bgr, kind, method, self.params(),
                        name=f"分割计算({method or 'kmeans'})",
                        on_done=self._apply, on_error=self._error)

    def _apply(self, result):
        self._result = result
        kind, method, res = result
        if kind == "gully":
            frac = (res["mask"] > 0).mean() * 100
            self.lbl_info.setText(f"{method} · 沟壑 {frac:.1f}% · 堆积体 {(res['debris']>0).mean()*100:.1f}%")
        else:
            frac = (res["mask"] > 0).mean() * 100
            self.lbl_info.setText(f"K-means k={len(res['stats'])} · 主坡体簇 #{res['cluster']} · {frac:.1f}%")
        self._draw()

    def _error(self, msg: str):
        QMessageBox.critical(self, "计算失败", msg)

    def _draw(self):
        if not self._result or not self.session.has_image():
            return
        kind, method, res = self._result
        view = self.view_combo.currentData()
        if kind == "slope" and view in ("boundary", "grad"):
            view = "overlay"
        yr = self.params().get("y_range") if kind == "gully" else None
        self._raw = render(self.session.bgr, kind, method, res, view, yr)
        self.view.set_array(self._raw, keep_view=True)

    def hit_mask(self, x: int, y: int) -> bool:
        """(x, y) 是否落在当前掩模内(供主窗口在状态栏提示)"""
        if not self._result:
            return False
        _kind, _m, res = self._result
        mask = res["mask"]
        h, w = mask.shape
        return 0 <= x < w and 0 <= y < h and mask[y, x] > 0

    # ---------------------------------------------------------------- 导出
    def _export(self, what: str):
        if not self._result:
            QMessageBox.information(self, "无结果", "请先载入图片并完成一次分割")
            return
        kind, method, res = self._result
        stem = f"{kind}_{method or 'kmeans'}"
        default = str(config.ROOT / f"{stem}_{what}.png")
        if what == "json":
            path, _ = QFileDialog.getSaveFileName(self, "导出参数", str(config.ROOT / f"{stem}_params.json"),
                                                  "JSON (*.json)")
            if not path:
                return
            data = {"image": self.session.image_path, "method": method, "kind": kind,
                    "params": self.params(), "exported_at": datetime.now().isoformat(timespec="seconds")}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.lbl_pick.setText(f"参数已导出:{path}")
            return

        path, _ = QFileDialog.getSaveFileName(self, "导出图片", default, "PNG (*.png)")
        if not path:
            return
        if what == "mask":
            img = res["mask"]
        else:
            yr = self.params().get("y_range") if kind == "gully" else None
            img = cv2.cvtColor(render(self.session.bgr, kind, method, res, "overlay", yr),
                               cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img)
        self.lbl_pick.setText(f"已导出:{path}")
