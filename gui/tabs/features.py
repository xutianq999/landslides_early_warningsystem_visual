"""③ 特征提取 + 入库:复用 features.features_from_image,参数取当前设备配置。"""

from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QMessageBox, QPushButton, QSplitter,
                               QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget)

import config
from gui.imageview import ImageView
from gui import style

ROI_TARGETS = [("gully", "沟壑"), ("debris", "堆积体"), ("both", "两者")]
ROI_MASK_KEYS = {"gully": "mask", "debris": "debris", "both": "mask_all"}


def extract(pil, device, classes, conf, max_depth, fov, roi_auto, roi_target, prev, to_db):
    """后台执行:特征提取(可选 ROI)→ 可选入库。返回 (row, overlay_rgb|None, db_msg)"""
    import db as dbm
    import features as F
    import segment_gully

    roi_mask, extra_masks, overlay = None, None, None
    if roi_auto:
        import numpy as _np
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        res = segment_gully.detect_gully(bgr)
        roi_mask = res[ROI_MASK_KEYS[roi_target]] > 0
        extra_masks = {"gully": res["mask"], "debris": res["debris"]}
        vis = bgr.copy()
        for m, color in ((res["mask"], (0, 255, 0)), (res["debris"], (0, 165, 255))):
            mm = m > 0
            if mm.any():
                tint = _np.zeros_like(bgr)
                tint[:] = color
                vis[mm] = (0.5 * bgr[mm] + 0.5 * tint[mm]).astype(_np.uint8)
        sel = res[ROI_MASK_KEYS[roi_target]] > 0
        edge = cv2.Canny((sel * 255).astype(np.uint8), 50, 150)
        vis[edge > 0] = (255, 0, 0)
        overlay = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    cls = [c.strip() for c in classes.split(",") if c.strip()] or None
    row, cur = F.features_from_image(pil, classes=cls, conf=conf, max_depth=max_depth, fov=fov,
                                     prev=prev, roi_mask=roi_mask, extra_masks=extra_masks)
    row = {"time": datetime.now().isoformat(timespec="seconds"), **row}

    db_msg = ""
    if to_db:
        params = {"classes": classes, "conf": conf, "max_depth": max_depth, "fov": fov,
                  "roi_auto": bool(roi_auto), "roi_target": roi_target if roi_auto else None}
        img_dir = config.images_dir() / device
        img_dir.mkdir(parents=True, exist_ok=True)
        stamp = row["time"].replace(":", "").replace("-", "")
        ipath = img_dir / f"{stamp}.jpg"
        pil.convert("RGB").save(ipath, quality=90)
        conn = dbm.connect()
        try:
            dbm.upsert_frames(conn, [{**row, "image_path": str(ipath.resolve()), "params": params}],
                              device_id=device)
            db_msg = f"已入库 → {dbm.resolve_db_path()}"
        finally:
            conn.close()
    return row, overlay, db_msg, cur


class FeaturesTab(QWidget):
    def __init__(self, session, runner, parent=None):
        super().__init__(parent)
        self.session = session
        self.runner = runner
        self._prev = None

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_panel())
        split.addWidget(self._build_result())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([400, 1020])
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)

        self.session.device_changed.connect(lambda *_: self._sync_device())

    def _build_panel(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(8, 8, 8, 8)

        self.gbox = QGroupBox("提取参数(默认取当前设备配置,可临时改)")
        f = QFormLayout(self.gbox)
        self.ed_classes = QTextEdit()
        self.ed_classes.setFixedHeight(58)
        f.addRow("分割类别", self.ed_classes)
        self.sp_conf = QDoubleSpinBox(); self.sp_conf.setRange(0.01, 0.99); self.sp_conf.setSingleStep(0.05)
        f.addRow("置信度", self.sp_conf)
        self.sp_max = QDoubleSpinBox(); self.sp_max.setRange(1, 200); self.sp_max.setSuffix(" m")
        f.addRow("最远距离", self.sp_max)
        self.sp_fov = QDoubleSpinBox(); self.sp_fov.setRange(10, 180); self.sp_fov.setSuffix(" °")
        f.addRow("FOV", self.sp_fov)
        v.addWidget(self.gbox)

        self.chk_roi = QCheckBox("只在自动检测到的区域算几何/深度/时序")
        self.cb_target = QComboBox()
        for key, label in ROI_TARGETS:
            self.cb_target.addItem(label, key)
        self.cb_target.setEnabled(False)
        self.chk_roi.toggled.connect(self.cb_target.setEnabled)
        v.addWidget(self.chk_roi)
        v.addWidget(self.cb_target)

        self.chk_db = QCheckBox("同时写入数据库(带设备号与参数)")
        self.chk_db.setChecked(True)
        v.addWidget(self.chk_db)

        btn = QPushButton("提取特征")
        btn.clicked.connect(self._run)
        v.addWidget(btn)
        self.lbl = QLabel("—")
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet(style.hint_style(self))
        v.addWidget(self.lbl)
        v.addStretch()

        self._sync_device()
        return panel

    def _build_result(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.addWidget(QLabel("区域检测(绿=沟壑,橙=堆积体,蓝线=选中区域边缘)"))
        self.view = ImageView()
        self.view.setMinimumHeight(300)
        v.addWidget(self.view, 2)
        v.addWidget(QLabel("本次特征"))
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["特征", "值"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        v.addWidget(self.table, 3)
        return w

    def analysis_params(self) -> dict:
        """当前分析参数(供导出配置文件)"""
        return {"classes": self.ed_classes.toPlainText().strip(), "conf": self.sp_conf.value(),
                "max_depth": self.sp_max.value(), "fov": self.sp_fov.value(),
                "roi": None, "roi_auto": self.chk_roi.isChecked(),
                "roi_target": self.cb_target.currentData()}

    def apply_analysis(self, a: dict) -> None:
        """把配置文件里的分析参数套到控件上"""
        if not a:
            return
        if a.get("classes"):
            self.ed_classes.setPlainText(str(a["classes"]))
        for key, w in (("conf", self.sp_conf), ("max_depth", self.sp_max), ("fov", self.sp_fov)):
            if key in a and a[key] is not None:
                w.setValue(float(a[key]))
        if "roi_auto" in a:
            self.chk_roi.setChecked(bool(a["roi_auto"]))
        if a.get("roi_target"):
            idx = self.cb_target.findData(a["roi_target"])
            if idx >= 0:
                self.cb_target.setCurrentIndex(idx)

    def _sync_device(self):
        import features as F
        p = F.resolve_params(self.session.device)
        self.ed_classes.setPlainText(p["classes"])
        self.sp_conf.setValue(float(p["conf"]))
        self.sp_max.setValue(float(p["max_depth"]))
        self.sp_fov.setValue(float(p["fov"]))
        self.chk_roi.setChecked(bool(p["roi_auto"]))
        idx = self.cb_target.findData(p["roi_target"])
        if idx >= 0:
            self.cb_target.setCurrentIndex(idx)

    def _run(self):
        if not self.session.has_image():
            QMessageBox.information(self, "无图片", "请先打开一张图片")
            return
        pil = Image.fromarray(self.session.rgb)
        args = (pil, self.session.device, self.ed_classes.toPlainText().strip(),
                self.sp_conf.value(), self.sp_max.value(), self.sp_fov.value(),
                self.chk_roi.isChecked(), self.cb_target.currentData(),
                self._prev, self.chk_db.isChecked())
        self.runner.run(extract, *args, name="特征提取", on_done=self._apply,
                        on_error=lambda m: QMessageBox.critical(self, "提取失败", m))

    def _apply(self, out):
        row, overlay, db_msg, cur = out
        self._prev = cur                     # 连续提取时自动与上一张比对算变化量
        if overlay is not None:
            self.view.set_array(overlay)
        elif self.session.has_image():
            self.view.set_array(self.session.rgb)
        self.table.setRowCount(len(row))
        for i, (k, val) in enumerate(row.items()):
            self.table.setItem(i, 0, QTableWidgetItem(str(k)))
            if isinstance(val, float):
                txt = "—" if val != val else f"{val:.4f}"
            else:
                txt = str(val)
            self.table.setItem(i, 1, QTableWidgetItem(txt))
        self.table.resizeColumnsToContents()
        self.lbl.setText(f"共 {len(row)} 项" + (f" · {db_msg}" if db_msg else " · 未入库"))
