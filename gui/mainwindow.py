"""主窗口:工具栏(打开图片 / 设备号)+ 标签页 + 状态栏(生效参数常驻显示)。"""

from PySide6.QtCore import QUrl
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (QComboBox, QFileDialog, QLabel, QMainWindow, QMessageBox,
                               QTabWidget, QToolBar)

import config
import core
from gui.session import Session
from gui.workers import TaskRunner
from gui import style

IMAGE_FILTER = "图片 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff);;所有文件 (*)"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("滑坡监测工作台")
        self.resize(1480, 920)

        self.session = Session()
        self.runner = TaskRunner(self)

        self._build_toolbar()
        self._build_tabs()
        self._build_statusbar()

        self.runner.started.connect(lambda name: self.statusBar().showMessage(f"{name}…"))
        self.runner.finished.connect(lambda: self.statusBar().showMessage("就绪"))
        self.session.device_changed.connect(self._refresh_params)
        self.session.image_changed.connect(self._refresh_params)

    # ---------------------------------------------------------------- 工具栏
    def _build_toolbar(self):
        tb = QToolBar("主工具栏")
        tb.setMovable(False)
        self.addToolBar(tb)

        act_open = QAction("打开图片…", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.open_image)
        tb.addAction(act_open)

        act_dir = QAction("打开项目目录", self)
        act_dir.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(config.ROOT))))
        tb.addAction(act_dir)

        tb.addSeparator()
        tb.addWidget(QLabel("  设备号 "))
        self.device_combo = QComboBox()
        self.device_combo.setEditable(True)                 # 允许直接输入新设备号
        self.device_combo.addItems(config.device_ids())
        self.device_combo.setCurrentText(self.session.device)
        self.device_combo.currentTextChanged.connect(self._on_device_combo)
        tb.addWidget(self.device_combo)

        tb.addSeparator()
        act_exp = QAction("导出配置…", self)
        act_exp.setToolTip("把当前分割/分析参数存成 profiles/*.json,给 runtime.py / monitor.py 用")
        act_exp.triggered.connect(self.export_profile)
        tb.addAction(act_exp)
        act_imp = QAction("载入配置…", self)
        act_imp.triggered.connect(self.import_profile)
        tb.addAction(act_imp)

        tb.addSeparator()
        act_about = QAction("关于", self)
        act_about.triggered.connect(self.about)
        tb.addAction(act_about)

    # ---------------------------------------------------------------- 标签页
    def _build_tabs(self):
        from gui.tabs.capture import CaptureTab
        from gui.tabs.data import DataTab
        from gui.tabs.depth import DepthTab
        from gui.tabs.features import FeaturesTab
        from gui.tabs.segment import SegmentTab

        self.tabs = QTabWidget()
        self.segment_tab = SegmentTab(self.session, self.runner)
        self.segment_tab.view.coords.connect(self._on_coords)
        self.depth_tab = DepthTab(self.session, self.runner)
        self.features_tab = FeaturesTab(self.session, self.runner)
        self.capture_tab = CaptureTab(self.session, self.runner)
        self.data_tab = DataTab(self.session, self.runner)
        self.tabs.addTab(self.segment_tab, "① 分割")
        self.tabs.addTab(self.depth_tab, "② 深度")
        self.tabs.addTab(self.features_tab, "③ 特征")
        self.tabs.addTab(self.capture_tab, "④ 抓图")
        self.tabs.addTab(self.data_tab, "⑤ 数据")
        self.setCentralWidget(self.tabs)

    # ---------------------------------------------------------------- 状态栏
    def _build_statusbar(self):
        sb = self.statusBar()
        self.lbl_coords = QLabel("鼠标 —")
        self.lbl_device = QLabel()
        self.lbl_params = QLabel()
        self.lbl_engine = QLabel()
        for w in (self.lbl_coords, self.lbl_device, self.lbl_params, self.lbl_engine):
            sb.addPermanentWidget(w)
        self._refresh_params()
        sb.showMessage("就绪")

    def _on_coords(self, x: int, y: int):
        extra = " · 掩模内" if self.segment_tab.hit_mask(x, y) else ""
        self.lbl_coords.setText(f"鼠标 ({x}, {y}){extra}")

    def _refresh_params(self):
        """状态栏常驻显示:当前设备号 + 生效参数 + 模型来源/推理设备"""
        import features as F
        dev = self.session.device
        p = F.resolve_params(dev)
        roi = f"自动({p['roi_target']})" if p["roi_auto"] else (str(p["roi"]) if p["roi"] else "全图")
        self.lbl_device.setText(f"设备 {dev}")
        self.lbl_params.setText(
            f"参数 fov {p['fov']}° · 最远 {p['max_depth']} m · conf {p['conf']} · ROI {roi}")
        self.lbl_engine.setText(f"模型 {core._DA2_DIR} · 推理 {core.DEVICE}")
        self.lbl_params.setStyleSheet("")
        self.lbl_engine.setStyleSheet(style.hint_style(self))

    # ---------------------------------------------------------------- 动作
    def _on_device_combo(self, text: str):
        self.session.set_device(text.strip())
        self._refresh_params()

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开图片", str(config.ROOT), IMAGE_FILTER)
        if not path:
            return
        if not self.session.load_image(path):
            QMessageBox.warning(self, "打开失败", f"读不到图片:\n{path}")
            return
        self.statusBar().showMessage(f"已载入 {path}", 4000)

    # ---------------------------------------------------------------- 配置文件
    def export_profile(self):
        """把当前界面上的分割/分析参数导出成配置文件(交给运行时用)"""
        import tuning as pm
        prof = pm.build(self.session.device)
        seg = self.segment_tab.segmentation_params()
        if seg:
            prof["segmentation"].update(seg)
        prof["analysis"].update(self.features_tab.analysis_params())
        default = str(config.ROOT / "profiles" / f"{self.session.device}.json")
        path, _ = QFileDialog.getSaveFileName(self, "导出配置文件", default, "JSON (*.json)")
        if not path:
            return
        try:
            p = pm.save(path, prof)
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))
            return
        QMessageBox.information(
            self, "已导出",
            f"{p}\n\n设备 {self.session.device}\n分割方法: {prof['segmentation'].get('method')}\n\n"
            f"运行时用法:\n  .venv/bin/python runtime.py            # 监视台,在下拉里选这份配置\n"
            f"  .venv/bin/python monitor.py --device {self.session.device} --profile {path}")

    def import_profile(self):
        import tuning as pm
        path, _ = QFileDialog.getOpenFileName(self, "载入配置文件",
                                              str(config.ROOT / "profiles"), "JSON (*.json)")
        if not path:
            return
        try:
            prof = pm.load(path)
        except Exception as e:
            QMessageBox.critical(self, "载入失败", str(e))
            return
        self.segment_tab.apply_profile(prof.get("segmentation") or {})
        self.features_tab.apply_analysis(prof.get("analysis") or {})
        QMessageBox.information(self, "已载入",
                                f"{path}\nversion={prof.get('version')}\n"
                                f"分割方法: {(prof.get('segmentation') or {}).get('method')}")

    def about(self):
        QMessageBox.information(
            self, "关于",
            "滑坡监测工作台(PySide6)\n\n"
            "分割 / 深度 / 特征 / 抓图 / 数据 统一入口。\n"
            "界面只做编排,计算全部复用命令行同一套模块,结果一致。\n\n"
            f"项目目录: {config.ROOT}\n"
            f"推理设备: {core.DEVICE}\n"
            f"数据库: {config.db_path()}")
