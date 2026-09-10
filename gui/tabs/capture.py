"""④ 抓图:RTSP 预览、手动抓帧入库、定时抓图。复用 capture 模块,不重写抓帧逻辑。"""

import logging

import cv2
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPushButton, QSplitter,
                               QTextEdit, QVBoxLayout, QWidget)

import config
from gui.imageview import ImageView


class _LogBridge(QObject, logging.Handler):
    """把 capture 模块的日志转发到界面(抓帧失败/重试都能看见)"""

    msg = Signal(str)

    def __init__(self):
        QObject.__init__(self)
        logging.Handler.__init__(self)
        self.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", "%H:%M:%S"))

    def emit(self, record):
        try:
            self.msg.emit(self.format(record))
        except Exception:
            pass


class CaptureTab(QWidget):
    def __init__(self, session, runner, parent=None):
        super().__init__(parent)
        self.session = session
        self.runner = runner

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_panel())
        split.addWidget(self._build_view())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([420, 1000])
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)

        self._bridge = _LogBridge()
        self._bridge.msg.connect(self._log)
        cap_log = logging.getLogger("capture")
        cap_log.addHandler(self._bridge)
        cap_log.setLevel(logging.INFO)      # 否则被 root 的 WARNING 级别挡掉,界面看不到抓帧日志

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        self.session.device_changed.connect(lambda *_: self._sync_device())
        self._sync_device()

    def _build_panel(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(8, 8, 8, 8)

        self.gbox = QGroupBox("设备与流地址(默认取当前设备配置,可临时改)")
        f = QFormLayout(self.gbox)
        self.ed_url = QLineEdit()
        self.ed_url.setPlaceholderText("rtsp://user:pw@ip:554/Streaming/Channels/102")
        f.addRow("RTSP 地址", self.ed_url)
        self.cb_transport = QComboBox()
        self.cb_transport.addItems(["tcp", "udp"])
        f.addRow("传输方式", self.cb_transport)
        v.addWidget(self.gbox)

        row = QHBoxLayout()
        btn_prev = QPushButton("预览一帧")
        btn_prev.setToolTip("连一次流抓一帧显示,不入库(无相机时可填本地图片/视频路径)")
        btn_prev.clicked.connect(self._preview)
        btn_grab = QPushButton("抓一帧并入库")
        btn_grab.setToolTip("走完整链路:抓帧 → 落盘 → 特征 → 入库 → 刷新报警")
        btn_grab.clicked.connect(self._grab_once)
        row.addWidget(btn_prev)
        row.addWidget(btn_grab)
        v.addLayout(row)

        g2 = QGroupBox("定时抓图")
        f2 = QFormLayout(g2)
        self.sp_interval = QDoubleSpinBox()
        self.sp_interval.setRange(0.1, 240)
        self.sp_interval.setDecimals(1)
        self.sp_interval.setSuffix(" 分钟")
        f2.addRow("采样间隔", self.sp_interval)
        self.btn_timer = QPushButton("开始定时")
        self.btn_timer.setCheckable(True)
        self.btn_timer.toggled.connect(self._toggle_timer)
        f2.addRow(self.btn_timer)
        v.addWidget(g2)

        self.lbl = QLabel("—")
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet("color:#333;")
        v.addWidget(self.lbl)
        v.addStretch()
        return panel

    def _build_view(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.addWidget(QLabel("预览"))
        self.view = ImageView()
        v.addWidget(self.view, 3)
        v.addWidget(QLabel("日志"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("font-family: Menlo, monospace; font-size: 11px;")
        v.addWidget(self.log, 2)
        return w

    # ---------------------------------------------------------------- 逻辑
    def _sync_device(self):
        meta = config.device_meta(self.session.device)
        self.ed_url.setText(meta.get("rtsp_url") or "")
        self.cb_transport.setCurrentText(config.for_device(self.session.device)
                                         .get("rtsp_transport", "tcp"))
        self.sp_interval.setValue(float(config.for_device(self.session.device)
                                        .get("interval_min", 5)))

    def _log(self, line: str):
        self.log.append(line)

    def _source(self) -> str | None:
        return self.ed_url.text().strip() or None

    def _preview(self):
        import capture
        src = self._source()
        if not src:
            QMessageBox.warning(self, "缺少地址", "该设备没有配置 rtsp_url,请先填写")
            return
        self.lbl.setText("正在抓帧…")

        def job():
            return capture.grab_with_retry(src, self.cb_transport.currentText(), attempts=2)

        def done(frame):
            if frame is None:
                self.lbl.setText("抓帧失败(看日志)")
                return
            self.view.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.lbl.setText(f"预览成功 {frame.shape[1]}×{frame.shape[0]}(未入库)")

        self.runner.run(job, name="抓帧预览", on_done=done,
                        on_error=lambda m: QMessageBox.critical(self, "抓帧失败", m))

    def _grab_once(self):
        import capture
        src = self._source()
        device = self.session.device

        def job():
            return capture.sample_once(device, source=src, do_alarm=True)

        def done(ok):
            self.lbl.setText("已抓帧并入库(报警已刷新)" if ok else "抓帧/处理失败,见日志")

        self.runner.run(job, name="抓帧入库", on_done=done,
                        on_error=lambda m: QMessageBox.critical(self, "抓帧失败", m))

    def _tick(self):
        if self.runner.busy():
            self._log("上一轮还在处理,本轮跳过")
            return
        self._grab_once()

    def _toggle_timer(self, on: bool):
        if on:
            ms = int(self.sp_interval.value() * 60_000)
            self._timer.start(ms)
            self.btn_timer.setText("停止定时")
            self._log(f"定时已启动,每 {self.sp_interval.value()} 分钟抓一次")
        else:
            self._timer.stop()
            self.btn_timer.setText("开始定时")
            self._log("定时已停止")
