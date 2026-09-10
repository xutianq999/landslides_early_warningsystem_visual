"""⑤ 数据:数据库概况、最近帧、报警等级,以及一键启停 API 服务。"""

import sys

from PySide6.QtCore import QProcess, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton,
                               QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget)

import config
import db as dbm

FRAME_COLS = ["captured_at", "device_id", "diff_frac", "slope_mean", "plane_rms",
              "disp_p50", "img_blur"]


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


class DataTab(QWidget):
    def __init__(self, session, runner, parent=None):
        super().__init__(parent)
        self.session = session
        self.runner = runner
        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(str(config.ROOT))
        self.proc.readyReadStandardOutput.connect(self._drain)
        self.proc.readyReadStandardError.connect(self._drain)
        self.proc.stateChanged.connect(self._on_proc_state)

        v = QVBoxLayout(self)
        v.setContentsMargins(10, 10, 10, 10)

        top = QHBoxLayout()
        btn = QPushButton("刷新")
        btn.clicked.connect(self.refresh)
        top.addWidget(btn)
        self.lbl_info = QLabel("—")
        self.lbl_info.setStyleSheet("color:#333;")
        top.addWidget(self.lbl_info)
        top.addStretch()
        v.addLayout(top)

        v.addWidget(QLabel("最近帧(最多 50 条)"))
        self.tbl_frames = self._table(FRAME_COLS)
        v.addWidget(self.tbl_frames, 3)

        v.addWidget(QLabel("各设备当前报警等级"))
        self.tbl_alarms = self._table(["device_id", "captured_at", "level", "level_name",
                                       "top_signal", "score", "camera_alarm"])
        v.addWidget(self.tbl_alarms, 2)

        gbox = QGroupBox("数据 API(供平台拉取)")
        g = QVBoxLayout(gbox)
        row = QHBoxLayout()
        self.btn_api = QPushButton("启动 API 服务")
        self.btn_api.clicked.connect(self._toggle_api)
        row.addWidget(self.btn_api)
        btn_docs = QPushButton("打开接口文档")
        btn_docs.clicked.connect(self._open_docs)
        row.addWidget(btn_docs)
        self.lbl_api = QLabel()
        row.addWidget(self.lbl_api)
        row.addStretch()
        g.addLayout(row)
        self.api_log = QTextEdit()
        self.api_log.setReadOnly(True)
        self.api_log.setFixedHeight(90)
        self.api_log.setStyleSheet("font-family: Menlo, monospace; font-size: 11px;")
        g.addWidget(self.api_log)
        v.addWidget(gbox)

        self.refresh()

    @staticmethod
    def _table(cols) -> QTableWidget:
        t = QTableWidget(0, len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.horizontalHeader().setStretchLastSection(True)
        t.verticalHeader().setVisible(False)
        return t

    @staticmethod
    def _fill(table: QTableWidget, rows: list[dict], cols: list[str]):
        table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                table.setItem(i, j, QTableWidgetItem(_fmt(r.get(c))))
        table.resizeColumnsToContents()

    # ---------------------------------------------------------------- 数据
    def refresh(self):
        try:
            conn = dbm.connect()
            try:
                s = dbm.stats(conn)
                frames = dbm.query_frames(conn, limit=50, order="desc")
                alarms = dbm.latest_alarms(conn)
            finally:
                conn.close()
        except Exception as e:
            self.lbl_info.setText(f"读取失败: {e}")
            return
        self.lbl_info.setText(
            f"库 {s['db_path']} · schema v{s['schema_version']} · 帧 {s['frames']} · "
            f"报警 {s['alarms']}(红/橙/黄 {s['alerts']}) · {s['first']} ~ {s['last']}")
        self._fill(self.tbl_frames, frames, FRAME_COLS)
        self._fill(self.tbl_alarms, alarms, ["device_id", "captured_at", "level", "level_name",
                                             "top_signal", "score", "camera_alarm"])

    # ---------------------------------------------------------------- API
    def _on_proc_state(self, state):
        running = state != QProcess.NotRunning
        self.btn_api.setText("停止 API 服务" if running else "启动 API 服务")
        host, port = config.CONFIG["api_host"], config.CONFIG["api_port"]
        self.lbl_api.setText(f"http://{host}:{port}/docs" if running else "未运行")

    def _toggle_api(self):
        if self.proc.state() != QProcess.NotRunning:
            self.proc.terminate()
            if not self.proc.waitForFinished(3000):
                self.proc.kill()
            return
        host, port = config.CONFIG["api_host"], config.CONFIG["api_port"]
        self.api_log.append(f"启动 api.py --host {host} --port {port}")
        self.proc.start(sys.executable, ["api.py", "--host", host, "--port", str(port)])

    def _drain(self):
        out = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "ignore")
        err = bytes(self.proc.readAllStandardError()).decode("utf-8", "ignore")
        for line in (out + err).splitlines():
            if line.strip():
                self.api_log.append(line)

    def _open_docs(self):
        host, port = config.CONFIG["api_host"], config.CONFIG["api_port"]
        QDesktopServices.openUrl(QUrl(f"http://{host}:{port}/docs"))
