"""运行时监视台:启停两套通道,看实时指标与报警。

与 studio.py(工作台)的分工:
  - studio.py   离线调参:对一张图试各种分割方法/参数,导出 profiles/*.json;
  - runtime.py  在线监护:读同一份配置,把两套通道跑起来,盯住变化率/等级/日志。

界面不含任何算法实现:启停与判定全部由 monitor.DeviceRuntime 完成(与命令行
`monitor.py` 是同一个类),这里只做编排与显示,所以两边的数一定一致。
"""

import collections
import logging
import os
import threading
import time

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton,
                               QTableWidget, QTableWidgetItem, QToolBar, QVBoxLayout, QWidget)

import config
import core
from gui import style

LOG = logging.getLogger("runtime")

COLS = ("启动", "设备号", "名称", "实时", "分析", "状态", "变化率", "速率/s",
        "加速度/s²", "有效像素", "基线", "帧源", "最近分析", "说明")

# 各列内容的最长样子(用于按字体量出列宽,字号改了也不会截断)
COLS_SAMPLE = ("启动", "HIK-01", "1号坡面", "实时", "分析", "橙色-预警",
               "100.00%", "-0.1750", "-0.1750", "100.00%", "99.9/10min",
               "已连 9.9fps 错9", "999s前 d=9.99 红", "说明")
# 超长列给个上限:内容再长也靠悬停提示看全,否则表格会被撑出横向滚动条、把「说明」挤没
COLS_MAX = {5: 110, 11: 170, 12: 170}

# 表格里用短名(全名在悬停提示里),省下来的宽度留给「说明」
LEVEL_SHORT = {"正常": "正常", "黄色-关注": "黄", "橙色-预警": "橙", "红色-紧急": "红"}


class _LogBridge(QObject, logging.Handler):
    """把帧源/通道/分析的日志转发到界面(重连、报警、异常都能看见)"""

    msg = Signal(str)

    def __init__(self):
        QObject.__init__(self)
        logging.Handler.__init__(self)
        self.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(name)s  %(message)s",
                                           "%H:%M:%S"))

    def emit(self, record):
        try:
            self.msg.emit(self.format(record))
        except Exception:
            pass


class RuntimeWorker(QThread):
    """后台线程:持有各点位的 DeviceRuntime,按节拍 step(),周期性把状态发给界面。

    所有启停命令都排队进 `_cmd`,由线程自己执行——停止要 join 帧源线程,放在界面
    线程里会把窗口卡住几秒。
    """

    snapshot = Signal(list)
    failed = Signal(str, str)          # (设备号, 原因)

    def __init__(self, db_path: str, parent=None):
        super().__init__(parent)
        self.db_path = db_path
        self.runtimes: dict = {}
        self._cmd = collections.deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()

    # ---- 供界面线程调用
    def start_devices(self, devices: list, opts: dict):
        with self._lock:
            self._cmd.append(("start", list(devices), dict(opts)))

    def stop_device(self, device: str):
        with self._lock:
            self._cmd.append(("stop_one", device))

    def stop_all(self):
        with self._lock:
            self._cmd.append(("stop_all",))

    def set_channels(self, device: str, realtime, analysis):
        with self._lock:
            self._cmd.append(("channels", device, realtime, analysis))

    def shutdown(self):
        self._stop.set()

    def running_devices(self) -> list:
        return list(self.runtimes)

    # ---- 线程主体
    def run(self):
        import monitor
        last_emit = 0.0
        while not self._stop.is_set():
            for cmd in self._drain():
                self._exec(cmd, monitor)
            for dev, r in list(self.runtimes.items()):
                try:
                    r.step()
                except Exception as e:                 # 单台异常不该拖垮其它点位
                    LOG.exception("设备 %s step 异常: %s", dev, e)
            now = time.time()
            if now - last_emit >= 0.5:
                last_emit = now
                self._emit()
            self._stop.wait(0.3)
        for r in self.runtimes.values():
            try:
                r.stop()
            except Exception:
                pass
        self.runtimes.clear()
        self.snapshot.emit([])

    def _drain(self) -> list:
        with self._lock:
            out = list(self._cmd)
            self._cmd.clear()
        return out

    def _exec(self, cmd, monitor):
        kind = cmd[0]
        if kind == "start":
            _, devices, opts = cmd
            for dev in devices:
                if dev in self.runtimes:
                    continue
                try:
                    self.runtimes[dev] = monitor.make_runtime(dev, self.db_path, **opts)
                    self.runtimes[dev].start()
                except (SystemExit, Exception) as e:   # 缺 rtsp_url / 参数非法都只跳过这台
                    LOG.error("设备 %s 启动失败: %s", dev, e)
                    self.failed.emit(dev, str(e))
        elif kind == "stop_one":
            dev = cmd[1]
            r = self.runtimes.pop(dev, None)
            if r is not None:
                r.stop()
                LOG.info("设备 %s 已停止", dev)
        elif kind == "stop_all":
            for dev, r in list(self.runtimes.items()):
                r.stop()
                self.runtimes.pop(dev, None)
            LOG.info("全部点位已停止")
        elif kind == "channels":
            _, dev, rt, an = cmd
            r = self.runtimes.get(dev)
            if r is not None:
                r.set_channels(realtime=rt, analysis=an)

    def _emit(self):
        rows = []
        for r in self.runtimes.values():
            try:
                rows.append(r.status())
            except Exception as e:
                LOG.warning("状态读取失败: %s", e)
        self.snapshot.emit(rows)


class RuntimeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("滑坡监测运行时监视台")
        self.resize(1560, 900)

        self.worker: RuntimeWorker | None = None
        self._syncing = False                  # 防止"程序改勾选"再触发一次命令
        self._devices = config.device_ids()

        self._build_toolbar()
        self._build_central()
        self._build_statusbar()

        self._bridge = _LogBridge()
        self._bridge.msg.connect(self.append_log)
        for name in ("runtime", "monitor", "realtime", "framesource", "capture", "alarm"):
            lg = logging.getLogger(name)
            lg.addHandler(self._bridge)
            lg.setLevel(logging.INFO)

    # ---------------------------------------------------------------- 构建
    def _build_toolbar(self):
        tb = QToolBar("主工具栏")
        tb.setMovable(False)
        self.addToolBar(tb)

        self.act_start = self._act(tb, "启动选中", self.start_selected,
                                   "启动勾选了「启动」列的点位(设备多时按顺序建连)")
        self.act_stop = self._act(tb, "全部停止", self.stop_all)
        tb.addSeparator()
        self._act(tb, "打开报警证据目录", self.open_alarms_dir,
                  "报警时保存的当前帧 + 基线帧图片")
        self._act(tb, "打开数据库目录", lambda: self._open(config.db_path().parent))
        self._act(tb, "接口文档", lambda: QDesktopServices.openUrl(
            QUrl(f"http://{config.CONFIG.get('api_host')}:{config.CONFIG.get('api_port')}/docs")),
            "需要先启动 api.py")
        self.act_start.setEnabled(True)

    def _act(self, tb, text, slot, tip=""):
        a = QAction(text, self)
        if tip:
            a.setToolTip(tip)
        a.triggered.connect(slot)
        tb.addAction(a)
        return a

    def _build_central(self):
        c = QWidget()
        v = QVBoxLayout(c)
        v.setContentsMargins(8, 8, 8, 8)
        v.addWidget(self._build_config_box())
        self.banner = QLabel("未运行")
        self.banner.setStyleSheet(style.banner_style(0, widget=self))
        v.addWidget(self.banner)
        v.addWidget(self._build_table(), 3)
        v.addWidget(QLabel("运行日志"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)          # 长跑不让日志把内存吃光
        self.log.setFont(style.mono_font())
        self.log.setStyleSheet(style.log_style())
        v.addWidget(self.log, 2)
        self.setCentralWidget(c)

    def _build_config_box(self) -> QWidget:
        g = QGroupBox("运行配置(算法参数来自工作台导出的配置文件)")
        grid = QGridLayout(g)

        grid.addWidget(QLabel("算法配置文件"), 0, 0)
        self.cb_profile = QComboBox()
        self.cb_profile.setMinimumWidth(320)
        self._reload_profiles()
        grid.addWidget(self.cb_profile, 0, 1)
        btn_browse = QPushButton("浏览…")
        btn_browse.clicked.connect(self._browse_profile)
        grid.addWidget(btn_browse, 0, 2)

        self.sp_interval = QDoubleSpinBox()
        self.sp_interval.setRange(1, 3600)
        self.sp_interval.setValue(10)
        self.sp_interval.setSuffix(" 秒")
        grid.addWidget(QLabel("分析间隔"), 0, 3)
        grid.addWidget(self.sp_interval, 0, 4)

        self.sp_rain = QDoubleSpinBox()
        self.sp_rain.setRange(1, 3600)
        self.sp_rain.setValue(5)
        self.sp_rain.setSuffix(" 秒")
        grid.addWidget(QLabel("降雨期间隔"), 0, 5)
        grid.addWidget(self.sp_rain, 0, 6)

        self.sp_rain_trigger = QDoubleSpinBox()
        self.sp_rain_trigger.setRange(0, 100)
        self.sp_rain_trigger.setValue(2.0)
        self.sp_rain_trigger.setSuffix(" mm/h")
        grid.addWidget(QLabel("降雨触发"), 0, 7)
        grid.addWidget(self.sp_rain_trigger, 0, 8)

        self.cb_imgsz = QComboBox()
        self.cb_imgsz.addItems(["416", "480", "640"])
        self.cb_imgsz.setCurrentText("480")
        grid.addWidget(QLabel("分割尺寸"), 1, 0)
        grid.addWidget(self.cb_imgsz, 1, 1)

        self.chk_mask = QCheckBox("剔除动态物体(YOLOE)")
        self.chk_mask.setChecked(True)
        self.chk_mask.setToolTip("关掉省算力;无 YOLOE 权重时也能跑,但人车会被算进变化率")
        grid.addWidget(self.chk_mask, 1, 2, 1, 2)

        self.cb_device = QComboBox()
        self.cb_device.addItems(["mps", "cpu"])
        self.cb_device.setCurrentText(core.DEVICE)
        self.cb_device.setToolTip("推理设备;须在首次推理前选定,模型加载后再改不生效")
        grid.addWidget(QLabel("推理设备"), 1, 4)
        grid.addWidget(self.cb_device, 1, 5)

        self.ed_source = QLineEdit()
        self.ed_source.setPlaceholderText("留空 = 用 config.json 里的 rtsp_url;可填本地视频/图片路径做无相机试跑")
        grid.addWidget(QLabel("临时视频源"), 1, 6, 1, 3)
        grid.addWidget(self.ed_source, 1, 9)
        grid.setColumnStretch(9, 1)
        return g

    def _build_table(self) -> QWidget:
        self.table = QTableWidget(len(self._devices), len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(len(COLS) - 1, QHeaderView.Stretch)      # 说明列吃掉余宽
        for i, dev in enumerate(self._devices):
            meta = config.device_meta(dev)
            chk = QCheckBox()
            chk.setChecked(config.for_device(dev).get("enabled", True) is not False)
            chk.toggled.connect(lambda on, d=dev: self._on_row_toggle(d, on))
            holder = QWidget()
            hl = QHBoxLayout(holder)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setAlignment(Qt.AlignCenter)
            hl.addWidget(chk)
            self.table.setCellWidget(i, 0, holder)
            self._set(i, 1, dev)
            self._set(i, 2, meta.get("name") or "")
            for col in (3, 4):
                cb = QCheckBox()
                cb.setChecked(True)
                cb.setEnabled(False)                     # 未启动时不可点
                cb.toggled.connect(lambda on, d=dev, c=col: self._on_channel_toggle(d, c, on))
                holder2 = QWidget()
                hl2 = QHBoxLayout(holder2)
                hl2.setContentsMargins(0, 0, 0, 0)
                hl2.setAlignment(Qt.AlignCenter)
                hl2.addWidget(cb)
                self.table.setCellWidget(i, col, holder2)
            self._set(i, 5, "未启动")
            self._set(i, 11, "—")
        self._fit_table()
        return self.table

    def _fit_table(self):
        """按当前字体量出列宽与行高。

        字号改大后写死的像素宽度会把数值截成 "35..."(以前就踩过),所以宽度从
        字体度量算:取"表头/样例值"里更宽的那个,再留一点内边距。
        """
        fm = self.table.fontMetrics()
        for col in range(len(COLS) - 1):                 # 最后一列由 Stretch 吃掉余宽
            if self.table.cellWidget(0, col) is not None:      # 勾选框列:给固定小宽度
                self.table.setColumnWidth(col, max(44, fm.horizontalAdvance("启动") + 24))
                continue
            want = max(fm.horizontalAdvance(COLS[col]), fm.horizontalAdvance(COLS_SAMPLE[col]))
            self.table.setColumnWidth(col, min(want + 26, COLS_MAX.get(col, 10**4)))
        self.table.verticalHeader().setDefaultSectionSize(max(28, fm.height() + 12))

    def _build_statusbar(self):
        sb = self.statusBar()
        self.lbl_run = QLabel("运行 0 台")
        self.lbl_db = QLabel(f"库 {config.db_path()}")
        self.lbl_engine = QLabel(f"推理 {core.DEVICE}")
        for w in (self.lbl_run, self.lbl_db, self.lbl_engine):
            sb.addPermanentWidget(w)
        sb.showMessage("就绪:勾选点位后点「启动选中」")

    # ---------------------------------------------------------------- 表格工具
    def _set(self, row: int, col: int, text, color: QColor | None = None, tip: str | None = None):
        item = self.table.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            self.table.setItem(row, col, item)
        item.setText(str(text))
        if color is not None:
            item.setForeground(color)
        else:
            item.setForeground(style.text_color(self))
        item.setToolTip(tip if tip is not None else str(text))

    def _row_of(self, device: str) -> int:
        try:
            return self._devices.index(device)
        except ValueError:
            return -1

    def _cell_checkbox(self, row: int, col: int) -> QCheckBox | None:
        w = self.table.cellWidget(row, col)
        return w.findChild(QCheckBox) if w else None

    # ---------------------------------------------------------------- 动作
    def _reload_profiles(self):
        self.cb_profile.clear()
        self.cb_profile.addItem("(内置默认)", "")
        d = config.ROOT / "profiles"
        if d.is_dir():
            for p in sorted(d.glob("*.json")):
                self.cb_profile.addItem(p.name, str(p))

    def _browse_profile(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择算法配置文件",
                                              str(config.ROOT / "profiles"), "JSON (*.json)")
        if not path:
            return
        self.cb_profile.addItem(os.path.basename(path), path)
        self.cb_profile.setCurrentIndex(self.cb_profile.count() - 1)

    def _options(self) -> dict:
        """组装 make_runtime 的参数;配置文件解析失败就直接报错,不静默用默认值"""
        opts = {"interval": self.sp_interval.value(), "interval_rain": self.sp_rain.value(),
                "rain_trigger": self.sp_rain_trigger.value(),
                "imgsz": int(self.cb_imgsz.currentText()),
                "use_mask": self.chk_mask.isChecked(), "pick_sharpest": True}
        path = self.cb_profile.currentData()
        if path:
            import tuning as pm
            prof = pm.load(path)
            errs = pm.validate(prof)
            if errs:
                raise ValueError("配置文件校验失败:\n  - " + "\n  - ".join(errs))
            opts["profile"] = prof
        src = self.ed_source.text().strip()
        if src:
            if not os.path.exists(src):
                raise FileNotFoundError(f"临时视频源不存在: {src}")
            opts["source"] = src
        return opts

    def start_selected(self):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "已在运行", "先「全部停止」再改配置重新启动")
            return
        chosen = [d for d in self._devices
                  if (cb := self._cell_checkbox(self._row_of(d), 0)) and cb.isChecked()]
        if not chosen:
            QMessageBox.warning(self, "没有点位", "请在「启动」列勾选至少一个点位")
            return
        try:
            opts = self._options()
        except Exception as e:
            QMessageBox.critical(self, "配置有误", str(e))
            return

        core.DEVICE = self.cb_device.currentText()       # 必须在模型首次加载前设好
        self.lbl_engine.setText(f"推理 {core.DEVICE}")

        import db as dbm
        conn = dbm.connect(str(config.db_path()))
        try:
            dbm.init_db(conn)                            # 先建表:API 立刻可查
        finally:
            conn.close()

        self.worker = RuntimeWorker(str(config.db_path()), self)
        self.worker.snapshot.connect(self.on_snapshot)
        self.worker.failed.connect(self.on_failed)
        self.worker.start_devices(chosen, opts)
        self.worker.start()
        for d in chosen:
            r = self._row_of(d)
            for col in (3, 4):
                cb = self._cell_checkbox(r, col)
                if cb:
                    cb.setEnabled(True)
            self._set(r, 5, "启动中…")
        self.act_start.setEnabled(False)
        self.statusBar().showMessage(f"正在启动 {len(chosen)} 个点位…")
        self.append_log(f"[界面] 启动 {', '.join(chosen)}")

    def stop_all(self):
        if self.worker is None:
            return
        self.append_log("[界面] 正在停止全部点位…")
        self.worker.stop_all()
        self.worker.shutdown()
        self.worker.wait(8000)
        self.worker = None
        self.act_start.setEnabled(True)
        for i, d in enumerate(self._devices):
            if self._cell_checkbox(i, 0).isChecked():
                self._set(i, 5, "已停止")
                self._set(i, 11, "—")
            for col in (3, 4):
                cb = self._cell_checkbox(i, col)
                if cb:
                    cb.setEnabled(False)
        self.banner.setText("已停止")
        self.banner.setStyleSheet(style.banner_style(0, widget=self))
        self.lbl_run.setText("运行 0 台")

    def _on_row_toggle(self, device: str, on: bool):
        """启动列:运行中勾选=启动这台,取消勾选=停这台"""
        if self._syncing or self.worker is None:
            return
        if on:
            try:
                opts = self._options()
            except Exception as e:
                QMessageBox.critical(self, "配置有误", str(e))
                return
            self.worker.start_devices([device], opts)
        else:
            self.worker.stop_device(device)
            r = self._row_of(device)
            self._set(r, 5, "已停止")
            for col in (3, 4):
                cb = self._cell_checkbox(r, col)
                if cb:
                    cb.setEnabled(False)

    def _on_channel_toggle(self, device: str, col: int, on: bool):
        if self._syncing or self.worker is None:
            return
        rt = self._cell_checkbox(self._row_of(device), 3)
        an = self._cell_checkbox(self._row_of(device), 4)
        self.worker.set_channels(device, rt.isChecked() if rt else None,
                                 an.isChecked() if an else None)

    def on_failed(self, device: str, msg: str):
        r = self._row_of(device)
        self._set(r, 5, "启动失败", style.level_color(3, self))
        self._set(r, len(COLS) - 1, msg)
        self.append_log(f"[界面] {device} 启动失败: {msg}")

    # ---------------------------------------------------------------- 快照渲染
    def on_snapshot(self, rows: list):
        self._syncing = True
        try:
            by_dev = {r["device"]: r for r in rows}
            worst, cam = 0, False
            for i, d in enumerate(self._devices):
                r = by_dev.get(d)
                if r is None:
                    cur = self.table.item(i, 5).text() if self.table.item(i, 5) else ""
                    if cur != "启动失败":            # 失败原因留在表上,别被下一次快照刷掉
                        self._set(i, 5, "启动中…" if cur == "启动中…" else "未启动")
                        for col in (6, 7, 8, 9, 11):
                            self._set(i, col, "—")
                    continue
                cb0 = self._cell_checkbox(i, 0)
                if cb0 and not cb0.isChecked():
                    cb0.setChecked(True)
                for col, key in ((3, "enable_realtime"), (4, "enable_analysis")):
                    cb = self._cell_checkbox(i, col)
                    if cb and cb.isChecked() != bool(r[key]):
                        cb.setChecked(bool(r[key]))
                name, color = self._status_text(r)
                self._set(i, 5, name, color)
                self._set(i, 6, self._pct(r["change_frac"]))
                self._set(i, 7, self._num(r["rate"], 4))
                self._set(i, 8, self._num(r["accel"], 4))
                self._set(i, 9, self._pct(r["valid_frac"]))
                self._set(i, 10, f"{r['history_s'] / 60:.1f}/{r['need_s'] / 60:.0f}min")
                self._set(i, 11, self._src_text(r), tip=self._src_tip(r))
                self._set(i, 12, self._analysis_text(r), tip=self._analysis_tip(r))
                notes = list(r["reasons"])
                if r["error"]:
                    notes.append(f"异常: {r['error']}")
                self._set(i, len(COLS) - 1, "; ".join(notes) or "—")
                worst = max(worst, r["level"])
                cam = cam or r["camera_alarm"]
            self._banner(worst, cam, len(rows))
            self.lbl_run.setText(f"运行 {len(rows)} 台")
        finally:
            self._syncing = False

    def _status_text(self, r: dict):
        if r["camera_alarm"]:
            return "相机异常", style.camera_color(self)
        # 已经有判定结果就显示等级(那是真实判定),别被"预热"盖掉
        if r["has_verdict"]:
            return r["level_name"], style.level_color(r["level"], self)
        if not r["enable_realtime"] and not r["enable_analysis"]:
            return "通道全关", style.muted_color(self)
        if r["enable_realtime"]:
            pct = min(100, 100 * r["history_s"] / max(r["need_s"], 1e-6))
            return f"预热 {pct:.0f}%", style.muted_color(self)
        return "仅分析", style.muted_color(self)

    def _banner(self, level: int, cam: bool, n: int):
        if cam:
            self.banner.setText("⚠ 相机异常(画面整体位移,疑似被碰动/晃动)——不计入滑坡判定")
        elif level >= 3:
            self.banner.setText(f"● 红色报警({n} 台在跑):变化率突增或加速度过快,已存证据图")
        elif level == 2:
            self.banner.setText(f"● 橙色预警({n} 台在跑):持续形变,建议核对现场")
        elif level == 1:
            self.banner.setText(f"● 黄色关注({n} 台在跑):变化率超过关注阈值")
        else:
            self.banner.setText(f"正常({n} 台在跑) · 变化率未超阈值")
        self.banner.setStyleSheet(style.banner_style(level, cam, widget=self))

    @staticmethod
    def _pct(x) -> str:
        return "—" if x is None else f"{100 * x:.2f}%"

    @staticmethod
    def _num(x, digits: int) -> str:
        return "—" if x is None else f"{x:.{digits}f}"

    @staticmethod
    def _src_text(r: dict) -> str:
        """表格里只放最要紧的:连接状态 + 帧率 + 错误数;帧数/重连次数在悬停提示里"""
        txt = ("已连" if r["connected"] else "断开") + f" {r['fps']:.1f}fps"
        if r["errors"]:
            txt += f" 错{r['errors']}"
        return txt

    @staticmethod
    def _src_tip(r: dict) -> str:
        t = (f"帧源 {'已连接' if r['connected'] else '已断开'}\n"
             f"已抽帧 {r['frames']} 帧 · {r['fps']:.1f} fps\n"
             f"错误 {r['errors']} 次 · 重连 {r['reconnects']} 次")
        if r["src_error"]:
            t += f"\n最近错误: {r['src_error']}"
        return t

    @staticmethod
    def _analysis_text(r: dict) -> str:
        if r["analysis_at"] is None:
            return "—"
        ago = time.time() - r["analysis_at"]
        txt = f"{ago:.0f}s前"
        if r["analysis_diff"] is not None and r["analysis_diff"] == r["analysis_diff"]:
            txt += f" d={r['analysis_diff']:.2f}"
        short = LEVEL_SHORT.get(r["analysis_alarm"] or "")
        return f"{txt} {short}" if short else txt

    @staticmethod
    def _analysis_tip(r: dict) -> str:
        if r["analysis_at"] is None:
            return "还没跑过分析(启动后约 2 秒开始第一轮)"
        ago = time.time() - r["analysis_at"]
        t = f"上次分析 {ago:.0f} 秒前"
        if r["analysis_diff"] is not None and r["analysis_diff"] == r["analysis_diff"]:
            t += f"\n帧间变化 diff_frac = {r['analysis_diff']:.4f}"
        if r["analysis_shift"] is not None and r["analysis_shift"] == r["analysis_shift"]:
            t += f"\n帧间位移 {r['analysis_shift']:.2f} px"
        if r["analysis_rain"] is not None and r["analysis_rain"] == r["analysis_rain"]:
            t += f"\n1h 降雨 {r['analysis_rain']:.1f} mm"
        if r["analysis_alarm"]:
            t += f"\n分析通道等级: {r['analysis_alarm']}"
        return t

    # ---------------------------------------------------------------- 循环
    def append_log(self, line: str):
        self.log.appendPlainText(line)

    def open_alarms_dir(self):
        d = config.ROOT / "alarms"
        d.mkdir(parents=True, exist_ok=True)
        self._open(d)

    def _open(self, path):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def closeEvent(self, ev):
        if self.worker is not None:
            self.worker.stop_all()
            self.worker.shutdown()
            self.worker.wait(8000)
        self.worker = None
        super().closeEvent(ev)
