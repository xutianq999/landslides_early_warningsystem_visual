"""界面外观统一设置:字号、颜色、表格行高。

**所有字号只在这里定义一份。** 以前是各处写死(日志 11px、横幅 15px、其余吃系统默认),
凑在同一个窗口里就不齐;换台机器/换个缩放比例更明显。

文字颜色一律走调色板,不写死 #333:深色模式下写死的深灰在深背景上几乎看不见
(用户截图就是深色模式)。

两个入口都调用 `apply(app)`,所以 studio.py 与 runtime.py 的字号天然一致。
"""

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

# ---------------------------------------------------------------- 字号(唯一来源)
BASE_PT = 15          # 正文、按钮、输入框、表格
SMALL_PT = 13         # 次要说明、提示文字
BANNER_PT = 17        # 报警横幅
LOG_PT = 13           # 日志(等宽)
MONO_FAMILIES = ("Menlo", "SF Mono", "Consolas", "DejaVu Sans Mono", "monospace")


def mono_font() -> QFont:
    f = QFont()
    f.setFamilies(list(MONO_FAMILIES))
    f.setStyleHint(QFont.Monospace)
    f.setPointSize(LOG_PT)
    return f


def is_dark(widget=None) -> bool:
    """当前是深色还是浅色主题(取控件所在窗口的调色板,拿不到就用全局)"""
    pal = widget.palette() if widget is not None else QApplication.palette()
    return pal.color(QPalette.Window).lightness() < 128


def text_color(widget=None) -> QColor:
    """常规文字色(跟随主题)"""
    pal = widget.palette() if widget is not None else QApplication.palette()
    return pal.color(QPalette.WindowText)


def muted_color(widget=None) -> QColor:
    """次要文字色:浅色下用中灰,深色下用亮灰(直接写死灰度会在另一边看不清)"""
    pal = widget.palette() if widget is not None else QApplication.palette()
    c = pal.color(QPalette.WindowText)
    if is_dark(widget):
        c.setAlpha(170)
    else:
        c.setAlpha(140)
    return c


def hint_style(widget=None) -> str:
    """次要说明标签的样式"""
    return f"color: rgba({muted_color(widget).red()}, {muted_color(widget).green()}, " \
           f"{muted_color(widget).blue()}, {muted_color(widget).alpha()});"


def log_style() -> str:
    """日志窗格:等宽 + 与正文协调又不喧宾夺主的字号"""
    return f"font-size: {LOG_PT}pt;"


# ---------------------------------------------------------------- 报警配色
# 每级两组色:浅色主题用深色字,深色主题用亮色字;底色统一用低透明度,两种主题都能读
_LEVEL_ACCENT = {
    0: ("#1b5e20", "#a5d6a7"),        # 正常 绿
    1: ("#8d6e00", "#ffe082"),        # 黄色关注
    2: ("#c25e00", "#ffb74d"),        # 橙色预警
    3: ("#b71c1c", "#ff8a80"),        # 红色紧急
}
_CAMERA_ACCENT = ("#4a148c", "#ce93d8")   # 相机异常(紫)


def level_color(level: int, widget=None) -> QColor:
    """表格里状态文字的颜色"""
    idx = 1 if is_dark(widget) else 0
    return QColor(_LEVEL_ACCENT.get(max(0, min(3, int(level))), _LEVEL_ACCENT[0])[idx])


def camera_color(widget=None) -> QColor:
    return QColor(_CAMERA_ACCENT[1 if is_dark(widget) else 0])


def banner_style(level: int = 0, camera: bool = False, widget=None) -> str:
    """报警横幅样式:底色用低透明度强调色,文字用与主题相配的高对比色"""
    if camera:
        fg, rgb = _CAMERA_ACCENT[1 if is_dark(widget) else 0], (74, 20, 140)
    else:
        fg, rgb = _LEVEL_ACCENT.get(max(0, min(3, int(level))), _LEVEL_ACCENT[0])[
            1 if is_dark(widget) else 0], {3: (183, 28, 28), 2: (194, 94, 0),
                                           1: (141, 110, 0), 0: (27, 94, 32)}[
            max(0, min(3, int(level)))]
    weight = 700 if (level >= 3 or camera) else 600
    return (f"font-size: {BANNER_PT}pt; font-weight: {weight}; padding: 8px;"
            f" border-radius: 4px; color: {fg};"
            f" background: rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.16);")


# ---------------------------------------------------------------- 全局应用

def apply(app: QApplication) -> QFont:
    """设置全局字体与控件级字号(两个入口都调用,保证一致)"""
    f = QFont()
    # 中文优先用系统黑体,回落到各平台默认;不指定具体字重,跟随系统
    f.setFamilies(["PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", "Helvetica Neue",
                   "Arial", "sans-serif"])
    f.setPointSize(BASE_PT)
    app.setFont(f)
    app.setStyleSheet(f"""
        QGroupBox {{
            font-size: {BASE_PT}pt;
            margin-top: 10px;
            padding-top: 6px;
        }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
        QTabBar::tab {{ font-size: {BASE_PT}pt; padding: 6px 12px; }}
        QTableWidget {{
            font-size: {BASE_PT}pt;
            gridline-color: rgba(128, 128, 128, 0.35);
        }}
        QHeaderView::section {{ font-size: {BASE_PT}pt; padding: 5px 6px; }}
        QPushButton {{ font-size: {BASE_PT}pt; padding: 4px 10px; }}
        QToolButton {{ font-size: {BASE_PT}pt; }}
        QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{ font-size: {BASE_PT}pt; padding: 2px 4px; }}
        QCheckBox, QRadioButton {{ font-size: {BASE_PT}pt; }}
        QLabel {{ font-size: {BASE_PT}pt; }}
        QStatusBar {{ font-size: {SMALL_PT}pt; }}
        QToolBar {{ font-size: {BASE_PT}pt; spacing: 4px; }}
    """)
    return f
