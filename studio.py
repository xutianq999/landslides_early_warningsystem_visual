#!/usr/bin/env python3
"""桌面工作台入口:一个窗口统管 分割 / 深度 / 特征 / 抓图 / 数据。

启动:
    .venv/bin/python studio.py

注意:启动时会切到项目根目录——YOLOE 的 mobileclip2_b.ts 由 ultralytics 按当前
工作目录解析,换目录会触发联网下载。
"""

import os
import sys


def main() -> int:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))   # 必须在导入 ultralytics 之前

    from PySide6.QtWidgets import QApplication
    from gui.mainwindow import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("滑坡监测工作台")
    app.setApplicationDisplayName("滑坡监测工作台")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
