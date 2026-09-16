#!/usr/bin/env python3
"""运行时监视台入口:启停两套通道,盯实时指标与报警。

启动:
    .venv/bin/python runtime.py

与 studio.py 的分工:studio.py 离线调参并导出 profiles/*.json,本程序读同一份配置
把 `monitor.DeviceRuntime` 跑起来(与命令行 monitor.py 是同一个类,参数与结果一致)。
"""

import os
import sys


def main() -> int:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))   # ultralytics 按 cwd 解析 mobileclip2_b.ts

    from PySide6.QtWidgets import QApplication
    from gui import style
    from gui.runtime_window import RuntimeWindow

    app = QApplication(sys.argv)
    app.setApplicationName("滑坡监测运行时监视台")
    app.setApplicationDisplayName("滑坡监测运行时监视台")
    style.apply(app)                      # 与 studio.py 同一份字号/配色
    win = RuntimeWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
