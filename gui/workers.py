"""后台任务:把推理/IO 放到 QThread,界面不冻结。

用法:
    self.runner.run(callable, arg1, on_done=self._ok, on_error=self._err, kw=1)
"""

import traceback

from PySide6.QtCore import QObject, QThread, Signal


class _Worker(QThread):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, args, kwargs, parent=None):
        super().__init__(parent)
        self._fn, self._args, self._kwargs = fn, args, kwargs

    def run(self):
        try:
            self.done.emit(self._fn(*self._args, **self._kwargs))
        except Exception as e:                       # 后台异常必须回传,不能静默
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


class TaskRunner(QObject):
    """同一时刻只跑一个后台任务;持有引用防止线程被 GC。"""

    started = Signal(str)          # 任务名(用于状态栏)
    finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: _Worker | None = None
        self._name = ""

    def busy(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def run(self, fn, *args, on_done=None, on_error=None, name="计算中", **kwargs):
        if self.busy():
            return False
        self._name = name
        self.started.emit(name)
        w = _Worker(fn, args, kwargs, parent=self)
        self._worker = w

        def _done(result):
            self._worker = None
            self.finished.emit()
            if on_done:
                on_done(result)

        def _fail(msg):
            self._worker = None
            self.finished.emit()
            if on_error:
                on_error(msg)

        w.done.connect(_done)
        w.failed.connect(_fail)
        w.start()
        return True
