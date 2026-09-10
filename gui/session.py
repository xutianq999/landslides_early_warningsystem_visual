"""标签页之间共享的状态:当前图片、当前设备号。"""

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

import config


class Session(QObject):
    image_changed = Signal()
    device_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.image_path: str | None = None
        self.rgb: np.ndarray | None = None      # uint8 (H,W,3) RGB
        self.bgr: np.ndarray | None = None      # uint8 (H,W,3) BGR
        self._device = config.CONFIG["device_id"]

    # ---- 图片 ----
    def load_image(self, path: str) -> bool:
        bgr = cv2.imread(path)
        if bgr is None:
            return False
        self.bgr = bgr
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.image_path = path
        self.image_changed.emit()
        return True

    def set_image(self, rgb: np.ndarray, path: str | None = None) -> None:
        self.rgb = np.ascontiguousarray(rgb)
        self.bgr = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)
        self.image_path = path
        self.image_changed.emit()

    def has_image(self) -> bool:
        return self.rgb is not None

    # ---- 设备 ----
    @property
    def device(self) -> str:
        return self._device

    def set_device(self, device: str) -> None:
        if device and device != self._device:
            self._device = device
            self.device_changed.emit(device)
