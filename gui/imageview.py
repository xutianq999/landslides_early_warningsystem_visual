"""图像查看器:缩放、平移、鼠标坐标,基于 QGraphicsView。"""

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


def np_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """numpy(uint8: HxW 灰度 或 HxWx3 RGB) → QPixmap(拷贝,不共享内存)"""
    a = np.ascontiguousarray(arr)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    h, w = a.shape[:2]
    if a.ndim == 2:
        img = QImage(a.data, w, h, w, QImage.Format_Grayscale8)
    elif a.shape[2] == 4:
        img = QImage(a.data, w, h, 4 * w, QImage.Format_RGBA8888)
    else:
        img = QImage(a.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(img.copy())   # copy:自己持有缓冲,避免 numpy 释放后花屏


class ImageView(QGraphicsView):
    """显示一张图并支持滚轮缩放 / 拖拽平移,鼠标移动时回报图像像素坐标。"""

    coords = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = QGraphicsPixmapItem()
        self._scene.addItem(self._item)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHints(self.renderHints())
        self.setBackgroundBrush(Qt.darkGray)
        self._empty = True

    def set_array(self, arr: np.ndarray | None, keep_view: bool = False) -> None:
        if arr is None:
            self._item.setPixmap(QPixmap())
            self._empty = True
            return
        self._item.setPixmap(np_to_qpixmap(arr))
        self._scene.setSceneRect(self._item.boundingRect())
        self._empty = False
        if not keep_view:
            self.fit()

    def fit(self) -> None:
        if not self._empty:
            self.fitInView(self._item, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        if self._empty:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mouseMoveEvent(self, event):
        if not self._empty:
            p = self.mapToScene(event.position().toPoint())
            self.coords.emit(int(p.x()), int(p.y()))
        super().mouseMoveEvent(event)
