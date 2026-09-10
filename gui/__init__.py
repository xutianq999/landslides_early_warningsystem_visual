"""桌面工作台(PySide6):一个入口统管 分割 / 深度 / 特征 / 抓图 / 数据。

设计原则:GUI 只做编排,不重写算法——所有计算调用既有模块
(segment_gully / segment_color / core / features / capture / alarm / db),
保证桌面台与命令行结果一致。
"""
