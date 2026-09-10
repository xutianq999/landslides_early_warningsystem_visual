"""把特征导出成 Excel(字段清单 + 原始数据),供人工挑选特征。

用法:
    .venv/bin/python export_features_excel.py [图片.jpg] [-o 输出.xlsx]
    # 若已存在 features.csv,优先用它;否则对给定图片(默认 3.jpg)跑一次提取

产出:
    特征清单  字段 / 类别 / 本次值 / 含义 / 单位范围 / 危险方向 / 保留(Y/N)
    原始数据  宽表(time + 所有特征列),即 xLSTM 的输入格式
"""

import argparse
import csv
import os
import sys

import numpy as np

XLSX_SKILL_DIR = ("/Users/Zhuanz/.zcode/cli/plugins/cache/zcode-plugins-official/"
                  "document-skills/0.1.4/skills/xlsx")
sys.path.insert(0, XLSX_SKILL_DIR)
sys.path.insert(0, os.path.join(XLSX_SKILL_DIR, "templates"))

import base  # noqa: E402  (技能自带的设计规范:配色/字体/样式工厂)

# macOS 本地中文字体优先
if os.path.exists("/System/Library/Fonts/Hiragino Sans GB.ttc"):
    base.FONT_NAME = "Hiragino Sans GB"

from openpyxl import Workbook  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402
from openpyxl.worksheet.datavalidation import DataValidation  # noqa: E402

# 字段文档:字段名 → (类别, 含义, 单位/范围, 危险方向)
DOCS = {
    "time": ("元信息", "本帧时间(ISO 格式)", "时间", "—"),
    "image": ("元信息", "源图片文件名", "—", "—"),
    # ① 图像层
    "img_brightness": ("图像质量", "平均亮度(噪声门控:太暗/过曝的帧不可信)", "0~1", "低于 0.12 或高于 0.95 应剔除"),
    "img_blur": ("图像质量", "拉普拉斯方差(越大越清晰,门控失焦)", "≥40 可用", "过小=失焦/雨雾,剔除该帧"),
    # 区域掩模(仅 --roi-auto 时)
    "gully_area_frac": ("区域掩模", "沟壑掩模面积占全图比例", "0~1", "扩大=沟壑在扩张/侧壁失稳"),
    "gully_width_min": ("区域掩模", "沟壑最小宽度(仅统计宽度≥1%图宽的有效行)", "0~1(占图宽)", "持续减小=沟壑收窄/被堵塞"),
    "gully_width_max": ("区域掩模", "沟壑最大宽度(逐行左右边界差的最大值)", "0~1(占图宽)", "增大=沟壑变宽"),
    "gully_width_std": ("区域掩模", "沟壑宽度沿纵深的波动(标准差)", "0~1(占图宽)", "增大=沟壑形状不规则化"),
    "gully_y_top": ("区域掩模", "沟壑掩模顶端位置(归一化 y)", "0~1", "上移=沟壑向上溯源侵蚀"),
    "gully_y_bottom": ("区域掩模", "沟壑掩模底端位置(归一化 y)", "0~1", "下移=沟壑向下延伸"),
    "gully_y_extent": ("区域掩模", "沟壑纵向跨度(y_bottom − y_top)", "0~1", "增大=沟壑纵深变长"),
    "debris_area_frac": ("区域掩模", "底部堆积体掩模面积占比", "0~1", "增大=堆积体增长"),
    "debris_width_min": ("区域掩模", "堆积体最小宽度(有效行)", "0~1(占图宽)", "减小=堆积体收缩"),
    "debris_width_max": ("区域掩模", "堆积体最大宽度", "0~1(占图宽)", "增大=堆积体横向扩张"),
    "debris_width_std": ("区域掩模", "堆积体宽度沿纵深的波动", "0~1(占图宽)", "增大=堆积体形状不规则化"),
    "debris_y_top": ("区域掩模", "堆积体顶端位置(归一化 y)", "0~1", "上移=堆积体向坡上扩展"),
    "debris_y_bottom": ("区域掩模", "堆积体底端位置(归一化 y)", "0~1", "—"),
    "debris_y_extent": ("区域掩模", "堆积体纵向跨度", "0~1", "增大=堆积体纵向增长"),
    # ② 深度层
    "disp_p05": ("深度分布", "归一化逆深度(视差)5% 分位,越小越远", "0~1", "下降=画面中远处占比变大"),
    "disp_p50": ("深度分布", "视差中位数(整体远近)", "0~1", "双向变化都值得关注"),
    "disp_p95": ("深度分布", "视差 95% 分位,越大越近", "0~1", "上升=有物体逼近(堆积体前缘)"),
    "disp_std": ("深度分布", "视差标准差(深度分布离散度)", "0~0.5", "骤降可能是深度模型失效"),
    # ③ 点云层
    "slope_mean": ("三维几何", "坡度角均值(点云法向量与上方向夹角)", "度 0~90", "持续增大=坡面变陡"),
    "slope_p95": ("三维几何", "坡度角 95% 分位", "度", "反映最陡区域,陡坎增多会上升"),
    "rough_local": ("三维几何", "局部粗糙度(坡度在 5x5 窗口的标准差)", "度", "增大=表面破碎化(裂缝发育/土体松散)"),
    "curv_mean": ("三维几何", "曲率均值(深度拉普拉斯归一化)", "0~1", "增大=局部凹凸加剧"),
    "plane_tilt": ("三维结构", "PCA 主平面倾角", "度 0~90", "增大=整体坡面变陡"),
    "plane_rms": ("三维结构", "相对主平面的归一化残差(平整度)", "0~1", "增大=表面解体/不平整"),
    "bulge_frac": ("三维结构", "凸起(朝相机外凸)面积占比", "0~1", "上升是坡脚鼓胀的经典前兆"),
    "edge_depth_corr": ("可信度", "图像边缘与深度边缘的相关系数", "-1~1", "接近 0 说明该帧深度不可信,应降权"),
    # ④ 时序层
    "shift_px": ("帧间变化", "与上一帧的配准位移模长", "像素", ">8 说明相机被碰动/漂移,本身即告警"),
    "shift_resp": ("可信度", "相位相关响应值(配准可靠度)", "0~1", "低于 0.2 该帧变化特征不可信"),
    "diff_mean": ("帧间变化", "深度差均值", "0~1", "增大=整体深度结构变化"),
    "diff_p95": ("帧间变化", "深度差 95% 分位", "0~1", "增大=局部剧烈变化"),
    "diff_frac": ("帧间变化", "深度变化超 10% 归一化范围的面积占比", "0~1", "核心形变指标,持续上升=坡体活动"),
    # 降雨
    "rain_1h": ("降雨", "抓图前 1 小时累积降雨", "mm", "短时强降雨是直接触发因素"),
    "rain_24h": ("降雨", "前 24 小时累积降雨", "mm", "累积降雨是最强预测因子"),
    "rain_72h": ("降雨", "前 72 小时累积降雨", "mm", "持续降雨导致孔隙水压上升"),
}

# 分割类字段:landslide / deep valley 统计面积,动态物体只统计数量
SEG_DOC = {
    "frac": ("分割", "该类别像素占全图比例", "0~1", "占比突变=坡面物质变化"),
    "max": ("分割", "最大连通域面积占比", "0~1", "增大=出现大面积同类区域"),
    "n": ("动态物体", "该类别实例个数(其区域已从几何/深度/变化计算中剔除)", "个", "数量本身是施工活动强度,不影响几何特征"),
}

NOTES = [
    "说明:",
    "1. 所有几何量都在 DA V2 重建坐标系下计算,形状比例正确但有仿射系统偏差(坡度绝对值偏大)。",
    "   同一相机下偏差恒定,监测请看时间序列的相对变化,不要看绝对值。",
    "2. disp_* 是归一化逆深度(视差),值越大越近,不是米制深度。",
    "3. 现场暂无尺度标定,位移为像素单位;米制量(体积/裂缝宽度)需放已知尺寸参照物后才能算。",
    "4. 每段第一帧的帧间变化特征为 nan;分割未检出的类别为 0。",
    "5. edge_depth_corr 和 shift_resp 低的行说明该帧不可信,建议降权或剔除。",
    "6. 无标签场景建议把 xLSTM 用作特征向量下一步预测器,预测残差即异常分;降雨特征用于降低误报。",
    "7. YOLOE 零样本对裂缝/堆积体等细结构概念检出为 0,默认只放实体类别,裂缝需另做检测。",
]


def field_doc(name):
    if name in DOCS:
        return DOCS[name]
    if name.startswith("seg_"):
        cls, _, kind = name[4:].rpartition("_")
        if kind in SEG_DOC:
            cat, meaning, unit, danger = SEG_DOC[kind]
            return cat, f"{cls}:{meaning}", unit, danger
    return "其他", "—", "—", "—"


def fmt(v):
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return "nan" if v != v else f"{v:.4f}"
    return str(v)


def load_rows(image_path):
    """优先读 features.csv;没有就对图片跑一次提取。返回 [{字段: 值}]"""
    if os.path.exists("features.csv"):
        with open("features.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            print(f"数据来源: features.csv({len(rows)} 行)")
            return rows
    import features as F
    from PIL import Image
    pil = Image.open(image_path).convert("RGB")
    row, _ = F.features_from_image(pil)
    print(f"数据来源: 对 {image_path} 现场提取")
    return [row]


def build(rows, out_path):
    fields = list(rows[0].keys())
    wb = Workbook()

    # ---------- Sheet 1: 特征清单 ----------
    ws = wb.active
    ws.title = "特征清单"
    headers = ["序号", "字段名", "类别", "本次值", "含义", "单位/范围", "危险方向",
               "保留(Y/N)", "备注"]
    last_col = len(headers) + 1
    base.setup_sheet(ws, title="滑坡监测特征清单", last_col=last_col)
    for c, h in enumerate(headers, 2):
        ws.cell(row=4, column=c, value=h)
    base.style_header_row(ws, row_num=4, col_start=2, col_end=last_col)

    for i, name in enumerate(fields):
        r = 5 + i
        cat, meaning, unit, danger = field_doc(name)
        values = [fmt(row.get(name, "")) for row in rows]
        val = values[0] if len(values) == 1 else f"{len(values)} 帧"
        for c, v in enumerate([i + 1, name, cat, val, meaning, unit, danger, "", ""], 2):
            ws.cell(row=r, column=c, value=v)
        base.style_data_row(ws, row_num=r, col_start=2, col_end=last_col, row_index=i)

    # 「保留」列加下拉框,方便挑选
    dv = DataValidation(type="list", formula1='"Y,N"', allow_blank=True)
    dv.prompt = "选 Y 保留该特征 / N 剔除"
    ws.add_data_validation(dv)
    dv.add(f"H5:H{4 + len(fields)}")

    # 列宽(文本列给足空间)
    for col, width in zip(range(2, last_col + 1), [6, 20, 12, 12, 46, 14, 34, 11, 20]):
        ws.column_dimensions[get_column_letter(col)].width = width
    base.auto_fit_row_heights(ws, header_row=4, data_start_row=5,
                              data_end_row=4 + len(fields))

    # 说明写在表格下方
    note_row = 5 + len(fields) + 2
    for j, line in enumerate(NOTES):
        cell = ws.cell(row=note_row + j, column=2, value=line)
        cell.font = base.font_caption() if j else Font(
            name=base.FONT_NAME, size=10, bold=True, color=base.PRIMARY)
        cell.alignment = Alignment(horizontal="left", vertical="center")
    ws.freeze_panes = "C5"
    ws.sheet_view.showGridLines = False

    # ---------- Sheet 2: 原始数据 ----------
    ws2 = wb.create_sheet("原始数据")
    last2 = len(fields) + 1
    base.setup_sheet(ws2, title="原始特征数据(xLSTM 输入格式)", last_col=last2)
    for c, h in enumerate(fields, 2):
        ws2.cell(row=4, column=c, value=h)
    base.style_header_row(ws2, row_num=4, col_start=2, col_end=last2)
    for i, row in enumerate(rows):
        for c, name in enumerate(fields, 2):
            v = row.get(name, "")
            if isinstance(v, str):
                try:
                    v = float(v)
                except ValueError:
                    pass
            ws2.cell(row=5 + i, column=c, value=v)
        base.style_data_row(ws2, row_num=5 + i, col_start=2, col_end=last2, row_index=i)
    base.auto_fit_columns(ws2, min_width=8, max_width=20, header_row=4, data_start_row=5)
    base.auto_fit_row_heights(ws2, header_row=4, data_start_row=5,
                              data_end_row=4 + len(rows))
    ws2.freeze_panes = "C5"
    ws2.sheet_view.showGridLines = False

    wb.properties.creator = "Z.ai"
    wb.save(out_path)
    return out_path, len(fields), len(rows)


def static_field_list():
    """不跑推理,直接从 features.py 的定义推导出全部字段(与运行结果一致)。"""
    import features as F
    fields = ["img_brightness", "img_blur"]
    for p in ("gully", "debris"):
        fields += [f"{p}_area_frac", f"{p}_width_min", f"{p}_width_max", f"{p}_width_std",
                   f"{p}_y_top", f"{p}_y_bottom", f"{p}_y_extent"]
    for c in [x.strip() for x in F.DEFAULT_CLASSES.split(",")]:
        if c in F.STATIC_CLASSES:
            fields += [f"seg_{c}_frac", f"seg_{c}_max"]
    for c in [x.strip() for x in F.DEFAULT_CLASSES.split(",")]:
        if c in F.DYNAMIC_CLASSES:
            fields += [f"seg_{c}_n"]
    fields += ["disp_p05", "disp_p50", "disp_p95", "disp_std",
               "slope_mean", "slope_p95", "rough_local", "curv_mean",
               "plane_tilt", "plane_rms", "bulge_frac", "edge_depth_corr",
               "shift_px", "shift_resp", "diff_mean", "diff_p95", "diff_frac",
               "rain_1h", "rain_24h", "rain_72h"]
    return fields


def build_fields_only(fields, out_path):
    """只输出字段清单(无本次值),用于挑选特征。"""
    wb = Workbook()
    ws = wb.active
    ws.title = "字段清单"
    headers = ["序号", "字段名", "类别", "含义", "单位/范围", "危险方向", "保留(Y/N)", "备注"]
    last_col = len(headers) + 1
    base.setup_sheet(ws, title="滑坡监测特征字段清单", last_col=last_col)
    for c, h in enumerate(headers, 2):
        ws.cell(row=4, column=c, value=h)
    base.style_header_row(ws, row_num=4, col_start=2, col_end=last_col)

    for i, name in enumerate(fields):
        r = 5 + i
        cat, meaning, unit, danger = field_doc(name)
        for c, v in enumerate([i + 1, name, cat, meaning, unit, danger, "", ""], 2):
            ws.cell(row=r, column=c, value=v)
        base.style_data_row(ws, row_num=r, col_start=2, col_end=last_col, row_index=i)

    dv = DataValidation(type="list", formula1='"Y,N"', allow_blank=True)
    dv.prompt = "选 Y 保留该特征 / N 剔除"
    ws.add_data_validation(dv)
    dv.add(f"G5:G{4 + len(fields)}")

    for col, width in zip(range(2, last_col + 1), [6, 20, 12, 48, 14, 34, 11, 20]):
        ws.column_dimensions[get_column_letter(col)].width = width
    base.auto_fit_row_heights(ws, header_row=4, data_start_row=5, data_end_row=4 + len(fields))

    note_row = 5 + len(fields) + 2
    for j, line in enumerate(NOTES):
        cell = ws.cell(row=note_row + j, column=2, value=line)
        cell.font = base.font_caption() if j else Font(
            name=base.FONT_NAME, size=10, bold=True, color=base.PRIMARY)
        cell.alignment = Alignment(horizontal="left", vertical="center")
    ws.freeze_panes = "C5"
    ws.sheet_view.showGridLines = False

    wb.properties.creator = "Z.ai"
    wb.save(out_path)
    return out_path, len(fields)


def main():
    ap = argparse.ArgumentParser(description="特征导出 Excel")
    ap.add_argument("image", nargs="?", default="3.jpg", help="features.csv 不存在时用这张图跑一次提取")
    ap.add_argument("-o", "--out", default="features.xlsx")
    ap.add_argument("--fields-only", action="store_true",
                    help="只输出字段清单(不跑推理、不含数值)")
    args = ap.parse_args()

    if args.fields_only:
        path, n_fields = build_fields_only(static_field_list(), args.out)
        print(f"已生成 {path}:{n_fields} 个字段(仅清单,无数值)")
        return

    rows = load_rows(args.image)
    path, n_fields, n_rows = build(rows, args.out)
    print(f"已生成 {path}:{n_fields} 个字段 × {n_rows} 帧")


if __name__ == "__main__":
    main()
