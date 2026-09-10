"""本地只读 REST API:平台通过 HTTP 拉取特征与报警。

启动:
    .venv/bin/python api.py
    # 或: uvicorn api:app --host 127.0.0.1 --port 8000

监听地址/端口/数据库路径见 config.py(config.json 或环境变量)。
交互式文档:启动后打开 http://<host>:<port>/docs

**不鉴权**。默认只监听 127.0.0.1(仅本机可访问);若改成 0.0.0.0 对局域网开放,
任何能连上该端口的人都能读取全部数据,请自行确保网络边界安全。
"""

import argparse
import csv
import io
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse

import config
import db as dbm

app = FastAPI(
    title="滑坡监测数据 API",
    version="1.0",
    description=(
        "边缘端(YOLOE 分割 + DA V2 深度)提取的滑坡监测特征与分级报警,供平台拉取。\n\n"
        "**注意**:坡度为 DA V2 相对深度重建结果,存在仿射系统偏差,越远越明显;"
        "位移为像素单位。请把数值当作**时间序列的相对变化**使用,不要当米制真值。"
        "字段含义见项目 FEATURES.md;ROI 掩模特征仅在采集端开启 `--roi-auto` 时才有值。"
    ),
)


@contextmanager
def _conn():
    conn = dbm.connect()
    try:
        yield conn
    finally:
        conn.close()


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/api/v1/health", tags=["系统"], summary="健康检查")
def health():
    """探活用:返回 schema 版本、库路径与帧数。"""
    with _conn() as conn:
        s = dbm.stats(conn)
    return {"status": "ok", "schema_version": s["schema_version"],
            "frames": s["frames"], "alarms": s["alarms"], "db_path": s["db_path"]}


@app.get("/api/v1/devices", tags=["设备"], summary="设备列表与当前状态")
def devices():
    """每台设备:设备号、名称/位置(来自 config.devices)、帧数、首末上报、当前报警等级。

    含已登记但暂无数据的设备;`rtsp_url` 不回传(内含相机口令),只给 `has_rtsp_url`。
    `params` 是该设备生效的计算参数(类别/FOV/ROI/报警阈值等),同样不含 rtsp_url。
    """
    with _conn() as conn:
        rows = dbm.list_devices(conn)
    for d in rows:
        d["params"] = config.safe_params(d["device_id"])
    return rows


@app.get("/api/v1/devices/{device_id}", tags=["设备"], summary="单台设备状态")
def device(device_id: str):
    with _conn() as conn:
        d = dbm.get_device(conn, device_id)
    if not d:
        raise HTTPException(status_code=404, detail=f"设备不存在: {device_id}")
    d["params"] = config.safe_params(device_id)
    return d


@app.get("/api/v1/frames", tags=["特征"], summary="按条件查询特征帧")
def frames(device_id: str | None = None, since: str | None = None, until: str | None = None,
           limit: int = Query(100, ge=1, le=5000), offset: int = Query(0, ge=0),
           order: str = Query("desc", pattern="^(asc|desc)$")):
    with _conn() as conn:
        items = dbm.query_frames(conn, device_id, since, until, limit, offset, order)
    return {"count": len(items), "items": items}


@app.get("/api/v1/frames/latest", tags=["特征"], summary="最近 N 帧(看板用)")
def frames_latest(device_id: str | None = None, n: int = Query(1, ge=1, le=1000)):
    with _conn() as conn:
        items = dbm.latest_frames(conn, n, device_id)
    return {"count": len(items), "items": items}


@app.get("/api/v1/frames/{frame_id}", tags=["特征"], summary="单帧全字段")
def frame(frame_id: int):
    with _conn() as conn:
        row = dbm.get_frame(conn, frame_id)
    if not row:
        raise HTTPException(status_code=404, detail="帧不存在")
    return row


@app.get("/api/v1/frames/{frame_id}/image", tags=["特征"], summary="该帧对应的抓图")
def frame_image(frame_id: int):
    with _conn() as conn:
        row = dbm.get_frame(conn, frame_id)
    if not row:
        raise HTTPException(status_code=404, detail="帧不存在")
    if not row.get("image_path"):
        raise HTTPException(status_code=404, detail="该帧没有关联图片(如网页端未落盘)")
    p = Path(row["image_path"])
    if not p.is_absolute():
        p = config.ROOT / p
    p = p.resolve()
    root = config.ROOT.resolve()
    if p != root and root not in p.parents:      # 防路径穿越:只允许项目目录内的文件
        raise HTTPException(status_code=403, detail="图片路径越界")
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"图片文件不存在: {p}")
    return FileResponse(str(p))


@app.get("/api/v1/series", tags=["特征"], summary="指定字段的时间序列")
def series(fields: str = Query(..., description="逗号分隔的字段名,如 diff_frac,slope_mean"),
           device_id: str | None = None, since: str | None = None, until: str | None = None,
           limit: int | None = Query(None, ge=1, le=100000)):
    names = [f.strip() for f in fields.split(",") if f.strip()]
    try:
        with _conn() as conn:
            return dbm.series(conn, names, device_id, since, until, limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/alarms", tags=["报警"], summary="查询报警判定")
def alarms(device_id: str | None = None, since: str | None = None, until: str | None = None,
           min_level: int = Query(0, ge=0, le=3), limit: int = Query(100, ge=1, le=5000),
           offset: int = Query(0, ge=0), order: str = Query("desc", pattern="^(asc|desc)$"),
           source: str | None = Query(None, pattern="^(realtime|analysis)$",
                                      description="realtime=突发报警, analysis=趋势预警")):
    with _conn() as conn:
        items = dbm.query_alarms(conn, device_id, since, until, min_level, limit, offset,
                                 order, source)
    return {"count": len(items), "items": items}


@app.get("/api/v1/realtime", tags=["报警"], summary="实时通道指标(变化率/速率/加速度)")
def realtime_metrics(device_id: str | None = None, since: str | None = None,
                     until: str | None = None, min_level: int = Query(0, ge=0, le=3),
                     limit: int = Query(200, ge=1, le=5000), offset: int = Query(0, ge=0),
                     order: str = Query("desc", pattern="^(asc|desc)$")):
    """1 Hz 判定、默认 10 s 落库的变化率序列;报警由实时通道产生(source=realtime)。"""
    with _conn() as conn:
        items = dbm.query_realtime(conn, device_id, since, until, min_level, limit, offset, order)
    return {"count": len(items), "items": items}


@app.get("/api/v1/alarms/latest", tags=["报警"], summary="每台设备当前报警等级")
def alarms_latest(device_id: str | None = None):
    with _conn() as conn:
        return dbm.latest_alarms(conn, device_id)


@app.get("/api/v1/export.csv", tags=["导出"], summary="特征帧 CSV 导出",
         response_class=PlainTextResponse)
def export_csv(device_id: str | None = None, since: str | None = None, until: str | None = None,
               limit: int = Query(10000, ge=1, le=1000000)):
    with _conn() as conn:
        rows = dbm.query_frames(conn, device_id, since, until, limit, offset=0, order="asc")
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return PlainTextResponse(buf.getvalue(), media_type="text/csv; charset=utf-8",
                             headers={"Content-Disposition": 'attachment; filename="frames.csv"'})


def main():
    import uvicorn
    ap = argparse.ArgumentParser(description="滑坡监测数据 API")
    ap.add_argument("--host", default=config.CONFIG["api_host"])
    ap.add_argument("--port", type=int, default=config.CONFIG["api_port"])
    ap.add_argument("--reload", action="store_true", help="开发用:改动自动重载")
    args = ap.parse_args()
    print(f"API 文档: http://{args.host}:{args.port}/docs")
    print(f"数据库  : {dbm.resolve_db_path()}")
    if args.host == "0.0.0.0":
        print("提示: 正在监听 0.0.0.0,局域网内任何能连上该端口的人都能读取全部数据。"
              "本服务不鉴权,请自行确保网络边界安全(只本机用就保持 127.0.0.1)")
    uvicorn.run("api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
