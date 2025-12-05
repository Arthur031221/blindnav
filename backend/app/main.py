import datetime
import math
import os
import re
import time
from typing import Any, Dict, List, Optional

# 🔹 一開始就設好 OpenMP workaround，後面所有 import 都會吃到這個設定
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from fastapi.responses import StreamingResponse
from fastapi import UploadFile, File
from app.bus_vision import (
    bus_video_generator,
    start_bus_vision,
    stop_bus_vision,
    reset_bus_vision,
    get_bus_status,
    receive_mobile_frame,
)
from app.pedestrian.service import (
    get_live_preview_response as get_pedestrian_live_preview,
    get_processed_video_response as get_pedestrian_video,
    get_progress as get_pedestrian_progress,
    start_processing as start_pedestrian_processing,
)

# === FastAPI app & static ===
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# === Google Maps API key ===
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
DEFAULT_ORIGIN = {"lat": 24.796123, "lng": 120.9935}


# === 資料模型 ===
class NavStep(BaseModel):
    index: int
    mode: str  # walk / bus_ride / arrive / etc.
    instruction: str
    distance_m: Optional[float] = None
    duration_min: Optional[float] = None
    bus_number: Optional[str] = None
    departure_stop: Optional[str] = None
    arrival_stop: Optional[str] = None
    target_lat: Optional[float] = None
    target_lng: Optional[float] = None
    bus_departure_timestamp: Optional[int] = None
    bus_departure_text: Optional[str] = None
    bus_stop_lat: Optional[float] = None   # departure stop latitude (公車站牌)
    bus_stop_lng: Optional[float] = None   # departure stop longitude (公車站牌)


class NavRoute(BaseModel):
    steps: List[NavStep]
    start_address: Optional[str] = None
    end_address: Optional[str] = None
    total_duration_min: Optional[float] = None
    summary: Optional[str] = None
    first_bus_wait_min: Optional[float] = None
    first_bus_number: Optional[str] = None
    overview_polyline: Optional[str] = None
    start_lat: Optional[float] = None
    start_lng: Optional[float] = None
    end_lat: Optional[float] = None
    end_lng: Optional[float] = None
    requested_departure_epoch: Optional[int] = None
    requested_departure_text: Optional[str] = None


class DeviceState(BaseModel):
    device_id: str
    created_at: datetime.datetime
    last_seen: datetime.datetime
    route: Optional[NavRoute] = None
    last_lat: Optional[float] = None
    last_lng: Optional[float] = None
    last_step_index: Optional[int] = None
    force_resume_requested: bool = False


class RegisterDeviceRequest(BaseModel):
    device_id: str


class ResetDeviceRequest(BaseModel):
    device_id: str


class RouteRequest(BaseModel):
    device_id: str


class GoogleRouteRequest(BaseModel):
    device_id: str
    destination: str
    origin_lat: float
    origin_lng: float
    departure_time: Optional[int] = None


class UpdateLocationRequest(BaseModel):
    device_id: str
    step_index: int
    lat: float
    lng: float


class UpdateLocationResponse(BaseModel):
    distance_to_target_m: Optional[float] = None
    message: Optional[str] = None


class DeviceSnapshot(BaseModel):
    device_id: str
    last_seen: datetime.datetime
    last_lat: Optional[float]
    last_lng: Optional[float]
    last_step_index: Optional[int]
    route: Optional[NavRoute]
    force_resume_requested: bool = False


class ForceResumeRequest(BaseModel):
    device_id: str
    clear: bool = False


# === in-memory 狀態 ===
devices: Dict[str, DeviceState] = {}


# === 工具函式 ===
def strip_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", "", html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(
        dlambda / 2.0
    ) ** 2
    c = 2.0 * math.asin(math.sqrt(a))
    return R * c


async def geocode_text_to_location(text: str) -> tuple[str, float, float]:
    if not GOOGLE_MAPS_API_KEY:
        raise HTTPException(
            status_code=500, detail="尚未設定 GOOGLE_MAPS_API_KEY 環境變數"
        )

    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": text,
        "region": "tw",
        "language": "zh-TW",
        "key": GOOGLE_MAPS_API_KEY,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, params=params)

    data = resp.json()
    status = data.get("status")
    if status != "OK":
        msg = data.get("error_message")
        raise HTTPException(
            status_code=500,
            detail=f"Geocoding 錯誤: {status}, msg={msg}",
        )

    results = data.get("results", [])
    if not results:
        raise HTTPException(status_code=404, detail="找不到相符的目的地地址")

    chosen = None
    for r in results:
        addr = r.get("formatted_address", "")
        if "台灣" in addr or "Taiwan" in addr:
            chosen = r
            break
    if chosen is None:
        chosen = results[0]

    formatted = chosen.get("formatted_address", text)
    loc = chosen.get("geometry", {}).get("location", {})
    lat = loc.get("lat")
    lng = loc.get("lng")
    if lat is None or lng is None:
        raise HTTPException(status_code=500, detail="Geocoding 結果缺少座標")

    return formatted, lat, lng


async def plan_route_with_google(
    origin_lat: float,
    origin_lng: float,
    destination_text: str,
    departure_time: Optional[int] = None,
) -> NavRoute:
    if not GOOGLE_MAPS_API_KEY:
        raise HTTPException(
            status_code=500, detail="尚未設定 GOOGLE_MAPS_API_KEY 環境變數"
        )

    resolved_dest_text, dest_lat, dest_lng = await geocode_text_to_location(
        destination_text
    )

    url = "https://maps.googleapis.com/maps/api/directions/json"
    params: Dict[str, Any] = {
        "origin": f"{origin_lat},{origin_lng}",
        "destination": f"{dest_lat},{dest_lng}",
        "mode": "transit",
        "transit_mode": "bus",
        "language": "zh-TW",
        "region": "tw",
        "key": GOOGLE_MAPS_API_KEY,
    }

    requested_dep_epoch: Optional[int] = None
    requested_dep_text: Optional[str] = None
    if departure_time is not None:
        params["departure_time"] = departure_time
        requested_dep_epoch = departure_time
        dt = datetime.datetime.fromtimestamp(departure_time)
        requested_dep_text = dt.strftime("%Y-%m-%d %H:%M")

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, params=params)

    print("=== Directions HTTP status:", resp.status_code)
    data = resp.json()
    print(
        "=== Directions summary ===",
        {"status": data.get("status"), "error_message": data.get("error_message")},
    )

    status = data.get("status")
    if status == "ZERO_RESULTS":
        raise HTTPException(
            status_code=404,
            detail="目前此時段查無公車路線，請嘗試不同出發時間或改用步行模式。",
        )
    if status != "OK":
        err_msg = data.get("error_message")
        raise HTTPException(
            status_code=500,
            detail=f"Google Directions API 錯誤: {status}, msg={err_msg}",
        )

    routes = data.get("routes", [])
    if not routes:
        raise HTTPException(status_code=500, detail="Directions 回傳資料異常，沒有 route。")

    chosen_route = None
    for r in routes:
        legs = r.get("legs", [])
        if not legs:
            continue
        steps_raw_candidate = legs[0].get("steps", [])
        if any(s.get("travel_mode") == "TRANSIT" for s in steps_raw_candidate):
            chosen_route = r
            break
    if chosen_route is None:
        chosen_route = routes[0]

    route0 = chosen_route
    leg0 = route0["legs"][0]
    steps_raw = leg0["steps"]

    start_addr = leg0.get("start_address")
    end_addr = leg0.get("end_address") or resolved_dest_text
    dur_s = leg0.get("duration", {}).get("value")
    total_duration_min = dur_s / 60.0 if dur_s is not None else None
    overview_polyline = route0.get("overview_polyline", {}).get("points")

    start_loc = leg0.get("start_location", {}) or {}
    dest_loc = leg0.get("end_location", {}) or {}

    steps: List[NavStep] = []
    idx = 0

    has_walk = False
    has_bus = False
    first_bus_wait_min: Optional[float] = None
    first_bus_number: Optional[str] = None

    for raw in steps_raw:
        mode = raw.get("travel_mode", "")

        if mode == "WALKING":
            has_walk = True
            substeps = raw.get("steps") or []
            if substeps:
                for sub in substeps:
                    html_inst = sub.get("html_instructions", "") or ""
                    inst_text = strip_html(html_inst) or "請步行至下一個路口。"
                    dist_m = sub.get("distance", {}).get("value")
                    dur_s2 = sub.get("duration", {}).get("value")
                    dur_min = dur_s2 / 60.0 if dur_s2 is not None else None
                    end_loc = sub.get("end_location", {}) or {}
                    tgt_lat = end_loc.get("lat")
                    tgt_lng = end_loc.get("lng")

                    steps.append(
                        NavStep(
                            index=idx,
                            mode="walk",
                            instruction=inst_text,
                            distance_m=dist_m,
                            duration_min=dur_min,
                            target_lat=tgt_lat,
                            target_lng=tgt_lng,
                        )
                    )
                    idx += 1
            else:
                html_inst = raw.get("html_instructions", "") or ""
                inst_text = strip_html(html_inst) or "請步行前往下一個地點。"
                dist_m = raw.get("distance", {}).get("value")
                dur_s2 = raw.get("duration", {}).get("value")
                dur_min = dur_s2 / 60.0 if dur_s2 is not None else None
                end_loc = raw.get("end_location", {}) or {}
                tgt_lat = end_loc.get("lat")
                tgt_lng = end_loc.get("lng")

                steps.append(
                    NavStep(
                        index=idx,
                        mode="walk",
                        instruction=inst_text,
                        distance_m=dist_m,
                        duration_min=dur_min,
                        target_lat=tgt_lat,
                        target_lng=tgt_lng,
                    )
                )
                idx += 1

        elif mode == "TRANSIT":
            has_bus = True
            html_inst = raw.get("html_instructions", "") or ""
            inst_text = strip_html(html_inst)

            td = raw.get("transit_details") or {}
            line = td.get("line") or {}
            bus_num = line.get("short_name") or line.get("name")

            dep_stop_obj = td.get("departure_stop") or {}
            arr_stop_obj = td.get("arrival_stop") or {}
            dep_stop_name = dep_stop_obj.get("name")
            arr_stop_name = arr_stop_obj.get("name")
            dep_stop_loc = dep_stop_obj.get("location") or {}
            dep_stop_lat = dep_stop_loc.get("lat")
            dep_stop_lng = dep_stop_loc.get("lng")

            dep_time = td.get("departure_time") or {}
            dep_ts = dep_time.get("value")
            dep_text = dep_time.get("text")

            if first_bus_number is None and bus_num:
                first_bus_number = bus_num
                if dep_ts is not None:
                    now_ts = int(time.time())
                    if dep_ts > now_ts:
                        first_bus_wait_min = (dep_ts - now_ts) / 60.0
                    else:
                        first_bus_wait_min = 0.0

            dist_m = raw.get("distance", {}).get("value")
            dur_s2 = raw.get("duration", {}).get("value")
            dur_min = dur_s2 / 60.0 if dur_s2 is not None else None
            end_loc = raw.get("end_location", {}) or {}
            tgt_lat = end_loc.get("lat")
            tgt_lng = end_loc.get("lng")

            if not inst_text:
                pieces = []
                if bus_num:
                    if dep_stop_name:
                        pieces.append(f"請在「{dep_stop_name}」搭乘公車 {bus_num}")
                    else:
                        pieces.append(f"請搭乘公車 {bus_num}")
                if arr_stop_name:
                    pieces.append(f"前往「{arr_stop_name}」")
                if pieces:
                    inst_text = "，".join(pieces) + "。"
                else:
                    inst_text = "請搭乘公車前往下一個站。"

            steps.append(
                NavStep(
                    index=idx,
                    mode="bus_ride",
                    instruction=inst_text,
                    distance_m=dist_m,
                    duration_min=dur_min,
                    bus_number=bus_num,
                    departure_stop=dep_stop_name,
                    arrival_stop=arr_stop_name,
                    target_lat=tgt_lat,
                    target_lng=tgt_lng,
                    bus_departure_timestamp=dep_ts,
                    bus_departure_text=dep_text,
                    bus_stop_lat=dep_stop_lat,
                    bus_stop_lng=dep_stop_lng,
                )
            )
            idx += 1

        else:
            html_inst = raw.get("html_instructions", "") or ""
            inst_text = strip_html(html_inst) or "請依照地圖指示移動。"
            dist_m = raw.get("distance", {}).get("value")
            dur_s2 = raw.get("duration", {}).get("value")
            dur_min = dur_s2 / 60.0 if dur_s2 is not None else None
            end_loc = raw.get("end_location", {}) or {}
            tgt_lat = end_loc.get("lat")
            tgt_lng = end_loc.get("lng")

            steps.append(
                NavStep(
                    index=idx,
                    mode=mode.lower() or "other",
                    instruction=inst_text,
                    distance_m=dist_m,
                    duration_min=dur_min,
                    target_lat=tgt_lat,
                    target_lng=tgt_lng,
                )
            )
            idx += 1

    dest_loc2 = leg0.get("end_location", {}) or {}
    steps.append(
        NavStep(
            index=idx,
            mode="arrive",
            instruction=f"已抵達目的地：{end_addr}。",
            target_lat=dest_loc2.get("lat"),
            target_lng=dest_loc2.get("lng"),
        )
    )

    summary_parts: List[str] = []
    if start_addr:
        summary_parts.append(f"目前位置在{start_addr}。")
    else:
        summary_parts.append("已取得 Google Maps 路線。")

    if total_duration_min is not None and end_addr:
        eta_min = round(total_duration_min)
        if eta_min <= 1:
            summary_parts.append(f"預計在一分鐘內抵達{end_addr}。")
        else:
            summary_parts.append(f"預計約 {eta_min} 分鐘後抵達{end_addr}。")
    elif end_addr:
        summary_parts.append(f"目的地為{end_addr}。")

    if has_walk:
        summary_parts.append("路程包含步行，會在每一段轉彎前給出距離提示。")
    if has_bus:
        summary_parts.append("中途會搭乘公車。")
        if first_bus_number:
            if first_bus_wait_min is not None:
                wait_min_round = round(first_bus_wait_min)
                if wait_min_round <= 0:
                    summary_parts.append(
                        f"第一班公車 {first_bus_number} 即將進站，請儘快前往站牌。"
                    )
                else:
                    summary_parts.append(
                        f"第一班公車 {first_bus_number} 大約 {wait_min_round} 分鐘後到站。"
                    )
            else:
                summary_parts.append(f"第一班公車為 {first_bus_number}。")
    else:
        summary_parts.append("目前 Google 傳回的路線為純步行路線，未包含公車資訊。")

    summary_text = "".join(summary_parts) if summary_parts else None

    return NavRoute(
        steps=steps,
        start_address=start_addr,
        end_address=end_addr,
        total_duration_min=total_duration_min,
        summary=summary_text,
        first_bus_wait_min=first_bus_wait_min,
        first_bus_number=first_bus_number,
        overview_polyline=overview_polyline,
        start_lat=start_loc.get("lat"),
        start_lng=start_loc.get("lng"),
        end_lat=dest_loc.get("lat"),
        end_lng=dest_loc.get("lng"),
        requested_departure_epoch=requested_dep_epoch,
        requested_departure_text=requested_dep_text,
    )


def get_default_route() -> NavRoute:
    demo_steps = [
        NavStep(
            index=0,
            mode="walk",
            instruction="請從清華大學校門口往前直走一百公尺，到公車站牌。",
            distance_m=100,
            target_lat=24.796123,
            target_lng=120.9935,
        ),
        NavStep(
            index=1,
            mode="bus_wait",
            instruction="你現在在清華大學公車站牌，等待公車五六八到站。",
        ),
        NavStep(
            index=2,
            mode="bus_ride",
            instruction="搭乘五六八公車，坐六站，在新竹火車站前站下車。",
            bus_number="568",
            bus_stop_lat=24.796123,
            bus_stop_lng=120.9935,
        ),
        NavStep(
            index=3,
            mode="walk",
            instruction="下車後往前步行一百五十公尺，目的地在你的右手邊。",
            distance_m=150,
        ),
        NavStep(index=4, mode="arrive", instruction="已經到達新竹火車站。"),
    ]

    summary = (
        "目前位置在國立清華大學附近。預計約二十五分鐘後抵達新竹火車站。"
        "中途會先步行到公車站，搭乘五六八公車六站，最後再步行一小段。"
        "第一班公車五六八預計約五分鐘後抵達站牌。"
    )

    return NavRoute(
        steps=demo_steps,
        start_address="國立清華大學",
        end_address="新竹火車站",
        total_duration_min=25.0,
        summary=summary,
        first_bus_wait_min=5.0,
        first_bus_number="568",
        overview_polyline=None,
        start_lat=24.796123,
        start_lng=120.9935,
        end_lat=24.8017,
        end_lng=120.9715,
        requested_departure_epoch=None,
        requested_departure_text=None,
    )


# === FastAPI 路由 ===
@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/monitor")
async def monitor():
    return FileResponse("static/monitor.html")

@app.get("/bus_monitor")
async def bus_monitor():
    return FileResponse("static/bus_monitor.html")


@app.get("/video_test")
async def video_test():
    return FileResponse("static/video_test.html")


@app.post("/traffic_light/start")
async def traffic_light_start():
    return await start_pedestrian_processing()


@app.get("/traffic_light/progress")
async def traffic_light_progress():
    return get_pedestrian_progress()


@app.get("/traffic_light/video/{filename}")
async def traffic_light_video(filename: str):
    return get_pedestrian_video(filename)


@app.get("/traffic_light/live_preview")
async def traffic_light_live_preview():
    return get_pedestrian_live_preview()


@app.post("/register_device")
async def register_device(req: RegisterDeviceRequest):
    now = datetime.datetime.utcnow()
    dev = devices.get(req.device_id)
    if dev is None:
        dev = DeviceState(
            device_id=req.device_id, created_at=now, last_seen=now, route=None
        )
        devices[req.device_id] = dev
    else:
        dev.last_seen = now
    return {"status": "ok", "device_id": req.device_id}


@app.post("/device_state/reset")
async def reset_device_state(req: ResetDeviceRequest):
    dev = devices.get(req.device_id)
    if dev is None:
        raise HTTPException(status_code=404, detail="找不到指定的裝置")
    dev.route = None
    dev.last_step_index = None
    dev.last_lat = DEFAULT_ORIGIN["lat"]
    dev.last_lng = DEFAULT_ORIGIN["lng"]
    dev.last_seen = datetime.datetime.utcnow()
    dev.force_resume_requested = False
    return {
        "status": "ok",
        "device_id": req.device_id,
        "lat": dev.last_lat,
        "lng": dev.last_lng,
    }


@app.post("/route")
async def route_demo(req: RouteRequest):
    dev = devices.get(req.device_id)
    if dev is None:
        raise HTTPException(status_code=400, detail="裝置尚未註冊，請先呼叫 /register_device")

    nav_route = get_default_route()
    dev.route = nav_route
    return {"route": nav_route}


@app.post("/route_google")
async def route_google(req: GoogleRouteRequest):
    dev = devices.get(req.device_id)
    if dev is None:
        raise HTTPException(status_code=400, detail="裝置尚未註冊，請先呼叫 /register_device")

    nav_route = await plan_route_with_google(
        origin_lat=req.origin_lat,
        origin_lng=req.origin_lng,
        destination_text=req.destination,
        departure_time=req.departure_time,
    )

    dev.route = nav_route
    return {"route": nav_route}


@app.post("/update_location", response_model=UpdateLocationResponse)
async def update_location(req: UpdateLocationRequest):
    dev = devices.get(req.device_id)
    if dev is None or dev.route is None:
        return UpdateLocationResponse(distance_to_target_m=None, message=None)

    dev.last_seen = datetime.datetime.utcnow()
    dev.last_lat = req.lat
    dev.last_lng = req.lng
    dev.last_step_index = req.step_index

    if req.step_index < 0 or req.step_index >= len(dev.route.steps):
        return UpdateLocationResponse(distance_to_target_m=None, message=None)

    step = dev.route.steps[req.step_index]
    if step.target_lat is None or step.target_lng is None:
        return UpdateLocationResponse(distance_to_target_m=None, message=None)

    d = haversine(req.lat, req.lng, step.target_lat, step.target_lng)
    return UpdateLocationResponse(distance_to_target_m=d, message=None)


@app.post("/force_resume")
async def force_resume(req: ForceResumeRequest):
    dev = devices.get(req.device_id)
    if dev is None:
        raise HTTPException(status_code=404, detail="找不到指定的裝置")
    if req.clear:
        dev.force_resume_requested = False
        return {"status": "cleared", "force_resume_requested": dev.force_resume_requested}
    dev.force_resume_requested = True
    return {"status": "requested", "force_resume_requested": dev.force_resume_requested}


@app.get("/force_resume/{device_id}")
async def force_resume_status(device_id: str, clear: bool = Query(False)):
    dev = devices.get(device_id)
    if dev is None:
        raise HTTPException(status_code=404, detail="找不到指定的裝置")
    flag = dev.force_resume_requested
    if flag and clear:
        dev.force_resume_requested = False
    return {"device_id": device_id, "force_resume_requested": flag}


@app.get("/device_state/{device_id}", response_model=DeviceSnapshot)
async def get_device_state(device_id: str):
    dev = devices.get(device_id)
    if dev is None:
        raise HTTPException(status_code=404, detail="找不到指定的裝置")
    return DeviceSnapshot(
        device_id=dev.device_id,
        last_seen=dev.last_seen,
        last_lat=dev.last_lat,
        last_lng=dev.last_lng,
        last_step_index=dev.last_step_index,
        route=dev.route,
        force_resume_requested=dev.force_resume_requested,
    )


# === Bus vision / 公車前方 LED 辨識 ===

@app.post("/bus_vision/start")
async def api_bus_vision_start(
    use_mobile: Optional[str] = Query(None, description="是否使用手機攝影機（'true' 或 'false'）"),
    video_file_path: Optional[str] = Query(None, description="影片檔案路徑"),
    start_sec: float = Query(0.0, description="影片開始時間（秒）"),
    end_sec: Optional[float] = Query(None, description="影片結束時間（秒）")
):
    """
    啟動公車辨識系統
    
    Args:
        use_mobile: 是否使用手機攝影機（字串 "true" 或 "false"，預設 False）
        video_file_path: 影片檔案路徑（如果提供，會優先使用影片檔案而非手機或電腦攝影機）
        start_sec: 影片開始時間（秒）
        end_sec: 影片結束時間（秒）
    """
    # 處理 use_mobile 參數（可能是字串 "true"/"false" 或布林值）
    use_mobile_bool = False
    if use_mobile is not None:
        if isinstance(use_mobile, str):
            use_mobile_bool = use_mobile.lower() == "true"
        else:
            use_mobile_bool = bool(use_mobile)
    
    started = start_bus_vision(
        use_mobile=use_mobile_bool, 
        video_file_path=video_file_path,
        start_sec=start_sec,
        end_sec=end_sec
    )
    status = "started" if started else "already_running"
    
    # 獲取實際使用的模式（從 bus_vision 模組）
    from app.bus_vision import _use_video_file, _video_file_path, _use_mobile_camera
    
    # 確定實際使用的模式
    if _use_video_file and _video_file_path:
        actual_mode = "video_file"
        actual_video_path = _video_file_path
        actual_use_mobile = False
    elif _use_mobile_camera:
        actual_mode = "mobile"
        actual_video_path = None
        actual_use_mobile = True
    else:
        actual_mode = "camera"
        actual_video_path = None
        actual_use_mobile = False
    
    return {
        "status": status,
        "use_mobile": actual_use_mobile,
        "video_file_path": actual_video_path,
        "mode": actual_mode
    }


@app.post("/bus_vision/stop")
async def api_bus_vision_stop():
    stop_bus_vision()
    return {"status": "stopped"}


@app.post("/bus_vision/reset")
async def api_bus_vision_reset():
    """重置所有公車辨識狀態和快取"""
    reset_bus_vision()
    return {"status": "reset"}


@app.get("/bus_vision/status")
async def api_bus_vision_status():
    # 回傳最近一次辨識結果（可能還沒有任何結果）
    return get_bus_status()


@app.get("/bus_vision/stream")
async def api_bus_vision_stream():
    # 注意：只有在 start_bus_vision() 之後，才會真的跑模型
    return StreamingResponse(
        bus_video_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/bus_vision/mobile_frame")
async def api_bus_vision_mobile_frame(file: UploadFile = File(...)):
    """接收來自手機的視訊 frame"""
    try:
        frame_bytes = await file.read()
        success = receive_mobile_frame(frame_bytes)
        return {"status": "ok" if success else "error"}
    except Exception as e:
        print(f"[main] 接收手機 frame 錯誤: {e}")
        return {"status": "error", "message": str(e)}
