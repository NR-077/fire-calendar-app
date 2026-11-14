import os
import json
import calendar
from datetime import datetime, timedelta
from functools import lru_cache
import ee
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Authenticate Earth Engine via Cloud Service Account (when deployed) or local credentials
try:
    if "EE_SERVICE_ACCOUNT_JSON" in os.environ:
        sa_info = json.loads(os.environ["EE_SERVICE_ACCOUNT_JSON"])
        credentials = ee.ServiceAccountCredentials(sa_info["client_email"], key_data=json.dumps(sa_info))
        ee.Initialize(credentials, project="fire-calendar-app")
    else:
        ee.Initialize(project="fire-calendar-app")
except Exception as e:
    print("Earth Engine init error:", e)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)

MODIS_TO_VIIRS_FACTOR = 1.35
MAX_SPAN_DEG = 0.45

def clamp_bbox(west: float, south: float, east: float, north: float):
    c_lon = (west + east) / 2.0
    c_lat = (south + north) / 2.0
    half_w = min(abs(east - west) / 2.0, MAX_SPAN_DEG / 2.0)
    half_h = min(abs(north - south) / 2.0, MAX_SPAN_DEG / 2.0)
    return round(c_lon - half_w, 3), round(c_lat - half_h, 3), round(c_lon + half_w, 3), round(c_lat + half_h, 3)

def map_landcover_to_fuel(lc_code: int) -> str:
    if lc_code == 10:
        return "dense_forest"
    elif lc_code in (20, 95):
        return "light_forest"
    elif lc_code in (30, 40, 90):
        return "grass_brush"
    elif lc_code == 80:
        return "water"
    else:
        return "barren_urban"

def classify_risk(count: int, total_frp: float) -> str:
    if count == 0:
        return "none"
    elif count < 15 and total_frp < 1500:
        return "low"
    elif count < 60 and total_frp < 8000:
        return "moderate"
    elif count < 150 and total_frp < 25000:
        return "high"
    else:
        return "extreme"

@lru_cache(maxsize=256)
def cached_month_risk(year: int, month: int, west: float, south: float, east: float, north: float):
    region = ee.Geometry.Rectangle([west, south, east, north])
    days_in_month = calendar.monthrange(year, month)[1]
    month_start = ee.Date.fromYMD(year, month, 1)
    month_end = month_start.advance(1, 'month')

    is_pre_viirs = (year < 2012) or (year == 2012 and month == 1)
    scale = 1000 if is_pre_viirs else 375
    frp_mult = (2.5 * MODIS_TO_VIIRS_FACTOR) if is_pre_viirs else 3.1

    col = ee.ImageCollection("FIRMS").filterDate(month_start, month_end).filterBounds(region)
    if col.size().getInfo() == 0:
        return {"year": year, "month": month, "risk_days": {}}

    day_indices = ee.List.sequence(0, days_in_month - 1)

    def summarize_day(offset):
        d_start = month_start.advance(offset, 'day')
        d_end = d_start.advance(1, 'day')
        day_col = col.filterDate(d_start, d_end)
        has_imgs = day_col.size().gt(0)

        def compute_stats():
            img = day_col.max().clip(region)
            has_t21 = img.bandNames().contains('T21')

            def calc():
                t21 = img.select(['T21'])
                fire_mask = t21.gt(300)
                frp_img = t21.subtract(300).max(0).multiply(frp_mult).updateMask(fire_mask)
                stats = frp_img.reduceRegion(
                    reducer=ee.Reducer.count().combine(ee.Reducer.sum(), sharedInputs=True),
                    geometry=region,
                    scale=scale,
                    maxPixels=1e6,
                    bestEffort=False
                )
                return ee.Feature(None, {
                    'date': d_start.format('YYYY-MM-dd'),
                    'count': stats.get('T21_count'),
                    'total_frp': stats.get('T21_sum')
                })

            empty = ee.Feature(None, {'date': d_start.format('YYYY-MM-dd'), 'count': 0, 'total_frp': 0})
            return ee.Algorithms.If(has_t21, calc(), empty)

        empty_feat = ee.Feature(None, {'date': d_start.format('YYYY-MM-dd'), 'count': 0, 'total_frp': 0})
        return ee.Algorithms.If(has_imgs, compute_stats(), empty_feat)

    fc = ee.FeatureCollection(day_indices.map(summarize_day)).getInfo()
    risk_map = {}
    for feat in fc.get('features', []):
        props = feat.get('properties', {})
        date_str = props.get('date')
        count = min(int(props.get('count') or 0), 1500)
        total_frp = float(props.get('total_frp') or 0.0)
        level = classify_risk(count, total_frp)
        if level != "none":
            risk_map[date_str] = level

    return {"year": year, "month": month, "risk_days": risk_map}

@app.get("/api/month_risk")
@limiter.limit("20/minute")
def get_month_risk(request: Request, year: int, month: int, west: float, south: float, east: float, north: float):
    try:
        w, s, e, n = clamp_bbox(west, south, east, north)
        return cached_month_risk(year, month, w, s, e, n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def fetch_single_day_hotspots(date_str: str, region):
    start_date = ee.Date(date_str)
    end_date = start_date.advance(1, 'day')
    is_pre_viirs = date_str < "2012-01-20"
    sample_scale = 1000 if is_pre_viirs else 375
    cell_deg = 0.009 if is_pre_viirs else 0.0035

    firms_img = ee.ImageCollection("FIRMS") \
        .filterDate(start_date, end_date) \
        .filterBounds(region) \
        .max().clip(region)

    if 'T21' not in firms_img.bandNames().getInfo():
        return []

    t21 = firms_img.select(['T21'])
    fire_only = t21.updateMask(t21.gt(300))

    samples = fire_only.sample(
        region=region,
        scale=sample_scale,
        numPixels=1500,
        geometries=True
    ).getInfo()

    hotspots = []
    for feat in samples.get('features', []):
        props = feat.get('properties', {})
        if props.get('T21') is not None:
            lon, lat = feat['geometry']['coordinates']
            if is_pre_viirs:
                frp_val = round(max(0, (props['T21'] - 300) * 2.5) * MODIS_TO_VIIRS_FACTOR, 2)
                sensor = "MODIS (Harmonized)"
            else:
                frp_val = round(max(0, (props['T21'] - 300) * 3.1), 2)
                sensor = "VIIRS/FIRMS"

            hotspots.append({
                "lat": round(lat, 5),
                "lon": round(lon, 5),
                "cell_deg": cell_deg,
                "harmonized_frp": frp_val,
                "sensor": sensor
            })
    return hotspots

@lru_cache(maxsize=128)
def cached_hotspots_range(start_date: str, end_date: str, west: float, south: float, east: float, north: float):
    d1 = datetime.strptime(start_date, "%Y-%m-%d")
    d2 = datetime.strptime(end_date, "%Y-%m-%d")
    if d2 < d1:
        d1, d2 = d2, d1
    if (d2 - d1).days > 13:
        d2 = d1 + timedelta(days=13)

    region = ee.Geometry.Rectangle([west, south, east, north])
    timeline = []
    prev_count = None

    curr = d1
    while curr <= d2:
        day_str = curr.strftime("%Y-%m-%d")
        spots = fetch_single_day_hotspots(day_str, region)
        count = len(spots)
        total_frp = round(sum(s["harmonized_frp"] for s in spots), 1)
        delta = 0 if prev_count is None else (count - prev_count)
        prev_count = count

        timeline.append({
            "date": day_str,
            "count": count,
            "delta_from_prev_day": delta,
            "total_frp": total_frp,
            "risk_level": classify_risk(count, total_frp),
            "hotspots": spots
        })
        curr += timedelta(days=1)

    landcover = ee.ImageCollection("ESA/WorldCover/v200").first().clip(region)
    rows, cols = 30, 30
    lat_step = (north - south) / rows
    lon_step = (east - west) / cols

    features = []
    for r in range(rows):
        for c in range(cols):
            pt_lat = south + (r + 0.5) * lat_step
            pt_lon = west + (c + 0.5) * lon_step
            features.append(ee.Feature(ee.Geometry.Point([pt_lon, pt_lat]), {"r": r, "c": c}))

    sampled_lc = landcover.reduceRegions(
        collection=ee.FeatureCollection(features),
        reducer=ee.Reducer.first(),
        scale=100
    ).getInfo()

    terrain_grid = [["grass_brush" for _ in range(cols)] for _ in range(rows)]
    for f in sampled_lc.get('features', []):
        p = f['properties']
        code = p.get('first')
        if code is not None:
            terrain_grid[p['r']][p['c']] = map_landcover_to_fuel(int(code))

    return {
        "start_date": d1.strftime("%Y-%m-%d"),
        "end_date": d2.strftime("%Y-%m-%d"),
        "bbox": [west, south, east, north],
        "days_count": len(timeline),
        "timeline": timeline,
        "grid_size": [rows, cols],
        "terrain_grid": terrain_grid
    }

@app.get("/api/hotspots_range")
@limiter.limit("10/minute")
def get_hotspots_range(request: Request, start_date: str, end_date: str, west: float, south: float, east: float, north: float):
    try:
        w, s, e, n = clamp_bbox(west, south, east, north)
        return cached_hotspots_range(start_date, end_date, w, s, e, n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))