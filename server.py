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

# Earth Engine Asset IDs
MODIS_ASSET = "FIRMS"
VIIRS_SNPP_ASSET = "NASA/LANCE/SNPP_VIIRS/C2"
VIIRS_NOAA20_ASSET = "NASA/LANCE/NOAA20_VIIRS/C2"

def clamp_bbox(west: float, south: float, east: float, north: float):
    c_lon = (west + east) / 2.0
    c_lat = (south + north) / 2.0
    half_w = min(abs(east - west) / 2.0, MAX_SPAN_DEG / 2.0)
    half_h = min(abs(north - south) / 2.0, MAX_SPAN_DEG / 2.0)
    return round(c_lon - half_w, 3), round(c_lat - half_h, 3), round(c_lon + half_w, 3), round(c_lat + half_h, 3)

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

def get_combined_viirs_image(start_date, end_date, region):
    """Merges Suomi-NPP and NOAA-20 375m VIIRS collections with nominal/high confidence filtering."""
    snpp = ee.ImageCollection(VIIRS_SNPP_ASSET).filterDate(start_date, end_date).filterBounds(region)
    noaa20 = ee.ImageCollection(VIIRS_NOAA20_ASSET).filterDate(start_date, end_date).filterBounds(region)
    merged = snpp.merge(noaa20)

    def mask_viirs(img):
        bands = img.bandNames()
        has_frp = bands.contains('FRP')
        has_ti4 = bands.contains('Bright_ti4')
        has_conf = bands.contains('confidence')

        def build():
            frp = ee.Image(ee.Algorithms.If(
                has_frp,
                img.select(['FRP']),
                img.select(['Bright_ti4']).subtract(300).max(0).multiply(3.1)
            )).rename('frp')

            # Filter out low confidence (1 = low, 2 = nominal, 3 = high) and weak thermal noise
            valid_mask = ee.Image(ee.Algorithms.If(
                has_conf,
                img.select(['confidence']).gte(2),
                img.select(['Bright_ti4']).gt(310)
            ))
            return frp.updateMask(valid_mask).updateMask(frp.gt(0)).toFloat()

        empty = ee.Image.constant(0).rename('frp').updateMask(ee.Image.constant(0)).toFloat()
        return ee.Image(ee.Algorithms.If(has_ti4, build(), empty))

    return merged.map(mask_viirs).max().clip(region)

def get_modis_image(start_date, end_date, region):
    """Extracts Terra & Aqua 1km MODIS hotspots with confidence >= 30% and harmonizes FRP."""
    modis_col = ee.ImageCollection(MODIS_ASSET).filterDate(start_date, end_date).filterBounds(region)

    def mask_modis(img):
        bands = img.bandNames()
        has_t21 = bands.contains('T21')
        has_conf = bands.contains('confidence')

        def build():
            t21 = img.select(['T21'])
            conf_mask = ee.Image(ee.Algorithms.If(
                has_conf,
                img.select(['confidence']).gte(30),
                t21.gt(310)
            ))
            fire_mask = t21.gt(305).And(conf_mask)
            frp = t21.subtract(300).max(0).multiply(2.5 * MODIS_TO_VIIRS_FACTOR).rename('frp')
            return frp.updateMask(fire_mask).toFloat()

        empty = ee.Image.constant(0).rename('frp').updateMask(ee.Image.constant(0)).toFloat()
        return ee.Image(ee.Algorithms.If(has_t21, build(), empty))

    return modis_col.map(mask_modis).max().clip(region)

@lru_cache(maxsize=256)
def cached_month_risk(year: int, month: int, west: float, south: float, east: float, north: float):
    region = ee.Geometry.Rectangle([west, south, east, north])
    days_in_month = calendar.monthrange(year, month)[1]
    month_start = ee.Date.fromYMD(year, month, 1)
    month_end = month_start.advance(1, 'month')

    is_pre_viirs = (year < 2012) or (year == 2012 and month == 1)
    scale = 1000 if is_pre_viirs else 375

    day_indices = ee.List.sequence(0, days_in_month - 1)

    def summarize_day(offset):
        d_start = month_start.advance(offset, 'day')
        d_end = d_start.advance(1, 'day')

        if is_pre_viirs:
            fire_img = get_modis_image(d_start, d_end, region)
        else:
            viirs_img = get_combined_viirs_image(d_start, d_end, region)
            modis_img = get_modis_image(d_start, d_end, region)
            # Combine VIIRS + MODIS so fires missed by one satellite pass are caught by the other
            fire_img = ee.ImageCollection([viirs_img, modis_img]).max().clip(region)

        stats = fire_img.reduceRegion(
            reducer=ee.Reducer.count().combine(ee.Reducer.sum(), sharedInputs=True),
            geometry=region,
            scale=scale,
            maxPixels=1e6,
            bestEffort=False
        )
        return ee.Feature(None, {
            'date': d_start.format('YYYY-MM-dd'),
            'count': stats.get('frp_count'),
            'total_frp': stats.get('frp_sum')
        })

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
    hotspots = []

    # 1. Sample 375m VIIRS (Suomi-NPP + NOAA-20) when available
    viirs_mask_for_dedup = None
    if not is_pre_viirs:
        viirs_img = get_combined_viirs_image(start_date, end_date, region)
        viirs_mask_for_dedup = viirs_img.mask()

        viirs_samples = viirs_img.sample(
            region=region,
            scale=375,
            numPixels=1500,
            geometries=True
        ).getInfo()

        for feat in viirs_samples.get('features', []):
            props = feat.get('properties', {})
            if props.get('frp') is not None:
                lon, lat = feat['geometry']['coordinates']
                hotspots.append({
                    "lat": round(lat, 5),
                    "lon": round(lon, 5),
                    "cell_deg": 0.0035,
                    "harmonized_frp": round(float(props['frp']), 2),
                    "sensor": "VIIRS 375m (SNPP/NOAA-20)"
                })

    # 2. Sample 1km MODIS (Terra & Aqua), masking out areas already covered by 375m VIIRS
    modis_img = get_modis_image(start_date, end_date, region)
    if viirs_mask_for_dedup is not None:
        # Focal max creates a ~500m buffer around VIIRS detections to prevent duplicate overlapping squares
        viirs_coverage = viirs_mask_for_dedup.focal_max(radius=500, units='meters').unmask(0)
        modis_img = modis_img.updateMask(viirs_coverage.Not())

    modis_samples = modis_img.sample(
        region=region,
        scale=1000,
        numPixels=1000,
        geometries=True
    ).getInfo()

    for feat in modis_samples.get('features', []):
        props = feat.get('properties', {})
        if props.get('frp') is not None:
            lon, lat = feat['geometry']['coordinates']
            hotspots.append({
                "lat": round(lat, 5),
                "lon": round(lon, 5),
                "cell_deg": 0.009,
                "harmonized_frp": round(float(props['frp']), 2),
                "sensor": "MODIS 1km (Terra/Aqua)"
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

    return {
        "start_date": d1.strftime("%Y-%m-%d"),
        "end_date": d2.strftime("%Y-%m-%d"),
        "bbox": [west, south, east, north],
        "days_count": len(timeline),
        "timeline": timeline
    }

@app.get("/api/hotspots_range")
@limiter.limit("10/minute")
def get_hotspots_range(request: Request, start_date: str, end_date: str, west: float, south: float, east: float, north: float):
    try:
        w, s, e, n = clamp_bbox(west, south, east, north)
        return cached_hotspots_range(start_date, end_date, w, s, e, n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@lru_cache(maxsize=128)
def cached_fire_events(west: float, south: float, east: float, north: float):
    region = ee.Geometry.Rectangle([west, south, east, north])
    today_str = datetime.utcnow().strftime("%Y-%m-%d")

    modis_col = ee.ImageCollection(MODIS_ASSET).filterDate("2000-11-01", today_str).filterBounds(region)
    snpp_col = ee.ImageCollection(VIIRS_SNPP_ASSET).filterDate("2012-01-20", today_str).filterBounds(region)
    noaa20_col = ee.ImageCollection(VIIRS_NOAA20_ASSET).filterDate("2018-01-01", today_str).filterBounds(region)

    def check_modis(img):
        has_t21 = img.bandNames().contains('T21')
        def compute():
            t21 = img.select(['T21'])
            conf = ee.Image(ee.Algorithms.If(img.bandNames().contains('confidence'), img.select(['confidence']).gte(30), t21.gt(310)))
            fire_mask = t21.gt(305).And(conf)
            frp_img = t21.subtract(300).max(0).multiply(2.5 * MODIS_TO_VIIRS_FACTOR).updateMask(fire_mask)
            stats = frp_img.reduceRegion(
                reducer=ee.Reducer.count().combine(ee.Reducer.sum(), sharedInputs=True),
                geometry=region, scale=1000, maxPixels=1e6, bestEffort=False
            )
            return ee.Feature(None, {
                'date': img.date().format('YYYY-MM-dd'),
                'count': stats.get('T21_count'),
                'harm_frp': stats.get('T21_sum')
            })
        empty = ee.Feature(None, {'date': img.date().format('YYYY-MM-dd'), 'count': 0, 'harm_frp': 0})
        return ee.Algorithms.If(has_t21, compute(), empty)

    def check_viirs(img):
        bands = img.bandNames()
        has_ti4 = bands.contains('Bright_ti4')
        def compute():
            frp = ee.Image(ee.Algorithms.If(
                bands.contains('FRP'),
                img.select(['FRP']),
                img.select(['Bright_ti4']).subtract(300).max(0).multiply(3.1)
            )).rename('frp')
            conf = ee.Image(ee.Algorithms.If(bands.contains('confidence'), img.select(['confidence']).gte(2), img.select(['Bright_ti4']).gt(310)))
            frp_img = frp.updateMask(conf).updateMask(frp.gt(0))
            stats = frp_img.reduceRegion(
                reducer=ee.Reducer.count().combine(ee.Reducer.sum(), sharedInputs=True),
                geometry=region, scale=375, maxPixels=1e6, bestEffort=False
            )
            return ee.Feature(None, {
                'date': img.date().format('YYYY-MM-dd'),
                'count': stats.get('frp_count'),
                'harm_frp': stats.get('frp_sum')
            })
        empty = ee.Feature(None, {'date': img.date().format('YYYY-MM-dd'), 'count': 0, 'harm_frp': 0})
        return ee.Algorithms.If(has_ti4, compute(), empty)

    fc_modis = ee.FeatureCollection(modis_col.map(check_modis)).filter(ee.Filter.gt('count', 0))
    fc_snpp = ee.FeatureCollection(snpp_col.map(check_viirs)).filter(ee.Filter.gt('count', 0))
    fc_noaa20 = ee.FeatureCollection(noaa20_col.map(check_viirs)).filter(ee.Filter.gt('count', 0))

    all_hits = fc_modis.merge(fc_snpp).merge(fc_noaa20).getInfo()

    daily_hits = {}
    for feat in all_hits.get('features', []):
        p = feat.get('properties', {})
        d_str = p.get('date')
        cnt = int(p.get('count') or 0)
        harm_frp = float(p.get('harm_frp') or 0.0)
        if not d_str or cnt <= 0:
            continue

        if d_str not in daily_hits:
            daily_hits[d_str] = {"count": 0, "frp": 0.0}
        daily_hits[d_str]["count"] = max(daily_hits[d_str]["count"], cnt)
        daily_hits[d_str]["frp"] = max(daily_hits[d_str]["frp"], harm_frp)

    sorted_dates = sorted(daily_hits.keys())
    events = []
    curr_event = None

    for d_str in sorted_dates:
        dt = datetime.strptime(d_str, "%Y-%m-%d")
        info = daily_hits[d_str]

        if curr_event is None:
            curr_event = {
                "start_date": d_str,
                "end_date": d_str,
                "last_dt": dt,
                "peak_count": info["count"],
                "total_frp": info["frp"]
            }
        else:
            gap_days = (dt - curr_event["last_dt"]).days
            if gap_days <= 3:
                curr_event["end_date"] = d_str
                curr_event["last_dt"] = dt
                curr_event["peak_count"] = max(curr_event["peak_count"], info["count"])
                curr_event["total_frp"] += info["frp"]
            else:
                events.append(curr_event)
                curr_event = {
                    "start_date": d_str,
                    "end_date": d_str,
                    "last_dt": dt,
                    "peak_count": info["count"],
                    "total_frp": info["frp"]
                }

    if curr_event is not None:
        events.append(curr_event)

    formatted_events = []
    for ev in reversed(events):
        d1 = datetime.strptime(ev["start_date"], "%Y-%m-%d")
        d2 = datetime.strptime(ev["end_date"], "%Y-%m-%d")
        duration = (d2 - d1).days + 1
        risk = classify_risk(ev["peak_count"], ev["total_frp"])
        formatted_events.append({
            "start_date": ev["start_date"],
            "end_date": ev["end_date"],
            "duration_days": duration,
            "peak_count": ev["peak_count"],
            "total_frp": round(ev["total_frp"], 1),
            "risk_level": risk.upper()
        })

    return {"events": formatted_events}

@app.get("/api/fire_events")
@limiter.limit("12/minute")
def get_fire_events(request: Request, west: float, south: float, east: float, north: float):
    try:
        w, s, e, n = clamp_bbox(west, south, east, north)
        return cached_fire_events(w, s, e, n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))