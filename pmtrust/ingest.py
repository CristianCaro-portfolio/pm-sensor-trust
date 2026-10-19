"""Download raw data from Sensor.Community and the UBA air data API.

Everything lands untouched under data/raw so a run can be repeated offline and the
bronze layer can always be rebuilt from the original files.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

SC_API = "https://data.sensor.community/airrohr/v1/filter/area={lat},{lon},{radius}"
SC_ARCHIVE = "https://archive.sensor.community/{day}/{day}_{kind}_sensor_{sid}.csv"
UBA_API = (
    "https://luftdaten.umweltbundesamt.de/api/air-data/v3/measures/json"
    "?date_from={start}&time_from=1&date_to={end}&time_to=24"
    "&station={station}&component={component}&scope=2"
)
USER_AGENT = "pm-sensor-trust/0.1 (research; github.com/CristianCaro-portfolio)"
PM_TYPES = {"SDS011"}
CLIMATE_TYPES = {"DHT22", "BME280"}
UBA_COMPONENTS = {"pm10": 1, "pm25": 9}
DWD_TU = (
    "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/"
    "hourly/air_temperature/recent/stundenwerte_TU_{station:05d}_akt.zip"
)


@dataclass(frozen=True)
class FetchResult:
    url: str
    path: Path | None
    status: str  # ok, cached, missing, error


def http_get(url: str, retries: int = 3, timeout: int = 60) -> bytes | None:
    """GET with bounded retries. Returns None on 404, raises after the last failure."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
        time.sleep(2**attempt)
    return None


def daterange(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def discover_sensors(lat: float, lon: float, radius_km: float, out: Path) -> list[dict]:
    """Snapshot of the sensors reporting inside the area right now.

    The live API only knows sensors that reported in the last five minutes, so the
    snapshot is saved and reused; a rerun months later works on the same sensor list.
    """
    if out.exists():
        return json.loads(out.read_text())
    payload = http_get(SC_API.format(lat=lat, lon=lon, radius=radius_km))
    if payload is None:
        raise RuntimeError("Sensor.Community area query returned 404")
    seen: dict[int, dict] = {}
    for rec in json.loads(payload):
        kind = rec["sensor"]["sensor_type"]["name"]
        if kind not in PM_TYPES | CLIMATE_TYPES:
            continue
        loc = rec["location"]
        seen[rec["sensor"]["id"]] = {
            "sensor_id": rec["sensor"]["id"],
            "sensor_type": kind,
            "location_id": loc["id"],
            "lat": float(loc["latitude"]),
            "lon": float(loc["longitude"]),
            "indoor": int(loc["indoor"]),
        }
    sensors = sorted(seen.values(), key=lambda s: s["sensor_id"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sensors, indent=1))
    return sensors


def paired_sensors(sensors: list[dict]) -> list[dict]:
    """Keep outdoor locations that have both a PM sensor and a humidity sensor."""
    by_loc: dict[int, list[dict]] = {}
    for s in sensors:
        if not s["indoor"]:
            by_loc.setdefault(s["location_id"], []).append(s)
    keep = []
    for group in by_loc.values():
        kinds = {s["sensor_type"] for s in group}
        if kinds & PM_TYPES and kinds & CLIMATE_TYPES:
            keep.extend(group)
    return keep


def _fetch_one(day: date, sensor: dict, raw_dir: Path) -> FetchResult:
    kind = sensor["sensor_type"].lower()
    url = SC_ARCHIVE.format(day=day.isoformat(), kind=kind, sid=sensor["sensor_id"])
    target = raw_dir / "sensor_community" / day.isoformat() / url.rsplit("/", 1)[1]
    missing = target.with_suffix(".missing")
    if target.exists():
        return FetchResult(url, target, "cached")
    if missing.exists():
        return FetchResult(url, None, "missing")
    try:
        body = http_get(url)
    except Exception as exc:  # network trouble is logged, the run continues
        log.warning("failed %s: %s", url, exc)
        return FetchResult(url, None, "error")
    target.parent.mkdir(parents=True, exist_ok=True)
    if body is None:
        missing.touch()
        return FetchResult(url, None, "missing")
    target.write_bytes(body)
    return FetchResult(url, target, "ok")


def fetch_sensor_days(sensors: list[dict], start: date, end: date, raw_dir: Path, workers: int = 8) -> dict[str, int]:
    jobs = [(d, s) for d in daterange(start, end) for s in sensors]
    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(lambda job: _fetch_one(job[0], job[1], raw_dir), jobs):
            counts[res.status] = counts.get(res.status, 0) + 1
    log.info("sensor files %s..%s: %s", start, end, counts)
    return counts


def fetch_reference(station: int, start: date, end: date, raw_dir: Path) -> list[Path]:
    paths = []
    for name, comp in UBA_COMPONENTS.items():
        target = raw_dir / "uba" / f"{station}_{name}_{start}_{end}.json"
        if not target.exists():
            body = http_get(UBA_API.format(start=start, end=end, station=station, component=comp))
            if body is None:
                raise RuntimeError(f"UBA returned 404 for {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
        paths.append(target)
    return paths


def fetch_weather(station: int, raw_dir: Path) -> Path:
    """Hourly temperature and humidity from a DWD synoptic station (values in UTC).

    The "recent" product rolls forward and covers roughly the last 500 days, so the zip
    is kept as downloaded; reruns read the same snapshot.
    """
    target = raw_dir / "dwd" / f"stundenwerte_TU_{station:05d}_akt.zip"
    if not target.exists():
        body = http_get(DWD_TU.format(station=station))
        if body is None:
            raise RuntimeError(f"DWD station {station} not found")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    return target
