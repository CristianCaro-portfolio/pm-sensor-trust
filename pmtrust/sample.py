"""Cut a small, real sample out of data/raw for tests and CI.

The sample keeps all raw files in one flat folder, gzips the CSV files and trims the
reference and weather files to the sample days, so the full pipeline runs on it offline.
"""

from __future__ import annotations

import gzip
import io
import json
import shutil
import zipfile
from datetime import date, timedelta
from pathlib import Path

import yaml

from pmtrust.config import DEFAULT_CONFIG
from pmtrust.ingest import daterange, paired_sensors

SAMPLE_WINDOWS = {
    "calibrate": ["2026-01-12", "2026-01-14"],
    "test_winter": ["2026-01-26", "2026-01-27"],
    "test_autumn": ["2026-10-01", "2026-10-02"],
}


def _pick_locations(raw: Path, n: int) -> list[dict]:
    """Paired locations with complete files on every sample day, nearest to the centre first."""
    sensors = paired_sensors(json.loads((raw / "sensors.json").read_text()))
    days = [d for s, e in SAMPLE_WINDOWS.values() for d in daterange(date.fromisoformat(s), date.fromisoformat(e))]
    by_loc: dict[int, list[dict]] = {}
    for s in sensors:
        by_loc.setdefault(s["location_id"], []).append(s)
    complete = []
    for loc, group in by_loc.items():
        pm = [s for s in group if s["sensor_type"] == "SDS011"][:1]
        clim = [s for s in group if s["sensor_type"] != "SDS011"][:1]
        chosen = pm + clim
        ok = all(
            (raw / "sensor_community" / d.isoformat() / f"{d}_{s['sensor_type'].lower()}_sensor_{s['sensor_id']}.csv").exists()
            for d in days
            for s in chosen
        )
        if ok and len(chosen) == 2:
            complete.append((abs(chosen[0]["lat"] - 48.78) + abs(chosen[0]["lon"] - 9.18), loc, chosen))
    complete.sort()
    return [s for _, _, chosen in complete[:n] for s in chosen]


def build_sample(raw: Path, out: Path, n_locations: int, _days: int) -> dict:
    if out.exists():
        shutil.rmtree(out)
    sensors = _pick_locations(raw, n_locations)
    (out / "raw").mkdir(parents=True)
    (out / "raw" / "sensors.json").write_text(json.dumps(sensors, indent=1))
    files = 0
    for start, end in SAMPLE_WINDOWS.values():
        for d in daterange(date.fromisoformat(start), date.fromisoformat(end)):
            for s in sensors:
                name = f"{d}_{s['sensor_type'].lower()}_sensor_{s['sensor_id']}.csv"
                src = raw / "sensor_community" / d.isoformat() / name
                dst = out / "raw" / (name + ".gz")
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(gzip.compress(src.read_bytes(), mtime=0))
                files += 1

    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    station = cfg["reference"]["station_id"]
    full = dict(zip(cfg["windows"], cfg["windows"].values(), strict=False))
    for name, (start, end) in SAMPLE_WINDOWS.items():
        fstart, fend = full[name]
        keep = {d.isoformat() for d in daterange(date.fromisoformat(start), date.fromisoformat(end))}
        for comp in ("pm10", "pm25"):
            src = json.loads((raw / "uba" / f"{station}_{comp}_{fstart}_{fend}.json").read_text())
            data = src["data"][str(station)]
            src["data"][str(station)] = {k: v for k, v in data.items() if k[:10] in keep}
            dst = out / "raw" / f"{station}_{comp}_{start}_{end}.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(src))

    zsrc = next((raw / "dwd").glob("stundenwerte_TU_*_akt.zip"))
    days = {
        d.strftime("%Y%m%d")
        for s, e in SAMPLE_WINDOWS.values()
        for d in daterange(date.fromisoformat(s) - timedelta(days=1), date.fromisoformat(e) + timedelta(days=1))
    }
    with zipfile.ZipFile(zsrc) as zf:
        name = next(n for n in zf.namelist() if n.startswith("produkt_tu_stunde"))
        lines = zf.read(name).decode("latin-1").splitlines()
    kept = [lines[0]] + [ln for ln in lines[1:] if ln.split(";")[1].strip()[:8] in days]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, "\n".join(kept) + "\n")
    (out / "raw" / zsrc.name).write_bytes(buf.getvalue())

    sample_cfg = dict(cfg, windows=SAMPLE_WINDOWS)
    sample_cfg["gold"] = dict(cfg["gold"], min_sensors=3)
    sample_cfg["qc"] = dict(cfg["qc"], drift_window_hours=48, neighbours=4)
    (out / "config.yaml").write_text(
        "# Same settings as the root config, shrunk to the sample size.\n" + yaml.safe_dump(sample_cfg, sort_keys=False)
    )
    return {"locations": len(sensors) // 2, "files": files}
