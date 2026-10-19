"""Download every raw file the study needs into data/raw.

    python scripts/download_data.py            # all windows in config.yaml
    python scripts/download_data.py --workers 4

Sources (all open, no key):
  Sensor.Community archive, ODbL 1.0     https://archive.sensor.community/
  UBA air data API, dl-de/by-2.0         https://luftdaten.umweltbundesamt.de/
  DWD Climate Data Center, CC BY 4.0     https://opendata.dwd.de/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pmtrust.config import ROOT, load_settings
from pmtrust.ingest import (
    discover_sensors,
    fetch_reference,
    fetch_sensor_days,
    fetch_weather,
    paired_sensors,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = load_settings()
    area = cfg.area
    sensors = discover_sensors(area["lat"], area["lon"], area["radius_km"], args.raw_dir / "sensors.json")
    pairs = paired_sensors(sensors)
    logging.info("%d sensors in area, %d on paired outdoor locations", len(sensors), len(pairs))

    fetch_weather(cfg.weather["station_id"], args.raw_dir)
    for name in cfg.window_names:
        start, end = cfg.window(name)
        fetch_reference(cfg.reference["station_id"], start, end, args.raw_dir)
        fetch_sensor_days(pairs, start, end, args.raw_dir, workers=args.workers)


if __name__ == "__main__":
    main()
