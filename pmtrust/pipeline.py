"""Plain functions for each stage. Dagster assets (defs.py) and the CLI both call these."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from pmtrust.bronze import load_bronze, merge_into
from pmtrust.calibrate import Calibration, apply_calibration, city_hourly, fit_calibration
from pmtrust.config import Settings
from pmtrust.contracts import GoldCityHourly
from pmtrust.evaluate import best_lag, fault_test, score
from pmtrust.qc import apply_qc, flag_summary
from pmtrust.silver import build_silver, neighbour_graph

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Paths:
    raw: Path
    lake: Path
    results: Path


def stage_bronze(cfg: Settings, paths: Paths) -> dict[str, dict]:
    ref = cfg.reference
    return {
        name: load_bronze(paths.raw, paths.lake, *cfg.window(name), ref["station_id"], ref["utc_offset_hours"])
        for name in cfg.window_names
    }


def stage_silver(cfg: Settings, paths: Paths) -> dict[str, pl.DataFrame]:
    """Hourly rows + QC flags per window, persisted to the silver_qc Delta table."""
    out = {}
    for name in cfg.window_names:
        silver = build_silver(paths.lake, *cfg.window(name))
        if silver.is_empty():
            log.warning("window %s has no data", name)
            continue
        edges = neighbour_graph(silver, cfg.qc["neighbours"])
        flags = apply_qc(silver, edges, cfg.qc).with_columns(pl.lit(name).alias("window"))
        merge_into(paths.lake / "silver_qc", flags, ["location_id", "hour_utc"], "day")
        out[name] = flags
    return out


def read_reference(paths: Paths) -> pl.DataFrame:
    return pl.read_delta(str(paths.lake / "bronze_reference")).select("hour_utc", "ref_pm10", "ref_pm25")


def stage_calibrate(cfg: Settings, paths: Paths, flags: dict[str, pl.DataFrame]) -> Calibration:
    reference = read_reference(paths)
    cal_flags = flags["calibrate"]
    min_sensors = cfg.gold["min_sensors"]
    cal = fit_calibration(cal_flags, reference, cfg.correction, min_sensors)
    paths.results.mkdir(parents=True, exist_ok=True)
    (paths.results / "calibration.json").write_text(json.dumps(cal.to_dict(), indent=2))
    return cal


def stage_gold(cfg: Settings, paths: Paths, flags: dict[str, pl.DataFrame], cal: Calibration) -> dict[str, pl.DataFrame]:
    out = {}
    for name, f in flags.items():
        city = apply_calibration(city_hourly(f, cal.kappa, cfg.correction["rh_cap"], cfg.gold["min_sensors"]), cal)
        city = city.with_columns(
            pl.when(pl.col("published")).then(pl.col("pm25_corrected")).alias("pm25_corrected"),
            pl.lit(name).alias("window"),
        )
        GoldCityHourly.validate(city.select(list(GoldCityHourly.to_schema().columns)))
        merge_into(paths.lake / "gold_city_hourly", city, ["hour_utc"])
        out[name] = city
    return out


def stage_evaluate(
    cfg: Settings, paths: Paths, flags: dict[str, pl.DataFrame], gold: dict[str, pl.DataFrame], cal: Calibration
) -> dict:
    reference = read_reference(paths)
    report: dict = {"calibration": cal.to_dict(), "windows": {}, "flags": {}}
    for name, city in gold.items():
        report["windows"][name] = score(city, reference)
        report["windows"][name]["lag_correlation"] = best_lag(city, reference)
        report["flags"][name] = flag_summary(flags[name]).to_dicts()
        report["windows"][name]["sensors"] = int(flags[name]["location_id"].n_unique())
    if "test_winter" in flags:
        silver_cols = [
            "location_id",
            "sensor_id",
            "lat",
            "lon",
            "hour_utc",
            "day",
            "n_samples",
            "pm25",
            "pm10",
            "rh",
            "temp",
            "rh_ref",
            "temp_ref",
        ]
        win = cfg.window("test_winter")
        silver = build_silver(paths.lake, *win)
        silver = silver.select(silver_cols + ["pm25_distinct"])
        edges = neighbour_graph(silver, cfg.qc["neighbours"])
        report["fault_test"] = fault_test(silver, edges, cfg.qc)
    (paths.results / "metrics.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def run_all(cfg: Settings, paths: Paths) -> dict:
    stage_bronze(cfg, paths)
    flags = stage_silver(cfg, paths)
    cal = stage_calibrate(cfg, paths, flags)
    gold = stage_gold(cfg, paths, flags, cal)
    return stage_evaluate(cfg, paths, flags, gold, cal)
