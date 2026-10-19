"""Dagster definitions: the same stages as pipeline.py, as assets with checks.

dagster dev -m pmtrust.defs        # UI on localhost:3000
pmtrust run                        # materialize everything in process
"""

from __future__ import annotations

import os
from pathlib import Path

import dagster as dg
import polars as pl

from pmtrust.calibrate import Calibration
from pmtrust.config import ROOT, load_settings
from pmtrust.pipeline import (
    Paths,
    stage_bronze,
    stage_calibrate,
    stage_evaluate,
    stage_gold,
    stage_silver,
)


class LakeResource(dg.ConfigurableResource):
    raw_dir: str = str(ROOT / "data" / "raw")
    lake_dir: str = str(ROOT / "data" / "lake")
    results_dir: str = str(ROOT / "results")
    config_path: str = str(ROOT / "config.yaml")

    def paths(self) -> Paths:
        return Paths(Path(self.raw_dir), Path(self.lake_dir), Path(self.results_dir))

    def settings(self):
        return load_settings(self.config_path)


@dg.asset(group_name="bronze", description="Raw CSV and JSON files merged into typed Delta tables, bad rows quarantined.")
def bronze_tables(lake: LakeResource) -> dg.MaterializeResult:
    stats = stage_bronze(lake.settings(), lake.paths())
    flat = {f"{w}.{k}": v for w, s in stats.items() for k, v in s.items()}
    return dg.MaterializeResult(metadata={k: dg.MetadataValue.int(int(v)) for k, v in flat.items()})


@dg.asset(group_name="silver", deps=[bronze_tables], description="Location-hour rows with every QC flag.")
def silver_qc(lake: LakeResource) -> dict[str, pl.DataFrame]:
    return stage_silver(lake.settings(), lake.paths())


@dg.asset(group_name="gold", description="kappa (blocked CV) and scale fitted on the calibration window only.")
def calibration(lake: LakeResource, silver_qc: dict[str, pl.DataFrame]) -> Calibration:
    return stage_calibrate(lake.settings(), lake.paths(), silver_qc)


@dg.asset(group_name="gold", description="City PM2.5 per hour: naive, QC'd and corrected estimates.")
def gold_city_hourly(lake: LakeResource, silver_qc: dict[str, pl.DataFrame], calibration: Calibration) -> dict[str, pl.DataFrame]:
    return stage_gold(lake.settings(), lake.paths(), silver_qc, calibration)


@dg.asset(group_name="report", description="Scores against the UBA reference station and the fault injection test.")
def evaluation(
    lake: LakeResource,
    silver_qc: dict[str, pl.DataFrame],
    gold_city_hourly: dict[str, pl.DataFrame],
    calibration: Calibration,
) -> dict:
    return stage_evaluate(lake.settings(), lake.paths(), silver_qc, gold_city_hourly, calibration)


@dg.asset_check(asset=silver_qc, description="Every window keeps at least half of its location-hours.")
def enough_trusted_hours(silver_qc: dict[str, pl.DataFrame]) -> dg.AssetCheckResult:
    shares = {w: float(f["trusted"].mean()) for w, f in silver_qc.items()}
    return dg.AssetCheckResult(
        passed=all(v >= 0.5 for v in shares.values()), metadata={w: round(v, 3) for w, v in shares.items()}
    )


@dg.asset_check(asset=gold_city_hourly, description="At least 90% of hours are published with enough trusted sensors.")
def publish_coverage(gold_city_hourly: dict[str, pl.DataFrame]) -> dg.AssetCheckResult:
    shares = {w: float(g["published"].mean()) for w, g in gold_city_hourly.items()}
    return dg.AssetCheckResult(
        passed=all(v >= 0.9 for v in shares.values()), metadata={w: round(v, 3) for w, v in shares.items()}
    )


@dg.asset_check(asset=evaluation, description="Corrected estimate beats the naive network mean on every test window.")
def corrected_beats_naive(evaluation: dict) -> dg.AssetCheckResult:
    meta, passed = {}, True
    for w, s in evaluation["windows"].items():
        if w == "calibrate":
            continue
        naive, ours = s["variants"]["raw_mean"]["mae"], s["variants"]["corrected"]["mae"]
        meta[f"{w}.mae_naive"], meta[f"{w}.mae_corrected"] = naive, ours
        passed = passed and ours < naive
    return dg.AssetCheckResult(passed=passed, metadata=meta)


@dg.asset_check(asset=evaluation, description="Reference and sensors line up in time: the best lag is within one hour.")
def time_alignment(evaluation: dict) -> dg.AssetCheckResult:
    lags = {}
    for w, s in evaluation["windows"].items():
        corr = {int(k): v for k, v in s["lag_correlation"].items()}
        lags[w] = max(corr, key=corr.get) if corr else 0
    return dg.AssetCheckResult(passed=all(abs(v) <= 1 for v in lags.values()), metadata=lags)


ASSETS = [bronze_tables, silver_qc, calibration, gold_city_hourly, evaluation]
CHECKS = [enough_trusted_hours, publish_coverage, corrected_beats_naive, time_alignment]


def lake_from_env() -> LakeResource:
    return LakeResource(
        raw_dir=os.environ.get("PMTRUST_RAW", str(ROOT / "data" / "raw")),
        lake_dir=os.environ.get("PMTRUST_LAKE", str(ROOT / "data" / "lake")),
        results_dir=os.environ.get("PMTRUST_RESULTS", str(ROOT / "results")),
        config_path=os.environ.get("PMTRUST_CONFIG", str(ROOT / "config.yaml")),
    )


defs = dg.Definitions(assets=ASSETS, asset_checks=CHECKS, resources={"lake": lake_from_env()})
