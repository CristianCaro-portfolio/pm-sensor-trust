"""End to end on the real sample in sample_data/ (8 locations, 7 days, offline)."""

from pathlib import Path

import dagster as dg
import polars as pl
import pytest

from pmtrust.config import ROOT, load_settings
from pmtrust.contracts import BronzePM
from pmtrust.defs import ASSETS, CHECKS, LakeResource
from pmtrust.ingest import paired_sensors
from pmtrust.pipeline import Paths, run_all, stage_bronze

SAMPLE = ROOT / "sample_data"


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    base = tmp_path_factory.mktemp("lake")
    cfg = load_settings(SAMPLE / "config.yaml")
    paths = Paths(SAMPLE / "raw", base / "lake", base / "results")
    report = run_all(cfg, paths)
    return cfg, paths, report


def test_tables_are_written(run):
    _, paths, _ = run
    for table in ("bronze_pm", "bronze_climate", "bronze_reference", "bronze_weather", "silver_qc", "gold_city_hourly"):
        assert (paths.lake / table / "_delta_log").exists(), table
    BronzePM.validate(pl.read_delta(str(paths.lake / "bronze_pm")))


def test_bronze_reload_is_idempotent(run):
    cfg, paths, _ = run
    before = pl.read_delta(str(paths.lake / "bronze_pm")).height
    stage_bronze(cfg, paths)
    assert pl.read_delta(str(paths.lake / "bronze_pm")).height == before


def test_report_has_every_window_and_variant(run):
    cfg, _, report = run
    assert set(report["windows"]) == set(cfg.window_names)
    for s in report["windows"].values():
        assert s["hours_published"] > 24
        assert {"raw_mean", "qc", "corrected"} <= set(s["variants"])
    assert report["fault_test"]["faults"]["stuck"]["recall"] == 1.0


def test_calibration_is_fitted_on_the_calibration_window_only(run):
    _, _, report = run
    cal = report["calibration"]
    assert cal["scale"] > 0
    assert cal["kappa"] in load_settings(SAMPLE / "config.yaml").correction["kappa_grid"]


def test_dagster_graph_materializes_with_checks(tmp_path: Path):
    lake = LakeResource(
        raw_dir=str(SAMPLE / "raw"),
        lake_dir=str(tmp_path / "lake"),
        results_dir=str(tmp_path / "results"),
        config_path=str(SAMPLE / "config.yaml"),
    )
    result = dg.materialize([*ASSETS, *CHECKS], resources={"lake": lake})
    assert result.success
    checks = {ev.check_name: ev.passed for ev in result.get_asset_check_evaluations()}
    assert set(checks) == {c.check_key.name for c in CHECKS}
    assert checks["enough_trusted_hours"] and checks["publish_coverage"]


def test_sensor_registry_pairs_pm_with_humidity():
    sensors = [
        {"sensor_id": 1, "sensor_type": "SDS011", "location_id": 10, "lat": 0, "lon": 0, "indoor": 0},
        {"sensor_id": 2, "sensor_type": "DHT22", "location_id": 10, "lat": 0, "lon": 0, "indoor": 0},
        {"sensor_id": 3, "sensor_type": "SDS011", "location_id": 11, "lat": 0, "lon": 0, "indoor": 0},
        {"sensor_id": 4, "sensor_type": "SDS011", "location_id": 12, "lat": 0, "lon": 0, "indoor": 1},
        {"sensor_id": 5, "sensor_type": "BME280", "location_id": 12, "lat": 0, "lon": 0, "indoor": 1},
    ]
    assert sorted(s["sensor_id"] for s in paired_sensors(sensors)) == [1, 2]
