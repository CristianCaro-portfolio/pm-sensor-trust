from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from pmtrust.bronze import _cast_with_quarantine, merge_into, parse_reference
from pmtrust.calibrate import city_hourly, growth_factor
from pmtrust.config import load_settings
from pmtrust.evaluate import metrics
from pmtrust.qc import apply_qc
from pmtrust.silver import haversine_km, neighbour_graph

QC = load_settings().qc


def _silver(n_locs: int = 6, hours: int = 72, seed: int = 0) -> pl.DataFrame:
    """Small network where every sensor follows the same signal plus noise."""
    rng = np.random.default_rng(seed)
    t0 = datetime(2026, 1, 12)
    signal = 15 + 8 * np.sin(np.arange(hours) / 6)
    rows = []
    for loc in range(1, n_locs + 1):
        for h in range(hours):
            pm25 = float(signal[h] * rng.uniform(0.9, 1.1))
            rows.append(
                {
                    "location_id": loc,
                    "sensor_id": 100 + loc,
                    "lat": 48.78 + loc * 0.01,
                    "lon": 9.18,
                    "hour_utc": t0 + timedelta(hours=h),
                    "n_samples": 24,
                    "pm25": pm25,
                    "pm10": pm25 * 1.4,
                    "pm25_distinct": 20,
                    "pm25_max": pm25 * 1.2,
                    "rh": 80.0,
                    "temp": 3.0,
                    "rh_ref": 82.0,
                    "temp_ref": 3.1,
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("hour_utc").dt.strftime("%Y-%m-%d").alias("day"))


def _qc(df: pl.DataFrame) -> pl.DataFrame:
    return apply_qc(df, neighbour_graph(df, 4), QC)


def test_clean_network_is_trusted():
    flags = _qc(_silver())
    assert flags["trusted"].mean() > 0.95


def test_stuck_sensor_is_flagged():
    df = _silver().with_columns(
        pl.when((pl.col("location_id") == 2) & (pl.col("hour_utc") >= datetime(2026, 1, 13)))
        .then(1)
        .otherwise(pl.col("pm25_distinct"))
        .alias("pm25_distinct")
    )
    flags = _qc(df).filter(pl.col("location_id") == 2, pl.col("hour_utc") >= datetime(2026, 1, 13))
    assert flags["f_flatline"].all()


def test_spike_is_flagged():
    df = _silver().with_columns(
        pl.when((pl.col("location_id") == 3) & (pl.col("hour_utc") == datetime(2026, 1, 13, 5)))
        .then(pl.col("pm25") * 8 + 40)
        .otherwise(pl.col("pm25"))
        .alias("pm25")
    )
    hit = _qc(df).filter(pl.col("location_id") == 3, pl.col("hour_utc") == datetime(2026, 1, 13, 5))
    assert hit["f_spike"].item()


def test_biased_sensor_is_flagged_by_drift_or_spatial():
    df = (
        _silver()
        .with_columns(pl.when(pl.col("location_id") == 4).then(pl.col("pm25") * 4).otherwise(pl.col("pm25")).alias("pm25"))
        .with_columns(pl.max_horizontal("pm10", "pm25").alias("pm10"))
    )
    flags = _qc(df).filter(pl.col("location_id") == 4, pl.col("hour_utc") >= datetime(2026, 1, 14))
    assert (flags["f_drift"] | flags["f_spatial"]).mean() > 0.9


def test_bad_humidity_sensor_does_not_block_pm():
    df = _silver().with_columns(pl.when(pl.col("location_id") == 5).then(14.0).otherwise(pl.col("rh")).alias("rh"))
    flags = _qc(df).filter(pl.col("location_id") == 5)
    assert flags["f_climate_offset"].all()
    assert flags["trusted"].mean() > 0.95


def test_range_and_pm_ratio():
    df = _silver().with_columns(pl.when(pl.col("location_id") == 1).then(1200.0).otherwise(pl.col("pm25")).alias("pm25"))
    flags = _qc(df).filter(pl.col("location_id") == 1)
    assert flags["f_range"].all()
    assert not flags["trusted"].any()


@pytest.mark.parametrize("rh,kappa,expected", [(0.0, 0.3, 1.0), (50.0, 0.2, 1.2), (99.0, 0.2, 1 + 0.2 * 0.95 / 0.05)])
def test_growth_factor(rh, kappa, expected):
    out = pl.DataFrame({"rh": [rh]}).select(growth_factor(pl.col("rh"), kappa, 95.0)).item()
    assert out == pytest.approx(expected)


def test_city_hourly_publishes_only_with_enough_trusted():
    flags = _qc(_silver(n_locs=4))
    city = city_hourly(flags, 0.1, 95.0, min_sensors=5)
    assert not city["published"].any()
    city = city_hourly(flags, 0.1, 95.0, min_sensors=3)
    assert city["published"].mean() > 0.9


def test_quarantine_routes_unparseable_values():
    raw = pl.DataFrame(
        {
            "sensor_id": ["1", "1", "2"],
            "timestamp": ["2026-01-12T00:00:01", "garbage", "2026-01-12T00:02:01"],
            "P2": ["3.5", "4.0", "n/a"],
        }
    )
    good, bad = _cast_with_quarantine(raw, {"sensor_id": pl.Int64, "timestamp": pl.Datetime("us"), "P2": pl.Float64}, "pm")
    assert good.height == 1
    assert sorted(bad["reason"].to_list()) == ["unparseable_P2", "unparseable_timestamp"]


def test_merge_is_idempotent(tmp_path):
    df = pl.DataFrame(
        {"sensor_id": [1, 2], "ts": [datetime(2026, 1, 1), datetime(2026, 1, 1)], "v": [1.0, 2.0], "day": ["2026-01-01"] * 2}
    )
    for _ in range(3):
        merge_into(tmp_path / "t", df, ["sensor_id", "ts"], "day")
    assert pl.read_delta(str(tmp_path / "t")).height == 2
    merge_into(tmp_path / "t", df.with_columns(pl.col("v") * 10), ["sensor_id", "ts"], "day")
    assert pl.read_delta(str(tmp_path / "t")).sort("sensor_id")["v"].to_list() == [10.0, 20.0]


def test_reference_is_shifted_to_utc(tmp_path):
    payload = {
        "data": {
            "224": {
                "2026-01-12 00:00:00": [9, 2, 20, "2026-01-12 01:00:00", "0"],
                "2026-01-12 01:00:00": [9, 2, None, "2026-01-12 02:00:00", "0"],
            }
        }
    }
    path = tmp_path / "224_pm25_x.json"
    path.write_text(__import__("json").dumps(payload))
    ref = parse_reference([path], 224, utc_offset_hours=1)
    assert ref["hour_utc"].to_list()[0] == datetime(2026, 1, 11, 23)
    assert ref["ref_pm25"].to_list() == [20.0, None]


def test_metrics_and_distance():
    m = metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
    assert m["mae"] == pytest.approx(0.33, abs=0.01)
    assert m["bias"] == pytest.approx(-0.33, abs=0.01)
    assert haversine_km(48.78, 9.18, 48.78, 9.18) == 0
    assert 110 < haversine_km(48.0, 9.0, 49.0, 9.0) < 112


def test_windows_do_not_overlap():
    cfg = load_settings()
    spans = sorted(cfg.window(w) for w in cfg.window_names)
    for (_s1, e1), (s2, _) in zip(spans, spans[1:], strict=False):
        assert e1 < s2
    assert all(isinstance(s, date) for s, _ in spans)
