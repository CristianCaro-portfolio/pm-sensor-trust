"""Quality control flags for every location-hour.

Each rule is a boolean column so the reason a reading was dropped is always visible.
Three groups:
  PM checks       range, coverage, flatline, pm_ratio, spike, spatial, drift
  climate checks  climate_missing, climate_invalid, climate_offset (vs the weather station)
  conditions      fog (the reading may be fine as a number but it is mostly water)

Climate checks describe the health of the network's own humidity sensors. They do not
gate the PM value, because the humidity correction uses the weather station.
"""

from __future__ import annotations

import polars as pl

PM_FLAGS = ["f_range", "f_coverage", "f_flatline", "f_pm_ratio", "f_spike", "f_spatial", "f_drift"]
CLIMATE_FLAGS = ["f_climate_missing", "f_climate_invalid", "f_climate_offset"]
CONDITION_FLAGS = ["f_fog"]
ALL_FLAGS = PM_FLAGS + CLIMATE_FLAGS + CONDITION_FLAGS
MAD_SCALE = 1.4826


def _run_length(mask: pl.Expr) -> pl.Expr:
    """Length of the consecutive run each row belongs to (per location)."""
    run_id = mask.rle_id().over("location_id")
    return pl.len().over("location_id", run_id)


def _neighbour_stats(df: pl.DataFrame, edges: pl.DataFrame, col: str) -> pl.DataFrame:
    """Median and MAD of `col` across each location's neighbours, hour by hour."""
    nb = df.select(pl.col("location_id").alias("neighbour_id"), "hour_utc", pl.col(col).alias("nb_value")).filter(
        pl.col("nb_value").is_not_null()
    )
    stats = (
        edges.join(nb, on="neighbour_id")
        .group_by("location_id", "hour_utc")
        .agg(
            pl.col("nb_value").median().alias(f"{col}_nb_median"),
            (pl.col("nb_value") - pl.col("nb_value").median()).abs().median().alias(f"{col}_nb_mad"),
            pl.len().alias(f"{col}_nb_n"),
        )
    )
    return df.join(stats, on=["location_id", "hour_utc"], how="left")


def apply_qc(silver: pl.DataFrame, edges: pl.DataFrame, qc: dict) -> pl.DataFrame:
    df = silver.sort("location_id", "hour_utc").with_columns(pl.col("pm25").clip(lower_bound=0).log1p().alias("log_pm25"))

    # point checks
    df = df.with_columns(
        (
            pl.col("pm25").is_null() | (pl.col("pm25") < 0) | (pl.col("pm25") > qc["max_pm"]) | (pl.col("pm10") > qc["max_pm"])
        ).alias("f_range"),
        (pl.col("n_samples") < qc["min_coverage"] * qc["expected_per_hour"]).alias("f_coverage"),
        (pl.col("pm25") > pl.col("pm10") * qc["pm25_pm10_tolerance"] + 1.0).fill_null(False).alias("f_pm_ratio"),
        pl.col("rh").is_null().alias("f_climate_missing"),
        (~pl.col("rh").is_between(*qc["rh_valid"]) | ~pl.col("temp").is_between(*qc["temp_valid"]))
        .fill_null(False)
        .alias("f_climate_invalid"),
        ((pl.col("rh") - pl.col("rh_ref")).abs() > qc["rh_offset"]).fill_null(False).alias("f_climate_offset"),
        (pl.col("rh_ref") >= qc["fog_rh"]).fill_null(False).alias("f_fog"),
    )

    # a stuck sensor repeats one value for hours
    stuck = (pl.col("pm25_distinct") <= 1) & (pl.col("n_samples") >= 2)
    df = df.with_columns(stuck.alias("_stuck")).with_columns(
        (pl.col("_stuck") & (_run_length(pl.col("_stuck")) >= qc["flatline_hours"])).alias("f_flatline")
    )

    # short spikes against the sensor's own recent history
    med = pl.col("pm25").rolling_median(window_size=5, center=True, min_samples=3).over("location_id")
    df = df.with_columns((pl.col("pm25") - med).alias("_resid"))
    mad = pl.col("_resid").abs().rolling_median(window_size=24, min_samples=6).over("location_id")
    df = df.with_columns(
        (pl.col("_resid") > pl.max_horizontal(qc["spike_mad"] * MAD_SCALE * mad, pl.lit(10.0))).fill_null(False).alias("f_spike")
    )

    # spatial consistency on log PM
    df = _neighbour_stats(df, edges, "log_pm25")
    z_pm = (pl.col("log_pm25") - pl.col("log_pm25_nb_median")).abs() / (MAD_SCALE * pl.col("log_pm25_nb_mad") + 0.15)
    df = df.with_columns(((z_pm > qc["spatial_z"]) & (pl.col("log_pm25_nb_n") >= 3)).fill_null(False).alias("_far"))
    df = df.with_columns((pl.col("_far") & (_run_length(pl.col("_far")) >= qc["spatial_persist_hours"])).alias("f_spatial"))

    # slow bias: the sensor's ratio to the network median over a week
    network = df.group_by("hour_utc").agg(pl.col("pm25").median().alias("net_pm25"))
    df = df.join(network, on="hour_utc", how="left").sort("location_id", "hour_utc")
    log_ratio = ((pl.col("pm25") + 1) / (pl.col("net_pm25") + 1)).log()
    lo, hi = qc["drift_ratio"]
    roll = log_ratio.rolling_median(window_size=qc["drift_window_hours"], min_samples=48).over("location_id")
    df = df.with_columns(roll.exp().alias("bias_ratio")).with_columns(
        (~pl.col("bias_ratio").is_between(lo, hi)).fill_null(False).alias("f_drift")
    )

    df = df.with_columns(
        (~pl.any_horizontal(PM_FLAGS)).alias("pm_ok"),
        (~pl.any_horizontal(CLIMATE_FLAGS)).alias("climate_ok"),
    )
    usable = pl.col("pm_ok") & pl.col("rh_ref").is_not_null()
    if qc.get("exclude_fog", False):
        usable = usable & ~pl.col("f_fog")
    df = df.with_columns(usable.alias("trusted"))

    keep = [
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
        "net_pm25",
        "bias_ratio",
        *ALL_FLAGS,
        "pm_ok",
        "climate_ok",
        "trusted",
    ]
    return df.select(keep)


def flag_summary(flags: pl.DataFrame) -> pl.DataFrame:
    """Share of location-hours raised by each rule."""
    total = flags.height
    return pl.DataFrame(
        {
            "flag": ALL_FLAGS + ["trusted"],
            "hours": [int(flags[c].sum()) for c in ALL_FLAGS + ["trusted"]],
        }
    ).with_columns((pl.col("hours") / max(total, 1)).round(4).alias("share"))
