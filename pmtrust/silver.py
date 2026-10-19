"""Bronze readings -> one row per location and UTC hour, PM joined with humidity."""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta
from pathlib import Path

import polars as pl


def _window_filter(start: date, end: date) -> pl.Expr:
    lo = datetime.combine(start, time.min)
    hi = datetime.combine(end + timedelta(days=1), time.min)
    return (pl.col("ts") >= lo) & (pl.col("ts") < hi)


def hourly_pm(bronze_pm: pl.LazyFrame, start: date, end: date) -> pl.LazyFrame:
    return (
        bronze_pm.filter(_window_filter(start, end))
        .with_columns(pl.col("ts").dt.truncate("1h").alias("hour_utc"))
        .group_by("location_id", "hour_utc")
        .agg(
            pl.col("sensor_id").first(),
            pl.col("lat").first(),
            pl.col("lon").first(),
            pl.len().alias("n_samples"),
            pl.col("pm25").mean().alias("pm25"),
            pl.col("pm10").mean().alias("pm10"),
            pl.col("pm25").n_unique().alias("pm25_distinct"),
            pl.col("pm25").max().alias("pm25_max"),
        )
    )


def hourly_climate(bronze_climate: pl.LazyFrame, start: date, end: date) -> pl.LazyFrame:
    # a location can have more than one climate sensor; the median of their hourly means is used
    per_sensor = (
        bronze_climate.filter(_window_filter(start, end))
        .with_columns(pl.col("ts").dt.truncate("1h").alias("hour_utc"))
        .group_by("location_id", "sensor_id", "hour_utc")
        .agg(pl.col("humidity").mean(), pl.col("temperature").mean())
    )
    return per_sensor.group_by("location_id", "hour_utc").agg(
        pl.col("humidity").median().alias("rh"),
        pl.col("temperature").median().alias("temp"),
    )


def build_silver(lake: Path, start: date, end: date) -> pl.DataFrame:
    pm = hourly_pm(pl.scan_delta(str(lake / "bronze_pm")), start, end)
    clim = hourly_climate(pl.scan_delta(str(lake / "bronze_climate")), start, end)
    weather_path = lake / "bronze_weather"
    if (weather_path / "_delta_log").exists():
        weather = pl.scan_delta(str(weather_path)).select("hour_utc", "rh_ref", "temp_ref")
    else:
        weather = pl.LazyFrame(schema={"hour_utc": pl.Datetime("us"), "rh_ref": pl.Float64, "temp_ref": pl.Float64})
    return (
        pm.join(clim, on=["location_id", "hour_utc"], how="left")
        .join(weather, on="hour_utc", how="left")
        .with_columns(pl.col("hour_utc").dt.strftime("%Y-%m-%d").alias("day"))
        .sort("location_id", "hour_utc")
        .collect()
    )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def neighbour_graph(silver: pl.DataFrame, k: int) -> pl.DataFrame:
    """Edges location -> its k nearest other locations, with distance in km."""
    locs = silver.group_by("location_id").agg(pl.col("lat").first(), pl.col("lon").first()).sort("location_id")
    rows = locs.rows()
    edges = []
    for loc, lat, lon in rows:
        dists = sorted((haversine_km(lat, lon, lat2, lon2), loc2) for loc2, lat2, lon2 in rows if loc2 != loc)
        edges.extend({"location_id": loc, "neighbour_id": n, "km": round(d, 3)} for d, n in dists[:k])
    return pl.DataFrame(edges, schema={"location_id": pl.Int64, "neighbour_id": pl.Int64, "km": pl.Float64})
