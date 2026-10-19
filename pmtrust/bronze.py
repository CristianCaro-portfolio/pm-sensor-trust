"""Raw files -> typed bronze Delta tables, with a quarantine for rows that fail parsing.

Writes are MERGEs on the natural key (sensor, timestamp), scoped to the day partition,
so loading the same day twice leaves the table unchanged.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from datetime import date, datetime, time, timedelta
from pathlib import Path

import polars as pl
from deltalake import DeltaTable

from pmtrust.contracts import BronzeClimate, BronzePM, BronzeReference, BronzeWeather
from pmtrust.ingest import daterange

log = logging.getLogger(__name__)

PM_KIND = "sds011"
CLIMATE_KINDS = ("dht22", "bme280")
TS_FORMAT = "%Y-%m-%dT%H:%M:%S"


def find_raw(raw_dir: Path, subdirs: list[str], pattern: str) -> list[Path]:
    """Files matching `pattern` in the standard layout (raw/<subdir>/...) or flat in raw/.

    data/raw keeps one folder per source and per day; sample_data/raw keeps everything
    in one folder. Both layouts load the same way.
    """
    found: set[Path] = set()
    for base in [raw_dir / s for s in subdirs] + [raw_dir]:
        if base.is_dir():
            found.update(base.glob(pattern))
    return sorted(found)


def _read_files(paths: list[Path]) -> pl.DataFrame:
    frames = [
        pl.read_csv(p, separator=";", infer_schema=False, truncate_ragged_lines=True) for p in paths if p.stat().st_size > 0
    ]
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def _cast_with_quarantine(raw: pl.DataFrame, casts: dict[str, pl.DataType], source: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Cast string columns; any non-empty value that fails to cast sends the row to quarantine."""
    typed = raw.with_columns(
        [pl.col(c).str.strip_chars().cast(t, strict=False).alias(f"__{c}") for c, t in casts.items() if c != "timestamp"]
        + [pl.col("timestamp").str.strptime(pl.Datetime("us"), TS_FORMAT, strict=False).alias("__timestamp")]
    )
    reasons = []
    for c in casts:
        failed = pl.col(c).is_not_null() & (pl.col(c).str.strip_chars() != "") & pl.col(f"__{c}").is_null()
        reasons.append(pl.when(failed).then(pl.lit(f"unparseable_{c}")))
    reasons.append(pl.when(pl.col("timestamp").is_null()).then(pl.lit("missing_timestamp")))
    typed = typed.with_columns(pl.coalesce(reasons).alias("reason"))
    bad = typed.filter(pl.col("reason").is_not_null()).select(
        pl.lit(source).alias("source"),
        pl.col("sensor_id").cast(pl.String),
        pl.col("timestamp").cast(pl.String),
        pl.col("reason"),
        pl.concat_str([pl.col(c).fill_null("") for c in casts], separator=";").alias("raw_values"),
    )
    good = typed.filter(pl.col("reason").is_null()).select([pl.col(f"__{c}").alias(c) for c in casts])
    return good, bad


def _dedupe(df: pl.DataFrame, source: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Keep the first row per (sensor_id, ts); later ones go to quarantine."""
    repeated = df.select(pl.struct("sensor_id", "ts").is_first_distinct().not_()).to_series()
    bad = df.filter(repeated).select(
        pl.lit(source).alias("source"),
        pl.col("sensor_id").cast(pl.String),
        pl.col("ts").dt.strftime(TS_FORMAT).alias("timestamp"),
        pl.lit("duplicate_key").alias("reason"),
        pl.lit("").alias("raw_values"),
    )
    return df.filter(~repeated), bad


def parse_pm_day(raw_dir: Path, day: date) -> tuple[pl.DataFrame, pl.DataFrame]:
    files = find_raw(raw_dir, [f"sensor_community/{day}", "sensor_community"], f"{day}_{PM_KIND}_sensor_*.csv*")
    raw = _read_files(files)
    if raw.is_empty():
        return pl.DataFrame(schema=BronzePM.to_schema().columns.keys()), pl.DataFrame()
    casts = {
        "sensor_id": pl.Int64,
        "location": pl.Int64,
        "lat": pl.Float64,
        "lon": pl.Float64,
        "timestamp": pl.Datetime("us"),
        "P1": pl.Float64,
        "P2": pl.Float64,
    }
    good, bad = _cast_with_quarantine(raw, casts, "pm")
    good = good.rename({"location": "location_id", "timestamp": "ts", "P1": "pm10", "P2": "pm25"})
    good = good.with_columns(pl.col("ts").dt.strftime("%Y-%m-%d").alias("day"))
    # a file for day D can carry a few rows of D+1 around midnight; keep them where they belong
    good, dup = _dedupe(good.sort("sensor_id", "ts"), "pm")
    good = BronzePM.validate(good.select(list(BronzePM.to_schema().columns)))
    return good, pl.concat([bad, dup], how="diagonal_relaxed")


def parse_climate_day(raw_dir: Path, day: date) -> tuple[pl.DataFrame, pl.DataFrame]:
    subdirs = [f"sensor_community/{day}", "sensor_community"]
    files = [p for k in CLIMATE_KINDS for p in find_raw(raw_dir, subdirs, f"{day}_{k}_sensor_*.csv*")]
    raw = _read_files(files)
    if raw.is_empty():
        return pl.DataFrame(schema=BronzeClimate.to_schema().columns.keys()), pl.DataFrame()
    casts = {
        "sensor_id": pl.Int64,
        "location": pl.Int64,
        "timestamp": pl.Datetime("us"),
        "temperature": pl.Float64,
        "humidity": pl.Float64,
    }
    good, bad = _cast_with_quarantine(raw, casts, "climate")
    good = good.rename({"location": "location_id", "timestamp": "ts"})
    good = good.with_columns(pl.col("ts").dt.strftime("%Y-%m-%d").alias("day"))
    good, dup = _dedupe(good.sort("sensor_id", "ts"), "climate")
    good = BronzeClimate.validate(good.select(list(BronzeClimate.to_schema().columns)))
    return good, pl.concat([bad, dup], how="diagonal_relaxed")


def parse_reference(paths: list[Path], station_id: int, utc_offset_hours: int) -> pl.DataFrame:
    """UBA hourly means are labelled by the start of the hour in CET (UTC+1, no DST)."""
    rows: dict[str, dict] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        for start, values in payload.get("data", {}).get(str(station_id), {}).items():
            comp, value = int(values[0]), values[2]
            col = {1: "ref_pm10", 9: "ref_pm25"}.get(comp)
            if col is None:
                continue
            rows.setdefault(start, {"start": start})[col] = None if value is None else float(value)
    if not rows:
        return pl.DataFrame(
            schema={"station_id": pl.Int64, "hour_utc": pl.Datetime("us"), "ref_pm10": pl.Float64, "ref_pm25": pl.Float64}
        )
    df = pl.DataFrame(list(rows.values()), infer_schema_length=None)
    for col in ("ref_pm10", "ref_pm25"):
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, pl.Float64).alias(col))
    df = (
        df.select(
            pl.lit(station_id, pl.Int64).alias("station_id"),
            (pl.col("start").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S") - timedelta(hours=utc_offset_hours)).alias(
                "hour_utc"
            ),
            pl.col("ref_pm10").cast(pl.Float64),
            pl.col("ref_pm25").cast(pl.Float64),
        )
        .unique("hour_utc", keep="last")
        .sort("hour_utc")
    )
    return BronzeReference.validate(df)


def parse_weather(zip_path: Path) -> pl.DataFrame:
    """DWD hourly TU product: MESS_DATUM is yyyymmddhh in UTC, -999 marks a missing value."""
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.startswith("produkt_tu_stunde"))
        body = zf.read(name)
    df = pl.read_csv(io.BytesIO(body), separator=";", infer_schema=False)
    df = df.rename({c: c.strip() for c in df.columns})
    num = lambda c: pl.col(c).str.strip_chars().cast(pl.Float64)  # noqa: E731
    df = (
        df.select(
            pl.col("STATIONS_ID").str.strip_chars().cast(pl.Int64).alias("station_id"),
            (pl.col("MESS_DATUM").str.strip_chars() + "00").str.strptime(pl.Datetime("us"), "%Y%m%d%H%M").alias("hour_utc"),
            pl.when(num("TT_TU") > -999).then(num("TT_TU")).alias("temp_ref"),
            pl.when(num("RF_TU") > -999).then(num("RF_TU")).alias("rh_ref"),
        )
        .unique("hour_utc", keep="last")
        .sort("hour_utc")
    )
    return BronzeWeather.validate(df)


def merge_into(path: Path, df: pl.DataFrame, keys: list[str], partition: str | None = None) -> None:
    """Idempotent upsert: insert new keys, update existing ones."""
    if df.is_empty():
        return
    if not (path / "_delta_log").exists():
        path.mkdir(parents=True, exist_ok=True)
        df.write_delta(path, delta_write_options={"partition_by": [partition]} if partition else None)
        return
    predicate = " AND ".join(f"t.{k} = s.{k}" for k in keys)
    if partition:
        values = ", ".join(f"'{v}'" for v in df[partition].unique().to_list())
        predicate += f" AND t.{partition} IN ({values})"
    (
        DeltaTable(str(path))
        .merge(df.to_arrow(), predicate=predicate, source_alias="s", target_alias="t")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute()
    )


def load_bronze(raw_dir: Path, lake: Path, start: date, end: date, station_id: int, utc_offset: int) -> dict[str, int]:
    stats = {"pm_rows": 0, "climate_rows": 0, "quarantined": 0}
    for day in daterange(start, end):
        pm, bad_pm = parse_pm_day(raw_dir, day)
        clim, bad_clim = parse_climate_day(raw_dir, day)
        merge_into(lake / "bronze_pm", pm, ["sensor_id", "ts"], "day")
        merge_into(lake / "bronze_climate", clim, ["sensor_id", "ts"], "day")
        bad = (
            pl.concat([b for b in (bad_pm, bad_clim) if not b.is_empty()], how="diagonal_relaxed")
            if (not bad_pm.is_empty() or not bad_clim.is_empty())
            else pl.DataFrame()
        )
        if not bad.is_empty():
            bad = bad.with_columns(pl.lit(day.isoformat()).alias("day"))
            merge_into(lake / "quarantine", bad, ["source", "sensor_id", "timestamp", "reason"], "day")
        stats["pm_rows"] += pm.height
        stats["climate_rows"] += clim.height
        stats["quarantined"] += bad.height
    ref_paths = find_raw(raw_dir, ["uba"], f"{station_id}_*_{start}_{end}.json")
    ref = parse_reference(ref_paths, station_id, utc_offset)
    merge_into(lake / "bronze_reference", ref, ["hour_utc"])
    stats["reference_hours"] = ref.height
    weather_zip = next(iter(find_raw(raw_dir, ["dwd"], "stundenwerte_TU_*_akt.zip")), None)
    if weather_zip is not None:
        weather = parse_weather(weather_zip)
        lo = datetime.combine(start, time.min)
        hi = datetime.combine(end + timedelta(days=1), time.min)
        weather = weather.filter(pl.col("hour_utc").is_between(lo, hi, closed="left"))
        merge_into(lake / "bronze_weather", weather, ["hour_utc"])
        stats["weather_hours"] = weather.height
    log.info("bronze %s..%s %s", start, end, stats)
    return stats
