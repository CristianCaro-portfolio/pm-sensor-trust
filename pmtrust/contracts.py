"""Data contracts between layers, enforced with Pandera on Polars frames.

Rows that cannot meet the contract never reach these schemas: bronze.py routes them to
the quarantine table with a reason. These schemas are the hard gate that fails the run
if a bug lets a bad row through.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl


class BronzePM(pa.DataFrameModel):
    sensor_id: int = pa.Field(gt=0)
    location_id: int = pa.Field(gt=0)
    lat: float = pa.Field(ge=-90, le=90)
    lon: float = pa.Field(ge=-180, le=180)
    ts: pl.Datetime = pa.Field(nullable=False)
    pm10: float = pa.Field(nullable=True)
    pm25: float = pa.Field(nullable=True)
    day: str = pa.Field(str_matches=r"^\d{4}-\d{2}-\d{2}$")

    class Config:
        strict = True
        unique = ["sensor_id", "ts"]


class BronzeClimate(pa.DataFrameModel):
    sensor_id: int = pa.Field(gt=0)
    location_id: int = pa.Field(gt=0)
    ts: pl.Datetime = pa.Field(nullable=False)
    temperature: float = pa.Field(nullable=True)
    humidity: float = pa.Field(nullable=True)
    day: str = pa.Field(str_matches=r"^\d{4}-\d{2}-\d{2}$")

    class Config:
        strict = True
        unique = ["sensor_id", "ts"]


class BronzeReference(pa.DataFrameModel):
    station_id: int = pa.Field(gt=0)
    hour_utc: pl.Datetime = pa.Field(nullable=False, unique=True)
    ref_pm10: float = pa.Field(nullable=True, ge=0)
    ref_pm25: float = pa.Field(nullable=True, ge=0)


class BronzeWeather(pa.DataFrameModel):
    station_id: int = pa.Field(gt=0)
    hour_utc: pl.Datetime = pa.Field(nullable=False, unique=True)
    temp_ref: float = pa.Field(nullable=True, ge=-40, le=50)
    rh_ref: float = pa.Field(nullable=True, ge=0, le=100)


class GoldCityHourly(pa.DataFrameModel):
    hour_utc: pl.Datetime = pa.Field(unique=True)
    n_sensors: int = pa.Field(ge=0)
    n_trusted: int = pa.Field(ge=0)
    published: bool
    pm25_raw_mean: float = pa.Field(nullable=True)
    pm25_qc: float = pa.Field(nullable=True, ge=0)
    pm25_corrected: float = pa.Field(nullable=True, ge=0)

    @pa.dataframe_check
    def trusted_le_total(cls, data) -> pl.LazyFrame:  # noqa: N805
        return data.lazyframe.select(pl.col("n_trusted") <= pl.col("n_sensors"))
