"""Humidity correction, city-level aggregation and calibration against the reference.

Optical sensors such as the SDS011 count particles by scattered light. In humid air the
particles take up water and grow, so the sensor reports more mass than a reference
instrument, which dries the sample first. The kappa-Koehler model describes that growth
with one hygroscopicity parameter:

    mass_wet / mass_dry = 1 + kappa * aw / (1 - aw),   aw = RH / 100

so the dry estimate is the reading divided by that factor. kappa is chosen on a
calibration window and frozen; a single scale factor is fitted on top of it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class Calibration:
    kappa: float
    scale: float  # applied after the humidity correction
    cv_mae: float  # blocked two-fold error used to choose kappa
    qc_scale: float  # scale-only variant, no humidity term
    lin_slope: float  # affine variant, no humidity term
    lin_intercept: float
    hours: int

    def to_dict(self) -> dict:
        return asdict(self)


def growth_factor(rh: pl.Expr, kappa: float, rh_cap: float) -> pl.Expr:
    aw = pl.min_horizontal(rh, pl.lit(rh_cap)) / 100.0
    return 1.0 + kappa * aw / (1.0 - aw)


def city_hourly(flags: pl.DataFrame, kappa: float, rh_cap: float, min_sensors: int) -> pl.DataFrame:
    """One row per hour: naive network averages next to the QC'd and corrected estimates."""
    df = flags.with_columns((pl.col("pm25") / growth_factor(pl.col("rh_ref"), kappa, rh_cap)).alias("pm25_dry"))
    trusted = pl.col("trusted")
    return (
        df.group_by("hour_utc")
        .agg(
            pl.len().alias("n_sensors"),
            trusted.sum().cast(pl.Int64).alias("n_trusted"),
            pl.col("pm25").filter(~pl.col("f_range")).mean().alias("pm25_raw_mean"),
            pl.col("pm25").filter(~pl.col("f_range")).median().alias("pm25_raw_median"),
            pl.col("pm25").filter(trusted).median().alias("pm25_qc"),
            pl.col("pm25_dry").filter(trusted).median().alias("pm25_dry"),
            pl.col("rh_ref").first().alias("rh_ref"),
            pl.col("rh").filter(pl.col("climate_ok")).median().alias("rh_network"),
            pl.col("f_fog").mean().alias("fog_share"),
        )
        .with_columns(
            pl.col("n_sensors").cast(pl.Int64),
            (pl.col("n_trusted") >= min_sensors).alias("published"),
        )
        .sort("hour_utc")
    )


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _scale(x: np.ndarray, y: np.ndarray) -> float:
    """Least squares through the origin: y ~ s * x."""
    return float(x @ y / (x @ x))


def blocked_cv_mae(x: np.ndarray, y: np.ndarray) -> float:
    """Fit the scale on one half of the window, score on the other, both ways.

    Halves are contiguous in time, so the score reflects a later period the scale
    has not seen, which is how the calibration is used.
    """
    h = len(x) // 2
    errs = []
    for fit, test in ((slice(0, h), slice(h, None)), (slice(h, None), slice(0, h))):
        s = _scale(x[fit], y[fit])
        errs.append(float(np.mean(np.abs(s * x[test] - y[test]))))
    return float(np.mean(errs))


def _paired(city: pl.DataFrame, reference: pl.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    joined = (
        city.join(reference, on="hour_utc")
        .filter(pl.col("published") & pl.col("ref_pm25").is_not_null() & pl.col(col).is_not_null())
        .sort("hour_utc")
    )
    return joined[col].to_numpy(), joined["ref_pm25"].to_numpy()


def fit_calibration(flags: pl.DataFrame, reference: pl.DataFrame, cfg: dict, min_sensors: int) -> Calibration:
    """Choose kappa by blocked cross-validation, then fit the scale on the whole window.

    A scale through the origin is used instead of slope + intercept: the intercept
    learned in one season encodes that season's background and does not transfer
    (see results/metrics.json, variant qc_linear).
    """
    best: tuple[float, float] | None = None
    for kappa in cfg["kappa_grid"]:
        x, y = _paired(city_hourly(flags, kappa, cfg["rh_cap"], min_sensors), reference, "pm25_dry")
        err = blocked_cv_mae(x, y)
        if best is None or err < best[0]:
            best = (err, kappa)
    assert best is not None
    cv_mae, kappa = best
    x, y = _paired(city_hourly(flags, kappa, cfg["rh_cap"], min_sensors), reference, "pm25_dry")
    xq, yq = _paired(city_hourly(flags, 0.0, cfg["rh_cap"], min_sensors), reference, "pm25_qc")
    lin_slope, lin_intercept = _ols(xq, yq)
    return Calibration(
        kappa=kappa,
        scale=round(_scale(x, y), 4),
        cv_mae=round(cv_mae, 3),
        qc_scale=round(_scale(xq, yq), 4),
        lin_slope=round(lin_slope, 4),
        lin_intercept=round(lin_intercept, 4),
        hours=len(x),
    )


def apply_calibration(city: pl.DataFrame, cal: Calibration) -> pl.DataFrame:
    return city.with_columns(
        (pl.col("pm25_qc") * cal.lin_slope + cal.lin_intercept).clip(lower_bound=0).alias("pm25_qc_linear"),
        (pl.col("pm25_qc") * cal.qc_scale).alias("pm25_qc_scale"),
        (pl.col("pm25_dry") * cal.scale).alias("pm25_corrected"),
    )
