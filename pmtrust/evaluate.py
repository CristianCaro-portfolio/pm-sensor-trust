"""Scoring against the reference station and a fault injection test for the QC rules."""

from __future__ import annotations

import numpy as np
import polars as pl

from pmtrust.qc import CLIMATE_FLAGS, PM_FLAGS, apply_qc

VARIANTS = {
    "raw_mean": "Network mean, no QC",
    "raw_median": "Network median, no QC",
    "qc": "Median of trusted sensors",
    "qc_linear": "Trusted + slope and intercept",
    "qc_scale": "Trusted + scale",
    "corrected": "Trusted + kappa humidity + scale (this repo)",
}
VARIANT_COLUMNS = {
    "raw_mean": "pm25_raw_mean",
    "raw_median": "pm25_raw_median",
    "qc": "pm25_qc",
    "qc_linear": "pm25_qc_linear",
    "qc_scale": "pm25_qc_scale",
    "corrected": "pm25_corrected",
}
RH_BINS = [(0, 70, "<70%"), (70, 85, "70-85%"), (85, 95, "85-95%"), (95, 101, ">=95% (fog)")]


def metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    err = pred - ref
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((ref - ref.mean()) ** 2)) or 1.0
    return {
        "n": int(len(ref)),
        "mae": round(float(np.mean(np.abs(err))), 2),
        "rmse": round(float(np.sqrt(np.mean(err**2))), 2),
        "bias": round(float(np.mean(err)), 2),
        "r2": round(1 - ss_res / ss_tot, 3),
    }


def best_lag(city: pl.DataFrame, reference: pl.DataFrame, max_lag: int = 3) -> dict[str, float]:
    """Correlation of the network median with the reference at hourly lags.

    A wrong time zone assumption shows up as a peak away from zero.
    """
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        shifted = reference.with_columns(pl.col("hour_utc") + pl.duration(hours=lag))
        j = city.join(shifted, on="hour_utc").drop_nulls(["pm25_raw_median", "ref_pm25"])
        if j.height > 10:
            out[lag] = round(float(np.corrcoef(j["pm25_raw_median"], j["ref_pm25"])[0, 1]), 4)
    return out


def score(city: pl.DataFrame, reference: pl.DataFrame) -> dict:
    """Score every variant on the same hours: published, reference present, every variant present."""
    cols = list(VARIANT_COLUMNS.values())
    j = city.join(reference, on="hour_utc").filter(pl.col("ref_pm25").is_not_null())
    all_hours = j.height
    common = j.filter(pl.col("published")).drop_nulls(cols)
    result = {
        "hours_with_reference": all_hours,
        "hours_published": common.height,
        "coverage": round(common.height / max(all_hours, 1), 3),
        "ref_mean": round(float(common["ref_pm25"].mean() or 0), 2),
        "variants": {},
        "by_rh": {},
    }
    ref = common["ref_pm25"].to_numpy()
    for name, col in VARIANT_COLUMNS.items():
        result["variants"][name] = metrics(common[col].to_numpy(), ref)

    # raw readings at every hour, including the ones QC would not publish (fog)
    raw_all = j.drop_nulls(["pm25_raw_mean", "rh_ref"])
    for lo, hi, label in RH_BINS:
        part = raw_all.filter(pl.col("rh_ref").is_between(lo, hi, closed="left"))
        sub = common.filter(pl.col("rh_ref").is_between(lo, hi, closed="left"))
        result["by_rh"][label] = {
            "raw_mean_all_hours": metrics(part["pm25_raw_mean"].to_numpy(), part["ref_pm25"].to_numpy()) if part.height else None,
            "corrected": metrics(sub["pm25_corrected"].to_numpy(), sub["ref_pm25"].to_numpy()) if sub.height else None,
            "raw_mean_ratio": round(float((part["pm25_raw_mean"] / (part["ref_pm25"] + 1)).median() or 0), 2)
            if part.height
            else None,
        }
    return result


def inject_faults(silver: pl.DataFrame, baseline: pl.DataFrame, seed: int = 7) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Plant known faults on real readings that passed QC, return (silver, labels).

    stuck      a sensor freezes on one value for 8 hours
    spike      a single hour jumps by 8x + 40 ug/m3 (insect, dust burst, firmware glitch)
    bias       a sensor reads 3x for the whole window (fouled optics, wrong unit)
    rh_fail    a humidity sensor reads a constant 14% (dead DHT22)
    """
    rng = np.random.default_rng(seed)
    ok = baseline.filter(pl.col("trusted")).select("location_id", "hour_utc")
    locs = ok["location_id"].unique().sort().to_numpy().copy()
    rng.shuffle(locs)
    bias_locs, rh_locs, rest = locs[:3], locs[3:6], locs[6:]

    labels = []
    df = silver.with_columns(pl.lit(None, pl.String).alias("fault"))

    df = df.with_columns(
        pl.when(pl.col("location_id").is_in(bias_locs.tolist())).then(pl.col("pm25") * 3).otherwise(pl.col("pm25")).alias("pm25"),
        pl.when(pl.col("location_id").is_in(bias_locs.tolist())).then(pl.lit("bias")).otherwise(pl.col("fault")).alias("fault"),
    ).with_columns(pl.max_horizontal("pm10", "pm25").alias("pm10"))

    df = df.with_columns(
        pl.when(pl.col("location_id").is_in(rh_locs.tolist())).then(pl.lit(14.0)).otherwise(pl.col("rh")).alias("rh"),
        pl.when(pl.col("location_id").is_in(rh_locs.tolist())).then(pl.lit("rh_fail")).otherwise(pl.col("fault")).alias("fault"),
    )

    ok_rest = ok.filter(pl.col("location_id").is_in(rest.tolist()))
    # stuck: 8 consecutive hours on 6 sensors
    starts = ok_rest.sample(n=6, seed=seed)
    stuck_keys = []
    for loc, h0 in starts.rows():
        for k in range(8):
            stuck_keys.append((loc, h0 + np.timedelta64(k, "h")))
    stuck = pl.DataFrame(stuck_keys, schema={"location_id": pl.Int64, "hour_utc": pl.Datetime("us")}, orient="row").with_columns(
        pl.lit(True).alias("_stuck")
    )
    first = (
        df.join(starts.with_columns(pl.lit(True).alias("_s")), on=["location_id", "hour_utc"], how="left")
        .filter(pl.col("_s"))
        .select("location_id", pl.col("pm25").alias("_frozen"))
    )
    df = (
        df.join(stuck, on=["location_id", "hour_utc"], how="left")
        .join(first, on="location_id", how="left")
        .with_columns(
            pl.when(pl.col("_stuck")).then(pl.col("_frozen")).otherwise(pl.col("pm25")).alias("pm25"),
            pl.when(pl.col("_stuck")).then(pl.lit(1, pl.UInt32)).otherwise(pl.col("pm25_distinct")).alias("pm25_distinct"),
            pl.when(pl.col("_stuck")).then(pl.lit("stuck")).otherwise(pl.col("fault")).alias("fault"),
        )
        .drop("_stuck", "_frozen")
    )

    # spikes: 40 single hours on other sensors, away from the stuck runs
    spike = (
        ok_rest.join(stuck, on=["location_id", "hour_utc"], how="anti")
        .sample(n=40, seed=seed + 1)
        .with_columns(pl.lit(True).alias("_spike"))
    )
    df = (
        df.join(spike, on=["location_id", "hour_utc"], how="left")
        .with_columns(
            pl.when(pl.col("_spike")).then(pl.col("pm25") * 8 + 40).otherwise(pl.col("pm25")).alias("pm25"),
            pl.when(pl.col("_spike")).then(pl.lit("spike")).otherwise(pl.col("fault")).alias("fault"),
        )
        .with_columns(pl.max_horizontal("pm10", "pm25").alias("pm10"))
        .drop("_spike")
    )
    labels = df.filter(pl.col("fault").is_not_null()).select("location_id", "hour_utc", "fault")
    return df.drop("fault"), labels


def fault_test(silver: pl.DataFrame, edges: pl.DataFrame, qc_cfg: dict, seed: int = 7) -> dict:
    baseline = apply_qc(silver, edges, qc_cfg)
    injected, labels = inject_faults(silver, baseline, seed)
    flagged = apply_qc(injected, edges, qc_cfg)
    j = flagged.join(labels, on=["location_id", "hour_utc"], how="left").join(
        baseline.select("location_id", "hour_utc", pl.col("trusted").alias("was_trusted")),
        on=["location_id", "hour_utc"],
    )
    out: dict = {"seed": seed, "faults": {}}
    for fault, rule_group in (("stuck", PM_FLAGS), ("spike", PM_FLAGS), ("bias", PM_FLAGS), ("rh_fail", CLIMATE_FLAGS)):
        part = j.filter((pl.col("fault") == fault) & pl.col("was_trusted"))
        caught = part.filter(pl.any_horizontal(rule_group))
        rules = {r: int(part[r].sum()) for r in rule_group if part.height and part[r].sum() > 0}
        out["faults"][fault] = {
            "injected_hours": part.height,
            "caught_hours": caught.height,
            "recall": round(caught.height / max(part.height, 1), 3),
            "rules": rules,
        }
    clean = j.filter(pl.col("fault").is_null() & pl.col("was_trusted"))
    out["collateral_flag_rate"] = round(float((~clean["trusted"]).mean()), 4)
    out["clean_hours"] = clean.height
    return out
