"""Charts and a markdown summary built from results/metrics.json and the gold table."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

from pmtrust.config import ROOT  # noqa: E402
from pmtrust.evaluate import VARIANTS  # noqa: E402

INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e2dc"
REF, NAIVE, QC, OURS = "#3d3d3a", "#eb6834", "#1baf7a", "#2a78d6"


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def results_figure(metrics: dict, lake: Path, out: Path) -> None:
    gold = pl.read_delta(str(lake / "gold_city_hourly"))
    ref = pl.read_delta(str(lake / "bronze_reference")).select("hour_utc", "ref_pm25")
    week = gold.filter(pl.col("window") == "test_winter").join(ref, on="hour_utc").sort("hour_utc")

    fig = plt.figure(figsize=(12, 6.4), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.45, wspace=0.25)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(week["hour_utc"], week["ref_pm25"], color=REF, lw=2, label="UBA reference DEBW013")
    ax.plot(week["hour_utc"], week["pm25_raw_mean"], color=NAIVE, lw=1.4, label="Network mean, no QC")
    ax.plot(week["hour_utc"], week["pm25_corrected"], color=OURS, lw=1.6, label="Trusted + kappa + scale (this repo)")
    ax.set_title("PM2.5 in Stuttgart, held-out winter window (ug/m3, hourly)", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, ncols=3, loc="upper left")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    _style(ax)

    ax = fig.add_subplot(gs[1, 0])
    names = ["raw_mean", "qc", "qc_linear", "qc_scale", "corrected"]
    windows = [w for w in ("test_winter", "test_autumn") if w in metrics["windows"]]
    width = 0.38
    for i, w in enumerate(windows):
        vals = [metrics["windows"][w]["variants"][n]["mae"] for n in names]
        xs = [j + (i - 0.5) * width for j in range(len(names))]
        bars = ax.bar(
            xs,
            vals,
            width=width - 0.04,
            color=[NAIVE, QC, MUTED, MUTED, OURS],
            alpha=1.0 if i == 0 else 0.55,
            label=w.replace("_", " "),
        )
        for b, v in zip(bars, vals, strict=False):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(
        range(len(names)),
        ["naive\nmean", "QC\nmedian", "QC + slope\n+ intercept", "QC +\nscale", "QC + kappa\n+ scale"],
        fontsize=8,
    )
    ax.set_title("MAE vs reference (ug/m3); solid = winter, light = October", loc="left", fontsize=10, color=INK)
    _style(ax)

    ax = fig.add_subplot(gs[1, 1])
    labels, ratios = [], []
    for label, b in metrics["windows"]["test_winter"]["by_rh"].items():
        if b["raw_mean_ratio"] is not None:
            labels.append(label.replace(" (fog)", "\nfog"))
            ratios.append(b["raw_mean_ratio"])
    bars = ax.bar(labels, ratios, color=NAIVE, width=0.55)
    for b, v in zip(bars, ratios, strict=False):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_title("Raw sensor / reference by station humidity (winter)", loc="left", fontsize=10, color=INK)
    ax.set_xlabel("relative humidity at DWD Stuttgart-Schnarrenberg", fontsize=8, color=MUTED)
    _style(ax)

    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architecture_figure(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.4), dpi=150)
    ax.set_xlim(0, 12)
    ax.set_ylim(0.1, 3.4)
    ax.axis("off")
    boxes = [
        (0.2, 2.5, "Sources", "Sensor.Community\nSDS011 + DHT22/BME280\nUBA reference PM\nDWD weather station"),
        (2.6, 2.5, "Bronze (Delta)", "typed rows, MERGE on\n(sensor, ts) per day\nPandera contracts\nquarantine table"),
        (5.0, 2.5, "Silver (Delta)", "location-hour rows\nPolars 2.0 streaming\n11 QC flags\nneighbour graph"),
        (7.4, 2.5, "Gold (Delta)", "city PM2.5 per hour\nkappa humidity fix\nscale calibration\npublish if >= 5 trusted"),
        (9.8, 2.5, "Evaluation", "MAE / R2 vs UBA\nby humidity band\nfault injection\nlag alignment"),
    ]
    for x, y, title, body in boxes:
        ax.add_patch(plt.Rectangle((x, y - 1.5), 2.0, 2.2, fill=False, ec=OURS, lw=1.4))
        ax.text(x + 1.0, y + 0.45, title, ha="center", fontsize=10, color=INK, weight="bold")
        ax.text(x + 1.0, y - 0.45, body, ha="center", va="center", fontsize=7.5, color=INK, linespacing=1.5)
    for x in (2.2, 4.6, 7.0, 9.4):
        ax.annotate("", xy=(x + 0.38, 2.0), xytext=(x + 0.02, 2.0), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    ax.text(
        6.0,
        0.35,
        "Dagster assets + asset checks orchestrate every stage; the CLI runs the same graph in process (CI smoke test)",
        ha="center",
        fontsize=8.5,
        color=MUTED,
    )
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def summary_markdown(metrics: dict) -> str:
    lines = ["# Results", "", "Generated by `pmtrust report` from `results/metrics.json`.", ""]
    cal = metrics["calibration"]
    lines += [
        f"Calibration window fit: kappa = {cal['kappa']} (blocked CV MAE {cal['cv_mae']} ug/m3), scale = {cal['scale']}, "
        f"{cal['hours']} hours.",
        "",
        "| Window | Hours | Ref mean | " + " | ".join(VARIANTS[v] for v in VARIANTS) + " |",
        "|---|---|---|" + "---|" * len(VARIANTS),
    ]
    for w, s in metrics["windows"].items():
        cells = [f"{s['variants'][v]['mae']} / {s['variants'][v]['r2']}" for v in VARIANTS]
        lines.append(f"| {w} | {s['hours_published']} | {s['ref_mean']} | " + " | ".join(cells) + " |")
    lines += ["", "Cells are MAE (ug/m3) / R2 on the same hours for every variant.", ""]
    if "fault_test" in metrics:
        ft = metrics["fault_test"]
        lines += ["## Fault injection (test_winter)", "", "| Fault | Injected hours | Caught | Recall |", "|---|---|---|---|"]
        for f, r in ft["faults"].items():
            lines.append(f"| {f} | {r['injected_hours']} | {r['caught_hours']} | {r['recall']} |")
        lines += ["", f"Collateral flag rate on clean hours: {ft['collateral_flag_rate']}", ""]
    return "\n".join(lines)


def write_report(metrics: dict, results: Path, docs: Path) -> None:
    lake = Path(os.environ.get("PMTRUST_LAKE", ROOT / "data" / "lake"))
    docs.mkdir(parents=True, exist_ok=True)
    results_figure(metrics, lake, docs / "results.png")
    architecture_figure(docs / "architecture.png")
    (results / "summary.md").write_text(summary_markdown(metrics))
