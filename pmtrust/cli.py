"""Command line entry point.

pmtrust run                     materialize all Dagster assets and checks in process
pmtrust report                  charts and summary.md from results/metrics.json
pmtrust sample                  rebuild the small offline sample under sample_data/
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import typer

from pmtrust.config import ROOT

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _env(raw: Path | None, lake: Path | None, results: Path | None, config: Path | None) -> None:
    for key, value in (("PMTRUST_RAW", raw), ("PMTRUST_LAKE", lake), ("PMTRUST_RESULTS", results), ("PMTRUST_CONFIG", config)):
        if value is not None:
            os.environ[key] = str(value)


@app.command()
def run(
    raw: Path = typer.Option(None, help="raw files (default data/raw)"),
    lake: Path = typer.Option(None, help="Delta tables (default data/lake)"),
    results: Path = typer.Option(None, help="metrics output (default results)"),
    config: Path = typer.Option(None, help="config file (default config.yaml)"),
) -> None:
    """Run the whole pipeline through Dagster and fail if a check fails."""
    _env(raw, lake, results, config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    import dagster as dg

    from pmtrust.defs import ASSETS, CHECKS, lake_from_env

    result = dg.materialize([*ASSETS, *CHECKS], resources={"lake": lake_from_env()}, raise_on_error=True)
    failed = []
    for ev in result.get_asset_check_evaluations():
        status = "pass" if ev.passed else "FAIL"
        typer.echo(f"check {ev.check_name:<24} {status}  {dict(ev.metadata) if not ev.passed else ''}")
        if not ev.passed:
            failed.append(ev.check_name)
    metrics = result.output_for_node("evaluation")
    for window, s in metrics["windows"].items():
        v = s["variants"]
        typer.echo(
            f"{window:<12} hours={s['hours_published']:>4} ref_mean={s['ref_mean']:>5}  "
            f"MAE naive={v['raw_mean']['mae']:>5}  qc={v['qc']['mae']:>5}  corrected={v['corrected']['mae']:>5}"
        )
    if failed:
        raise typer.Exit(code=1)


@app.command()
def report(results: Path = typer.Option(ROOT / "results"), docs: Path = typer.Option(ROOT / "docs")) -> None:
    """Draw the result charts and write results/summary.md."""
    from pmtrust.report import write_report

    write_report(json.loads((results / "metrics.json").read_text()), results, docs)


@app.command()
def sample(
    raw: Path = typer.Option(ROOT / "data" / "raw"),
    out: Path = typer.Option(ROOT / "sample_data"),
    sensors: int = typer.Option(14),
    days: int = typer.Option(3),
) -> None:
    """Copy a few real sensors and days into sample_data/ for tests and CI."""
    from pmtrust.sample import build_sample

    typer.echo(json.dumps(build_sample(raw, out, sensors, days), indent=1))


if __name__ == "__main__":
    app()
