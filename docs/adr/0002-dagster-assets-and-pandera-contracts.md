# 0002. Dagster assets with checks, Pandera contracts between layers

Status: accepted

## Context

The value of this project is not one number but a set of claims: the bronze rows are
typed and unique, each hour has enough trusted sensors, the corrected series beats
the naive one on data the calibration never saw, and the reference station lines up
in time. Those claims should fail loudly, every run, in CI and in production.

## Decision

- Model the pipeline as Dagster software-defined assets (`bronze_tables`, `silver_qc`,
  `calibration`, `gold_city_hourly`, `evaluation`) and express the claims as asset
  checks (`enough_trusted_hours`, `publish_coverage`, `corrected_beats_naive`,
  `time_alignment`). `pmtrust run` materializes the graph in process and exits non
  zero when a check fails, so the CI smoke job is the same code path as a real run.
- Enforce schemas with Pandera on Polars frames at each layer boundary
  (`pmtrust/contracts.py`). Rows that cannot be parsed never reach the contract: they
  go to a `quarantine` Delta table with the reason, so nothing is dropped silently.

## Alternatives

- Airflow: task oriented, I used it in spark-self-heal; for a data product whose unit
  is a table, assets and checks describe the intent better and run locally without a
  scheduler.
- Prefect: good for flows, weaker built-in notion of asset lineage and checks.
- Great Expectations: heavier than needed for four schemas; Pandera keeps the contract
  next to the code and validates Polars natively.

## Consequences

- The whole graph and its checks are visible in `dagster dev -m pmtrust.defs`.
- Contracts are strict (`strict = True`), so an upstream column change fails the run
  instead of flowing through.
