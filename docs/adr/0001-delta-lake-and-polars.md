# 0001. Delta Lake tables written with Polars 2.0

Status: accepted

## Context

Sensor.Community publishes one CSV per sensor per day. Files arrive late, some are
re-published, and a download can fail half way. The pipeline must be safe to rerun on
any day without creating duplicates, and the layers (bronze, silver, gold) must be
queryable without a server. Everything has to run on a laptop and in CI.

## Decision

Store every layer as a Delta Lake table through `deltalake` (delta-rs, no JVM) and do
all transformations in Polars 2.0.

- Writes are `MERGE` on the natural key (`sensor_id, ts` in bronze, `location_id,
  hour_utc` in silver, `hour_utc` in gold). The merge predicate also names the day
  partition, so a reload of one day only touches that partition.
- Bronze is partitioned by day because that is the unit of arrival.
- Polars 2.0 (6 October 2026) runs lazy queries on the streaming engine by default,
  which keeps memory flat when the window grows from two weeks to a season.

## Alternatives

- Plain Parquet folders: no atomic upsert, a rerun would need delete-then-write logic.
- Apache Iceberg with PyIceberg: used in an earlier project of mine (lidar-line-clearance);
  its Python write path for row-level upserts is still less direct than delta-rs `merge`.
- DuckDB file: good for queries, but a single file is a weaker fit for partitioned
  append-heavy sensor data and for sharing tables with Spark later.

## Consequences

- `tests/test_pipeline.py::test_bronze_reload_is_idempotent` reloads the bronze layer and
  checks the row count does not change.
- Polars 2.0 no longer guarantees row order after joins and group-bys, so every step
  that depends on order (rolling windows, run lengths) sorts explicitly first.
