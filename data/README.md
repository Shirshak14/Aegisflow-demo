# data/

- `raw/` — untouched dataset files exactly as downloaded. Never committed
  (see `.gitignore`); see `docs/data_pipeline.md` for how to populate
  `raw/cic_ids2017/`.
- `interim/` — output of `python -m aegisflow ingest`: canonical, cleaned,
  labeled Parquet, one file per dataset (`<dataset_key>.parquet`).
- `processed/` — output of Phase 2 windowing (not yet implemented): temporal
  windows ready for model training, plus `aegisflow.db` (SQLite metadata,
  Phase 6).
