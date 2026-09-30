# Data pipeline (Phase 1)

## Flow

```
raw dataset files
    -> DatasetAdapter.load_raw()      (aegisflow/ml/datasets/cic_ids2017.py)
    -> clean_canonical_frame()        (aegisflow/ml/preprocessing/cleaning.py)
    -> apply_stage_mapping()          (aegisflow/ml/preprocessing/labels.py)
    -> data/interim/<dataset>.parquet
    -> run_eda()                      (aegisflow/ml/eda.py) -> reports/eda/, reports/figures/
```

Driven by `python -m aegisflow ingest --dataset cic_ids2017` /
`python -m aegisflow eda --dataset cic_ids2017`, or both together via
`python scripts/prepare_data.py --dataset cic_ids2017`.

## Setting up CIC-IDS2017

1. `python scripts/download_data.py --dataset cic_ids2017` — prints the exact
   official instructions (also reproduced below).
2. Go to https://www.unb.ca/cic/datasets/ids-2017.html and download
   **`GeneratedLabelledFlows.zip`** (not `MachineLearningCSV.zip` — that
   variant drops IP/timestamp columns AegisFlow needs).
3. Unzip so the 8 per-day CSVs end up somewhere under
   `data/raw/cic_ids2017/` (the official archive nests them in a
   `TrafficLabelling ` folder; AegisFlow searches recursively, so you can
   keep that folder as-is).
4. `python -m aegisflow validate-dataset --dataset cic_ids2017` — confirms
   the files are found and readable before you spend time on a full ingest.
5. `python -m aegisflow preprocess --dataset cic_ids2017 --reingest --sample-size 500000`
   for a fast first pass, then run it without `--sample-size` for the full
   2.8M-row dataset once you're happy with the pipeline.

**How `--sample-size N` selects rows (CIC-IDS2017).** The adapter counts the
data lines in every source CSV. That count includes the ~288k blank padding
rows in `Thursday-WorkingHours-Morning-WebAttacks`, which are later dropped
as `missing_timestamp`. It applies one global fraction `f = N / total_lines`
to every file, and draws `round(f * lines_in_file)` rows uniformly without
replacement from the **whole** file, never its head. Draws use
`numpy.random.default_rng(random_seed)`, consumed in sorted file order, so a
given seed and file set always select the same rows. Only the selected lines
are parsed; labels and columns are untouched. Every file, and every part of
each capture day, is thinned by the same factor, so per-window flow counts
are about `f` times their full-data values. Use the full dataset for
count-sensitive experiments. (`preprocess --sample-size` without `--reingest`
still truncates an existing interim file to its first N rows by time; always
pass `--reingest` when sampling.)

**Timestamps.** The TrafficLabelling CSVs use a 12-hour clock with no AM/PM
marker (for example, `Friday-...-Afternoon-DDos` starts at `7/7/2017 3:30`).
Parsed hours below 8 are shifted +12 h (`3:30` → 15:30); hours 8–12 are
unchanged.

Expected directory layout:

```
data/raw/cic_ids2017/
    TrafficLabelling/
        Monday-WorkingHours.pcap_ISCX.csv
        Tuesday-WorkingHours.pcap_ISCX.csv
        Wednesday-workingHours.pcap_ISCX.csv
        Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
        Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
        Friday-WorkingHours-Morning.pcap_ISCX.csv
        Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
        Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

If the directory is missing or empty, every AegisFlow command that needs it
fails immediately with `DatasetNotFoundError` and reprints these
instructions — it never substitutes fake data.

## Canonical schema

See `aegisflow/schema.py` for the authoritative column list, and run:

```
python -m aegisflow feature-registry
```

for a Markdown table of every canonical column, where CIC-IDS2017 sources
it from, and whether it's required.

Two columns are structurally **not available** from CIC-IDS2017's
CICFlowMeter output and are always `NA` for this dataset: `ttl_mean` /
`ttl_std` (no TTL field in the CSV) and `retransmission_count` (not
reported directly). This is documented, not silently patched.

## Attack-stage mapping

`configs/stages.yaml` maps each of CIC-IDS2017's 15 raw labels to a
`normalized_attack_class` and an `attack_stage`. `Benign` is the only
`ground_truth` label; every attack-stage value is `inferred` — see that
file's header comment for the full reasoning and caveats (e.g. `Infiltration`
and `Heartbleed` being mapped to "Lateral Movement" is a weak proxy, flagged
as such). Ingestion raises `LabelMappingError` for any label not in this
file, so a schema change in the dataset can't silently produce wrong
stages.

## Cleaning

`aegisflow/ml/preprocessing/cleaning.py` drops (and counts) rows with:
missing timestamp, missing source/destination IP, negative flow duration,
any infinite numeric value, or exact duplicate flows (same timestamp +
5-tuple). Every drop is logged with a count; nothing is silently discarded.

## PCAP / NetFlow support

Only the CSV path is implemented in Phase 1. PCAP ingestion (via PyShark/
TShark) and NetFlow parsing are planned for Phase 2+ once the CSV pipeline
and windowing are proven; see `docs/architecture.md` phase table. When
added, PCAP support requires TShark installed separately (`apt install
tshark` on Linux/WSL, or the Wireshark installer on Windows with "Install
TShark" checked) — the adapter will detect its absence and fail with
install instructions rather than silently skip PCAP files.
