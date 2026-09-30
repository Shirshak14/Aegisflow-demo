# AegisFlow Phase 2: Data Quality & Temporal Distribution Report

**Dataset Key:** `cic_ids2017`  
**Temporal Duration:** 2017-07-03 08:56:22 to 2017-07-07 17:02:00 (104.09 hours)  
**Unique Monitored Hosts:** 9971  

---

## 1. Flow Ingestion & Cleaning Summary

| Metric | Count | Percentage |
|---|---|---|
| Raw Rows Loaded | 500,001 | 100.0% |
| Cleaned Rows Kept | 446,137 | 89.23% |
| Rows Dropped | 53,864 | 10.77% |

### Dropped Rows Breakdown

| Reason | Dropped Count |
|---|---|
| `missing_timestamp` | 46,256 |
| `exact_duplicate_flow` | 7,589 |
| `negative_duration` | 19 |
| `missing_source_ip` | 0 |
| `missing_destination_ip` | 0 |
| `infinite_values` | 0 |

---

## 2. Temporal Representation Layer

| Representation Level | Total Count | Attack-Positive | Benign | Attack Ratio |
|---|---|---|---|---|
| Cleaned Flows | 446,137 | 85,883 | 360,254 | 19.25% |
| Host Windows (60s / 30s stride) | 132,460 | 1,038 | 131,422 | 0.78% |
| Sequences (Len=10, Horizon=1) | 85,077 | 1,012 | 84,065 | 1.19% |

**Target gap (target start - final input end):** min 30.0 s, median 60.0 s, max 350670.0 s; nonpositive 0 sequences; overlaps 0. Horizon counts eligible same-host windows whose start is strictly after the final input window end.

---

## 3. Class & Attack Stage Distribution Across Pipeline Tiers

### Normalized Attack Classes

| Class | Flow Count | Window Dominant Count | Sequence Target Count |
|---|---|---|---|
| **Benign** | 360,254 | 131,422 | 84,065 |
| **Botnet** | 348 | 422 | 409 |
| **Brute Force** | 2,128 | 250 | 238 |
| **Denial of Service** | 57,610 | 186 | 185 |
| **Infiltration** | 5 | 10 | 11 |
| **Reconnaissance** | 25,419 | 40 | 39 |
| **Web Attack** | 373 | 130 | 130 |

### Inferred Cyber Kill-Chain Stages (Proxy)

| Kill-Chain Stage | Flow Count | Window Dominant Count | Sequence Target Count |
|---|---|---|---|
| **Benign** | 360,254 | 131,422 | 84,065 |
| **Command and Control** | 348 | 422 | 409 |
| **Impact** | 57,610 | 186 | 185 |
| **Initial Access** | 2,501 | 380 | 368 |
| **Lateral Movement** | 5 | 10 | 11 |
| **Reconnaissance** | 25,419 | 40 | 39 |

---

## 4. Chronological Train / Val / Test Split Breakdown

- **Val Cutoff Timestamp:** `2017-07-06 09:20:22`
- **Test Cutoff Timestamp:** `2017-07-07 09:15:22`

| Split Partition | Sequence Count | Percentage |
|---|---|---|
| Training (`train`) | 51,093 | 60.06% |
| Validation (`val`) | 11,970 | 14.07% |
| Testing (`test`) | 11,379 | 13.37% |
| Boundary Quarantined (`boundary_excluded`) | 10,635 | 12.50% |

### Positive Targets: Onset vs Continuation

Onset = attack-positive target with no attack-present window among the sequence inputs; continuation = attack-positive target whose inputs already contain attack traffic.

| Split Partition | Positive Targets | Onset | Continuation |
|---|---|---|---|
| `train` | 382 | 9 | 373 |
| `val` | 127 | 6 | 121 |
| `test` | 487 | 69 | 418 |
| `boundary_excluded` | 16 | 2 | 14 |

---

## 5. Structurally Unavailable Features

The following features are not reported by CIC-IDS2017 sensor output and are explicitly marked as unavailable in the feature registry (never fabricated):

- `ttl_mean` (retained as NA / omitted from numerical tensors)
- `ttl_std` (retained as NA / omitted from numerical tensors)
- `retransmission_count` (retained as NA / omitted from numerical tensors)

---

## 6. Critical Observations & Data Warnings

> [!WARNING]
> Attack stages (Reconnaissance, Initial Access, Lateral Movement, C2, Impact) are PROXY INFERENCES derived from dataset labels (configs/stages.yaml) and must not be presented as ground truth.
