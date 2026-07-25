# OFA Ethereum analysis

A compact collection of Python scripts for measuring Ethereum builder
concentration and short-run persistence from public
[Xatu](https://ethpandaops.io/data/xatu/) datasets. The repository contains
analysis code only; raw data, caches, and generated outputs are intentionally
excluded.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Workflow

1. Download relay payloads and compute the key-level baseline:

   ```bash
   python scripts/compute_xatu_builder_persistence.py \
     --cache-dir data/xatu-relay \
     --out-dir outputs/key-baseline \
     --start 2024-09-16 \
     --end 2025-10-05 \
     --permutations 1000
   ```

2. Join the relay payloads with canonical execution blocks to construct the
   normalized `extra_data` identity proxy:

   ```bash
   python scripts/compute_xatu_extra_data_sensitivity.py \
     --payload-cache data/xatu-relay \
     --cache-dir data/xatu-canonical \
     --out-dir outputs/identity-proxy \
     --permutations 1000
   ```

   The downstream scripts use
   `outputs/identity-proxy/xatu_canonical_payloads_by_key_reduced.parquet`.
   It contains one canonical payload per retained slot and the columns
   `slot`, `block_number`, `slot_start_date_time`, `builder_pubkey`,
   `extra_data_tag`, and `builder_identity_proxy`.

3. Run focused robustness checks as needed:

   ```bash
   python scripts/compute_xatu_common_support.py \
     --payloads outputs/identity-proxy/xatu_canonical_payloads_by_key_reduced.parquet \
     --out-dir outputs/common-support

   python scripts/validate_builder_identity_proxy.py \
     --payloads outputs/identity-proxy/xatu_canonical_payloads_by_key_reduced.parquet \
     --out-dir outputs/identity-validation

   python scripts/analyze_persistence_stability.py \
     --payloads outputs/identity-proxy/xatu_canonical_payloads_by_key_reduced.parquet \
     --out-dir outputs/stability

   python scripts/calibrate_persistence_null.py \
     --payloads outputs/identity-proxy/xatu_canonical_payloads_by_key_reduced.parquet \
     --out-dir outputs/null-calibration

   python scripts/adjust_persistence_threshold_family.py \
     --payloads outputs/identity-proxy/xatu_canonical_payloads_by_key_reduced.parquet \
     --out-dir outputs/threshold-family
   ```

Use smaller permutation counts for smoke tests. The defaults in the
computationally intensive scripts are intended for full runs and may require
multiple CPU cores.

## Included scripts

- `compute_xatu_builder_persistence.py`: downloads public relay payloads and
  computes monthly concentration and run-length statistics.
- `compute_xatu_extra_data_sensitivity.py`: joins canonical blocks and compares
  public-key and normalized `extra_data` identity units.
- `compute_xatu_common_support.py`: evaluates both identity units on identical
  slot support.
- `validate_builder_identity_proxy.py`: checks key-to-tag consistency,
  aggregation, temporal stability, and concentration sensitivity.
- `analyze_persistence_stability.py`: chronological, leave-one-identity-out,
  bootstrap, and permutation stability checks.
- `calibrate_persistence_null.py`: simulation-based null calibration and power
  analysis.
- `adjust_persistence_threshold_family.py`: family-wise permutation inference
  across run-length thresholds.

All scripts expose their parameters through `--help`. Randomized procedures use
explicit seeds and write machine-readable CSV or JSON summaries alongside
figures.

## Data and outputs

Xatu data are downloaded only when a script is run. Confirm the upstream data
terms and availability before large downloads. Local datasets, Parquet files,
NumPy arrays, caches, and generated outputs are ignored by Git.

## License

The code in this repository is available under the Apache License 2.0.
