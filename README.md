# OFA Ethereum analysis

Code for the measurements in the paper: builder concentration under different
identity units, external identity attribution, historical order-flow shares and
conditional run-length (consecutive-win) inference.

## Setup

Linux, Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s analysis -p 'test_*.py'
```

## Data

Data packages are in a read-only
[OneDrive folder](https://1drv.ms/f/c/cdc0f83bf736d892/IgD5jmPgG7mPR4tunC0UTX4zAZj9UtpOkchfKvWCPQ_mO-o).
[DATASETS.md](DATASETS.md) describes the packages and input formats.

## Analysis

Output directories must not already exist. Use `--help` for all options; small
permutation counts are enough for quick checks.

### 1. Calibration and run-length inference

```bash
python scripts/calibrate_persistence_null.py \
  --payloads data/xatu/common_support.parquet --out-dir outputs/calibration \
  --null-replications 1000 --power-replications 300 \
  --permutations 999 --workers 48

python scripts/adjust_persistence_threshold_family.py \
  --payloads data/xatu/common_support.parquet --out-dir outputs/full-sample \
  --permutations 10000 --workers 48

python scripts/compute_xatu_common_support.py \
  --payloads data/xatu/common_support.parquet --out-dir outputs/windows \
  --permutations 10000 --workers 48
```

- Calibration runs a fresh permutation test on every simulated dataset.
- Inference uses a joint maximum statistic over 14 coordinates (seven run
  thresholds, two identity units).
- The window analysis covers 8 widths (50-10,000 slots), 2 origins and 2 run
  definitions; `across_settings_p` applies the 32-setting correction.

### 2. External identities

```bash
python analysis/analyze_identity.py --root data/evidence \
  --input data/xatu/common_support.parquet --out outputs/identity
python analysis/check_source_sensitivity.py --root data/evidence \
  --input data/xatu/common_support.parquet --out outputs/identity-sources
python analysis/run_identity_windows.py \
  --payloads outputs/identity/analysis_labels.parquet \
  --out-dir outputs/external-windows --permutations 10000 --workers 48
python analysis/analyze_partition_decomposition.py \
  --root outputs/identity --input data/xatu/common_support.parquet \
  --out outputs/partition-differences
```

### 3. Contributions and simulation controls

```bash
python scripts/analyze_persistence_stability.py \
  --payloads outputs/identity/analysis_labels.parquet \
  --columns raw_key slot_tag historical_service historical_network \
  --out-dir outputs/contributions

python analysis/run_controls.py formal \
  --labels outputs/identity/analysis_labels.parquet \
  --output-root outputs/controls --workers 48
python analysis/run_controls.py resolution \
  --labels outputs/identity/analysis_labels.parquet \
  --output-root outputs/controls --workers 48
python analysis/summarize_control_endpoints.py \
  --input outputs/controls/controls_formal_v1 --out outputs/control-endpoints

python analysis/run_joint_controls.py formal \
  --labels outputs/identity/analysis_labels.parquet \
  --baseline outputs/controls/controls_formal_v1 \
  --output-root outputs/joint-controls --workers 48
```

- Contributions: by run-start identity and period, with exact within-window
  expectations.
- Controls: stationary, smooth and step-changing shares, and three copy
  strengths; 200 samples x 1,999 permutations (resolution check: first 20
  samples x 9,999).
- Joint controls: smooth or step-changing shares combined with copy
  probabilities 0.005, 0.020 and 0.050, paired with the zero-copy baselines
  (first 100 samples, same permutation streams).

### 4. Historical summaries

```bash
python analysis/summarize_history.py final \
  --input data/history/final_v1 --out outputs/history
```

## License

[Apache License 2.0](LICENSE).
