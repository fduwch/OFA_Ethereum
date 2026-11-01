# OFA Ethereum analysis

Python tools for Ethereum builder concentration, identity attribution, historical
order-flow measurements and conditional run-length inference.

## Setup

Linux, Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s analysis -p 'test_*.py'
```

## Data

See [DATASETS.md](DATASETS.md) for the input schema, public Xatu preparation
commands and dataset availability. Additional datasets will be uploaded to
OneDrive, with download links and manifests added here.

## Analysis

### Calibration and empirical inference

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

Calibration applies fresh inner permutations to each outer dataset. Inference
uses pooled standardization, inclusive ranks and a joint maximum statistic over
two identity units and seven run thresholds. Window analysis covers eight widths
from 50 to 10,000 slots, two origins and two run definitions; its
`across_settings_p` includes the 32-setting correction for that identity pair.

### External identities

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

### Contributions and simulation controls

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
```

Contributions are allocated by run-start identity and chronological period,
using exact within-window expectations. Controls comprise stationary IID,
smooth and step-changing IID shares, and three sticky-process strengths.
The formal stage uses 200 outer samples per model and 1,999 permutations;
the paired resolution stage uses the first 20 samples and 9,999 permutations.
Endpoint summaries separate positive and negative rejections for all thresholds
and for runs of at least eight or ten slots.

### Historical summaries

```bash
python analysis/summarize_history.py final \
  --input data/history/final_v1 --out outputs/history
```

All analysis output directories must be new. Use `--help` for individual
parameters. Small permutation counts are available for execution checks.

## License

[Apache License 2.0](LICENSE).
