# Datasets

The analyses use five data collections. Additional datasets and saved simulation
arrays will be uploaded to OneDrive; download links and file manifests will be
added here when available.

| Collection | Contents | Local path |
| --- | --- | --- |
| Canonical relay payloads | Slots, execution blocks, builder public keys and normalized extra-data tags | `data/xatu/common_support.parquet` |
| External identity evidence | First-party key disclosures, registry snapshots and source manifests | `data/evidence/` |
| PoW rewards | Block number, timestamp and beneficiary address | `data/rewards/` |
| Historical monitoring | Retained monitor records, candidate transactions, hourly counts and canonical transaction denominators | `data/history/` |
| Simulation outputs | Observed run counts, permutation arrays and per-sample inference | `data/simulations/` |

The frozen common-support input contains 2,332,997 rows. Its SHA-256 is
`06bac99220ff55814d91d43a7d2082f6409968bd98078f4dec946bab485c7c78`.
External analysis produces `analysis_labels.parquet`, whose SHA-256 is
`ed984f72d574ed69352ad983366d9df2c0325a02eb22523b71953384dad099cd`.
The external-identity and control runners use these frozen inputs.

Public relay and canonical-block exports are available from
[Xatu](https://ethpandaops.io/data/xatu/). To prepare a local input:

```bash
python scripts/compute_xatu_builder_persistence.py \
  --cache-dir data/xatu-relay --out-dir outputs/relay --prepare-only
python scripts/compute_xatu_extra_data_sensitivity.py \
  --payload-cache data/xatu-relay --cache-dir data/xatu-canonical \
  --out-dir outputs/canonical --prepare-only
```

The canonical common-support file is
`outputs/canonical/xatu_canonical_payloads_by_key_reduced.parquet`.
Download dates are set with `--start` and `--end` on the relay script.

Historical monitoring availability varies by period. The complete 2020–2022 raw
monitor archive is unavailable; retained candidate records and aggregates are
listed separately in the data package. The matched-hour analysis uses explicit
source-availability flags and canonical inclusion-hour denominators.

External evidence follows the `sources/`, `registries_complete/` and
`archive_transfer/` layout inside `data/evidence/`. Each source snapshot is
identified by its manifest entry and checksum.
