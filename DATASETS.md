# Datasets

## Download

All packages are in a read-only
[OneDrive folder](https://1drv.ms/f/c/cdc0f83bf736d892/IgD5jmPgG7mPR4tunC0UTX4zAZj9UtpOkchfKvWCPQ_mO-o).

| File | Contents |
| --- | --- |
| `ofa-reproducibility.zip` | Inputs, identity evidence, historical aggregates, code and reference outputs |
| `upgrade-experiments-complete-controls-development-20261028.zip` | Multi-scale analysis, simulation controls, external identities, historical comparison and saved simulation outputs |
| `ofa-directional-supplement-20261030.zip` | Directional endpoint summaries for the controls package |
| `ofa-joint-controls-supplement-20261108.zip` | Joint drift-plus-dependence controls for the controls package |
| `source-data/` | Source collections; `INDEX.json` lists each archive |

Extract the first two archives into separate directories, because both use the
top-level directory `ofa-reproducibility/`. Each archive contains its own README
and file manifest.

## Inputs

| Collection | Contents | Local path |
| --- | --- | --- |
| Canonical relay payloads | Slots, execution blocks, builder public keys and normalized extra-data tags | `data/xatu/common_support.parquet` |
| External identity evidence | First-party key disclosures, registry snapshots and source manifests | `data/evidence/` |
| PoW rewards | Block number, timestamp and beneficiary address | `data/rewards/` |
| Historical monitoring | Retained monitor records, candidate transactions, hourly counts and canonical transaction denominators | `data/history/` |
| Simulation outputs | Observed run counts, permutation arrays and per-sample inference | `data/simulations/` |

Frozen inputs used by the identity and control runners:

- `common_support.parquet`: 2,332,997 rows, SHA-256
  `06bac99220ff55814d91d43a7d2082f6409968bd98078f4dec946bab485c7c78`.
- `analysis_labels.parquet` (output of the identity analysis): SHA-256
  `ed984f72d574ed69352ad983366d9df2c0325a02eb22523b71953384dad099cd`.

## Preparing the Xatu input

Relay and canonical-block exports are public at
[Xatu](https://ethpandaops.io/data/xatu/).

```bash
python scripts/compute_xatu_builder_persistence.py \
  --cache-dir data/xatu-relay --out-dir outputs/relay --prepare-only
python scripts/compute_xatu_extra_data_sensitivity.py \
  --payload-cache data/xatu-relay --cache-dir data/xatu-canonical \
  --out-dir outputs/canonical --prepare-only
```

The result is `outputs/canonical/xatu_canonical_payloads_by_key_reduced.parquet`.
Set the download period with `--start` and `--end` on the relay script.

## Notes

- Historical monitoring: the complete 2020-2022 raw monitor archive is not
  available. Retained candidate records and aggregates are provided, and the
  matched-hour analysis uses source-availability flags and canonical
  inclusion-hour denominators.
- External evidence: `data/evidence/` uses the `sources/`,
  `registries_complete/` and `archive_transfer/` layout; each snapshot has a
  manifest entry and checksum.
