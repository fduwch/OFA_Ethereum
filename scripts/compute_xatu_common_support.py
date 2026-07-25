#!/usr/bin/env python3
"""Common-support sensitivity for Xatu builder identity measurements.

The primary identity analysis uses every slot that is valid for each observable
unit.  This script instead restricts both public-key and extra-data-proxy
measurements to the unique-key sample, so differences cannot be attributed to
the additional multi-key slots recovered by the proxy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import compute_xatu_builder_persistence as core


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payloads", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window-slots", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(args.payloads)
    required = {
        "slot",
        "block_number",
        "slot_start_date_time",
        "builder_pubkey",
        "builder_identity_proxy",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")
    frame = frame.drop_duplicates("slot").sort_values("slot").reset_index(drop=True)

    monthly_frames: list[pd.DataFrame] = []
    persistence_rows: list[dict[str, object]] = []
    units = ("builder_pubkey", "builder_identity_proxy")
    for unit_index, unit in enumerate(units):
        working = frame.copy()
        working["builder_pubkey"] = working[unit]

        monthly = core.concentration_by_month(working)
        monthly.insert(0, "label_unit", unit)
        monthly.insert(1, "support", "common_unique_key_slots")
        monthly_frames.append(monthly)

        observed, simulations, run_details = core.permutation_null(
            working,
            "slot",
            args.window_slots,
            args.permutations,
            args.seed + unit_index,
        )
        analytical, _ = core.analytic_expected(
            working,
            "slot",
            args.window_slots,
        )
        for threshold_index, threshold in enumerate(core.THRESHOLDS):
            simulated = simulations[:, threshold_index]
            p_ge = float(
                (1 + (simulated >= observed[threshold_index]).sum())
                / (1 + len(simulated))
            )
            p_le = float(
                (1 + (simulated <= observed[threshold_index]).sum())
                / (1 + len(simulated))
            )
            persistence_rows.append(
                {
                    "label_unit": unit,
                    "support": "common_unique_key_slots",
                    "continuity": "slot",
                    "window_slots": args.window_slots,
                    "threshold": int(threshold),
                    "observed_runs": int(observed[threshold_index]),
                    "analytical_expected_runs": float(analytical[threshold_index]),
                    "observed_expected_ratio": (
                        float(observed[threshold_index] / analytical[threshold_index])
                        if analytical[threshold_index] > 0
                        else np.nan
                    ),
                    "permutation_mean": float(simulated.mean()),
                    "permutation_ci_low": float(np.quantile(simulated, 0.025)),
                    "permutation_ci_high": float(np.quantile(simulated, 0.975)),
                    "permutation_p_ge_observed": p_ge,
                    "permutation_p_le_observed": p_le,
                    "permutation_p_two_sided": min(1.0, 2.0 * min(p_ge, p_le)),
                    "permutations": args.permutations,
                    "observed_maximal_runs": int(len(run_details)),
                }
            )
        print(f"completed common-support unit={unit}", flush=True)

    monthly_output = pd.concat(monthly_frames, ignore_index=True)
    persistence_output = pd.DataFrame(persistence_rows)
    monthly_output.to_csv(out_dir / "xatu_common_support_monthly.csv", index=False)
    persistence_output.to_csv(
        out_dir / "xatu_common_support_persistence.csv",
        index=False,
    )

    quality = {
        "support": "unique-key slots with both public-key and proxy labels",
        "slots": int(len(frame)),
        "first_slot": int(frame.slot.min()),
        "last_slot": int(frame.slot.max()),
        "window_slots": args.window_slots,
        "permutations": args.permutations,
        "seed": args.seed,
        "public_keys": int(frame.builder_pubkey.nunique()),
        "identity_proxies": int(frame.builder_identity_proxy.nunique()),
    }
    (out_dir / "xatu_common_support_quality.json").write_text(
        json.dumps(quality, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
