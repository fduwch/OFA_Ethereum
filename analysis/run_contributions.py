"""Additive run-count contributions by identity and chronological period."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from direct_contribution_engine import direct_contributions, THRESHOLDS
from persistence_test import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payloads", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--columns", nargs="+")
    parser.add_argument("--windows", nargs="+", type=int,
                        default=[10000, 5000, 2500, 1000, 500, 250, 100, 50])
    parser.add_argument("--periods", type=int, default=4)
    args = parser.parse_args()
    if args.periods < 1 or len(set(args.windows)) != len(args.windows):
        parser.error("Use positive periods and distinct window widths")
    if any(w < 10 or w % 2 for w in args.windows):
        parser.error("Window widths must be even integers of at least ten")
    frame = pd.read_parquet(args.payloads).sort_values("slot").reset_index(drop=True)
    slots = frame.slot.to_numpy()
    if not np.issubdtype(slots.dtype, np.integer) or len(slots) == 0:
        raise ValueError("Nonempty integer slot support required")
    if (slots < 0).any() or (np.diff(slots) <= 0).any():
        raise ValueError("Slots must be nonnegative and unique")
    windows = np.unique(slots // 10000)
    if len(windows) < args.periods:
        raise ValueError("Fewer occupied 10000-slot windows than periods")
    period_map = {int(w): i for i, group in enumerate(np.array_split(windows, args.periods))
                  for w in group}
    periods = np.array([period_map[int(w)] for w in slots // 10000], dtype=np.int64)
    if args.columns:
        columns = args.columns
    else:
        if frame.builder_pubkey.isna().any():
            raise ValueError("Missing builder public key")
        key = frame.builder_pubkey.astype(str).str.casefold()
        tag = frame.extra_data_tag.fillna("").astype(str)
        frame["key"] = "key:" + key
        frame["proxy"] = np.where(tag.ne(""), "tag:" + tag, "key:" + key)
        columns = ["key", "proxy"]
    if frame[columns].isna().any().any():
        raise ValueError("Missing identity labels")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for unit in columns:
        codes, names = pd.factorize(frame[unit], sort=True)
        for width in args.windows:
            for offset in (0, width // 2):
                observed, expected, _, _ = direct_contributions(slots, codes, periods, width, offset)
                observed = np.concatenate([observed.sum(axis=1, keepdims=True), observed], axis=1)
                expected = np.concatenate([expected.sum(axis=1, keepdims=True), expected], axis=1)
                for d, definition in enumerate(("window_bounded", "gap_only")):
                    for period in range(args.periods + 1):
                        for label, identity in enumerate(names):
                            for k, threshold in enumerate(THRESHOLDS):
                                o, e = int(observed[d, period, label, k]), float(expected[d, period, label, k])
                                rows.append(dict(unit=unit, width_slots=width, offset_slots=offset,
                                    definition=definition, period=period, identity=str(identity),
                                    threshold=int(threshold), observed=o, exact_expectation=e,
                                    excess_runs=o-e, ratio=o/e if e > 0 else None))
    result = args.out_dir / "contributions.csv"
    pd.DataFrame(rows).to_csv(result, index=False)
    metadata = dict(rows=len(frame), units=columns, widths=args.windows, periods=args.periods,
                    period_zero="full_sample", run_assignment="starting_slot",
                    input_sha256=sha256(args.payloads), output_sha256=sha256(result))
    (args.out_dir / "completion.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
