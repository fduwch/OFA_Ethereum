#!/usr/bin/env python3
"""Family-wise permutation inference for builder-run thresholds.

The test uses the common-support payload sample.  Within every slot window, one
shared row permutation is applied to the public-key and extra-data-proxy labels.
This preserves each unit's exact marginal frequencies, their cross-unit
association, all observed slot gaps, and the dependence among run thresholds.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd


THRESHOLDS = np.array([2, 3, 4, 5, 6, 8, 10], dtype=np.int16)
UNITS = ("key", "proxy")

_KEY_LABELS: np.ndarray | None = None
_PROXY_LABELS: np.ndarray | None = None
_BREAK_BEFORE: np.ndarray | None = None
_WINDOW_STARTS: np.ndarray | None = None
_WINDOW_ENDS: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payloads", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--window-slots", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


def threshold_counts(labels: np.ndarray, break_before: np.ndarray) -> np.ndarray:
    if len(labels) == 0:
        return np.zeros(len(THRESHOLDS), dtype=np.int32)
    continuation = (~break_before[1:]) & (labels[1:] == labels[:-1])
    run_starts = np.r_[True, ~continuation]
    starts = np.flatnonzero(run_starts)
    lengths = np.diff(np.r_[starts, len(labels)])
    return np.array([(lengths >= k).sum() for k in THRESHOLDS], dtype=np.int32)


def _permutation_batch(task: tuple[int, int]) -> np.ndarray:
    seed, repetitions = task
    assert _KEY_LABELS is not None
    assert _PROXY_LABELS is not None
    assert _BREAK_BEFORE is not None
    assert _WINDOW_STARTS is not None
    assert _WINDOW_ENDS is not None

    rng = np.random.default_rng(seed)
    output = np.empty((repetitions, len(UNITS), len(THRESHOLDS)), dtype=np.int32)
    base_order = np.arange(len(_KEY_LABELS), dtype=np.int32)
    for repetition in range(repetitions):
        order = base_order.copy()
        for start, end in zip(_WINDOW_STARTS, _WINDOW_ENDS):
            rng.shuffle(order[start:end])
        output[repetition, 0] = threshold_counts(
            _KEY_LABELS[order], _BREAK_BEFORE
        )
        output[repetition, 1] = threshold_counts(
            _PROXY_LABELS[order], _BREAK_BEFORE
        )
    return output


def finite_p(samples: np.ndarray, observed: float, upper: bool) -> float:
    if upper:
        exceedances = np.count_nonzero(samples >= observed)
    else:
        exceedances = np.count_nonzero(samples <= observed)
    return float((1 + exceedances) / (len(samples) + 1))


def finite_two_sided_p(samples: np.ndarray, observed: float) -> float:
    """Return the doubled-smaller-tail finite-sample permutation p-value."""
    upper = finite_p(samples, observed, True)
    lower = finite_p(samples, observed, False)
    return min(1.0, 2.0 * min(upper, lower))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(
        args.payloads,
        columns=["slot", "builder_pubkey", "extra_data_tag"],
    ).sort_values("slot")
    frame["builder_pubkey"] = frame.builder_pubkey.astype(str).str.casefold()
    frame["extra_data_tag"] = frame.extra_data_tag.fillna("").astype(str)
    frame["proxy"] = np.where(
        frame.extra_data_tag.str.len() > 0,
        "tag:" + frame.extra_data_tag,
        "key:" + frame.builder_pubkey,
    )

    slots = frame.slot.to_numpy(dtype=np.int64)
    window_ids = slots // args.window_slots
    window_break = np.r_[True, window_ids[1:] != window_ids[:-1]]
    continuity_break = np.r_[True, np.diff(slots) != 1]
    break_before = window_break | continuity_break
    starts = np.r_[0, np.flatnonzero(window_break[1:]) + 1].astype(np.int64)
    ends = np.r_[starts[1:], len(frame)].astype(np.int64)

    key_labels = pd.factorize(frame.builder_pubkey, sort=True)[0].astype(np.int32)
    proxy_labels = pd.factorize(frame.proxy, sort=True)[0].astype(np.int32)
    observed = np.stack(
        [
            threshold_counts(key_labels, break_before),
            threshold_counts(proxy_labels, break_before),
        ]
    )

    global _KEY_LABELS, _PROXY_LABELS, _BREAK_BEFORE
    global _WINDOW_STARTS, _WINDOW_ENDS
    _KEY_LABELS = key_labels
    _PROXY_LABELS = proxy_labels
    _BREAK_BEFORE = break_before
    _WINDOW_STARTS = starts
    _WINDOW_ENDS = ends

    task_count = min(args.permutations, max(args.workers * 4, 1))
    repetitions = np.full(task_count, args.permutations // task_count, dtype=int)
    repetitions[: args.permutations % task_count] += 1
    children = np.random.SeedSequence(args.seed).spawn(task_count)
    tasks = [
        (int(child.generate_state(1)[0]), int(count))
        for child, count in zip(children, repetitions)
    ]
    with mp.get_context("fork").Pool(processes=args.workers) as pool:
        simulations = np.concatenate(
            pool.map(_permutation_batch, tasks, chunksize=1), axis=0
        )

    means = simulations.mean(axis=0)
    standard_deviations = simulations.std(axis=0, ddof=1)
    valid = standard_deviations > 0
    observed_z = np.full_like(means, np.nan, dtype=float)
    simulated_z = np.full_like(simulations, np.nan, dtype=float)
    observed_z[valid] = (observed[valid] - means[valid]) / standard_deviations[valid]
    simulated_z[:, valid] = (
        simulations[:, valid] - means[valid]
    ) / standard_deviations[valid]

    absolute_simulated_z = np.abs(simulated_z)
    per_unit_max = np.nanmax(absolute_simulated_z, axis=2)
    family_max = np.nanmax(
        absolute_simulated_z.reshape(len(simulations), -1), axis=1
    )
    family_critical_95 = float(np.quantile(family_max, 0.95, method="higher"))

    records: list[dict[str, object]] = []
    for unit_index, unit in enumerate(UNITS):
        for threshold_index, threshold in enumerate(THRESHOLDS):
            samples = simulations[:, unit_index, threshold_index]
            z_value = observed_z[unit_index, threshold_index]
            absolute_z = abs(z_value)
            records.append(
                {
                    "unit": unit,
                    "threshold": int(threshold),
                    "observed": int(observed[unit_index, threshold_index]),
                    "permutation_mean": float(means[unit_index, threshold_index]),
                    "permutation_sd": float(standard_deviations[unit_index, threshold_index]),
                    "z": float(z_value),
                    "raw_upper_p": finite_p(
                        samples, observed[unit_index, threshold_index], True
                    ),
                    "raw_lower_p": finite_p(
                        samples, observed[unit_index, threshold_index], False
                    ),
                    "raw_two_sided_p": finite_two_sided_p(
                        samples, observed[unit_index, threshold_index]
                    ),
                    "unit_family_two_sided_p": float(
                        (
                            1
                            + np.count_nonzero(
                                per_unit_max[:, unit_index] >= absolute_z
                            )
                        )
                        / (len(simulations) + 1)
                    ),
                    "all_tests_family_two_sided_p": float(
                        (1 + np.count_nonzero(family_max >= absolute_z))
                        / (len(simulations) + 1)
                    ),
                    "simultaneous_95_lower": float(
                        means[unit_index, threshold_index]
                        - family_critical_95
                        * standard_deviations[unit_index, threshold_index]
                    ),
                    "simultaneous_95_upper": float(
                        means[unit_index, threshold_index]
                        + family_critical_95
                        * standard_deviations[unit_index, threshold_index]
                    ),
                    "permutations": int(len(simulations)),
                }
            )

    summary = pd.DataFrame.from_records(records)
    summary.to_csv(args.out_dir / "threshold_family_summary.csv", index=False)
    np.save(args.out_dir / "threshold_family_permutation_counts.npy", simulations)
    metadata = {
        "input": str(args.payloads),
        "rows": int(len(frame)),
        "window_slots": int(args.window_slots),
        "thresholds": THRESHOLDS.tolist(),
        "units": list(UNITS),
        "permutations": int(args.permutations),
        "workers": int(args.workers),
        "seed": int(args.seed),
        "family_critical_95": family_critical_95,
        "procedure": (
            "Shared within-window row permutations; two-sided studentized "
            "max-T adjustment across both identity units and all thresholds."
        ),
        "results": summary.to_dict("records"),
    }
    (args.out_dir / "threshold_family_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
