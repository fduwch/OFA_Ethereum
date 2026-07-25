#!/usr/bin/env python3
"""Stability analyses for the two principal persistence statistics.

The script operates on the common-support payload sample and evaluates public
keys at k=5 and normalized extra-data proxies at k=8.  It produces chronological
split estimates, leave-one-major-identity-out estimates, window-block bootstrap
intervals, and a high-resolution local-share permutation test.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import BLUE, DOUBLE_FIGSIZE, RED, apply_publication_style, finish_axis


_PERM_LABELS: np.ndarray | None = None
_PERM_BREAK_BEFORE: np.ndarray | None = None
_PERM_STARTS: np.ndarray | None = None
_PERM_ENDS: np.ndarray | None = None
_PERM_K: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payloads", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--window-slots", type=int, default=10_000)
    parser.add_argument("--time-periods", type=int, default=4)
    parser.add_argument("--top-identities", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=5_000)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260719)
    return parser.parse_args()


def count_runs_at_least(labels: np.ndarray, break_before: np.ndarray, k: int) -> int:
    """Count maximal label runs of length at least k."""
    n = len(labels)
    if n < k:
        return 0
    continuation = (~break_before[1:]) & (labels[1:] == labels[:-1])
    width = n - k + 1
    reaches_k = np.ones(width, dtype=bool)
    for offset in range(k - 1):
        reaches_k &= continuation[offset : offset + width]
    starts_here = np.r_[True, ~continuation[: width - 1]]
    return int(np.count_nonzero(reaches_k & starts_here))


def analytical_window_stats(
    frame: pd.DataFrame,
    label_column: str,
    k: int,
    window_slots: int,
) -> pd.DataFrame:
    working = frame[["slot", "slot_start_date_time", label_column]].sort_values("slot")
    labels = pd.factorize(working[label_column], sort=True)[0].astype(np.int32)
    slots = working.slot.to_numpy(dtype=np.int64)
    dates = working.slot_start_date_time.to_numpy()
    window_ids = slots // window_slots
    starts = np.r_[0, np.flatnonzero(window_ids[1:] != window_ids[:-1]) + 1]
    ends = np.r_[starts[1:], len(working)]
    records: list[dict[str, object]] = []
    for start, end in zip(starts, ends):
        local_labels = labels[start:end]
        local_slots = slots[start:end]
        break_before = np.r_[True, np.diff(local_slots) != 1]
        segment_starts = np.flatnonzero(break_before)
        segment_lengths = np.diff(np.r_[segment_starts, len(local_labels)])
        eligible = segment_lengths[segment_lengths >= k]
        eligible_count = len(eligible)
        eligible_excess = int((eligible - k).sum())
        _, counts = np.unique(local_labels, return_counts=True)
        shares = counts / counts.sum()
        expected = float(
            np.sum(np.power(shares, k) * (eligible_count + eligible_excess * (1.0 - shares)))
        )
        records.append(
            {
                "window_id": int(window_ids[start]),
                "n_slots": int(end - start),
                "start_time": pd.Timestamp(dates[start]),
                "end_time": pd.Timestamp(dates[end - 1]),
                "observed": count_runs_at_least(local_labels, break_before, k),
                "analytical_expected": expected,
            }
        )
    return pd.DataFrame(records)


def bootstrap_ratio(
    stats: pd.DataFrame, replicates: int, seed: int
) -> tuple[float, float, float]:
    observed = stats.observed.to_numpy(dtype=float)
    expected = stats.analytical_expected.to_numpy(dtype=float)
    estimate = float(observed.sum() / expected.sum())
    rng = np.random.default_rng(seed)
    ratios = np.empty(replicates, dtype=float)
    batch_size = 500
    for start in range(0, replicates, batch_size):
        size = min(batch_size, replicates - start)
        indices = rng.integers(0, len(stats), size=(size, len(stats)))
        ratios[start : start + size] = observed[indices].sum(axis=1) / expected[indices].sum(axis=1)
    low, high = np.quantile(ratios, [0.025, 0.975])
    return estimate, float(low), float(high)


def _permutation_batch(task: tuple[int, int]) -> np.ndarray:
    seed, repetitions = task
    assert _PERM_LABELS is not None
    assert _PERM_BREAK_BEFORE is not None
    assert _PERM_STARTS is not None
    assert _PERM_ENDS is not None
    assert _PERM_K is not None
    rng = np.random.default_rng(seed)
    output = np.empty(repetitions, dtype=np.int32)
    for index in range(repetitions):
        permuted = _PERM_LABELS.copy()
        for start, end in zip(_PERM_STARTS, _PERM_ENDS):
            rng.shuffle(permuted[start:end])
        output[index] = count_runs_at_least(permuted, _PERM_BREAK_BEFORE, _PERM_K)
    return output


def permutation_test(
    frame: pd.DataFrame,
    label_column: str,
    k: int,
    window_slots: int,
    permutations: int,
    workers: int,
    seed: int,
) -> tuple[int, np.ndarray]:
    global _PERM_LABELS, _PERM_BREAK_BEFORE, _PERM_STARTS, _PERM_ENDS, _PERM_K
    working = frame[["slot", label_column]].sort_values("slot")
    labels = pd.factorize(working[label_column], sort=True)[0].astype(np.int32)
    slots = working.slot.to_numpy(dtype=np.int64)
    window_ids = slots // window_slots
    group_break = np.r_[True, window_ids[1:] != window_ids[:-1]]
    continuity_break = np.r_[True, np.diff(slots) != 1]
    break_before = group_break | continuity_break
    starts = np.r_[0, np.flatnonzero(group_break[1:]) + 1]
    ends = np.r_[starts[1:], len(labels)]
    observed = count_runs_at_least(labels, break_before, k)

    _PERM_LABELS = labels
    _PERM_BREAK_BEFORE = break_before
    _PERM_STARTS = starts
    _PERM_ENDS = ends
    _PERM_K = k
    task_count = min(permutations, max(workers * 4, 1))
    repetitions = np.full(task_count, permutations // task_count, dtype=int)
    repetitions[: permutations % task_count] += 1
    seeds = np.random.SeedSequence(seed).spawn(task_count)
    tasks = [(int(child.generate_state(1)[0]), int(count)) for child, count in zip(seeds, repetitions)]
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        samples = pool.map(_permutation_batch, tasks, chunksize=1)
    return observed, np.concatenate(samples)


def summarize_estimate(
    stats: pd.DataFrame, bootstrap_replicates: int, seed: int
) -> dict[str, object]:
    estimate, low, high = bootstrap_ratio(stats, bootstrap_replicates, seed)
    return {
        "windows": int(len(stats)),
        "slots": int(stats.n_slots.sum()),
        "observed": int(stats.observed.sum()),
        "analytical_expected": float(stats.analytical_expected.sum()),
        "ratio": estimate,
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
    }


def plot_stability(temporal: pd.DataFrame, leave_out: pd.DataFrame, out_dir: Path) -> None:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_FIGSIZE)
    styles = {"key": (BLUE, "o"), "proxy": (RED, "s")}
    for unit, group in temporal.groupby("unit"):
        color, marker = styles[unit]
        x = group.period.to_numpy(dtype=int)
        y = group.ratio.to_numpy(dtype=float)
        yerr = np.vstack((y - group.bootstrap_ci_low, group.bootstrap_ci_high - y))
        axes[0].errorbar(x, y, yerr=yerr, color=color, marker=marker, capsize=2,
                         markersize=3, label=f"{unit}, k={int(group.threshold.iloc[0])}")
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    axes[0].set(xticks=sorted(temporal.period.unique()), xlabel="Chronological period",
                ylabel="Observed / expected")
    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        columnspacing=0.9,
        borderaxespad=0,
    )

    for unit, group in leave_out.groupby("unit"):
        color, marker = styles[unit]
        x = group.removed_rank.to_numpy(dtype=int)
        y = group.ratio.to_numpy(dtype=float)
        yerr = np.vstack((y - group.bootstrap_ci_low, group.bootstrap_ci_high - y))
        axes[1].errorbar(x, y, yerr=yerr, color=color, marker=marker, capsize=2,
                         markersize=3, label=f"{unit}, k={int(group.threshold.iloc[0])}")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    axes[1].set(xticks=sorted(leave_out.removed_rank.unique()),
                xticklabels=["All", "Remove 1", "2", "3", "4", "5"],
                xlabel="Identity removed by market-share rank", ylabel="Observed / expected")
    axes[1].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        columnspacing=0.9,
        borderaxespad=0,
    )
    for axis in axes:
        finish_axis(axis)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.25, top=0.75, wspace=0.34)
    fig.savefig(out_dir / "Fig_persistence_stability.pdf")
    fig.savefig(out_dir / "Fig_persistence_stability.png", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(
        args.payloads,
        columns=["slot", "block_number", "slot_start_date_time", "builder_pubkey", "extra_data_tag"],
    ).sort_values("slot")
    frame["slot_start_date_time"] = pd.to_datetime(frame.slot_start_date_time, utc=True)
    frame["builder_pubkey"] = frame.builder_pubkey.astype(str).str.casefold()
    frame["extra_data_tag"] = frame.extra_data_tag.fillna("").astype(str)
    frame["proxy"] = np.where(
        frame.extra_data_tag.str.len() > 0,
        "tag:" + frame.extra_data_tag,
        "key:" + frame.builder_pubkey,
    )
    specifications = [("key", "builder_pubkey", 5), ("proxy", "proxy", 8)]

    temporal_records: list[dict[str, object]] = []
    leave_out_records: list[dict[str, object]] = []
    bootstrap_records: list[dict[str, object]] = []
    permutation_records: list[dict[str, object]] = []
    selected_windows: dict[str, list[int]] = {}

    for unit_index, (unit, column, threshold) in enumerate(specifications):
        print(f"preparing {unit}, k={threshold}", flush=True)
        stats = analytical_window_stats(frame, column, threshold, args.window_slots)
        full = summarize_estimate(
            stats, args.bootstrap_replicates, args.seed + unit_index * 10_000
        )
        bootstrap_records.append({"unit": unit, "threshold": threshold, **full})

        window_partitions = np.array_split(stats.window_id.to_numpy(), args.time_periods)
        selected_windows[unit] = [int(value) for value in stats.window_id]
        for period_index, window_ids in enumerate(window_partitions, start=1):
            subset = stats[stats.window_id.isin(window_ids)]
            estimate = summarize_estimate(
                subset,
                args.bootstrap_replicates,
                args.seed + unit_index * 10_000 + period_index,
            )
            temporal_records.append(
                {
                    "unit": unit,
                    "threshold": threshold,
                    "period": period_index,
                    "start_time": subset.start_time.min(),
                    "end_time": subset.end_time.max(),
                    **estimate,
                }
            )

        top = frame[column].value_counts().head(args.top_identities)
        baseline = summarize_estimate(
            stats, args.bootstrap_replicates, args.seed + unit_index * 10_000 + 100
        )
        leave_out_records.append(
            {
                "unit": unit,
                "threshold": threshold,
                "removed_rank": 0,
                "removed_identity": "none",
                "removed_share": 0.0,
                **baseline,
            }
        )
        for rank, (identity, count) in enumerate(top.items(), start=1):
            reduced = frame[frame[column] != identity]
            reduced_stats = analytical_window_stats(reduced, column, threshold, args.window_slots)
            estimate = summarize_estimate(
                reduced_stats,
                args.bootstrap_replicates,
                args.seed + unit_index * 10_000 + 100 + rank,
            )
            leave_out_records.append(
                {
                    "unit": unit,
                    "threshold": threshold,
                    "removed_rank": rank,
                    "removed_identity": identity,
                    "removed_share": float(count / len(frame)),
                    **estimate,
                }
            )

        print(f"running {args.permutations} permutations for {unit}, k={threshold}", flush=True)
        observed, samples = permutation_test(
            frame,
            column,
            threshold,
            args.window_slots,
            args.permutations,
            args.workers,
            args.seed + unit_index * 1_000_000,
        )
        expected_mean = float(samples.mean())
        permutation_records.append(
            {
                "unit": unit,
                "threshold": threshold,
                "observed": observed,
                "permutation_expected": expected_mean,
                "permutation_ci_low": float(np.quantile(samples, 0.025)),
                "permutation_ci_high": float(np.quantile(samples, 0.975)),
                "observed_expected_ratio": float(observed / expected_mean),
                "p_upper": float((1 + np.count_nonzero(samples >= observed)) / (len(samples) + 1)),
                "permutations": int(len(samples)),
            }
        )
        np.save(args.out_dir / f"{unit}_k{threshold}_permutation_counts.npy", samples)
        print(f"completed {unit}, k={threshold}", flush=True)

    temporal = pd.DataFrame(temporal_records)
    leave_out = pd.DataFrame(leave_out_records)
    bootstrap = pd.DataFrame(bootstrap_records)
    permutations = pd.DataFrame(permutation_records)
    temporal.to_csv(args.out_dir / "temporal_stability.csv", index=False)
    leave_out.to_csv(args.out_dir / "leave_one_major_identity_out.csv", index=False)
    bootstrap.to_csv(args.out_dir / "window_bootstrap_summary.csv", index=False)
    permutations.to_csv(args.out_dir / "permutation_10000_summary.csv", index=False)
    plot_stability(temporal, leave_out, args.out_dir)
    summary = {
        "input": str(args.payloads),
        "window_slots": args.window_slots,
        "time_periods": args.time_periods,
        "top_identities": args.top_identities,
        "bootstrap_replicates": args.bootstrap_replicates,
        "permutations": args.permutations,
        "workers": args.workers,
        "seed": args.seed,
        "bootstrap": bootstrap.to_dict("records"),
        "permutation": permutations.to_dict("records"),
    }
    (args.out_dir / "persistence_stability_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
