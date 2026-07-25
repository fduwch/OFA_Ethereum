#!/usr/bin/env python3
"""Calibrate the share-conditioned persistence test and compare nulls.

All simulations run on a stratified set of empirical 10,000-slot windows. The
window-specific label distributions and observed slot gaps determine the
synthetic profiles. A sticky categorical process injects serial dependence
while retaining the window-specific marginal distribution in expectation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import (
    BLUE,
    DOUBLE_FIGSIZE,
    GREEN,
    LIGHT_BLUE,
    RED,
    apply_publication_style,
    finish_axis,
)

import compute_xatu_builder_persistence as core


THRESHOLDS = np.array([5, 8], dtype=int)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payloads", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--window-slots", type=int, default=10_000)
    parser.add_argument("--profile-windows", type=int, default=24)
    parser.add_argument("--null-simulations", type=int, default=600)
    parser.add_argument("--power-simulations", type=int, default=300)
    parser.add_argument("--baseline-permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate the figure from CSV files already present in --out-dir.",
    )
    return parser.parse_args()


def run_counts(labels: np.ndarray, break_before: np.ndarray) -> np.ndarray:
    starts = np.r_[True, break_before[1:] | (labels[1:] != labels[:-1])]
    indices = np.flatnonzero(starts)
    lengths = np.diff(np.r_[indices, len(labels)])
    return np.array([(lengths >= k).sum() for k in THRESHOLDS], dtype=np.int64)


def select_window_ids(ids: np.ndarray, count: int) -> np.ndarray:
    unique = np.unique(ids)
    if len(unique) <= count:
        return unique
    positions = np.linspace(0, len(unique) - 1, num=count, dtype=int)
    return unique[positions]


def empirical_profiles(
    frame: pd.DataFrame, label_column: str, window_slots: int, count: int
) -> tuple[list[dict[str, np.ndarray]], list[int]]:
    working = frame[["slot", label_column]].sort_values("slot").copy()
    working["window"] = working.slot // window_slots
    selected = select_window_ids(working.window.to_numpy(), count)
    profiles: list[dict[str, np.ndarray]] = []
    used: list[int] = []
    for window_id in selected:
        group = working[working.window == window_id]
        if len(group) < 1000:
            continue
        slots = group.slot.to_numpy(dtype=np.int64)
        labels = pd.factorize(group[label_column], sort=True)[0]
        counts = np.bincount(labels)
        probabilities = counts / counts.sum()
        breaks = np.r_[True, np.diff(slots) != 1]
        profiles.append(
            {
                "probabilities": probabilities,
                "break_before": breaks,
                "length": np.array([len(group)], dtype=np.int64),
            }
        )
        used.append(int(window_id))
    return profiles, used


def simulate_profiles(
    profiles: list[dict[str, np.ndarray]],
    rho: float,
    simulations: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.zeros((simulations, len(THRESHOLDS)), dtype=np.int64)
    for simulation in range(simulations):
        totals = np.zeros(len(THRESHOLDS), dtype=np.int64)
        for profile in profiles:
            probabilities = profile["probabilities"]
            breaks = profile["break_before"]
            length = int(profile["length"][0])
            base = rng.choice(len(probabilities), size=length, p=probabilities)
            if rho > 0:
                fresh = rng.random(length) >= rho
                fresh[breaks] = True
                anchors = np.maximum.accumulate(
                    np.where(fresh, np.arange(length, dtype=np.int64), 0)
                )
                labels = base[anchors]
            else:
                labels = base
            totals += run_counts(labels, breaks)
        output[simulation] = totals
    return output


def leave_one_out_pvalues(values: np.ndarray) -> np.ndarray:
    pvalues = np.empty(len(values), dtype=float)
    for index, value in enumerate(values):
        greater_equal = int((values >= value).sum()) - 1
        pvalues[index] = (1 + greater_equal) / len(values)
    return pvalues


def ranges_from_groups(group_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.r_[0, np.flatnonzero(group_ids[1:] != group_ids[:-1]) + 1]
    ends = np.r_[starts[1:], len(group_ids)]
    return starts, ends


def permutation_baseline(
    labels: np.ndarray,
    slots: np.ndarray,
    group_ids: np.ndarray,
    break_at_group_boundary: bool,
    permutations: int,
    seed: int,
    progress_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    continuity_break = np.r_[True, np.diff(slots) != 1]
    group_break = np.r_[True, group_ids[1:] != group_ids[:-1]]
    break_before = continuity_break | group_break if break_at_group_boundary else continuity_break
    observed = run_counts(labels, break_before)
    starts, ends = ranges_from_groups(group_ids)
    rng = np.random.default_rng(seed)
    simulations = np.zeros((permutations, len(THRESHOLDS)), dtype=np.int64)
    for simulation in range(permutations):
        permuted = labels.copy()
        for start, end in zip(starts, ends):
            rng.shuffle(permuted[start:end])
        simulations[simulation] = run_counts(permuted, break_before)
        if (simulation + 1) % 100 == 0:
            print(f"{progress_label}: {simulation + 1}/{permutations}", flush=True)
    return observed, simulations


def baseline_rows(
    unit: str,
    grouping: str,
    observed: np.ndarray,
    simulations: np.ndarray,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, threshold in enumerate(THRESHOLDS):
        sample = simulations[:, index]
        expected_mean = float(sample.mean())
        records.append(
            {
                "unit": unit,
                "grouping": grouping,
                "threshold": int(threshold),
                "observed": int(observed[index]),
                "expected_mean": expected_mean,
                "ci_low": float(np.quantile(sample, 0.025)),
                "ci_high": float(np.quantile(sample, 0.975)),
                "observed_expected_ratio": (
                    float(observed[index] / expected_mean)
                    if expected_mean > 0
                    else float("nan")
                ),
                "p_upper": float((1 + (sample >= observed[index]).sum()) / (len(sample) + 1)),
                "p_lower": float((1 + (sample <= observed[index]).sum()) / (len(sample) + 1)),
                "permutations": int(len(sample)),
            }
        )
    return records


def plot_results(
    calibration: pd.DataFrame,
    power: pd.DataFrame,
    baselines: pd.DataFrame,
    out_dir: Path,
) -> None:
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=DOUBLE_FIGSIZE)

    labels = [f"{row.unit}\nk={row.threshold}" for row in calibration.itertuples()]
    axes[0].bar(np.arange(len(calibration)), calibration.type_i_error, color=BLUE)
    axes[0].axhline(0.05, color=RED, linestyle="--", linewidth=1.0)
    axes[0].set_xticks(np.arange(len(calibration)), labels, rotation=20)
    axes[0].set(ylabel="Type-I error", ylim=(0, max(0.08, calibration.type_i_error.max() * 1.25)))

    styles = {("key", 5): (BLUE, "o"), ("key", 8): (LIGHT_BLUE, "s"),
              ("proxy", 5): (RED, "o"), ("proxy", 8): (GREEN, "s")}
    for (unit, threshold), group in power.groupby(["unit", "threshold"]):
        color, marker = styles[(unit, int(threshold))]
        axes[1].plot(group.rho, group.power, color=color, marker=marker,
                     markersize=3, label=f"{unit}, k={int(threshold)}")
    axes[1].set(xlabel="Injected persistence $\\rho$", ylabel="Detection power", ylim=(0, 1.02))
    axes[1].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        columnspacing=0.7,
        handletextpad=0.35,
        borderaxespad=0,
    )

    selected = baselines[
        ((baselines.unit == "key") & (baselines.threshold == 5))
        | ((baselines.unit == "proxy") & (baselines.threshold == 8))
    ].copy()
    order = ["global", "monthly", "local_10k", "analytical_local_10k"]
    display_names = {
        "global": "Global",
        "monthly": "Monthly",
        "local_10k": "Local",
        "analytical_local_10k": "Analytical",
    }
    selected["grouping"] = pd.Categorical(selected.grouping, categories=order, ordered=True)
    selected = selected.sort_values(["unit", "grouping"])
    for unit, group in selected.groupby("unit", observed=True):
        axes[2].plot(
            group.grouping.astype(str).map(display_names),
            group.observed_expected_ratio,
            marker="o" if unit == "key" else "s",
            markersize=3,
            color=BLUE if unit == "key" else RED,
            label="key, k=5" if unit == "key" else "proxy, k=8",
        )
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    axes[2].set_yscale("log")
    axes[2].tick_params(axis="x", rotation=20)
    plt.setp(axes[2].get_xticklabels(), ha="right", rotation_mode="anchor")
    axes[2].set(ylabel="Observed / expected")
    axes[2].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        columnspacing=0.7,
        handletextpad=0.35,
        borderaxespad=0,
    )

    for axis in axes:
        finish_axis(axis)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.26, top=0.70, wspace=0.46)
    fig.savefig(out_dir / "Fig_persistence_calibration.pdf")
    fig.savefig(out_dir / "Fig_persistence_calibration.png", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        calibration = pd.read_csv(args.out_dir / "null_calibration.csv")
        power = pd.read_csv(args.out_dir / "persistence_power_curve.csv")
        baselines = pd.read_csv(args.out_dir / "real_null_baseline_comparison.csv")
        plot_results(calibration, power, baselines, args.out_dir)
        print(f"regenerated figure in {args.out_dir}")
        return
    frame = pd.read_parquet(
        args.payloads,
        columns=["slot", "block_number", "slot_start_date_time", "builder_pubkey", "extra_data_tag"],
    )
    frame["slot_start_date_time"] = pd.to_datetime(frame.slot_start_date_time, utc=True)
    frame["builder_pubkey"] = frame.builder_pubkey.astype(str).str.casefold()
    frame["extra_data_tag"] = frame.extra_data_tag.fillna("").astype(str)
    frame["proxy"] = np.where(
        frame.extra_data_tag.str.len() > 0,
        "tag:" + frame.extra_data_tag,
        "key:" + frame.builder_pubkey,
    )
    frame = frame.sort_values("slot").reset_index(drop=True)

    calibration_records: list[dict[str, object]] = []
    power_records: list[dict[str, object]] = []
    selected_windows: dict[str, list[int]] = {}
    for unit, column in [("key", "builder_pubkey"), ("proxy", "proxy")]:
        profiles, used = empirical_profiles(
            frame, column, args.window_slots, args.profile_windows
        )
        selected_windows[unit] = used
        null = simulate_profiles(
            profiles, 0.0, args.null_simulations, args.seed + (0 if unit == "key" else 1000)
        )
        for index, threshold in enumerate(THRESHOLDS):
            pvalues = leave_one_out_pvalues(null[:, index])
            calibration_records.append(
                {
                    "unit": unit,
                    "threshold": int(threshold),
                    "type_i_error": float((pvalues <= 0.05).mean()),
                    "pvalue_q10": float(np.quantile(pvalues, 0.10)),
                    "pvalue_median": float(np.median(pvalues)),
                    "pvalue_q90": float(np.quantile(pvalues, 0.90)),
                    "null_mean": float(null[:, index].mean()),
                    "null_sd": float(null[:, index].std(ddof=1)),
                    "simulations": int(len(null)),
                }
            )
            power_records.append(
                {
                    "unit": unit,
                    "threshold": int(threshold),
                    "rho": 0.0,
                    "power": float((pvalues <= 0.05).mean()),
                    "mean_count": float(null[:, index].mean()),
                    "mean_null_ratio": 1.0,
                    "simulations": int(len(null)),
                }
            )
        for rho_index, rho in enumerate((0.01, 0.025, 0.05, 0.10), start=1):
            alternative = simulate_profiles(
                profiles,
                rho,
                args.power_simulations,
                args.seed + 10_000 * rho_index + (0 if unit == "key" else 1000),
            )
            for index, threshold in enumerate(THRESHOLDS):
                reference = null[:, index]
                pvalues = np.array(
                    [
                        (1 + (reference >= value).sum()) / (len(reference) + 1)
                        for value in alternative[:, index]
                    ]
                )
                power_records.append(
                    {
                        "unit": unit,
                        "threshold": int(threshold),
                        "rho": rho,
                        "power": float((pvalues <= 0.05).mean()),
                        "mean_count": float(alternative[:, index].mean()),
                        "mean_null_ratio": float(
                            alternative[:, index].mean() / reference.mean()
                        ),
                        "simulations": int(len(alternative)),
                    }
                )
        print(f"completed synthetic calibration for {unit}", flush=True)

    baseline_records: list[dict[str, object]] = []
    slots = frame.slot.to_numpy(dtype=np.int64)
    global_groups = np.zeros(len(frame), dtype=np.int8)
    monthly_groups = frame.slot_start_date_time.dt.tz_localize(None).dt.to_period("M").astype(str).to_numpy()
    local_groups = (frame.slot // args.window_slots).to_numpy(dtype=np.int64)
    for unit_index, (unit, column) in enumerate([("key", "builder_pubkey"), ("proxy", "proxy")]):
        labels = pd.factorize(frame[column], sort=True)[0].astype(np.int16)
        for grouping_index, (name, groups, break_groups) in enumerate(
            [
                ("global", global_groups, False),
                ("monthly", monthly_groups, False),
                ("local_10k", local_groups, True),
            ]
        ):
            observed, simulations = permutation_baseline(
                labels,
                slots,
                groups,
                break_groups,
                args.baseline_permutations,
                args.seed + 100_000 * unit_index + 10_000 * grouping_index,
                f"{unit}/{name}",
            )
            baseline_records.extend(
                baseline_rows(unit, name, observed, simulations)
            )

        analytical_frame = frame[["slot", "block_number", column]].rename(
            columns={column: "builder_pubkey"}
        )
        expected, _ = core.analytic_expected(
            analytical_frame, "slot", args.window_slots
        )
        threshold_indices = [int(np.flatnonzero(core.THRESHOLDS == k)[0]) for k in THRESHOLDS]
        local_observed, _ = permutation_baseline(
            labels,
            slots,
            local_groups,
            True,
            1,
            args.seed,
            f"{unit}/analytical-observed",
        )
        for index, threshold_index in enumerate(threshold_indices):
            value = float(expected[threshold_index])
            baseline_records.append(
                {
                    "unit": unit,
                    "grouping": "analytical_local_10k",
                    "threshold": int(THRESHOLDS[index]),
                    "observed": int(local_observed[index]),
                    "expected_mean": value,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "observed_expected_ratio": float(local_observed[index] / value),
                    "p_upper": np.nan,
                    "p_lower": np.nan,
                    "permutations": 0,
                }
            )

    calibration = pd.DataFrame(calibration_records)
    power = pd.DataFrame(power_records)
    baselines = pd.DataFrame(baseline_records)
    calibration.to_csv(args.out_dir / "null_calibration.csv", index=False)
    power.to_csv(args.out_dir / "persistence_power_curve.csv", index=False)
    baselines.to_csv(args.out_dir / "real_null_baseline_comparison.csv", index=False)

    summary = {
        "input": str(args.payloads),
        "window_slots": args.window_slots,
        "profile_windows_requested": args.profile_windows,
        "selected_window_ids": selected_windows,
        "null_simulations": args.null_simulations,
        "power_simulations": args.power_simulations,
        "baseline_permutations": args.baseline_permutations,
        "seed": args.seed,
        "calibration": calibration.to_dict("records"),
        "power": power.to_dict("records"),
        "real_baselines": baselines.to_dict("records"),
    }
    (args.out_dir / "null_calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot_results(calibration, power, baselines, args.out_dir)
    print(json.dumps({"calibration": calibration.to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
