#!/usr/bin/env python3
"""Measure builder concentration and serial persistence with public Xatu data.

The source table is EthPandaOps' public daily Parquet export of
``mev_relay_proposer_payload_delivered``.  Builder public keys are used as the
unit of analysis, so the concentration estimates are conservative when a
single builder rotates or uses multiple keys.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests


BASE_URL = (
    "https://data.ethpandaops.io/xatu/mainnet/databases/default/"
    "mev_relay_proposer_payload_delivered/{year}/{month}/{day}.parquet"
)
THRESHOLDS = np.array([2, 3, 4, 5, 6, 8, 10], dtype=int)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start", default="2024-09-16")
    parser.add_argument("--end", default="2025-10-05")
    parser.add_argument("--window-slots", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--request-timeout", type=int, default=90)
    return parser.parse_args()


def daterange(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def download_day(
    session: requests.Session, day: date, cache_dir: Path, timeout: int
) -> tuple[Path | None, str]:
    target = cache_dir / f"{day.isoformat()}.parquet"
    if target.exists() and target.stat().st_size > 0:
        return target, "cached"
    url = BASE_URL.format(year=day.year, month=day.month, day=day.day)
    for attempt in range(4):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code == 404:
                return None, "missing"
            response.raise_for_status()
            temporary = target.with_suffix(".parquet.part")
            temporary.write_bytes(response.content)
            os.replace(temporary, target)
            return target, "downloaded"
        except requests.RequestException:
            if attempt == 3:
                return None, "error"
            time.sleep(2**attempt)
    return None, "error"


def reduce_day(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    columns = [
        "slot",
        "slot_start_date_time",
        "block_number",
        "relay_name",
        "block_hash",
        "builder_pubkey",
    ]
    frame = pq.read_table(path, columns=columns).to_pandas()
    raw_rows = len(frame)
    frame = frame.dropna(subset=["slot", "builder_pubkey", "block_hash"])
    frame["slot"] = frame.slot.astype("int64")
    frame["block_number"] = pd.to_numeric(frame.block_number, errors="coerce")
    # Xatu stores this field as a uint32 Unix timestamp, not an Arrow timestamp.
    # Passing the integers to pd.to_datetime without a unit silently interprets
    # them as nanoseconds after 1970 and corrupts every monthly aggregation.
    frame["slot_start_date_time"] = pd.to_datetime(
        frame.slot_start_date_time, unit="s", utc=True
    )
    def to_text(value: object) -> str:
        return (
            value.decode("utf-8", errors="replace")
            if isinstance(value, (bytes, bytearray))
            else str(value)
        )
    frame["builder_pubkey"] = frame.builder_pubkey.map(to_text).str.lower()
    frame["block_hash"] = frame.block_hash.map(to_text).str.lower()
    frame["relay_name"] = frame.relay_name.map(to_text)
    frame = frame[frame.builder_pubkey.str.len() > 2]
    exact = frame.drop_duplicates(
        ["slot", "block_hash", "builder_pubkey", "relay_name"]
    )
    candidates = (
        exact.groupby(["slot", "block_hash", "builder_pubkey"], as_index=False)
        .agg(
            relay_count=("relay_name", "nunique"),
            relay_names=("relay_name", lambda value: "|".join(sorted(set(value)))),
            slot_start_date_time=("slot_start_date_time", "min"),
            block_number=("block_number", "max"),
        )
        .sort_values(["slot", "relay_count", "block_hash"], ascending=[True, False, True])
    )
    candidate_count = candidates.groupby("slot").size()
    ambiguous_slots = set(candidate_count[candidate_count > 1].index.astype(int))
    chosen = candidates[~candidates.slot.isin(ambiguous_slots)].copy()
    chosen = chosen.drop_duplicates("slot").sort_values("slot")
    quality = {
        "raw_rows": int(raw_rows),
        "exact_rows": int(len(exact)),
        "candidate_payloads": int(len(candidates)),
        "ambiguous_slots_excluded": int(len(ambiguous_slots)),
        "unique_slots_retained": int(len(chosen)),
    }
    return chosen, quality


def run_lengths(labels: np.ndarray, break_before: np.ndarray):
    if len(labels) == 0:
        return np.array([], dtype=int), np.array([], dtype=labels.dtype)
    starts = np.r_[True, break_before[1:] | (labels[1:] != labels[:-1])]
    indices = np.flatnonzero(starts)
    lengths = np.diff(np.r_[indices, len(labels)])
    return lengths, labels[indices]


def threshold_counts(lengths: np.ndarray) -> np.ndarray:
    return np.array([(lengths >= threshold).sum() for threshold in THRESHOLDS])


def analytic_expected(
    frame: pd.DataFrame, continuity: str, window_slots: int
) -> tuple[np.ndarray, dict[str, float]]:
    totals = np.zeros(len(THRESHOLDS), dtype=float)
    builder_expected: dict[str, float] = defaultdict(float)
    for _, window in frame.groupby(frame.slot // window_slots, sort=True):
        window = window.sort_values("slot")
        labels = window.builder_pubkey.to_numpy()
        if continuity == "slot":
            coordinate = window.slot.to_numpy(dtype=np.int64)
        else:
            coordinate = window.block_number.to_numpy(dtype=float)
        segment_break = np.r_[True, np.diff(coordinate) != 1]
        segment_starts = np.flatnonzero(segment_break)
        segment_lengths = np.diff(np.r_[segment_starts, len(window)])
        eligible_segments = np.array(
            [(segment_lengths >= threshold).sum() for threshold in THRESHOLDS],
            dtype=float,
        )
        eligible_excess_lengths = np.array(
            [
                (segment_lengths[segment_lengths >= threshold] - threshold).sum()
                for threshold in THRESHOLDS
            ],
            dtype=float,
        )
        probabilities = pd.Series(labels).value_counts(normalize=True)
        for builder, probability in probabilities.items():
            p = float(probability)
            expected = np.power(p, THRESHOLDS) * (
                eligible_segments + (1.0 - p) * eligible_excess_lengths
            )
            totals += expected
            threshold_five_index = int(np.flatnonzero(THRESHOLDS == 5)[0])
            builder_expected[str(builder)] += float(expected[threshold_five_index])
    return totals, builder_expected


def permutation_null(
    frame: pd.DataFrame,
    continuity: str,
    window_slots: int,
    permutations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = frame.builder_pubkey.to_numpy(copy=True)
    slots = frame.slot.to_numpy(dtype=np.int64)
    blocks = frame.block_number.to_numpy(dtype=float)
    window_ids = slots // window_slots
    base_break = np.r_[True, window_ids[1:] != window_ids[:-1]]
    coordinate = slots if continuity == "slot" else blocks
    continuity_break = np.r_[True, np.diff(coordinate) != 1]
    break_before = base_break | continuity_break
    observed_lengths, observed_builders = run_lengths(labels, break_before)
    observed = threshold_counts(observed_lengths)

    starts = np.r_[0, np.flatnonzero(window_ids[1:] != window_ids[:-1]) + 1]
    ends = np.r_[starts[1:], len(labels)]
    rng = np.random.default_rng(seed)
    simulations = np.zeros((permutations, len(THRESHOLDS)), dtype=np.int64)
    for simulation in range(permutations):
        permuted = labels.copy()
        for start, end in zip(starts, ends):
            rng.shuffle(permuted[start:end])
        lengths, _ = run_lengths(permuted, break_before)
        simulations[simulation] = threshold_counts(lengths)
    return observed, simulations, np.column_stack((observed_lengths, observed_builders))


def concentration_by_month(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for month, group in frame.groupby(frame.slot_start_date_time.dt.to_period("M")):
        counts = group.builder_pubkey.value_counts()
        shares = counts / counts.sum()
        records.append(
            {
                "month": str(month),
                "date": month.to_timestamp(),
                "delivered_blocks": int(len(group)),
                "builder_pubkeys": int(len(counts)),
                "hhi": float(np.square(shares).sum()),
                "cr3": float(shares.head(3).sum()),
                "cr5": float(shares.head(5).sum()),
                "effective_builder_pubkeys": float(1.0 / np.square(shares).sum()),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    # Version the lightweight cache because v1 was produced before the explicit
    # Unix-second conversion above.  Keeping it separate also avoids deleting
    # any prior artifact while making stale-cache reuse impossible.
    reduced_dir = cache_dir / "reduced_v2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    reduced_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    frames: list[pd.DataFrame] = []
    daily_quality: list[dict[str, object]] = []
    for index, day in enumerate(daterange(start, end), start=1):
        reduced_path = reduced_dir / f"{day.isoformat()}.parquet"
        if reduced_path.exists():
            reduced = pd.read_parquet(reduced_path)
            status = "reduced-cache"
            quality = {
                "raw_rows": -1,
                "exact_rows": -1,
                "candidate_payloads": -1,
                "ambiguous_slots_excluded": -1,
                "unique_slots_retained": int(len(reduced)),
            }
        else:
            path, status = download_day(
                session, day, cache_dir, timeout=args.request_timeout
            )
            if path is None:
                daily_quality.append({"date": day.isoformat(), "status": status})
                continue
            reduced, quality = reduce_day(path)
            reduced.to_parquet(reduced_path, index=False)
        daily_quality.append({"date": day.isoformat(), "status": status, **quality})
        frames.append(reduced)
        if index % 30 == 0:
            print(f"processed {index} calendar days through {day}", flush=True)

    if not frames:
        raise RuntimeError("No Xatu daily data were loaded")
    frame = pd.concat(frames, ignore_index=True)
    frame["slot_start_date_time"] = pd.to_datetime(
        frame.slot_start_date_time, utc=True
    )
    frame = frame.drop_duplicates("slot").sort_values("slot").reset_index(drop=True)
    # Daily files do not all use the same Arrow timestamp resolution.  Pandas
    # normalizes them to ns when concatenating, while the cluster's older
    # pyarrow defaults to us on write.  Slot times are second-granular, so this
    # explicit conversion is lossless and keeps the combined cache portable.
    cache_frame = frame.copy()
    cache_frame["slot_start_date_time"] = (
        cache_frame.slot_start_date_time.dt.floor("us")
    )
    cache_frame.to_parquet(
        out_dir / "xatu_delivered_blocks_reduced.parquet",
        index=False,
        coerce_timestamps="us",
        allow_truncated_timestamps=True,
    )

    monthly = concentration_by_month(frame)
    monthly.to_csv(out_dir / "xatu_monthly_concentration.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    builder_rows: list[dict[str, object]] = []
    for mode_index, continuity in enumerate(("slot", "block")):
        observed, simulations, observed_runs = permutation_null(
            frame,
            continuity,
            args.window_slots,
            args.permutations,
            args.seed + mode_index,
        )
        analytical, builder_expected = analytic_expected(
            frame, continuity, args.window_slots
        )
        for index, threshold in enumerate(THRESHOLDS):
            simulated = simulations[:, index]
            summary_rows.append(
                {
                    "continuity": continuity,
                    "threshold": int(threshold),
                    "observed_runs": int(observed[index]),
                    "analytical_expected_runs": float(analytical[index]),
                    "observed_expected_ratio": float(observed[index] / analytical[index])
                    if analytical[index] > 0
                    else np.nan,
                    "permutation_mean": float(simulated.mean()),
                    "permutation_ci_low": float(np.quantile(simulated, 0.025)),
                    "permutation_ci_high": float(np.quantile(simulated, 0.975)),
                    "permutation_p_ge_observed": float(
                        (1 + (simulated >= observed[index]).sum())
                        / (1 + len(simulated))
                    ),
                    "permutations": int(args.permutations),
                }
            )

        labels = frame.builder_pubkey.to_numpy()
        slots = frame.slot.to_numpy(dtype=np.int64)
        blocks = frame.block_number.to_numpy(dtype=float)
        window_ids = slots // args.window_slots
        coordinate = slots if continuity == "slot" else blocks
        break_before = np.r_[
            True,
            (window_ids[1:] != window_ids[:-1]) | (np.diff(coordinate) != 1),
        ]
        lengths, run_builders = run_lengths(labels, break_before)
        observed_by_builder = Counter(run_builders[lengths >= 5])
        market_counts = frame.builder_pubkey.value_counts()
        for builder, count in market_counts.head(30).items():
            expected = builder_expected.get(str(builder), 0.0)
            observed_count = int(observed_by_builder.get(str(builder), 0))
            builder_rows.append(
                {
                    "continuity": continuity,
                    "builder_pubkey": str(builder),
                    "builder_key_short": f"{str(builder)[:10]}...{str(builder)[-6:]}",
                    "delivered_blocks": int(count),
                    "market_share": float(count / len(frame)),
                    "observed_runs_ge5": observed_count,
                    "expected_runs_ge5": float(expected),
                    "observed_expected_ratio": float(observed_count / expected)
                    if expected > 0
                    else np.nan,
                }
            )

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(out_dir / "xatu_builder_persistence.csv", index=False)
    pd.DataFrame(builder_rows).to_csv(
        out_dir / "xatu_builder_persistence_by_key.csv", index=False
    )

    quality_summary = {
        "source": "EthPandaOps Xatu public mev_relay_proposer_payload_delivered Parquet",
        "source_url_template": BASE_URL,
        "requested_start": args.start,
        "requested_end": args.end,
        "loaded_days": int(len(frames)),
        "missing_or_error_days": int(
            sum(item.get("status") in {"missing", "error"} for item in daily_quality)
        ),
        "retained_unique_slots": int(len(frame)),
        "first_slot": int(frame.slot.min()),
        "last_slot": int(frame.slot.max()),
        "first_block": int(frame.block_number.min()),
        "last_block": int(frame.block_number.max()),
        "first_time": frame.slot_start_date_time.min().isoformat(),
        "last_time": frame.slot_start_date_time.max().isoformat(),
        "builder_unit": "builder public key (entity-level concentration is at least as high)",
        "window_slots": args.window_slots,
        "permutations": args.permutations,
        "daily_quality": daily_quality,
    }
    (out_dir / "xatu_data_quality.json").write_text(
        json.dumps(quality_summary, indent=2), encoding="utf-8"
    )

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0))
    slot_result = summary_frame[summary_frame.continuity == "slot"]
    axes[0].plot(
        slot_result.threshold,
        slot_result.observed_expected_ratio,
        marker="o",
        color="#0072B2",
        linewidth=1.8,
    )
    axes[0].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Run-length threshold")
    axes[0].set_ylabel("Observed / expected runs")
    axes[0].set_title("(a) Persistence beyond market share")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(
        monthly.date,
        monthly.hhi,
        marker="o",
        markersize=2.5,
        color="#0072B2",
        label="HHI",
    )
    axes[1].plot(
        monthly.date,
        monthly.cr3,
        marker="s",
        markersize=2.3,
        color="#D55E00",
        label="CR3",
    )
    axes[1].set_title("(b) Builder-key concentration")
    axes[1].set_xlabel("Month (UTC)")
    axes[1].set_ylabel("Concentration")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, ncol=2, loc="upper center")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in axes[1].get_xticklabels():
        label.set_rotation(35)
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_dir / "Fig_builder_persistence.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "Fig_builder_persistence.png", dpi=240, bbox_inches="tight")

    print(summary_frame.to_string(index=False))
    print(json.dumps({k: v for k, v in quality_summary.items() if k != "daily_quality"}, indent=2))


if __name__ == "__main__":
    main()
