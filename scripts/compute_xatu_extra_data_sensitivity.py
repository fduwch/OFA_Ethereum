#!/usr/bin/env python3
"""Canonical-chain and identity-unit validation for Xatu persistence.

The relay export can contain several payload candidates for one slot.  This
script joins every candidate to Xatu's canonical execution-block export by
block number and hash instead of discarding the whole slot.  It then repeats
the concentration and permutation analyses for builder public keys and for an
exact normalized version of the winning block's self-reported extra data.

The tag is a sensitivity proxy, not verified ownership: builders can change or
spoof it.  Empty tags fall back to the original key so unrelated unknown blocks
are never collapsed into one actor.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re
import time
import unicodedata

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

import compute_xatu_builder_persistence as core


BASE_URL = (
    "https://data.ethpandaops.io/xatu/mainnet/databases/default/"
    "canonical_execution_block/1000/{chunk}.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-cache", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--window-slots", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--request-timeout", type=int, default=90)
    return parser.parse_args()


def to_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def normalize_tag(value: object) -> str:
    text = unicodedata.normalize("NFKC", to_text(value))
    text = " ".join(text.split()).strip().casefold()
    # Treat control-only, replacement-character-only, and hex-looking values as
    # uninformative.  Do not hand-maintain brand aliases: exact normalized tags
    # keep the sensitivity deterministic and versionable.
    printable = "".join(char for char in text if char.isprintable()).strip()
    if not printable or "�" in printable or re.fullmatch(r"0x[0-9a-f]+", printable):
        return ""
    return printable


def decode_extra_data(value: object) -> str:
    """Decode Xatu's hex-encoded raw extra_data without trusting UTF-8."""
    text = to_text(value).strip()
    if text.startswith("0x"):
        try:
            return bytes.fromhex(text[2:]).decode("utf-8", errors="replace")
        except ValueError:
            return text
    return text


def download_chunk(cache_dir: Path, chunk: int, timeout: int) -> tuple[int, str]:
    target = cache_dir / f"{chunk}.parquet"
    if target.exists() and target.stat().st_size > 0:
        return chunk, "cached"
    url = BASE_URL.format(chunk=chunk)
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            temporary = target.with_suffix(".parquet.part")
            temporary.write_bytes(response.content)
            os.replace(temporary, target)
            return chunk, "downloaded"
        except requests.RequestException:
            if attempt == 3:
                return chunk, "error"
            time.sleep(2**attempt)
    return chunk, "error"


def read_chunk(path: Path) -> pd.DataFrame:
    table = pq.read_table(
        path,
        columns=["block_number", "block_hash", "extra_data"],
    )
    # Some blocks contain arbitrary non-UTF-8 extra data.  Older pyarrow builds
    # can fail while wrapping such binary columns for pandas, especially across
    # threads.  A Python dictionary preserves the bytes for our tolerant decoder.
    frame = pd.DataFrame(table.to_pydict())
    frame["block_number"] = frame.block_number.astype("int64")
    frame["block_hash"] = frame.block_hash.map(to_text).str.lower()
    frame["extra_data_tag"] = frame.extra_data.map(decode_extra_data).map(normalize_tag)
    return frame[["block_number", "block_hash", "extra_data_tag"]]


def read_payload_day(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    table = pq.read_table(
        path,
        columns=[
            "slot",
            "slot_start_date_time",
            "block_number",
            "relay_name",
            "block_hash",
            "builder_pubkey",
        ],
    )
    frame = pd.DataFrame(table.to_pydict())
    raw_rows = len(frame)
    frame = frame.dropna(
        subset=["slot", "block_number", "builder_pubkey", "block_hash"]
    )
    frame["slot"] = frame.slot.astype("int64")
    frame["block_number"] = pd.to_numeric(
        frame.block_number, errors="coerce"
    ).astype("int64")
    frame["slot_start_date_time"] = pd.to_datetime(
        frame.slot_start_date_time, unit="s", utc=True
    )
    frame["builder_pubkey"] = frame.builder_pubkey.map(to_text).str.lower()
    frame["block_hash"] = frame.block_hash.map(to_text).str.lower()
    frame["relay_name"] = frame.relay_name.map(to_text)
    exact = frame.drop_duplicates(
        ["slot", "block_number", "block_hash", "builder_pubkey", "relay_name"]
    )
    candidates = exact.groupby(
        ["slot", "block_number", "block_hash", "builder_pubkey"],
        as_index=False,
    ).agg(
        relay_count=("relay_name", "nunique"),
        relay_names=("relay_name", lambda values: "|".join(sorted(set(values)))),
        slot_start_date_time=("slot_start_date_time", "min"),
    )
    quality = {
        "raw_rows": int(raw_rows),
        "exact_relay_rows": int(len(exact)),
        "payload_candidates": int(len(candidates)),
        "candidate_slots": int(candidates.slot.nunique()),
    }
    return candidates, quality


def summary_for(frame: pd.DataFrame, label_unit: str, args: argparse.Namespace):
    working = frame.copy()
    working["builder_pubkey"] = working[label_unit]
    rows: list[dict[str, object]] = []
    for mode_index, continuity in enumerate(("slot", "block")):
        observed, simulations, _ = core.permutation_null(
            working,
            continuity,
            args.window_slots,
            args.permutations,
            args.seed + mode_index,
        )
        analytical, _ = core.analytic_expected(
            working, continuity, args.window_slots
        )
        for index, threshold in enumerate(core.THRESHOLDS):
            simulated = simulations[:, index]
            p_ge = float(
                (1 + (simulated >= observed[index]).sum())
                / (1 + len(simulated))
            )
            p_le = float(
                (1 + (simulated <= observed[index]).sum())
                / (1 + len(simulated))
            )
            rows.append(
                {
                    "label_unit": label_unit,
                    "continuity": continuity,
                    "threshold": int(threshold),
                    "observed_runs": int(observed[index]),
                    "analytical_expected_runs": float(analytical[index]),
                    "observed_expected_ratio": (
                        float(observed[index] / analytical[index])
                        if analytical[index] > 0
                        else np.nan
                    ),
                    "permutation_mean": float(simulated.mean()),
                    "permutation_ci_low": float(np.quantile(simulated, 0.025)),
                    "permutation_ci_high": float(np.quantile(simulated, 0.975)),
                    "permutation_p_ge_observed": p_ge,
                    "permutation_p_le_observed": p_le,
                    "permutation_p_two_sided": min(1.0, 2.0 * min(p_ge, p_le)),
                    "permutations": args.permutations,
                }
            )
    working_monthly = core.concentration_by_month(working)
    working_monthly.insert(0, "label_unit", label_unit)
    return pd.DataFrame(rows), working_monthly


def main() -> None:
    args = parse_args()
    payload_cache = Path(args.payload_cache)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload_paths = sorted(payload_cache.glob("????-??-??.parquet"))
    if not payload_paths:
        raise RuntimeError(f"No daily payload files found in {payload_cache}")
    candidate_frames: list[pd.DataFrame] = []
    daily_quality: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(read_payload_day, path): path for path in payload_paths}
        for index, future in enumerate(as_completed(futures), start=1):
            candidates, quality = future.result()
            candidate_frames.append(candidates)
            daily_quality.append({"file": futures[future].name, **quality})
            if index % 30 == 0:
                print(f"read {index}/{len(payload_paths)} payload days", flush=True)
    payload_candidates = pd.concat(candidate_frames, ignore_index=True)
    payload_candidates = payload_candidates.drop_duplicates(
        ["slot", "block_number", "block_hash", "builder_pubkey"]
    )
    candidate_counts = payload_candidates.groupby("slot").size()
    ambiguous_slots = set(candidate_counts[candidate_counts > 1].index.astype(int))

    first_chunk = int(payload_candidates.block_number.min() // 1000 * 1000)
    last_chunk = int(payload_candidates.block_number.max() // 1000 * 1000)
    chunks = list(range(first_chunk, last_chunk + 1000, 1000))

    statuses: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_chunk, cache_dir, chunk, args.request_timeout): chunk
            for chunk in chunks
        }
        for index, future in enumerate(as_completed(futures), start=1):
            chunk, status = future.result()
            statuses[chunk] = status
            if index % 250 == 0:
                print(f"fetched {index}/{len(chunks)} canonical chunks", flush=True)

    good_paths = [
        cache_dir / f"{chunk}.parquet"
        for chunk in chunks
        if statuses.get(chunk) != "error"
    ]
    canonical_frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(read_chunk, path) for path in good_paths]
        for index, future in enumerate(as_completed(futures), start=1):
            canonical_frames.append(future.result())
            if index % 500 == 0:
                print(f"read {index}/{len(good_paths)} canonical chunks", flush=True)
    canonical = pd.concat(canonical_frames, ignore_index=True)
    canonical = canonical.drop_duplicates(["block_number", "block_hash"])

    canonical_matches = payload_candidates.merge(
        canonical,
        on=["block_number", "block_hash"],
        how="inner",
        validate="many_to_one",
    )
    canonical_matches["extra_data_tag"] = canonical_matches.extra_data_tag.fillna("")
    canonical_matches["builder_identity_proxy"] = np.where(
        canonical_matches.extra_data_tag.str.len() > 0,
        "tag:" + canonical_matches.extra_data_tag,
        "key:" + canonical_matches.builder_pubkey,
    )

    # One canonical payload may be reported under several relay-specific builder
    # keys.  Such a slot is invalid for the key analysis but still has one block
    # extra-data identity.  Build a valid sample separately for each unit rather
    # than forcing both analyses onto the smaller unique-key subset.
    key_counts = canonical_matches.groupby("slot").builder_pubkey.nunique()
    key_ambiguous_slots = set(key_counts[key_counts > 1].index.astype(int))
    key_selected = canonical_matches[
        ~canonical_matches.slot.isin(key_ambiguous_slots)
    ].drop_duplicates("slot")
    key_selected = key_selected.sort_values("slot").reset_index(drop=True)

    proxy_counts = canonical_matches.groupby("slot").builder_identity_proxy.nunique()
    proxy_ambiguous_slots = set(proxy_counts[proxy_counts > 1].index.astype(int))
    proxy_selected = canonical_matches[
        ~canonical_matches.slot.isin(proxy_ambiguous_slots)
    ].drop_duplicates("slot")
    proxy_selected = proxy_selected.sort_values("slot").reset_index(drop=True)

    key_selected.to_parquet(
        out_dir / "xatu_canonical_payloads_by_key_reduced.parquet",
        index=False,
        coerce_timestamps="us",
        allow_truncated_timestamps=True,
    )
    proxy_selected.to_parquet(
        out_dir / "xatu_canonical_payloads_by_identity_reduced.parquet",
        index=False,
        coerce_timestamps="us",
        allow_truncated_timestamps=True,
    )

    result_frames: list[pd.DataFrame] = []
    monthly_frames: list[pd.DataFrame] = []
    analysis_frames = {
        "builder_identity_proxy": proxy_selected,
        "builder_pubkey": key_selected,
    }
    for unit, analysis_frame in analysis_frames.items():
        results, monthly = summary_for(analysis_frame, unit, args)
        result_frames.append(results)
        monthly_frames.append(monthly)
    results = pd.concat(result_frames, ignore_index=True)
    monthly = pd.concat(monthly_frames, ignore_index=True)
    results.to_csv(out_dir / "xatu_builder_identity_persistence.csv", index=False)
    monthly.to_csv(out_dir / "xatu_builder_identity_monthly.csv", index=False)

    tag_counts = proxy_selected.loc[
        proxy_selected.extra_data_tag.str.len() > 0, "extra_data_tag"
    ].value_counts()
    tag_counts.rename_axis("extra_data_tag").reset_index(name="blocks").to_csv(
        out_dir / "xatu_extra_data_tag_counts.csv", index=False
    )
    quality = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_source_url_template": BASE_URL,
        "first_chunk": first_chunk,
        "last_chunk": last_chunk,
        "requested_chunks": len(chunks),
        "error_chunks": [chunk for chunk, status in statuses.items() if status == "error"],
        "payload_days": len(payload_paths),
        "raw_relay_rows": int(sum(item["raw_rows"] for item in daily_quality)),
        "exact_relay_rows": int(sum(item["exact_relay_rows"] for item in daily_quality)),
        "payload_candidate_rows": int(len(payload_candidates)),
        "payload_candidate_slots": int(payload_candidates.slot.nunique()),
        "ambiguous_slots_before_canonical_match": int(len(ambiguous_slots)),
        "canonical_candidate_rows": int(len(canonical_matches)),
        "canonical_matched_slots": int(canonical_matches.slot.nunique()),
        "ambiguous_slots_with_canonical_hash": int(
            len(ambiguous_slots & set(canonical_matches.slot.astype(int)))
        ),
        "unique_key_slots_retained": int(len(key_selected)),
        "multi_key_slots_excluded_from_key_analysis": int(len(key_ambiguous_slots)),
        "unique_identity_proxy_slots_retained": int(len(proxy_selected)),
        "multi_proxy_slots_excluded_from_identity_analysis": int(
            len(proxy_ambiguous_slots)
        ),
        "ambiguous_slots_recovered_for_identity_analysis": int(
            len(ambiguous_slots & set(proxy_selected.slot.astype(int)))
        ),
        "unmatched_candidate_slots": int(
            payload_candidates.slot.nunique() - canonical_matches.slot.nunique()
        ),
        "first_slot": int(proxy_selected.slot.min()),
        "last_slot": int(proxy_selected.slot.max()),
        "first_time": proxy_selected.slot_start_date_time.min().isoformat(),
        "last_time": proxy_selected.slot_start_date_time.max().isoformat(),
        "nonempty_normalized_tag_rows": int((proxy_selected.extra_data_tag.str.len() > 0).sum()),
        "nonempty_normalized_tag_share": float((proxy_selected.extra_data_tag.str.len() > 0).mean()),
        "unique_normalized_tags": int((proxy_selected.extra_data_tag[proxy_selected.extra_data_tag.str.len() > 0]).nunique()),
        "label_definition": (
            "exact NFKC/casefold/whitespace-normalized execution extra_data_string; "
            "empty tags fall back to builder public key"
        ),
        "limitation": "self-reported tag is an entity proxy, not verified ownership",
        "window_slots": args.window_slots,
        "permutations": args.permutations,
        "daily_quality": sorted(daily_quality, key=lambda item: item["file"]),
    }
    (out_dir / "xatu_builder_identity_quality.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0))
    for unit, label, color, marker in (
        ("builder_identity_proxy", "Extra-data proxy", "#D55E00", "s"),
        ("builder_pubkey", "Public key", "#0072B2", "o"),
    ):
        subset = results[
            (results.label_unit == unit) & (results.continuity == "slot")
        ]
        plotted = (
            subset[subset.threshold <= 6]
            if unit == "builder_pubkey"
            else subset
        )
        axes[0].plot(
            plotted.threshold,
            plotted.observed_expected_ratio,
            label=label,
            color=color,
            linewidth=1.7,
        )
        significant = (
            (plotted.observed_runs < plotted.permutation_ci_low)
            | (plotted.observed_runs > plotted.permutation_ci_high)
        )
        axes[0].scatter(
            plotted.threshold,
            plotted.observed_expected_ratio,
            marker=marker,
            edgecolor=color,
            facecolor=np.where(significant, color, "white"),
            zorder=3,
        )
    axes[0].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Run-length threshold")
    axes[0].set_ylabel("Observed / expected runs")
    axes[0].set_title("(a) Persistence beyond market share")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    styles = {
        "builder_identity_proxy": ("Extra-data proxy", "-"),
        "builder_pubkey": ("Public key", "--"),
    }
    for unit, (label, style) in styles.items():
        subset = monthly[monthly.label_unit == unit]
        axes[1].plot(
            pd.to_datetime(subset.date),
            subset.cr3,
            linestyle=style,
            linewidth=1.6,
            label=f"CR3, {label}",
        )
        axes[1].plot(
            pd.to_datetime(subset.date),
            subset.hhi,
            linestyle=style,
            linewidth=1.2,
            alpha=0.75,
            label=f"HHI, {label}",
        )
    axes[1].set_title("(b) Concentration by identity unit")
    axes[1].set_xlabel("Month (UTC)")
    axes[1].set_ylabel("Concentration")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in axes[1].get_xticklabels():
        label.set_rotation(35)
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_dir / "Fig_builder_identity_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(
        out_dir / "Fig_builder_identity_sensitivity.png",
        dpi=240,
        bbox_inches="tight",
    )

    print(results.to_string(index=False))
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
