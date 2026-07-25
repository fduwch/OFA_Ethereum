#!/usr/bin/env python3
"""Validate the stability of an extra-data builder identity proxy.

The analysis is deliberately restricted to the canonical, unique-key support
already produced by the Xatu pipeline. It quantifies key-to-tag consistency,
tag-to-key aggregation, temporal stability, and the sensitivity of monthly
concentration to four transparent identity constructions.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from plot_style import (
    BLUE,
    DOUBLE_FIGSIZE,
    GREEN,
    LIGHT_BLUE,
    PURPLE,
    RED,
    apply_publication_style,
    finish_axis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payloads", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--external-html", type=Path)
    parser.add_argument("--stable-threshold", type=float, default=0.95)
    parser.add_argument("--min-key-slots", type=int, default=100)
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def normalized_entropy(counts: pd.Series) -> float:
    probabilities = counts.to_numpy(dtype=float)
    probabilities /= probabilities.sum()
    if len(probabilities) <= 1:
        return 0.0
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(len(probabilities))


def concentration(frame: pd.DataFrame, identity_column: str, variant: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for month, group in frame.groupby("month", sort=True):
        shares = group[identity_column].value_counts(normalize=True)
        hhi = float(np.square(shares).sum())
        records.append(
            {
                "month": str(month),
                "date": month.to_timestamp(),
                "variant": variant,
                "slots": int(len(group)),
                "identities": int(len(shares)),
                "hhi": hhi,
                "cr3": float(shares.head(3).sum()),
                "cr5": float(shares.head(5).sum()),
                "effective_identities": float(1.0 / hhi),
            }
        )
    return pd.DataFrame.from_records(records)


def canonical_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return " ".join(text.split())


def brand_text(value: object) -> str:
    text = canonical_text(value)
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"[^a-z0-9]+", "", text)
    for token in ("builder", "build", "www", "org", "xyz", "com"):
        text = text.replace(token, "")
    return text


def parse_external_table(path: Path) -> pd.DataFrame:
    html = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"const\s+tableData\s*=\s*(\[.*?\])\s*;", html, re.S)
    if not match:
        return pd.DataFrame(columns=["builder_pubkey", "external_builder"])
    payload = match.group(1)
    # The dashboard embeds this JavaScript inside a JSON-encoded HTML block,
    # so quotes and newlines may still be escaped once after reading the page.
    if r'\"' in payload:
        payload = json.loads('"' + payload + '"')
    rows = json.loads(payload)
    external = pd.DataFrame(rows)
    if not {"builder_pubkey", "builder"}.issubset(external.columns):
        return pd.DataFrame(columns=["builder_pubkey", "external_builder"])
    external = external.rename(columns={"builder": "external_builder"})
    external["builder_pubkey"] = external.builder_pubkey.astype(str).str.casefold()
    external = external.sort_values("14d_total", ascending=False).drop_duplicates(
        "builder_pubkey"
    )
    return external[["builder_pubkey", "external_builder", "14d_total", "total"]]


def plot_validation(key_metrics: pd.DataFrame, monthly: pd.DataFrame, out_dir: Path) -> None:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_FIGSIZE)

    eligible = key_metrics[key_metrics.slots >= 100].copy()
    if len(eligible):
        weights = eligible.slots.to_numpy(dtype=float)
        weights /= weights.sum()
        shares = eligible.dominant_share.to_numpy(dtype=float)
        masks = (
            shares < 0.90,
            (shares >= 0.90) & (shares < 0.95),
            (shares >= 0.95) & (shares < 0.99),
            (shares >= 0.99) & (~np.isclose(shares, 1.0)),
            np.isclose(shares, 1.0),
        )
        bin_weights = np.asarray([weights[mask].sum() for mask in masks])
        x = np.arange(len(bin_weights))
        bars = axes[0].bar(
            x,
            bin_weights,
            color=["#B8B8B8", "#B8B8B8", LIGHT_BLUE, BLUE, BLUE],
            width=0.72,
        )
        axes[0].set_xticks(
            x,
            [
                r"$<0.90$",
                r"$[0.90,0.95)$",
                r"$[0.95,0.99)$",
                r"$[0.99,1)$",
                r"$1$",
            ],
            rotation=20,
            ha="right",
        )
        for bar, value in zip(bars, bin_weights):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                f"{value:.1%}",
                ha="center",
                va="bottom",
                fontsize=6.2,
            )
    axes[0].set(xlabel="Dominant-tag share per key", ylabel="Share of eligible slots", ylim=(0, 0.96))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

    colors = {
        "raw_key": BLUE,
        "slot_tag": RED,
        "global_dominant_tag": GREEN,
        "stable_else_key": PURPLE,
    }
    labels = {
        "raw_key": "Raw key",
        "slot_tag": "Slot tag",
        "global_dominant_tag": "Key's dominant tag",
        "stable_else_key": "Stable tag; key fallback",
    }
    for variant in labels:
        group = monthly[monthly.variant == variant]
        axes[1].plot(
            pd.to_datetime(group.date),
            group.hhi,
            marker="o",
            markersize=2.5,
            color=colors.get(variant),
            label=labels.get(variant, variant),
        )
    axes[1].set(xlabel="Month", ylabel="HHI", ylim=(0, None))
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        columnspacing=0.8,
        borderaxespad=0,
    )

    finish_axis(axes[0], grid_axis="y")
    finish_axis(axes[1])
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.25, top=0.72, wspace=0.32)
    fig.savefig(out_dir / "Fig_identity_proxy_validation.pdf")
    fig.savefig(out_dir / "Fig_identity_proxy_validation.png", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        key_metrics = pd.read_csv(args.out_dir / "key_tag_metrics.csv")
        monthly = pd.read_csv(args.out_dir / "identity_concentration_variants.csv")
        plot_validation(key_metrics, monthly, args.out_dir)
        print(f"regenerated figure in {args.out_dir}")
        return

    columns = [
        "slot",
        "block_number",
        "builder_pubkey",
        "slot_start_date_time",
        "extra_data_tag",
    ]
    frame = pd.read_parquet(args.payloads, columns=columns)
    frame["builder_pubkey"] = frame.builder_pubkey.astype(str).str.casefold()
    frame["extra_data_tag"] = frame.extra_data_tag.fillna("").map(canonical_text)
    frame["slot_start_date_time"] = pd.to_datetime(
        frame.slot_start_date_time, utc=True
    )
    frame = frame.sort_values(["builder_pubkey", "slot_start_date_time", "slot"])
    frame["month"] = frame.slot_start_date_time.dt.tz_localize(None).dt.to_period("M")

    tagged = frame[frame.extra_data_tag.str.len() > 0].copy()
    key_tag_counts = (
        tagged.groupby(["builder_pubkey", "extra_data_tag"], observed=True)
        .size()
        .rename("slots")
        .reset_index()
    )
    key_totals = key_tag_counts.groupby("builder_pubkey").slots.sum()
    key_n_tags = key_tag_counts.groupby("builder_pubkey").extra_data_tag.nunique()
    dominant_index = key_tag_counts.groupby("builder_pubkey").slots.idxmax()
    dominant = key_tag_counts.loc[dominant_index].set_index("builder_pubkey")

    entropy = key_tag_counts.groupby("builder_pubkey").slots.apply(
        normalized_entropy
    )
    key_metrics = pd.DataFrame(
        {
            "slots": key_totals,
            "tags": key_n_tags,
            "dominant_tag": dominant.extra_data_tag,
            "dominant_slots": dominant.slots,
            "normalized_tag_entropy": entropy,
        }
    )
    key_metrics["dominant_share"] = key_metrics.dominant_slots / key_metrics.slots

    transitions = tagged[["builder_pubkey", "extra_data_tag"]].copy()
    transitions["previous"] = transitions.groupby("builder_pubkey").extra_data_tag.shift()
    transitions = transitions[transitions.previous.notna()]
    transition_metrics = transitions.assign(
        changed=transitions.extra_data_tag != transitions.previous
    ).groupby("builder_pubkey").changed.agg(["size", "sum"])
    transition_metrics = transition_metrics.rename(
        columns={"size": "tag_transitions", "sum": "tag_changes"}
    )
    transition_metrics["tag_change_rate"] = (
        transition_metrics.tag_changes / transition_metrics.tag_transitions
    )
    key_metrics = key_metrics.join(transition_metrics, how="left").fillna(
        {"tag_transitions": 0, "tag_changes": 0, "tag_change_rate": 0.0}
    )

    monthly_key_tag = (
        tagged.groupby(["builder_pubkey", "month", "extra_data_tag"], observed=True)
        .size()
        .rename("slots")
        .reset_index()
    )
    monthly_dominant = monthly_key_tag.loc[
        monthly_key_tag.groupby(["builder_pubkey", "month"]).slots.idxmax()
    ].sort_values(["builder_pubkey", "month"])
    monthly_dominant["previous_tag"] = monthly_dominant.groupby(
        "builder_pubkey"
    ).extra_data_tag.shift()
    monthly_dominant["previous_month"] = monthly_dominant.groupby(
        "builder_pubkey"
    ).month.shift()
    comparable = monthly_dominant[monthly_dominant.previous_tag.notna()].copy()
    comparable["consecutive"] = (
        comparable.month.astype(int) - comparable.previous_month.astype(int)
    ) == 1
    comparable = comparable[comparable.consecutive]
    comparable["same_tag"] = comparable.extra_data_tag == comparable.previous_tag

    tag_metrics = (
        key_tag_counts.groupby("extra_data_tag")
        .agg(slots=("slots", "sum"), keys=("builder_pubkey", "nunique"))
        .sort_values("slots", ascending=False)
    )
    tag_dominant = key_tag_counts.loc[
        key_tag_counts.groupby("extra_data_tag").slots.idxmax()
    ].set_index("extra_data_tag")
    tag_metrics["dominant_key"] = tag_dominant.builder_pubkey
    tag_metrics["dominant_key_share"] = tag_dominant.slots / tag_metrics.slots

    dominant_map = key_metrics.dominant_tag.to_dict()
    stable_map = (
        key_metrics.dominant_share >= args.stable_threshold
    ).to_dict()
    frame["raw_key"] = "key:" + frame.builder_pubkey
    frame["slot_tag"] = np.where(
        frame.extra_data_tag.str.len() > 0,
        "tag:" + frame.extra_data_tag,
        frame.raw_key,
    )
    frame["global_dominant_tag"] = frame.builder_pubkey.map(dominant_map)
    frame["global_dominant_tag"] = np.where(
        frame.global_dominant_tag.notna(),
        "tag:" + frame.global_dominant_tag.fillna(""),
        frame.raw_key,
    )
    frame["stable_else_key"] = np.where(
        frame.builder_pubkey.map(stable_map).fillna(False),
        frame.global_dominant_tag,
        frame.raw_key,
    )

    monthly_frames = [
        concentration(frame, column, column)
        for column in [
            "raw_key",
            "slot_tag",
            "global_dominant_tag",
            "stable_else_key",
        ]
    ]
    for threshold in (0.90, 0.99):
        column = f"stable_{int(threshold * 100)}_else_key"
        threshold_map = (key_metrics.dominant_share >= threshold).to_dict()
        frame[column] = np.where(
            frame.builder_pubkey.map(threshold_map).fillna(False),
            frame.global_dominant_tag,
            frame.raw_key,
        )
        monthly_frames.append(concentration(frame, column, column))
    monthly = pd.concat(monthly_frames, ignore_index=True)

    external_summary: dict[str, object] = {"available": False}
    external_crosscheck = pd.DataFrame()
    if args.external_html and args.external_html.exists():
        external = parse_external_table(args.external_html)
        external_crosscheck = key_metrics.reset_index().merge(
            external, on="builder_pubkey", how="inner"
        )
        if len(external_crosscheck):
            external_crosscheck["exact_string_match"] = (
                external_crosscheck.dominant_tag.map(canonical_text)
                == external_crosscheck.external_builder.map(canonical_text)
            )
            external_crosscheck["brand_string_match"] = (
                external_crosscheck.dominant_tag.map(brand_text)
                == external_crosscheck.external_builder.map(brand_text)
            )
            external_summary = {
                "available": True,
                "external_rows": int(len(external)),
                "overlap_keys": int(len(external_crosscheck)),
                "overlap_slots": int(external_crosscheck.slots.sum()),
                "exact_string_match_key_share": float(
                    external_crosscheck.exact_string_match.mean()
                ),
                "exact_string_match_slot_share": float(
                    np.average(
                        external_crosscheck.exact_string_match,
                        weights=external_crosscheck.slots,
                    )
                ),
                "brand_string_match_slot_share": float(
                    np.average(
                        external_crosscheck.brand_string_match,
                        weights=external_crosscheck.slots,
                    )
                ),
            }

    eligible = key_metrics[key_metrics.slots >= args.min_key_slots]
    stability_by_min_slots = pd.DataFrame(
        [
            {
                "min_key_slots": minimum,
                "eligible_keys": int(len(subset)),
                "slot_share": float(subset.slots.sum() / key_metrics.slots.sum()),
                "median_dominant_share": float(subset.dominant_share.median()),
                "keys_at_least_90_share": float((subset.dominant_share >= 0.90).mean()),
                "keys_at_least_95_share": float((subset.dominant_share >= 0.95).mean()),
                "keys_at_least_99_share": float((subset.dominant_share >= 0.99).mean()),
            }
            for minimum in (1, 10, 100, 1000)
            for subset in [key_metrics[key_metrics.slots >= minimum]]
        ]
    )
    slot_weighted_dominant = float(key_metrics.dominant_slots.sum() / key_metrics.slots.sum())
    stable_slot_share = float(
        key_metrics.loc[
            key_metrics.dominant_share >= args.stable_threshold, "slots"
        ].sum()
        / key_metrics.slots.sum()
    )
    changed_transitions = float(
        transitions.extra_data_tag.ne(transitions.previous).sum() / len(transitions)
    ) if len(transitions) else 0.0
    monthly_same_share = float(
        np.average(comparable.same_tag, weights=comparable.slots)
    ) if len(comparable) else float("nan")

    concentration_summary = (
        monthly.groupby("variant")[["hhi", "cr3", "cr5", "effective_identities"]]
        .median()
        .reset_index()
    )
    summary = {
        "input": str(args.payloads),
        "rows": int(len(frame)),
        "tagged_rows": int(len(tagged)),
        "tag_coverage": float(len(tagged) / len(frame)),
        "unique_keys": int(frame.builder_pubkey.nunique()),
        "unique_tags": int(tagged.extra_data_tag.nunique()),
        "slot_weighted_key_dominant_tag_share": slot_weighted_dominant,
        "stable_threshold": args.stable_threshold,
        "stable_key_slot_share": stable_slot_share,
        "eligible_keys_min_slots": int(len(eligible)),
        "eligible_key_median_dominant_share": float(eligible.dominant_share.median()),
        "eligible_keys_at_least_threshold_share": float(
            (eligible.dominant_share >= args.stable_threshold).mean()
        ),
        "adjacent_observation_tag_change_rate": changed_transitions,
        "consecutive_month_dominant_tag_agreement_slot_weighted": monthly_same_share,
        "slots_under_multi_key_tags_share": float(
            tag_metrics.loc[tag_metrics["keys"] > 1, "slots"].sum()
            / tag_metrics.slots.sum()
        ),
        "median_monthly_concentration": concentration_summary.to_dict("records"),
        "external_temporal_crosscheck": external_summary,
    }

    key_metrics.reset_index().to_csv(args.out_dir / "key_tag_metrics.csv", index=False)
    tag_metrics.reset_index().to_csv(args.out_dir / "tag_key_metrics.csv", index=False)
    monthly_dominant.to_csv(args.out_dir / "monthly_key_dominant_tags.csv", index=False)
    monthly.to_csv(args.out_dir / "identity_concentration_variants.csv", index=False)
    concentration_summary.to_csv(
        args.out_dir / "identity_concentration_summary.csv", index=False
    )
    stability_by_min_slots.to_csv(
        args.out_dir / "key_stability_threshold_sensitivity.csv", index=False
    )
    if len(external_crosscheck):
        external_crosscheck.to_csv(
            args.out_dir / "external_temporal_crosscheck.csv", index=False
        )
    (args.out_dir / "identity_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_validation(key_metrics.reset_index(), monthly, args.out_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
