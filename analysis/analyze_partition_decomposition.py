"""Decompose identity-dependent HHI on the unchanged, frozen payload support.

For two partitions A and B and their common contingency table n_ab,
HHI(B)-HHI(A) = sum_b (n_b^2-sum_a n_ab^2)/N^2
                 - sum_a (n_a^2-sum_b n_ab^2)/N^2.
Each component is calculated from the same payload contingency table.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

COMMON_SHA = "06bac99220ff55814d91d43a7d2082f6409968bd98078f4dec946bab485c7c78"
LABEL_SHA = "ed984f72d574ed69352ad983366d9df2c0325a02eb22523b71953384dad099cd"
PAIRS = (
    ("raw_key", "slot_tag"),
    ("raw_key", "historical_service"),
    ("slot_tag", "historical_service"),
    ("historical_service", "historical_network"),
    ("raw_key", "official_historical_service"),
    ("official_historical_service", "historical_service"),
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def decompose(table):
    table = np.asarray(table)
    if table.ndim != 2 or not np.issubdtype(table.dtype, np.integer):
        raise ValueError("Expected a two-dimensional integer contingency table")
    if np.any(table < 0):
        raise ValueError("Negative counts")
    n = sum(int(x) for x in table.ravel())
    if not 0 < n <= 3_000_000_000:
        raise ValueError("Empty or unsupported sample size")
    table = table.astype(np.int64)
    row = table.sum(axis=1)
    col = table.sum(axis=0)
    squares = table * table
    merge_numerators = col * col - squares.sum(axis=0)
    split_numerators = row * row - squares.sum(axis=1)
    source_square = sum(int(x) ** 2 for x in row)
    target_square = sum(int(x) ** 2 for x in col)
    merge = sum(int(x) for x in merge_numerators)
    split = sum(int(x) for x in split_numerators)
    if merge - split != target_square - source_square:
        raise AssertionError("Integer partition identity failed")
    return {
        "slots": n,
        "source_identities": int(np.count_nonzero(row)),
        "target_identities": int(np.count_nonzero(col)),
        "source_hhi": source_square / n**2,
        "target_hhi": target_square / n**2,
        "merge_hhi": merge / n**2,
        "split_hhi": split / n**2,
        "delta_hhi": (merge - split) / n**2,
        "merge_ordered_pairs": merge,
        "split_ordered_pairs": split,
    }, row, col, merge_numerators, split_numerators


def contingency(a, b, n_a, n_b):
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b):
        raise ValueError("Unaligned labels")
    if not np.issubdtype(a.dtype, np.integer) or not np.issubdtype(b.dtype, np.integer):
        raise ValueError("Labels must be integer codes")
    if np.any(a < 0) or np.any(a >= n_a) or np.any(b < 0) or np.any(b >= n_b):
        raise ValueError("Code outside its dictionary")
    return np.bincount(a.astype(np.int64) * n_b + b, minlength=n_a*n_b).reshape(n_a, n_b)


def code_labels(labels):
    code, names = pd.factorize(labels, sort=True)
    if np.any(code < 0):
        raise ValueError("Missing identity")
    return code.astype(np.int32), np.asarray(names)


def run(args):
    source = args.root
    verified = json.loads((source / "completion.json").read_text())
    pins = {args.input: COMMON_SHA, source / "analysis_labels.parquet": LABEL_SHA}
    for name in ("concentration.csv", "key_mapping.csv", "identity_tag_contingency.csv"):
        pins[source / name] = verified["outputs"][name]
    for path, expected in pins.items():
        if sha256(path) != expected:
            raise ValueError(f"Input checksum mismatch: {path}")
    if args.out.exists():
        raise FileExistsError(f"Output must be new: {args.out}")

    frame = pd.read_parquet(args.input, columns=[
        "slot", "slot_start_date_time", "builder_pubkey", "extra_data_tag"
    ]).sort_values("slot").reset_index(drop=True)
    labels = pd.read_parquet(source / "analysis_labels.parquet")
    if len(frame) != 2332997 or frame.slot.duplicated().any():
        raise ValueError("Unexpected canonical support")
    if not np.array_equal(frame.slot, labels.slot):
        raise ValueError("Frozen label support differs")
    frame["builder_pubkey"] = frame.builder_pubkey.str.casefold()
    frame["extra_data_tag"] = frame.extra_data_tag.fillna("")
    months = pd.to_datetime(frame.slot_start_date_time, utc=True).dt.strftime("%Y-%m").to_numpy()
    raw_names = ("key:" + frame.builder_pubkey).to_numpy()
    tag_names = np.where(frame.extra_data_tag.ne(""), "tag:" + frame.extra_data_tag, raw_names)
    codes, names = {}, {}
    for variant, values in (("raw_key", raw_names), ("slot_tag", tag_names)):
        codes[variant], names[variant] = code_labels(values)
        if not np.array_equal(codes[variant], labels[variant]):
            raise ValueError(f"Canonical identity reconstruction differs: {variant}")
    mapping = pd.read_csv(source / "key_mapping.csv", keep_default_na=False)
    for variant in ("historical_service", "historical_network", "official_historical_service"):
        sub = mapping[mapping.variant.eq(variant)].set_index("builder_pubkey")
        if sub.index.duplicated().any() or set(frame.builder_pubkey) != set(sub.index):
            raise ValueError("Mapping does not cover keys exactly once")
        mapped = frame.builder_pubkey.map(sub.name)
        values = np.where(mapped.ne(""), "external:" + mapped, raw_names)
        codes[variant], names[variant] = code_labels(values)
        if variant in labels and not np.array_equal(codes[variant], labels[variant]):
            raise ValueError(f"Frozen external labels differ: {variant}")

    reference = pd.read_csv(source / "concentration.csv").set_index(["variant", "month"])
    unique_months = sorted(set(months))
    summary, detail, cells, checks = [], [], [], []
    for month in ["all", *unique_months]:
        keep = np.ones(len(frame), dtype=bool) if month == "all" else months == month
        n = int(keep.sum())
        for variant in codes:
            count = np.bincount(codes[variant][keep], minlength=len(names[variant]))
            share = np.sort(count.astype(float) / n)[::-1]
            actual = dict(slots=n, identities=int(np.count_nonzero(count)),
                          hhi=float(share @ share), cr3=float(share[:3].sum()), cr5=float(share[:5].sum()))
            expected = reference.loc[(variant, month)]
            for metric, value in actual.items():
                if not np.isclose(value, expected[metric], rtol=0, atol=1e-12):
                    raise ValueError(f"Existing result differs: {variant}, {month}, {metric}")
            checks.append(dict(variant=variant, month=month, **actual))
        for a, b in PAIRS:
            tab = contingency(codes[a][keep], codes[b][keep], len(names[a]), len(names[b]))
            stat, row, col, merge, split = decompose(tab)
            if a == "raw_key" and b != "slot_tag" and stat["split_ordered_pairs"]:
                raise ValueError("Static key mapping unexpectedly splits keys")
            meta = dict(source_unit=a, target_unit=b, month=month,
                        partial_edge_month=month in (unique_months[0], unique_months[-1]))
            summary.append(dict(**meta, **stat))
            for direction, unit, counts, numerator in (("merge", b, col, merge), ("split", a, row, split)):
                for idx in np.flatnonzero(numerator):
                    detail.append(dict(**meta, direction=direction, identity=names[unit][idx],
                                       identity_slots=int(counts[idx]), ordered_pairs=int(numerator[idx]),
                                       hhi_component=int(numerator[idx])/n**2))
            if month == "all":
                for i, j in zip(*np.nonzero(tab)):
                    cells.append(dict(source_unit=a, target_unit=b,
                                      source_identity=names[a][i], target_identity=names[b][j], slots=int(tab[i,j])))

    result = pd.DataFrame(summary)
    aggregates = []
    for (a, b), group in result[result.month.ne("all")].groupby(["source_unit", "target_unit"]):
        for scope in ("all_14_months", "full_12_months"):
            selected = group if scope == "all_14_months" else group[~group.partial_edge_month]
            mean = selected[["source_hhi", "target_hhi", "merge_hhi", "split_hhi", "delta_hhi"]].mean()
            aggregates.append(dict(source_unit=a, target_unit=b, scope=scope, months=len(selected),
                                   **{f"mean_{k}": float(v) for k,v in mean.items()},
                                   median_paired_delta_hhi=float(selected.delta_hhi.median()),
                                   difference_of_median_hhi=float(selected.target_hhi.median()-selected.source_hhi.median()),
                                   min_paired_delta_hhi=float(selected.delta_hhi.min()),
                                   max_paired_delta_hhi=float(selected.delta_hhi.max())))
    args.out.mkdir(parents=True, exist_ok=False)
    result.to_csv(args.out / "partition_differences.csv", index=False)
    pd.DataFrame(detail).to_csv(args.out / "identity_contributions.csv", index=False)
    pd.DataFrame(cells).to_csv(args.out / "full_period_contingencies.csv", index=False)
    pd.DataFrame(checks).to_csv(args.out / "reference_reproduction.csv", index=False)
    pd.DataFrame(aggregates).to_csv(args.out / "paired_month_summaries.csv", index=False)
    completion = dict(status="verified", rows=len(frame), comparisons=len(result),
                      reproduced_reference_rows=len(checks), reproduced_reference_values=len(checks)*5,
                      fixed_pairs=[list(x) for x in PAIRS],
                      input_sha256={str(p):h for p,h in pins.items()},
                      code_sha256=sha256(__file__),
                      finished_at=datetime.now(timezone.utc).isoformat(),
                      python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      interpretation="Identity-partition decomposition on identical payload support.")
    completion["output_sha256"] = {p.name:sha256(p) for p in sorted(args.out.iterdir()) if p.is_file()}
    (args.out / "completion.json").write_text(json.dumps(completion, indent=2)+"\n", encoding="utf-8")
    print(result[result.month.eq("all")].to_string(index=False), flush=True)
    print(json.dumps({k:completion[k] for k in ("status", "rows", "comparisons", "reproduced_reference_values")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
