"""Matched-support summaries, independent arithmetic checks and plot inputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

COHORTS = {"source_three_hour": "common_source_guard", "retained_raw_three_hour": "common_raw_guard"}
SCOPES = ("all_used_intervals", "original_visible_x_range")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def close(a, b):
    if pd.isna(a) and pd.isna(b):
        return
    np.testing.assert_allclose(float(a), float(b), rtol=1e-12, atol=1e-12)


def validate_hours(frame, paired, final):
    require(len(frame) > 0, "Empty study calendar")
    for name, f in (("hourly", frame), ("paired", paired)):
        require(not f.hour_epoch.duplicated().any(), name + " repeats an hour")
        require((f.hour_epoch % 3600).eq(0).all(), name + " has a non-hour timestamp")
        require((~f.common_raw_guard | f.common_source_guard).all(), "Raw cohort is not nested")
        require(f.month.astype(str).eq(pd.to_datetime(f.hour_epoch, unit="s", utc=True).dt.strftime("%Y%m")).all(), "Month labels disagree with UTC hours")
        require((f.transactions >= 0).all(), "Negative canonical denominator")
        for unit in (("baseline", "corrected") if final else ("baseline",)):
            require(((f[unit + "_candidates"] >= 0) & (f[unit + "_candidates"] <= f.transactions)).all(), "Invalid candidate count")
            np.testing.assert_allclose(f[unit + "_fraction"], f[unit + "_candidates"] / f.transactions.replace(0, np.nan), rtol=1e-12, atol=1e-12, equal_nan=True)
        if not final:
            require(not any(c.startswith(("corrected_", "extraction_")) for c in f.columns), "Baseline contains unrun corrections")
    require(set(paired.hour_epoch) <= set(frame.hour_epoch), "Paired rows outside hourly sample")
    matched = frame.set_index("hour_epoch").loc[paired.hour_epoch]
    for col in ("transactions", "baseline_candidates", "common_source_guard", "common_raw_guard"):
        np.testing.assert_array_equal(paired[col].to_numpy(), matched[col].to_numpy())
    if final:
        np.testing.assert_array_equal(paired.corrected_candidates.to_numpy(), matched.corrected_candidates.to_numpy())
    np.testing.assert_allclose(paired.candidate_row_ratio, paired.candidate_observation_rows / paired.denominator_transactions.replace(0, np.nan), rtol=1e-12, atol=1e-12, equal_nan=True)


def scalar(records, final):
    n = len(records)
    total = sum(int(x["transactions"]) for x in records)
    out = dict(hours=n, canonical_transactions=total)
    for unit in (("baseline", "corrected") if final else ("baseline",)):
        count = sum(int(x[unit + "_candidates"]) for x in records)
        out[unit + "_candidates"] = count
        out[unit + "_weighted"] = count / total if total else np.nan
        out[unit + "_median"] = median([float(x[unit + "_fraction"]) for x in records]) if records else np.nan
    return out


def monthly_calendar(frame, final):
    dates = pd.to_datetime(frame.hour_epoch, unit="s", utc=True)
    months = pd.period_range(dates.min().strftime("%Y-%m"), dates.max().strftime("%Y-%m"), freq="M")
    rows = []
    for cohort, flag in COHORTS.items():
        for period in months:
            month = period.strftime("%Y%m")
            support = frame[frame.month.eq(month)]
            selected = support[support[flag] & support.transactions.gt(0)]
            nstudy = len(support)
            raw = int(selected.common_raw_guard.sum())
            rows.append(dict(cohort=cohort, month=month, calendar_month_hours=int(period.days_in_month * 24),
                             study_hours=nstudy, eligible_hours=len(selected), raw_supported_hours=raw,
                             eligible_fraction_of_study=len(selected) / nstudy if nstudy else np.nan,
                             raw_supported_fraction_of_eligible=raw / len(selected) if len(selected) else np.nan,
                             **scalar(selected.to_dict("records"), final)))
    return pd.DataFrame(rows)


def selected_pairs(paired, cohort, scope):
    mask = paired[COHORTS[cohort]] & paired.used_in_plot & np.isfinite(paired.candidate_row_ratio) & paired.denominator_transactions.gt(0) & paired.transactions.gt(0)
    if scope == "original_visible_x_range":
        mask &= paired.original_x_visible
    return paired[mask]


def decomposition(records, final):
    require(bool(records), "No paired observations")
    R = sum(int(x["candidate_observation_rows"]) for x in records)
    D = sum(int(x["denominator_transactions"]) for x in records)
    T = sum(int(x["transactions"]) for x in records)
    require(D > 0 and T > 0, "Nonpositive paired denominator")
    summary = scalar(records, final)
    legacy, intermediate, aligned = R / D, R / T, summary["baseline_weighted"]
    summary.update(legacy_observation_rows=R, legacy_interval_transactions=D,
                   legacy_interval_weighted=legacy, row_count_canonical_ratio=intermediate,
                   legacy_median=median([float(x["candidate_row_ratio"]) for x in records]),
                   denominator_step_pp=100 * (intermediate - legacy),
                   candidate_reconstruction_step_pp=100 * (aligned - intermediate),
                   alignment_total_pp=100 * (aligned - legacy))
    changes = [100 * (float(x["baseline_fraction"]) - float(x["candidate_row_ratio"])) for x in records]
    summary["median_alignment_delta_pp"] = median(changes)
    summary["p95_abs_alignment_delta_pp"] = float(np.quantile(np.abs(changes), .95))
    close(summary["denominator_step_pp"] + summary["candidate_reconstruction_step_pp"], summary["alignment_total_pp"])
    if final:
        summary["extraction_step_pp"] = 100 * (summary["corrected_weighted"] - aligned)
        summary["final_total_pp"] = 100 * (summary["corrected_weighted"] - legacy)
        delta = [100 * (float(x["corrected_fraction"]) - float(x["baseline_fraction"])) for x in records]
        summary["median_extraction_delta_pp"] = median(delta)
        summary["max_abs_extraction_delta_pp"] = max(abs(x) for x in delta)
        close(summary["alignment_total_pp"] + summary["extraction_step_pp"], summary["final_total_pp"])
    return summary


def paired_decompositions(paired, final):
    rows = []
    for cohort in COHORTS:
        for scope in SCOPES:
            selected = selected_pairs(paired, cohort, scope)
            if len(selected):
                rows.append(dict(cohort=cohort, scope=scope, month="all", **decomposition(selected.to_dict("records"), final)))
            for month in sorted(selected.month.unique()):
                rows.append(dict(cohort=cohort, scope=scope, month=month,
                                 **decomposition(selected[selected.month.eq(month)].to_dict("records"), final)))
    return pd.DataFrame(rows)


def verify_tables(frame, paired, calendar, steps, existing, summary, final):
    checked = 0
    for name, actual, keys in (("monthly.parquet", calendar[calendar.eligible_hours.gt(0)], ["cohort", "month"]),
                              ("paired_monthly.parquet", steps[steps.month.ne("all")], ["cohort", "scope", "month"])):
        reference = existing[name].set_index(keys).sort_index()
        computed = actual.set_index(keys).sort_index()
        require(list(computed.index) == list(reference.index), "Summary rows changed: " + name)
        for col in reference:
            target = "study_hours" if col == "target_calendar_hours" else col
            require(target in computed, "Missing independently computed field: " + target)
            for a, b in zip(computed[target], reference[col]):
                close(a, b)
                checked += 1
    for row in summary["cohorts"]:
        selected = frame[frame[COHORTS[row["cohort"]]] & frame.transactions.gt(0)]
        computed = scalar(selected.to_dict("records"), final)
        for key, value in row.items():
            if key != "cohort":
                close(value, computed[key]); checked += 1
    # Rebuild original groups from ordered rows before applying availability filters.
    used = paired[paired.used_in_plot].reset_index(drop=True)
    original = existing["legacy_plot_groups.parquet"]
    require(len(original) == 2 * ((len(used) + 49) // 50), "Display group coverage changed")
    for row in original.to_dict("records"):
        group = used.iloc[50 * row["group_index"]:50 * (row["group_index"] + 1)]
        selected = selected_pairs(group, row["cohort"], "all_used_intervals")
        require(len(selected) == row["paired_hours"] and len(group) == row["original_intervals"], "Display support differs")
        require(int(group.iloc[0].block_interval_start) == row["block_number"], "Display position changed")
        for col in (["legacy_median", "baseline_median", "corrected_median"] if final else ["legacy_median", "baseline_median"]):
            values = selected["candidate_row_ratio" if col == "legacy_median" else col.replace("median", "fraction")].tolist()
            close(median(values) if values else np.nan, row[col]); checked += 1
    rebuilt = []
    for cohort in COHORTS:
        rows = calendar[calendar.cohort.eq(cohort)].to_dict("records")
        for a, b in zip(rows, rows[1:]):
            if a["eligible_hours"] and b["eligible_hours"]:
                r = dict(cohort=cohort, first_month=a["month"], second_month=b["month"])
                for unit in (("baseline", "corrected") if final else ("baseline",)):
                    r[unit + "_change_pp"] = 100 * (b[unit + "_weighted"] - a[unit + "_weighted"])
                rebuilt.append(r)
    keys = ["cohort", "first_month", "second_month"]
    actual = pd.DataFrame(rebuilt).set_index(keys).sort_index()
    expected = existing["monthly_changes.parquet"].set_index(keys).sort_index()
    require(list(actual.index) == list(expected.index), "Month change support differs")
    for col in expected:
        for a, b in zip(actual[col], expected[col]):
            close(a, b); checked += 1
    return checked


def load_completed(folder, mode):
    expected = "final_paired_comparison_complete" if mode == "final" else "baseline_alignment_comparison_complete"
    manifest = json.loads((folder / "completion.json").read_text())
    require(manifest["status"] == expected, "Comparison completion status mismatch")
    for name, digest in manifest["output_hashes"].items():
        require((folder / name).resolve().is_relative_to(folder.resolve()), "Unsafe manifest path")
        require(sha256(folder / name) == digest, "Changed completed artifact: " + name)
    summary = json.loads((folder / "summary.json").read_text())
    require(summary["status"] == expected and summary["final_extraction_comparison"] == (mode == "final"), "Stage summary mismatch")
    existing = {name: pd.read_parquet(folder / name) for name in ("hourly.parquet", "paired_hours.parquet", "monthly.parquet", "paired_monthly.parquet", "legacy_plot_groups.parquet", "monthly_changes.parquet")}
    return existing, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["baseline", "final"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    folder = args.input.resolve()
    out = args.out.resolve()
    require(not out.exists(), "Output directory must be new")
    names = ["coverage_calendar.csv", "paired_estimator_decomposition.csv", "measurement_summary.json", "summary_validation.json"]
    require(not out.is_relative_to(folder), "Output must be outside the input directory")
    tables, previous = load_completed(folder, args.mode)
    final = args.mode == "final"
    frame, paired = tables["hourly.parquet"], tables["paired_hours.parquet"]
    validate_hours(frame, paired, final)
    calendar = monthly_calendar(frame, final)
    steps = paired_decompositions(paired, final)
    comparisons = verify_tables(frame, paired, calendar, steps, tables, previous, final)
    result = dict(stage=args.mode, final_extraction_comparison=final,
                  scope="Matched-hour estimators and source availability",
                  calendar_months=int(calendar.month.nunique()), calendar_rows=len(calendar), paired_summary_rows=len(steps),
                  cohort_totals=previous["cohorts"],
                  primary_paired_totals=steps[steps.month.eq("all") & steps.scope.eq("original_visible_x_range")].to_dict("records"),
                  missing_estimates_preserved=True, raw_reextraction_claims_restricted_to_raw_cohort=True,
                  independently_recomputed_values=comparisons)
    out.mkdir(parents=True, exist_ok=False)
    calendar.to_csv(out / names[0], index=False)
    steps.to_csv(out / names[1], index=False)
    (out / names[2]).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    validation = dict(status="summary_verified", mode=args.mode, finished_utc=datetime.now(timezone.utc).isoformat(),
                      comparisons=comparisons, input_completion_sha256=sha256(folder / "completion.json"),
                      code_sha256=sha256(__file__),
                      files={name: sha256(out / name) for name in names[:-1]})
    (out / names[3]).write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
