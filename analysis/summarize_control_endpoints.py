"""Summarize complete controls without altering their tests or multiplicity family."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ("stationary_iid", "smooth_iid", "step_iid", "sticky_005", "sticky_020", "sticky_050")
KEYS = ["model", "replication", "width_slots", "offset_slots", "definition", "unit", "threshold"]
DIRECTIONS = ("two_sided", "positive", "negative")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def rate(values):
    values = np.asarray(values, dtype=bool)
    require(values.ndim == 1 and len(values) > 0, "Empty or invalid outer events")
    n, k = len(values), int(values.sum())
    z = 1.959963984540054
    denominator = 1 + z*z/n
    center = (k/n + z*z/(2*n))/denominator
    radius = z*math.sqrt(k/n*(1-k/n)/n + z*z/(4*n*n))/denominator
    return dict(outer_samples=n, rejections=k, rejection_rate=k/n,
                wilson_95_low=max(0., center-radius), wilson_95_high=min(1., center+radius))


def validate_grid(frame, config):
    require(not frame.empty and not frame[KEYS].isna().any().any(), "Empty grid or missing keys")
    require(not frame.duplicated(KEYS).any(), "Duplicate coordinate")
    for key in ("replication", "width_slots", "offset_slots", "threshold"):
        require(np.equal(frame[key], np.floor(frame[key])).all(), "Noninteger grid key")
    n = config["outer_samples_per_model"]
    require(frame.replication.between(0, n-1).all(), "Unexpected replication")
    for key, values in (("model", config["models"]), ("width_slots", config["widths"]),
                        ("definition", config["definitions"]), ("unit", config["units"]),
                        ("threshold", config["thresholds"])):
        require(frame[key].isin(values).all(), "Unexpected grid member: " + key)
    require((frame.offset_slots.eq(0) | frame.offset_slots.eq(frame.width_slots//2)).all(), "Unexpected origin")
    expected = n*len(config["models"])*len(config["widths"])*2*len(config["definitions"])*len(config["units"])*len(config["thresholds"])
    require(len(frame) == expected, "Incomplete grid")
    for key in ("observed", "permutation_mean", "family_adjusted_p", "across_settings_p"):
        require(np.isfinite(frame[key]).all() and frame[key].ge(0).all(), "Invalid numeric value: " + key)
    for key in ("family_adjusted_p", "across_settings_p"):
        require(frame[key].le(1).all(), "Invalid p value")
    require(np.equal(frame.observed, np.floor(frame.observed)).all(), "Noninteger observed count")
    np.testing.assert_allclose(frame.across_settings_p,
                               np.minimum(1., config["multiplicity_settings"]*frame.family_adjusted_p), rtol=0, atol=1e-12)


def endpoint_summaries(frame):
    records, global_records = [], []
    dimensions = ["model", "width_slots", "offset_slots", "definition"]
    for endpoint, subset in (("all_coordinates", frame), ("long_runs_8_10", frame[frame.threshold.isin([8, 10])])):
        require(not subset.empty, "Missing endpoint")
        events = subset[KEYS].copy()
        rejected = subset.across_settings_p.le(.05)
        events["two_sided"] = rejected
        events["positive"] = rejected & subset.observed.gt(subset.permutation_mean)
        events["negative"] = rejected & subset.observed.lt(subset.permutation_mean)
        require((events.positive | events.negative).equals(events.two_sided), "Rejected zero-direction coordinate")
        setting_events = events.groupby(dimensions+["replication"])[list(DIRECTIONS)].any()
        for values, group in setting_events.groupby(level=dimensions, sort=True):
            for direction in DIRECTIONS:
                records.append(dict(zip(dimensions, values), endpoint=endpoint, direction=direction,
                                    **rate(group[direction])))
        global_events = events.groupby(["model", "replication"])[list(DIRECTIONS)].any()
        for model, group in global_events.groupby(level="model", sort=True):
            for direction in DIRECTIONS:
                global_records.append(dict(model=model, endpoint=endpoint, direction=direction,
                                           **rate(group[direction])))
    return pd.DataFrame(records), pd.DataFrame(global_records)


def verify_original_global(summary, original):
    new = summary[summary.endpoint.eq("all_coordinates") & summary.direction.eq("two_sided")].set_index("model").sort_index()
    old = original.set_index("model").sort_index()
    require(new.index.equals(old.index), "Original global model set differs")
    for key in ("outer_samples", "rejections", "rejection_rate", "wilson_95_low", "wilson_95_high"):
        np.testing.assert_allclose(new[key], old[key], rtol=0, atol=1e-12)
    return len(new)*5


def load_control_run(source):
    source = Path(source).resolve()
    marker = json.loads((source / "completion.json").read_text())
    config = json.loads((source / "configuration.json").read_text())
    require(marker["status"] == "complete", "Control run is incomplete")
    require(config["stage"] in ("formal", "resolution"), "Use formal or resolution controls")
    require(marker["configuration_sha256"] == sha(source / "configuration.json"), "Configuration changed")
    for name, digest in marker["output_sha256"].items():
        path = source / name
        require(path.resolve().is_relative_to(source) and sha(path) == digest, "Output changed: " + name)
    frame = pd.read_parquet(source / "all_outer_inference.parquet")
    validate_grid(frame, config)
    return frame, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.input.resolve(), args.out.resolve()
    require(not out.exists() and not out.is_relative_to(source), "Use a new output directory")
    frame, config = load_control_run(source)
    settings, global_rates = endpoint_summaries(frame)
    checks = verify_original_global(global_rates, pd.read_csv(source / "global_rejection_rates.csv"))
    out.mkdir(parents=True, exist_ok=False)
    settings.to_csv(out / "setting_endpoint_rates.csv", index=False)
    global_rates.to_csv(out / "global_endpoint_rates.csv", index=False)
    metadata = dict(stage=config["stage"], outer_samples_per_model=config["outer_samples_per_model"],
                    inference_rows=len(frame), original_global_values_reproduced=checks,
                    output_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()})
    (out / "completion.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
