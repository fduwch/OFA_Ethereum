"""Paired drift-plus-dependence controls; reuse the frozen permutation engine."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import run_controls as base
import summarize_control_endpoints as endpoints

HERE = Path(__file__).resolve().parent
CORE = base.core
ENGINE = base.engine
MODELS = tuple(f"{profile}_r{suffix}" for profile in ("smooth", "step")
               for suffix in ("005", "020", "050"))
RHOS = {"005": .005, "020": .020, "050": .050}
STAGES = {"pilot": (1, 99), "formal": (100, 1999)}
# Portable run_controls.py; generator code identical to the frozen server version.
BASE_CODE_SHA = "cd2e5189768e06b476873e2632561bf5dd42906b757f096385e33bc6f45f9fbf"
STATE = None


def specification(model):
    if model not in MODELS:
        raise ValueError("Unknown joint model")
    profile, suffix = model.split("_r")
    return profile + "_iid", RHOS[suffix]


def copy_mask(slots, baseline_model, replication, rho, seed=base.SEED):
    if baseline_model not in ("smooth_iid", "step_iid") or not 0 <= rho <= 1 or replication < 0:
        raise ValueError("Invalid joint generator specification")
    if len(slots) == 0 or np.any(np.diff(slots) <= 0):
        raise ValueError("Invalid slot support")
    rng = np.random.default_rng([seed, 2, base.MODELS.index(baseline_model), replication])
    copy = rng.random(len(slots)) < rho
    copy[np.r_[True, np.diff(slots) != 1]] = False
    return copy


def generate_at_rho(labels, slots, early, late, baseline_model, replication, rho, seed=base.SEED):
    fresh = base.generate(labels, slots, early, late, baseline_model, replication, seed)
    copy = copy_mask(slots, baseline_model, replication, rho, seed)
    return base.copy_at_anchors(fresh, ~copy)


def generate(labels, slots, early, late, model, replication):
    baseline_model, rho = specification(model)
    return generate_at_rho(labels, slots, early, late, baseline_model, replication, rho)


def task_name(task):
    model, rep, width, offset = task
    return f"{model}_r{rep:03d}_w{width:05d}_o{offset:04d}"


def execute_task(task):
    labels, slots, early, late, out, permutations, config_hash, stage = STATE
    model, rep, width, offset = task
    dest = out / "cases" / task_name(task)
    dest.mkdir(exist_ok=False)
    started = time.monotonic()
    synthetic = generate(labels, slots, early, late, model, rep)
    bounded, gaps, starts, ends = ENGINE.layout(slots, width, offset)
    observed = np.stack([CORE.count_runs(synthetic, bounded), CORE.count_runs(synthetic, gaps)])
    baseline_model, rho = specification(model)
    draws = ENGINE.draw_batch(synthetic, bounded, gaps, starts, ends, permutations,
                              base.inner_rng(baseline_model, rep, width, offset))
    diagnostic = {}
    if stage == "pilot":
        exact = ENGINE.exact_window_expectation(synthetic, bounded, starts, ends)
        sd = draws[:, 0].std(0, ddof=1)
        error = np.divide(draws[:, 0].mean(0)-exact, sd/np.sqrt(permutations),
                          out=np.zeros_like(exact), where=sd > 0)
        diagnostic = dict(exact_bounded_mean=exact.tolist(),
                          max_abs_mean_zerror=float(np.max(np.abs(error))),
                          rare_constant_discrepancies=int(np.sum((sd == 0) &
                              (np.abs(draws[:, 0].mean(0)-exact) > 1e-8))))
    records = base.inference_rows(observed, draws, model, rep, width, offset)
    if draws.max() >= 2**31:
        raise ValueError("Output count overflow")
    with (dest / "counts.npz.tmp").open("wb") as handle:
        np.savez_compressed(handle, observed=observed.astype(np.int32), permutations=draws.astype(np.int32))
    os.replace(dest / "counts.npz.tmp", dest / "counts.npz")
    base.write_json(dest / "inference.json", records)
    base.write_json(dest / "completion.json", dict(
        status="complete", task=list(task), configuration_sha256=config_hash,
        synthetic_sha256=base.array_hash(synthetic), baseline_model=baseline_model,
        copy_probability=rho, rows=len(slots), permutations=permutations,
        elapsed_seconds=time.monotonic()-started, finished_utc=base.now(), **diagnostic,
        output_sha256={name: CORE.sha256(dest/name) for name in ("counts.npz", "inference.json")}))
    return task_name(task)


def verify_cases(out, tasks, config_hash):
    complete = set()
    for task in tasks:
        dest = out / "cases" / task_name(task)
        if not dest.exists():
            continue
        marker = dest / "completion.json"
        if not marker.exists():
            raise RuntimeError(f"Uncommitted case requires inspection: {dest}")
        meta = json.loads(marker.read_text())
        if meta["configuration_sha256"] != config_hash or meta["task"] != list(task):
            raise ValueError("Case configuration mismatch")
        for name, digest in meta["output_sha256"].items():
            if CORE.sha256(dest/name) != digest:
                raise ValueError("Case output changed")
        complete.add(task_name(task))
    return complete


def baseline_preflight(labels, slots, early, late, source):
    """Check exact rho=0 nesting and two permutation draws against archived cases."""
    checks = []
    for model in ("smooth_iid", "step_iid"):
        for replication in (0, 99):
            synthetic = generate_at_rho(labels, slots, early, late, model, replication, 0.)
            for width, offset in ((10000, 0), (50, 25)):
                task = (model, replication, width, offset)
                dest = source / "cases" / base.task_name(task)
                meta = json.loads((dest / "completion.json").read_text())
                if base.array_hash(synthetic) != meta["synthetic_sha256"]:
                    raise ValueError("Baseline outer sample not reproduced")
                for name, digest in meta["output_sha256"].items():
                    if CORE.sha256(dest/name) != digest:
                        raise ValueError("Archived baseline case changed")
                bounded, gaps, starts, ends = ENGINE.layout(slots, width, offset)
                observed = np.stack([CORE.count_runs(synthetic, bounded), CORE.count_runs(synthetic, gaps)])
                draws = ENGINE.draw_batch(synthetic, bounded, gaps, starts, ends, 2,
                                          base.inner_rng(model, replication, width, offset))
                with np.load(dest / "counts.npz") as old:
                    np.testing.assert_array_equal(observed, old["observed"])
                    np.testing.assert_array_equal(draws, old["permutations"][:2])
                checks.append(dict(task=list(task), outer_sample_and_observed_and_draw_prefix_identical=True))
    return checks


def paired_event_counts(frame, baseline):
    """Paired event counts at each setting; no independence-assuming difference CI."""
    group = ["replication", "width_slots", "offset_slots", "definition"]
    rows = []
    for model in MODELS:
        base_model, rho = specification(model)
        left = frame[frame.model.eq(model)]
        right = baseline[baseline.model.eq(base_model)]
        for endpoint, thresholds in (("all_coordinates", [2, 3, 4, 5, 6, 8, 10]),
                                      ("long_runs_8_10", [8, 10])):
            for direction in ("two_sided", "positive", "negative"):
                def events(part):
                    part = part[part.threshold.isin(thresholds)]
                    rejected = part.across_settings_p.le(.05)
                    if direction == "positive":
                        rejected &= part.observed.gt(part.permutation_mean)
                    elif direction == "negative":
                        rejected &= part.observed.lt(part.permutation_mean)
                    return part.assign(event=rejected).groupby(group).event.any()
                a, b = events(left), events(right)
                if not a.index.equals(b.index):
                    raise ValueError("Unpaired inference grid")
                paired = pd.DataFrame({"joint": a, "baseline": b}).reset_index()
                for key, sub in paired.groupby(group[1:]):
                    n11 = int((sub.joint & sub.baseline).sum())
                    n10 = int((sub.joint & ~sub.baseline).sum())
                    n01 = int((~sub.joint & sub.baseline).sum())
                    n00 = int((~sub.joint & ~sub.baseline).sum())
                    rows.append(dict(model=model, baseline_model=base_model, copy_probability=rho,
                        **dict(zip(group[1:], key)), endpoint=endpoint, direction=direction,
                        outer_pairs=len(sub), both_reject=n11, joint_only=n10, baseline_only=n01,
                        neither_reject=n00, paired_rejection_difference=(n10-n01)/len(sub)))
    return pd.DataFrame(rows)


def finish(out, tasks, config, pins, baseline):
    records, manifest, sample_hashes, gap_observed = [], {}, {}, {}
    for task in tasks:
        dest = out / "cases" / task_name(task)
        meta = json.loads((dest / "completion.json").read_text())
        manifest[task_name(task)] = CORE.sha256(dest / "completion.json")
        outer = task[:2]
        if sample_hashes.setdefault(outer, meta["synthetic_sha256"]) != meta["synthetic_sha256"]:
            raise ValueError("Outer sample differs across settings")
        with np.load(dest / "counts.npz") as data:
            if not np.array_equal(gap_observed.setdefault(outer, data["observed"][1]), data["observed"][1]):
                raise ValueError("Gap-only observation differs across settings")
            # Independently regenerate each inference row from saved counts.
            reconstructed = base.inference_rows(data["observed"], data["permutations"], *task)
        stored = json.loads((dest / "inference.json").read_text())
        if reconstructed != stored:
            raise ValueError("Saved inference does not reproduce from saved counts")
        records.extend(stored)
    detail = pd.DataFrame(records).sort_values(endpoints.KEYS)
    endpoints.validate_grid(detail, config)
    detail.to_parquet(out / "all_outer_inference.parquet", index=False)
    for table, name in zip(base.summarize(detail), ("coordinate", "setting", "global")):
        table.to_csv(out / f"{name}_rejection_rates.csv", index=False)
    setting_rates, global_rates = endpoints.endpoint_summaries(detail)
    setting_rates.to_csv(out / "setting_endpoint_rates.csv", index=False)
    global_rates.to_csv(out / "global_endpoint_rates.csv", index=False)
    if config["stage"] == "formal":
        prior = pd.read_parquet(baseline / "all_outer_inference.parquet")
        prior = prior[prior.replication.lt(config["outer_samples_per_model"]) &
                      prior.model.isin(["smooth_iid", "step_iid"])]
        prior_config = dict(config, models=["smooth_iid", "step_iid"])
        endpoints.validate_grid(prior, prior_config)
        paired_event_counts(detail, prior).to_csv(out / "paired_setting_event_counts.csv", index=False)
        ps, pg = endpoints.endpoint_summaries(prior)
        ps.to_csv(out / "paired_baseline_setting_rates.csv", index=False)
        pg.to_csv(out / "paired_baseline_global_rates.csv", index=False)
    base.write_json(out / "case_manifest.json", manifest)
    for path, expected in pins.items():
        if CORE.sha256(path) != expected:
            raise ValueError(f"Pinned source changed: {path}")
    base.write_json(out / "completion.json", dict(status="complete", finished_utc=base.now(),
        stage=config["stage"], tasks=len(tasks), outer_samples=len(sample_hashes),
        inference_rows=len(detail), inference_recomputed_from_all_saved_counts=True,
        configuration_sha256=CORE.sha256(out / "configuration.json"),
        output_sha256={p.name: CORE.sha256(p) for p in sorted(out.iterdir())
                      if p.is_file() and p.name not in ("progress.json", "completion.json")}))


def run(args):
    if not 1 <= args.workers <= 48:
        raise ValueError("Use 1-48 workers")
    outer, permutations = STAGES[args.stage]
    out = args.output_root.resolve() / f"controls_joint_{args.stage}_v1"
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / (out.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = args.labels.resolve()
        pins = {source: base.LABEL_SHA, HERE / "persistence_test.py": base.CORE_SHA,
                HERE / "window_engine.py": base.ENGINE_SHA, HERE / "run_controls.py": BASE_CODE_SHA}
        for path, expected in pins.items():
            if CORE.sha256(path) != expected:
                raise ValueError(f"Input/source hash mismatch: {path}")
        baseline = args.baseline.resolve()
        marker = json.loads((baseline / "completion.json").read_text())
        if marker["status"] != "complete" or marker["tasks"] != 19200:
            raise ValueError("Baseline formal experiment incomplete")
        for name in ("all_outer_inference.parquet", "configuration.json"):
            path = baseline / name
            if CORE.sha256(path) != marker["output_sha256"][name]:
                raise ValueError("Baseline aggregate changed")
            pins[path] = CORE.sha256(path)
        for path in (Path(__file__), HERE / "test_joint_controls.py", HERE / "summarize_control_endpoints.py",
                     baseline / "completion.json"):
            pins[path] = CORE.sha256(path)
        frame = pd.read_parquet(source)
        slots = frame.slot.to_numpy(dtype=np.int64)
        labels = np.ascontiguousarray(frame[list(base.UNITS)].to_numpy(dtype=np.int32))
        if len(frame) != 2332997 or np.any(np.diff(slots) <= 0):
            raise ValueError("Unexpected sample support")
        early, late = base.make_profiles(slots)
        config = dict(stage=args.stage, seed=base.SEED, copy_seed_stream=2,
            outer_samples_per_model=outer, permutations=permutations, models=list(MODELS),
            units=list(base.UNITS), widths=list(ENGINE.WIDTHS), origins="zero_and_half_width",
            definitions=list(ENGINE.DEFINITIONS), thresholds=CORE.THRESHOLDS.tolist(),
            multiplicity_settings=32, rows=len(slots), baseline_replications=list(range(outer)),
            rho_zero_baselines=["smooth_iid", "step_iid"],
            input_sha256={str(p): h for p, h in pins.items()},
            python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__)
        if args.resume:
            if json.loads((out / "configuration.json").read_text()) != config:
                raise ValueError("Resume configuration differs")
            if (out / "completion.json").exists():
                raise ValueError("Already complete")
        else:
            out.mkdir(exist_ok=False)
            (out / "cases").mkdir()
            base.write_json(out / "configuration.json", config)
        config_hash = CORE.sha256(out / "configuration.json")
        base.write_json(out / "baseline_preflight.json", dict(status="verified",
                        checks=baseline_preflight(labels, slots, early, late, baseline)))
        tasks = [(m, r, w, o) for r in range(outer) for w in ENGINE.WIDTHS
                 for m in MODELS for o in (0, w//2)]
        done = verify_cases(out, tasks, config_hash)
        pending = [t for t in tasks if task_name(t) not in done]
        warm = ENGINE.layout(np.arange(24, dtype=np.int64), 50, 0)
        ENGINE.draw_batch(labels[:24], *warm, 2, np.random.default_rng(0))
        ENGINE.exact_window_expectation(labels[:24], warm[0], warm[2], warm[3])
        global STATE
        STATE = labels, slots, early, late, out, permutations, config_hash, args.stage
        started = time.monotonic()
        def progress(status="running"):
            base.write_json(out / "progress.json", dict(status=status, pid=os.getpid(),
                updated_utc=base.now(), completed_tasks=len(done), total_tasks=len(tasks),
                workers=args.workers, elapsed_this_session_seconds=time.monotonic()-started))
        progress()
        with mp.get_context("fork").Pool(args.workers) as pool:
            iterator = pool.imap_unordered(execute_task, pending, chunksize=1)
            while len(done) < len(tasks):
                try:
                    name = iterator.next(timeout=30)
                except mp.TimeoutError:
                    progress()
                    continue
                done.add(name)
                progress()
                print(f"{base.now()} {args.stage} {len(done)}/{len(tasks)} {name}", flush=True)
        verify_cases(out, tasks, config_hash)
        progress("validating")
        finish(out, tasks, config, pins, baseline)
        progress("complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--labels", type=Path, required=True,
                        help="analysis_labels.parquet from the external-identity analysis")
    parser.add_argument("--baseline", type=Path, required=True,
                        help="completed controls_formal_v1 directory (rho=0 baselines)")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.detach:
        logs = args.output_root.resolve() / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / f"joint_controls_{args.stage}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}.log"
        command = [sys.executable, "-u", str(Path(__file__).resolve()), args.stage,
                   "--labels", str(args.labels.resolve()), "--baseline", str(args.baseline.resolve()),
                   "--output-root", str(args.output_root.resolve()), "--workers", str(args.workers)]
        if args.resume:
            command.append("--resume")
        with log.open("xb") as stream:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(pid=proc.pid, log=str(log), stage=args.stage)), flush=True)
    else:
        os.nice(10)
        run(args)
