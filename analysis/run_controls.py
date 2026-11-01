"""Full-support, genuinely nested multi-scale calibration and power controls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

import persistence_test as core
import window_engine as engine

HERE = Path(__file__).resolve().parent
SEED = 2026102407
MODELS = ("stationary_iid", "smooth_iid", "step_iid", "sticky_005", "sticky_020", "sticky_050")
RHOS = {"sticky_005": .005, "sticky_020": .020, "sticky_050": .050}
UNITS = ("historical_service", "historical_network")
STAGES = {"pilot": (1, 99), "formal": (200, 1999), "resolution": (20, 9999)}
LABEL_SHA = "ed984f72d574ed69352ad983366d9df2c0325a02eb22523b71953384dad099cd"
CORE_SHA = "7e79f2ac926a77d05c8f905b120f5e1aab339affe13fed422e04a991d2798890"
ENGINE_SHA = "54535edf74552b3b6d4bc373e39e9a6ad0feaebec7192560f8a24c9fc2388b6f"
_STATE = None


def write_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(tmp, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def array_hash(array):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def make_profiles(slots):
    slots = np.asarray(slots)
    if len(slots) == 0 or np.any(np.diff(slots) <= 0):
        raise ValueError("Unsorted, duplicate or empty slot support")
    groups = np.array_split(np.unique(slots//10000), 4)
    if any(len(g) == 0 for g in groups):
        raise ValueError("Need at least four occupied 10000-slot windows")
    return tuple(np.flatnonzero(np.isin(slots//10000, g)) for g in (groups[0], groups[-1]))


def copy_at_anchors(fresh_labels, resets):
    if len(fresh_labels) != len(resets) or len(resets) == 0 or not resets[0]:
        raise ValueError("Invalid anchor mask")
    anchors = np.maximum.accumulate(np.where(resets, np.arange(len(resets)), 0))
    return np.ascontiguousarray(fresh_labels[anchors])


def generate(labels, slots, early, late, model, replication, seed=SEED):
    if model not in MODELS or replication < 0:
        raise ValueError("Invalid generator specification")
    if len(labels) != len(slots) or len(labels) == 0:
        raise ValueError("Unaligned labels and support")
    rng = np.random.default_rng([seed, 0, MODELS.index(model), replication])
    n = len(labels)
    if model in ("smooth_iid", "step_iid"):
        t = slots - slots[0]
        probability = ((1 + np.sin(2*np.pi*t/10000))/2 if model == "smooth_iid"
                       else ((t+137)//2500 % 2).astype(float))
        choose_late = rng.random(n) < probability
        initial = early[rng.integers(0, len(early), size=n)]
        alternate = late[rng.integers(0, len(late), size=n)]
        indices = np.where(choose_late, alternate, initial)
        return np.ascontiguousarray(labels[indices])
    result = labels[rng.integers(0, n, size=n)]
    if model in RHOS:
        resets = rng.random(n) >= RHOS[model]
        resets[np.r_[True, np.diff(slots) != 1]] = True
        result = copy_at_anchors(result, resets)
    return np.ascontiguousarray(result)


def inner_rng(model, replication, width, offset, seed=SEED):
    return np.random.default_rng([seed, 1, MODELS.index(model), replication, width, offset])


def task_name(task):
    model, rep, width, offset = task
    return f"m{MODELS.index(model)}_r{rep:03d}_w{width:05d}_o{offset:04d}"


def inference_rows(observed, draws, model, rep, width, offset):
    records = []
    for d, definition in enumerate(engine.DEFINITIONS):
        result = core.infer_from_counts(observed[d], draws[:, d])
        for u, unit in enumerate(UNITS):
            for k, threshold in enumerate(core.THRESHOLDS):
                row = dict(model=model, replication=int(rep), width_slots=int(width), offset_slots=int(offset),
                           definition=definition, unit=unit, threshold=int(threshold))
                for name in ("observed", "permutation_mean", "permutation_sd", "pooled_center", "pooled_sd", "z", *core.P_COLUMNS):
                    row[name] = float(result[name][u, k])
                row["across_settings_p"] = min(1., row["family_adjusted_p"] * 32)
                mean = row["permutation_mean"]
                row["observed_expected_ratio"] = row["observed"]/mean if mean > 0 else None
                row["excess_runs"] = row["observed"] - mean
                row["zero_pooled_sd"] = bool(result["zero_pooled_sd"][u, k])
                records.append(row)
    return records


def execute_task(task):
    labels, slots, early, late, out, permutations, config_hash, stage = _STATE
    model, rep, width, offset = task
    dest = out / "cases" / task_name(task)
    if dest.exists():
        raise FileExistsError(f"Unexpected unfinished or duplicate task: {dest}")
    dest.mkdir()
    start = time.monotonic()
    synthetic = generate(labels, slots, early, late, model, rep)
    bounded, gaps, starts, ends = engine.layout(slots, width, offset)
    observed = np.stack([core.count_runs(synthetic, bounded), core.count_runs(synthetic, gaps)])
    draws = engine.draw_batch(synthetic, bounded, gaps, starts, ends, permutations,
                              inner_rng(model, rep, width, offset))
    if stage == "pilot":
        exact = engine.exact_window_expectation(synthetic, bounded, starts, ends)
        sd = draws[:,0].std(0, ddof=1)
        error = np.divide(draws[:,0].mean(0)-exact, sd/np.sqrt(permutations),
                          out=np.zeros_like(exact), where=sd>0)
        if np.any((sd == 0) & (np.abs(draws[:,0].mean(0)-exact) > 1e-8)):
            # Record exact-mean discrepancies for constant Monte Carlo counts.
            rare_constant_discrepancies = int(np.sum((sd == 0) & (np.abs(draws[:,0].mean(0)-exact) > 1e-8)))
        else:
            rare_constant_discrepancies = 0
        diagnostic = dict(exact_bounded_mean=exact.tolist(), max_abs_mean_zerror=float(np.max(np.abs(error))),
                          rare_constant_discrepancies=rare_constant_discrepancies)
    else:
        diagnostic = {}
    if stage == "resolution":
        prior = out.parent / "controls_formal_v1/cases" / task_name(task)
        committed = json.loads((prior / "completion.json").read_text())
        for name, digest in committed["output_sha256"].items():
            if core.sha256(prior/name) != digest:
                raise ValueError("Paired formal output changed")
        with np.load(prior / "counts.npz") as old:
            np.testing.assert_array_equal(observed, old["observed"])
            np.testing.assert_array_equal(draws[:1999], old["permutations"])
        if committed["synthetic_sha256"] != array_hash(synthetic):
            raise ValueError("Paired outer sample changed")
        diagnostic["paired_1999_prefix_identical"] = True
    records = inference_rows(observed, draws, model, rep, width, offset)
    if draws.max() >= 2**31:
        raise ValueError("Count exceeds int32 output range")
    with (dest / "counts.npz.tmp").open("wb") as handle:
        np.savez_compressed(handle, observed=observed.astype(np.int32), permutations=draws.astype(np.int32))
    os.replace(dest / "counts.npz.tmp", dest / "counts.npz")
    write_json(dest / "inference.json", records)
    metadata = dict(status="complete", task=list(task), configuration_sha256=config_hash,
                    synthetic_sha256=array_hash(synthetic), rows=len(labels), permutations=permutations,
                    elapsed_seconds=time.monotonic()-start, finished_utc=now(), **diagnostic,
                    output_sha256={name:core.sha256(dest/name) for name in ("counts.npz", "inference.json")})
    write_json(dest / "completion.json", metadata)
    return task_name(task)


def rate_record(values, **meta):
    count = int(np.sum(np.asarray(values) <= .05))
    lo, hi = core.wilson(count, len(values))
    return dict(**meta, rejections=count, outer_samples=len(values), rejection_rate=count/len(values),
                wilson_95_low=lo, wilson_95_high=hi)


def summarize(frame):
    rates = []
    dimensions = ["model", "width_slots", "offset_slots", "definition", "unit", "threshold"]
    for values, group in frame.groupby(dimensions, sort=True):
        for metric in ("family_adjusted_p", "across_settings_p"):
            rates.append(rate_record(group[metric], **dict(zip(dimensions, values)), metric=metric))
    families = []
    for values, group in frame.groupby(dimensions[:4], sort=True):
        for metric in ("family_adjusted_p", "across_settings_p"):
            minima = group.groupby("replication")[metric].min()
            families.append(rate_record(minima, **dict(zip(dimensions[:4], values)), metric=metric))
    global_rates = []
    for model, group in frame.groupby("model", sort=True):
        minimum = group.groupby("replication").across_settings_p.min()
        global_rates.append(rate_record(minimum, model=model, metric="any_of_all_448_coordinates"))
    return pd.DataFrame(rates), pd.DataFrame(families), pd.DataFrame(global_rates)


def verified_cases(out, tasks, config_hash):
    complete = set()
    for task in tasks:
        folder = out / "cases" / task_name(task)
        if not folder.exists():
            continue
        marker = folder / "completion.json"
        if not marker.exists():
            raise RuntimeError(f"Uncommitted task retained for inspection: {folder}")
        meta = json.loads(marker.read_text())
        if meta["configuration_sha256"] != config_hash or meta["task"] != list(task):
            raise ValueError("Task configuration mismatch")
        for name, digest in meta["output_sha256"].items():
            if core.sha256(folder/name) != digest:
                raise ValueError("Committed task output changed")
        complete.add(task_name(task))
    return complete


def run(args):
    count, permutations = STAGES[args.stage]
    out = args.output_root.resolve() / f"controls_{args.stage}_v1"
    if not 1 <= args.workers <= 48:
        raise ValueError("Use 1-48 workers")
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / (out.name+".lock")).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = args.labels.resolve()
        pins = {source:LABEL_SHA, HERE/"persistence_test.py":CORE_SHA, HERE/"window_engine.py":ENGINE_SHA}
        for path, expected in pins.items():
            if core.sha256(path) != expected:
                raise ValueError(f"Pinned input mismatch: {path}")
        for path in (Path(__file__), HERE/"test_controls.py"):
            pins[path] = core.sha256(path)
        if args.stage == "resolution":
            formal = out.parent / "controls_formal_v1"
            if not (formal / "completion.json").exists():
                raise ValueError("Resolution run requires completed formal run")
        frame = pd.read_parquet(source)
        slots = frame.slot.to_numpy(dtype=np.int64)
        labels = np.ascontiguousarray(frame[list(UNITS)].to_numpy(dtype=np.int32))
        if len(frame) != 2332997 or not np.all(np.diff(slots) > 0):
            raise ValueError("Unexpected sample support")
        early, late = make_profiles(slots)
        config = dict(stage=args.stage, seed=SEED, outer_samples_per_model=count,
                      permutations=permutations, models=list(MODELS), units=list(UNITS),
                      widths=list(engine.WIDTHS), origins="zero_and_half_width",
                      definitions=list(engine.DEFINITIONS), thresholds=core.THRESHOLDS.tolist(),
                      multiplicity_settings=32, rows=len(slots), gaps=int(np.sum(np.r_[True,np.diff(slots)!=1])),
                      early_profile_rows=len(early), late_profile_rows=len(late),
                      input_sha256={str(p):h for p,h in pins.items()},
                      python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__)
        if args.resume:
            old = json.loads((out/"configuration.json").read_text())
            if old != config:
                raise ValueError("Resume configuration/source differs")
            if (out/"completion.json").exists():
                raise ValueError("Already complete")
        else:
            out.mkdir(exist_ok=False)
            (out/"cases").mkdir()
            write_json(out/"configuration.json", config)
        config_hash = core.sha256(out/"configuration.json")
        tasks = [(model, rep, width, offset) for rep in range(count) for width in engine.WIDTHS
                 for model in MODELS for offset in (0,width//2)]
        done = verified_cases(out, tasks, config_hash)
        pending = [task for task in tasks if task_name(task) not in done]
        warm_slots = np.arange(24,dtype=np.int64)
        bounded,gaps,starts,ends = engine.layout(warm_slots,50,0)
        engine.draw_batch(labels[:24],bounded,gaps,starts,ends,2,np.random.default_rng(0))
        engine.exact_window_expectation(labels[:24],bounded,starts,ends)
        global _STATE
        _STATE = (labels,slots,early,late,out,permutations,config_hash,args.stage)
        started = time.monotonic()
        def progress(status="running"):
            write_json(out/"progress.json", dict(status=status,pid=os.getpid(),updated_utc=now(),
                       completed_tasks=len(done),total_tasks=len(tasks),workers=args.workers,
                       elapsed_this_session_seconds=time.monotonic()-started))
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
                print(f"{now()} {args.stage} completed {len(done)}/{len(tasks)} {name}",flush=True)
        verified_cases(out,tasks,config_hash)
        records = []
        sample_hashes = {}
        gap_observed = {}
        case_manifest = {}
        for task in tasks:
            folder=out/"cases"/task_name(task)
            metadata=json.loads((folder/"completion.json").read_text())
            case_manifest[task_name(task)]=core.sha256(folder/"completion.json")
            outer=task[:2]
            if sample_hashes.setdefault(outer,metadata["synthetic_sha256"]) != metadata["synthetic_sha256"]:
                raise ValueError("Outer sample changed across settings")
            with np.load(folder/"counts.npz") as data:
                if not np.array_equal(gap_observed.setdefault(outer,data["observed"][1]),data["observed"][1]):
                    raise ValueError("Gap-only observation changed with conditioning scale")
            records.extend(json.loads((folder/"inference.json").read_text()))
        detail=pd.DataFrame(records).sort_values(["model","replication","width_slots","offset_slots","definition","unit","threshold"])
        if len(detail)!=len(tasks)*28:
            raise ValueError("Incomplete inference grid")
        detail.to_parquet(out/"all_outer_inference.parquet",index=False)
        rates,families,global_rates=summarize(detail)
        rates.to_csv(out/"coordinate_rejection_rates.csv",index=False)
        families.to_csv(out/"setting_rejection_rates.csv",index=False)
        global_rates.to_csv(out/"global_rejection_rates.csv",index=False)
        write_json(out/"case_manifest.json",case_manifest)
        for path, expected in pins.items():
            if core.sha256(path)!=expected:
                raise ValueError(f"Input/source changed while running: {path}")
        completion=dict(status="complete",finished_utc=now(),tasks=len(tasks),outer_samples=len(sample_hashes),
                        configuration_sha256=config_hash,rows_per_outer_sample=len(slots),
                        output_sha256={p.name:core.sha256(p) for p in sorted(out.iterdir())
                                       if p.is_file() and p.name not in ("progress.json","completion.json")})
        write_json(out/"completion.json",completion)
        progress("complete")
        print(global_rates.to_string(index=False),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.nice(10)
    run(args)
