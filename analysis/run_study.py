"""Nested null calibration, power estimation, and full-sample inference."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

import persistence_test as core

_ARRAYS = None
_ARGS = None


def worker(task):
    mode, rho, replication, seed_words = task
    labels, breaks, starts, ends = _ARRAYS
    rng = np.random.default_rng(np.asarray(seed_words, dtype=np.uint32))
    synthetic = core.synthetic_labels(labels, breaks, starts, ends, rng, mode, rho)
    result = core.complete_test(synthetic, breaks, starts, ends, _ARGS.permutations, rng, _ARGS.alpha)
    return core.result_rows(result, mode=mode, rho=rho, replication=replication)


def real_worker(task):
    seed_words, repetitions = task
    return core.permutation_counts(*_ARRAYS, repetitions, np.random.default_rng(seed_words))


def summarize(details, alpha):
    marginal = []
    for identity, group in details.groupby(["mode", "rho", "unit", "threshold"]):
        mode, rho, unit, threshold = identity
        for metric in core.P_COLUMNS:
            errors = int((group[metric] <= alpha).sum())
            lo, hi = core.wilson(errors, len(group))
            marginal.append(dict(mode=mode, rho=rho, unit=unit, threshold=int(threshold),
                                 test=metric, rejections=errors, replications=len(group),
                                 rejection_rate=errors/len(group), ci_low=lo, ci_high=hi))
    families = []
    for (mode, rho), group in details.groupby(["mode", "rho"]):
        groups = [("all_tests", group, "family_adjusted_p")]
        groups += [(unit, group[group.unit == unit], "unit_adjusted_p") for unit in core.UNITS]
        for family, subset, metric in groups:
            minima = subset.groupby("replication")[metric].min()
            count = int((minima <= alpha).sum())
            lo, hi = core.wilson(count, len(minima))
            families.append(dict(mode=mode, rho=rho, family=family, rejections=count,
                                 replications=len(minima), rejection_rate=count/len(minima),
                                 ci_low=lo, ci_high=hi))
    return pd.DataFrame(marginal), pd.DataFrame(families)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payloads", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["nested", "real"], default="nested")
    parser.add_argument("--null-replications", type=int, default=1000)
    parser.add_argument("--power-replications", type=int, default=300)
    parser.add_argument("--models", nargs="+", choices=["conditional", "iid_sticky"], default=["conditional", "iid_sticky"])
    parser.add_argument("--rhos", type=float, nargs="+", default=[.01, .025, .05, .1])
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--profile-windows", type=int, default=24)
    parser.add_argument("--window-slots", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    if args.workers < 1 or args.permutations < 2 or args.null_replications < 1 or args.power_replications < 0:
        parser.error("Invalid worker, permutation, or replication count")
    if not 0 < args.alpha < 1 or args.window_slots < 1 or args.profile_windows < 0:
        parser.error("Invalid alpha or window settings")
    if any(not 0 < rho <= 1 for rho in args.rhos) or len(set(args.rhos)) != len(args.rhos):
        parser.error("Power rho values must be distinct and in (0,1]")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    input_hash = core.sha256(args.payloads)
    frame, selected = core.load_frame(args.payloads, args.window_slots,
                                      0 if args.mode == "real" else args.profile_windows)
    global _ARRAYS, _ARGS
    _ARRAYS = core.arrays_from_frame(frame, args.window_slots)
    _ARGS = args
    core.count_runs(_ARRAYS[0][:16], _ARRAYS[1][:16])
    metadata = {key: str(value) if isinstance(value, Path) else value for key,value in vars(args).items()}
    metadata.update(input_sha256=input_hash, selected_window_ids=selected, rows=len(frame),
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    source_hashes={name: core.sha256(Path(__file__).parent/name)
                                   for name in ("persistence_test.py", "run_study.py")},
                    status="running", started_unix=started)
    (args.out_dir/"configuration.json").write_text(json.dumps(metadata, indent=2)+"\n")
    if args.mode == "real":
        count = min(args.permutations, args.workers*4)
        sizes = np.full(count, args.permutations//count)
        sizes[:args.permutations % count] += 1
        children = np.random.SeedSequence(args.seed).spawn(count)
        tasks = [(child.generate_state(4), int(n)) for child,n in zip(children,sizes)]
        with mp.get_context("fork").Pool(args.workers) as pool:
            chunks = list(pool.imap(real_worker, tasks))
        permutations = np.concatenate(chunks)
        observed = core.count_runs(_ARRAYS[0], _ARRAYS[1])
        result = core.infer_from_counts(observed, permutations, args.alpha)
        pd.DataFrame(core.result_rows(result)).to_csv(args.out_dir/"observed_inference.csv", index=False)
        np.save(args.out_dir/"permutation_counts.npy", permutations)
        print(pd.DataFrame(core.result_rows(result)).to_string(index=False), flush=True)
    else:
        tasks = []
        for model in args.models:
            settings = [0.] + (args.rhos if model == "iid_sticky" and args.power_replications else [])
            for rho in settings:
                n = args.null_replications if rho == 0 else args.power_replications
                scenario = np.random.SeedSequence([args.seed, int(model == "iid_sticky"), int(round(rho*1000000))])
                tasks.extend((model, rho, index, child.generate_state(4))
                             for index,child in enumerate(scenario.spawn(n)))
        detail_path = args.out_dir/"outer_tests.csv"
        with detail_path.open("x") as handle, mp.get_context("fork").Pool(args.workers) as pool:
            for done, rows in enumerate(pool.imap_unordered(worker, tasks, chunksize=1), 1):
                pd.DataFrame(rows).to_csv(handle, index=False, header=done == 1)
                handle.flush()
                if done % 20 == 0 or done == len(tasks):
                    print(f"completed {done}/{len(tasks)}; elapsed={time.time()-started:.1f}s", flush=True)
        details = pd.read_csv(detail_path).sort_values(["mode", "rho", "replication", "unit", "threshold"])
        details.to_csv(args.out_dir/"outer_tests_sorted.csv", index=False)
        marginal, families = summarize(details, args.alpha)
        marginal.to_csv(args.out_dir/"rejection_rates.csv", index=False)
        families.to_csv(args.out_dir/"family_rejection_rates.csv", index=False)
        print(families.to_string(index=False), flush=True)
    if core.sha256(args.payloads) != input_hash:
        raise RuntimeError("Input hash changed during execution")
    metadata.update(status="complete", elapsed_seconds=time.time()-started)
    metadata["output_sha256"] = {path.name: core.sha256(path) for path in sorted(args.out_dir.iterdir())
                                 if path.is_file() and path.name != "configuration.json"}
    (args.out_dir/"completion.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"complete: {time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
