"""Joint within-window permutation inference for maximal builder runs."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

THRESHOLDS = np.array([2, 3, 4, 5, 6, 8, 10], dtype=np.int64)
UNITS = ("key", "proxy")
P_COLUMNS = ("raw_upper_p", "raw_lower_p", "raw_two_sided_p", "unit_adjusted_p", "family_adjusted_p")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frame(path, window_slots=10000, profile_windows=24):
    frame = pd.read_parquet(path, columns=["slot", "builder_pubkey", "extra_data_tag"])
    if frame.empty or frame.slot.isna().any() or frame.builder_pubkey.isna().any():
        raise ValueError("Empty input or missing slot/public-key values")
    if not np.issubdtype(frame.slot.dtype, np.integer) or (frame.slot < 0).any():
        raise ValueError("Slots must be nonnegative integers")
    if frame.slot.duplicated().any():
        raise ValueError("Duplicate slots in common-support input")
    frame["builder_pubkey"] = frame.builder_pubkey.astype(str).str.casefold()
    if (frame.builder_pubkey.str.len() == 0).any():
        raise ValueError("Empty public keys")
    frame["extra_data_tag"] = frame.extra_data_tag.fillna("").astype(str)
    frame["proxy"] = np.where(frame.extra_data_tag.str.len() > 0,
                              "tag:" + frame.extra_data_tag, "key:" + frame.builder_pubkey)
    available = np.unique(frame.slot.to_numpy() // window_slots)
    if profile_windows and profile_windows < len(available):
        selected = available[np.rint(np.linspace(0, len(available) - 1, profile_windows)).astype(int)]
        frame = frame.loc[np.isin(frame.slot.to_numpy() // window_slots, selected)].copy()
    else:
        selected = available
    return frame.sort_values("slot").reset_index(drop=True), selected.tolist()


def arrays_from_frame(frame, window_slots=10000):
    slots = frame.slot.to_numpy(dtype=np.int64)
    if len(slots) == 0 or np.any(np.diff(slots) <= 0):
        raise ValueError("Input slots must be nonempty, unique, and sorted")
    windows = slots // window_slots
    window_break = np.r_[True, np.diff(windows) != 0]
    breaks = window_break | np.r_[True, np.diff(slots) != 1]
    starts = np.flatnonzero(window_break).astype(np.int64)
    ends = np.r_[starts[1:], len(slots)].astype(np.int64)
    labels = np.column_stack([pd.factorize(frame[column], sort=True)[0]
                              for column in ("builder_pubkey", "proxy")]).astype(np.int32)
    return labels, breaks, starts, ends


@njit(cache=True)
def ordered_counts(labels, order, breaks):
    """Count both identity units in one pass; cap run lengths at ten."""
    histogram = np.zeros((2, 11), dtype=np.int64)
    if len(order) == 0:
        return np.zeros((2, 7), dtype=np.int64)
    lengths = np.ones(2, dtype=np.int64)
    for position in range(1, len(order)):
        for unit in range(2):
            if not breaks[position] and labels[order[position], unit] == labels[order[position-1], unit]:
                lengths[unit] = min(lengths[unit] + 1, 10)
            else:
                histogram[unit, lengths[unit]] += 1
                lengths[unit] = 1
    for unit in range(2):
        histogram[unit, lengths[unit]] += 1
    output = np.zeros((2, 7), dtype=np.int64)
    for unit in range(2):
        for index in range(7):
            for length in range(THRESHOLDS[index], 11):
                output[unit, index] += histogram[unit, length]
    return output


def count_runs(labels, breaks):
    return ordered_counts(labels, np.arange(len(labels), dtype=np.int64), breaks)


def permutation_order(rng, size, starts, ends):
    order = np.arange(size, dtype=np.int64)
    for start, end in zip(starts, ends):
        rng.shuffle(order[start:end])
    return order


def permutation_counts(labels, breaks, starts, ends, repetitions, rng):
    if repetitions < 2:
        raise ValueError("At least two permutations are required")
    result = np.empty((repetitions, 2, 7), dtype=np.int64)
    for index in range(repetitions):
        order = permutation_order(rng, len(labels), starts, ends)
        result[index] = ordered_counts(labels, order, breaks)
    return result


def infer_from_counts(observed, permutations, alpha=0.05):
    """Symmetric studentization and inclusive ranks over B+1 vectors.

    Each coordinate is centered and scaled using the pooled observed and
    permuted vectors. Constant coordinates receive zero standardized scores.
    Descriptive expectations and standard deviations use only permutations.
    """
    observed = np.asarray(observed)
    permutations = np.asarray(permutations)
    if observed.shape != (2, 7) or permutations.ndim != 3 or permutations.shape[1:] != (2, 7):
        raise ValueError("Expected observed (2,7) and permutations (B,2,7)")
    if len(permutations) < 2 or not 0 < alpha < 1:
        raise ValueError("Invalid permutation count or alpha")
    pooled = np.concatenate([observed[None], permutations]).astype(np.float64)
    if not np.isfinite(pooled).all() or (pooled < 0).any():
        raise ValueError("Run counts must be finite and nonnegative")
    center = pooled.mean(axis=0)
    scale = pooled.std(axis=0, ddof=1)
    z = np.divide(pooled - center, scale, out=np.zeros_like(pooled), where=scale > 0)
    absolute = np.abs(z)
    unit_max = absolute.max(axis=2)
    family_max = absolute.max(axis=(1, 2))
    upper = (pooled >= observed).mean(axis=0)
    lower = (pooled <= observed).mean(axis=0)
    unit_p = (unit_max[:, :, None] >= absolute[0][None]).mean(axis=0)
    family_p = (family_max[:, None, None] >= absolute[0][None]).mean(axis=0)
    result = {
        "observed": observed,
        "permutation_mean": permutations.mean(axis=0),
        "permutation_sd": permutations.std(axis=0, ddof=1),
        "pooled_center": center, "pooled_sd": scale,
        "z": z[0], "raw_upper_p": upper, "raw_lower_p": lower,
        "raw_two_sided_p": np.minimum(1., 2 * np.minimum(upper, lower)),
        "unit_adjusted_p": unit_p, "family_adjusted_p": family_p,
        "unit_reject": (unit_p <= alpha).any(axis=1),
        "family_reject": bool((family_p <= alpha).any()),
        "zero_permutation_sd": permutations.std(axis=0, ddof=1) == 0,
        "zero_pooled_sd": scale == 0,
    }
    return result


def complete_test(labels, breaks, starts, ends, repetitions, rng, alpha=0.05):
    observed = count_runs(labels, breaks)
    permutations = permutation_counts(labels, breaks, starts, ends, repetitions, rng)
    return infer_from_counts(observed, permutations, alpha)


def synthetic_labels(labels, breaks, starts, ends, rng, model, rho=0.):
    if model == "conditional":
        if rho != 0:
            raise ValueError("Conditional permutations require rho=0")
        return labels[permutation_order(rng, len(labels), starts, ends)]
    if model != "iid_sticky" or not 0 <= rho <= 1:
        raise ValueError("Unknown synthetic model or invalid rho")
    selected = np.empty(len(labels), dtype=np.int64)
    for start, end in zip(starts, ends):
        selected[start:end] = rng.integers(start, end, size=end-start)
    if rho > 0:
        fresh = rng.random(len(labels)) >= rho
        fresh[breaks] = True
        anchors = np.maximum.accumulate(np.where(fresh, np.arange(len(labels)), 0))
        selected = selected[anchors]
    return labels[selected]


def wilson(successes, total):
    z = 1.959963984540054
    rate = successes / total
    divisor = 1 + z*z/total
    center = (rate + z*z/(2*total))/divisor
    radius = z*math.sqrt(rate*(1-rate)/total+z*z/(4*total*total))/divisor
    return max(0., center-radius), min(1., center+radius)


def result_rows(result, **identity):
    rows = []
    for u, unit in enumerate(UNITS):
        for k, threshold in enumerate(THRESHOLDS):
            row = dict(identity, unit=unit, threshold=int(threshold))
            for field in ("observed", "permutation_mean", "permutation_sd", "pooled_center", "pooled_sd", "z", *P_COLUMNS):
                row[field] = float(result[field][u, k])
            row["zero_permutation_sd"] = bool(result["zero_permutation_sd"][u,k])
            row["zero_pooled_sd"] = bool(result["zero_pooled_sd"][u,k])
            rows.append(row)
    return rows
