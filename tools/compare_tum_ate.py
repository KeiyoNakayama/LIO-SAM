#!/usr/bin/env python3
"""Compute ATE between two TUM trajectories with nearest-time association."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def read_tum(path: Path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                raise ValueError(f"bad TUM row in {path}: {line}")
            rows.append((float(parts[0]), np.array([float(parts[1]), float(parts[2]), float(parts[3])])))
    if not rows:
        raise ValueError(f"no poses in {path}")
    return rows


def associate(reference, estimate, max_dt):
    matches = []
    est_i = 0
    for ref_t, ref_p in reference:
        while est_i + 1 < len(estimate) and estimate[est_i + 1][0] < ref_t:
            est_i += 1
        candidates = [estimate[est_i]]
        if est_i + 1 < len(estimate):
            candidates.append(estimate[est_i + 1])
        est_t, est_p = min(candidates, key=lambda row: abs(row[0] - ref_t))
        if abs(est_t - ref_t) <= max_dt:
            matches.append((ref_t, ref_p, est_t, est_p))
    if len(matches) < 2:
        raise ValueError("not enough synchronized poses; increase --max-dt or check timestamps")
    return matches


def umeyama(src, dst, with_scale=False):
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = dst_c.T @ src_c / src.shape[0]
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1, -1] = -1
    r = u @ s @ vt
    scale = 1.0
    if with_scale:
        scale = np.trace(np.diag(d) @ s) / np.mean(np.sum(src_c * src_c, axis=1))
    t = dst_mean - scale * r @ src_mean
    return scale, r, t


def stats(errors):
    return {
        "count": len(errors),
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95)),
        "max": float(np.max(errors)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("groundtruth", type=Path)
    parser.add_argument("estimate", type=Path)
    parser.add_argument("--max-dt", type=float, default=0.05)
    parser.add_argument("--align", action="store_true", help="rigidly align estimate to ground truth")
    parser.add_argument("--scale", action="store_true", help="allow scale correction during alignment")
    parser.add_argument("--csv", type=Path, help="write per-pose errors")
    args = parser.parse_args()

    gt = read_tum(args.groundtruth)
    est = read_tum(args.estimate)
    matches = associate(gt, est, args.max_dt)
    gt_xyz = np.array([m[1] for m in matches])
    est_xyz = np.array([m[3] for m in matches])

    if args.align:
        scale, rot, trans = umeyama(est_xyz, gt_xyz, args.scale)
        est_xyz = (scale * (rot @ est_xyz.T)).T + trans

    errors = np.linalg.norm(gt_xyz - est_xyz, axis=1)
    result = stats(errors)
    for key, value in result.items():
        if key == "count":
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.6f} m")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t_groundtruth", "t_estimate", "error_m"])
            for match, error in zip(matches, errors):
                writer.writerow([f"{match[0]:.9f}", f"{match[2]:.9f}", f"{error:.9f}"])


if __name__ == "__main__":
    main()
