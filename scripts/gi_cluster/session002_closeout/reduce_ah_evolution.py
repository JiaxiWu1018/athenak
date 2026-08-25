#!/usr/bin/env python3
"""Create auditable apparent-horizon evolution tables for GI Session 002.

Only shape-file blocks are genuine FastFlow convergences.  Each block is reconstructed
with AthenaK's Wigner-d harmonic convention and passed through the same independent
geometric, resolution, expansion, and mass gates as ``analyze_ah.py``.  Summary-file
rows without a shape block never enter the evolution table.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import hashlib
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_ah as AH  # noqa: E402


AUDIT_FIELDS = [
    "shape_index", "time", "summary_time", "summary_time_offset", "m_christodoulou",
    "m_irr", "area", "spin", "center_time", "center_x", "center_y", "center_z",
    "r_mean", "r_min_summary", "r_min_surface", "r_max_surface", "r_min_cells",
    "r_max_cells", "non_sphericity", "expansion_residual", "lmax", "passes_gates",
    "gate_failures", "pass_sequence", "in_longest_persistent_sequence",
    "after_persistent_validation",
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_one(run_dir: str, pattern: str) -> str | None:
    out = os.path.join(run_dir, "out")
    for folder in [run_dir, out, *glob.glob(os.path.join(out, "*"))]:
        paths = sorted(glob.glob(os.path.join(folder, pattern)))
        if paths:
            return paths[0]
    return None


def pass_sequences(records: list[dict]) -> tuple[int, int, list[tuple[int, int]]]:
    sequences: list[tuple[int, int]] = []
    start = None
    for i, record in enumerate(records):
        if record["passes_gates"] and start is None:
            start = i
        if start is not None and (not record["passes_gates"] or i == len(records) - 1):
            end = i if record["passes_gates"] and i == len(records) - 1 else i - 1
            sequences.append((start, end))
            start = None
    if not sequences:
        return -1, -1, []
    longest = max(sequences, key=lambda ab: ab[1] - ab[0] + 1)
    return longest[0], longest[1], sequences


def report_reconciliation(report_path: str | None, event: dict) -> dict:
    if not report_path or not os.path.exists(report_path):
        return {"report_path": report_path, "available": False}
    text = open(report_path, errors="replace").read()
    patterns = {
        "genuine_converged": r"GENUINE converged finds \(shape-file blocks\):\s*(\d+)",
        "passing": r"finds passing every gate:\s*(\d+) of (\d+)",
        "longest": r"longest run of consecutive passing finds:\s*(\d+)",
        "first": r"first validated find at t =\s*([-+0-9.eE]+)",
        "persistent": r"first of a persistent run at t =\s*([-+0-9.eE]+)",
    }
    parsed = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            parsed[key] = [float(x) if any(c in x for c in ".eE") else int(x)
                           for x in match.groups()]
    checks = {
        "genuine_converged": parsed.get("genuine_converged", [None])[0]
        == event["genuine_converged_shape_blocks"],
        "passing": parsed.get("passing", [None])[0] == event["passing_all_gates"],
        "passing_denominator": (len(parsed.get("passing", [])) > 1
                                and parsed["passing"][1]
                                == event["genuine_converged_shape_blocks"]),
        "longest": parsed.get("longest", [None])[0]
        == event["longest_passing_sequence"],
        "first": ("first" in parsed and math.isclose(parsed["first"][0],
                  event["first_gate_passing_time"], abs_tol=5e-6)),
        "persistent": ("persistent" in parsed and math.isclose(parsed["persistent"][0],
                       event["persistent_sequence_start_time"], abs_tol=5e-6)),
    }
    return {
        "report_path": os.path.abspath(report_path),
        "available": True,
        "parsed": parsed,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def evaluate_run(run_dir: str, dx: float, rmax_gate: float, mass_adm: float,
                 hmean_rel: float, persist: int) -> tuple[list[dict], list[dict], dict]:
    fsum = find_one(run_dir, "*.horizon_summary_0.txt")
    fshape = find_one(run_dir, "*.horizon_shape_0.txt")
    ftracker = find_one(run_dir, "*.co_0.txt")
    if not fsum or not fshape:
        raise FileNotFoundError(f"{run_dir}: missing summary or shape file")
    summary = AH.read_summary(fsum)
    shapes = AH.read_shape(fshape)
    tracker = AH.read_tracker(ftracker) if ftracker else np.zeros((0, 4))

    theta = np.linspace(1e-6, math.pi - 1e-6, 41)
    phi = np.linspace(0.0, 2 * math.pi, 80, endpoint=False)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    records: list[dict] = []
    for index, (_, time, coef) in enumerate(shapes):
        lmax = next((level for level in range(33)
                     if AH.shape_ncoef(level) == len(coef)), None)
        if lmax is None:
            raise ValueError(f"cannot infer lmax at t={time} from {len(coef)} coefficients")
        k = int(np.argmin(np.abs(summary[:, 1] - time)))
        row = summary[k]
        _, summary_time, mchr, _, _, _, spin, area, _, hmean, rmean, rmin_summary = row
        radius = AH.surface_radius(coef, lmax, th, ph)
        rmin = float(np.nanmin(radius))
        rmax = float(np.nanmax(radius))
        mirr = math.sqrt(area / (16 * math.pi)) if area > 0 else math.nan
        hrel = abs(hmean) / area if area > 0 else math.inf
        failures = []
        if not (rmin > 2 * dx):
            failures.append("MINRAD_LT_2DX")
        if not (rmin > 0):
            failures.append("NEGATIVE_RADIUS")
        if not (rmax <= rmax_gate):
            failures.append("OFF_FINE_REGION")
        if not (hrel < hmean_rel):
            failures.append("EXPANSION")
        if not (0 < mirr <= mass_adm):
            failures.append("MASS")
        if not np.isfinite([mchr, area, rmean, rmin_summary, rmin, rmax]).all():
            failures.append("NONFINITE")
        if len(tracker):
            kt = int(np.argmin(np.abs(tracker[:, 0] - time)))
            center_time, cx, cy, cz = tracker[kt]
        else:
            center_time, cx, cy, cz = [math.nan] * 4
        records.append({
            "shape_index": index,
            "time": float(time),
            "summary_time": float(summary_time),
            "summary_time_offset": float(summary_time - time),
            "m_christodoulou": float(mchr),
            "m_irr": float(mirr),
            "area": float(area),
            "spin": float(spin),
            "center_time": float(center_time),
            "center_x": float(cx), "center_y": float(cy), "center_z": float(cz),
            "r_mean": float(rmean),
            "r_min_summary": float(rmin_summary),
            "r_min_surface": rmin,
            "r_max_surface": rmax,
            "r_min_cells": rmin / dx,
            "r_max_cells": rmax / dx,
            "non_sphericity": (rmax - rmin) / rmean,
            "expansion_residual": hrel,
            "lmax": int(lmax),
            "passes_gates": not failures,
            "gate_failures": ";".join(failures),
            "pass_sequence": -1,
            "in_longest_persistent_sequence": False,
            "after_persistent_validation": False,
            "coef": coef,
        })

    longest_start, longest_end, sequences = pass_sequences(records)
    qualifying = [(start, end) for start, end in sequences
                  if end - start + 1 >= persist]
    if not qualifying:
        raise ValueError(f"{run_dir}: no sequence satisfies persistence={persist}")
    first_persistent_start, first_persistent_end = qualifying[0]
    for sequence_number, (start, end) in enumerate(sequences):
        for i in range(start, end + 1):
            records[i]["pass_sequence"] = sequence_number
    if longest_start >= 0:
        for i in range(longest_start, longest_end + 1):
            records[i]["in_longest_persistent_sequence"] = True
    for i in range(first_persistent_start, len(records)):
        records[i]["after_persistent_validation"] = True

    good = [record for record in records if record["passes_gates"]]
    if not good or longest_start < 0 or longest_end - longest_start + 1 < persist:
        raise ValueError(f"{run_dir}: no persistent independently validated horizon")
    post_persistent_good = [record for record in records[first_persistent_start:]
                            if record["passes_gates"]]
    comfortable = next((record for record in post_persistent_good
                        if record["r_min_cells"] >= 8.0), None)
    event = {
        "run": os.path.basename(os.path.normpath(run_dir)),
        "dx_fine": dx,
        "genuine_converged_shape_blocks": len(records),
        "passing_all_gates": len(good),
        "first_fastflow_candidate_time": records[0]["time"],
        "first_gate_passing_time": good[0]["time"],
        "first_gate_passing_m_irr": good[0]["m_irr"],
        "persistent_sequence_start_time": records[first_persistent_start]["time"],
        "first_persistent_sequence_end_time": records[first_persistent_end]["time"],
        "first_persistent_sequence_length": first_persistent_end - first_persistent_start + 1,
        "longest_sequence_start_time": records[longest_start]["time"],
        "longest_sequence_end_time": records[longest_end]["time"],
        "longest_passing_sequence": longest_end - longest_start + 1,
        "comfortably_resolved_rmin_ge_8dx_time": (
            comfortable["time"] if comfortable else None
        ),
        "final_validated_time": good[-1]["time"],
        "final_m_irr": good[-1]["m_irr"],
        "final_r_min_cells": good[-1]["r_min_cells"],
        "final_non_sphericity": good[-1]["non_sphericity"],
        "final_expansion_residual": good[-1]["expansion_residual"],
        "final_spin": good[-1]["spin"],
        "files": {
            "summary": {"path": os.path.abspath(fsum), "sha256": sha256_file(fsum)},
            "shape": {"path": os.path.abspath(fshape), "sha256": sha256_file(fshape)},
            "tracker": ({"path": os.path.abspath(ftracker), "sha256": sha256_file(ftracker)}
                        if ftracker else None),
        },
        "max_abs_summary_time_offset": max(abs(r["summary_time_offset"]) for r in records),
        "shape_block_count_matches_verbose_report": None,
    }
    report_path = os.path.join(run_dir, "analysis", "ah_report.txt")
    reconciliation = report_reconciliation(report_path, event)
    event["validator_report_reconciliation"] = reconciliation
    if reconciliation.get("available") and not reconciliation.get("all_checks_pass"):
        raise ValueError(f"{run_dir}: reduced data disagree with archived validator report")

    clean_records = [{key: value for key, value in record.items() if key != "coef"}
                     for record in records]
    clean_good = [record for record in clean_records if record["passes_gates"]]
    return clean_records, clean_good, event


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_root")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--mass-adm", type=float, default=0.73)
    parser.add_argument("--rmax-gate", type=float, default=0.40)
    parser.add_argument("--hmean-rel", type=float, default=0.20)
    parser.add_argument("--persist", type=int, default=3)
    args = parser.parse_args()

    root = os.path.abspath(args.session_root)
    outdir = os.path.abspath(args.outdir or os.path.join(root, "reduced_data"))
    os.makedirs(outdir, exist_ok=True)
    specs = [
        ("g2_L128", os.path.join(root, "evidence", "runs", "g2_L128_383659"), 1 / 128),
        ("g2_L64", os.path.join(root, "evidence", "runs", "g2_L64_383658"), 1 / 64),
    ]
    events = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gate_definition": {
            "genuine_find": "one actual FastFlow horizon_shape block",
            "min_surface_radius": "r_min > 2 dx_fine",
            "positive_radius": "r_min > 0",
            "on_fine_region": f"r_max <= {args.rmax_gate}",
            "expansion": f"abs(int H dA)/area < {args.hmean_rel}",
            "mass": f"0 < M_irr <= {args.mass_adm}",
            "persistence": f"at least {args.persist} consecutive passing shape blocks",
        },
        "runs": {},
    }
    for label, run_dir, dx in specs:
        audit, valid, event = evaluate_run(
            run_dir, dx, args.rmax_gate, args.mass_adm, args.hmean_rel, args.persist
        )
        audit_path = os.path.join(outdir, f"ah_surface_gate_audit_{label}.csv")
        valid_path = os.path.join(outdir, f"ah_evolution_{label}.csv")
        write_csv(audit_path, audit)
        write_csv(valid_path, valid)
        event["outputs"] = {"audit_csv": audit_path, "validated_csv": valid_path}
        events["runs"][label] = event
        print(f"{label}: {len(audit)} shape blocks, {len(valid)} pass; "
              f"persistent t={event['persistent_sequence_start_time']:.5f}; "
              f"final M_irr={event['final_m_irr']:.6f}")

    events_path = os.path.join(outdir, "ah_validation_events.json")
    with open(events_path, "w") as stream:
        json.dump(events, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"wrote {events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
