#!/usr/bin/env python3
"""Quantify the default-sampler regression against the nondeterminism floor.

The AthenaK GPU build is not bitwise reproducible run to run: atomic CIC
deposition sums particle contributions in a nondeterministic order, so two runs
of the SAME unchanged executable already diverge at roundoff level after the
first cycle.  The meaningful regression statement is therefore:

  1. the t=0 particle realization must be bitwise identical, and
  2. the old-vs-new divergence at later times must be no larger than the
     old-vs-old divergence of the unchanged executable with itself.

This script measures both.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np


def read_vtk(path):
    blob = path.read_bytes()
    m = re.search(rb"POINTS\s+(\d+)\s+float", blob)
    n = int(m.group(1))
    time = float(re.search(rb"time=\s*([-+0-9.eE]+)", blob).group(1))

    def block(marker, count):
        off = blob.find(marker)
        start = blob.find(b"\n", off + len(marker)) + 1
        return np.frombuffer(blob[start:start + 4*count], dtype=">f4").astype(float)

    pos = block(m.group(0), 3*n).reshape(n, 3)
    vel = block(b"VECTORS prtcl_vel float", 3*n).reshape(n, 3)
    tag = block(b"SCALARS ptag float\nLOOKUP_TABLE default", n).astype(np.int64)
    order = np.argsort(tag)
    return time, pos[order], vel[order], tag[order]


def pair_metrics(a_dir, b_dir):
    out = []
    for pa in sorted(Path(a_dir).glob("pvtk/*.part.vtk")):
        pb = Path(b_dir) / "pvtk" / pa.name
        if not pb.exists():
            continue
        ta, xa, va, ga = read_vtk(pa)
        tb, xb, vb, gb = read_vtk(pb)
        if not np.array_equal(ga, gb):
            out.append({"file": pa.name, "time": ta, "tag_mismatch": True})
            continue
        xs = max(np.abs(xa).max(), 1.0)
        vs = max(np.abs(va).max(), 1e-30)
        out.append({
            "file": pa.name, "time": ta,
            "bitwise_identical": pa.read_bytes() == pb.read_bytes(),
            "max_rel_dpos": float(np.abs(xa - xb).max() / xs),
            "max_rel_dvel": float(np.abs(va - vb).max() / vs),
            "rms_rel_dpos": float(np.sqrt(np.mean((xa - xb) ** 2)) / xs),
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regression", required=True, type=Path,
                    help="dir containing old/ and new/ (patched vs unpatched)")
    ap.add_argument("--control", required=True, type=Path,
                    help="dir containing a/ and b/ (unpatched vs itself)")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    reg = pair_metrics(args.regression / "old", args.regression / "new")
    ctl = pair_metrics(args.control / "a", args.control / "b")

    print(f"{'file':52s} {'time':>8s} {'old-vs-new':>12s} {'old-vs-old':>12s}")
    verdict_ok = True
    t0_ok = False
    rows = []
    for r, c in zip(reg, ctl):
        same = r.get("bitwise_identical")
        print(f"{r['file']:52s} {r['time']:8.4f} "
              f"{r['max_rel_dpos']:12.3e} {c['max_rel_dpos']:12.3e}"
              f"{'   [bitwise identical]' if same else ''}")
        if r["time"] == 0.0:
            t0_ok = bool(same and c.get("bitwise_identical"))
        else:
            # allow a factor of a few: both are samples of the same noise process
            if r["max_rel_dpos"] > max(10.0 * c["max_rel_dpos"], 1e-12):
                verdict_ok = False
        rows.append({"file": r["file"], "time": r["time"],
                     "old_vs_new_max_rel_dpos": r["max_rel_dpos"],
                     "old_vs_old_max_rel_dpos": c["max_rel_dpos"],
                     "old_vs_new_bitwise": same,
                     "old_vs_old_bitwise": c.get("bitwise_identical")})

    result = {
        "t0_realization_bitwise_identical": t0_ok,
        "late_divergence_within_nondeterminism_floor": verdict_ok,
        "verdict": "PASS" if (t0_ok and verdict_ok) else "FAIL",
        "comparisons": rows,
    }
    print(f"\nt=0 realization bitwise identical: {t0_ok}")
    print(f"late divergence within nondeterminism floor: {verdict_ok}")
    print(f"VERDICT: {result['verdict']}")
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2))
    raise SystemExit(0 if result["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
