#!/usr/bin/env python3
"""Collect the t=0 Z4c constraint norms, i.e. the constraints after the
finite-particle matter source has been deposited.

The pgen never re-solves the Einstein constraints against the realized
particulate T_munu: the metric is the exact continuum mean-field solution, so
the t=0 constraint violation is a direct measure of how far each sampler's
deposited source sits from the continuum source that metric assumes.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


def read_hst(path):
    labels, rows = None, []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            if "=" in line and "[" in line:
                labels = re.findall(r"\[\d+\]=(\S+)", line)
            continue
        if line.strip():
            rows.append([float(v) for v in line.split()])
    if not rows or not labels:
        return None
    arr = np.array(rows)
    return {lab: arr[:, i] for i, lab in enumerate(labels[:arr.shape[1]])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--stage", default="live_q6p1")
    args = ap.parse_args()

    manifest = json.loads((args.root / "inputs" / "input_manifest.json").read_text())
    rows = []
    for case in [c for c in manifest if c.get("stage") == args.stage]:
        run = args.root / "runs" / case["name"]
        z = run / f"{case['name']}.z4c.user.hst"
        u = run / f"{case['name']}.user.hst"
        if not z.exists():
            continue
        zd, ud = read_hst(z), (read_hst(u) if u.exists() else None)
        if zd is None:
            continue
        rec = {"name": case["name"], "sampler": case["sampler"],
               "seed": case["seed"], "model": case["model"], "q": case["q"],
               "t0": float(zd["time"][0])}
        for lab, key in (("H-norm2", "ham"), ("M-norm2", "mom"),
                         ("C-norm2", "cnorm"), ("Z-norm2", "znorm"),
                         ("Mx-norm2", "momx"), ("My-norm2", "momy"),
                         ("Mz-norm2", "momz")):
            if lab in zd:
                rec[f"{key}_t0"] = float(zd[lab][0])
        if ud is not None:
            for lab, key in (("M0_alive", "M0"), ("E_part", "E_part"),
                             ("L_scalar", "L_scalar"), ("Jpart_x", "Jx"),
                             ("Jpart_y", "Jy"), ("Jpart_z", "Jz"),
                             ("rho_max", "rho_max"), ("rho_ctr", "rho_ctr"),
                             ("N_alive", "N_alive")):
                if lab in ud:
                    rec[f"{key}_t0"] = float(ud[lab][0])
            if all(f"J{a}_t0" in rec for a in "xyz"):
                rec["J_norm_t0"] = float(np.linalg.norm(
                    [rec["Jx_t0"], rec["Jy_t0"], rec["Jz_t0"]]))
        rows.append(rec)

    if not rows:
        raise SystemExit("no history files found yet")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)

    print(f"{'case':44s} {'H(t=0)':>12s} {'M(t=0)':>12s} {'|J|(t=0)':>12s} "
          f"{'M0':>12s}")
    for r in sorted(rows, key=lambda x: (x["sampler"], x["seed"])):
        print(f"{r['name']:44s} {r.get('ham_t0', float('nan')):12.5e} "
              f"{r.get('mom_t0', float('nan')):12.5e} "
              f"{r.get('J_norm_t0', float('nan')):12.5e} "
              f"{r.get('M0_t0', float('nan')):12.8f}")
    print(f"\nwrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
