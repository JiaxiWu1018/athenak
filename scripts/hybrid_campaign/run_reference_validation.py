#!/usr/bin/env python3
"""Validate the Python reference sampler against every executable t=0 dump.

The pre-evolution multipole and CDF sweep is computed from the Python mirror of
the pgen rather than from the solver, so that many samplers and seeds can be
swept cheaply.  That is only legitimate if the mirror reproduces what AthenaK
actually placed.  This compares them particle-by-particle for every sampler,
using the fixed-background smoke dumps at t=0.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_sampler_reference as ref
from validate_reference_sampler import read_particle_vtk

SAMPLERS = list(ref.SAMPLERS)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1985)
    ap.add_argument("--radius-over-mass", type=float, default=6.1)
    ap.add_argument("--nradial", type=int, default=32)
    ap.add_argument("--nangular", type=int, default=240)
    ap.add_argument("--tolerance", type=float, default=5.0e-6)
    args = ap.parse_args()

    out = args.root / "analysis" / "reference_validation"
    out.mkdir(parents=True, exist_ok=True)
    results, ok_all = [], True
    for s in SAMPLERS:
        dumps = sorted((args.root / "runs" / f"fixed_{s}_s{args.seed}" / "pvtk")
                       .glob("*.part.vtk"))
        if not dumps:
            print(f"{s:24s} NO DUMP")
            ok_all = False
            continue
        time, pos, vel, tag, mass = read_particle_vtk(dumps[0])
        r = ref.realize(s, args.seed, radius_over_mass=args.radius_over_mass,
                        nradial=args.nradial, nangular=args.nangular,
                        octahedral=True)
        tags_match = bool(np.array_equal(tag, r["tag"]))
        pscale = max(float(np.abs(pos).max()), 1.0)
        vscale = max(float(np.abs(vel).max()), 1e-30)
        dpos = float(np.abs(pos - r["pos"]).max() / pscale)
        dvel = float(np.abs(vel - r["vel"]).max() / vscale)
        dmass = float(abs(mass.mean() - r["particle_mass"]) / r["particle_mass"])
        ok = bool(tags_match and dpos < args.tolerance
                  and dvel < args.tolerance and dmass < args.tolerance)
        ok_all &= ok
        rec = {"sampler": s, "seed": args.seed, "time": time,
               "npart": int(len(tag)), "tags_match": tags_match,
               "max_rel_position_error": dpos, "max_rel_velocity_error": dvel,
               "rel_particle_mass_error": dmass,
               "tolerance": args.tolerance, "agrees": ok}
        results.append(rec)
        (out / f"{s}.json").write_text(json.dumps(rec, indent=2))
        print("{:24s} N={:6d} t={:.1f} dpos={:.2e} dvel={:.2e} dmass={:.2e} "
              "tags={} -> {}".format(s, len(tag), time, dpos, dvel, dmass,
                                     tags_match, "AGREES" if ok else "DIFFERS"))

    verdict = {"all_agree": ok_all, "tolerance": args.tolerance,
               "note": "float32 particle dumps; tolerance is a few ulp of "
                       "single precision", "results": results}
    (out / "reference_validation.json").write_text(json.dumps(verdict, indent=2))
    print(f"\nall_agree = {ok_all}")
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
