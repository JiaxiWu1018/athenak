#!/usr/bin/env python3
"""Subsample particle positions from every pvtk frame into one small npz for movies.

The pvtk dumps are 81 MB/frame and there are a few hundred of them, so the raw dumps stay
on the compute side; this reduces a run to a few tens of MB by keeping a FIXED subset of
particle IDs and, for each, only its position and its initial radial group.

Subset choice: a fixed STRIDE over pair index (tag // 2), not a random draw. Because the
stratum index equals the pair index and the radial quantile is monotone in it, a stride in
pair index is a uniform sample in ENCLOSED REST MASS -- it spans the core, the half-mass
region and the halo in the correct proportion -- and it is reproducible from the stride
alone. Both members of each selected pair are kept so the subset carries zero net
momentum, exactly like the full set.

This is a VISUALISATION subset only. The scientific reductions (reduce_run.py) use every
particle.

Colouring: cohort = (tag // 2) * ncohort // npair, the particle's initial radial group.
Particles keep their colour for all time, so any colour blending on screen is physical
shell mixing, not a rendering artefact.

Usage: extract_movie_frames.py --rundir DIR --out FILE.npz [--nsub 200000] [--ncohort 32]
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pvtk_reader import read_pvtk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nsub", type=int, default=200000)
    ap.add_argument("--ncohort", type=int, default=32)
    ap.add_argument("--npair", type=int, default=1056768)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.rundir, "pvtk", "*.part.vtk")))
    if not files:
        sys.exit("no pvtk frames under %s/pvtk" % a.rundir)
    print("%d frames" % len(files), flush=True)

    npair_keep = max(1, a.nsub//2)
    stride = max(1, a.npair//npair_keep)
    pairs = np.arange(0, a.npair, stride, dtype=np.int64)
    keep_tags = np.sort(np.concatenate([2*pairs, 2*pairs + 1]))
    coh = ((keep_tags//2)*a.ncohort)//a.npair
    print("subset: stride %d in pair index -> %d pairs, %d particles, cohorts %d..%d"
          % (stride, pairs.size, keep_tags.size, coh.min(), coh.max()), flush=True)

    times, X, Y, Z = [], [], [], []
    for k, f in enumerate(files):
        d = read_pvtk(f, {"pos", "ptag"})
        pos = np.asarray(d["pos"], dtype=np.float32)
        tag = np.asarray(d["ptag"]).astype(np.int64)
        # select on TAG, not row index: particle ORDER changes between frames after
        # migration, so only the tag keeps the same physical particles across the movie
        sel = np.isin(tag, keep_tags)
        p, t = pos[sel], tag[sel]
        order = np.argsort(t)                    # canonical order every frame
        p, t = p[order], t[order]
        if t.size != keep_tags.size or not np.array_equal(t, keep_tags):
            print("  WARNING frame %d: kept %d of %d requested tags" % (k, t.size,
                  keep_tags.size), flush=True)
        times.append(float(d.get("time", np.nan)))
        X.append(p[:, 0]); Y.append(p[:, 1]); Z.append(p[:, 2])
        if k % 20 == 0 or k == len(files) - 1:
            print("  frame %4d  t=%12.5f  kept %d" % (k, times[-1], p.shape[0]),
                  flush=True)
        del d, pos, tag

    np.savez_compressed(a.out, times=np.array(times),
                        x=np.array(X, dtype=np.float32), y=np.array(Y, dtype=np.float32),
                        z=np.array(Z, dtype=np.float32),
                        cohort=coh.astype(np.int16), tags=keep_tags.astype(np.int64),
                        ncohort=a.ncohort, npair=a.npair, stride=stride)
    print("wrote %s (%.1f MB)" % (a.out, os.path.getsize(a.out)/1e6), flush=True)


if __name__ == "__main__":
    main()
