"""Particle stress-energy feedback (Tmunu deposition) regression tier (NRPIC Stage 4c).

These tests need a build with the part_tmunu_test user pgen (NOT in the built-in
dispatch). Dedicated invocation -- the space inside the quoted flag matters
(argparse rejects a bare "-DPROBLEM=..." value):

    cd tst
    python3 run_test_suite.py --mpicpu "-D PROBLEM=part_tmunu_test" \\
        --test test_suite/particles/test_part_tmunu_mpicpu.py

In the stock full-suite run this module SKIPs (the build has
PROBLEM=built_in_pgens). Never run the FULL test suite with -DPROBLEM set.

Standalone dev loop (run_test_suite.py deletes tst/build at the end of each run):
    cmake -B tst/build -DAthena_ENABLE_MPI=ON -DPROBLEM=part_tmunu_test
    make -C tst/build -j10
    ln -s ../../inputs tst/build/src/inputs
    (cd tst/build/src && python3 -m pytest ../../test_suite/particles -v)

The Stage-4c deposit folds every particle's own-block cloud and its cross-block
ghost images into ONE canonically-sorted (target_m, tag, off_code) stream, so the
per-cell Tmunu is bitwise identical for every rank decomposition (CPU/serial-host).
Cross-rank images travel on the particle communicator and merge into that same sorted
stream. Each run has <particles> debug=1, so the in-code float64 conservation identity
Sum_cells E sqrt(gamma) dV == Sum_p m W f_p (reduced across ranks) is fatal -- a
nonzero exit fails the test before any artifact assert.
"""
import filecmp
import glob
import os

import test_suite.particles.part_helpers as ph

ph.require_problem("part_tmunu_test")

FLAT = "inputs/part_tmunu_flat_pytest.athinput"
SMR = "inputs/part_tmunu_smr_pytest.athinput"
W2 = ["problem/pux=1.0", "problem/puy=1.0", "problem/puz=1.0"]  # u_i=(1,1,1): W=2 flat


def _tmunu_dumps(d):
    """Sorted Tmunu bin dumps (gathered, gid-ordered -> a single file per dump whose
    bytes are decomposition-independent: the header carries no rank-count and the
    payload is ordered by global block id)."""
    files = sorted(glob.glob(os.path.join(d, "bin", "*.tmunu.*.bin")))
    assert files, f"no tmunu bin dumps in {d}"
    return files


def _assert_tmunu_bitwise(da, db):
    a, b = _tmunu_dumps(da), _tmunu_dumps(db)
    assert len(a) == len(b), f"dump count differs: {len(a)} ({da}) vs {len(b)} ({db})"
    for fa, fb in zip(a, b):
        assert filecmp.cmp(fa, fb, shallow=False), (
            f"Tmunu not bitwise identical: {os.path.basename(fa)} ({da} vs {db})")


def test_flat_sweep_np_invariance():
    """T1: the 11^3 W=2 sweep lattice (faces/edges/corners + periodic wrap across the
    2x2x2 decomposition) deposited once as a seed. Per-cell Tmunu bitwise identical
    across np {1,2,4}; the in-code conservation identity gates correctness."""
    nps = ph.rank_counts((1, 2, 4))
    dirs = {n: f"tmu_t1_np{n}" for n in nps}
    try:
        for n in nps:
            ph.rmdirs(dirs[n])
            ph.run_args(["-i", FLAT, "-d", dirs[n], "problem/mode=sweep"] + W2,
                        threads=n)
        for n in nps[1:]:
            _assert_tmunu_bitwise(dirs[nps[0]], dirs[n])
    finally:
        ph.rmdirs(*dirs.values())


def test_smr_corner_cross_rank_images():
    """T1-SMR: a particle at the shared corner (0.5,0.5,0.5) of the 8 level-1 fine
    blocks generates 7 same-level images, some crossing ranks at np>1 (matter stays
    on the finest level -- no coarse-fine interface touch). Tmunu bitwise np {1,2,4}."""
    nps = ph.rank_counts((1, 2, 4))
    dirs = {n: f"tmu_t2_np{n}" for n in nps}
    pos = ["problem/mode=single", "problem/px=0.5", "problem/py=0.5", "problem/pz=0.5"]
    try:
        for n in nps:
            ph.rmdirs(dirs[n])
            ph.run_args(["-i", SMR, "-d", dirs[n]] + pos + W2, threads=n)
        for n in nps[1:]:
            _assert_tmunu_bitwise(dirs[nps[0]], dirs[n])
    finally:
        ph.rmdirs(*dirs.values())


def test_periodic_wrap_cross_rank():
    """T1-wrap: the sweep anchored at the domain corner (-0.5) wraps through the
    periodic boundary -- at np>1 the wrap neighbor is on another rank, so the images
    travel cross-rank with no wrapped-position arithmetic. Tmunu bitwise np {1,2}."""
    nps = ph.rank_counts((1, 2))
    dirs = {n: f"tmu_t3_np{n}" for n in nps}
    corner = ["problem/mode=sweep", "problem/sweep_x1=-0.5", "problem/sweep_x2=-0.5",
              "problem/sweep_x3=-0.5"]
    try:
        for n in nps:
            ph.rmdirs(dirs[n])
            ph.run_args(["-i", FLAT, "-d", dirs[n]] + corner + W2, threads=n)
        for n in nps[1:]:
            _assert_tmunu_bitwise(dirs[nps[0]], dirs[n])
    finally:
        ph.rmdirs(*dirs.values())
