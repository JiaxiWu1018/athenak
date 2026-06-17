"""Particle redistribution through dynamic-AMR regrids (NRPIC Stage 5a).

Same build/invocation as the Stage-3d migration tier (the part_crossing user pgen):

    cd tst
    python3 run_test_suite.py --mpicpu "-D PROBLEM=part_crossing" \\
        --test test_suite/particles/test_part_amr_mpicpu.py

A scripted moving box (problem/amr=moving_box) drives a deterministic, particle-
independent stream of refine/derefine events while drift particles stream through. Every
run has <particles> debug=1, so the in-code post-regrid validator (containment, PGID
range, two-sided conservation ledger) is fatal -- a nonzero exit fails the test before any
artifact assert. The drift pusher is purely kinematic (AMR moves blocks, not particles),
so the analytic drift law x0 + v*t (mod L) must still hold THROUGH the regrids, and the
per-tag positions must be bitwise identical across rank counts -- the Stage-3/4
determinism contract extended across regrid events.
"""
import glob
import os

import pytest

import test_suite.particles.part_helpers as ph

ph.require_problem("part_crossing")

INPUT = "inputs/part_crossing_amr_pytest.athinput"


def test_amr_lattice_across_ranks():
    """5a-T1: drift lattice under a scripted moving-box AMR, np {1,2,4}. Particles
    survive every regrid (debug=1 validator + ledger fatal on violation), the analytic
    drift law still holds through the regrids, and per-tag positions are bitwise
    identical across rank counts (determinism contract through regrid events)."""
    nps = ph.rank_counts((1, 2, 4))
    dirs = {n: f"prt_amr_t1_np{n}" for n in nps}
    try:
        for n in nps:
            ph.rmdirs(dirs[n])
            ph.run_args(["-i", INPUT, "-d", dirs[n]], threads=n)
            ph.assert_analytic_drift(dirs[n])
        for n in nps[1:]:
            ph.assert_last_dumps_bitwise(dirs[nps[0]], dirs[n])
    finally:
        ph.rmdirs(*dirs.values())


def test_amr_restart_straddling_regrid():
    """5a-T2: rst written on np1 mid-run (after several regrids), restarted on np2 to a
    later cycle, vs an uninterrupted np2 reference. Final dumps bitwise per tag with
    tag-SET equality -- the grid-agnostic bbox re-placement (FindContainingMeshBlock)
    plus the regrid remap survive a decomposition change straddling regrid events. First
    post-restart dump compared time-matched (restart shifts the file-number ladder)."""
    if (os.cpu_count() or 1) < 2:
        pytest.skip("needs >= 2 cores")
    chain, ref = "prt_amr_t2_chain", "prt_amr_t2_ref"
    try:
        ph.rmdirs(chain, ref)
        ph.run_args(["-i", INPUT, "-d", chain, "time/nlim=15"], threads=1)
        seg1_dumps = set(ph.list_dumps(chain))
        rsts = sorted(glob.glob(os.path.join(chain, "rst", "*.rst")))
        assert rsts, "stage 1 wrote no restart file"
        ph.run_args(["-r", rsts[-1], "-d", chain, "time/nlim=30"], threads=2)
        ph.run_args(["-i", INPUT, "-d", ref, "time/nlim=30"], threads=2)
        ph.assert_last_dumps_bitwise(chain, ref)
        new = [f for f in ph.list_dumps(chain) if f not in seg1_dumps]
        assert new, "restarted segment produced no dumps"
        ph.assert_bitwise(new[0],
                          ph.pick_dump(ref, "match", ph.dump_time(new[0])))
    finally:
        ph.rmdirs(chain, ref)
