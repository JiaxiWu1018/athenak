"""Particle migration / destruction / restart regression tier (NRPIC Stage 3d).

These tests need a build with the part_crossing user pgen (NOT in the built-in
dispatch). Dedicated invocation -- the space inside the quoted flag matters
(argparse rejects a bare "-DPROBLEM=..." value):

    cd tst
    python3 run_test_suite.py --mpicpu "-D PROBLEM=part_crossing" \\
        --test test_suite/particles/test_part_crossing_mpicpu.py

In the stock full-suite run this module SKIPs (the build has
PROBLEM=built_in_pgens). Never run the FULL test suite with -DPROBLEM set: such a
build compiles the built-in problem generators out and every other test fails.

Standalone dev loop (run_test_suite.py deletes tst/build at the end of each run):
    cmake -B tst/build -DAthena_ENABLE_MPI=ON -DPROBLEM=part_crossing
    make -C tst/build -j10
    ln -s ../../inputs tst/build/src/inputs
    (cd tst/build/src && python3 -m pytest ../../test_suite/particles -v)

This tier is the fast repo-resident gate; the ~400-run research campaign suites
remain the deep-verification instrument. Every run has <particles> debug=1, so the
in-code per-cycle validator (containment, PGID range, two-sided conservation
ledger) is fatal -- a nonzero exit fails the test before any artifact assert.
"""
import glob
import os

import pytest

import test_suite.particles.part_helpers as ph

ph.require_problem("part_crossing")

INPUT = "inputs/part_crossing_pytest.athinput"
SMR_INPUT = "inputs/part_crossing_smr_pytest.athinput"
BASE = "part_crossing_pytest"
NLAT = 512  # 8^3 lattice; tag = i + 8*(j + 8*k)


def test_uniform_lattice_across_ranks():
    """D-T1: drift lattice on the uniform 8-block grid, np {1,2,4}: per-tag positions
    bitwise identical across rank counts; analytic drift x0 + v*t (mod L)."""
    nps = ph.rank_counts((1, 2, 4))
    dirs = {n: f"prt_t1_np{n}" for n in nps}
    try:
        for n in nps:
            ph.rmdirs(dirs[n])
            ph.run_args(["-i", INPUT, "-d", dirs[n]], threads=n)
            ph.assert_analytic_drift(dirs[n])
        for n in nps[1:]:
            ph.assert_last_dumps_bitwise(dirs[nps[0]], dirs[n])
    finally:
        ph.rmdirs(*dirs.values())


def test_smr_lattice_across_ranks():
    """D-T2: same checks on the two-level SMR grid (level boundary carries both
    block parities -- the parity-dependent coarse-lookup configuration)."""
    nps = ph.rank_counts((1, 4))
    dirs = {n: f"prt_t2_np{n}" for n in nps}
    try:
        for n in nps:
            ph.rmdirs(dirs[n])
            ph.run_args(["-i", SMR_INPUT, "-d", dirs[n]], threads=n)
            ph.assert_analytic_drift(dirs[n])
        for n in nps[1:]:
            ph.assert_last_dumps_bitwise(dirs[nps[0]], dirs[n])
    finally:
        ph.rmdirs(*dirs.values())


OUTFLOW = ["mesh/{}_bc=outflow".format(k)
           for k in ("ix1", "ox1", "ix2", "ox2", "ix3", "ox3")]


def test_outflow_destruction_census():
    """D-T3: all-faces outflow drain, np {1,2}. Analytic census: only the rest
    particles (tag % 27 == 26 in the lattice velocity pattern) survive; everything
    else exits. Death CSV complete, exit-only, per-dump set algebra holds, and the
    records are identical across rank counts."""
    nps = ph.rank_counts((1, 2))
    dirs = {n: f"prt_t3_np{n}" for n in nps}
    survivors = {t for t in range(NLAT) if t % 27 == 26}  # 18 rest particles
    rows_by_np = {}
    try:
        for n in nps:
            ph.rmdirs(dirs[n])
            ph.run_args(["-i", INPUT, "-d", dirs[n]] + OUTFLOW, threads=n)
            _, _, _, _, tag = ph.read_part_vtk(ph.pick_dump(dirs[n], "last"))
            assert set(tag.tolist()) == survivors, "survivor set != rest particles"
            rows = ph.read_death_csv(
                os.path.join(dirs[n], BASE + ".prtcl_destroy.csv"))
            rows_by_np[n] = rows
            assert len(rows) == NLAT - len(survivors), "death CSV incomplete"
            assert all(r["reason"] == "exit" for r in rows)
            ph.assert_death_invariants(rows)
            dead = {r["tag"] for r in rows}
            assert not (dead & survivors), "a survivor has a death record"
            assert dead | survivors == set(range(NLAT)), "tags unaccounted for"
            ph.assert_death_set_algebra(dirs[n], rows)
        if len(nps) > 1:
            a = {r["tag"]: ph.death_key(r) for r in rows_by_np[nps[0]]}
            b = {r["tag"]: ph.death_key(r) for r in rows_by_np[nps[1]]}
            assert a == b, "death records differ across rank counts"
    finally:
        ph.rmdirs(*dirs.values())


def test_restart_continuation_changed_ranks():
    """D-T4: rst written on np1 at nlim=10, restarted on np2 to nlim=20, vs an
    uninterrupted np2 reference. Final dumps bitwise per tag with tag-SET equality
    (no re-tagging through restart); first post-restart dump compared time-matched
    (restart runs skip the initial output and seg1's Finalize dump shifts the
    output-threshold ladder, so file numbers never align -- match by header time)."""
    if (os.cpu_count() or 1) < 2:
        pytest.skip("needs >= 2 cores")
    chain, ref = "prt_t4_chain", "prt_t4_ref"
    try:
        ph.rmdirs(chain, ref)
        ph.run_args(["-i", INPUT, "-d", chain, "time/nlim=10"], threads=1)
        seg1_dumps = set(ph.list_dumps(chain))
        rsts = sorted(glob.glob(os.path.join(chain, "rst", "*.rst")))
        assert rsts, "stage 1 wrote no restart file"
        # -r is resolved before the -d chdir -> path relative to launch cwd; seg2
        # takes its parameters from the rst (no -i) with CLI overrides on top
        ph.run_args(["-r", rsts[-1], "-d", chain, "time/nlim=20"], threads=2)
        ph.run_args(["-i", INPUT, "-d", ref, "time/nlim=20"], threads=2)
        ph.assert_last_dumps_bitwise(chain, ref)
        new = [f for f in ph.list_dumps(chain) if f not in seg1_dumps]
        assert new, "restarted segment produced no dumps"
        ph.assert_bitwise(new[0],
                          ph.pick_dump(ref, "match", ph.dump_time(new[0])))
    finally:
        ph.rmdirs(chain, ref)
