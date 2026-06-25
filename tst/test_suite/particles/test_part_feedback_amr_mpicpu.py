"""Feedback + dynamic chi-AMR determinism gate (NRPIC Stage 5c).

PROBLEM=z4c_one_puncture with <particles> feedback=true and chi-driven adaptive AMR --
the case that was FATAL-guarded through Stage 5b. The particles source Tmunu (-> the Z4c
RHS) through the regrids: Stage 5c relabels + ships them to their new blocks AND
re-deposits Tmunu on the new grid before the next CalcRHS, so the gr_boris geodesic no
longer reads a stale wrong-block metric. With <particles> debug=1 the post-regrid
containment + ledger and the per-cycle E-conservation identity (Sum E*sqrt(gamma)*dV ==
Sum m*W, exact for scheme A even across the freshly-refined seam) are fatal, so a clean
exit IS the per-cycle oracle.
On top of that, the per-tag particle positions and velocities AND the deposited Tmunu
must be bitwise identical across rank counts through the regrids -- the determinism
contract under feedback + AMR (CPU/serial-host).

    cd tst
    python3 run_test_suite.py --mpicpu "-D PROBLEM=z4c_one_puncture" \\
        --test test_suite/particles/test_part_feedback_amr_mpicpu.py
"""
import test_suite.particles.part_helpers as ph

ph.require_problem("z4c_one_puncture")

INPUT = "inputs/z4c_op_amr_fb_pytest.athinput"


def test_feedback_amr_bitwise_across_ranks():
    """5c-T1: feedback + chi-AMR, np {1,2,4}. Particles survive every regrid (debug=1
    ledger fatal), the deposit conserves across the chi-created seams every cycle
    (identity fatal), and both the per-tag positions and the deposited Tmunu are bitwise
    identical across rank counts through the regrids."""
    nps = ph.rank_counts((1, 2, 4))
    dirs = {n: f"prt_fbamr_np{n}" for n in nps}
    logs = {}
    try:
        for n in nps:
            ph.rmdirs(dirs[n])
            logs[n] = ph.run_args(["-i", INPUT, "-d", dirs[n]], threads=n)
            ph.assert_refined(dirs[n])
            ph.assert_cross_level(logs[n])
            ph.assert_regridded(logs[n])
        for n in nps[1:]:
            ph.assert_last_dumps_bitwise(dirs[nps[0]], dirs[n])
            ph.assert_tmunu_last_bitwise(dirs[nps[0]], dirs[n])
    finally:
        ph.rmdirs(*dirs.values())
