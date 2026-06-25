"""Oppenheimer-Snyder collapse + live chi-AMR + feedback: bitwise acceptance (NRPIC 5c).

PROBLEM=nr_pic_os with <particles> feedback=true and chi-driven adaptive AMR -- the
headline configuration (README test 15). The collapsing dust sources Tmunu through the
chi-created regrids; debug=1 makes the post-regrid ledger + the per-cycle E-conservation
identity (exact for scheme A across the freshly-refined seam) fatal, so a clean exit is
the per-cycle oracle. On top of that the per-tag particle positions and velocities AND
the deposited Tmunu must be bitwise identical across rank counts through the regrids --
the determinism contract for the full OS-AMR-feedback composition (CPU/serial-host).

    cd tst
    python3 run_test_suite.py --mpicpu "-D PROBLEM=nr_pic_os" \\
        --test test_suite/particles/test_os_amr_mpicpu.py
"""
import glob
import os

import pytest

import test_suite.particles.part_helpers as ph

ph.require_problem("nr_pic_os")

INPUT = "inputs/nr_pic_os_amr_pytest.athinput"


def test_os_amr_bitwise_across_ranks():
    """5c-T2 (acceptance): OS dust collapse + chi-AMR + feedback, np {1,2,4}. The dust
    survives the chi-created regrids (debug=1 ledger fatal), deposits conservatively
    across the seams every cycle (identity fatal), and both the per-tag positions and the
    deposited Tmunu are bitwise identical across rank counts through the regrids."""
    nps = ph.rank_counts((1, 2, 4))
    dirs = {n: f"os_amr_np{n}" for n in nps}
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


def test_os_amr_restart_straddling_regrid():
    """5c-T3 (restart-under-fire): np1 stops just after its first OS-AMR regrid, then
    restarts on np2 and must regrid again, versus an uninterrupted np2 reference. Particle
    positions and velocities plus Tmunu are bitwise equal at the first post-restart and
    final matched dumps. This exercises bbox re-placement, regrid remapping, fresh Tmunu
    deposition, and gr_boris metric-snapshot seeding across a decomposition change."""
    if (os.cpu_count() or 1) < 2:
        pytest.skip("needs >= 2 cores")
    chain, ref = "os_amr_t3_chain", "os_amr_t3_ref"
    try:
        ph.rmdirs(chain, ref)
        log1 = ph.run_args(["-i", INPUT, "-d", chain, "time/nlim=2"], threads=1)
        seg1 = set(ph.list_dumps(chain))
        tseg1 = set(ph.tmunu_dumps(chain))
        rsts = sorted(glob.glob(os.path.join(chain, "rst", "*.rst")))
        assert rsts, "stage 1 wrote no restart file"
        log2 = ph.run_args(["-r", rsts[-1], "-d", chain, "time/nlim=5"], threads=2)
        log_ref = ph.run_args(["-i", INPUT, "-d", ref, "time/nlim=5"], threads=2)
        ph.assert_regridded(log1)
        ph.assert_regridded(log2)
        ph.assert_regridded(log_ref)
        ph.assert_cross_level(log1)
        ph.assert_cross_level(log2)
        ph.assert_cross_level(log_ref)
        ph.assert_refined(chain)
        ph.assert_refined(ref)
        # final state: particles AND re-deposited Tmunu match the uninterrupted ref
        ph.assert_last_dumps_bitwise(chain, ref)
        ph.assert_tmunu_last_bitwise(chain, ref)
        # first post-restart dump (time-matched): the redeposition right after regrid
        new = [f for f in ph.list_dumps(chain) if f not in seg1]
        assert new, "restarted segment produced no dumps"
        ph.assert_bitwise(new[0], ph.pick_dump(ref, "match", ph.dump_time(new[0])))
        tnew = [f for f in ph.tmunu_dumps(chain) if f not in tseg1]
        assert tnew, "restarted segment produced no tmunu dumps"
        ph.assert_tmunu_bitwise(
            tnew[0], ph.pick_tmunu(ref, "match", ph.tmunu_time(tnew[0])))
    finally:
        ph.rmdirs(chain, ref)
