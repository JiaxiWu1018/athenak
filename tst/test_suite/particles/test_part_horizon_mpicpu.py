"""GR horizon-capture regression (NRPIC Stage 3d, D-T5).

Schwarzschild (CKS, M=1): a rest shell at r0=5 plunges through a 2-level SMR grid
and is excised by the sphere criterion inside the horizon; an equatorial ring at
r0=8 survives. Exercises gr_boris x {migration, SMR, MPI, destruction} and the
decomposition-determinism contract (np1 == np2 bitwise, death records included).

Needs a build with the part_kerr_schild user pgen -- separate invocation from the
part_crossing tests (one PROBLEM per build):

    cd tst
    python3 run_test_suite.py --mpicpu "-D PROBLEM=part_kerr_schild" \\
        --test test_suite/particles/test_part_horizon_mpicpu.py

In the stock full-suite run this module SKIPs. Never run the FULL suite with
-DPROBLEM set. Standalone dev loop: as in test_part_crossing_mpicpu.py with
-DPROBLEM=part_kerr_schild.
"""
import math
import os

import test_suite.particles.part_helpers as ph

ph.require_problem("part_kerr_schild")

INPUT = "inputs/part_horizon_schw_pytest.athinput"
BASE = "horizon_schw_pytest"
SHELL = set(range(128))      # tags [0,128): rest shell, all captured
RING = set(range(128, 192))  # tags [128,192): circular ring, all survive


def test_horizon_capture_sphere():
    """D-T5: whole shell captured by excise_radius=1.0 strictly inside the horizon
    (r_death < 2M); ring survives to tlim; bitwise across np {1,2} including the
    death records (rank column excluded -- gather order is an MPI artifact)."""
    nps = ph.rank_counts((1, 2))
    dirs = {n: f"prt_t5_np{n}" for n in nps}
    rows_by_np = {}
    try:
        for n in nps:
            ph.rmdirs(dirs[n])
            ph.run_args(["-i", INPUT, "-d", dirs[n],
                         "particles/excise_radius=1.0"], threads=n)
            rows = ph.read_death_csv(
                os.path.join(dirs[n], BASE + ".prtcl_destroy.csv"))
            rows_by_np[n] = rows
            assert {r["tag"] for r in rows} == SHELL, "shell not fully captured"
            assert all(r["reason"] == "sphere" for r in rows)
            # marked at the first sample with r < R: crit (= r at death) sits just
            # under the excision radius, far inside the horizon r_h = 2M
            assert all(0.9 < r["crit"] < 1.0 for r in rows)
            assert all(math.sqrt(r["x"]**2 + r["y"]**2 + r["z"]**2) < 2.0
                       for r in rows), "a particle was excised outside the horizon"
            ph.assert_death_invariants(rows)
            _, _, _, _, tag = ph.read_part_vtk(ph.pick_dump(dirs[n], "last"))
            assert set(tag.tolist()) == RING, "ring did not survive intact"
        if len(nps) > 1:
            ph.assert_last_dumps_bitwise(dirs[nps[0]], dirs[nps[1]])
            a = {r["tag"]: ph.death_key(r) for r in rows_by_np[nps[0]]}
            b = {r["tag"]: ph.death_key(r) for r in rows_by_np[nps[1]]}
            assert a == b, "death records differ across rank counts"
    finally:
        ph.rmdirs(*dirs.values())
