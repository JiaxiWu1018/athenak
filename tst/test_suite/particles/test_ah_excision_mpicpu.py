"""MPI regression for <particles> excise_ah: apparent-horizon particle excision.

Run from ``tst`` with::

    python3 run_test_suite.py \
        --mpicpu "-D PROBLEM=z4c/z4c_one_puncture" \
        --test test_suite/particles/test_ah_excision_mpicpu.py

The fixture is one M = 1 puncture with the shipped AH-finder AMR setup and two
equatorial rings of geodesic particles: 16 at r = 0.30, inside the horizon
(Brill-Lindquist isotropic AH at r = M/2), and 16 at r = 4.00, outside it.
``excise_lapse`` is 0, so a nonzero ``lapse`` count would mean the reason
attribution is wrong.
"""

import math
import os

import test_suite.particles.part_helpers as helpers


helpers.require_problem("z4c/z4c_one_puncture")

INPUT = "inputs/ah_excision_onepunc_pytest.athinput"
CSV = "ah_excision_onepunc_pytest.prtcl_destroy.csv"
SUMMARY = "ah_excision_onepunc_pytest.horizon_summary_0.txt"
INNER = set(range(16))          # r = 0.30, inside the horizon
OUTER = set(range(16, 32))      # r = 4.00, outside it


def _first_found_min_radius():
    """FastFlow's own minimum surface radius at the first converged find.

    The inner ring dies at the first find, and for r < rmin the recorded crit is
    r/rmin, so radius/crit must reproduce this value -- comparing against the
    finder's own file catches a packing/normalization slip in the staged surface.
    FastFlow writes this file to the working directory, not the -d run dir, and
    each run overwrites it, so read it right after the run it belongs to.
    """
    with open(SUMMARY, encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("#"):
                continue
            rmin = float(line.split()[11])
            assert math.isfinite(rmin) and rmin > 0.0, (
                "first FastFlow find did not converge; the fixture assumes it does"
            )
            return rmin
    raise AssertionError("FastFlow wrote no summary rows")


def test_ah_excision_removes_only_the_inside_ring():
    """Inside the horizon is destroyed, outside survives, and the reason is 'horizon'."""
    ranks = helpers.rank_counts((1, 2, 4))
    run_dirs = {count: f"ah_excision_np{count}" for count in ranks}
    deaths = {}
    try:
        for count in ranks:
            helpers.remove_dirs(run_dirs[count])
            log = helpers.run_case(INPUT, run_dirs[count], count)

            # A horizon that is fully resolved on the mesh must not be gated away.
            assert "not published to consumers" not in log

            rows = helpers.read_death_csv(os.path.join(run_dirs[count], CSV))
            deaths[count] = rows
            assert {row["tag"] for row in rows} == INNER
            assert all(row["reason"] == "horizon" for row in rows)
            # crit is the containment ratio r / R: < 1 means genuinely inside.
            assert all(0.0 < row["crit"] < 1.0 for row in rows)
            # and it must reproduce the surface radius the finder itself reported
            r_ah_min = _first_found_min_radius()
            for row in rows:
                radius = math.sqrt(row["x"]**2 + row["y"]**2 + row["z"]**2)
                assert math.isclose(radius/row["crit"], r_ah_min, rel_tol=1.0e-6)

            _, _, _, tags = helpers.read_particle_vtk(
                helpers.particle_dumps(run_dirs[count])[-1]
            )
            assert set(tags.tolist()) == OUTER

            assert "conservation OK" in log
            assert " horizon=16" in log

        for count in ranks[1:]:
            reference = {row["tag"]: helpers.death_key(row) for row in deaths[ranks[0]]}
            candidate = {row["tag"]: helpers.death_key(row) for row in deaths[count]}
            assert candidate == reference, "death records differ across rank counts"
    finally:
        helpers.remove_dirs(*run_dirs.values())


def test_ah_excision_is_off_by_default_and_inert_without_it():
    """With excise_ah off, nothing is destroyed even though an AH exists every cycle."""
    run_dir = "ah_excision_off"
    try:
        helpers.remove_dirs(run_dir)
        log = helpers.run_case(INPUT, run_dir, helpers.rank_counts((1,))[0],
                               extra_args=["particles/excise_ah=false"])
        assert not os.path.exists(os.path.join(run_dir, CSV))
        assert "destroyed: exit=0 sphere=0 lapse=0 horizon=0" in log
        assert "conservation OK" in log
    finally:
        helpers.remove_dirs(run_dir)
