"""MPI regression for <particles> excise_ah: apparent-horizon particle excision.

Run from ``tst`` with::

    python3 run_test_suite.py \
        --mpicpu "-D PROBLEM=z4c/z4c_one_puncture" \
        --test test_suite/particles/test_ah_excision_mpicpu.py

The fixture is one M = 1 puncture with the shipped AH-finder AMR setup and two equatorial
rings of geodesic particles: 16 at r = 0.30, inside the horizon (FastFlow recovers
r_AH = 0.4920 against the exact Brill-Lindquist isotropic value 0.5), and 16 at r = 4.00,
outside it. ``excise_lapse`` is 0, so a nonzero ``lapse`` count would mean the reason
attribution is wrong.
"""

import math
import os
import re

import test_suite.particles.part_helpers as helpers


helpers.require_problem("z4c/z4c_one_puncture")

INPUT = "inputs/ah_excision_onepunc_pytest.athinput"
CSV = "ah_excision_onepunc_pytest.prtcl_destroy.csv"
INNER = set(range(16))          # r = 0.30, inside the horizon
OUTER = set(range(16, 32))      # r = 4.00, outside it

# FastFlow's own reported minimum radius for this configuration at t = 0.
R_AH_MIN = 0.4894837848062050


def _self_check_deviation(log):
    """Worst deviation reported by FastFlow's off-grid surface self-check."""
    match = re.search(r"AH surface self-check .*worst relative deviation = ([-\d.eE+]+)",
                      log)
    assert match, "FastFlow did not emit the AH surface self-check line"
    return float(match.group(1))


def test_ah_excision_removes_only_the_inside_ring():
    """Inside the horizon is destroyed, outside survives, and the reason is 'horizon'."""
    ranks = helpers.rank_counts((1, 2, 4))
    run_dirs = {count: f"ah_excision_np{count}" for count in ranks}
    deaths = {}
    try:
        for count in ranks:
            helpers.remove_dirs(run_dirs[count])
            log = helpers.run_case(INPUT, run_dirs[count], count)

            # The consumer must evaluate the SAME surface the finder produced. A packing,
            # normalization or theta/phi slip here would be silent and would misclassify
            # exactly the particles closest to the horizon.
            assert _self_check_deviation(log) == 0.0

            # A horizon that is fully resolved on the mesh must not be gated away.
            assert "AH surface REJECTED for excision" not in log

            rows = helpers.read_death_csv(os.path.join(run_dirs[count], CSV))
            deaths[count] = rows
            assert {row["tag"] for row in rows} == INNER
            assert all(row["reason"] == "horizon" for row in rows)
            # crit is the containment ratio r / R(theta,phi): < 1 means genuinely inside.
            assert all(0.0 < row["crit"] < 1.0 for row in rows)
            # and it must reproduce the surface radius the finder reported
            for row in rows:
                radius = math.sqrt(row["x"]**2 + row["y"]**2 + row["z"]**2)
                assert math.isclose(radius/row["crit"], R_AH_MIN, rel_tol=1.0e-6)

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
        assert "EXPERIMENTAL: <particles> excise_ah is ON" not in log
        assert not os.path.exists(os.path.join(run_dir, CSV))
        assert "destroyed: exit=0 sphere=0 lapse=0 horizon=0" in log
        assert "conservation OK" in log
    finally:
        helpers.remove_dirs(run_dir)
