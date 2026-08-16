"""MPI regression for GR particle migration and destruction inside a horizon.

Run from ``tst`` with::

    python3 run_test_suite.py \
        --mpicpu "-D PROBLEM=particles/part_kerr_schild" \
        --test test_suite/particles/test_part_horizon_mpicpu.py
"""

import math
import os

import test_suite.particles.part_helpers as helpers


helpers.require_problem("particles/part_kerr_schild")

INPUT = "inputs/part_horizon_schw_pytest.athinput"
SHELL = set(range(128))
RING = set(range(128, 192))


def test_schwarzschild_horizon_capture_is_rank_independent():
    """The plunging shell is excised, while the orbiting ring survives."""
    ranks = helpers.rank_counts((1, 2))
    run_dirs = {count: f"particle_horizon_np{count}" for count in ranks}
    deaths = {}
    try:
        for count in ranks:
            helpers.remove_dirs(run_dirs[count])
            helpers.run_case(INPUT, run_dirs[count], count)
            rows = helpers.read_death_csv(
                os.path.join(run_dirs[count], "horizon_schw_pytest.prtcl_destroy.csv")
            )
            deaths[count] = rows
            assert {row["tag"] for row in rows} == SHELL
            assert all(row["reason"] == "sphere" for row in rows)
            assert all(0.9 < row["crit"] < 1.0 for row in rows)
            assert all(
                math.sqrt(row["x"] ** 2 + row["y"] ** 2 + row["z"] ** 2) < 2.0
                for row in rows
            )
            _, _, _, tags = helpers.read_particle_vtk(
                helpers.particle_dumps(run_dirs[count])[-1]
            )
            assert set(tags.tolist()) == RING

        for count in ranks[1:]:
            helpers.assert_final_positions_bitwise(run_dirs[ranks[0]], run_dirs[count])
            reference = {row["tag"]: helpers.death_key(row) for row in deaths[ranks[0]]}
            candidate = {row["tag"]: helpers.death_key(row) for row in deaths[count]}
            assert candidate == reference, "death records differ across rank counts"
    finally:
        helpers.remove_dirs(*run_dirs.values())
