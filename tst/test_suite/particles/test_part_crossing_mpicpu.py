"""MPI regression for particle migration across a two-level SMR mesh.

Run from ``tst`` with::

    python3 run_test_suite.py \
        --mpicpu "-D PROBLEM=particles/part_crossing" \
        --test test_suite/particles/test_part_crossing_mpicpu.py
"""

import test_suite.particles.part_helpers as helpers


helpers.require_problem("particles/part_crossing")

INPUT = "inputs/part_crossing_smr_pytest.athinput"


def test_smr_lattice_is_rank_independent():
    """Check analytic drift and bitwise results across MPI decompositions."""
    ranks = helpers.rank_counts((1, 4))
    run_dirs = {count: f"particle_crossing_np{count}" for count in ranks}
    try:
        for count in ranks:
            helpers.remove_dirs(run_dirs[count])
            helpers.run_case(INPUT, run_dirs[count], count)
            helpers.assert_analytic_periodic_drift(run_dirs[count])
        for count in ranks[1:]:
            helpers.assert_final_positions_bitwise(run_dirs[ranks[0]], run_dirs[count])
    finally:
        helpers.remove_dirs(*run_dirs.values())
