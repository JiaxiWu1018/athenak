"""Particle redistribution through dynamic-AMR regrids."""

import test_suite.particles.part_helpers as helpers


helpers.require_problem("particles/part_crossing")

INPUT = "inputs/part_crossing_amr_pytest.athinput"


def test_amr_lattice_across_ranks():
    """Particles retain their identities and analytic drift through repeated regrids."""
    ranks = helpers.rank_counts((1, 2, 4))
    directories = {rank: f"prt_amr_np{rank}" for rank in ranks}
    try:
        for rank in ranks:
            helpers.remove_dirs(directories[rank])
            helpers.run_case(INPUT, directories[rank], rank)
            helpers.assert_analytic_periodic_drift(directories[rank])
        for rank in ranks[1:]:
            helpers.assert_final_positions_bitwise(
                directories[ranks[0]], directories[rank]
            )
    finally:
        helpers.remove_dirs(*directories.values())
