"""Particle feedback through dynamic-AMR regrids."""

import os

import test_suite.particles.part_helpers as helpers


helpers.require_problem("z4c/z4c_one_puncture")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
INPUT = os.path.join(REPO, "tst", "inputs", "z4c_op_amr_fb_pytest.athinput")


def test_feedback_amr_is_rank_invariant():
    """Particles and re-deposited Tmunu remain deterministic through live regrids."""
    ranks = helpers.rank_counts((1, 2, 4))
    directories = {count: f"prt_fbamr_np{count}" for count in ranks}
    try:
        for count in ranks:
            helpers.remove_dirs(directories[count])
            log = helpers.run_case(INPUT, directories[count], count)
            helpers.assert_regridded(log)
            helpers.assert_mixed_level_output(directories[count])
            helpers.assert_cross_level_deposition(log)
        for count in ranks[1:]:
            helpers.assert_final_particle_state_bitwise(
                directories[ranks[0]], directories[count]
            )
            helpers.assert_final_tmunu_bitwise(
                directories[ranks[0]], directories[count]
            )
    finally:
        helpers.remove_dirs(*directories.values())
