"""Oppenheimer-Snyder collapse with particle feedback and live chi-AMR."""

import glob
import os

import pytest

import test_suite.particles.part_helpers as helpers


helpers.require_problem("particles/nr_pic_os")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
INPUT = os.path.join(REPO, "tst", "inputs", "nr_pic_os_amr_pytest.athinput")


def _assert_exercised(run_dir, log):
    helpers.assert_regridded(log)
    helpers.assert_mixed_level_output(run_dir)
    helpers.assert_cross_level_deposition(log)


def test_os_amr_is_rank_invariant():
    """The complete OS-AMR-feedback composition is deterministic across MPI ranks."""
    ranks = helpers.rank_counts((1, 2, 4))
    directories = {count: f"os_amr_np{count}" for count in ranks}
    try:
        for count in ranks:
            helpers.remove_dirs(directories[count])
            log = helpers.run_case(INPUT, directories[count], count)
            _assert_exercised(directories[count], log)
        for count in ranks[1:]:
            helpers.assert_final_particle_state_bitwise(
                directories[ranks[0]], directories[count]
            )
            helpers.assert_final_tmunu_bitwise(
                directories[ranks[0]], directories[count]
            )
    finally:
        helpers.remove_dirs(*directories.values())


def test_os_amr_restart_across_rank_change():
    """A restart after a live regrid matches an uninterrupted two-rank evolution."""
    if (os.cpu_count() or 1) < 2:
        pytest.skip("needs at least two CPU cores")
    chain = "os_amr_restart_chain"
    reference = "os_amr_restart_reference"
    try:
        helpers.remove_dirs(chain, reference)
        first_log = helpers.run_case(
            INPUT, chain, 1, extra_args=["time/nlim=2"]
        )
        restart_files = sorted(glob.glob(os.path.join(chain, "rst", "*.rst")))
        assert restart_files, "first segment wrote no restart"
        restart_log = helpers.run_args(
            ["-r", restart_files[-1], "-d", chain, "time/nlim=5"], threads=2
        )
        reference_log = helpers.run_case(
            INPUT, reference, 2, extra_args=["time/nlim=5"]
        )
        helpers.assert_regridded(first_log)
        helpers.assert_regridded(restart_log)
        _assert_exercised(reference, reference_log)
        helpers.assert_mixed_level_output(chain)
        helpers.assert_cross_level_deposition(first_log + restart_log)
        helpers.assert_final_particle_state_bitwise(chain, reference)
        helpers.assert_final_tmunu_bitwise(chain, reference)
    finally:
        helpers.remove_dirs(chain, reference)
