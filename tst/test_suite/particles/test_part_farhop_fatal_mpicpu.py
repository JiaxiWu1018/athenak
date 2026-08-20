"""Particle motion beyond the supported migration range must be fatal, not repaired.

Migration resolves a destination through the 56-slot neighbour array, i.e. the 26
immediate neighbours -- one MeshBlock width in each direction. A longer hop has no
representable destination, so it aborts instead of being re-homed.

Both directions of that threshold, on 1 and 4 ranks:

* ``farhop`` (hops = 2,3, ``debug = 0``): all 3328 particles travel 2 or 3 block widths
  in one update, so the run must abort with the migration diagnostic in the log. Matching
  the message, not just a nonzero exit, distinguishes this from an unrelated crash, and
  ``debug = 0`` shows detection does not need the debug instrumentation.
* ``nearhop`` (hops = 1): exactly one block width is a supported crossing, so the run
  must complete, follow the analytic periodic drift, and be decomposition-independent.

Run from ``tst`` with::

    python3 run_test_suite.py \
        --mpicpu "-D PROBLEM=particles/part_crossing" \
        --test test_suite/particles/test_part_farhop_fatal_mpicpu.py
"""

import re

import pytest

import test_suite.particles.part_helpers as helpers


helpers.require_problem("particles/part_crossing")

FAR_INPUT = "inputs/part_crossing_farhop_pytest.athinput"
NEAR_INPUT = "inputs/part_crossing_nearhop_pytest.athinput"
BLOCKS = 64                       # 32^3 mesh in 8^3 MeshBlocks
FAR_PARTICLES = BLOCKS * 26 * 2   # 26 offset directions x hop counts {2,3}
NEAR_PARTICLES = BLOCKS * 26 * 1
SUMMARY_RE = re.compile(
    r"particle migration cannot place (\d+) particle\(s\) on rank (\d+)"
)
NGHBR_RE = re.compile(
    r"(\d+) moved within the supported range but found no neighbour"
)


def test_multiblock_hops_are_fatal():
    """2- and 3-block hops in all 26 directions must stop the run, on 1 and 4 ranks."""
    ranks = helpers.rank_counts((1, 4))
    run_dirs = {count: f"particle_farhop_fatal_np{count}" for count in ranks}
    try:
        for count in ranks:
            helpers.remove_dirs(run_dirs[count])
            log = helpers.run_case(
                FAR_INPUT, run_dirs[count], count, expect_failure=True
            )
            assert f"placed {FAR_PARTICLES} particles" in log, (
                "pgen did not stage the expected particle set"
            )
            assert "FATAL ERROR particle migration" in log, (
                f"{count} rank(s): no per-particle migration diagnostic in the log"
            )
            assert (
                "particle moved MORE THAN ONE MeshBlock width in one update" in log
            ), f"{count} rank(s): the hop was not classified as unsupported motion"
            # Pin the counts, not just the presence of a summary: every particle in this
            # fixture must be rejected, so a prefix match alone would miss a partial
            # regression. Blocks are spread evenly, 26 directions x 2 hop counts each.
            summaries = SUMMARY_RE.findall(log)
            assert summaries, f"{count} rank(s): no host-side fatal summary in the log"
            expected_per_rank = (BLOCKS // count) * 26 * 2
            for got, rank in summaries:
                assert int(got) == expected_per_rank, (
                    f"{count} rank(s): rank {rank} reported {got} unplaceable "
                    f"particles, expected {expected_per_rank} -- some hops were NOT "
                    "detected as unsupported motion"
                )
            # ...and attributed to overspeed, not to a broken mesh. This also pins the
            # two counters against being transposed, which no fixture can test directly
            # (a 2:1-balance violation cannot be built by the tree).
            nghbr_counts = NGHBR_RE.findall(log)
            assert nghbr_counts, f"{count} rank(s): summary is missing the cause split"
            assert all(int(n) == 0 for n in nghbr_counts), (
                f"{count} rank(s): overspeed hops were misattributed to a broken "
                f"neighbour contract: {nghbr_counts}"
            )
            # the diagnostic must be actionable: particle identity and hop size
            assert "tag=" in log and "MeshBlock widths" in log, (
                f"{count} rank(s): diagnostic is missing tag/displacement detail"
            )
    finally:
        helpers.remove_dirs(*run_dirs.values())
    if len(ranks) < 2:
        pytest.skip(
            f"only rank counts {ranks} were available on this runner, so the "
            "multi-rank case did not run (it is not a silent pass)"
        )


def test_supported_crossings_do_not_abort_without_debug():
    """A healthy crossing must not abort when <particles> debug = 0.

    The nearhop fixture runs with debug = 1, and the only debug = 0 fixture is the one
    required to abort, so without this case nothing would catch the detection becoming
    debug-dependent and aborting on a legal single-block crossing.
    """
    ranks = helpers.rank_counts((1, 4))
    run_dirs = {count: f"particle_nearhop_nodebug_np{count}" for count in ranks}
    try:
        for count in ranks:
            helpers.remove_dirs(run_dirs[count])
            log = helpers.run_case(
                NEAR_INPUT, run_dirs[count], count, extra_args=["particles/debug=0"]
            )
            assert f"placed {NEAR_PARTICLES} particles" in log
            assert "FATAL" not in log, (
                f"{count} rank(s): a supported crossing aborted with debug = 0:\n"
                + "\n".join(ln for ln in log.splitlines() if "FATAL" in ln)[:800]
            )
            helpers.assert_analytic_periodic_drift(run_dirs[count])
    finally:
        helpers.remove_dirs(*run_dirs.values())


def test_single_block_hops_still_migrate():
    """A hop of exactly one MeshBlock width is supported and must NOT be fatal."""
    ranks = helpers.rank_counts((1, 4))
    run_dirs = {count: f"particle_nearhop_np{count}" for count in ranks}
    try:
        for count in ranks:
            helpers.remove_dirs(run_dirs[count])
            log = helpers.run_case(NEAR_INPUT, run_dirs[count], count)
            assert f"placed {NEAR_PARTICLES} particles" in log, (
                "pgen did not stage the expected particle set"
            )
            assert "FATAL" not in log, (
                f"{count} rank(s): a supported one-block hop was rejected:\n"
                + "\n".join(ln for ln in log.splitlines() if "FATAL" in ln)[:800]
            )
            helpers.assert_analytic_periodic_drift(run_dirs[count])
        for count in ranks[1:]:
            helpers.assert_final_positions_bitwise(run_dirs[ranks[0]], run_dirs[count])
    finally:
        helpers.remove_dirs(*run_dirs.values())
    if len(ranks) < 2:
        pytest.skip(
            f"only rank counts {ranks} were available on this runner, so the bitwise "
            "decomposition-independence assertion never ran (it is not a silent pass)"
        )
