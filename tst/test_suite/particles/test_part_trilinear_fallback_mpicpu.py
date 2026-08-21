"""The trilinear geodesic fallback must repair rejected pushes and change nothing else.

``<particles> trilinear_fallback`` re-solves a geodesic substep with a trilinear
interpolant when the high-order one yields geometry that is not a valid 3-metric. The
retry is entered ONLY for substeps the production code would reject, so the two
properties worth pinning are:

1. every push the production code accepts is bit-for-bit unchanged, whichever mode is
   set -- checked here on the well-resolved outer ring of an under-resolved puncture,
   and, when no rejection happens at all, on the whole ensemble;
2. when a rejection does happen the retry is attempted, counted, and reported, and the
   run still finishes with a conserved ledger and no non-finite state.

Note what is NOT asserted: that the TOTAL retry count over the run equals the total
rejection count of the ``none`` run. It does not, and must not -- a rescued push moves a
particle the production run leaves frozen, so the two trajectories separate and the later
cycles are no longer comparable. The cross-run equality is asserted only at the FIRST
cycle that reports anything, where it is exact by construction: the flag cannot act
before the first rejection, so up to that cycle the two runs execute identical code on
identical data.

The deck puts one ring at r = 0.3 on a dx = 1/3 grid, inside the first cell of the
trumpet. Measured: it does NOT in fact produce a rejection -- see the deck's own header
for the three configurations tried -- so on this fixture the test takes the "no rejection
anywhere, therefore nothing may change" branch and asserts the whole ensemble is bitwise
identical across all three modes. The rescue path is exercised by the y_c = 0.700
acceptance run, not here; the branch below is written and kept so that the day a fixture
does reject, the assertions are already in place. Run from ``tst`` with::

    python3 run_test_suite.py \
        --mpicpu "-D PROBLEM=z4c/z4c_one_puncture" \
        --test test_suite/particles/test_part_trilinear_fallback_mpicpu.py
"""

import os
import re

import numpy as np

import test_suite.particles.part_helpers as helpers


helpers.require_problem("z4c/z4c_one_puncture")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
INPUT = os.path.join(REPO, "tst", "inputs", "z4c_op_trilinear_pytest.athinput")
MODES = ("none", "mirror", "consistent")
OUTER_TAG0 = 32          # <problem> prtcl_np in the deck; the pgen tags ring 1 first

REJECT_RE = re.compile(r"update rejected: rank (\d+) cycle (\d+): (\d+) of (\d+) "
                       r"particles kept their step-n state \(geodesic (\d+), "
                       r"write-back (\d+)")
FALLBACK_RE = re.compile(r"trilinear fallback: rank (\d+) cycle (\d+): (\d+) attempted, "
                         r"(\d+) succeeded \((\d+) converged, (\d+) Euler\), "
                         r"(\d+) still rejected; grid corners invalid in (\d+), "
                         r"non-finite in (\d+)")


def _rejects_by_cycle(log):
    """cycle -> [geodesic rejections, write-back refusals], summed over ranks."""
    out = {}
    for m in REJECT_RE.findall(log):
        c = int(m[1])
        row = out.setdefault(c, [0, 0])
        row[0] += int(m[4])
        row[1] += int(m[5])
    return out


def _fallbacks_by_cycle(log):
    """cycle -> [attempted, succeeded, still rejected, grid invalid, grid non-finite]."""
    out = {}
    for m in FALLBACK_RE.findall(log):
        c = int(m[1])
        row = out.setdefault(c, [0, 0, 0, 0, 0])
        for i, g in enumerate((2, 3, 6, 7, 8)):
            row[i] += int(m[g])
    return out


def _outer_ring(run_dir):
    """Final position and velocity of the well-resolved ring, keyed by tag.

    Note the comparison below is at the precision of the particle VTK, which stores
    float32; "bitwise" here means bit-for-bit in that output, not in the Real the pusher
    computed. That is still the right check for "the flag changed nothing", because a
    perturbed push would move a particle by far more than a float32 ulp.
    """
    path = helpers.particle_dumps(run_dir)[-1]
    _, x, v, tag = helpers.read_particle_vtk(path)
    keep = tag >= OUTER_TAG0
    assert keep.sum() > 0, (
        f"{run_dir}: no particle with tag >= {OUTER_TAG0} survived, so the "
        f"well-resolved control ring is empty and the comparison below is vacuous"
    )
    order = np.argsort(tag[keep])
    return x[keep][order], v[keep][order], tag[keep][order]


def test_trilinear_fallback():
    runs = {}
    dirs = {}
    try:
        for mode in MODES:
            run_dir = f"prt_trilinear_{mode}"
            dirs[mode] = run_dir
            helpers.remove_dirs(run_dir)
            runs[mode] = helpers.run_case(
                INPUT, run_dir, 1,
                extra_args=[f"particles/trilinear_fallback={mode}"],
            )

        for mode, log in runs.items():
            assert "nan" not in log.lower(), f"{mode}: nan in the log"
            assert "FATAL" not in log, f"{mode}: fatal error"
            assert all(v[1] == 0 for v in _rejects_by_cycle(log).values()), (
                f"{mode}: a non-finite write-back was refused"
            )

        base_rej = _rejects_by_cycle(runs["none"])

        # 1. mode = none must be reported as doing nothing at all
        assert not _fallbacks_by_cycle(runs["none"]), (
            "trilinear_fallback = none must not attempt a retry"
        )
        assert "trilinear fallback" not in runs["none"], (
            "trilinear_fallback = none must not emit the fallback summary"
        )

        # 2. the well-resolved ring is untouched by the flag in every mode
        ref = _outer_ring(dirs["none"])
        for mode in ("mirror", "consistent"):
            got = _outer_ring(dirs[mode])
            assert np.array_equal(ref[2], got[2]), f"{mode}: outer-ring tags differ"
            assert np.array_equal(ref[0], got[0]), (
                f"{mode}: outer-ring positions differ -- the fallback "
                f"perturbed a push the high-order path accepted"
            )
            assert np.array_equal(ref[1], got[1]), (
                f"{mode}: outer-ring velocities differ"
            )

        if not base_rej:
            # no rejection anywhere: then NOTHING may change, for either mode
            for mode in ("mirror", "consistent"):
                assert not _fallbacks_by_cycle(runs[mode]), (
                    f"{mode}: retried a substep the high-order path did not reject"
                )
                helpers.assert_final_particle_state_bitwise(dirs["none"], dirs[mode])
            return

        first_rej_cycle = min(base_rej)
        for mode in ("mirror", "consistent"):
            fb = _fallbacks_by_cycle(runs[mode])
            rej = _rejects_by_cycle(runs[mode])
            assert fb, f"{mode}: the production run rejected but no retry was attempted"

            # 3a. per cycle, inside one run, the census must add up and must agree with
            #     the rejection summary printed by the same run
            for c, (tried, ok, still, bad_grid, nonfinite_grid) in fb.items():
                assert ok + still == tried, f"{mode}: cycle {c} census does not add up"
                assert rej.get(c, [0, 0])[0] == still, (
                    f"{mode}: cycle {c} reports {rej.get(c, [0, 0])[0]} rejections for "
                    f"{still} failed retries"
                )
                assert bad_grid <= tried and nonfinite_grid <= tried

            # 3b. at the first cycle that reports anything the two runs are still
            #     identical, so every rejection there must have been retried exactly once
            assert min(fb) == first_rej_cycle, (
                f"{mode}: first retry at cycle {min(fb)}, first production rejection at "
                f"cycle {first_rej_cycle} -- the flag cannot act before the first "
                f"rejection, so these must coincide"
            )
            assert fb[first_rej_cycle][0] == base_rej[first_rej_cycle][0], (
                f"{mode}: {fb[first_rej_cycle][0]} retries for "
                f"{base_rej[first_rej_cycle][0]} rejections at the first affected cycle"
            )
            assert "TRILINEAR geodesic fallback for the first time" in runs[mode]
    finally:
        helpers.remove_dirs(*dirs.values())
