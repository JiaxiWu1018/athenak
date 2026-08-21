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

The deck puts one ring at r = 0.3 on a dx = 1/3 grid, i.e. inside a single cell of the
trumpet, which is the regime that produces the rejections. Run from ``tst`` with::

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

REJECT_RE = re.compile(r"update rejected: rank (\d+) cycle (\d+): (\d+) of (\d+) "
                       r"particles kept their step-n state \(geodesic (\d+), "
                       r"write-back (\d+)")
FALLBACK_RE = re.compile(r"trilinear fallback: rank (\d+) cycle (\d+): (\d+) attempted, "
                         r"(\d+) succeeded \((\d+) converged, (\d+) Euler\), "
                         r"(\d+) still rejected; grid corners invalid in (\d+), "
                         r"non-finite in (\d+)")


def _rejects(log):
    """(geodesic rejections, write-back refusals) summed over every reported cycle."""
    geo = sum(int(m[4]) for m in REJECT_RE.findall(log))
    out = sum(int(m[5]) for m in REJECT_RE.findall(log))
    return geo, out


def _fallbacks(log):
    """(attempted, succeeded, still rejected, grid-invalid, grid-non-finite)."""
    rows = FALLBACK_RE.findall(log)
    cols = [sum(int(r[i]) for r in rows) for i in (2, 3, 6, 7, 8)]
    return tuple(cols)


def _outer_ring(run_dir):
    """Final position and velocity of the well-resolved ring, keyed by tag.

    The pgen tags ring 1 first, so tags >= prtcl_np belong to the outer ring.
    """
    path = helpers.particle_dumps(run_dir)[-1]
    _, x, v, tag = helpers.read_particle_vtk(path)
    keep = tag >= 32
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
            assert _rejects(log)[1] == 0, f"{mode}: a non-finite write-back was refused"

        base_geo, _ = _rejects(runs["none"])

        # 1. mode = none must be reported as doing nothing at all
        assert _fallbacks(runs["none"]) == (0, 0, 0, 0, 0), (
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
                f"{mode}: outer-ring positions are not bitwise equal -- the fallback "
                f"perturbed a push the high-order path accepted"
            )
            assert np.array_equal(ref[1], got[1]), (
                f"{mode}: outer-ring velocities are not bitwise equal"
            )

        if base_geo == 0:
            # no rejection anywhere: then NOTHING may change, for either mode
            for mode in ("mirror", "consistent"):
                assert _fallbacks(runs[mode])[0] == 0, (
                    f"{mode}: retried a substep the high-order path did not reject"
                )
                helpers.assert_final_particle_state_bitwise(dirs["none"], dirs[mode])
            return

        # 3. every rejection of the production run is retried, counted and accounted for
        for mode in ("mirror", "consistent"):
            tried, ok, still, bad_grid, nonfinite_grid = _fallbacks(runs[mode])
            assert tried == base_geo, (
                f"{mode}: {tried} retries for {base_geo} production rejections -- the "
                f"retry must be entered exactly once per rejected substep"
            )
            assert ok + still == tried, f"{mode}: fallback census does not add up"
            assert _rejects(runs[mode])[0] == still, (
                f"{mode}: reported rejections do not match the failed retries"
            )
            assert bad_grid <= tried and nonfinite_grid <= tried
            assert "trilinear geodesic fallback for the first time" in runs[mode]
    finally:
        helpers.remove_dirs(*dirs.values())
