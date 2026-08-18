"""The gr_boris non-convergence diagnostic must be bounded, not O(N_particle).

When the implicit geodesic substep does not converge, the step falls back to forward
Euler. That is a legitimate, documented first-order fallback (it appears at large CFL and
disappears as dt is reduced), so it MUST stay visible -- but warning once per particle per
cycle produced one log line per failure, which for a large ensemble means hundreds of
millions of lines. The kernel now counts every failure and prints detail for at most
Particles::kBorisDetail of them per cycle, with one summary line per rank per cycle.

Run from ``tst`` with::

    python3 run_test_suite.py \
        --mpicpu "-D PROBLEM=z4c/z4c_one_puncture" \
        --test test_suite/particles/test_part_boris_warn_mpicpu.py
"""

import os
import re

import test_suite.particles.part_helpers as helpers


helpers.require_problem("z4c/z4c_one_puncture")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
INPUT = os.path.join(REPO, "tst", "inputs", "z4c_op_amr_fb_pytest.athinput")
DETAIL_BUDGET = 3          # Particles::kBorisDetail


def _counts(log):
    """Return (failures reported, gr_boris log lines, detailed lines, summary lines)."""
    totals = [int(v) for v in re.findall(r"rank total (\d+)", log)]
    summaries = re.findall(
        r"non-convergence: rank (\d+) cycle (\d+): (\d+) of (\d+)", log
    )
    return (
        max(totals) if totals else 0,
        len([ln for ln in log.splitlines() if "gr_boris" in ln]),
        len(re.findall(r"forward-Euler fallback used", log)),
        summaries,
    )


def test_boris_fallback_diagnostic_is_bounded():
    """More particles must mean more counted failures but NOT more log output."""
    runs = {}
    dirs = {}
    try:
        # prtcl_np is per ring; the deck has two rings, so the ensemble is 2 x prtcl_np
        for prtcl_np in (16, 64):
            run_dir = f"prt_boriswarn_np{prtcl_np}"
            dirs[prtcl_np] = run_dir
            helpers.remove_dirs(run_dir)
            runs[prtcl_np] = helpers.run_case(
                INPUT, run_dir, 1, extra_args=[f"problem/prtcl_np={prtcl_np}"]
            )

        small = _counts(runs[16])
        large = _counts(runs[64])

        # the fallback must still be REPORTED, not silently swallowed
        assert small[0] > 0, "no gr_boris fallback reported; the diagnostic is hidden"
        assert small[3], "no per-cycle summary line emitted"

        # every failure is counted: 4x the particles must give 4x the failures
        assert large[0] == 4 * small[0], (
            f"failure count did not scale with the ensemble: {small[0]} -> {large[0]}"
        )

        # ...while the LOG SIZE stays bounded and does not grow with particle number
        assert large[1] == small[1], (
            f"log output grew with particle count: {small[1]} -> {large[1]} lines"
        )
        assert large[2] == small[2], (
            f"detailed lines grew with particle count: {small[2]} -> {large[2]}"
        )
        assert large[1] < large[0], (
            f"log lines ({large[1]}) not fewer than failures ({large[0]}): unbounded"
        )

        # detail is capped per cycle, and the summary accounts for all of them
        per_cycle = {}
        for _, cycle, nfail, _ in large[3]:
            per_cycle[int(cycle)] = int(nfail)
        assert sum(per_cycle.values()) == large[0], (
            "per-cycle summaries do not add up to the reported total"
        )
        assert large[2] <= DETAIL_BUDGET * len(per_cycle), (
            f"detailed lines {large[2]} exceed the per-cycle budget "
            f"{DETAIL_BUDGET} x {len(per_cycle)}"
        )

        # the diagnostic must remain USEFUL: first occurrence explains itself, and the
        # detailed lines carry enough particle state to debug the failure
        assert "did not converge for the first time this run" in runs[16]
        assert "forward-Euler" in runs[16]
        detail = re.search(
            r"forward-Euler fallback used \| tag=(-?\d+) gid=(-?\d+) "
            r"x=\(\s*([-0-9.e+]+),\s*([-0-9.e+]+),\s*([-0-9.e+]+)\) "
            r"u_i=\(\s*([-0-9.e+]+),\s*([-0-9.e+]+),\s*([-0-9.e+]+)\) dt=([-0-9.e+]+)",
            runs[16],
        )
        assert detail is not None, "detailed line lost its particle state"
        assert int(detail.group(1)) >= 0, "tag missing from the detailed line"
        assert float(detail.group(9)) > 0.0, "dt missing from the detailed line"
    finally:
        helpers.remove_dirs(*dirs.values())
