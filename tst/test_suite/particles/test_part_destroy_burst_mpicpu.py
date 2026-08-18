"""Death-ledger regression for the grow-only death-record arrays.

A 32^3 lattice drains through outflow boundaries, producing a >1000-particle destruction
burst followed by dozens of progressively SMALLER events. That is exactly the sequence
that leaves nloc < capacity in Particles::FlushDeathLog, which used to abort HIP/CUDA
runs (a column subview of the (7, cap) LayoutRight record arrays is non-contiguous, so it
cannot be mirrored to the host) and which must never leak the unused capacity tail into
the CSV on any backend.

Run from ``tst`` with::

    python3 run_test_suite.py \
        --mpicpu "-D PROBLEM=particles/part_crossing" \
        --test test_suite/particles/test_part_destroy_burst_mpicpu.py
"""

import collections
import os

import test_suite.particles.part_helpers as helpers


helpers.require_problem("particles/part_crossing")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
INPUT = os.path.join(REPO, "tst", "inputs", "part_destroy_burst_pytest.athinput")
NPART = 32 * 32 * 32


def _census(run_dir):
    rows = helpers.read_death_csv(
        os.path.join(run_dir, "part_destroy_burst.prtcl_destroy.csv")
    )
    return rows, collections.Counter(row["cycle"] for row in rows)


def test_death_ledger_survives_shrinking_events():
    """Many events of decreasing size must leave an exact, padding-free death ledger."""
    ranks = helpers.rank_counts((1, 2))
    run_dirs = {count: f"prt_destroy_burst_np{count}" for count in ranks}
    census = {}
    try:
        for count in ranks:
            helpers.remove_dirs(run_dirs[count])
            log = helpers.run_case(INPUT, run_dirs[count], count)
            rows, per_cycle = _census(run_dirs[count])
            census[count] = rows

            # the run must have destroyed a lot, but not everything
            assert len(rows) > 10000, f"only {len(rows)} deaths; lattice did not drain"
            assert len(per_cycle) >= 10, f"only {len(per_cycle)} destruction events"

            # a >1000-particle burst followed by a strictly smaller LATER event is what
            # drives nloc < capacity; assert the reproducer really has that shape
            biggest_cycle = max(per_cycle, key=per_cycle.get)
            assert per_cycle[biggest_cycle] > 1000, (
                f"largest event only {per_cycle[biggest_cycle]} particles"
            )
            later_smaller = [
                c for c in per_cycle
                if c > biggest_cycle and per_cycle[c] < per_cycle[biggest_cycle]
            ]
            assert later_smaller, "no smaller event after the burst: nloc < cap untested"

            # ledger closes exactly and no capacity padding leaked into the CSV
            tags = [row["tag"] for row in rows]
            assert len(set(tags)) == len(tags), "a destroyed tag appears twice"
            assert all(0 <= tag < NPART for tag in tags), "tag outside the lattice range"
            assert all(row["reason"] == "exit" for row in rows), (
                f"unexpected destruction reasons: "
                f"{sorted({row['reason'] for row in rows})}"
            )
            assert f"destroyed: exit={len(rows)} " in log, (
                "in-code census disagrees with the CSV row count"
            )
            assert "[conservation OK]" in log, "particle-number conservation violated"

            # every destroyed particle must have left the domain
            for row in rows:
                assert max(abs(row["x"]), abs(row["y"]), abs(row["z"])) > 0.5, (
                    f"tag {row['tag']} destroyed inside the domain at "
                    f"({row['x']}, {row['y']}, {row['z']})"
                )

        # the death set is a physical result: identical across MPI decompositions
        for count in ranks[1:]:
            reference = {row["tag"]: helpers.death_key(row) for row in census[ranks[0]]}
            candidate = {row["tag"]: helpers.death_key(row) for row in census[count]}
            assert candidate == reference, "death records differ across rank counts"
    finally:
        helpers.remove_dirs(*run_dirs.values())
