"""MPI regression for rank-invariant particle stress-energy deposition.

Run from tst with:

    python3 run_test_suite.py \
        --mpicpu "-D PROBLEM=particles/part_tmunu_test" \
        --test test_suite/particles/test_part_tmunu_mpicpu.py
"""

import filecmp
import glob
import os

import test_suite.particles.part_helpers as helpers


helpers.require_problem("particles/part_tmunu_test")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FLAT = os.path.join(REPO, "inputs", "particles", "part_tmunu_flat.athinput")
SMR = os.path.join(REPO, "tst", "inputs", "part_tmunu_smr_pytest.athinput")
W2 = ["problem/pux=1.0", "problem/puy=1.0", "problem/puz=1.0"]


def _tmunu_dumps(run_dir):
    paths = sorted(glob.glob(os.path.join(run_dir, "bin", "*.tmunu.*.bin")))
    assert paths, f"no Tmunu binary output in {run_dir}"
    return paths


def _assert_tmunu_bitwise(first_dir, second_dir):
    first = _tmunu_dumps(first_dir)
    second = _tmunu_dumps(second_dir)
    assert len(first) == len(second), "Tmunu dump counts differ"
    for first_path, second_path in zip(first, second):
        assert filecmp.cmp(first_path, second_path, shallow=False), (
            f"Tmunu differs: {os.path.basename(first_path)}"
        )


def _run_rank_invariance(input_file, prefix, extra_args):
    ranks = helpers.rank_counts((1, 2, 4))
    run_dirs = {count: f"{prefix}_np{count}" for count in ranks}
    try:
        for count in ranks:
            helpers.remove_dirs(run_dirs[count])
            helpers.run_case(
                input_file, run_dirs[count], count, extra_args=W2 + extra_args
            )
        for count in ranks[1:]:
            _assert_tmunu_bitwise(run_dirs[ranks[0]], run_dirs[count])
    finally:
        helpers.remove_dirs(*run_dirs.values())


def test_periodic_sweep_is_rank_invariant():
    """Exercise faces, edges, corners, empty ranks, and periodic image transport."""
    _run_rank_invariance(
        FLAT,
        "tmunu_periodic",
        [
            "problem/mode=sweep",
            "problem/sweep_x1=-0.5",
            "problem/sweep_x2=-0.5",
            "problem/sweep_x3=-0.5",
        ],
    )


def test_static_refinement_corner_is_rank_invariant():
    """Exercise seven same-level images at a refined-block corner."""
    _run_rank_invariance(SMR, "tmunu_smr", [])
