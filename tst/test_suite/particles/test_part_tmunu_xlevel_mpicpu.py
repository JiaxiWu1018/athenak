"""MPI rank-invariance regressions for target-native cross-level Tmunu deposition."""

import filecmp
import glob
import os

import pytest

import test_suite.particles.part_helpers as helpers


helpers.require_problem("particles/part_tmunu_test")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SMR = os.path.join(REPO, "tst", "inputs", "part_tmunu_smr_pytest.athinput")
SMR_HALF = os.path.join(REPO, "tst", "inputs", "part_tmunu_smr_half_pytest.athinput")
W2 = ["problem/pux=1.0", "problem/puy=1.0", "problem/puz=1.0"]

CASES = [
    # Fine-to-coarse and coarse-to-fine face crossings exercise native clipping/fanout.
    ("fine_to_coarse", SMR, ["problem/px=0.0078125", "problem/py=0.25", "problem/pz=0.25"]),
    ("coarse_to_fine", SMR, ["problem/px=-0.0078125", "problem/py=0.25", "problem/pz=0.25"]),
    # Two- and three-dimensional seams exercise distinct cross-level edge/corner targets.
    ("edge", SMR, ["problem/px=0.0078125", "problem/py=0.0078125", "problem/pz=0.25"]),
    ("corner", SMR, ["problem/px=0.0078125", "problem/py=0.0078125", "problem/pz=0.0078125"]),
    # The half-domain geometry demotes a diagonal onto an already targeted coarse face.
    ("demotion_dedup", SMR_HALF,
     ["problem/px=-0.0078125", "problem/py=0.4921875", "problem/pz=0.25"]),
]


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


@pytest.mark.parametrize("case,input_file,position", CASES, ids=[case[0] for case in CASES])
def test_cross_level_tmunu_is_rank_invariant(case, input_file, position):
    """Cross-level routing must exit cleanly and produce identical Tmunu for np=1/2/4."""
    ranks = helpers.rank_counts((1, 2, 4))
    run_dirs = {count: f"tmunu_xlevel_{case}_np{count}" for count in ranks}
    try:
        for count in ranks:
            helpers.remove_dirs(run_dirs[count])
            helpers.run_case(
                input_file,
                run_dirs[count],
                count,
                extra_args=["problem/mode=single"] + position + W2,
            )
        for count in ranks[1:]:
            _assert_tmunu_bitwise(run_dirs[ranks[0]], run_dirs[count])
    finally:
        helpers.remove_dirs(*run_dirs.values())
