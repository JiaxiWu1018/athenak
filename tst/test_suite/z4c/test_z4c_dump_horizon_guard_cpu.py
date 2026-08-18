"""Requesting more horizon dumps than compact-object trackers must fail cleanly.

Z4c::DumpHorizons centres phorizon_dump[i] on ptracker[i], but the two vectors are sized
by independent input parameters (<z4c> dump_horizon_<n> and <z4c> co_<n>_type). A
mismatch used to index ptracker out of bounds and segfault inside DumpHorizons; it must
now be rejected at setup with an actionable message.
"""

import os
import subprocess


def test_horizon_dump_tracker_mismatch_is_rejected():
    """Two dump_horizon_<n> with one co_<n>_type: clean exit, no segfault."""
    process = subprocess.Popen(
        ["./athena", "-i", "inputs/z4c_dump_horizon_mismatch_pytest.athinput"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    output = process.communicate()[0]

    # a segfault/abort would give a negative return code (signal); require a clean exit
    assert process.returncode > 0, (
        f"expected a clean nonzero exit, got {process.returncode} (a negative code means "
        f"the process died on a signal)\n{output[-2000:]}"
    )
    assert "FATAL ERROR" in output, f"no fatal diagnostic emitted:\n{output[-2000:]}"
    assert "dump_horizon" in output and "co_<n>_type" in output, (
        f"diagnostic does not explain the pairing:\n{output[-2000:]}"
    )
    assert "Segmentation fault" not in output, f"still segfaulting:\n{output[-2000:]}"
    assert not os.path.exists("horizon_1"), (
        "a horizon output directory was created for a dump with no tracker"
    )
