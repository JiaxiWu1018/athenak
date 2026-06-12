"""Shared setup for the particles pytest tier.

Makes the tests runnable from BOTH entry points:
  - the harness (run_test_suite.py), which imports testutils from tst/ and runs pytest
    with cwd tst/build/src -- everything here is then a no-op;
  - a standalone `pytest` during development (tst/build prebuilt), where two
    cwd-at-import-time dependences inside testutils.py must be satisfied first:
    sys.path.insert(0, "../vis/python") and the eagerly-opened log file at
    abspath("../tst/test_log.txt").
"""
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_TST = os.path.join(_REPO, "tst")
_BUILD_SRC = os.path.join(_TST, "build", "src")

# absolute-path anchors so imports survive any launch cwd
for _p in (os.path.join(_REPO, "vis", "python"), _TST):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# import testutils with cwd at tst/ (no-op if the harness already imported it)
_cwd = os.getcwd()
os.chdir(_TST)
try:
    import test_suite.testutils  # noqa: F401
finally:
    os.chdir(_cwd)

import pytest  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _anchor_cwd():
    """Run from tst/build/src (./athena + the inputs/ symlink) regardless of how
    pytest was launched; restore afterwards. No-op under the harness."""
    old = os.getcwd()
    if not os.path.exists("./athena") and os.path.exists(
            os.path.join(_BUILD_SRC, "athena")):
        os.chdir(_BUILD_SRC)
    yield
    os.chdir(old)
