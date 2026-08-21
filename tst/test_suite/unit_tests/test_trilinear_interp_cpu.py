"""
Unit tests for the trilinear particle-interpolation kernels used by the gr_boris
low-order geodesic fallback.

The checks live in src/pgen/unit_tests/trilinear_interp.cpp so that they run in device
kernels against the same code the pusher calls; that pgen exits non-zero on the first
failure, so running it is the assertion.
"""

# Modules
import os
import re

import pytest

import test_suite.testutils as testutils

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_CACHE = os.path.join(_REPO, "tst", "build", "CMakeCache.txt")

# <problem> pgen_name is short-circuited by USER_PROBLEM_ENABLED, so in a -D PROBLEM=...
# build this deck silently runs that problem generator instead of the unit test. Skip
# rather than pass vacuously.
if os.path.exists(_CACHE):
    for _line in open(_CACHE):
        _m = re.match(r"PROBLEM:\w+=(.*?)\s*$", _line)
        if _m and _m.group(1) != "built_in_pgens":
            pytest.skip(
                f"needs the default built_in_pgens build "
                f"(cache has PROBLEM={_m.group(1)})",
                allow_module_level=True,
            )


def test_trilinear_interp():
    input_file = "inputs/ut_trilinear_interp.athinput"
    testutils.run(input_file)
