"""
Unit tests for the trilinear particle-interpolation kernels used by the gr_boris
low-order geodesic fallback.

The checks live in src/pgen/unit_tests/trilinear_interp.cpp so that they run in device
kernels against the same code the pusher calls; that pgen exits non-zero on the first
failure, so running it is the assertion.
"""

# Modules
import test_suite.testutils as testutils


def test_trilinear_interp():
    input_file = "inputs/ut_trilinear_interp.athinput"
    testutils.run(input_file)
