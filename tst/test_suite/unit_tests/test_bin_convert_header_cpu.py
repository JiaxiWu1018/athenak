"""Regression test for binary-header parsing in vis/python/bin_convert.py.

An AthenaK binary dump embeds the full input file in its header. A parameter whose
VALUE legitimately contains '=' (for example the <comment> line
"reference = gr_boris pusher validation (q=0 geodesic limit), spin a=0, mass M=1" in
inputs/particles/part_schwarzschild.athinput, or "configure = -b --prob=shock_tube" in
tst/inputs/rj2a.athinput) must not break the parser: splitting on every '=' raises
"ValueError: too many values to unpack (expected 2)". The parser therefore has to split
on the FIRST '=' only.
"""

import os
import struct
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if os.path.join(_REPO, "vis", "python") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "vis", "python"))

import bin_convert  # noqa: E402


def _write_minimal_binary(path, extra_comment):
    """Write a single-MeshBlock Athena binary-v1.1 file whose embedded input deck
    contains `extra_comment`, a parameter line with more than one '='."""
    deck = "\n".join([
        "<comment>",
        "problem = header parsing regression",
        extra_comment,
        "<mesh>",
        "nghost = 2",
        "nx1 = 4",
        "x1min = -0.5",
        "x1max = 0.5",
        "nx2 = 4",
        "x2min = -0.5",
        "x2max = 0.5",
        "nx3 = 4",
        "x3min = -0.5",
        "x3max = 0.5",
        "<meshblock>",
        "nx1 = 4",
        "nx2 = 4",
        "nx3 = 4",
        "",
    ]).encode()

    pre = [b"time=0.0", b"cycle=0", b"size of location=8", b"size of variable=8"]
    with open(path, "wb") as fp:
        fp.write(b"Athena binary output version=1.1\n")
        fp.write(f"  size of preheader={len(pre) + 1}\n".encode())
        for line in pre:
            fp.write(b"  " + line + b"\n")
        fp.write(b"  number of variables=1\n")
        fp.write(b"  variables:  dens\n")
        fp.write(f"  header offset={len(deck)}\n".encode())
        fp.write(deck)
        # one MeshBlock: index range, logical location, geometry, then the payload
        fp.write(struct.pack("=6i", 0, 3, 0, 3, 0, 3))
        fp.write(struct.pack("=4i", 0, 0, 0, 0))
        fp.write(struct.pack("=6d", -0.5, 0.5, -0.5, 0.5, -0.5, 0.5))
        fp.write(struct.pack("=64d", *[float(i) for i in range(64)]))


def test_bin_convert_header_with_multiple_equals(tmp_path):
    """read_binary must parse a header whose parameter value contains '='."""
    path = str(tmp_path / "multi_equals.bin")
    _write_minimal_binary(
        path,
        "reference = gr_boris validation (q=0 geodesic limit), spin a=0, mass M=1",
    )
    data = bin_convert.read_binary(path)

    assert data["Nx1"] == 4 and data["Nx2"] == 4 and data["Nx3"] == 4
    assert data["x1min"] == -0.5 and data["x1max"] == 0.5
    assert data["n_mbs"] == 1
    assert data["var_names"] == ["dens"]
    # the payload must survive intact
    assert data["mb_data"]["dens"][0].size == 64
    assert data["mb_data"]["dens"][0].flatten()[0] == 0.0
    assert data["mb_data"]["dens"][0].flatten()[-1] == 63.0
    # the multi-'=' value must be retained whole, not truncated at the second '='
    ref = [ln for ln in data["header"] if ln.startswith("reference")]
    assert len(ref) == 1, data["header"]
    assert "q=0" in ref[0] and "a=0" in ref[0] and "M=1" in ref[0]
