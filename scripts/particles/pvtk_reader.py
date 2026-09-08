#!/usr/bin/env python3
"""Self-contained reader for AthenaK legacy-VTK particle dumps (`pvtk/*.part.vtk`).

numpy only.  The layout is fixed by src/outputs/vtk_prtcl.cpp:

    # vtk DataFile Version 2.0
    # AthenaK particle data at time= T  nranks= R  cycle=C  variables=V
    BINARY
    DATASET UNSTRUCTURED_GRID
    <LF>POINTS N float<LF>                       3*N  big-endian float32
    <LF><LF>POINT_DATA N<LF>
    <LF>SCALARS gid float<LF>LOOKUP_TABLE default<LF>       N  float32
    <LF>SCALARS ptag float<LF>LOOKUP_TABLE default<LF>      N  float32
    <LF>VECTORS prtcl_vel float<LF>                        3*N float32   (covariant u_i)
    [ <LF>VECTORS prtcl_du_dt float<LF>                    3*N float32 ]  gr_boris_diagnostics
    [ <LF>VECTORS prtcl_dL_dt float<LF>                    3*N float32 ]
    [ <LF>VECTORS prtcl_raw_live_du_dt ... , prtcl_raw_live_dL_dt ... ]   live_monopole only
    <LF>SCALARS prtcl_energy float<LF>LOOKUP_TABLE default<LF>   N float32   (-u_t)
    <LF>SCALARS prtcl_mass  float<LF>LOOKUP_TABLE default<LF>    N float32

Every numeric block is big-endian float32.  Rather than trusting the offsets we
scan for the ASCII section markers, so the reader survives an extra optional
block being present or absent.
"""
import re
import numpy as np

_MARK = re.compile(rb"\n(POINTS|POINT_DATA|SCALARS|VECTORS|LOOKUP_TABLE)[^\n]*\n")


def _sections(buf):
    """Yield (kind, name, ncomp, data_start) for every binary block, in file order."""
    out = []
    pos = 0
    n = None
    while True:
        m = _MARK.search(buf, pos)
        if m is None:
            break
        line = m.group(0).strip().decode("ascii")
        tok = line.split()
        kind = tok[0]
        pos = m.end()
        if kind == "POINTS":
            n = int(tok[1])
            out.append(("POINTS", "pos", 3, pos))
        elif kind == "POINT_DATA":
            n = int(tok[1])
        elif kind == "VECTORS":
            out.append(("VECTORS", tok[1], 3, pos))
        elif kind == "SCALARS":
            # the binary payload starts after the following LOOKUP_TABLE line
            m2 = _MARK.search(buf, pos - 1)
            if m2 is None or not m2.group(0).strip().startswith(b"LOOKUP_TABLE"):
                raise ValueError("SCALARS %s not followed by LOOKUP_TABLE" % tok[1])
            out.append(("SCALARS", tok[1], 1, m2.end()))
            pos = m2.end()
        # LOOKUP_TABLE handled above
    return n, out


def read_pvtk(path, want=None):
    """Read a particle vtk dump.

    Returns dict with 'time', 'cycle', 'nranks', 'n', and one entry per block:
    'pos' (n,3), 'gid' (n,), 'ptag' (n,), 'prtcl_vel' (n,3), ... as float32/int64.
    `want` optionally restricts which named blocks are materialised (saves RAM);
    'pos' and 'ptag' are always read.
    """
    with open(path, "rb") as f:
        buf = f.read()
    head_end = buf.find(b"\nPOINTS ")
    if head_end < 0:
        raise ValueError("not an AthenaK particle vtk file: %s" % path)
    head = buf[:head_end].decode("ascii", "replace")
    mt = re.search(r"time=\s*([-+0-9.eE]+)", head)
    mc = re.search(r"cycle=\s*([0-9]+)", head)
    mr = re.search(r"nranks=\s*([0-9]+)", head)
    n, secs = _sections(buf)
    res = {
        "time": float(mt.group(1)) if mt else float("nan"),
        "cycle": int(mc.group(1)) if mc else -1,
        "nranks": int(mr.group(1)) if mr else -1,
        "n": n,
        "path": path,
        "blocks": [s[1] for s in secs],
    }
    always = {"pos", "ptag"}
    for kind, name, ncomp, start in secs:
        if want is not None and name not in want and name not in always:
            continue
        cnt = n * ncomp
        a = np.frombuffer(buf, dtype=">f4", count=cnt, offset=start)
        a = a.astype(np.float32) if ncomp == 1 else a.astype(np.float32).reshape(n, 3)
        res[name] = a
    if "ptag" in res:
        # ptag is written through float32; exact for |tag| < 2^24 only.
        res["ptag_f"] = res["ptag"]
        res["ptag"] = res["ptag"].astype(np.int64)
    if "gid" in res:
        res["gid"] = res["gid"].astype(np.int64)
    return res


def ptag_is_exact(n):
    """float32 stores integers exactly only up to 2**24 = 16777216."""
    return n <= 2 ** 24
