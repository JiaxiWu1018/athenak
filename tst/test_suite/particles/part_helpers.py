"""Helpers for the particles pytest tier (no test_ prefix -> never collected).

Ports of the proven parsers/checks from the NRPIC Stage-3 research tooling
(analyze_crossing.py): binary particle-vtk reader, dump selection by header time,
death-record CSV reader, and assert-style wrappers (per-tag bitwise comparison,
analytic drift check, death-ledger invariants and per-dump set algebra).

All assertions are made on artifacts (exit codes, pvtk dumps, the death CSV) --
testutils funnels every run's stdout into one shared log file, so stdout is not
usable for per-run checks. With <particles> debug=1 the in-code per-cycle validator
makes any migration/conservation violation fatal, so a nonzero exit code (run_args
raises) is the per-cycle oracle.
"""
import glob
import os
import re
import shutil
import struct

import numpy as np
import pytest

import test_suite.testutils as testutils

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


# ------------------------------------------------------------------ build / skip guard
def require_problem(name):
    """Module-level guard: SKIP unless tst/build was configured with -DPROBLEM=<name>.

    The particle pgens are user problems (not in the built-in dispatch), so these
    tests need a dedicated build/invocation, e.g.
        python3 run_test_suite.py --mpicpu "-D PROBLEM=<name>" \\
            --test test_suite/particles/<test_file>
    In the stock full-suite run (PROBLEM=built_in_pgens) the module skips cleanly.
    """
    cache = os.path.join(_REPO, "tst", "build", "CMakeCache.txt")
    prob = None
    if os.path.exists(cache):
        with open(cache) as f:
            for line in f:
                m = re.match(r"PROBLEM:\w+=(.*?)\s*$", line)
                if m:
                    prob = m.group(1)
    if prob != name:
        pytest.skip(f"needs a -DPROBLEM={name} build (cache has PROBLEM={prob})",
                    allow_module_level=True)


# ------------------------------------------------------------------------- run wrapper
def run_args(args, threads=1):
    """mpirun -np <threads> ./athena <args>, via testutils.run_command (shared log).

    Unlike testutils.run/mpi_run this does not hardcode '-i', so restart runs ('-r')
    and run-dir runs ('-d') are expressible. Raises on nonzero exit.
    """
    cmd = ["mpirun", "-np", str(threads), "./athena"] + args
    if not testutils.run_command(cmd):
        raise RuntimeError(f"athena failed: {' '.join(cmd)}")


def rank_counts(want):
    """Filter requested rank counts to this machine's cores (never oversubscribe)."""
    cap = min(4, os.cpu_count() or 1)
    return [n for n in want if n <= cap]


def rmdirs(*names):
    for n in names:
        shutil.rmtree(n, ignore_errors=True)


# ------------------------------------------------- particle-vtk parsing (proven port)
def read_part_vtk(fp):
    """Read one binary (big-endian) particle vtk: returns (time, pos[N,3], vel[N,3],
    gid[N], tag[N])."""
    with open(fp, "rb") as f:
        b = f.read()
    time = float(re.search(rb"time=\s*([-+0-9.eE]+)", b).group(1))
    m = re.search(rb"POINTS\s+(\d+)\s+float", b)
    n = int(m.group(1))

    def blk(marker, count):
        i = b.find(marker)
        if i < 0:
            pytest.fail(f"{fp}: missing '{marker.decode()}'")
        s = b.find(b"\n", i + len(marker)) + 1
        return np.array(struct.unpack(">%df" % count, b[s:s + 4 * count]))

    pos = blk(m.group(0), 3 * n).reshape(n, 3)
    vel = blk(b"VECTORS prtcl_vel float", 3 * n).reshape(n, 3)
    gid = blk(b"SCALARS gid float\nLOOKUP_TABLE default", n).astype(int)
    tag = blk(b"SCALARS ptag float\nLOOKUP_TABLE default", n).astype(int)
    return time, pos, vel, gid, tag


def list_dumps(d):
    files = sorted(glob.glob(os.path.join(d, "pvtk", "*.part.vtk")))
    if not files:
        files = sorted(glob.glob(os.path.join(d, "*.part.vtk")))
    if not files:
        pytest.fail(f"no part.vtk dumps in {d}")
    return files


def dump_time(fp):
    """Dump time from the vtk header (identical ASCII for equal doubles across runs)."""
    with open(fp, "rb") as f:
        head = f.read(4096)
    m = re.search(rb"time=\s*([-+0-9.eE]+)", head)
    if not m:
        pytest.fail(f"no time= in header of {fp}")
    return float(m.group(1))


def pick_dump(d, sel, other_time=None):
    """Select a dump: first | last | match (= other_time, exact parsed equality).
    Restart runs skip the initial output and their threshold ladder is shifted, so
    file NUMBERS never align with an uninterrupted reference -- restart comparisons
    must match by time."""
    files = list_dumps(d)
    if sel == "last":
        return files[-1]
    if sel == "first":
        return files[0]
    if sel == "match":
        for f in files:
            if dump_time(f) == other_time:
                return f
        pytest.fail(f"no dump in {d} with time {other_time!r}")
    pytest.fail(f"bad pick selector '{sel}'")


# --------------------------------------------------------------- death CSV (proven port)
def read_death_csv(path):
    """Parse one <basename>.prtcl_destroy.csv into a list of dict rows."""
    import csv as _csv
    rows = []
    with open(path) as f:
        for r in _csv.reader(f):
            if not r or r[0].startswith("#"):
                continue
            rows.append(dict(cycle=int(r[0]), time=float(r[1]), tag=int(r[2]),
                             reason=r[3], x=float(r[4]), y=float(r[5]), z=float(r[6]),
                             vx=float(r[7]), vy=float(r[8]), vz=float(r[9]),
                             gid=int(r[10]), rank=int(r[11]), crit=float(r[12])))
    return rows


def death_key(r):
    """Decomposition-invariant identity of a death record. The rank column is
    excluded (CSV row order and ownership are MPI-gather artifacts); same key as the
    research-suite comparator."""
    return (r["cycle"], r["time"], r["reason"], r["x"], r["y"], r["z"],
            r["vx"], r["vy"], r["vz"], r["crit"])


# ---------------------------------------------------------------------------- asserts
def assert_bitwise(fa, fb):
    """Per-tag bitwise position equality between two dumps, plus dump-time equality
    and tag-SET equality (catches re-tagging as well as divergence)."""
    ta, pa, _, _, taga = read_part_vtk(fa)
    tb, pb, _, _, tagb = read_part_vtk(fb)
    assert ta == tb, f"dump times differ: {ta} vs {tb} ({fa} vs {fb})"
    ia, ib = np.argsort(taga), np.argsort(tagb)
    assert np.array_equal(taga[ia], tagb[ib]), f"tag sets differ ({fa} vs {fb})"
    dmax = np.abs(pa[ia] - pb[ib]).max() if len(taga) else 0.0
    assert dmax == 0.0, f"not bitwise: max |dpos| = {dmax} ({fa} vs {fb})"


def assert_last_dumps_bitwise(da, db):
    assert_bitwise(pick_dump(da, "last"), pick_dump(db, "last"))


def assert_analytic_drift(d, xmin=-0.5, xmax=0.5, tol=5e-7):
    """First-vs-last dump: tags unique and conserved, every position equals the
    analytic drift x0 + v*(t1-t0) (periodically wrapped). The drift pusher is exact;
    tol covers the float32 dump precision."""
    files = list_dumps(d)
    assert len(files) >= 2, f"need >=2 dumps in {d}"
    t0, p0, v0, _, tag0 = read_part_vtk(files[0])
    t1, p1, _, _, tag1 = read_part_vtk(files[-1])
    assert len(np.unique(tag0)) == len(tag0), "initial tags not unique"
    assert len(tag0) == len(tag1), f"particle count changed {len(tag0)}->{len(tag1)}"
    i0, i1 = np.argsort(tag0), np.argsort(tag1)
    assert np.array_equal(tag0[i0], tag1[i1]), "tag sets differ first vs last dump"
    L = xmax - xmin
    expect = (p0[i0] + v0[i0] * (t1 - t0) - xmin) % L + xmin
    err = np.abs(p1[i1] - expect)
    err = np.minimum(err, L - err)  # wrap-around distance
    worst = int(np.argmax(err.max(axis=1)))
    assert err.max() <= tol, (
        f"analytic drift violated in {d}: max err {err.max():.3e} > tol {tol:g} "
        f"(tag {tag1[i1][worst]}: pos {p1[i1][worst]} expect {expect[worst]})")


def assert_death_invariants(rows):
    """Within one CSV segment: every tag dies at most once, cycles are monotone."""
    tags = [r["tag"] for r in rows]
    assert len(tags) == len(set(tags)), "a tag died more than once"
    cyc = [r["cycle"] for r in rows]
    assert cyc == sorted(cyc), "death cycles not monotone"


def assert_death_set_algebra(d, rows):
    """Per-dump set algebra over the run's dump series: alive(k) == alive(k-1) minus
    the deaths with time in (t_{k-1}, t_k] -- destroyed particles never reappear and
    no survivor is lost, at any output cadence. (eps absorbs the ulp mismatch between
    the CSV's full-precision time and the dump header's ASCII-rounded time when a
    death lands exactly on a dump boundary.)"""
    t_prev, alive_prev = None, None
    for fp in list_dumps(d):
        t, _, _, _, tag = read_part_vtk(fp)
        alive = set(tag.tolist())
        if alive_prev is not None:
            eps = 1e-12 * (abs(t) + 1.0)
            died = {r["tag"] for r in rows if t_prev + eps < r["time"] <= t + eps}
            want = alive_prev - died
            assert alive == want, (
                f"set algebra broken at t={t}: "
                f"reappeared {sorted(alive - want)[:5]}, "
                f"lost {sorted(want - alive)[:5]}")
        t_prev, alive_prev = t, alive
