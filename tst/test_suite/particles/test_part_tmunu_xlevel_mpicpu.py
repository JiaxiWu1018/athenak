"""Cross-level Tmunu deposition tier (both schemes) -- NRPIC Stage 5b.

Extends the Stage-4c same-level deposit to clouds that straddle a static-SMR coarse-fine
interface, delivered by cross-level images riding the same canonical (target_m, tag,
off_code, lev) stream as the same-level images. A cross-level image is made UNIQUE per
(tag, target gid) at generation (EnumerateParticleTargets). The <particles>
cross_level_deposit flag selects the kernel:

  * native (B, 5b(a)) -- each cross-level image deposits at its TARGET block's own
    resolution (DepositCloudNative). NON-conservative across a seam by construction: the
    per-cycle identity reports a measured O(straddle) residual (closed-form values
    asserted by run_suite_5bb.sh tier t4), NOT a fatal.
  * conservative (A, 5b(b), DEFAULT) -- the whole cloud is deposited at the FINEST level
    it touches and fine cells over coarser leaves are RESTRICTED into the coarse cell
    (DepositCloudRestrict). The identity Sum E sqrt(gamma) dV == Sum m W is recovered
    EXACTLY across the seam, so with debug=1 a residual above tol is FATAL (run_args
    raises on the nonzero exit) -- a clean exit IS the conservation oracle.

Both tiers also assert the decomposition-invariant artifact the harness can check
directly: the deposited Tmunu is BITWISE identical across rank counts (cross-level image
transport + the extended canonical sort + the dedup + the restrict arithmetic are all
rank-count invariant; a dropped/duplicated child, a demotion double-deposit, or a
misrouted restrict would break this).

Needs a build with the part_tmunu_test pgen (see test_part_tmunu_mpicpu.py for the
dedicated invocation); SKIPs cleanly in the stock full-suite run.
"""
import filecmp
import glob
import os

import test_suite.particles.part_helpers as ph

ph.require_problem("part_tmunu_test")

# octant SMR: [0.25,0.75]^3 refines the [0,1]^3 root block (dx_fine = 1/32); coarse-fine
# seams at x1=x2=x3=0. A fine particle at +dx_fine/4 from 1/2/3 seams straddles 1/2/3 ways
# (F->C cross_level 1 / 3 / 7); a coarse particle at -dx_fine/4 (C2F) deposits the whole
# cloud at the fine sublevel (scheme A) / fans out to the fine children (scheme B).
SMR = "inputs/part_tmunu_smr_pytest.athinput"
# half-domain SMR: x1<0 refined, full x2/x3 -> a seam fine block has transverse fine
# siblings, so an x1x2 overhang DEMOTES onto the coarse x1-face -> the dedup is exercised.
SMR_HALF = "inputs/part_tmunu_smr_half_pytest.athinput"
# boundary SMR: +x half refined, y faces OUTFLOW (closed). A coarse -x particle straddling
# x1=0 (cfine: deposited at the fine sublevel) near a closed y boundary -- scheme A's
# boundary clip must be at FINE resolution (regression for the coarse-stencil clip bug).
SMR_BDY = "inputs/part_tmunu_smr_bdy_pytest.athinput"

F2C = ["problem/mode=single", "problem/px=0.0078125",
       "problem/py=0.25", "problem/pz=0.25"]
C2F = ["problem/mode=single", "problem/px=-0.0078125",
       "problem/py=0.25", "problem/pz=0.25"]
EDGE = ["problem/mode=single", "problem/px=0.0078125",
        "problem/py=0.0078125", "problem/pz=0.25"]
CORNER = ["problem/mode=single", "problem/px=0.0078125",
          "problem/py=0.0078125", "problem/pz=0.0078125"]
DEMOTE = ["problem/mode=single", "problem/px=-0.0078125",
          "problem/py=0.4921875", "problem/pz=0.25"]
# coarse -x particle straddling x1=0 (cfine) near a CLOSED y boundary (SMR_BDY): the c->f
# deposit is at fine res, so the closed-y clip is a fine-cell clip. On the buggy code the
# coarse-stencil clip mismatches (particle-side 0.75 vs cell-side 1.0 -> residual 0.25).
BDY_LOWER = ["problem/mode=single", "problem/px=-0.0078125",
             "problem/py=-0.984375", "problem/pz=-0.25"]   # near the lower-y outflow edge
BDY_UPPER = ["problem/mode=single", "problem/px=-0.0078125",
             "problem/py=0.984375", "problem/pz=-0.25"]    # symmetric upper-y edge
W2 = ["problem/pux=1.0", "problem/puy=1.0", "problem/puz=1.0"]  # u_i=(1,1,1): W=2 flat

SCHEME_A = ["particles/cross_level_deposit=conservative"]
SCHEME_B = ["particles/cross_level_deposit=native"]


def _tmunu_dumps(d):
    files = sorted(glob.glob(os.path.join(d, "bin", "*.tmunu.*.bin")))
    assert files, f"no tmunu bin dumps in {d}"
    return files


def _assert_tmunu_bitwise(da, db):
    a, b = _tmunu_dumps(da), _tmunu_dumps(db)
    assert len(a) == len(b), f"dump count differs: {len(a)} ({da}) vs {len(b)} ({db})"
    for fa, fb in zip(a, b):
        assert filecmp.cmp(fa, fb, shallow=False), (
            f"Tmunu not bitwise identical: {os.path.basename(fa)} ({da} vs {db})")


def _np_invariance(tag, inp, overrides):
    """Run inp at np {1,2,4} (W=2) and assert the deposited Tmunu is bitwise identical.
    A nonzero exit (e.g. debug=1 conservation FATAL under scheme A, or a count!=appends /
    duplicate-image FATAL under either scheme) makes run_args raise -- so a passing test
    also certifies the run completed without a deposit error."""
    nps = ph.rank_counts((1, 2, 4))
    dirs = {n: f"xl_{tag}_np{n}" for n in nps}
    try:
        for n in nps:
            ph.rmdirs(dirs[n])
            ph.run_args(["-i", inp, "-d", dirs[n]] + overrides + W2, threads=n)
        for n in nps[1:]:
            _assert_tmunu_bitwise(dirs[nps[0]], dirs[n])
    finally:
        ph.rmdirs(*dirs.values())


# ---------------------------------------------------------------------------------------
# Scheme B (native, 5b(a) regression): non-conservative but smooth + bitwise np-invariant.
# Pinned to cross_level_deposit=native (the default flipped to conservative in 5b(b)).
# ---------------------------------------------------------------------------------------
def test_xlevel_B_fine2coarse_np_invariance():
    """B F->C: a fine-block particle straddling x1=0 deposits its overhang onto the coarse
    neighbor at coarse resolution (one cross-level image). No FATAL; Tmunu bitwise."""
    _np_invariance("B_f2c", SMR, F2C + SCHEME_B)


def test_xlevel_B_coarse2fine_np_invariance():
    """B C->F: a coarse-block particle straddling x1=0 fans out to the fine children it
    spans (up to 4 on a face; unreached ones clip to zero). No FATAL; Tmunu bitwise."""
    _np_invariance("B_c2f", SMR, C2F + SCHEME_B)


def test_xlevel_B_edge_np_invariance():
    """B EDGE: a fine particle straddling TWO octant seams (x1+x2) emits 3 cross-level
    images (2 faces + 1 corner block, all distinct). No FATAL; Tmunu bitwise."""
    _np_invariance("B_edge", SMR, EDGE + SCHEME_B)


def test_xlevel_B_corner_np_invariance():
    """B CORNER: a fine particle straddling THREE octant seams emits 7 cross-level images
    (3 faces + 3 edges + 1 corner, all distinct). No FATAL; Tmunu bitwise."""
    _np_invariance("B_corner", SMR, CORNER + SCHEME_B)


def test_xlevel_B_demotion_dedup_np_invariance():
    """B DEMOTION: the half-domain seam fine block whose x1x2 overhang DEMOTES onto the
    coarse x1-face already targeted by the x1-face subset. The cross-level dedup keeps one
    record per (tag, target gid); a clean exit means the post-sort (lev>=0) duplicate
    invariant did NOT fire (no double-deposit), and Tmunu is bitwise across np {1,2,4}."""
    _np_invariance("B_demote", SMR_HALF, DEMOTE + SCHEME_B)


# ---------------------------------------------------------------------------------------
# Scheme A (conservative, 5b(b), DEFAULT): EXACT identity across a seam (clean exit under
# debug=1 is the conservation oracle) + bitwise np-invariant.
# ---------------------------------------------------------------------------------------
def test_xlevel_A_fine2coarse_conservation_npinv():
    """A F->C: the fine overhang is RESTRICTED into the coarse neighbor cell. debug=1
    asserts Sum E sqrt(g) dV == Sum m W EXACTLY across the seam (clean exit); Tmunu
    bitwise np {1,2,4}."""
    _np_invariance("A_f2c", SMR, F2C + SCHEME_A)


def test_xlevel_A_coarse2fine_conservation_npinv():
    """A C->F: the coarse particle deposits its WHOLE cloud at the fine sublevel -- the
    fine-neighbor part native, the own-block part restricted back into the coarse cell.
    Exact identity (clean exit); Tmunu bitwise np {1,2,4}."""
    _np_invariance("A_c2f", SMR, C2F + SCHEME_A)


def test_xlevel_A_edge_conservation_npinv():
    """A EDGE: two seams; each fine overhang restricts into its coarse neighbor. Exact
    identity (clean exit); Tmunu bitwise np {1,2,4}."""
    _np_invariance("A_edge", SMR, EDGE + SCHEME_A)


def test_xlevel_A_corner_conservation_npinv():
    """A CORNER: three seams (3 faces + 3 edges + 1 corner restricts). Exact identity
    (clean exit); Tmunu bitwise np {1,2,4}."""
    _np_invariance("A_corner", SMR, CORNER + SCHEME_A)


def test_xlevel_A_demotion_conservation_npinv():
    """A DEMOTION: the half-domain demoted-edge overhang restricts (deduped) into the
    coarse x1-face. Exact identity (clean exit); Tmunu bitwise np {1,2,4}."""
    _np_invariance("A_demote", SMR_HALF, DEMOTE + SCHEME_A)


def test_xlevel_A_seam_plus_lower_boundary_conservation_npinv():
    """A SEAM + CLOSED BOUNDARY (lower y): a coarse particle straddling x1=0 deposits at
    the fine sublevel (cfine) while its cloud also touches the CLOSED lower-y outflow.
    The boundary clip must be the fine-cell clip matching the deposit -- regression for
    the coarse-stencil clip bug (particle-side 0.75 vs cell-side 1.0 -> residual 0.25).
    Exact identity (clean exit); Tmunu bitwise np {1,2,4}."""
    _np_invariance("A_bdy_lo", SMR_BDY, BDY_LOWER + SCHEME_A)


def test_xlevel_A_seam_plus_upper_boundary_conservation_npinv():
    """A SEAM + CLOSED BOUNDARY (upper y): the symmetric upper-y outflow case. Exact
    identity (clean exit); Tmunu bitwise np {1,2,4}."""
    _np_invariance("A_bdy_hi", SMR_BDY, BDY_UPPER + SCHEME_A)


# ---------------------------------------------------------------------------------------
# Flag / default (test 12): the default is scheme A (conservative); both schemes are
# selectable and run clean on the same straddle geometry.
# ---------------------------------------------------------------------------------------
def test_xlevel_flag_default_is_conservative():
    """No cross_level_deposit override picks scheme A: the F->C straddle then conserves
    EXACTLY (debug=1 would FATAL otherwise) -- a clean exit certifies the default is the
    conservative kernel, not native (which is non-conservative and would not assert). The
    explicit =native run also completes clean (its non-conservation is just logged)."""
    out = "xl_flagdefault"
    try:
        ph.rmdirs(out)
        ph.run_args(["-i", SMR, "-d", out] + F2C, threads=1)      # default -> A (exact)
        ph.rmdirs(out)
        ph.run_args(["-i", SMR, "-d", out] + F2C + SCHEME_B, threads=1)  # B (logged)
    finally:
        ph.rmdirs(out)
