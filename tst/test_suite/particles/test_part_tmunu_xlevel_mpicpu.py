"""Cross-level Tmunu deposition (scheme B, native-resolution) tier -- NRPIC Stage 5b(a).

Extends the Stage-4c same-level deposit to clouds that straddle a static-SMR coarse-fine
interface. The Stage-4 level-interface FATAL becomes a deposit at the TARGET block's own
resolution (DepositCloudNative), delivered by cross-level images that ride the same
canonical (target_m, tag, off_code, lev) stream as the same-level images. A cross-level
image is made UNIQUE per (tag, target gid) at generation (EnumerateParticleTargets): the
native deposit covers the WHOLE clipped cloud, so when FindDestinationIndex demotes a
diagonal overhang onto an already-targeted coarse face the duplicate is dropped (else the
target is double-deposited). Scheme B is by construction NON-conservative across a seam:
the per-cycle identity reports a measured O(straddle) residual, not a fatal. The residual
VALUES are asserted against the closed form by run_suite_5b.sh; here we assert the
decomposition-invariant artifact the harness can check directly --

  (i)  the straddle / demotion run no longer FATALs (run_args raises on any nonzero exit;
       with <particles> debug=1 a count!=appends, a cross-level (target_m,tag) duplicate,
       or a same-level identity violation fails the test);
  (ii) the deposited Tmunu is BITWISE identical across rank counts (cross-level image
       transport + the extended canonical sort + the dedup are rank-count invariant; a
       dropped/duplicated child or a demotion double-deposit would break this).

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
# (cross_level 1 / 3 / 7); all targets distinct (no demotion), so dedup is a no-op here.
SMR = "inputs/part_tmunu_smr_pytest.athinput"
# half-domain SMR: x1<0 refined, full x2/x3 -> a seam fine block has transverse fine
# siblings, so an x1x2 overhang DEMOTES onto the coarse x1-face -> the dedup is exercised.
SMR_HALF = "inputs/part_tmunu_smr_half_pytest.athinput"

F2C = ["problem/mode=single", "problem/px=0.0078125",
       "problem/py=0.25", "problem/pz=0.25"]
C2F = ["problem/mode=single", "problem/px=-0.0078125",
       "problem/py=0.25", "problem/pz=0.25"]
EDGE = ["problem/mode=single", "problem/px=0.0078125",
        "problem/py=0.0078125", "problem/pz=0.25"]
CORNER = ["problem/mode=single", "problem/px=0.0078125",
          "problem/py=0.0078125", "problem/pz=0.0078125"]
W2 = ["problem/pux=1.0", "problem/puy=1.0", "problem/puz=1.0"]  # u_i=(1,1,1): W=2 flat


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


def test_xlevel_fine2coarse_np_invariance():
    """F->C: a fine-block particle straddling x1=0 deposits its overhang onto the coarse
    neighbor at coarse resolution (one cross-level image). No FATAL; Tmunu bitwise."""
    _np_invariance("f2c", SMR, F2C)


def test_xlevel_coarse2fine_np_invariance():
    """C->F: a coarse-block particle straddling x1=0 fans out to the fine children it
    spans (up to 4 on a face; unreached ones clip to zero). No FATAL; Tmunu bitwise."""
    _np_invariance("c2f", SMR, C2F)


def test_xlevel_edge_np_invariance():
    """EDGE: a fine particle straddling TWO octant seams (x1+x2) emits 3 cross-level
    images (2 faces + 1 corner block, all distinct). No FATAL; Tmunu bitwise."""
    _np_invariance("edge", SMR, EDGE)


def test_xlevel_corner_np_invariance():
    """CORNER: a fine particle straddling THREE octant seams emits 7 cross-level images
    (3 faces + 3 edges + 1 corner, all distinct). No FATAL; Tmunu bitwise."""
    _np_invariance("corner", SMR, CORNER)


def test_xlevel_demotion_dedup_np_invariance():
    """DEMOTION: the half-domain seam fine block whose x1x2 overhang DEMOTES onto the
    coarse x1-face already targeted by the x1-face subset. The cross-level dedup keeps one
    record per (tag, target gid); a clean exit means the post-sort (lev>=0) duplicate
    invariant did NOT fire (no double-deposit), and Tmunu is bitwise across np {1,2,4}."""
    _np_invariance("demote", SMR_HALF, EDGE[:1] + ["problem/px=-0.0078125",
                   "problem/py=0.4921875", "problem/pz=0.25"])
