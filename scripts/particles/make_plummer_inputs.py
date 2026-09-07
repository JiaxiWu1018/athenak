#!/usr/bin/env python3
"""Derive the Plummer session-1 preflight decks from the single production deck.

Every preflight deck is the production deck with an explicit, auditable list of key
changes -- so "what differs from production" is mechanically checkable rather than a
claim in a comment.  Run with --check to re-derive and diff against the files on disk.

    python3 make_plummer_inputs.py --outdir ../inputs
    python3 make_plummer_inputs.py --outdir ../inputs --check
"""
import argparse
import difflib
import os
import re
import sys

PROD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "code", "inputs", "particles", "nr_pic_plummer.athinput")

P_HALF = 1192.496781


def set_key(text, block, key, value):
    """Replace `key = ...` inside <block>.  Fatal if the block or key is absent."""
    bpat = re.compile(r"(^<%s>\s*$)(.*?)(?=^<|\Z)" % re.escape(block), re.M | re.S)
    m = bpat.search(text)
    if not m:
        sys.exit("block <%s> not found" % block)
    body = m.group(2)
    kpat = re.compile(r"^(\s*%s\s*=\s*)(\S.*?)(\s*)$" % re.escape(key), re.M)
    if not kpat.search(body):
        sys.exit("key %s not found in <%s>" % (key, block))
    new_body = kpat.sub(lambda mm: "%s%s%s" % (mm.group(1), value, mm.group(3)), body,
                        count=1)
    return text[:m.start(2)] + new_body + text[m.end(2):]


def drop_block(text, block):
    bpat = re.compile(r"^<%s>\s*$.*?(?=^<|\Z)" % re.escape(block), re.M | re.S)
    if not bpat.search(text):
        sys.exit("block <%s> not found" % block)
    return bpat.sub("", text, count=1)


def add_block(text, name, body):
    return text.rstrip() + "\n\n<%s>\n%s\n" % (name, body.strip())


def header(title, lines):
    out = ["# " + "=" * 77,
           "# " + title,
           "# " + "-" * 77,
           "# DERIVED FROM inputs/particles/nr_pic_plummer.athinput by",
           "# scripts/make_plummer_inputs.py.  Changes relative to production:"]
    out += ["#   * " + l for l in lines]
    out += ["# Everything else is inherited VERBATIM.",
            "# " + "=" * 77, ""]
    return "\n".join(out)


def build(prod):
    decks = {}

    # ---- PF-A: t = 0 diagnostic, fully coupled, 4 cycles ---------------------
    t = prod
    t = set_key(t, "job", "basename", "pf_t0_plummer")
    t = set_key(t, "time", "nlim", "4")
    t = set_key(t, "time", "tlim", "1.0e30")
    t = set_key(t, "time", "ndiag", "1")
    t = set_key(t, "particles", "debug", "1")
    for i, dt in [(1, 0.5), (2, 0.5), (3, 0.5), (4, 0.5), (5, 0.5), (6, 1.0e30),
                  (7, 1.0e30), (8, 0.5)]:
        t = set_key(t, "output%d" % i, "dt", repr(dt))
    decks["pf_t0_plummer.athinput"] = header(
        "NRPIC Plummer session 1 -- PREFLIGHT A: t = 0 diagnostic (4 cycles)",
        ["<job>/basename       -> pf_t0_plummer",
         "<time>/nlim          -> 4      (stop after four cycles)",
         "<time>/tlim          -> 1e30   (nlim is the stopping condition)",
         "<time>/ndiag         -> 1      (per-cycle timestep/perf line)",
         "<particles>/debug    -> 1      (post-migration validation + ledger; FATAL on"
         " any containment, GID-range or count violation)",
         "output dt            -> 0.5 M for hst/pvtk/bin/trk so t=0 and the first cycles"
         " are dumped; cbin and rst effectively disabled (dt = 1e30)",
         "PURPOSE: leaf MeshBlock count (expect 400), per-rank memory, the actual"
         " timestep from all restrictions, seconds/cycle, output sizes, the initial"
         " Hamiltonian/momentum constraint norms and profile, the sampler banner"
         " (M_0, mu, exact momentum/angular-momentum cancellation), and the particle"
         " ledger."])+ t

    # ---- PF-B: frozen analytic metric, full N, one reference period ----------
    t = prod
    t = set_key(t, "job", "basename", "pf_frozen_plummer")
    t = drop_block(t, "z4c")
    t = add_block(t, "adm", "# empty: analytic ADM background, no spacetime evolution")
    t = set_key(t, "particles", "feedback", "false")
    t = set_key(t, "time", "tlim", repr(P_HALF))
    t = set_key(t, "time", "ndiag", "200")
    t = set_key(t, "output2", "dt", repr(P_HALF/100.0))
    # <output3> con and <output5> z4c REQUIRE a Z4c object (basetype_output.cpp:153
    # is a hard fatal error, and it fires even with a huge dt because the output object
    # is constructed and validated at startup regardless).  <output4> tmunu likewise
    # needs the Tmunu object, which only exists with particle feedback.  All three must
    # be REMOVED, not merely disabled.
    for i in (3, 4, 5):
        t = drop_block(t, "output%d" % i)
    t = set_key(t, "output6", "dt", repr(P_HALF/4.0))
    t = set_key(t, "output7", "dt", repr(P_HALF/4.0))
    t = set_key(t, "output1", "dt", repr(P_HALF/200.0))
    t = set_key(t, "output8", "dt", repr(P_HALF/400.0))
    decks["pf_frozen_plummer.athinput"] = header(
        "NRPIC Plummer session 1 -- PREFLIGHT B: frozen-metric orbit test, full N, 1 P_1/2",
        ["<job>/basename           -> pf_frozen_plummer",
         "<z4c> block REMOVED, empty <adm> block ADDED: the metric is the analytic"
         " static background on the production grid and does not evolve",
         "<particles>/feedback     -> false (required with <adm>; no back-reaction)",
         "<time>/tlim              -> P_1/2 = %.6f M" % P_HALF,
         "<time>/ndiag             -> 200",
         "<output3> (con), <output4> (tmunu) and <output5> (z4c) REMOVED: all three"
         " require objects that do not exist without <z4c>/feedback, and"
         " basetype_output.cpp:153 makes that a startup fatal error regardless of dt",
         "<output6> (cbin adm) and <output7> (rst) at P/4; pvtk at P/100, hst at P/200,"
         " trk at P/400",
         "PURPOSE: the production pusher, gather, migration and cross-refinement"
         " machinery exercised at the production particle count in the production"
         " gridded metric.  Measures radial drift and the conservation of E = alpha W"
         " and L = r W v_c for every particle -- core, half-mass, halo, and the"
         " refinement faces at R = 64, 128, 256 M which the stratified sample crosses"
         " by construction."]) + t

    # ---- PF-C: short fully coupled preflight + restart continuity -----------
    t = prod
    t = set_key(t, "job", "basename", "pf_live_plummer")
    t = set_key(t, "time", "tlim", "100.0")
    t = set_key(t, "time", "ndiag", "50")
    t = set_key(t, "output1", "dt", repr(P_HALF/200.0))
    t = set_key(t, "output2", "dt", "25.0")
    for i in (3, 4, 5):
        t = set_key(t, "output%d" % i, "dt", "25.0")
    t = set_key(t, "output6", "dt", "50.0")
    t = set_key(t, "output7", "dt", "50.0")
    t = set_key(t, "output8", "dt", "5.0")
    decks["pf_live_plummer.athinput"] = header(
        "NRPIC Plummer session 1 -- PREFLIGHT C: short fully coupled run + restart",
        ["<job>/basename  -> pf_live_plummer",
         "<time>/tlim     -> 100 M (~400 cycles at dt = 0.25 M)",
         "<time>/ndiag    -> 50",
         "output cadences compressed (pvtk/bin 25 M, cbin/rst 50 M, trk 5 M) so the"
         " short run exercises every output path and produces two restart files",
         "PURPOSE: stable fully coupled evolution, restart continuity (restart from"
         " the 50 M checkpoint must reproduce the 100 M state), particle accounting,"
         " memory growth, measured seconds/cycle and node-hours to P_1/2 and 3 P_1/2,"
         " output sizes, and constraint behaviour."]) + t
    return decks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    prod = open(PROD).read()
    decks = build(prod)
    bad = 0
    for name, text in decks.items():
        path = os.path.join(a.outdir, name)
        if a.check:
            if not os.path.exists(path):
                print("MISSING", path)
                bad += 1
                continue
            cur = open(path).read()
            if cur != text:
                bad += 1
                print("DIFFERS:", path)
                for line in difflib.unified_diff(cur.splitlines(), text.splitlines(),
                                                 "on-disk", "re-derived", lineterm="",
                                                 n=1):
                    print("   ", line)
            else:
                print("OK", path)
        else:
            os.makedirs(a.outdir, exist_ok=True)
            open(path, "w").write(text)
            print("wrote", path)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
