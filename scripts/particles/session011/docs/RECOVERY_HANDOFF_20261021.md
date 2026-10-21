# Session 011 continuation state

## Active goal — 2026-10-21 16:39 PDT

Complete the single AMD trajectory to t75 with validated AH, population accounting,
reports/figures/two1080p movies, and monitor node hours. User explicitly authorizes
necessary debugging. Goal ACTIVE and incomplete. This goal turn made concrete
progress: diagnosed the stop, validated corrected tooling, audited/promoted the
latest state, and submitted same-state continuation. No replay/extra trajectory.

## Authoritative production state

AMD root: /work1/eliasmost/jiaxiwu/gi_s011_amd_20261021.
Production473967, segment0004,6nodes/24ranks,cap256: PENDING/Priority at last poll.
Do not call it running until scheduler/startup logs confirm. Check this specific job.
Release473947 COMPLETED/0:0 (454s/1node) and submitted it. No PAUSED marker.
Latest fully audited state: runs/segment_0003/rst/s11_amd.00005.rst,
cycle1902,t8.803125000000511,60291241129bytes,
SHA2561fd79b708bbb8ba5547d3c0390b914e63e90d802a377f4989d62ea4a9842d3f2.
4198470alive+801530removed=5000000; exact all-tag partition, per-tag rest weights,
and native/online component Jz pass. Unknown native IPEN330 (165/clump) explicitly
recorded, unknown rest mass2.1654597172130847e-5 per clump. Physical fields finite.
Both individual AHs first accepted t8.4859375/cycle1699; pair state00004 protected,
60301921689bytes. Final segment3 valid_bits3,strict_common0,trackers both live,
coordinate separation6.06151958. New segment4 must reacquire horizons normally.
Raw staysAMD. Recovery rotation happened only through the existing hash/audit
controller during successful release; consult control/checkpoint_inventory.json
for retained/deleted entries, not prior status notes. No Mac mirror or raw transfer.

## Root cause and released correction

473899 zero-evolution particle inspection (2s/1node) isolated IPEN only.
473909 zero-evolution saved-metric diagnosis (94s/1node): all330 failed particles
have non-positive-definite high-order interpolated spatial metrics and negative
energy discriminants; all330 trilinear interpolants valid. All25 full saved grid
fields finite. Reconstruction matches330 finite native energies within2.84e-15.
IPEN is diagnostic/output-only, not a pusher/deposition input.

QA473946 COMPLETED (43s/1node):16 controller+9 product tests, full synthetic
report/two1080p movies/decode, isolated real full-population audit and22real cohort
files. Corrected audit rejects nonfinite physical fields, records unknown energy
counts/rest mass; products preserve unknown native totals across mixed shells;
cohort geometry is usable with explicit energy-quality counts. Finalization errors
now create PAUSED.json. Scheduler lookup uses the live user queue so completed
IDs aged out of squeue do not break continuation.

Production executable remains exactly evidence/final_473059/athena_production,
SHA256e4fce096299d9d0ef852e7b713e3636afccb08a1d12a9c44cca9a657b9aa35ba,
engine df6ab41d5cdd95c7e33a8b70726891d52691c0cf; canonical input SHA256
11fb21b93abacf02e5410d77c7922967773535e329aabca44939a59a4026ef31.
No physical input, grid/dt, pusher, deposition, removal or AH acceptance change.
Live AMD checkout and deployed Python revision42823d8212e21debb01f8b69a4344f78b54401b8.
Release preserved prior gate/control/scripts in evidence/nonfinite_20261021/release_473947/.
New VALIDATION_PASS.json references metric/QA evidence and exact updated hashes.
Local evidence/nonfinite_20261021/ contains compact diagnosis, QA and release receipts.
Local scripts/ mirrors the four deployed Python files from42823d82.

## Next actions (required; goal not complete)

1. Poll473967, confirm restart8.803125 and AH reacquisition, monitor subsequent
   checkpoint audits/unknown-energy counts, GPU/storage and costs through75.
2. Local branch experiment/GI-in-cluster-session011 has committed/pushed pending
   endpoint fixes in baff767c (NOT deployed to AMD): bitwise per-Real comparison in
   src/z4c/frozen_ah_checks.hpp (unchanged NaN otherwise compares unequal) with
   unknown-energy coverage count, and products.py filtering foreign-cluster job
   receipts plus a regression test. These still need Slurm build/testing.
   Prefer a separate hashed ANALYSIS-ONLY binary for frozen saved-AH tests,
   preserving the immutable production executable. Build using existing AMD build
   cache after updating checkout; only header affects driver compilation.
   Test on real saved00005 with330unknown IPEN, nlim0, same6nodes/24ranks/cap256,
   selected horizon0, frozen searches6, unique output directory, suppress ordinary
   outputs/GI accounting as endpoint.py does. Require nonmutation pass with unknown
   count>0 and accepted AH. Prioritize production allocation; a dependency on
   production start can keep the analysis from taking its nodes first.
   Route endpoint.py probes to a validated analysis-binary receipt with checksum;
   endpoint.py currently uses gate['executable'] and has NOT been changed yet.
3. Test/deploy endpoint-only Python changes with preserved previous seal and exact
   new hashes. Coordinate changes with launches; never invalidate sealed scripts
   during a launch. Do NOT replace production executable with the analysis build.
   The one-off release_energy_audit.py is finished and must not be rerun: pins
   cycle1230/segment3 and forbids C++ diffs. Use a new versioned endpoint release.
4. Fix final resource report before claiming complete: AMD sacct must not resolve
   copied Perseus job18291 as unrelated AMD embedding18291. Pending baff767c filters
   foreign receipts; verify it. Account Perseus separately using Perseus sacct.
5. Verify all promised endpoint AH checks, all five science hypotheses, figures,
   full-population ledgers, movies/full decode, report and one-way manifest archive.
   Stop at75; extension150 requires a new user decision.

## Costs and retained failure evidence

Production through473355:49.2866667nodeh. Current recovery/debug jobs total0.175nodeh:
473899=2s,473909=94s,473917=1s,473938=33s,473942=3s,473946=43s,473947=454s,
all1node.473937 cancelled unstarted=0.473967 adds production cost once allocated.
473917 staging lacked make_input.py; corrected dependency and rerun.
473938 passed15+9 tests;473942 release stopped before mutation because old473355
had aged out of squeue. Scheduler fix validated by473946. Incorrect pending helper
473937 cancelled before allocation. All logs retained and registered for accounting.
Current whole-user storage QA473938=489559235933bytes (below1.5TiB warning).
Raw accounting_20261021.txt includes the foreign18291 collision; its uncorrected
51.042222 total is invalid. Corrected AMD total at that snapshot (release182s elapsed)
was49.9005556nodeh, including original AMD builds/checks. Re-query for current costs.
Original HIP4GPU validation473059 and CUDA18291 remain valid for unchanged engine.
Archive watcher last receipt18557; it is read-only and not proof of production state.
