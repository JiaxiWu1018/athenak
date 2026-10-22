# Session011 saved-state parameter repair and continuation

## New audited checkpoint and saved-AH launch repair — 2026-10-22T03:23:36.845017+00:00

Production473967 COMPLETED0:0 after7374seconds on6nodes (12.290000nodeh).
Checkpoint runs/segment_0004/rst/s11_amd.00006.rst is fully audited at
10.078125000000801/cycle2718:3,755,767alive+1,244,233lapse removals=5,000,000.
Exact ID partition, per-tag rest weights and all component native/onlineJz pass.
Native diagnostic-energy unknown_count=0 at this state; earlier unknowns remain
recorded, not retroactively erased.4572blocks,dt0.0015625,58,822,222,521bytes,
SHA25638e9fdfd937d0e060f77dd36ff082e9f3fabb8b68eec285e0522c809bb2b1187.

Continuation474260 RUNNING in segment0005 on6nodes,24ranks/cap256. Startup
readback exactly matches00006 metadata/hash. Both individual AHs reacquired at
cycle2721/time10.082812500000802 after3searches. Preserve the restart validity gap;
no uninterrupted-versus-restarted equivalence claim. Latest inspected pair accepted
cycle2984/time10.493750000000896. Segment5 final population audit remains pending.
At this snapshot474260 elapsed2322s/6nodes=3.870000nodeh. Completed production
61.576667nodeh; including this live allocation65.446667nodeh. All known AMD
allocations66.298611nodeh, including0.851944nonproduction. Perseus actual accounting
remains unavailable and excluded. Pending/cancelled unallocated jobs cost0nodeh.

Saved-state regression473992 FAILED1:0 after58seconds/6nodes (0.096667nodeh).
No AH search/evolution occurred: native parser rejected command-line require_accept
absent from the production checkpoint header. Header inspection also found all16
output dcycle parameters absent. Historical command, log and failure preserved.

Repair commits548df6c1 andf89d3fc4925e0b8a9777b88abf2806792cdb1648 pushed.
Analysis-only endpoint helper writes a minimal diagnostic_checks input overlay with
frozen_ah_searches=6,require_accept,horizon, loaded via-r plus-i. The checkpoint
retains every physical parameter and saved finder seed. Outputs are disabled only
through their existing cadence key (dt or dcycle). The saved-state regression first
runs native-n and compares every effective parameter against the exact saved header
plus explicit analysis controls, before allocating/reading the full mesh. It still
requires acceptedH0,unchangedtime/cycle,bitwise particle/grid nonmutation,unknown
energy coverage and immutable production executable. No C++ rebuild or engine change.

QA474287 (first overlay repair) and474292 (complete parser repair) each COMPLETED
32s/1node. Latest474292:21controller+10product+3plot regressions, synthetic movies
and real exact-ID/rest-weight/spectral-AH render checks pass. Products.py hash remains
27e2a56f0bf93d6db7f3ac7bf95fa1a49d81b9b517f17330d423b1b45db72bd9.
Endpoint.py hash822ef48a5e850e736ddb8eb73d91dd28c53cc8944993c3dba3bcb417e38bb33e.
Actual native parameter preflight and saved-solver verdict are still pending.

Current chain:474293 saved-state retry PENDING/Resources, using QA474292 isolated
source plus analysis binary473989 (SHA2562f5c21ce1fce83794015884825bcd57719745fcebcab4fd13a366521647c1337).
474294 guarded combined analysis/renderer deployment afterok474293. It installs all
four tested tools from the successful saved-check receipt; no separate renderer
installer is needed.474002 and474224 cancelled while unallocated; never retry them.
474166 was already superseded/cancelled. Installer still requires initializedRUNNING
production under submit.lock, intact seals/binary, and otherwise exits75/deferred
without changes. Check actual receipts; reschedule deferred deployment only during
a suitable live interval. Remote checkout remains5fa3100a; live scripts remain old
until474294 passes. Production474260 is independent of these analysis dependencies.

Compact evidence: evidence/continuation_474260/ (36 files checksum-verified on
Perseus), including completed audit/manifest, restart readback, first/latestAH rows,
failure logs, replacement submission receipt and both QA receipts/test logs.
Raw checkpoint staysAMD at the exact path above. Previous milestone/regression
states remain protected; no new raw transfer or extra evolved trajectory.

Next: monitor474260; inspect474293 native parser and solver/nonmutation verdicts;
inspect474294 deployment or safe deferral; then mirror actual deployed tools and
receipts. Continue the one trajectory to75 and finish all required AH angular,
full-ledger, scientific/report/figure/movie and checksum-archive validation.
Goal remains ACTIVE/incomplete. Previous goal turn was a verified wait followed
by completed checkpoint evidence; this turn repaired the failed analysis launcher.
