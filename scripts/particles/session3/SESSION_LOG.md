# Session 03 running record

## 2026-10-05 — implementation started

The user approved the full benchmark plan. Created the isolated session archive,
copied the prompt and derivation with matching SHA-256, and cloned the clean
project/Plummer-cluster source and Kokkos independently. No previous campaign data
was deleted. AMD ST-migration job 452339 was running and is unrelated.

Next: implement and independently verify the isotropic phase-space construction,
health gates and physical diagnostics, then run allocated preflight jobs. Long
production is gated on successful validation and a measured budget <=200 hours.

## 2026-10-05 — independent continuum check

Perseus Slurm job 11041 completed the standalone C++ probe and independent Python
reference (70-digit PDF closed form, adaptive quadrature). All 354 numerical checks
passed at 1e-8; maximum error 7.85e-10. Evidence: `evidence/profile_agreement.json`,
`initial_data/cpp_profile.csv`, `logs/profile_11041.log`. P_ref=461.99570043659014 M,
untruncated rest mass 1.0134873008561143. Cutoffs omit 0.014800997% and 0.003700288%.

Perseus CPU/MPI build job 11042 is running. Isotropic initializer, physical stresses,
frozen E/L invariants, ADM mass and guarded termination are undergoing compilation;
they have not yet passed a simulation. No AMD session-3 job is running yet.

## 2026-10-05 — compiled initialization and mesh validation

CPU build 11042 succeeded. Mesh inspection 11045 reported 624 leaves but crashed
in cleanup; diagnosed uninitialized MeshBlockPack.pcoord in the pre-physics -m
path, fixed and pushed in 4210d505. Rebuild 11046 and inspection 11047 passed:
624 baseline and 568 coarse leaves. Evidence includes both mesh structure files.
Initializer/diagnostics commit 16656468 and dense-radius/sampling commit b63917f9
are pushed to origin/project/Plummer-cluster.

AMD build 452534 and dependent preflights 452548/452549 had a six-hour resource
forecast and were cancelled without touching unrelated jobs. Switched the build
to the idle devel VM: 452552 completed in 294 seconds (0.08167 node-hours), pinned
ROCm 6.4.1, executable hash recorded in CONTINUATION.md. Its dependent four-GPU
preflights 452553/452554 remain held pending the corrected cache-finalization build.

Full-N CPU initialization exposed two startup/finalization bugs in the new code,
preserved in 11052/11056 logs and corrected in the current source. Frozen t0 11058
exits cleanly; its sampling validator initially failed JSON serialization, fixed
and rerun in 11064. Frozen validation passed all actual tag/pair and continuum
shell-moment checks; dump/in-code energy and stress discrepancies <=1.25e-10.
Job 11064 continues with live t0. No evolved-equilibrium result exists yet.

Perseus raw t0 evidence is retained under runs/; completed frozen trees each
occupy about 11 GiB. Exact size/manifests and Anta checksum archive remain to do.

## 2026-10-05 22:58 UTC — healthy preflights; budget gate stops production

Live CPU t0/sampling11064 completed successfully. Primary angular ledgers now
use common untruncated-rest-mass quartiles about COM;11072/11073/11077 passed.
Health/restart adversarial checks11079 passed. Independent profile comparison
11084 uses relative CDF errors throughout:354checks pass, max3.712e-9.

Corrected AMD build452585 at4ebd7565 succeeded; four-GPU preflights452594/452595
finish healthy at cycle100,t=7.8125M. One live Boris fallback atcycle91 is retained.
Verified checkpoint hashes/compact ledgers/logs/manifests: evidence/amd_runs/.
New COM diagnostic build452612 at5c0f38b0 compiles; CPU runtime passed, GPU
runtime of the added kernel remains. Supplementary one-rank startup452606
exceeded VM memory before initialization, preserved/excluded from throughput.
No unrelated job was changed.

AMD reductions452625/452651 complete. Budget452637 projects403.08node-hours,
above200; exit2 is intentional. Continuation records stage=stopped and exits.
No pilots/production launched. Actual AMD use through reductions0.5925hours.
User budget/scope question pending; alternatives191.69hours(shortened full
matrix),165.26hours(main-only5P), proposed425-hour ceiling(original matrix).
No option approved; original production decks remain intact.

Initialization PDF/PNG11078 and corrected preflight PDF/PNG11087 generated and
visually checked. Reports state insufficient coverage for equilibrium judgement.
Full-run movies/period products pending. Perseus inventory11081 records40.217GB
retained raw evidence. Anta /data2 has zero free bytes; archive transfer pending.
No deletion authorized or performed.
