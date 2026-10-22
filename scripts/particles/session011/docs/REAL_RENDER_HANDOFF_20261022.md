## Real-data render validation queued — 2026-10-22 00:20 UTC

Current goal turn made progress: added a non-evolving integration check of real
particle/AH rendering, committed/pushed0569f58e. It uses the renderer whose hash
passed build473989, audits exact initial-cohort ID subset and unchanged rest weights,
selects the latest audited segment3 frame with both AHs accepted at exactly the
same cycle/time, writes wide/close1920x1080 QA previews, and checks render input
bytes unchanged (including unknown energies). Preview only, not final75 products.
No simulation, replay, field/particle mutation, or production tooling edit.

Job474110 (AMD devel,1node/4CPUs,5min bound) PENDING/Resources,0allocated node-hours.
Latest scheduler start estimate19:38:31CDT Oct21 =00:38:31UTC Oct22; tentative.
Script AMD evidence/check_real_render_20261022.py; versioned locally under
code/scripts/particles/session011/diagnostics/check_real_render.py.
Read evidence/real_render_474110.slurm.log and real_render_474110/RESULT.json;
if pass, copy its manifest-listed wide/close PNGs, exact accepted_horizons.csv and
receipt to Perseus, verify hashes and inspect both images. Diagnose failure before
claiming real-output validation. Do not treat it as the pending473992 frozen-AH
regression: those prove different properties.

Production473967 stillPENDING/Priority;473992 and474002 stillPENDING/Dependency.
All handles confirmed live; no duplicate submission. Last science remains audited
8.803125. Debugging allocation total stays0.1941667nodeh until new jobs run.
The full goal through75 and final AH/report/movie validation remains ACTIVE.
