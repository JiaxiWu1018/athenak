# Portable analysis and one-way archival

All analysis, hashing, plotting, movies and bulk transfers run in allocations. Install numpy,matplotlib,scipy,pillow,h5py,imageio-ffmpeg in a fresh ARM/Linux environment underSCRATCH. Do not reuse os/.venv or Mac paths.

analysis_support/analyze_s9.py retains the reviewed current particle, horizon association, rawR40 waveform and fixed Cartesian density/movie algorithms; despite its historical filename it now requires explicit --root,--athenak and repeatable ordered --run paths. Pass actual closed production output directories, never split/restart validation branches. Each requires SCIENCE_VERIFIED.json or ARCHIVE_VERIFIED.json from allocated checksum verification. Side files must share the per-run cwd; nested bin/planes are found recursively. Pure helpers do not contact AMD or publish a scientific stop. The revised reporter uses native checkpoint time and actual allocation rows, keeps tracker coverage separate, and issues no stale-time forecast. Scientific tail duration comes from control/reviewed_limits.json; absent agreement, ringdown termination is deferred.

Example only, after data and reviewed limits exist:

```sh
python analysis_support/analyze_s9.py --root "$RUN_ROOT" --athenak "$SOURCE_TREE" --milestone final --run "$RUN_ROOT/segments/segment_001/out"
```

The actual repaired Session007 renderers are retained with explicit CLI paths:render_s7_interim_visuals.py and render_progress_movie.py. Their historical names do not select the scientific model. Review their sidecar/accepted-horizon handling before promotion; use current unique accepted-horizon joins. Raw complex multipoles remain primary; FFI requires causal coverage, convention verification and gap/cutoff checks. Inspect representative figures/movie frames and decode movies before promotion.

Archive destination candidate:anta.caltech.edu:/data3/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013/. Verify current space and site allocations there. Copy closed run directories plus exact inputs/source/executable provenance and checkpoints; produce SHA256 inventory on Vista compute and independently verify every copied file in Anta compute. Retain source/destination manifests and receipts. No deletion is authorized by this package. Prefer Vista analysis first, but archive earlier if10-day purge exposure or queue/transfer delays threaten data. No bidirectional sync, AMD storage caps or old archival cron is inherited.

On Vista compute, `scripts/seal_run.py --run OUT --result RESULT.json` hashes all closed output files and verifies the native final checkpoint, producing a relative-path inventory in SCIENCE_VERIFIED.json. After an allocated copy to Anta, run `scripts/verify_archive.py --run ANTA_OUT` on Anta compute. It independently rehashes every inventoried file and verifies the relocated native checkpoint, then writes ARCHIVE_VERIFIED.json. These receipts certify data integrity; scientific acceptance still requires diagnostic review. Keep both manifests, the exact transfer command and the source/executable provenance. Neither helper transfers or deletes files.

Session009's TT convention and wave-gate receipt are retained in reference_reports for provenance and reviewed reuse. The analysis reads an explicitly accepted `evidence/wave_gate.json` in the new run root; without one it defers validated waveform/ringdown claims. Review source/executable/convention compatibility before accepting that evidence or doing focused revalidation. Do not repeat the historical full campaign by default.
