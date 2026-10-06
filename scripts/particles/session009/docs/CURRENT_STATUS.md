# Current status — Session009 R40 fresh start queued

Checked 2026-10-06T22:47:06.919839+00:00. Live AMD `control/state.json` and Slurm are authoritative.

## Submitted and tested

User approved extraction R40 and reset from t0 on October6. Fine wave floor extends through R46; exact seeded count5384, down2072 (27.79%) from R50. Five million particles, approved local boost0.133215, central ceiling1/256, all other approved physics/cadences/caps retained. No Vista migration.

- Scientific input commit ae89f066 pushed; operations6b9806eb and batch receipt path repairddf31ebd pushed.
- Perseus allocated QA11150:14 tests plus syntax/input-contract checks passed. No simulation/build/plot reduction on a login node.
- Actual AMD input preflight453630 completed0:0 in4s, all11 regression checks passed and three compiled-executable input commands accepted. Receipt evidence/r40_input_preflight.json onAMD.
- Full startup **453631 PENDING(Resources)**,12nodes/48GPUs,4h; inspector **453632 PENDING(Dependency)**. It schedules production only after mesh/ledger/memory/restart/output gates pass. No current binary evolution, verified checkpoint or new scientific result. Scheduling forecast is provisional; see r40_queue_start_estimate.txt.
- Passed80M linear propagation and short Weyl convention tests reused unchanged. New complete AMD mesh/full-particle memory/restart/output validation remains pending.

## Preserved failure history

R50 startup452580 failed before initialization because output7/last_time was not an explicit input key. Old failed runs, configs, scripts, reports and inventory preserved onAMD andAnta in history/r50_before_reset_20261006/. Anta2353 had checksum-verified both failed runs. Corresponding Perseus old reports/plan/status retained there. No Session008 product changed.

R40 preflight453626 accepted all simulation inputs but its final inline receipt import failed because batch working directory was not scripts/. The helper path was repaired explicitly and pushedddf31ebd.453627 cancelled without allocation;453628 completed the failure inspection. Reviewed evidence/state/config preserved in AMD history/r40_receipt_import_failure_453626/.453630 then passed. No numerical evolution was retried. All historical jobs remain in accounting.

## Durable automation

- Anta metadata cron every5min, marker JEANS9_20261005_METADATA, reinstalled and successfully ticked. Allocated Anta jobs pull sealed data, checksum-verify, then analyze initial/t12/every50M/final and update plots/reports/movies. All3 prior archive jobs count toward96; original deadline retained. Old R50 final report pointer moved to history; no current science plot is claimed.
- Perseus two-hour cron at even UTC hours, marker JEANS9_R40_TWO_HOUR_WAKE. Same existing Codex thread01a0f9d8-f1e1-7250-b3b3-0bb0c6dfa014 is resumed when idle; a lock and active-turn check avoid overlap. Exact command/source in wake_check.py. Minimal-cron-environment auth and AMD metadata probe passed. Actual idle resume delivery is pending first due check after this active turn; receipts distinguish installation/probes/delivery. Evidence/two_hour_checks.jsonl records checks. Schedule ends at original45-day deadline or sticky userstop. Runtime continuation does not require a wake to work.
- No automatic numerical retries. Failures stop for review; scientific/resource stops and user-stop remain sticky.

## Resources and next action

AMD actual0.9225rawnodeh; pending startup+inspector reserve48.5, maximum49.4225 so far. Anta2333/2334/2353 completed115/116/118s, total0.096944444node/GPUh;3/96jobs. Monetary tariff unavailable. Current AMD storage/48GPU memory sample awaits the allocated startup monitor; do not quote old samples as current measurements.

Original hardt400/10000rawAMDnodeh,90segments,≤12h jobs/application11h20, deadline2026-11-19T22:01:43.732344Z. AMD1.25TiBsession plus whole-user guards; Anta16TiB. Latest3verified checkpoints kept after archive verification; no checkpoint exists yet. Earlier ringdownstop only accepted common enclosure + usable/gapchecked/inband outgoing ringdown +100savedM after observed wavepeak. Merger not guaranteed.

Next automatic action:453631 performs full startup;453632 either halts with preserved failure or submits first production targett12 from the accepted startup checkpoint, then finite one-successor segments toward50M milestones. Inspect representative figures/frames once verified data arrive. Do not manually submit duplicate jobs or change physics to bypass a gate.

## Exact controls

```sh
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/workflow.py status'
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/workflow.py stop'
ssh hpcfund.amd.com 'python3 /work1/eliasmost/jiaxiwu/gi_s009_amd_20261005/scripts/workflow.py cancel'
python3 /data/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/code/scripts/particles/session009/wake_check.py remove
```

Stop requests a clean checkpoint when possible and blocks successors; cancel is campaign-scoped emergency cancellation. Never cancel unrelated ST `s9_prod`. Removing the wake only removes its own cron entry. Anta archive trigger control is scripts/archive_trigger.py remove onAnta; use AMDstop first when stopping the scientific campaign.

Compiled source6892be3e3f04ec573f91cb2034bc9d3009a3bdff unchanged; inputSHA256 deb6f631f7b06f8df6927662e01d6a3fa88bda4b0322ade3a1a07ff10f05dffd. Frozen per-file bindings in evidence/frozen_config.json. README/manifest map all roots. No Session009 raw particle/volume/checkpoint files onPerseus; small QA/source/docs retained here.
