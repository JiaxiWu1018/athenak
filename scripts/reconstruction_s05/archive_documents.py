"""Update current reports from preserved pre-cleanup versions after gated removal."""
import json
from pathlib import Path
import subprocess

from archive_closeout import SESSION, ARCHIVE, RESTART_DEST, now, write


def main():
    p = SESSION
    backup = p/'evidence/before_amd_cleanup_20261016'
    removed = json.loads((p/'evidence/AMD_CLEANUP_RECEIPT.json').read_text())
    verified = json.loads((p/'evidence/AMD_ARCHIVE_VERIFIED.json').read_text())
    reproduced = json.loads((p/'evidence/AMD_ARCHIVE_REPRODUCED.json').read_text())
    assert removed['status']=='REMOVED' and verified['status']=='VERIFIED' and reproduced['status']=='REPRODUCED'
    manifest = json.loads((p/ARCHIVE/'closeout_records/KEEP_MANIFEST.json').read_text())
    source = subprocess.check_output(['git','-C',str(p/'src/athenak'),'rev-parse','HEAD'],text=True).strip()
    accounting = json.loads((p/'evidence/AMD_ACCOUNTING.json').read_text())
    if not any(r[0]=='464739' for r in accounting['rows']):
        accounting['rows'].append(['464739','COMPLETED','14','1'])
    cost = sum(int(r[2])*int(r[3]) for r in accounting['rows'])/3600
    accounting.update({'allocated_node_hours':cost,'updated_utc':now(),
                       'scope':'Original survey, particle visualization extraction and 14s archive preparation; Perseus archive/reproduction/audit jobs recorded separately.'})
    write(p/'evidence/AMD_ACCOUNTING.json',accounting)
    allocated = int(subprocess.check_output(['du','-s','-B1',str(p)],text=True).split()[0])
    write(p/'evidence/STORAGE_POST_CLEANUP.json',{
        'measured_utc':now(),'amd_session_allocated_bytes':0,
        'amd_session_allocated_bytes_before':removed['campaign_allocated_bytes_before'],
        'perseus_session_allocated_bytes':allocated,
        'perseus_df':subprocess.check_output(['df','-B1',str(p)],text=True),
        'retained_amd_manifest_bytes':manifest['retained_logical_bytes'],
        'retained_amd_manifest_files':len(manifest['files']),
        'raw_retention':'Only seed1 proper-stratified 300M restart and initial conditions; evolved native particle/field dumps and all other checkpoints deliberately discarded.',
        'post_audit_measurement':'evidence/STORAGE_AFTER_ARCHIVE_AUDIT.json'})

    report = (backup/'reports/REPORT_session5.md').read_text()
    report = report.replace('Each case has **151 particle snapshots**','Each case produced **151 particle snapshots**')
    report = report.replace('Failed attempts and superseded outputs are preserved with diagnoses',
                            'Failed-attempt logs, verification receipts and diagnoses are preserved')
    start = report.index('Production raw manifests total **510,609,619,953 logical bytes**')
    end = report.index('\n\n## Implication',start)
    report = report[:start]+(
        'The original production raw manifests total **510,609,619,953 logical bytes**; '
        'they now describe historical outputs, not current availability. Following the user-approved '
        'archive plan, all Session 5 storage was removed from AMD after checksum verification and '
        'Perseus-only plotting tests. **AMD Session 5 storage is now zero**, releasing approximately '
        f'**{removed["campaign_allocated_bytes_before"]/1e9:.1f} GB**. Compact results, exact initialization '
        'inputs/evidence, source/build provenance, figures and movies remain on Perseus, together '
        'with only the proper-profile stratified seed 1 restart at 300M. '
        '[Archive closeout](ARCHIVE_CLOSEOUT.md) gives the exact retention/deletion inventory and '
        '[restart instructions](../restarts/proper_stratified_s1/README.md). Native evolved particle '
        'snapshots, field dumps and other checkpoints were deliberately discarded; no full raw '
        'backup is claimed. Other AMD campaigns were untouched. Nothing was transferred to the Mac.'
    )+report[end:]
    report = report.replace('Cache and rendering frames remain on the clusters for reproduction.',
                            'Cache and rendering frames remain on Perseus for reproduction; the AMD originals were subsequently cleaned as recorded below.')
    report += ('\n\n## Perseus archive and AMD cleanup\n\n'
               'The approved compact-data archive is complete. All **7,134 selected AMD files** '
               '(7.592 GB logical, including the restart) passed SHA256 checks on Perseus. All '
               '24 double-precision initialization dumps match their actual AMD input positions, '
               'momenta, IDs and masses exactly. The retained 300M restart passed full-payload, '
               'original-ID and per-ID mass checks. Slurm job 13890 regenerated all **34 original '
               'PNG/PDF pairs**, both 151-frame movies and the initial-position figures using '
               'Perseus data alone. Accepted original figures/movies were preserved.\n\n'
               f'Archive preparation added 14s; cumulative AMD cost is **{cost:.5f} node-hours**. '
               'Monitoring remains stopped. The next session can continue proper-profile '
               'stratified seed 1 toward five surface periods from the saved 300M state; no '
               'extension was launched here. See [the archive report](ARCHIVE_CLOSEOUT.md), '
               '[verification](../evidence/AMD_ARCHIVE_VERIFIED.json), '
               '[plotting test](../evidence/AMD_ARCHIVE_REPRODUCED.json) and '
               '[deletion receipt](../evidence/AMD_CLEANUP_RECEIPT.json).\n')
    (p/'reports/REPORT_session5.md').write_text(report)

    readme = (backup/'README.md').read_text()
    start = readme.index('AMD raw root:')
    end = readme.index('\n\n[HANDOFF.md]',start)
    readme = readme[:start]+(
        '**Archived on Perseus; AMD Session 5 storage is zero.** '
        '[Archive closeout](reports/ARCHIVE_CLOSEOUT.md) records checksum-verified retention, '
        'Perseus-only plotting reproduction and removal of the entire AMD campaign directory. '
        'All compact results and actual initialization inputs/evidence remain here. '
        '[Proper-profile stratified seed 1 restart at 300M](restarts/proper_stratified_s1/README.md) '
        'is saved for next session. Other evolved native dumps/checkpoints were discarded. '
        f'Cumulative AMD cost including visualization and archiving is {cost:.5f} node-hours. '
        'Historical raw manifests remain as provenance; they do not indicate retained raw files.'
    )+readme[end:]
    readme = readme.replace('monitoring stops at completed delivery.','monitoring remains stopped.')
    (p/'README.md').write_text(readme)
    (p/'runs/README.md').write_text(
        '# Compact production metadata mirror\n\n'
        'These 24 directories contain JSON receipts, memory samples, native histories, solver logs '
        'and completion gates copied by checksum from AMD. The AMD campaign directory has now '
        'been removed under the user-approved compact-retention plan. Original raw manifests '
        'describe historical products, most of which were intentionally discarded.\n\n'
        'All 24 full double-precision initial particle dumps and actual input binaries are in '
        '`../archive/amd_closeout_20261016/runs/CASE/initial_particles.bin` and '
        '`../archive/amd_closeout_20261016/samples/CASE.bin`. Only the proper-profile '
        'stratified seed 1 checkpoint at 300M is retained: '
        '[restart instructions](../restarts/proper_stratified_s1/README.md). '
        'See [archive closeout](../reports/ARCHIVE_CLOSEOUT.md) for the inventory and verification.\n')

    guide = (backup/'reports/REPRODUCE.md').read_text()
    guide = guide.replace('AMD root: `/work1/','As-run AMD root (subsequently removed): `/work1/')
    guide = guide.replace('`memory_samples.jsonl` on AMD.','`memory_samples.jsonl`, now retained in the Perseus compact mirror and archive.')
    guide = guide.replace('production binary samples stay on AMD.',
        'actual production binary samples are now retained under `archive/amd_closeout_20261016/samples/` on Perseus.')
    guide = guide.replace('Native binary data\nremain on AMD; `runs/` here is a compact metadata/history mirror.',
        'Native evolved particle/field dumps were subsequently discarded under the approved archive plan. '
        '`runs/` here remains the compact metadata/history mirror; only the selected final restart is retained.')
    start = guide.index('Raw particle/field/checkpoint products remain on AMD')
    end = guide.index('\nWhole-user and session storage',start)
    guide = guide[:start]+(
        'The approved archive/cleanup is recorded in [ARCHIVE_CLOSEOUT.md](ARCHIVE_CLOSEOUT.md). '
        'AMD Session 5 storage is zero. All compact products, exact initial inputs/evidence, '
        'protected executables and source/build provenance are on Perseus. Only proper-profile '
        'stratified seed 1 at 300M is retained as a restart. All other evolved native '
        'particle/field/checkpoint files were deliberately discarded; their original hashes '
        'remain in the historical completion manifests. No Mac transfer occurred.\n'
    )+guide[end:]
    guide = guide.replace("AMD's whole\nuser root is monitored every five minutes,", "During production, AMD's whole\nuser root was monitored every five minutes,")
    guide = guide.replace('No deletion or Mac transfer occurred.',
        'No deletion had occurred at that original delivery. The subsequent approved AMD cleanup supersedes its retention status; no Mac transfer occurred.')
    guide = guide.replace('All cache, native raw files and rendering frames are retained, with sizes/locations in the addendum receipt.',
        'Cache and rendering frames are retained on Perseus. Native raw files were subsequently discarded except for the selected restart and initial conditions; archive closeout supersedes the addendum retention status.')
    guide += ('\n\n## Replotting after AMD cleanup\n\n'
              'From the active source checkout, submit `sbatch scripts/reconstruction_s05/archive_perseus.sbatch reproduce`. '
              'This runs entirely from Perseus data in a new `archive_validation/JOB/` directory, '
              'leaving accepted plots unchanged. It regenerates all five original figure '
              'families at 100M and 300M and both cached movies/initial figures. '
              '`evidence/samples/` now contains the verified actual-AMD input receipts required '
              'by the plotting interface. The original source/reduction manifests retain their '
              'historical absolute paths and hashes; they do not cause the rendering stage to '
              'read the deleted AMD files. Job 13890 demonstrated this complete workflow.\n\n'
              'The initial 1.5-million-particle states remain complete for all 24 cases. '
              'For evolved particles, the movie cache contains fixed 20,000-ID positions and '
              'all-particle 2D count projections for the two seed 1 stratified runs, not general '
              'phase-space snapshots for all runs. New arbitrary evolved particle/field analyses '
              'outside the retained summaries/cache/restart require rerunning simulations. '
              'See the [restart handoff](../restarts/proper_stratified_s1/README.md) for next session.\n')
    (p/'reports/REPRODUCE.md').write_text(guide)
    figures = (backup/'reports/FIGURES.md').read_text()
    figures += ('\n\n## Archive availability\n\n'
                'All figures, movies, compact source tables/profiles and movie cache remain on '
                'Perseus and passed the independent Perseus-only reproduction test. The AMD '
                'campaign directory was removed after verification. Original raw-source hashes '
                'describe historical inputs; native evolved snapshots/field dumps were discarded. '
                'See [archive closeout](ARCHIVE_CLOSEOUT.md) for current data availability.\n')
    (p/'reports/FIGURES.md').write_text(figures)

    (p/'reports/ARCHIVE_CLOSEOUT.md').write_text(f'''# Session 5 archive and AMD cleanup

Closed {removed['created_utc']}. **Session 5 occupies zero storage on AMD.**
The entire `/work1/eliasmost/jiaxiwu/st_reconstruction_s05_20261014` tree was
removed after the approved checksum and plotting gates passed, releasing
{removed['campaign_allocated_bytes_before']:,} allocated bytes ({removed['campaign_allocated_bytes_before']/1e9:.1f} GB).
Other AMD campaigns were untouched. Monitoring remains stopped.

The authoritative Perseus session is
`{p}`. Reports remain in `reports/`, accepted figures and movies in `plots/final/`,
with the original 100M interim in `plots/interim100/`.

## Retained data

| Product | Location relative to this session | Retention |
|---|---|---|
| Reports, plots and movies | `reports/`, `plots/` | All accepted originals and historical versions |
| All 24 compact results/histories | `reduced/`, `reduced_interim/`, `runs/`, `science_review/`, `report_tables/` | Full saved time histories and original report inputs |
| Actual AMD initial inputs | `{ARCHIVE}/samples/` | 24 DP binaries; 2,304,001,332 bytes |
| Native initialization evidence | `{ARCHIVE}/runs/CASE/initial_particles.bin` | All 24 full DP initial states with per-particle masses; 2,592,010,224 bytes |
| Continuation restart | `{RESTART_DEST}` | Proper-profile stratified seed 1, 300M; 2,372,815,454 bytes |
| Source/build/job evidence | `{ARCHIVE}/` and `src/athenak/` | Documents, logs, snapshots, Git bundles, protected executables, inputs, toolchain and QA |
| Particle movie cache | `particle_visualization_data/464699/` | Reproduces both existing movies and initial-position panels |

The selected AMD-file manifest contains **{len(manifest['files']):,} files and
{manifest['retained_logical_bytes']:,} logical bytes**, including the restart;
bundles and inventory records are additional. See [archive map](../{ARCHIVE}/README_ARCHIVE.md)
and [restart instructions](../restarts/proper_stratified_s1/README.md).
The separate `qa_samples/` files are Perseus validation inputs; use the actual
AMD inputs above for exact reproduction, especially for cell-aware sampling.

## Discarded data and plotting limits

All native evolved particle snapshots, native field dumps and other checkpoints
were deliberately discarded, together with bulk preflight states, intermediate
build products and AMD duplicate copies. The original inventory covers
{removed['inventoried_original_files']:,} files before temporary archival records.
Logs, diagnoses and QA receipts survive. There is no full raw backup.

The compact archive reproduces the existing shell, constraint, occupancy,
effective-count and error figures. All initial conditions remain complete.
The movie cache retains 20,000 fixed-ID positions and all-particle 2D count
projections for the two seed 1 stratified cases. Arbitrary new evolved
particle/field analyses outside these products and the selected restart need
new simulations. Current scientific conclusions and their limitations are unchanged.

## Verification and provenance

- AMD job **464739** prepared the full retain/delete inventory, source bundles
  and retained-file hashes in 14s. All source worktrees were clean.
- Perseus job **13889** transferred and SHA256-verified all {len(manifest['files']):,} selected
  files, verified source bundles, checked all 24 actual initial states exactly
  against their input IDs/positions/momenta/masses, and validated the full
  300M checkpoint payload with unchanged per-ID masses. Runtime: 149s.
- Independent Perseus job **13890** regenerated 17 PNG/PDF pairs at each endpoint,
  both 151-frame movies and the two initial-condition PNG/PDF pairs using only
  Perseus data. It decoded both movies completely. Runtime: 143s.
- Cleanup rechecked the frozen inventory and absence of campaign writers,
  removed the exact approved root, and found no remaining named Session 5
  candidates in the AMD work/home roots. Its receipt reports zero campaign bytes.

[Retain manifest](../{ARCHIVE}/closeout_records/KEEP_MANIFEST.json),
[complete inventory](../{ARCHIVE}/closeout_records/INVENTORY.json),
[verification](../evidence/AMD_ARCHIVE_VERIFIED.json),
[reproduction](../evidence/AMD_ARCHIVE_REPRODUCED.json),
[cleanup receipt](../evidence/AMD_CLEANUP_RECEIPT.json),
[storage measurement](../evidence/STORAGE_POST_CLEANUP.json) and
[final artifact audit](../evidence/ARCHIVE_DELIVERY_AUDIT_OK.json) are retained.
Pre-cleanup documents and delivery manifest are preserved under
`evidence/before_amd_cleanup_20261016/`; archived AMD documents retain their historical wording.

Cumulative allocated AMD cost is **{cost:.5f} node-hours**, including the original
survey, visualization extraction and archive preparation. All substantial
computation used Slurm. Archive/reproduction/audit scripts are versioned on
`experiment/ST-reconstruction-s05-sampling`. No new evolution or Mac transfer was launched.
The next session can extend the retained proper-profile stratified seed 1 run
toward five surface periods after fixing that next session's period convention.
''')

    (p/'HANDOFF.md').write_text(f'''# Reconstruction Session 5 — closed and archived

Read `/data/jiaxiwu/NRPIC/AGENT_LOGISTICS.md`, then
[REPORT_session5.md](reports/REPORT_session5.md) and [ARCHIVE_CLOSEOUT.md](reports/ARCHIVE_CLOSEOUT.md).

All 24 live-Z4c runs finished 300M. Scientific findings are unchanged: the proper
radial measure strongly suppresses the historical shell response; no sampler
has a consistent overall shell advantage. Fixed resolution, short duration and
unbounded boundary effects limit interpretation.

**AMD Session 5 storage is zero.** The approved original root was removed after
all {len(manifest['files']):,} selected files passed SHA256 checks and a full Perseus-only
replot/movie test. Native evolved particle/field dumps and all other checkpoints
were deliberately discarded. Original source-file manifests are historical
evidence, not a statement that raw outputs still exist. All documents, compact
results, actual initial inputs/evidence, protected binaries and source/build/job
provenance are preserved here. No unrelated AMD data was touched.

The single continuation restart is `{RESTART_DEST}`.
See [restart README](restarts/proper_stratified_s1/README.md) and `VALIDATED.json`.
It is the complete DP proper-profile stratified seed 1 state at 300M, with
1,500,000 original IDs and unchanged masses; SHA256
`bd785c67bee977b007c43a6dd9acdbaecf4855065f222f36f5d047786a805d4c`.
Next session's objective is an extension to five surface periods; no extension
is running. Use a new session directory and the preserved ROCm 6.4.1 stack.

Accepted products remain `plots/final/`, `plots/interim100/` and
`plots/seed_review_13673/`. All 24 compact reductions/histories and the particle
visualization cache remain on Perseus. `{ARCHIVE}/` stores original-relative-path
AMD evidence and Git bundles; its snapshots are not active Git worktrees.

Jobs: AMD 464739 completed 14s; Perseus 13889 verified transfer/initials/restart
in 149s; 13890 reproduced all 34 original figure pairs and both 151-frame movies
plus initial figures in 143s. Cumulative AMD cost {cost:.5f} node-hours. Final
delivery-audit status is in `evidence/AMD_CLEANUP_STATE.json` and
`evidence/ARCHIVE_DELIVERY_AUDIT_OK.json`.

Source branch `experiment/ST-reconstruction-s05-sampling`, active clone
`src/athenak`; cleanup source `{source}`. Compiled physics source remains
`5a5db71c9d02498bfe3402c73c8a2b4b3180e377`; protected executable SHA256
`5fb4fb9f2131984854b3dd2eef455e8acc6808152e800698fcf5349e5bc35482`.
Monitoring remains stopped. Controller closed=true; original PID no longer exists.
Do not relaunch jobs from old events. Previous full handoff/state/report versions
are under `evidence/before_amd_cleanup_20261016/` and earlier dated handoffs.
''')
    state = json.loads((p/'evidence/state.json').read_text())
    state.update({'amd_root_exists':False,'last_active_utc':now(),'local_source_commit':source,
        'source_note':'AMD original checkout removed after verified archival; local pushed source includes archive tooling. Compiled physics and scientific results unchanged.',
        'continuation':'All actual campaign events acknowledged; scientific and visualization work complete; monitoring stopped.',
        'latest_amd_storage':{'campaign_allocated_bytes':0,'whole_user_allocated_bytes':removed['whole_user_bytes_after'],
                              'measured_utc':removed['created_utc'],'retained':False,'receipt':'evidence/AMD_CLEANUP_RECEIPT.json'},
        'perseus_retention':{'path':str(p),'allocated_bytes':allocated,'measured_utc':now(),
                            'status':'Compact results, all documents/plots/movies and actual initialization inputs/evidence retained, with one proper stratified seed1 300M restart.'},
        'amd_cleanup':{'status':'data archived and AMD removed','report':'reports/ARCHIVE_CLOSEOUT.md',
                       'verification':'evidence/AMD_ARCHIVE_VERIFIED.json','reproduction':'evidence/AMD_ARCHIVE_REPRODUCED.json',
                       'receipt':'evidence/AMD_CLEANUP_RECEIPT.json','restart':RESTART_DEST,'final_delivery_audit':'pending'},
        'next':'Final archive artifact audit pending; no evolution or persistent monitor remains.'})
    state['latest_storage']={'measured_utc':now(),'amd_campaign_allocated_bytes':0,'perseus_campaign_allocated_bytes':allocated,
                             'receipt':'evidence/STORAGE_POST_CLEANUP.json','status':'Approved compact archive verified; AMD Session5 removed.'}
    state['amd_allocated_cost']=accounting
    for role,job,host in [('archive_prepare',464739,'amd'),('archive_transfer_verify',13889,'perseus'),('archive_reproduction',13890,'perseus')]:
        if not any(r['id']==job and r['host']==host for r in state['jobs']):
            state['jobs'].append({'role':role,'id':job,'host':host,'status':'COMPLETED'})
    write(p/'evidence/state.json',state)
    close = json.loads((p/'evidence/AMD_CLEANUP_STATE.json').read_text())
    close.update({'status':'AMD removed; final documents updated; delivery audit pending','updated_utc':now(),
                  'amd_session_allocated_bytes':0,'cumulative_AMD_node_hours':cost,'source_commit':source,
                  'next':'Run archive_perseus.sbatch audit, inspect PASSED receipt and confirm no campaign jobs remain.'})
    write(p/'evidence/AMD_CLEANUP_STATE.json',close)
    marker = json.loads((p/'evidence/CAMPAIGN_COMPLETE').read_text())
    marker.update({'allocated_AMD_node_hours_including_archive':cost,'amd_session_allocated_bytes':0,
                   'amd_cleanup_completed_utc':removed['created_utc'],'archive_report':'reports/ARCHIVE_CLOSEOUT.md',
                   'monitor_stop_requested':True})
    write(p/'evidence/CAMPAIGN_COMPLETE',marker)
    write(p/'evidence/ARCHIVE_DELIVERY_AUDIT_OK.json',{'status':'pending','reason':'Final reports updated after cleanup; artifact audit is the remaining closure check.'})
    log = p.parent/'SESSION_LOG.md'
    title = '## Session 05 compact archive and AMD cleanup — 2026-10-16 UTC'
    text = log.read_text()
    if title not in text:
        log.write_text(text+f'''\n\n{title}

Implemented the approved compact-retention plan. AMD Session5 root removed;
zero campaign bytes remain, releasing {removed['campaign_allocated_bytes_before']/1e9:.1f} GB.
All reports/figures/movies, 24 compact result sets, actual initialization inputs
and DP evidence, source/build/job records and protected executables are retained
on Perseus. Only proper-profile stratified seed1 300M restart retained for the
next session's five-surface-period extension. Other evolved raw states discarded.
AMD 464739 prepared hashes/bundles in14s; Perseus13889 verified all7134 files,
24 initials and full restart in149s;13890 regenerated all34 figure pairs and both
151-frame movies plus initials without AMD access in143s. Cumulative AMD
{cost:.5f} nodehours. No simulation or persistent monitor restarted; unrelated
AMD campaigns untouched. Current retention, checks and exact paths:
`05_sampling_shell_response_20261014/reports/ARCHIVE_CLOSEOUT.md`.
''')
    print('ARCHIVE_DOCUMENTS_UPDATED',source,cost,flush=True)


if __name__=='__main__':
    main()
