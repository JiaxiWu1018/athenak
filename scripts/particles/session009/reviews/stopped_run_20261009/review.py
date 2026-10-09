#!/usr/bin/env python3
"""Review sealed stopped data in Slurm; keep all production stop controls intact."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
import time

ROOT = Path('/data3/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005')
OUT = ROOT / 'analysis/update_final_2408'
REVIEW = ROOT / 'analysis/stopped_review_20261009'


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Saved metric-plane review requires a Slurm allocation')
    receipt_path = REVIEW / 'REVIEW_COMPLETE.json'
    if receipt_path.exists():
        return
    REVIEW.mkdir(parents=True, exist_ok=True)
    original = REVIEW / 'original_reports'
    original.mkdir(exist_ok=True)
    for name in ('resources.json', 'numerical_health.json', 'REPORT_Jeans9.md', 'REPORT_AGENT.md'):
        target = original / name
        if not target.exists():
            shutil.copy2(OUT / name, target)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in original.iterdir() if p.is_file()}

    # Frozen health9.py searches run/*.bin, but production outputs use run/bin/.
    # Present the three most recent sealed planes through a small symlink view.
    # This leaves the frozen implementation and the raw output files unchanged.
    run = ROOT / 'runs/segment_001'
    archived = json.loads((run / 'ARCHIVE_VERIFIED.json').read_text())
    files = {v['path']: v for v in archived['files']}
    view = REVIEW / 'plane_view'
    view.mkdir(exist_ok=True)
    selected = []
    for plane in ('xy', 'xz', 'yz'):
        candidates = sorted(run.rglob('*.z4c_' + plane + '.*.bin'))
        if not candidates:
            raise RuntimeError('No saved ' + plane + ' plane')
        path = candidates[-1]
        row = files[str(path.relative_to(run))]
        if path.stat().st_size != row['bytes']:
            raise RuntimeError('Verified plane size changed')
        link = view / path.name
        if not link.exists():
            link.symlink_to(path)
        selected.append(dict(path=str(path), archive_sha256=row['sha256'], bytes=row['bytes']))
    sys.path.insert(0, str(ROOT / 'scripts'))
    from health9 import assess
    assess(ROOT, [view], REVIEW)
    health = json.loads((REVIEW / 'numerical_health.json').read_text())
    if len(health['samples']) != 3 or not all(v['finite'] for v in health['samples']):
        raise RuntimeError('Incomplete or nonfinite saved metric-plane assessment')
    health['path_review'] = 'Latest xy/xz/yz planes from sealed run/bin presented via symlink view; frozen health9.py unchanged.'
    health['selected_planes'] = selected
    (REVIEW / 'numerical_health.json').write_text(json.dumps(health, indent=2) + '\n')
    shutil.copy2(REVIEW / 'numerical_health.json', OUT / 'numerical_health.json')

    # REQUEST_STOP prevents controller checkpoint acceptance, not final checkpoint
    # verification. Use the actual archived verified checkpoint for time coverage.
    resources = json.loads((original / 'resources.json').read_text())
    checkpoint = json.loads((ROOT / 'evidence/latest_checkpoint.json').read_text())
    checkpoint_file = run / 'rst/s9_orbit_gw.00005.rst'
    item = files[str(checkpoint_file.relative_to(run))]
    if checkpoint['sha256'] != item['sha256'] or checkpoint['bytes'] != item['bytes']:
        raise RuntimeError('Checkpoint and archive verification disagree')
    start = float(json.loads((ROOT / 'evidence/gate_receipt.json').read_text())['checkpoint']['time'])
    end = float(checkpoint['time'])
    if not end > start:
        raise RuntimeError('No verified production time interval')
    state = resources['state']
    segment_ids = {v['id'] for v in state['jobs'] if v['name'].startswith('segment')}
    node_hours = sum(v['raw_node_hours'] for v in resources['AMD_jobs'] if v['job'] in segment_ids)
    historical_rate = node_hours / (end - start)
    resources.update(
        accounting_correction_utc=time.time(),
        latest_verified_checkpoint=checkpoint,
        controller_time_is_stale=True,
        measured_production_interval=[start, end],
        production_allocated_node_hours=node_hours,
        mean_production_node_hours_per_M=historical_rate,
        estimated_remaining_node_hours_to_t400=None,
        estimate_limitations='Historical allocation divided by actual production time interval; not a forecast. GPU memory resource stop requires review before any resumed allocation. Old controller t0.03125 is not production coverage.',
        original_reporting_sha256=before,
    )
    (OUT / 'resources.json').write_text(json.dumps(resources, indent=2) + '\n')
    text = f'''## Stopped-run review — 2026-10-09 UTC

This is the closeout of a run stopped by the GPU memory guard, not a completed orbit/merger/ringdown calculation. The latest verified, checksum-archived checkpoint is t={end:.8g}M_ref, despite the controller still displaying t={state['time']} because REQUEST_STOP prevented its continuation inspector. No stop flag was cleared and no evolution job was submitted.

The original report's production-cost estimate used that stale controller time. The recorded production allocation used {node_hours:.6f} raw node-hours for the verified interval t={start:.8g} to {end:.8g}: {historical_rate:.6f} raw node-hours/M_ref as a historical average. Remaining-cost estimates are withdrawn while the memory stop requires review; this average includes the inexpensive early evolution and is not a collapse/orbit speed forecast. Original reports and resources are preserved under {original}.

The prior numerical_health.json had zero samples because its path search missed run/bin/. Allocated job {os.environ['SLURM_JOB_ID']} checked the actual three latest saved diagnostic planes; all metric fields in these planes are finite. This is a sampled plane check, not full3D accuracy or convergence evidence. Characteristic estimates and exact selected file hashes are in {OUT / 'numerical_health.json'}.

'''
    for name in ('REPORT_Jeans9.md', 'REPORT_AGENT.md'):
        document = (original / name).read_text()
        if name == 'REPORT_Jeans9.md':
            old = next(line for line in document.splitlines() if line.startswith('Recent production estimate:'))
            document = document.replace(old, 'Historical production average: ' + str(historical_rate) + ' raw node-hours/M_ref over verified t=' + str(start) + '..' + str(end) + '. No remaining-cost forecast is issued while the GPU memory stop requires review. See the stopped-run correction above.')
        else:
            # Replace the embedded stale resource record while preserving its original.
            document = document.replace(json.dumps(json.loads((original / 'resources.json').read_text()), indent=2), json.dumps(resources, indent=2))
        (OUT / name).write_text(text + document)
    receipt = dict(job=os.environ['SLURM_JOB_ID'], utc=time.time(), production_stop_preserved=True,
                   health_samples=3, all_saved_plane_fields_finite=True,
                   original_sha256=before, latest_verified_checkpoint=checkpoint,
                   production_node_hours=node_hours, production_interval=[start,end],
                   historical_node_hours_per_M=historical_rate,
                   remaining_cost_forecast=None, frozen_runtime_files_modified=False)
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
