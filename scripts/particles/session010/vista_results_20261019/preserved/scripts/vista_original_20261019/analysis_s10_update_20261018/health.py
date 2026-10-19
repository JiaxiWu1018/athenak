"""Read-only simulation health and restart-readiness summary for the frozen branch."""
import csv
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
from run import BASE, RUN, OUT, SOURCE, atomic, check_time, digest, freeze
from science import numeric


def main():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'gg':
        raise RuntimeError('Allocated GG compute required')
    check_time()
    manifest = freeze()
    state = json.loads((RUN / 'segment_002.json').read_text())
    new_chunks = [c for c in state['chunks'] if str(RUN) in c['output']]
    audit = json.loads((OUT / 'RESTART_HEALTH.json').read_text())
    if not audit['passed'] or audit['checkpoint']['sha256'] != state['checkpoint']['sha256']:
        raise RuntimeError('Native audit does not bind this checkpoint')
    quality = {}
    for i in range(3):
        last = None; total = 0; accepted = 0
        for root in manifest['ordered_outputs']:
            with (Path(root) / f's10_orbit_gw.horizon_consumer_{i}.csv').open() as stream:
                for row in csv.DictReader(stream):
                    total += 1
                    if all(row[k] == '1' for k in ('published_this_candidate', 'association_ok', 'quality_geometry_ok', 'quality_persist_ok')):
                        last = row; accepted += 1
                    final = row
        quality[str(i)] = dict(total_candidates=total, accepted_candidates=accepted,
            last_accepted_time=float(last['time']) if last else None,
            latest_candidate=final, configured_hrel_limit=.2)
    histories = [numeric(Path(p) / 's10_orbit_gw.z4c.user.hst') for p in manifest['ordered_outputs']]
    hist = np.concatenate(histories)
    finite = bool(np.isfinite(hist).all())
    good = (hist[:, 0] > 0) & (hist[:, 10] > 0) & np.isfinite(hist).all(axis=1)
    selected = hist[good]; norms = np.sqrt(selected[:, [3, 4]] / selected[:, 10, None])
    events = []
    for c in new_chunks:
        p = Path(c['output']).parent / 'telemetry_read_events.jsonl'
        if p.exists():
            events.extend(json.loads(line) for line in p.read_text().splitlines() if line.strip())
    health = dict(checked_epoch=time.time(), host=socket.gethostname(), job=os.environ['SLURM_JOB_ID'],
        checkpoint=state['checkpoint'], completed_segment_status=state['status'],
        native_audit_passed=True, new_chunks=len(new_chunks),
        all_new_native_exits_zero=all(c['exit_code'] == 0 for c in new_chunks),
        no_new_guard_stop=all(c['stop_reason'] is None for c in new_chunks),
        gpu_peak_fraction=max(max(c['memory']['gpu_peak_fractions'].values()) for c in new_chunks),
        host_peak_fraction=max(max(c['memory']['host_peak_fractions'].values()) for c in new_chunks),
        horizon_quality=quality, history_all_finite=finite,
        constraint_norms=dict(last_time=float(selected[-1, 0]), last_H_RMS=float(norms[-1, 0]),
            last_M_RMS=float(norms[-1, 1]), max_H_RMS=float(norms[:, 0].max()), max_M_RMS=float(norms[:, 1].max()),
            mask='proper-volume chi>=0.0625; not AH-exterior'),
        telemetry_events=events, production_modified=False, submitted_jobs=False,
        continuation='Operationally restartable; horizon-quality loss is an unresolved scientific diagnostic limitation, not a failed native checkpoint.')
    atomic(OUT / 'SIMULATION_HEALTH.json', health)
    print(json.dumps(health, indent=2))


if __name__ == '__main__':
    main()
