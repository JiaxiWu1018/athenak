"""Allocated audit of legacy control failures and nominal timestep histories."""
import datetime
import json
import os
import re
import socket
import numpy as np
from campaign import ROOT, resource_snapshot, sample_array, validate_initial
from state_io import require_job, sha, write_json, read_rst


def audit(work):
    case = work.parent.name
    log = work / 'run.log'
    text = log.read_text(errors='replace')
    patterns = {
        'euler_fallback': r'gr_boris non-convergence: rank (\d+) cycle (\d+): (\d+) of \d+ particles fell back to forward Euler \(rank total (\d+)\)',
        'rejected_update': r'gr_boris update rejected: rank (\d+) cycle (\d+): (\d+) of \d+ particles kept their step-n state.*?rank total (\d+)\)',
    }
    counts = {}
    for label, pattern in patterns.items():
        rows = [tuple(map(int, m)) for m in re.findall(pattern, text)]
        totals = {}
        for rank, cycle, count, cumulative in rows:
            totals[rank] = max(totals.get(rank, 0), cumulative)
        counts[label] = dict(parsed_events=sum(x[2] for x in rows),
                            last_cumulative_by_rank=totals,
                            cumulative_sum=sum(totals.values()),
                            summaries_consistent=sum(x[2] for x in rows)==sum(totals.values()),
                            affected_rank_cycles=len(rows),
                            first_cycle=min((x[1] for x in rows), default=None))
    histories = []
    for path in sorted(work.glob('*.hst')):
        data = np.atleast_2d(np.loadtxt(path))
        bad = np.argwhere(~np.isfinite(data))
        histories.append(dict(path=str(path), sha256=sha(path), rows=len(data),
            last_time=float(data[-1, 0]), all_finite=not len(bad),
            first_nonfinite=None if not len(bad) else dict(time=float(data[bad[0, 0], 0]), column_one_based=int(bad[0, 1])+1)))
    launches = json.loads((work/'LAUNCH.json').read_text())
    steps = [dict(cycle=int(c), time=float(t), dt=float(h)) for c,t,h in
             re.findall(r'cycle=(\d+) time=([^\s]+) dt=([^\s]+)', text)]
    checkpoints = [read_rst(path, validate=False) for path in sorted((work/'rst').glob('*.rst'))]
    return dict(case=case, work=str(work), completed=(work/'COMPLETE.json').exists(),
        exit=json.loads((work/'EXIT.json').read_text()), counts=counts, histories=histories,
        diagnostic_steps=steps, checkpoints=checkpoints,
        first_fatal_lines=[line for line in text.splitlines() if 'FATAL' in line][:8],
        launch=launches, sources=[dict(path=str(q),sha256=sha(q)) for q in (log,work/'LAUNCH.json',work/'EXIT.json')])


def main():
    require_job()
    out=ROOT/'analysis'/('control_audit_'+os.environ['SLURM_JOB_ID'])
    out.mkdir(exist_ok=False)
    records=[]
    for case in ('legacy_1m_h020','legacy_1m_h010','legacy_1m_h005','frozen_rk3_1m_h020'):
        for work in sorted((ROOT/'runs'/case).glob('to_*')):
            if (work/'EXIT.json').exists(): records.append(audit(work))
    frozen=ROOT/'runs/frozen_rk3_1m_h020/to_020_job495337'
    initial=validate_initial(frozen,sample_array(ROOT/'reference/session7/samples/mc_1m_s1.bin'))
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        host=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],script_sha256=sha(__file__),
        records=records,frozen_initial=initial,resources=resource_snapshot(),
        caveats=['Counts use per-cycle summary counters, not capped per-particle warnings.',
                 'Aborted MPI logs may omit or interleave final summaries; frozen counts are recorded events only.',
                 'Diagnostic steps are sampled every ten cycles, not a complete timestep sequence.',
                 'Nominal h is a maximum; restart alignment and growth transients prevent strictly scaled uniform grids.',
                 'Frozen-source h=.2 failure is an experimental outcome; no valid20M endpoint.'])
    write_json(out/'AUDIT.json',result)
    print(json.dumps(dict(output=str(out),frozen_initial=initial,resources=result['resources'])))

if __name__=='__main__': main()
