#!/usr/bin/env python3
"""Endpoint-only submission and non-evolving saved-state angular checks."""
import argparse
import csv
import fcntl
import json
import os
import subprocess
from pathlib import Path
from runtime import atomic, require_compute, digest
from make_input import parse


def submit_endpoint(root):
    control=root/'control'
    if json.loads((control/'ENDPOINT_75.json').read_text())['time']<75-1e-10:
        raise RuntimeError('endpoint has not been reached')
    with (control/'endpoint_submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        layout=json.loads((control/'submission.json').read_text())
        parent=None
        for kind,script in [('ah','amd_saved_ah.sbatch'),('products','amd_products.sbatch')]:
            receipt=control/f'endpoint_{kind}_submission.json'
            if receipt.exists():
                prior=json.loads(receipt.read_text())
                if prior['status']!='submitted':raise RuntimeError('ambiguous endpoint submission; manual receipt review required')
                parent=prior['job_id'];continue
            command=['sbatch','--parsable',f'--chdir={root}',f'--export=ALL,SESSION_ROOT={root}',
                     f'--output={root}/evidence/endpoint_{kind}_%j.slurm.log']
            if kind=='ah':command += [f"--nodes={layout['nodes']}",f"--ntasks={4*layout['nodes']}",'--ntasks-per-node=4']
            if kind=='products':command += [f'--dependency=afterany:{parent}']
            command += [str(root/'scripts'/script)]
            atomic(receipt,dict(status='prepared',command=command))
            output=subprocess.check_output(command,text=True).strip()
            job=output.splitlines()[-1].split(';')[0]
            if not job.isdigit():raise RuntimeError('unrecognized endpoint scheduler receipt')
            atomic(receipt,dict(status='submitted',job_id=job,command=command))
            parent=job


def saved_ah_executable(root,gate):
    """Resolve a separately validated analysis binary without changing evolution."""
    receipt=root/'control/SAVED_AH_TOOLING.json'
    if not receipt.exists():
        if gate.get('diagnostic_energy_repair'):
            raise RuntimeError('missing validated NaN-safe frozen AH tooling')
        return gate['executable']
    record=json.loads(receipt.read_text())
    if (record.get('status')!='pass' or record.get('evolution_advances')!=0
        or not record.get('bitwise_particle_check')
        or record.get('production_executable_sha256')!=gate['executable_sha256']):
        raise RuntimeError('incompatible saved AH tooling receipt')
    if digest(record['executable'])!=record['executable_sha256']:
        raise RuntimeError('saved AH executable checksum mismatch')
    return record['executable']


def saved_ah_controls(path,horizon,require_accept):
    """Add analysis-only knobs absent from ordinary production restart headers.

    Athena loads this minimal -i overlay after -r. Never reload the production
    input: the checkpoint's physical parameters and serialized finder seeds stay
    authoritative. Command-line overrides alone cannot add missing parameters.
    """
    if horizon not in (0,1,2) or not isinstance(require_accept,bool):
        raise ValueError('invalid saved-state AH selection')
    path.write_text('<diagnostic_checks>\nfrozen_ah_searches = 6\n'
                    f'require_accept = {str(require_accept).lower()}\n'
                    f'horizon = {horizon}\n')
    return path


def probe(root):
    require_compute()
    out=root/'evidence'/f'saved_ah_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(exist_ok=False)
    gate=json.loads((root/'control/VALIDATION_PASS.json').read_text())
    analysis_executable=saved_ah_executable(root,gate)
    atomic(out/'analysis_executable.json',dict(path=analysis_executable,sha256=digest(analysis_executable),production_executable=gate['executable']))
    layout=json.loads((root/'control/submission.json').read_text())
    inventory=json.loads((root/'control/checkpoint_inventory.json').read_text())
    events={}
    for path in sorted((root/'runs').glob('segment_*/*.gi_milestones.csv')):
        for row in csv.DictReader(path.open()):events[row['event']]=int(row['cycle'])
    selected={}
    for label,cycle in events.items():
        if 'individual' in label or 'strict_common' in label:
            selected[cycle]=label
    endpoint=json.loads((root/'control/ENDPOINT_75.json').read_text())
    selected[endpoint['cycle']]='final75'
    blocks=parse((root/'inputs/production.athinput').read_text())
    disable=[f'{block}/dcycle=0' for block in blocks if block.startswith('output')]
    disable += [f'{block}/dt=-1' for block in blocks if block.startswith('output')]
    results=[]
    common=any('strict_common' in key for key in events)
    for cycle,label in sorted(selected.items()):
        checkpoint=next(r for r in inventory if r['cycle']==cycle and not r.get('deleted'))
        horizons=[2] if ('strict_common' in label or (label=='final75' and common)) else [0,1]
        for ntheta in (10,14):
            for horizon in horizons:
                run=out/f'{label}_c{cycle}_n{ntheta}_h{horizon}';run.mkdir()
                overlay=saved_ah_controls(run/'diagnostic_controls.athinput',horizon,False)
                command=['prun',analysis_executable,'--kokkos-map-device-id-by=mpi_rank',
                         '-r',checkpoint['path'],'-i',str(overlay),'-d',str(run),
                         'time/nlim=0','time/tlim=75','gi_diagnostics/enabled=false',
                         f'fastflow/ntheta={ntheta}',f"mesh_refinement/max_nmb_per_rank={layout['capacity']}"]+disable
                atomic(run/'command.json',command)
                with (run/'run.log').open('w') as f:
                    rc=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT).returncode
                log=(run/'run.log').read_text()
                rows=[]
                for path in run.glob('*.ah_events_*.csv'):
                    rows += [r for r in csv.DictReader(path.open()) if r['published']=='1']
                results.append(dict(event=label,cycle=cycle,ntheta=ntheta,horizon=horizon,
                                    returncode=rc,nonmutating_pass=rc==0 and 'FROZEN_AH_CHECKS_PASS' in log,
                                    accepted=rows[-1] if rows else None,run=str(run),checkpoint_sha256=checkpoint['sha256'],
                                    diagnostic_overlay_sha256=digest(overlay)))
                atomic(out/'RESULT.json',dict(job_id=os.environ['SLURM_JOB_ID'],evolution_advances=0,checks=results))
                if rc:raise RuntimeError(f'saved-state probe failed: {run}')
    atomic(root/'control/SAVED_AH_CHECKS.json',dict(path=str(out/'RESULT.json'),checks=results))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['submit','probe']);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    (submit_endpoint if a.action=='submit' else probe)(a.root)
