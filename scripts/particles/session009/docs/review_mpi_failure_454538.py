#!/usr/bin/env python3
"""One reviewed operations repair; never an automatic numerical retry."""
import argparse
import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/work1/eliasmost/jiaxiwu/gi_s009_amd_20261005')
HISTORY = ROOT / 'history/mpi_failure_454538_20261007'
REVISION = 'ff4b5cb42bc67375f0b9f852de78dfe0dc073a49'
sys.path.insert(0, str(ROOT / 'scripts'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)

def preconditions():
    state = json.loads((ROOT / 'control/state.json').read_text())
    assert state['status'] == 'configuration_failure'
    assert state['time'] == 0 and state['stop_requested'] is False
    assert not state.get('pending_submission')
    for name in ('USER_STOP', 'REQUEST_STOP', 'RESOURCE_STOP.json', 'ARCHIVE_ERROR'):
        assert not (ROOT / 'control' / name).exists(), name
    live = {s.split('|')[0] for s in subprocess.check_output(
        ['squeue', '-h', '-u', 'jiaxiwu', '-o', '%i|%T'], text=True).splitlines()}
    assert not any(str(j['id']) in live for j in state['jobs'])
    ids = ','.join(str(j['id']) for j in state['jobs'])
    accounting = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', ids,
        '--format=JobIDRaw,State%30,ExitCode,ElapsedRaw,AllocNodes'], text=True)
    rows = {s.split('|')[0]: s.split('|') for s in accounting.splitlines()}
    closed = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL',
              'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
    assert all(rows[str(j['id'])][1].split()[0].rstrip('+') in closed for j in state['jobs'])
    assert rows['454538'][1:4] == ['FAILED', '1:0', '6']
    assert not (ROOT / 'control/latest_checkpoint.json').exists()
    return state, accounting

def preserve(state, accounting):
    assert not HISTORY.exists(), 'History already exists; inspect before repeating'
    HISTORY.mkdir(parents=True)
    for name in ('control', 'scripts', 'inputs'):
        shutil.copytree(ROOT / name, HISTORY / name)
    for name in ('REPORT_Jeans9.md', 'REPORT_AGENT.md', 'README.md', 'manifest.json'):
        if (ROOT / name).exists(): shutil.copy2(ROOT / name, HISTORY / name)
    (HISTORY / 'evidence').mkdir()
    for name in ('gate_provenance.txt', 'mpi_48rank_gate.txt', 'CURRENT_STATUS.md', 'frozen_config.json'):
        if (ROOT / 'evidence' / name).exists():
            shutil.copy2(ROOT / 'evidence' / name, HISTORY / 'evidence' / name)
    (HISTORY / 'logs').mkdir()
    for p in (ROOT / 'logs').glob('45453[789]*'):
        shutil.copy2(p, HISTORY / 'logs' / p.name)
    (HISTORY / 'accounting.psv').write_text(accounting)
    write(HISTORY / 'PRESERVATION_INVENTORY.json', dict(utc=time.time(),
        reason='MPI_Init PML mismatch on k003-009 before geometry/particles; no numerical evolution',
        state=state, files=[dict(path=str(p.relative_to(HISTORY)), bytes=p.stat().st_size,
        sha256=sha(p)) for p in sorted(HISTORY.rglob('*')) if p.is_file()]))

def reset(state):
    assert (HISTORY / 'PRESERVATION_INVENTORY.json').exists()
    assert json.loads((HISTORY / 'control/state.json').read_text()) == state
    from mpi_nodes import NODE_SPEC, WITNESS, witness_hosts
    config = json.loads((ROOT / 'control/config.json').read_text())
    assert config['compiled_source_commit'] == '6892be3e3f04ec573f91cb2034bc9d3009a3bdff'
    assert config['input_sha256'] == sha(ROOT / 'inputs/gi_cluster_s9.athinput')
    assert config['created_utc'] == 1791237703.7323442
    assert config['deadline_utc'] == 1795125703.7323442
    config.update(mpi_recovery_operations_revision=REVISION,
        mpi_recovery_utc=time.time(), requested_node_spec=NODE_SPEC,
        validated_mpi_nodes=witness_hosts(WITNESS.read_text()), mpi_witness_sha256=sha(WITNESS),
        excluded_nodes=['k003-009', 'k003-010'], job_prefix='jn9_20261007_known12_',
        reuse_wave_gate=True)
    config['script_hashes'] = {p.name: sha(p) for p in (ROOT / 'scripts').iterdir() if p.is_file()}
    assert len(config['script_hashes']) == 37
    state.setdefault('attempt_history', []).append(dict(status=state['status'],
        detail=state['detail'], radius=40, time=0, history=str(HISTORY), utc=time.time(),
        reviewed_action='Use exact previously successful twelve-node group; no physics/transport change'))
    for j in state['jobs']:
        if j['id'] in (454537, 454538, 454539): j['name'] = 'mpi454538_failure_' + j['name']
    state.update(status='r40_preparing', gates_submitted=False, gates_passed=False,
        detail='Reviewed pre-initialization MPI failure; pinned to historical passed48-rank allocation',
        pending_submission=None, updated_utc=time.time())
    write(ROOT / 'control/config.json', config)
    write(ROOT / 'evidence/frozen_config.json', config)
    write(ROOT / 'control/state.json', state)
    write(ROOT / 'evidence/MPI_NODE_SELECTION_REVIEW_20261007.json', dict(utc=time.time(),
        revision=REVISION, history=str(HISTORY), witness=str(WITNESS),
        witness_sha256=sha(WITNESS), nodes=config['validated_mpi_nodes'],
        config=config, actual_full_startup_pending=True))
    print(json.dumps(dict(status=state['status'], config=config), indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['preserve', 'reset']); a = p.parse_args()
    with (ROOT / 'control/workflow.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state, accounting = preconditions()
        if a.action == 'preserve': preserve(state, accounting)
        else: reset(state)
