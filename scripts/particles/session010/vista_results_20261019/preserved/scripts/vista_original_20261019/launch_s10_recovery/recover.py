#!/usr/bin/env python3
"""Explicit two-job restart from a reviewed monitoring-only stop. Never auto-submits."""
import argparse
import contextlib
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import pwd
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PREVIOUS = BASE / 'runtime_s10/runs/20261014_gh_retry_01'
RUN = BASE / 'runtime_s10/runs/20261016_gh_resume_01'
READY = BASE / 'runtime_s10/RESTART_READY_20261016.json'
spec = importlib.util.spec_from_file_location('session010_frozen_launcher', BASE / 'launch_s10/s10.py')
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
from telemetry_reader import StableSamplerReader
ORIGINAL_ARGS = base.sbatch_args
ORIGINAL_ACCOUNTING = base.accounting
SPECS = [('production', 'gh', 16, 24, 2), ('production', 'gh', 16, 24, 3)]
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL', 'PREEMPTED', 'REVOKED', 'DEADLINE'}
IMMUTABLE_INPUTS = ('production.athinput', 'compact.athinput', 'config.json', 'PROVENANCE.json',
                    'DEBUG_VERIFIED.json', 'FULL_SIZE_VERIFIED.json', 'segment_001.json',
                    'SEED_STATE.json', 'RESTART_VERIFIED.json', 'PREVIOUS_ACCOUNTING.json')
RECOVERY_FILES = ('recover.py', 'telemetry_reader.py', 'job.sbatch', 'submit.sh')


def recovery_args(specification, run, dependency=None):
    args = ORIGINAL_ARGS(specification, run, dependency)
    args[-6] = str(HERE / 'job.sbatch')
    args[-2] = str(HERE)
    return args


def no_previous_live_jobs(previous=PREVIOUS):
    ledger = base.read(previous / 'submission.json')
    if ledger.get('pending'):
        raise RuntimeError('Previous submission remains ambiguous; no restart submission')
    ids = {j['job_id'] for j in ledger['jobs']}
    user = pwd.getpwuid(os.getuid()).pw_name
    p = subprocess.run(['squeue', '-h', '-u', user, '-o', '%i|%T'], capture_output=True, text=True, check=True, timeout=20)
    if any(line.split('|', 1)[0].strip() in ids for line in p.stdout.splitlines()):
        raise RuntimeError('Previous production jobs are still live; no duplicate restart')


def validate_stop(state, result):
    if state['status'] != 'telemetry_failure' or result['stop_reason'] != 'telemetry_failure' or result['exit_code'] != 0:
        raise RuntimeError('Recovery authorizes only this reviewed clean monitoring stop, not numerical/resource/user failures')
    cp = result.get('checkpoint')
    if not cp or cp != state['checkpoint'] or result.get('verification_error'):
        raise RuntimeError('Stopped-state checkpoint is not verified')
    if cp['time'] <= 36.55 or cp['cycle'] != 31488:
        raise RuntimeError('Unexpected recovery checkpoint')
    if max(result['memory']['gpu_peak_fractions'].values()) >= .85 or max(result['memory']['host_peak_fractions'].values()) >= .85:
        raise RuntimeError('Actual memory safety stop requires separate review')
    return cp


def prepare():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'gg':
        raise RuntimeError('Prepare and hash the restart on allocated GG compute')
    if RUN.exists():
        raise RuntimeError('Prepared recovery already exists; preserve it and use --dry-run')
    no_previous_live_jobs()
    base.verify_package()
    state = base.read(PREVIOUS / 'state.json')
    last = PREVIOUS / 'segments/segment_002/chunk_002'
    result = base.read(last / 'result.json')
    cp = validate_stop(state, result)
    print('VERIFY checkpoint SHA256 and native layout', flush=True)
    base.verify_restart(cp, 3)
    base.checkpoint_matches_exit(cp, last / 'athena.log')
    print('VERIFY finite full native grid and particle identities', flush=True)
    finite = base.audit_native(cp, fields=True)
    if not finite.get('finite') or not finite.get('full_grid_finite_checked'):
        raise RuntimeError('Native state gate failed')
    old_accounting = ORIGINAL_ACCOUNTING(PREVIOUS)
    if not old_accounting['final']:
        raise RuntimeError('Previous chain accounting is not terminal; inspect Slurm before restart')
    cfg = base.read(PREVIOUS / 'config.json'); base.validate_config(cfg)
    full = base.read(PREVIOUS / 'FULL_SIZE_VERIFIED.json')
    if not full.get('passed') or full['executable_sha256'] != base.digest(base.EXE):
        raise RuntimeError('Matching full-size gate is required')
    RUN.mkdir(); (RUN / 'logs').mkdir()
    for name in ('production.athinput', 'compact.athinput', 'config.json', 'PROVENANCE.json',
                 'DEBUG_VERIFIED.json', 'FULL_SIZE_VERIFIED.json', 'segment_001.json'):
        shutil.copy2(PREVIOUS / name, RUN / name)
    base.atomic(RUN / 'PREVIOUS_ACCOUNTING.json', old_accounting)
    source_hashes = {name: base.digest(PREVIOUS / name) for name in
                     ('state.json', 'FAILURE_production_2.json', 'submission.json', 'FULL_SIZE_VERIFIED.json')}
    base.atomic(RUN / 'RESTART_VERIFIED.json', dict(passed=True, checkpoint=cp, native=finite,
                source_run=str(PREVIOUS), source_case=str(last), source_hashes=source_hashes,
                executable_sha256=base.digest(base.EXE), native_executable_unchanged=True,
                restart_from_t0=False, review='Explicit user-requested recovery from monitoring-only clean stop',
                resources=dict(nodes=16, partition='gh', wall_hours=[24, 24]), verified_epoch=time.time()))
    # Preserve all earlier science directories in order; no validation branch is added.
    seed = dict(state)
    seed.update(status='verified_segment_end', segment=1, recovery_source=str(PREVIOUS),
                recovery_authorized=True, accounting=old_accounting, scientific_completion=False)
    for key in ('failure', 'failed_stage', 'failed_segment'):
        seed.pop(key, None)
    base.atomic(RUN / 'SEED_STATE.json', seed)
    base.save_state(RUN, seed)
    base.atomic(RUN / 'INPUT_SHA256.json', {name: base.digest(RUN / name) for name in IMMUTABLE_INPUTS})
    # Original frozen launcher is left byte-for-byte unchanged and still verified.
    base.atomic(RUN / 'LAUNCHER_SHA256.json', base.read(PREVIOUS / 'LAUNCHER_SHA256.json'))
    base.atomic(RUN / 'RECOVERY_SHA256.json', {name: base.digest(HERE / name) for name in RECOVERY_FILES})
    base.atomic(READY, dict(run=str(RUN), previous=str(PREVIOUS), status='prepared_not_submitted',
                           checkpoint_time=cp['time'], jobs=2, nodes=16, wall_hours=[24, 24]))
    print('PREPARED', RUN, 'from t=', cp['time'], '; no jobs submitted', flush=True)


def finalize_preparation():
    """Finish an unsubmitted preparation after the final code/test review."""
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'gg':
        raise RuntimeError('Preparation review requires allocated GG compute')
    if (RUN / 'submission.json').exists():
        raise RuntimeError('Never update a launcher after submission')
    if base.read(BASE / 'runtime_s10/ACTIVE_RUN.json')['run'] != str(PREVIOUS):
        raise RuntimeError('Active run changed; stop preparation review')
    no_previous_live_jobs()
    receipt = base.read(RUN / 'RESTART_VERIFIED.json')
    if not receipt['passed'] or not receipt['native']['full_grid_finite_checked']:
        raise RuntimeError('Full native-state audit was not completed')
    for name, sha in receipt['source_hashes'].items():
        if base.digest(PREVIOUS / name) != sha:
            raise RuntimeError('Source recovery evidence changed')
    for name, sha in base.read(RUN / 'INPUT_SHA256.json').items():
        if base.digest(RUN / name) != sha:
            raise RuntimeError('Prepared scientific input changed')
    base.verify_restart(receipt['checkpoint'], 3)
    history = RUN / 'preparation_history'; history.mkdir(exist_ok=True)
    for name in ('INPUT_SHA256.json', 'RECOVERY_SHA256.json'):
        target = history / (name + '.before_final_review')
        if not target.exists(): shutil.copy2(RUN / name, target)
    if not (RUN / 'SEED_STATE.json').exists():
        base.atomic(RUN / 'SEED_STATE.json', base.read(RUN / 'state.json'))
    base.atomic(RUN / 'INPUT_SHA256.json', {name: base.digest(RUN / name) for name in IMMUTABLE_INPUTS})
    base.atomic(RUN / 'RECOVERY_SHA256.json', {name: base.digest(HERE / name) for name in RECOVERY_FILES})
    base.atomic(RUN / 'PREPARATION_REVIEW.json', dict(epoch=time.time(), final_user_requested_wall_hours=[24, 24],
                reason='Final reviewed recovery code and immutable seed/gate bindings; no production input changes',
                checkpoint_sha256=receipt['checkpoint']['sha256'], submitted=False))
    print('FINAL PREPARATION VERIFIED; no jobs submitted', flush=True)


def verify_frozen(run):
    for name, sha in base.read(run / 'INPUT_SHA256.json').items():
        if base.digest(run / name) != sha:
            raise RuntimeError('Prepared input/configuration changed: ' + name)
    for name, sha in base.read(run / 'LAUNCHER_SHA256.json').items():
        if base.digest(base.HERE / name) != sha:
            raise RuntimeError('Original frozen launcher changed: ' + name)
    for name, sha in base.read(run / 'RECOVERY_SHA256.json').items():
        if base.digest(HERE / name) != sha:
            raise RuntimeError('Recovery launcher changed after preparation: ' + name)
    restart = base.read(run / 'RESTART_VERIFIED.json')
    if not restart['passed'] or restart['executable_sha256'] != base.digest(base.EXE):
        raise RuntimeError('Restart/build gate changed')
    if (run / 'STOP_REQUESTED').exists() or (PREVIOUS / 'STOP_REQUESTED').exists():
        raise RuntimeError('A user stop blocks recovery')


def combined_accounting(run):
    old = base.read(run / 'PREVIOUS_ACCOUNTING.json')
    ledger = base.read(run / 'submission.json') if (run / 'submission.json').exists() else {}
    new = ORIGINAL_ACCOUNTING(run) if ledger.get('jobs') else dict(rows=[], final=not ledger.get('pending'))
    rows = {r['job_id']: r for r in old['rows']}
    rows.update({r['job_id']: r for r in new['rows']})
    return dict(rows=list(rows.values()), actual_node_hours=sum(r['node_hours'] for r in rows.values()),
                final=old['final'] and new['final'], prior_chain_included=True,
                note='Actual allocated node-hours, not a tariff invoice; running rows remain provisional.')


def submit(dry_run=False, runner=subprocess.run):
    if not READY.exists():
        raise RuntimeError('Prepare and verify the recovery on GG before submission')
    run = Path(base.read(READY)['run'])
    if dry_run:
        verify_frozen(run)
        print(json.dumps(dict(submitted=False, restart_time=base.read(run / 'RESTART_VERIFIED.json')['checkpoint']['time'],
            jobs=[dict(partition='gh', nodes=16, wall_hours=24, segment=s[4]) for s in SPECS],
            commands=[shlex.join(recovery_args(s, run, 'FIRST_JOB_ID' if i else None)) for i, s in enumerate(SPECS)]), indent=2))
        return
    if not re.match(r'^login\d+(?:\.|$)', socket.gethostname().lower()) or os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit from a Vista login shell outside idev; no submission on compute')
    with (READY.parent / 'submit.lock').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        verify_frozen(run)
        no_previous_live_jobs()
        if (run / 'submission.json').exists():
            raise RuntimeError('Recovery submission already attempted; inspect the ledger, never blindly retry')
        if not base.build_ready(full=False):
            raise RuntimeError('Verified native build is required; no build/debug queue job will be submitted')
        pointer = BASE / 'runtime_s10/ACTIVE_RUN.json'
        active = base.read(pointer)
        if active['run'] not in (str(PREVIOUS), str(run)):
            raise RuntimeError('Another run is active; no recovery submission')
        base.atomic(run / 'PREVIOUS_ACTIVE_RUN.json', active)
        base.atomic(pointer, dict(run=str(run), previous_run=active['run'], created_epoch=time.time()))
        base.sbatch_args = recovery_args
        result = base.submit_chain(run, SPECS, runner=runner)
        print(json.dumps(dict(run=str(run), **result), indent=2))


def job(run, segment):
    if run.resolve() != RUN.resolve() or segment not in (2, 3):
        raise RuntimeError('This reviewed recovery is limited to its two prepared segments')
    verify_frozen(run)
    base.sampler_rows = StableSamplerReader()
    base.accounting = combined_accounting
    base.job('production', run, segment)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    sub.add_parser('finalize-prepare')
    s = sub.add_parser('submit'); s.add_argument('--dry-run', action='store_true')
    j = sub.add_parser('job'); j.add_argument('run', type=Path); j.add_argument('--segment', required=True, type=int)
    a = p.parse_args()
    if a.command == 'prepare': prepare()
    elif a.command == 'finalize-prepare': finalize_preparation()
    elif a.command == 'submit': submit(dry_run=a.dry_run)
    else: job(a.run, a.segment)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('SESSION010 RECOVERY:', exc, file=sys.stderr)
        sys.exit(42)
