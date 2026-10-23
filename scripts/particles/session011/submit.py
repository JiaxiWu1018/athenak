"""User-operated submission, with no retries, job chaining, transfers, or cleanup."""
import argparse
import datetime as dt
import fcntl
import getpass
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

from prepare import HERE, SESSION, atomic
from workflow import BRANCH, git, manifest, ready

ACCOUNT = 'AST21006'
JOBS = {'validation': ('gh-dev', 4, '02:00:00', 'validate.sbatch', 8),
        'production': ('gh', 16, '36:00:00', 'production.sbatch', 576)}


def command(args, *, timeout=12, **kwargs):
    result = subprocess.run(args, text=True, capture_output=True, timeout=timeout, **kwargs)
    if result.returncode:
        raise RuntimeError(f'{args[0]} failed: {result.stderr.strip() or result.stdout.strip()}')
    return result.stdout.strip()


def allocation(stage, etc=Path('/usr/local/etc')):
    """Read TACC's locally refreshed project, membership, and Vista usage maps."""
    paths = [etc / name for name in ('project.map', 'projectuser.map', 'usage.map', 'projectbalance.map')]
    if any(not 0 <= time.time() - p.stat().st_mtime <= 86400 for p in paths):
        raise RuntimeError('TACC allocation maps are missing or more than one day old')
    projects = dict(line.split() for line in paths[0].read_text().splitlines() if line.strip())
    group = projects.get(ACCOUNT)
    member = any(line.split() == [getpass.getuser(), group] for line in paths[1].read_text().splitlines())
    if not group or not member or int(group) not in os.getgroups() + [os.getgid()]:
        raise RuntimeError('Account membership is not verified for ' + ACCOUNT)
    rows = [line.split(':') for line in paths[2].read_text().splitlines()
            if line.startswith(ACCOUNT + ':Vista:')]
    if len(rows) != 1 or len(rows[0]) != 7:
        raise RuntimeError('Cannot identify the Vista allocation balance')
    _, _, start, end, total, used, remaining = rows[0]
    today = dt.date.today()
    if not dt.date.fromisoformat(start) <= today <= dt.date.fromisoformat(end):
        raise RuntimeError('Vista allocation is not active on this date')
    total, used, remaining = map(float, (total, used, remaining))
    balances = dict(line.split() for line in paths[3].read_text().splitlines() if line.strip())
    balance = float(balances[group])
    if (not all(map(math.isfinite, (total, used, remaining, balance)))
            or abs(total - used - remaining) > .02 or used < 0):
        raise RuntimeError('Inconsistent TACC allocation balance')
    available = min(balance, remaining)
    needed = JOBS[stage][4]
    if available < needed:
        raise RuntimeError(f'{available:g} SUs available; {needed} required for {stage}')
    return dict(account=ACCOUNT, membership_verified=True, available_su=available,
                requested_node_hours=needed, maps={p.name: p.stat().st_mtime for p in paths})


def duration(value):
    if value in ('UNLIMITED', 'INFINITE'):
        return math.inf
    days, clock = value.split('-', 1) if '-' in value else ('0', value)
    h, m, s = map(int, clock.split(':'))
    return int(days) * 86400 + h * 3600 + m * 60 + s


def partition(stage):
    name, nodes, wall, _, _ = JOBS[stage]
    raw = command(['scontrol', 'show', 'partition', name, '-o'])
    values = dict(re.findall(r'(\w+)=(\S+)', raw))
    maximum = values['MaxNodes']
    maxnodes = math.inf if maximum in ('UNLIMITED', 'INFINITE') else int(maximum)
    if (values['State'] != 'UP' or maxnodes < nodes
            or duration(values['MaxTime']) < duration(wall)
            or int(values.get('MinNodes', '1')) > nodes):
        raise RuntimeError('Live partition limits do not permit the requested job')
    return dict(partition=name, live_configuration=raw)


def existing_jobs():
    raw = command(['squeue', '-h', '-u', getpass.getuser(), '-o', '%i|%j|%T'])
    active = [line for line in raw.splitlines() if '|s11-' in line]
    if active:
        raise RuntimeError('Existing Session 11 job(s): ' + '; '.join(active))
    return dict(checked=True, active_session11_jobs=[])


def modules():
    raw = command(['bash', '-lc', '\n'.join([
        'set -euo pipefail', 'module reset', 'module load nvidia/24.7 openmpi/5.0.5',
        'module list', "python3 -B -c 'import numpy'", 'command -v ibrun'])], timeout=25)
    return dict(modules=['nvidia/24.7', 'openmpi/5.0.5'], numpy_imported=True, launcher=raw)


def storage(stage):
    fs = os.statvfs(SESSION)
    available = fs.f_bavail * fs.f_frsize
    required = (3 * 1024 ** 4 if stage == 'production' else 0) + 150 * 1024 ** 3
    if available < required:
        raise RuntimeError(f'Insufficient output headroom: {available} bytes available, {required} required')
    return dict(available_bytes=available, required_bytes=required,
                expected_production_output='1-3 TiB; actual size/rate must be measured')


def checkout():
    if git('branch', '--show-current') != BRANCH or git('status', '--porcelain'):
        raise RuntimeError('The campaign checkout must be clean and committed before submission')
    return dict(commit=git('rev-parse', 'HEAD'), branch=BRANCH)


def submission_location():
    host = socket.getfqdn()
    if not re.fullmatch(r'login\d+\.vista\.tacc\.utexas\.edu', host):
        raise RuntimeError('Run submission commands from a Vista login node; current host is ' + host)
    return dict(host=host)


def preflight(stage, *, remote=False):
    checks = {}
    for label, fn in [('submission_location', submission_location),
                      ('native_inputs_and_tools', manifest), ('checkout', checkout),
                      ('allocation', lambda: allocation(stage)),
                      ('queue_limits', lambda: partition(stage)), ('existing_jobs', existing_jobs),
                      ('modules', modules), ('storage', lambda: storage(stage))]:
        try:
            checks[label] = dict(passed=True, details=fn())
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
            checks[label] = dict(passed=False, error=str(exc))
    if stage == 'production':
        try:
            ready(verify_remote=remote)
            checks['gpu_validation_and_push'] = dict(passed=True)
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
            checks['gpu_validation_and_push'] = dict(passed=False, error=str(exc))
    return checks


def show_checks(stage, checks):
    print(f'{stage.upper()}: ' + ('ready for manual submission' if all(c['passed'] for c in checks.values()) else 'not ready'))
    for label, result in checks.items():
        print('  ' + ('OK ' if result['passed'] else 'BLOCKED ') + label.replace('_', ' ')
              + (': ' + result['error'] if not result['passed'] else ''))


def pending_attempts(stage):
    # A prepared record also blocks: the process may have died after Slurm accepted it.
    folder = SESSION / 'evidence/submissions'
    previous = list(folder.glob(stage + '_*.json')) if folder.exists() else []
    if previous or list((SESSION / 'runtime').glob(stage + '_*')):
        raise RuntimeError(f'A {stage} attempt already exists. Inspect its record, squeue and sacct; '
                           'do not retry without explicit review in the next chat.')


def sbatch_args(stage):
    partition_name, nodes, wall, script, _ = JOBS[stage]
    return ['sbatch', '--parsable', '-A', ACCOUNT, '-p', partition_name, '-N', str(nodes),
            '-n', str(nodes), '--ntasks-per-node=1', '-t', wall,
            '--chdir=' + str(SESSION), '--output=' + str(SESSION / 'slurm-%j.out'), str(HERE / script)]


def submit(stage):
    submission_location()
    folder = SESSION / 'evidence/submissions'; folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending_attempts(stage)
        checks = preflight(stage, remote=stage == 'production')
        show_checks(stage, checks)
        if not all(c['passed'] for c in checks.values()):
            raise RuntimeError('Preflight failed; no job submitted')
        env = os.environ.copy(); env['S11_ROOT'] = str(SESSION)
        if any(k.startswith('SBATCH_') for k in env):
            raise RuntimeError('Unset SBATCH_* environment overrides before using this launcher')
        args = sbatch_args(stage)
        # Controller-side validation uses the exact request, without allocating a job.
        dry = args.copy(); dry.insert(1, '--test-only'); dry.remove('--parsable')
        command(dry, cwd=SESSION, env=env, timeout=20)
        token = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:10]
        record_path = folder / (stage + '_' + token + '.json')
        record = dict(stage=stage, state='prepared', epoch=time.time(), command=args,
                      preflight=checks, automatic_retry=False,
                      log_pattern=str(SESSION / 'slurm-%j.out'))
        atomic(record_path, record)
        try:
            result = subprocess.run(args, cwd=SESSION, env=env, capture_output=True, text=True, timeout=30)
            record.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
            match = re.fullmatch(r'(\d+)(?:;[\w.-]+)?', result.stdout.strip())
            if result.returncode or not match:
                raise RuntimeError('Slurm response is ambiguous; inspect the retained submission record and queue')
            record.update(state='accepted', job_id=match[1],
                          log=str(SESSION / ('slurm-' + match[1] + '.out')),
                          runtime=str(SESSION / 'runtime' / (stage + '_' + match[1])))
        except BaseException as exc:
            record.update(state='ambiguous', error=repr(exc))
            raise
        finally:
            atomic(record_path, record)
            with (SESSION / 'REPORT.md').open('a') as stream:
                stream.write(f"\nManual {stage} submission attempt: `{record_path}`; "
                             f"state={record['state']}, job={record.get('job_id', 'unknown')}. "
                             "No automatic retry or chained submission.\n")
        print(f"Submitted {stage} job {record['job_id']}\nLog: {record['log']}\n"
              f"Outputs: {record['runtime']}\nReceipt: {record_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'validation', 'production'], nargs='?', default='check')
    args = parser.parse_args()
    try:
        if args.action == 'check':
            # Deliberately no mkdir, receipt writes, sbatch, simulation, or push here.
            for stage in JOBS:
                checks = preflight(stage)
                try:
                    pending_attempts(stage)
                except RuntimeError as exc:
                    checks['previous_attempt'] = dict(passed=False, error=str(exc))
                show_checks(stage, checks)
            print('No job submitted. Run validation first; review its real GPU results before production.')
        else:
            submit(args.action)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
