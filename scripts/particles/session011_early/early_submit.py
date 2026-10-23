"""Manual Session 11 submission with the user's explicit early-production policy."""
import argparse
import datetime as dt
import fcntl
import getpass
import json
import os
import re
import subprocess
import sys
import time
import uuid

from early_gate import HERE, MODE, SESSION, digest, manifest
import submit as legacy
from prepare import atomic


def job_id(stdout):
    """Vista's submit plugin prints a banner even when --parsable is requested."""
    lines = stdout.strip().splitlines()
    terminal = lines[-1].strip() if lines else ''
    match = re.fullmatch(r'(?:Submitted batch job )?(\d+)(?:;[\w.-]+)?', terminal)
    if not match:
        raise RuntimeError('Slurm response is ambiguous; inspect the retained submission record and queue')
    return match[1]


def existing_jobs(stage):
    raw = legacy.command(['squeue', '-h', '-u', getpass.getuser(), '-o', '%i|%80j|%T'])
    companion, duplicate = [], []
    own = 's11-final-' if stage == 'production' else 's11-ah-validation'
    other = 's11-ah-validation' if stage == 'production' else 's11-final-'
    for line in raw.splitlines():
        fields = line.split('|')
        if len(fields) != 3:
            raise RuntimeError('Unexpected squeue response; cannot rule out a duplicate job')
        name = fields[1].strip()
        if name.startswith(own) or (name.startswith('s11-') and not name.startswith(other)):
            duplicate.append(line)
        elif name.startswith(other):
            companion.append(line)
    if duplicate:
        raise RuntimeError('Existing same-stage Session 11 job(s): ' + '; '.join(duplicate))
    return dict(checked=True, same_stage_jobs=[], other_stage_jobs_allowed=companion)


def preflight(stage):
    checks = {}
    for label, fn in [
        ('submission_location', legacy.submission_location),
        ('native_inputs_and_tools', manifest if stage == 'production' else legacy.manifest),
        ('checkout', legacy.checkout), ('allocation', lambda: legacy.allocation(stage)),
        ('queue_limits', lambda: legacy.partition(stage)),
        ('existing_jobs', lambda: existing_jobs(stage)), ('modules', legacy.modules),
        ('storage', lambda: legacy.storage(stage)),
    ]:
        try:
            checks[label] = dict(passed=True, details=fn())
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
            checks[label] = dict(passed=False, error=str(exc))
    if stage == 'production':
        # Authorization is checked against the exact native/input hashes above.
        # This is a launch-policy override, never a manufactured validation PASS.
        checks['early_production_policy'] = dict(
            passed=checks['native_inputs_and_tools']['passed'], mode=MODE,
            gpu_validation_verified=False, push_verified=False)
        if not checks['early_production_policy']['passed']:
            checks['early_production_policy']['error'] = 'Early-production hash authorization is not verified'
    return checks


def show_checks(stage, checks):
    label = 'PRODUCTION (USER-AUTHORIZED, UNVALIDATED)' if stage == 'production' else 'VALIDATION'
    print(label + ': ' + ('ready for manual submission' if all(c['passed'] for c in checks.values()) else 'not ready'))
    for name, result in checks.items():
        print('  ' + ('OK ' if result['passed'] else 'BLOCKED ') + name.replace('_', ' ')
              + (': ' + result['error'] if not result['passed'] else ''))


def submit_job(stage):
    legacy.submission_location()
    folder = SESSION / 'evidence/submissions'; folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another submission is in progress; no job submitted') from exc
        legacy.pending_attempts(stage)
        checks = preflight(stage)
        show_checks(stage, checks)
        if not all(c['passed'] for c in checks.values()):
            raise RuntimeError('Preflight failed; no job submitted')
        env = os.environ.copy(); env['S11_ROOT'] = str(SESSION)
        if any(k.startswith('SBATCH_') for k in env):
            raise RuntimeError('Unset SBATCH_* environment overrides before using this launcher')
        args = legacy.sbatch_args(stage)
        if stage == 'production':
            args[-1] = str(HERE / 'production.sbatch')
            env['S11_PRODUCTION_MANIFEST_SHA256'] = digest(checks['native_inputs_and_tools']['details'])
            env['S11_PRODUCTION_COMMIT'] = checks['checkout']['details']['commit']
        dry = args.copy(); dry.insert(1, '--test-only'); dry.remove('--parsable')
        legacy.command(dry, cwd=SESSION, env=env, timeout=20)
        token = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:10]
        path = folder / (stage + '_' + token + '.json')
        record = dict(stage=stage, state='prepared', epoch=time.time(), command=args,
                      preflight=checks, automatic_retry=False,
                      mode=MODE if stage == 'production' else 'validation',
                      log_pattern=str(SESSION / 'slurm-%j.out'))
        if stage == 'production':
            record.update(gpu_validation_verified=False, push_verified=False,
                          submitted_manifest_sha256=env['S11_PRODUCTION_MANIFEST_SHA256'],
                          submitted_commit=env['S11_PRODUCTION_COMMIT'])
        atomic(path, record)
        try:
            result = subprocess.run(args, cwd=SESSION, env=env, capture_output=True, text=True, timeout=30)
            record.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
            if result.returncode:
                raise RuntimeError('Slurm submission failed or is ambiguous; inspect retained response before any retry')
            job = job_id(result.stdout)
            record.update(state='accepted', job_id=job,
                          log=str(SESSION / ('slurm-' + job + '.out')),
                          runtime=str(SESSION / 'runtime' / (stage + '_' + job)))
        except BaseException as exc:
            record.update(state='ambiguous', error=repr(exc))
            if isinstance(exc, subprocess.TimeoutExpired):
                def retained(value):
                    return value.decode(errors='replace') if isinstance(value, bytes) else value
                record.update(stdout=retained(exc.stdout), stderr=retained(exc.stderr))
            raise
        finally:
            atomic(path, record)
            with (SESSION / 'REPORT.md').open('a') as stream:
                stream.write(f"\nManual {stage} submission: `{path}`; state={record['state']}, "
                             f"job={record.get('job_id', 'unknown')}, mode={record['mode']}. "
                             "Raw sbatch response retained; no automatic retry or chained job.\n")
        print(f"Submitted {stage} job {job}\nLog: {record['log']}\n"
              f"Outputs: {record['runtime']}\nReceipt: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'validation', 'production'], nargs='?', default='check')
    args = parser.parse_args()
    try:
        if args.action == 'check':
            for stage in legacy.JOBS:
                checks = preflight(stage)
                try:
                    legacy.pending_attempts(stage)
                except RuntimeError as exc:
                    checks['previous_attempt'] = dict(passed=False, error=str(exc))
                show_checks(stage, checks)
            print('No job submitted. Production may be submitted manually while validation is pending.')
        else:
            submit_job(args.action)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
