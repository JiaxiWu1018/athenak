"""User-run update of one pending job's hash binding. Never submits or cancels."""
import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
import uuid

from early_gate import SESSION, MODE, digest
from early_submit import legacy, preflight, show_checks
from prepare import atomic, sha
from queued_binding import job_info


def pending_job(job):
    fields, raw = job_info(job)
    if fields.get('JobState') != 'PENDING':
        raise RuntimeError('Job is no longer pending; executable updates are refused')
    if (fields.get('Account', '').upper() != legacy.ACCOUNT
            or fields.get('Partition') != 'gh'
            or fields.get('NumNodes') != '16' or fields.get('NumTasks') != '16'
            or legacy.duration(fields.get('TimeLimit', '00:00:00')) != 36*3600
            or fields.get('WorkDir') != str(SESSION)
            or not fields.get('JobName', '').startswith('s11-final-')
            or not fields.get('UserId', '').endswith('('+str(os.getuid())+')')):
        raise RuntimeError('Queued job does not match the approved owner, resource and output contract')
    return fields, raw


def update(job):
    if not re.fullmatch(r'\d+', job):
        raise RuntimeError('Specify the queued production job ID')
    legacy.submission_location()
    folder = SESSION/'evidence/submissions'
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before, raw = pending_job(job)
        paths = list(folder.glob('production_*.json'))
        if not paths:
            raise RuntimeError('No original production submission receipt')
        original_path = max(paths, key=lambda p: json.loads(p.read_text())['epoch'])
        original = json.loads(original_path.read_text())
        if original.get('job_id') != job or original.get('state') != 'accepted':
            raise RuntimeError('Only the latest accepted production attempt can be updated')
        checks = preflight('production', updating_job=job)
        show_checks('production', checks)
        if not all(c['passed'] for c in checks.values()):
            raise RuntimeError('Preflight failed; job has not been updated')
        current = checks['native_inputs_and_tools']['details']
        old = original['preflight']['native_inputs_and_tools']['details']
        if (current['native_source_sha256'] != old['native_source_sha256']
                or current['kokkos_commit'] != old['kokkos_commit']
                or {p: h for p, h in current['files'].items() if '/inputs/' in p}
                   != {p: h for p, h in old['files'].items() if '/inputs/' in p}):
            raise RuntimeError('Same-job repair requires unchanged native physics, Kokkos and inputs')
        for kind in ('gi', 'os', 'spin'):
            receipt = json.loads((SESSION/f'evidence/build/{kind}.json').read_text())
            if receipt.get('clean_build') is not True:
                raise RuntimeError('A fresh native build is required: '+kind)
        root = SESSION/'evidence/production_updates'; root.mkdir(exist_ok=True)
        path = root/(job+'.json')
        binding = dict(schema_version=1, job_id=job, mode=MODE, epoch=time.time(),
                       authorization='User requested repairing the executable while retaining the pending production job.',
                       original_submission_file=original_path.name,
                       original_submission_sha256=sha(original_path),
                       original_manifest_sha256=original['submitted_manifest_sha256'],
                       original_commit=original['submitted_commit'],
                       replacement_manifest=current, replacement_manifest_sha256=digest(current),
                       replacement_commit=checks['checkout']['details']['commit'],
                       native_or_physics_changed=False, gpu_validation_verified=False,
                       push_verified=False, automatic_submission=False, automatic_retry=False)
        if path.exists():
            saved = json.loads(path.read_text())
            if {k: v for k, v in saved.items() if k != 'epoch'} != {k: v for k, v in binding.items() if k != 'epoch'}:
                raise RuntimeError('A different retained update already exists; review it before further changes')
            binding = saved
        else:
            atomic(path, binding)
        tag = 'S11_UPDATE_'+sha(path)
        attempt = root/(job+'_controller_'+uuid.uuid4().hex[:10]+'.json')
        record = dict(job_id=job, binding=str(path), binding_sha256=sha(path),
                      epoch=time.time(), state='prepared', preflight=checks,
                      before=raw, automatic_submission=False, automatic_retry=False)
        atomic(attempt, record)
        try:
            # No hold, release or priority change; only attach the immutable new hashes.
            pending_job(job)
            result = subprocess.run(['scontrol', 'update', 'JobId='+job, 'Comment='+tag],
                                    capture_output=True, text=True, timeout=20)
            record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            if result.returncode:
                raise RuntimeError('Slurm rejected the update; original and replacement receipts retained')
            after, after_raw = job_info(job)
            record['after'] = after_raw
            if after.get('Comment') != tag:
                raise RuntimeError('Cannot verify updated Slurm hash binding; inspect retained receipt')
            record['state'] = 'verified'
        except BaseException as exc:
            record.update(state='failed_or_ambiguous', error=repr(exc))
            raise
        finally:
            atomic(attempt, record)
            with (SESSION/'REPORT.md').open('a') as out:
                out.write(f'\nUser-run production update for job {job}: {record["state"]}; '
                          f'binding `{path}`, controller receipt `{attempt}`. '
                          'Native source/physics/inputs/resources unchanged; GPU validation and push unverified. '
                          'No submission, cancellation, hold, retry or priority change.\n')
        print(f'Updated pending production job {job}; keep this job, no resubmission needed.\n'
              f'New executable SHA256: {current["executable_sha256"]["gi"]}\nReceipt: {attempt}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job_id')
    args = parser.parse_args()
    try:
        update(args.job_id)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
