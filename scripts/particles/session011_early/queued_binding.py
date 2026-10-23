"""Verify a user-amended queued submission against its Slurm Comment digest."""
import json
import os
import re
import subprocess

from prepare import SESSION, sha


def job_info(job):
    result = subprocess.run(['scontrol', 'show', 'job', job, '-o'], capture_output=True,
                            text=True, timeout=15)
    if result.returncode:
        raise RuntimeError('Cannot verify Slurm submission update: ' + result.stderr.strip())
    fields = dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]*)', result.stdout))
    if fields.get('JobId') != job:
        raise RuntimeError('Unexpected job record from Slurm')
    return fields, result.stdout


def verify(current, current_digest):
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdigit():
        raise RuntimeError('Missing job ID for submission update')
    path = SESSION/'evidence/production_updates'/(job+'.json')
    binding = json.loads(path.read_text())
    original_path = SESSION/'evidence/submissions'/binding['original_submission_file']
    if original_path.parent != SESSION/'evidence/submissions':
        raise RuntimeError('Invalid original submission path')
    original = json.loads(original_path.read_text())
    submitted = original['preflight']['native_inputs_and_tools']['details']
    if (binding.get('schema_version') != 1 or binding.get('job_id') != job
            or binding.get('mode') != 'user_authorized_unvalidated'
            or binding.get('gpu_validation_verified') is not False
            or binding.get('push_verified') is not False
            or binding.get('native_or_physics_changed') is not False
            or binding.get('original_submission_sha256') != sha(original_path)
            or original.get('job_id') != job or original.get('state') != 'accepted'
            or binding.get('original_manifest_sha256') != original['submitted_manifest_sha256']
            or binding.get('original_commit') != original['submitted_commit']
            or binding.get('original_manifest_sha256') != os.environ.get('S11_PRODUCTION_MANIFEST_SHA256')
            or binding.get('original_commit') != os.environ.get('S11_PRODUCTION_COMMIT')
            or binding.get('replacement_manifest_sha256') != current_digest
            or binding.get('replacement_manifest') != current
            or current['native_source_sha256'] != submitted['native_source_sha256']
            or current['kokkos_commit'] != submitted['kokkos_commit']
            or {p: h for p, h in current['files'].items() if '/inputs/' in p}
               != {p: h for p, h in submitted['files'].items() if '/inputs/' in p}):
        raise RuntimeError('Queued submission update does not match the original and replacement hashes')
    fields, raw = job_info(job)
    if fields.get('Comment') != 'S11_UPDATE_'+sha(path):
        raise RuntimeError('Replacement hashes have not been bound to this job in Slurm')
    return dict(receipt=str(path), receipt_sha256=sha(path),
                original_manifest_sha256=binding['original_manifest_sha256'],
                original_commit=binding['original_commit'],
                replacement_manifest_sha256=current_digest,
                replacement_commit=binding['replacement_commit'],
                slurm_comment=fields['Comment'], slurm_job_record=raw)
