"""Explicit early-production authorization, separate from frozen validation tools."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / 'session011'
sys.path.insert(0, str(LEGACY))
from prepare import SESSION, atomic, sha
from workflow import BRANCH, git, manifest as validation_manifest

MODE = 'user_authorized_unvalidated'


def manifest():
    current = validation_manifest()
    policy = json.loads((HERE / 'production_policy.json').read_text())
    if (policy.get('schema_version') != 1 or policy.get('mode') != MODE
            or set(policy.get('waived_prerequisites', [])) != {
                'gpu_validation_before_production', 'verified_push_before_production'}
            or policy.get('native_source_sha256') != current['native_source_sha256']
            or policy.get('gi_executable_sha256') != current['executable_sha256']['gi']
            or policy.get('production_input_sha256') != sha(LEGACY / 'inputs/production.athinput')
            or policy.get('physics_and_resource_guards_changed') is not False
            or policy.get('automatic_submission') is not False
            or policy.get('automatic_retry') is not False
            or policy.get('push_before_native_validation_authorized') is not False):
        raise RuntimeError('Early-production authorization does not match the current frozen model/build/input')
    paths = sorted([*HERE.glob('*.py'), *HERE.glob('*.sh'), *HERE.glob('*.sbatch'),
                    HERE / 'production_policy.json'])
    current['early_production'] = dict(
        mode=MODE, gpu_validation_verified=False, push_verified=False,
        user_authorization=policy,
        files={p.name: sha(p) for p in paths})
    return current


def digest(current):
    return hashlib.sha256(json.dumps(current, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def readiness():
    """Used only by the early production process; never changes validation or push."""
    current = manifest()
    expected_commit = os.environ.get('S11_PRODUCTION_COMMIT')
    amendment = None
    if os.environ.get('S11_PRODUCTION_MANIFEST_SHA256') != digest(current):
        from queued_binding import verify
        try:
            amendment = verify(current, digest(current))
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
            raise RuntimeError('Production hashes changed without a verified Slurm submission update: '+str(exc)) from exc
        expected_commit = amendment['replacement_commit']
    if (git('branch', '--show-current') != BRANCH or git('status', '--porcelain')
            or git('rev-parse', 'HEAD') != expected_commit):
        raise RuntimeError('Production checkout changed after submission')
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdigit():
        raise RuntimeError('Missing production allocation job ID')
    root = SESSION / 'runtime' / ('production_' + job)
    # The unchanged production runner creates this directory before readiness.
    if not root.is_dir():
        raise RuntimeError('Missing production run directory')
    atomic(root / 'RUN_MODE.json', dict(
        mode=MODE, job_id=job, commit=git('rev-parse', 'HEAD'), manifest=current,
        gpu_validation_verified=False, push_verified=False,
        validation_requirement_waived_by_user=True, push_requirement_waived_for_launch=True,
        validation_receipt_present=(SESSION / 'evidence/VALIDATION_PASS.json').exists(),
        push_receipt_present=(SESSION / 'evidence/PUSH_VERIFIED.json').exists(),
        scientifically_validated=False, resource_guards_retained=True, automatic_retry=False,
        submission_update=amendment))
    return current
