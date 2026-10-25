"""Post-terminal metadata addendum after actual allocated destination verification."""
import argparse
import datetime
import fcntl
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from accounting import summarize, TERMINAL
from status_local import SESSION, AMD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Perseus-job', required=True)
    args = parser.parse_args()
    assert args.Perseus_job.isdigit()
    mirror = SESSION / 'review_mirror'
    verification = json.loads((mirror / 'evidence/THIN_REVIEW_VERIFIED.json').read_text())
    assert verification['destination_SHA256_verified']
    assert verification['job'] == args.Perseus_job
    manifest = mirror / 'evidence/REVIEW_PRODUCT_MANIFEST.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == verification['source_manifest_sha256']
    acceptance_path = mirror / 'evidence/FINAL_ACCEPTANCE.json'
    acceptance = json.loads(acceptance_path.read_text())
    assert acceptance['all_five_numerical_acceptance'] and acceptance['actual_PDF_and_encoded_frame_inspection']
    assert acceptance['resource_limits_passed']
    remote = '''
import json,subprocess,sys,os,time
from pathlib import Path
os.environ['TZ']='UTC';time.tzset()
root=Path(AMD_ROOT);sys.path.insert(0,str(root/'scripts'))
from accounting import summarize
registry=json.loads((root/'evidence/job_registry.json').read_text())
ids=','.join(r['job_id'] for r in registry['jobs'])
query=subprocess.run(['sacct','-X','-n','-P','-j',ids,'-o','JobIDRaw,State,ExitCode,ElapsedRaw,AllocNodes,Start,End'],capture_output=True,text=True)
assert query.returncode==0,query.stderr
preserved={p.stem:json.loads(p.read_text())['scheduler_fields'] for p in (root/'evidence/scheduler_receipts/amd').glob('*.json')}
account=summarize(registry,query.stdout,preserved)
queue=subprocess.run(['squeue','-u','jiaxiwu','-h','-o','%i|%T|%D'],capture_output=True,text=True)
assert queue.returncode==0,queue.stderr
known={r['job_id'] for r in registry['jobs']}
assert not [r for r in queue.stdout.splitlines() if r.split('|')[0] in known],queue.stdout
print(json.dumps(dict(account=account,scheduler_stdout=query.stdout,scheduler_stderr=query.stderr,queue_stdout=queue.stdout,registry=registry)))
'''
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'hpcfund.amd.com', 'python3 -'],
                            input='AMD_ROOT=' + repr(AMD) + '\n' + remote,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    amd = json.loads(result.stdout)
    env = dict(__import__('os').environ, TZ='UTC')
    query = subprocess.run(['sacct', '-X', '-n', '-P', '-j', args.Perseus_job, '-o',
                            'JobIDRaw,State,ExitCode,ElapsedRaw,AllocNodes,Start,End'],
                           capture_output=True, text=True, env=env)
    accounting_disabled = query.returncode != 0
    if accounting_disabled:
        assert 'Slurm accounting storage is disabled' in query.stderr, query.stderr
    queued = subprocess.run(['squeue', '-u', 'jiaxiwu', '-h', '-o', '%i|%T|%D'],
                            capture_output=True, text=True, env=env)
    assert queued.returncode == 0, queued.stderr
    assert not [row for row in queued.stdout.splitlines() if row.split('|')[0] == args.Perseus_job]
    ps_registry = json.loads((SESSION / 'evidence/perseus_jobs.json').read_text())
    assert [row['job_id'] for row in ps_registry['jobs']] == [args.Perseus_job]
    preserved_ps = {p.stem: json.loads(p.read_text())['scheduler_fields']
                    for p in (SESSION / 'evidence/scheduler_receipts/perseus').glob('*.json')}
    ps = summarize(ps_registry, '' if accounting_disabled else query.stdout, preserved_ps)
    if accounting_disabled:
        assert ps['all_terminal_evidence'], 'Preserved terminal Perseus receipt required'
    combined = amd['account']['allocated_node_hour_upper'] + ps['allocated_node_hour_upper']
    assert combined <= 40
    raw = json.loads((mirror / 'evidence/RAW_RETENTION_SUMMARY.json').read_text())
    storage_upper = (raw['storage']['own'] + verification['allocated_Perseus_session_bytes'] +
                     acceptance['closing_reserve_bytes'])
    assert storage_upper < 400 * 1024 ** 3
    publisher = acceptance['job']
    assert publisher in {row['job_id'] for row in amd['account']['jobs']}
    post = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                AMD_accounting=amd, Perseus_accounting=ps,
                Perseus_scheduler_stdout=query.stdout, Perseus_scheduler_stderr=query.stderr,
                Perseus_accounting_storage_disabled=accounting_disabled,
                Perseus_preserved_terminal_receipts=preserved_ps,
                Perseus_queue_stdout=queued.stdout,
                finalizer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                allocated_node_hour_upper=combined, combined_storage_upper_bytes=storage_upper,
                closing_reserve_bytes=acceptance['closing_reserve_bytes'],
                publisher_job=publisher, destination_verification_job=args.Perseus_job,
                all_registered_handles_quiescent=True,
                destination_manifest_sha256=verification['source_manifest_sha256'],
                scope='Fresh post-terminal metadata. Missing terminal rows retain requested bounds; quiescence is confirmed from actual scheduler handles. Not an invoice. No additional data deletion or native evolution.')
    with (SESSION / 'evidence/.closeout.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        evidence = mirror / 'evidence'
        post_path = evidence / 'FINAL_POST_TERMINAL_ACCOUNTING.json'
        post_path.write_text(json.dumps(post, indent=2) + '\n')
        acceptance.update(destination_copy_verified=True, publisher_terminal_addendum_required=False,
                          post_terminal_accounting_sha256=hashlib.sha256(post_path.read_bytes()).hexdigest(),
                          allocated_node_hour_upper=combined,
                          combined_storage_upper_GiB=storage_upper / 1024 ** 3,
                          all_registered_handles_quiescent=True, complete=True, remaining=[])
        acceptance_path.write_text(json.dumps(acceptance, indent=2) + '\n')
        files = [post_path, acceptance_path, evidence / 'THIN_REVIEW_VERIFIED.json']
        subprocess.run(['rsync', '-a'] + [str(p) for p in files] +
                       ['hpcfund.amd.com:' + AMD + '/final/evidence/'], check=True)
        checks = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        verifier = ('import json,hashlib;from pathlib import Path;' +
                    'root=Path(' + repr(AMD + '/final/evidence') + ');' +
                    'expected=' + repr(checks) + ';' +
                    'actual={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in expected};' +
                    'assert actual==expected;print(json.dumps(actual))')
        verify = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'hpcfund.amd.com',
                                 'python3 -c ' + shlex.quote(verifier)],
                                capture_output=True, text=True)
        assert verify.returncode == 0, verify.stderr
        assert json.loads(verify.stdout) == checks
        closure = dict(UTC=post['UTC'], complete=True, metadata_SHA256_verified_at_AMD_and_Perseus=checks,
                       fixed_product_manifest_sha256=verification['source_manifest_sha256'],
                       allocated_node_hour_upper=combined, combined_storage_upper_bytes=storage_upper,
                       all_registered_handles_quiescent=True)
        (SESSION / 'evidence/COMPLETE.json').write_text(json.dumps(closure, indent=2) + '\n')
    print(json.dumps(closure, indent=2))


if __name__ == '__main__':
    main()
