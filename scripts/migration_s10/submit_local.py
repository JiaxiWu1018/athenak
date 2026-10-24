"""Registered one-node submissions with a strict50h requested-time ceiling."""
import argparse
import datetime
import fcntl
import json
import shlex
import subprocess
from pathlib import Path

LOCAL = Path('/data/jiaxiwu/NRPIC/ST_migration/session_10_yc0500_outward_20261024')
REMOTE = '/work1/eliasmost/jiaxiwu/st_migration_s10_20261024'

def atomic(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('role', choices=['reference', 'production', 'analysis', 'products'])
    p.add_argument('--case', choices=['yc0500_outward', 'yc0500_control'], default='yc0500_outward')
    p.add_argument('--target', type=float, default=20.)
    p.add_argument('--minutes', type=int)
    p.add_argument('--after', default='')
    a = p.parse_args()
    assert 0 < a.target <= 1500
    seconds = 60*(a.minutes or dict(reference=30, production=120, analysis=60, products=120)[a.role])
    assert 0 < seconds <= 7200
    registry = LOCAL/'evidence/job_registry.json'
    with (LOCAL/'evidence/submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = json.loads(registry.read_text())
        # Requested-time reservation is intentionally stricter than elapsed accounting.
        assert sum(row['requested_seconds'] for row in rows)+seconds <= 50*3600, 'Hard50h reservation exhausted'
        if a.after:
            assert all(i.isdigit() for i in a.after.split(':'))
        serial = [r for r in rows if r['role'] == 'production'] if a.role == 'production' else []
        dependency = a.after or (serial[-1]['id'] if serial else '')
        comment = f'st10:20261024:{a.role}:{len(rows)+1:03d}'
        command = ['sbatch', '--parsable', f'--time={seconds//60}', '--comment='+comment,
            '--export=ALL,S10_CASE='+a.case+',S10_ENDPOINT='+str(a.target)+',S10_U='+('0' if a.case.endswith('control') else '0.005')]
        if dependency:
            command.append('--dependency=afterok:'+dependency)
        command.append(REMOTE+'/scripts/'+a.role+'.sbatch')
        ticket = dict(role=a.role, case=a.case, target=a.target, requested_seconds=seconds,
            nodes=1, comment=comment, command=command,
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        ticket_path = LOCAL/'evidence'/f'submission_{len(rows)+1:03d}.json'
        assert not ticket_path.exists(), 'Unresolved prior submission ticket; check scheduler before retry'
        atomic(ticket_path, ticket)
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            'hpcfund.amd.com', shlex.join(command)], capture_output=True, text=True, timeout=60)
        (LOCAL/'logs'/f'submission_{len(rows)+1:03d}.log').write_text(result.stdout+result.stderr)
        assert result.returncode == 0, result.stderr
        ident = result.stdout.strip().splitlines()[-1].split(';')[0]
        assert ident.isdigit(), result.stdout
        ticket['id'] = ident
        rows.append(ticket)
        atomic(registry, rows)
        atomic(ticket_path, ticket)
        upload = 'python3 -c '+shlex.quote('from pathlib import Path; import sys; p=Path("'+REMOTE+'/evidence/job_registry.json"); t=p.with_suffix(".upload"); t.write_text(sys.stdin.read()); t.replace(p)')
        subprocess.run(['ssh', '-o', 'BatchMode=yes', 'hpcfund.amd.com', upload],
            input=registry.read_text(), text=True, check=True, timeout=60)
        print(json.dumps(ticket, indent=2))

if __name__ == '__main__':
    main()
