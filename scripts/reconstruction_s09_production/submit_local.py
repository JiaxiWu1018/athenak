"""Budgeted, registered submissions; two GPU lanes and one serial support lane."""
import argparse
import datetime
import fcntl
import json
import shlex
import subprocess
from status_local import SESSION, AMD, refresh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['support', 'production'])
    parser.add_argument('--case', choices=['yc0283', 'yc0605', 'yc0819', 'yc0900'])
    parser.add_argument('--hours', type=float, default=2)
    parser.add_argument('--review-root')
    parser.add_argument('args', nargs='*')
    args = parser.parse_args()
    path = SESSION / 'evidence/job_registry.json'
    with (SESSION / 'evidence/.submission.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        snapshot = refresh()
        registry = json.loads(path.read_text())
        seconds = 1800 if args.kind == 'support' else int(args.hours * 3600)
        assert 0 < seconds <= 28800
        # Preserve room for destination verification and scheduler teardown;
        # production also leaves one node-hour for final review work.
        reserve = 1.5 if args.kind == 'production' else .55
        assert snapshot['committed_node_hour_upper'] + seconds / 3600 + reserve <= 40
        if args.kind == 'support':
            partition, script = 'devel', 'support.sbatch'
            previous = [j for j in registry['jobs'] if j['kind'] != 'production']
        else:
            assert args.case
            partition = 'mi2104x' if args.case in ['yc0283', 'yc0819'] else 'mi2508x'
            script = args.case + '.sbatch'
            previous = [j for j in registry['jobs']
                        if j['kind'] == 'production' and j['partition'] == partition]
        command = ['sbatch', '--parsable', f'--time={seconds // 60}']
        if previous:
            command.append('--dependency=afterany:' + previous[-1]['job_id'])
        if args.review_root:
            assert args.review_root.startswith(AMD + '/')
            command.append('--export=ALL,ST9_REVIEW_ROOT=' + args.review_root)
        command += [AMD + '/scripts/' + script] + args.args
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'hpcfund.amd.com',
                                 shlex.join(command)], capture_output=True, text=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
        (SESSION / 'logs' / f'submit_{stamp}.txt').write_text(result.stdout + '\n' + result.stderr)
        assert result.returncode == 0, result.stderr
        ident = result.stdout.strip().splitlines()[-1].split(';')[0]
        assert ident.isdigit(), result.stdout
        registry['jobs'].append(dict(job_id=ident, kind=args.kind, partition=partition,
                                     nodes=1, requested_seconds=seconds, script=script,
                                     command=command, case=args.case,
                                     submitted_UTC=stamp))
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(registry, indent=2) + '\n')
        temporary.replace(path)
        subprocess.run(['rsync', '-a', str(path),
                        'hpcfund.amd.com:' + AMD + '/evidence/'], check=True)
        subprocess.run(['rsync', '-a', str(SESSION / 'evidence/scheduler_receipts'),
                        'hpcfund.amd.com:' + AMD + '/evidence/'], check=True)
        print('REGISTERED', ident, shlex.join(command))


if __name__ == '__main__':
    main()
