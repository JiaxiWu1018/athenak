"""Use actual Slurm allocation metadata rather than stale exported rank counts."""
import re


def actual_ranks(raw, policy):
    values = dict(re.findall(r'\b(NumTasks|NumNodes|Partition|OverSubscribe)=(\S+)', raw))
    assert values['NumNodes'] == '1' and values['OverSubscribe'] == 'NO'
    partition = values['Partition']
    assert partition in policy['allowed_launches']
    ranks = int(values['NumTasks'])
    assert ranks == policy['allowed_launches'][partition]['ranks']
    return ranks


def main():
    from common import job, resource_policy, subprocess, atomic, EXT, now, os
    raw = subprocess.check_output(['scontrol', 'show', 'job', job(), '-o'], text=True)
    ranks = actual_ranks(raw, resource_policy())
    names = ['SLURM_NTASKS', 'SLURM_NPROCS', 'SLURM_TASKS_PER_NODE', 'SLURM_NTASKS_PER_NODE']
    atomic(EXT/'evidence/allocations'/job()/'LAUNCH_LAYOUT.json', dict(
        utc=now(), job=job(), actual_ranks=ranks, scheduler=raw.strip(),
        inherited_environment={k: os.environ.get(k) for k in names},
        normalized_environment={k: str(ranks) for k in names},
        reason='Single-node native MPI launch follows actual allocated ranks'))
    print(ranks)


if __name__ == '__main__':
    main()
