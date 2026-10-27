"""Bound workflow clean-stop timing by the actual allocated Slurm limit."""
import re


def limit_seconds(value):
    days = 0
    if '-' in value:
        day, value = value.split('-', 1)
        assert day.isdigit()
        days = int(day)
    parts = value.split(':')
    assert 1 <= len(parts) <= 3 and all(p.isdigit() for p in parts), value
    fields = list(map(int, parts))
    if len(fields) == 3:
        hours, minutes, seconds = fields
    elif len(fields) == 2:
        hours, minutes, seconds = 0, fields[0], fields[1]
    else:
        hours, minutes, seconds = 0, fields[0], 0
    assert 0 <= minutes < 60 or len(fields) < 3
    assert 0 <= seconds < 60
    total = days*86400+hours*3600+minutes*60+seconds
    assert total > 0
    return total


def scheduler_limit(text, ident):
    assert ident.isdigit()
    assert re.search(r'\bJobId='+re.escape(ident)+r'\b', text)
    match = re.search(r'\bTimeLimit=(\S+)', text)
    assert match, 'Missing actual Slurm TimeLimit'
    return limit_seconds(match[1])


def main():
    from common import atomic, job, now, EXT, os, subprocess, load
    ident = job()
    inherited_environment_seconds = int(os.environ.get('ST8X_WALL_SECONDS', '7200'))
    ticket = next(x for x in load(EXT/'evidence/job_registry.json') if x['id'] == ident)
    requested = int(ticket['requested_seconds'])
    assert 0 < requested <= 4*86400  # Site wall limit; the campaign has no compute cap.
    raw = subprocess.check_output(['scontrol', 'show', 'job', '-o', ident],
        text=True, timeout=15)
    allocated = scheduler_limit(raw, ident)
    effective = min(requested, allocated)
    atomic(EXT/'evidence/allocations'/ident/'WALL_BUDGET.json', dict(
        utc=now(), job=ident, configured_seconds=requested,
        inherited_environment_seconds=inherited_environment_seconds,
        configuration_source='Registered request, including explicit pending-job wall updates',
        actual_slurm_limit_seconds=allocated, effective_workflow_seconds=effective,
        clean_stop_reserve_seconds=240,
        scheduler=raw.strip(),
        scope='Workflow elapsed clock starts before modules; trajectory subtracts a further240s clean-stop reserve. Slurm remains the allocation hard limit.'))
    print(effective)


if __name__ == '__main__':
    main()
