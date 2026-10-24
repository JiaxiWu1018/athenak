"""Exact terminal accounting with explicit reservations for unresolved jobs."""
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'PREEMPTED',
            'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def summarize(registry, scheduler_text):
    records = {}
    for line in scheduler_text.splitlines():
        values = line.split('|')
        if len(values) >= 7:
            records[values[0]] = values
    rows = []
    for entry in registry['jobs']:
        fields = records.get(entry['job_id'])
        upper = entry['nodes'] * entry['requested_seconds'] / 3600
        status = 'UNKNOWN'
        basis = 'requested allocation upper bound; terminal evidence unavailable'
        exact = None
        if fields:
            status = fields[1].split()[0].rstrip('+')
            if status in TERMINAL:
                elapsed, nodes = int(fields[3]), int(fields[4])
                assert elapsed >= 0 and nodes >= 0
                assert elapsed == 0 or nodes > 0
                exact = elapsed * nodes / 3600
                upper, basis = exact, 'terminal scheduler'
            elif status == 'PENDING':
                basis = 'pending requested allocation reservation'
            else:
                basis = 'active requested allocation upper bound'
        rows.append(dict(job_id=entry['job_id'], kind=entry['kind'],
                         partition=entry['partition'], status=status,
                         upper_node_hours=upper, exact_node_hours=exact,
                         basis=basis, scheduler_fields=fields))
    return dict(jobs=rows, allocated_node_hour_upper=sum(r['upper_node_hours'] for r in rows),
                exact_terminal_node_hours=sum(r['exact_node_hours'] or 0 for r in rows),
                all_terminal_evidence=all(r['status'] in TERMINAL for r in rows),
                unresolved_job_ids=[r['job_id'] for r in rows if r['status'] not in TERMINAL],
                scope='Allocated node-hours, including failed work. Reservations and missing-terminal bounds are explicit; not an invoice.')
