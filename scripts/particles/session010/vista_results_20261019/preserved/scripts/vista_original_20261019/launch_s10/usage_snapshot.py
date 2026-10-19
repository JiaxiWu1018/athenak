#!/usr/bin/env python3
"""Read Slurm allocation rows and estimate Vista's published minimum/rate charges."""
import datetime as dt
import json
import os
from pathlib import Path
import pwd
import subprocess

import s10

RATES = {"gh": 1.0, "gh-dev": 1.0, "gg": .33}
CONFIRMED = {"1069850", "1069961", "1069970", "1069971", "1069972", "1069976", "1070201", "1071690"}


def estimate(row):
    raw = row["nodes"] * row["elapsed_seconds"] / 3600
    billed = row["nodes"] * max(row["elapsed_seconds"] / 3600, .25) if row["nodes"] else 0
    rate = RATES.get(row["partition"])
    if rate is None:
        raise ValueError("unknown queue charging rate")
    return dict(row, actual_node_hours=raw, minimum_billed_node_hours=billed,
                published_rate_su_per_node_hour=rate, estimated_service_units=billed * rate,
                confirmed_session=row["job_id"] in CONFIRMED,
                allocation_finished=not row["state"].startswith(("RUNNING", "PENDING", "COMPLETING", "CONFIGURING")))


def snapshot():
    user = pwd.getpwuid(os.getuid()).pw_name
    command = ["sacct", "-n", "-X", "-P", "--starttime", "2026-10-13T14:05:00", "--user", user,
               "--accounts=AST21006", "--format=JobIDRaw,Partition,State,ElapsedRaw,AllocNodes,Start,End"]
    p = subprocess.run(command, text=True, capture_output=True, check=True, timeout=30)
    rows = []
    for line in p.stdout.splitlines():
        a = line.split("|")
        if len(a) >= 7 and a[0].isdigit() and a[3].isdigit() and a[4].isdigit():
            rows.append(estimate(dict(job_id=a[0], partition=a[1], state=a[2], elapsed_seconds=int(a[3]), nodes=int(a[4]), start=a[5], end=a[6])))
    if len(rows) != len({r["job_id"] for r in rows}):
        raise ValueError("duplicate allocation row")
    gh = [r for r in rows if r["job_id"] in {"1069961", "1069970", "1069971", "1069972"}]
    confirmed = [r for r in rows if r["confirmed_session"]]
    receipt = dict(as_of_utc=dt.datetime.now(dt.timezone.utc).isoformat(), rows=rows,
            simulation_chain_actual_node_hours=sum(r["actual_node_hours"] for r in gh),
            simulation_chain_billed_gh_node_hours=sum(r["minimum_billed_node_hours"] for r in gh),
            confirmed_session_estimated_service_units=sum(r["estimated_service_units"] for r in confirmed),
            all_vista_allocations_in_period_estimated_service_units=sum(r["estimated_service_units"] for r in rows),
            all_vista_allocations_in_period_actual_node_hours=sum(r["actual_node_hours"] for r in rows),
            tariff_source="https://docs.tacc.utexas.edu/hpc/vista/#job-accounting",
            scope_note="1070169 is a nearby GG allocation whose Session010 association is unproven; included only in the full-period total.",
            charge_note="Estimate from actual Slurm allocation rows and posted 15-minute minimum plus GH=1/GG=.33 rates, not a portal invoice. The current GG allocation is provisional.", command=command)
    root = s10.HANDOFF / "runtime_s10/repair_20261014"
    root.mkdir(exist_ok=True)
    s10.atomic(root / "USAGE.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    snapshot()
