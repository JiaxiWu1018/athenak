"""Finalize verified progress before interpreting stop intent; no submission."""
from copy import deepcopy
import math

def finalize(state, checkpoint, accounting, stop_reason, segment):
    result=deepcopy(state)
    required=("path","sha256","bytes","time","cycle")
    if not all(k in checkpoint for k in required):raise ValueError("unverified checkpoint receipt")
    if not math.isfinite(checkpoint["time"]) or checkpoint["bytes"]<=0:raise ValueError("invalid checkpoint")
    if checkpoint["time"]<state.get("time",0) or checkpoint["cycle"]<state.get("cycle",0):raise ValueError("regressing checkpoint")
    result.update(time=checkpoint["time"],cycle=checkpoint["cycle"],checkpoint=deepcopy(checkpoint),segments_completed=max(segment,state.get("segments_completed",0)),status=stop_reason,resources=deepcopy(accounting))
    return result

def allocated_node_hours(rows):
    seen=set();total=0.0
    for r in rows:
        if r["job_id"] in seen:raise ValueError("duplicate allocation accounting")
        seen.add(r["job_id"])
        if r["elapsed_seconds"]<0 or r["nodes"]<0:raise ValueError("invalid allocation accounting")
        total+=r["elapsed_seconds"]*r["nodes"]/3600
    return total
