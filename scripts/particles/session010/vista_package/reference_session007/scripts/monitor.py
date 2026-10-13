#!/usr/bin/env python3
"""Low-cost in-allocation monitor. It never performs particle reductions."""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import pathlib
import subprocess
import tempfile
import time


def atomic_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def bytes_used(path: pathlib.Path) -> int:
    proc = subprocess.run(
        ["du", "-sb", str(path)], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, check=False,
    )
    try:
        return int(proc.stdout.split()[0])
    except (IndexError, ValueError):
        return -1


def latest_tracker_mtime(run: pathlib.Path) -> float:
    files = list(run.rglob("*.co_0.txt")) + list(run.rglob("*.co_1.txt"))
    return max((x.stat().st_mtime for x in files), default=0.0)


def last_tracker(run: pathlib.Path, index: int):
    matches = sorted(run.rglob(f"*.co_{index}.txt"))
    if not matches:
        return None
    last = None
    with matches[-1].open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.strip() and not line.lstrip().startswith("#"):
                fields = line.split()
                if len(fields) >= 5:
                    last = (int(fields[0]), float(fields[1]), *(float(x) for x in fields[2:5]))
    return last


def accepted_common_horizon(run: pathlib.Path, cfg: dict) -> dict:
    required = int(cfg.get("common_horizon_persistence_records", 3))
    tolerance = float(cfg.get("common_horizon_tracker_time_tolerance", 0.2))
    matches = sorted(run.rglob("*.horizon_consumer_2.csv"))
    trackers = (last_tracker(run, 0), last_tracker(run, 1))
    result = {"qualified": False, "records": 0, "reason": "unavailable"}
    if not matches or any(row is None for row in trackers):
        return result
    with matches[-1].open("r", encoding="utf-8", errors="replace", newline="") as stream:
        rows = list(csv.DictReader(stream))
    accepted = []
    for row in reversed(rows):
        try:
            ok = (int(row["published_this_candidate"]) == 1
                  and int(row["quality_persist_ok"]) == 1
                  and int(row.get("association_ok", "1")) == 1)
        except (KeyError, TypeError, ValueError):
            break
        if not ok:
            break
        accepted.append(row)
        if len(accepted) >= required:
            break
    result["records"] = len(accepted)
    if len(accepted) < required:
        result["reason"] = "insufficient_consecutive_accepted_records"
        return result
    row = accepted[0]
    try:
        t = float(row["time"]); rmin = float(row["rmin"])
        center = tuple(float(row[f"center_{axis}"]) for axis in "xyz")
        distances = [math.sqrt(sum((tr[a + 2] - center[a])**2 for a in range(3)))
                     for tr in trackers]
        current = all(abs(tr[1] - t) <= tolerance for tr in trackers)
        encloses = all(distance < rmin for distance in distances)
        finite = math.isfinite(t) and math.isfinite(rmin) and rmin > 0 and all(
            math.isfinite(x) for x in (*center, *distances))
    except (KeyError, TypeError, ValueError):
        result["reason"] = "malformed_common_horizon_or_tracker_record"
        return result
    result.update({"time": t, "rmin": rmin, "center": center,
                   "tracker_distances": distances, "time_matched": current,
                   "encloses_both": encloses, "qualified": finite and current and encloses,
                   "reason": "qualified" if finite and current and encloses else
                             "coverage_or_time_gate_failed"})
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=pathlib.Path)
    p.add_argument("--run", required=True, type=pathlib.Path)
    p.add_argument("--campaign-root", required=True, type=pathlib.Path)
    p.add_argument("--log", required=True, type=pathlib.Path)
    p.add_argument("--started-epoch", required=True, type=float)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cfg = {k: os.path.expandvars(v) if isinstance(v, str) else v for k, v in cfg.items()}
    interval = int(cfg["monitor_seconds"])
    startup_grace = int(cfg["startup_telemetry_grace_seconds"])
    stale_grace = int(cfg["telemetry_grace_seconds"])
    warn = int(cfg["storage_warn_bytes"])
    stop = int(cfg["storage_stop_bytes"])
    control = pathlib.Path(cfg["work_root"]) / "control"
    telemetry = control / "telemetry.json"

    while True:
        now = time.time()
        storage = bytes_used(args.campaign_root)
        mtime = latest_tracker_mtime(args.run)
        age = None if mtime == 0.0 else now - mtime
        reasons = []
        common = accepted_common_horizon(args.run, cfg)
        if bool(cfg.get("authorized_early_stop", False)) and common["qualified"]:
            reasons.append("authorized_scientific_stop")
        if (control / "STOP_REQUESTED").exists():
            reasons.append("user_stop")
        if storage >= stop:
            reasons.append("resource_limit_storage")
        if now - args.started_epoch > startup_grace and (
                mtime == 0.0 or (age is not None and age > stale_grace)):
            reasons.append("diagnostic_stale")
        fatal = False
        if args.log.exists():
            with args.log.open("rb") as stream:
                stream.seek(max(0, args.log.stat().st_size - 262144))
                tail = stream.read().decode("utf-8", errors="replace")
            fatal = "### FATAL ERROR" in tail or "PARTICLE LEDGER FAIL" in tail
            if fatal:
                reasons.append("numerical_failure")
        if reasons:
            (args.run / "REQUEST_STOP").touch(exist_ok=True)
        atomic_json(telemetry, {
            "campaign_id": cfg["campaign_id"],
            "wall_elapsed_seconds": now - args.started_epoch,
            "storage_bytes": storage,
            "storage_warning": storage >= warn,
            "latest_tracker_age_seconds": age,
            "fatal_pattern": fatal,
            "common_horizon_stop_gate": common,
            "authorized_common_horizon_stop": (
                bool(cfg.get("authorized_early_stop", False)) and common["qualified"]
            ),
            "stop_reasons": reasons,
            "request_stop_present": (args.run / "REQUEST_STOP").exists(),
            "updated_epoch": now,
        })
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
