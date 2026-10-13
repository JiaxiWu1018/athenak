#!/usr/bin/env python3
"""Finalize one segment into an atomic, compact result and output manifest."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pathlib
import tempfile
import time
from typing import Iterable, Optional, Tuple


def atomic_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def digest(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def last_data(path: pathlib.Path) -> Optional[list[str]]:
    if not path.exists():
        return None
    last = None
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.strip() and not line.lstrip().startswith("#"):
                last = line.split()
    return last


def tracker_progress(run: pathlib.Path) -> Tuple[int, float, bool]:
    rows = []
    for idx in (0, 1):
        matches = sorted(run.rglob(f"*.co_{idx}.txt"))
        if not matches:
            matches = sorted(run.rglob(f"*.co_{idx}*.txt"))
        row = last_data(matches[-1]) if matches else None
        if row is None or len(row) < 3:
            return -1, float("nan"), False
        rows.append((int(row[0]), float(row[1])))
    same = rows[0][0] == rows[1][0] and abs(rows[0][1] - rows[1][1]) <= 1.0e-8
    return rows[0][0], rows[0][1], same


def total_bytes(root: pathlib.Path) -> int:
    total = 0
    for base, _, files in os.walk(root):
        for name in files:
            try:
                total += (pathlib.Path(base) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=pathlib.Path)
    p.add_argument("--campaign-root", required=True, type=pathlib.Path)
    p.add_argument("--segment", required=True, type=int)
    p.add_argument("--exit-code", required=True, type=int)
    p.add_argument("--nodes", required=True, type=int)
    p.add_argument("--started-epoch", required=True, type=float)
    p.add_argument("--log", required=True, type=pathlib.Path)
    p.add_argument("--telemetry", required=True, type=pathlib.Path)
    p.add_argument("--minimum-checkpoint-bytes", required=True, type=int)
    args = p.parse_args()

    restarts = sorted(
        (x for x in args.run.rglob("*.rst") if x.is_file() and x.stat().st_size > 0),
        key=lambda x: (x.stat().st_mtime_ns, x.name, str(x)),
    )
    checkpoint = restarts[-1] if restarts else None
    checkpoint_hash = digest(checkpoint) if checkpoint else None
    checkpoint_bytes = checkpoint.stat().st_size if checkpoint else 0
    telemetry = {}
    if args.telemetry.exists():
        try:
            telemetry = json.loads(args.telemetry.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            telemetry = {"stop_reasons": ["malformed_telemetry"]}
    cycle, sim_time, trackers_match = tracker_progress(args.run)
    fatal = False
    if args.log.exists():
        with args.log.open("rb") as stream:
            stream.seek(max(0, args.log.stat().st_size - 4 * 1024 * 1024))
            tail = stream.read().decode("utf-8", errors="replace")
        fatal = "### FATAL ERROR" in tail or "PARTICLE LEDGER FAIL" in tail

    manifest = args.run / "segment_manifest.tsv"
    with manifest.open("w", encoding="utf-8") as stream:
        stream.write("relative_path\tbytes\n")
        for base, _, files in os.walk(args.run):
            for name in sorted(files):
                path = pathlib.Path(base) / name
                if path == manifest:
                    continue
                try:
                    stream.write(f"{path.relative_to(args.run)}\t{path.stat().st_size}\n")
                except FileNotFoundError:
                    pass
    manifest_hash = digest(manifest)
    elapsed = max(0.0, time.time() - args.started_epoch)
    payload = {
        "schema_version": 1,
        "segment": args.segment,
        "exit_code": args.exit_code,
        "fatal_pattern": fatal,
        "cycle": cycle,
        "time": sim_time,
        "trackers_same_sample": trackers_match,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_complete": bool(
            checkpoint and checkpoint_hash and checkpoint_bytes >= args.minimum_checkpoint_bytes
            and trackers_match and not fatal and args.exit_code == 0
        ),
        "stop_reasons": telemetry.get("stop_reasons", []),
        "authorized_common_horizon_stop": bool(
            telemetry.get("authorized_common_horizon_stop", False)
        ),
        "common_horizon_stop_gate": telemetry.get("common_horizon_stop_gate"),
        "manifest": str(manifest),
        "manifest_sha256": manifest_hash,
        "storage_bytes": total_bytes(args.campaign_root),
        "elapsed_seconds": elapsed,
        "node_hours": args.nodes * elapsed / 3600.0,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(args.run / "result.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["checkpoint_complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
