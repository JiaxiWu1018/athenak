#!/usr/bin/env python3
"""Session010 first-production analysis. Allocated compute only; never submits jobs."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parents[1]
RUN = BASE / "runtime_s10/runs/20261014_gh_retry_01"
OUT = BASE / "runtime_s10/analysis/first_production_20261015"
SOURCE = BASE / "runtime_s10/build/source/athenak"
os.environ.setdefault("MPLCONFIGDIR", str(BASE / "runtime_s10/analysis/matplotlib_cache"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(BASE / "analysis_support"))
from runtime import atomic


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def remaining():
    return min(float(os.environ.get("SLURM_JOB_END_TIME", 0)), START + 7200) - time.time()


def check_time(reserve=120):
    if remaining() < reserve:
        raise TimeoutError("Analysis allocation reserve reached; resume in another GG allocation")


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "INPUTS.json"
    if target.exists():
        return json.loads(target.read_text())
    state = json.loads((RUN / "segment_001.json").read_text())
    if state["status"] != "verified_segment_end":
        raise RuntimeError("First production segment is not closed and verified")
    manifest = dict(schema=1, run=str(RUN), cutoff=state["checkpoint"]["time"],
                    segment_receipt_sha256=digest(RUN / "segment_001.json"),
                    production_input_sha256=digest(RUN / "production.athinput"),
                    executable_sha256=json.loads((RUN / "FULL_SIZE_VERIFIED.json").read_text())["executable_sha256"],
                    source_commit="386870f88f60e25e1a8fac76fe04a071b87a01cf",
                    ordered_outputs=[c["output"] for c in state["chunks"]],
                    cases=[str(RUN / c["label"]) for c in state["chunks"]],
                    checkpoint=state["checkpoint"], created_epoch=time.time(),
                    note="Only closed first-production branch. Validation and later segments excluded.")
    atomic(target, manifest)
    return manifest


def seal(manifest):
    previous = json.loads((OUT / "SEALED.json").read_text()) if (OUT / "SEALED.json").exists() else None
    if previous:
        for receipt in previous["receipts"]:
            if digest(receipt["path"]) != receipt["sha256"]:
                raise RuntimeError("An existing checksum receipt changed; do not silently reseal")
            for row in receipt.get("stats", []):
                st = Path(row["path"]).stat()
                if (st.st_size, st.st_mtime_ns) != (row["bytes"], row["mtime_ns"]):
                    raise RuntimeError("A sealed input changed; preserve and review it: " + row["path"])
    def one(case):
        check_time()
        case = Path(case)
        receipt = case / "out/SCIENCE_VERIFIED.json"
        if not receipt.exists():
            print("SEAL start", case.name, flush=True)
            subprocess.run([sys.executable, str(BASE / "scripts/seal_run.py"),
                            "--run", str(case / "out"), "--result", str(case / "SEAL_RESULT.json")],
                           check=True, timeout=max(1, remaining() - 90))
        d = json.loads(receipt.read_text())
        # Check the relative inventory still names exactly these closed files.
        expected = {r["path"] for r in d["files"]}
        actual = {str(p.relative_to(case / "out")) for p in (case / "out").rglob("*")
                  if p.is_file() and p.name not in ("SCIENCE_VERIFIED.json", "ARCHIVE_VERIFIED.json")}
        if expected != actual:
            raise RuntimeError("Sealed inventory membership changed: " + str(case))
        if any((case / "out" / r["path"]).stat().st_size != r["bytes"] for r in d["files"]):
            raise RuntimeError("Sealed file size changed: " + str(case))
        print("SEAL done", case.name, len(d["files"]), flush=True)
        stats = [dict(path=str(case / "out" / r["path"]), bytes=r["bytes"],
                      mtime_ns=(case / "out" / r["path"]).stat().st_mtime_ns) for r in d["files"]]
        return dict(path=str(receipt), sha256=digest(receipt), bytes=sum(r["bytes"] for r in d["files"]), stats=stats)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(one, manifest["cases"]))
    atomic(OUT / "SEALED.json", dict(passed=True, receipts=rows, total_bytes=sum(r["bytes"] for r in rows)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["seal", "reduce", "fields", "render", "verify", "all"], default="all")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") != "gg":
        raise RuntimeError("Run this analysis inside an allocated Vista GG job; no submission is performed")
    check_time()
    manifest = freeze()
    actual_commit = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    actual_status = subprocess.check_output(["git", "-C", str(SOURCE), "status", "--porcelain"], text=True)
    if actual_commit != manifest["source_commit"] or actual_status:
        raise RuntimeError("Pinned native source changed; analysis reader provenance requires review")
    if digest(RUN / "production.athinput") != manifest["production_input_sha256"]:
        raise RuntimeError("Frozen production input changed")
    if digest(RUN / "segment_001.json") != manifest["segment_receipt_sha256"]:
        raise RuntimeError("Frozen segment receipt changed")
    executable = BASE / "runtime_s10/build/build_cuda_gi/src/athena"
    if digest(executable) != manifest["executable_sha256"]:
        raise RuntimeError("Native executable changed")
    atomic(OUT / "ATTEMPT.json", dict(job=os.environ["SLURM_JOB_ID"], host=socket.gethostname(),
           started_epoch=START, stage=args.stage, python=sys.version, command=sys.argv,
           script_sha256=digest(__file__), source_status=actual_status))
    stages = ["seal", "reduce", "fields", "render", "verify"] if args.stage == "all" else [args.stage]
    try:
        for stage in stages:
            check_time()
            if stage == "seal":
                seal(manifest)
            else:
                if not (OUT / "SEALED.json").exists():
                    raise RuntimeError("Run checksum-sealing before analysis")
                sealed = json.loads((OUT / "SEALED.json").read_text())
                for receipt in sealed["receipts"]:
                    if digest(receipt["path"]) != receipt["sha256"]:
                        raise RuntimeError("Input checksum receipt changed")
                    for row in receipt.get("stats", []):
                        st = Path(row["path"]).stat()
                        if (st.st_size, st.st_mtime_ns) != (row["bytes"], row["mtime_ns"]):
                            raise RuntimeError("Sealed input changed: " + row["path"])
                from science import reduce, fields, render, verify
                {"reduce": reduce, "fields": fields, "render": render, "verify": verify}[stage](manifest)
            atomic(OUT / "PROGRESS.json", dict(last_completed_stage=stage, updated_epoch=time.time()))
    except BaseException as exc:
        failure = dict(stage=stage, error=repr(exc), epoch=time.time(), resumable=True)
        atomic(OUT / "LAST_FAILURE.json", failure)
        atomic(OUT / "failures" / f"{time.time_ns()}_{stage}.json", failure)
        raise


START = time.time()
if __name__ == "__main__":
    # Imported helpers share this same timer and immutable paths, not a second cap.
    sys.modules["run"] = sys.modules["__main__"]
    main()
