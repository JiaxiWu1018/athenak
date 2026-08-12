#!/usr/bin/env python3
"""Collect the run table: source commit, pgen hash, input hash, executable hash,
seed, sampler, particle count, wall time, and completion state for every case.
"""
import argparse
import csv
import json
from pathlib import Path


def parse_kv(path):
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    manifest = json.loads((args.root / "inputs" / "input_manifest.json").read_text())
    rows = []
    for case in manifest:
        name = case["name"]
        recs = sorted((args.root / "provenance").glob(f"{name}_job*"))
        run = args.root / "runs" / name
        rec = {
            "case": name, "stage": case.get("stage", ""), "model": case["model"],
            "q": case["q"], "sampler": case["sampler"], "seed": case["seed"],
            "kind": case.get("kind", "live"), "npart": case["npart"],
            "nradial": case["nradial"], "nangular": case["nangular"],
            "tlim": case["tlim"], "period": case["period"],
            "input_sha256": case["input_sha256"],
            "completed": (run / "COMPLETED").exists(),
            "failed": (run / "FAILED").exists(),
            "slurm_jobs": ";".join(p.name.split("_job")[-1] for p in recs),
        }
        if recs:
            p = parse_kv(recs[-1] / "provenance.txt")
            rec.update({
                "source_commit": p.get("source_commit", ""),
                "source_uncommitted_files": p.get("source_uncommitted_files", ""),
                "pgen_sha256": p.get("pgen_sha256", ""),
                "exe_sha256": p.get("exe_sha256", ""),
                "input_sha256_run": p.get("input_sha256", ""),
                "partition": p.get("partition", ""),
                "host": p.get("host", ""),
                "elapsed_s": p.get("elapsed_s", ""),
                "exit_status": p.get("exit_status", ""),
                "date_utc": p.get("date_utc", ""),
            })
            rec["input_hash_matches"] = (rec.get("input_sha256_run", "")
                                         == case["input_sha256"])
        rows.append(rec)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    order = ["case", "stage", "model", "q", "sampler", "seed", "kind",
             "npart", "tlim", "completed", "failed", "elapsed_s"]
    fields = order + [f for f in fields if f not in order]
    with args.output.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    done = sum(1 for r in rows if r["completed"])
    fail = sum(1 for r in rows if r["failed"])
    commits = {r.get("source_commit", "") for r in rows if r.get("source_commit")}
    pgens = {r.get("pgen_sha256", "") for r in rows if r.get("pgen_sha256")}
    exes = {r.get("exe_sha256", "") for r in rows if r.get("exe_sha256")}
    print(f"{len(rows)} cases: {done} completed, {fail} failed")
    print(f"distinct source commits : {commits}")
    print(f"distinct pgen hashes    : {pgens}")
    print(f"distinct exe hashes     : {exes}")
    bad = [r["case"] for r in rows
           if r.get("input_hash_matches") is False]
    print(f"input-hash mismatches   : {bad if bad else 'none'}")


if __name__ == "__main__":
    main()
