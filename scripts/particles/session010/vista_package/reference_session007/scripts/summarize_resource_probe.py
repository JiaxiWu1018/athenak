#!/usr/bin/env python3
"""Select a tested-safe Vista MeshBlock capacity and conservative node count."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def memory_rows(path: Path) -> list[tuple[float, float]]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) >= 4:
            try:
                rows.append((float(fields[-2]), float(fields[-1])))
            except ValueError:
                pass
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--ranks", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cases = []
    with (args.root / "cases.tsv").open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            cap = int(row["capacity"])
            files = sorted(args.root.glob(f"gpu_memory_cap{cap}_*.csv"))
            samples = [sample for path in files for sample in memory_rows(path)]
            peak = max((used / total for used, total in samples if total > 0), default=None)
            passed = (int(row["exit_code"]) == 0 and len(files) == args.ranks
                      and peak is not None and peak <= 0.80)
            cases.append({"capacity": cap, "exit_code": int(row["exit_code"]),
                          "sampled_nodes": len(files), "peak_hbm_fraction": peak,
                          "selection_pass_at_0p80": passed})
    safe = [case["capacity"] for case in cases if case["selection_pass_at_0p80"]]
    tested_safe = max(safe) if safe else None
    initial_blocks = 1352
    forecast_blocks = 3 * initial_blocks
    choices = [8, 12, 16, 24, 32, 48, 64]
    required = (math.ceil(forecast_blocks / (0.75 * tested_safe))
                if tested_safe else None)
    recommended = (next((value for value in choices if value >= required), None)
                   if required is not None else None)
    payload = {
        "schema_version": 1,
        "h200_documented_hbm_gb": 96,
        "probe_ranks": args.ranks,
        "selection_memory_fraction": 0.80,
        "hard_quicktest_memory_fraction": 0.85,
        "cases": cases,
        "tested_safe_max_nmb_per_rank": tested_safe,
        "initial_leaf_meshblocks": initial_blocks,
        "planning_peak_meshblocks": forecast_blocks,
        "block_capacity_utilization_limit": 0.75,
        "minimum_ranks_from_capacity": required,
        "recommended_candidate_nodes": recommended,
        "qualification": (
            "candidate only; final nodes are max(capacity recommendation, measured "
            "time-to-solution recommendation), and the full exact quick test must pass"
        ),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if tested_safe is not None and recommended is not None else 3


if __name__ == "__main__":
    raise SystemExit(main())
