#!/usr/bin/env python3
"""Create a machine-readable acceptance receipt for the exact Vista quick tests."""
from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path


def total_bytes(root: Path) -> int:
    total = 0
    for base, _, files in os.walk(root):
        for name in files:
            try:
                total += (Path(base)/name).stat().st_size
            except FileNotFoundError:
                pass
    return total


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    exact = args.root/"uninterrupted"
    production = args.root/"production_output"
    logs = sorted(args.root.glob("*.log"))
    fatal = []
    for log in logs:
        text = log.read_text(errors="replace")
        if "### FATAL ERROR" in text or "PARTICLE LEDGER FAIL" in text:
            fatal.append(str(log))
    required = {
        "history": list(exact.rglob("*.hst")),
        "tmunu_slice": list(exact.rglob("*.tmunu.*.bin")),
        "constraint_slice": list(exact.rglob("*.con.*.bin")),
        "constraint_3d": list(exact.rglob("*.con3d.*.bin")),
        "z4c_slice": list(exact.rglob("*.z4c.*.bin")),
        "particles": list(exact.rglob("*.part.vtk")),
        "energy_3d": list(exact.rglob("*.E3d.*.bin")),
        "restart": list(exact.rglob("*.rst")),
        "coarsened_chi": list(exact.rglob("*.cbin")),
        "tracker_left": list(exact.rglob("*.co_0.txt")),
        "tracker_right": list(exact.rglob("*.co_1.txt")),
        "individual_consumer_left": list(exact.rglob("*.horizon_consumer_0.csv")),
        "individual_consumer_right": list(exact.rglob("*.horizon_consumer_1.csv")),
        "common_consumer": list(exact.rglob("*.horizon_consumer_2.csv")),
        "profile": list(exact.rglob("gi_profile_M076_two_clump_s7.txt")),
    }
    missing = [key for key, paths in required.items() if not paths]
    production_required = {
        "xy_slices": list(production.rglob("*.tmunu_xy.*.bin"))
            + list(production.rglob("*.con_xy.*.bin"))
            + list(production.rglob("*.z4c_xy.*.bin")),
        "xz_slices": list(production.rglob("*.tmunu_xz.*.bin"))
            + list(production.rglob("*.con_xz.*.bin"))
            + list(production.rglob("*.z4c_xz.*.bin")),
        "yz_slices": list(production.rglob("*.tmunu_yz.*.bin"))
            + list(production.rglob("*.con_yz.*.bin"))
            + list(production.rglob("*.z4c_yz.*.bin")),
        "weyl_slices": list(production.rglob("*.weyl_xy.*.bin"))
            + list(production.rglob("*.weyl_xz.*.bin"))
            + list(production.rglob("*.weyl_yz.*.bin")),
        "full_3d": list(production.rglob("*.con3d.*.bin"))
            + list(production.rglob("*.z4c3d.*.bin"))
            + list(production.rglob("*.tmunu3d.*.bin"))
            + list(production.rglob("*.weyl3d.*.bin")),
        "waveform_real": list(production.rglob("waveforms/rpsi4_real_*.txt")),
        "waveform_imag": list(production.rglob("waveforms/rpsi4_imag_*.txt")),
    }
    missing.extend(f"production_{key}" for key, paths in production_required.items()
                   if not paths)
    init = json.loads((args.root/"initial_validation.json").read_text())
    restart = json.loads((args.root/"restart_validation.json").read_text())
    production_restart_checks = {"configured": False}
    if cfg.get("initial_checkpoint"):
        production_restart = args.root/"production_restart"
        restart_logs = (args.root/"production_restart.log",
                        args.root/"production_restart_parse.log")
        terminal = []
        if restart_logs[0].is_file():
            terminal = re.findall(
                r"^time=([0-9.eE+-]+)\s+cycle=([0-9]+)\s*$",
                restart_logs[0].read_text(errors="replace"), re.MULTILINE)
        output_restarts = sorted(production_restart.rglob("*.rst"))
        output_restart = output_restarts[-1] if output_restarts else None
        marker_offset = -1
        if output_restart:
            with output_restart.open("rb") as stream:
                marker_offset = stream.read(1024*1024 + 1).find(b"<par_end>")
        last_time = float(terminal[-1][0]) if terminal else float("nan")
        last_cycle = int(terminal[-1][1]) if terminal else -1
        production_restart_checks = {
            "configured": bool(cfg.get("production_restart_smoke")),
            "advanced_cycle": last_cycle > int(cfg.get("initial_cycle", -1)),
            "advanced_time": last_time > float(cfg.get("initial_time", float("inf"))),
            "checkpoint_written": bool(
                output_restart and output_restart.stat().st_size
                >= int(cfg["minimum_checkpoint_bytes"])
            ),
            "checkpoint_header_within_1_mib": 0 <= marker_offset < 1024*1024,
            "checkpoint_parameter_end_offset": marker_offset,
            "last_cycle": last_cycle,
            "last_time": last_time,
            "parse_smoke_log_clean": all(
                path.is_file() and "### FATAL ERROR" not in path.read_text(errors="replace")
                for path in restart_logs
            ),
        }
    storage = total_bytes(args.root)
    storage_cap = int(cfg["quicktest_storage_stop_bytes"])
    memory_rows = []
    sampled_files = {}
    for path in sorted(args.root.glob("gpu_memory_*.csv")):
        valid_rows = 0
        for line in path.read_text(errors="replace").splitlines():
            fields = [x.strip() for x in line.split(",")]
            if len(fields) >= 4:
                try:
                    memory_rows.append((float(fields[-2]), float(fields[-1]), str(path)))
                    valid_rows += 1
                except ValueError:
                    pass
        if valid_rows:
            sampled_files[path.name] = valid_rows
    expected = {"parse": 1, "mesh": 1, "uninterrupted": int(cfg["startup_nodes"]),
                "production_output": int(cfg["startup_nodes"]),
                "split_a": int(cfg["startup_nodes"]),
                "restart_b": int(cfg["restart_test_ranks"]),
                "clean_stop": int(cfg["restart_test_ranks"]),
                "unboosted": int(cfg["startup_nodes"])}
    if cfg.get("initial_checkpoint"):
        expected["production_restart"] = int(cfg["startup_nodes"])
        expected["production_restart_parse"] = 1
    sampled_counts = {
        label: sum(name.startswith(f"gpu_memory_{label}_") for name in sampled_files)
        for label in expected
    }
    missing_memory_nodes = {
        label: expected[label] - sampled_counts[label]
        for label in expected if sampled_counts[label] < expected[label]
    }
    peak_fraction = max((used/total for used, total, _ in memory_rows if total > 0),
                        default=None)
    memory_pass = (peak_fraction is not None and not missing_memory_nodes
                   and peak_fraction <= float(cfg["quicktest_gpu_memory_fraction_limit"]))
    assertions_pass = all(init.get("assertions", {}).values())
    production_restart_pass = (
        not cfg.get("initial_checkpoint")
        or all(value for key, value in production_restart_checks.items()
               if key not in {"checkpoint_parameter_end_offset", "last_cycle", "last_time"})
    )
    passed = (not fatal and not missing and storage <= storage_cap and memory_pass
              and assertions_pass and production_restart_pass)
    payload = {
        "schema_version": 1, "status": "pass" if passed else "fail",
        "fatal_logs": fatal, "missing_output_classes": missing,
        "outputs": {key: [str(x) for x in paths] for key, paths in required.items()},
        "production_outputs": {
            key: [str(x) for x in paths] for key, paths in production_required.items()
        },
        "storage_bytes": storage, "storage_cap_bytes": storage_cap,
        "sampled_all_node_gpu_peak_fraction": peak_fraction,
        "sampled_gpu_files": sampled_files,
        "sampled_node_counts": sampled_counts,
        "missing_memory_nodes": missing_memory_nodes,
        "gpu_fraction_limit": cfg["quicktest_gpu_memory_fraction_limit"],
        "memory_scope": "one nvidia-smi sampler per participating Vista node via an overlapping Slurm step",
        "initial_assertions": init.get("assertions", {}),
        "restart_particle_max_abs": restart.get("particles", {}).get("max_abs"),
        "production_restart_checks": production_restart_checks,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 3

if __name__ == "__main__":
    raise SystemExit(main())
