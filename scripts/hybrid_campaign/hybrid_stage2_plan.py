#!/usr/bin/env python3
"""Stage-2 validation plan for the stratified_antithetic hybrid sampler.

One fixed-background pusher smoke plus three matched 1P live runs at R/M=6.1
on the SAME reduced grid, particle count, physics, boundaries, damping, and
diagnostics as the 2026-08-01 sampler-causality campaign, so the new sampler
compares directly against that campaign's preserved monte_carlo_antithetic,
stratified_random, and shell_fibonacci results without re-running them.
"""
import json
from pathlib import Path

PERIOD_Q6P1 = 94.661770035504048

plan = [{
    "name": "fixed_stratified_antithetic_s1985",
    "model": "q6p1", "sampler": "stratified_antithetic", "seed": 1985,
    "kind": "fixed", "tlim": 20.0, "pvtk_dt": 2.0,
    "nradial": 32, "nangular": 240,
    "stage": "fixed_smoke",
}]
for seed in (1985, 424242, 20260801):
    plan.append({
        "name": f"q6p1_stratified_antithetic_s{seed}",
        "model": "q6p1", "sampler": "stratified_antithetic", "seed": seed,
        "kind": "live", "nperiod": 1.0,
        "pvtk_dt": 5.0,
        "cbin_dt": round(PERIOD_Q6P1 / 4.0, 9),
        "rst_dt": PERIOD_Q6P1,
        "stage": "live_q6p1_stage2",
    })

Path("stage2_plan.json").write_text(json.dumps(plan, indent=2))
print(f"{len(plan)} cases written to stage2_plan.json")
