#!/usr/bin/env python3
"""Exercise the exact pinned C++ parser only (-n), inside an allocated GG job."""
import os
import subprocess

import s10


def check():
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") != "gg":
        raise RuntimeError("allocated GG required")
    root = s10.HANDOFF / "runtime_s10/build_parser_check"
    evidence = root / "evidence"
    exe = root / "src/athena"
    failed = s10.HANDOFF / "runtime_s10/runs/20261013T200009Z/production.athinput"
    production = s10.HERE / "production.athinput"
    prior = s10.read(s10.HANDOFF / "runtime_s10/runs/20261013T200009Z/DEBUG_VERIFIED.json")
    checkpoint = prior["clean_stop"]["checkpoint"]["path"]
    cases = [("original_missing_nlim", ["-i", str(failed)], s10.production_overrides(nlim=2, walk_every=1, tracker_every=1), False, None)]
    cases.extend((f"fixed_nlim_{n}", ["-i", str(production)], s10.production_overrides(nlim=n, walk_every=1 if n > 0 else 16, tracker_every=1 if n > 0 else 20), True, n) for n in (-1, 1, 2))
    cases.append(("restart_unlimited", ["-r", checkpoint], s10.production_overrides(), True, -1))
    rows = []
    for label, mode, overrides, expect_success, nlim in cases:
        command = ["ibrun", "-n", "1", str(exe), *mode, "-n", *overrides]
        p = subprocess.run(command, text=True, capture_output=True, timeout=120, check=False)
        (evidence / f"{label}.stdout").write_text(p.stdout)
        (evidence / f"{label}.stderr").write_text(p.stderr)
        passed = p.returncode == 0 if expect_success else p.returncode != 0 and "Parameter 'nlim' in block 'time' on command line not found" in p.stdout
        if expect_success:
            dumped = s10.parse_deck(p.stdout)
            passed = passed and int(dumped.get("time", {}).get("nlim", 999)) == nlim
        rows.append(dict(label=label, command=command, exit_code=p.returncode, passed=passed,
                         stdout_sha256=s10.digest(evidence / f"{label}.stdout"), stderr_sha256=s10.digest(evidence / f"{label}.stderr")))
        print(label, "PASS" if passed else "FAIL", "exit", p.returncode, flush=True)
    receipt = dict(passed=all(r["passed"] for r in rows), cases=rows, simulations_executed=0, job_id=os.environ["SLURM_JOB_ID"],
            production_input_sha256=s10.digest(production), native_cpu_executable_sha256=s10.digest(exe),
            native_gpu_executable_sha256=s10.digest(s10.EXE), source_commit=s10.read(s10.HANDOFF / "PROVENANCE.json")["source_commit"],
            parser_source_sha256=s10.digest(s10.SOURCE / "src/parameter_input.cpp"))
    s10.atomic(evidence / "PARSER_VERIFIED.json", receipt)
    if not receipt["passed"]:
        raise RuntimeError("native parser check failed")


if __name__ == "__main__":
    check()
