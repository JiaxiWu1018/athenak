#!/usr/bin/env python3
"""Session010 finite, user-submitted simulation chain; no batch-side submissions."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
HANDOFF = HERE.parent
WORK = HANDOFF / "runtime_s10/build"
SOURCE = WORK / "source/athenak"
EXE = WORK / "build_cuda_gi/src/athena"
sys.path.insert(0, str(HANDOFF / "scripts"))
from checkpoint_io import checkpoint, digest, atomic


def read(path):
    return json.loads(Path(path).read_text())


def config():
    return read(HERE / "config.json")


def parse_deck(text):
    blocks = {}
    block = None
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("<") and line.endswith(">"):
            block = line[1:-1]
            blocks.setdefault(block, {})
        elif block and "=" in line:
            k, v = line.split("=", 1)
            blocks[block][k.strip()] = v.strip()
    return blocks


def production_deck(original):
    """Only reviewed time/output changes. Preserve physics byte-for-byte."""
    block = None
    lines = []
    for line in original.splitlines():
        match = re.fullmatch(r"<([^>]+)>", line.strip())
        if match:
            block = match[1]
        key = line.split("=", 1)[0].strip() if "=" in line else None
        value = None
        if block == "time" and key == "nlim":
            value = "-1"
        elif block == "time" and key == "tlim":
            value = "1e30"
        elif block == "fastflow" and key in {f"stop_time_{i}" for i in range(3)}:
            value = "1e30"
        elif block and re.fullmatch(r"output\d+", block) and key == "dt":
            i = int(block[6:])
            value = ("0.025" if i in (1, 8) else "2.0" if i == 5 else
                     "1e30" if i == 7 else "25.0" if i in (6, 18, 19, 20, 21)
                     else "0.5")
        lines.append(f"{key} = {value}" if value is not None else line)
    result = "\n".join(lines) + "\n"
    before, after = parse_deck(original), parse_deck(result)
    for name, values in before.items():
        if name == "time" or name.startswith("output"):
            continue
        expected = dict(values)
        if name == "fastflow":
            for i in range(3):
                expected[f"stop_time_{i}"] = "1e30"
        if after[name] != expected:
            raise ValueError(f"unreviewed physics change: {name}")
    return result


def validate_overrides(text, overrides):
    """The native CLI can replace existing parameters, but cannot add them."""
    blocks = parse_deck(text)
    for argument in overrides:
        if "=" not in argument or "/" not in argument.split("=", 1)[0]:
            continue
        block, name = argument.split("=", 1)[0].split("/", 1)
        if block not in blocks or name not in blocks[block]:
            raise ValueError(f"native command-line override requires existing input key: {block}/{name}")


def production_overrides(stop_path="REQUEST_STOP", nlim=-1, tracker_every=20, walk_every=16):
    return [f"time/stop_file={stop_path}", "time/tlim=1e30", f"time/nlim={nlim}",
            "mesh_refinement/max_nmb_per_rank=400", f"z4c/tracker_walk_every={walk_every}",
            f"z4c/co_0_out_every={tracker_every}", f"z4c/co_1_out_every={tracker_every}"]


def validate_config(cfg):
    if cfg["email"] != "jiaxiwu@caltech.edu" or not cfg["session010_review_complete"]:
        raise ValueError("Session010 review/email mismatch")
    if cfg["production_nodes"] != 16 or cfg["production_wall_hours"] != [8, 24, 24]:
        raise ValueError("change the reviewed chain only after resource review")
    if cfg["max_nmb_per_rank"] != 400 or cfg["automatic_retries"] != 0:
        raise ValueError("unreviewed capacity or retry policy")
    if not 0 < cfg["memory_warning_fraction"] < cfg["memory_stop_fraction"] < 1:
        raise ValueError("invalid memory guards")
    if not 0 < cfg["capacity_stop_fraction"] < 1:
        raise ValueError("invalid mesh capacity guard")
    if cfg["analysis_jobs"] or cfg["archive_jobs"]:
        raise ValueError("this launcher submits simulation jobs only")


def build_ready(full=True):
    receipt = WORK / "BUILD_VERIFIED.json"
    if not receipt.exists():
        return False
    r = read(receipt)
    pins = read(HANDOFF / "PROVENANCE.json")
    if any(r.get(k) != pins[k] for k in ("source_commit", "kokkos_commit")):
        raise ValueError("build pins differ from handoff")
    if r["architecture"] != "aarch64" or Path(r["executable"]) != EXE:
        raise ValueError("wrong native executable")
    if not EXE.is_file():
        raise ValueError("native executable missing")
    if full:
        for row in r["files"]:
            if digest(row["path"]) != row["sha256"]:
                raise ValueError(f"build evidence changed: {row['path']}")
        for repo, pin in ((SOURCE, pins["source_commit"]),
                          (SOURCE / "kokkos", pins["kokkos_commit"])):
            if subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != pin:
                raise ValueError("source checkout pin changed")
            if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
                raise ValueError("compiled source has tracked changes")
    return True


def verify_package():
    subprocess.run(["sha256sum", "--quiet", "-c", "PACKAGE_SHA256SUMS.txt"],
                   cwd=HANDOFF, check=True)
    if not build_ready():
        raise RuntimeError("native build gate absent")


def submission_specs(cfg, native_ready):
    specs = []
    if not native_ready:
        specs.append(("build", "gg", 1, cfg["build_wall_hours"], 0))
    specs.append(("debug", "gh-dev", cfg["debug_nodes"], cfg["debug_wall_hours"], 0))
    specs.extend(("production", "gh", cfg["production_nodes"], hours, i)
                 for i, hours in enumerate(cfg["production_wall_hours"], 1))
    return specs


def gh_submission_specs(cfg):
    return [("production", "gh", cfg["production_nodes"], hours, i)
            for i, hours in enumerate(cfg["production_wall_hours"], 1)]


def sbatch_args(spec, run, dependency=None):
    cfg = read(run / "config.json") if (run / "config.json").exists() else config()
    stage, partition, nodes, hours, segment = spec
    name = f"s10-{run.name}-{stage}-{segment}"
    args = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes", "--no-requeue",
            "--account=" + cfg["account"], "--partition=" + partition,
            f"--nodes={nodes}", f"--ntasks={nodes}", "--ntasks-per-node=1",
            "--cpus-per-task=" + ("16" if stage == "build" else "1"),
            f"--time={hours:02d}:00:00", "--job-name=" + name,
            "--mail-user=" + cfg["email"], "--mail-type=BEGIN,END,FAIL",
            "--chdir=" + str(HANDOFF), "--output=" + str(run / "logs" / f"{stage}_{segment}.%j.log")]
    if dependency:
        args.append("--dependency=afterok:" + str(dependency))
    return args + [str(HERE / "job.sbatch"), stage, str(run), str(segment), str(HERE), sys.executable]


def submission_id(stdout):
    # TACC's site plugin prints its banner/checks on stdout before --parsable's ID.
    ids = {m[1] for line in stdout.splitlines()
           if (m := re.fullmatch(r"(\d+)(?:;[^\s]+)?", line.strip()))}
    return next(iter(ids)) if len(ids) == 1 else None


def confirm_job(job_id, name):
    p = subprocess.run(["sacct", "-n", "-X", "-P", "-j", job_id,
                        "--format=JobIDRaw,JobName%128,State,User,Account"],
                       text=True, capture_output=True, timeout=20, check=True)
    import pwd
    user = pwd.getpwuid(os.getuid()).pw_name
    rows = [line.split("|") for line in p.stdout.splitlines()]
    matches = [r for r in rows if len(r) >= 5 and r[0] == job_id and r[1].strip() == name
               and r[3].strip() == user and r[4].strip().lower() == config()["account"].lower()]
    if len(matches) != 1:
        raise RuntimeError("cannot independently confirm accepted job identity; no resubmission")
    if matches[0][2].split("+")[0] not in {"PENDING", "RUNNING", "COMPLETING", "COMPLETED", "CONFIGURING"}:
        raise RuntimeError("accepted predecessor has failed/stopped; inspect its logs before continuing")
    return dict(job_id=job_id, row=matches[0], confirmed_epoch=time.time())


def submit_chain(run, specs, runner=subprocess.run, resume=False):
    """One-shot ledger: uncertain sbatch responses forbid blind resubmission."""
    ledger = run / "submission.json"
    if ledger.exists() and not resume:
        raise RuntimeError("submission ledger already exists; inspect it, do not resubmit")
    state = read(ledger) if resume else {"status": "submitting", "jobs": [], "pending": None, "created_epoch": time.time()}
    if state["status"] == "finite_chain_submitted":
        raise RuntimeError("finite chain is already submitted; no duplicate jobs")
    if resume and state.get("pending"):
        attempt = state.get("last_attempt", {})
        job_id = submission_id(attempt.get("stdout", ""))
        if attempt.get("returncode") != 0 or not job_id:
            raise RuntimeError("uncertain pending submission has no unambiguous success ID; no automatic retry")
        spec = specs[len(state["jobs"])]
        name = f"s10-{run.name}-{spec[0]}-{spec[4]}"
        confirmed = confirm_job(job_id, name)
        state["jobs"].append(dict(stage=spec[0], segment=spec[4], job_id=job_id,
                                  nodes=spec[2], wall_hours=spec[3], recovered=confirmed, **attempt))
        state["pending"] = None
        state["status"] = "partial_chain_recovered"
        atomic(ledger, state)
    if resume:
        for i, recorded in enumerate(state["jobs"]):
            if i >= len(specs) or (recorded["stage"], recorded["segment"]) != (specs[i][0], specs[i][4]):
                raise RuntimeError("recorded chain differs from reviewed job sequence")
            confirm_job(recorded["job_id"], f"s10-{run.name}-{recorded['stage']}-{recorded['segment']}")
        # Only submission parsing changed. Preserve the original frozen manifest;
        # refresh launcher hashes before releasing any production job.
        if any(j["stage"] == "production" for j in state["jobs"]):
            raise RuntimeError("production already submitted; this recovery is limited to the preparation stage")
        hashes = read(run / "LAUNCHER_SHA256.json")
        archive = run / "LAUNCHER_SHA256.before_recovery.json"
        if not archive.exists():
            atomic(archive, hashes)
        atomic(run / "LAUNCHER_SHA256.json", {name: digest(HERE / name) for name in hashes})
    atomic(ledger, state)
    predecessor = state["jobs"][-1]["job_id"] if state["jobs"] else None
    for spec in specs[len(state["jobs"]):]:
        args = sbatch_args(spec, run, predecessor)
        state["pending"] = args
        atomic(ledger, state)
        proc = runner(args, text=True, capture_output=True, check=False)
        attempt = dict(command=args, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        state["last_attempt"] = attempt
        job_id = submission_id(proc.stdout)
        if proc.returncode or not job_id:
            state["status"] = "submission_incomplete_or_uncertain"
            atomic(ledger, state)
            raise RuntimeError("sbatch failed/ambiguous; inspect submission.json and squeue; no automatic retry")
        predecessor = job_id
        state["jobs"].append(dict(stage=spec[0], segment=spec[4], job_id=predecessor,
                                  nodes=spec[2], wall_hours=spec[3], **attempt))
        state["pending"] = None
        atomic(ledger, state)
    state["status"] = "finite_chain_submitted"
    atomic(ledger, state)
    return state


def accounting(run):
    ledger = run / "submission.json"
    ids = [j["job_id"] for j in read(ledger)["jobs"]] if ledger.exists() else []
    if not ids:
        ids = [os.environ["SLURM_JOB_ID"]] if os.environ.get("SLURM_JOB_ID") else []
    rows = []
    if ids:
        p = subprocess.run(["sacct", "-n", "-X", "-P", "-j", ",".join(ids),
                            "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocNodes"],
                           text=True, capture_output=True, timeout=20, check=False)
        for line in p.stdout.splitlines():
            a = line.split("|")
            if len(a) >= 5 and a[0] in ids and a[3].isdigit() and a[4].isdigit():
                rows.append(dict(job_id=a[0], state=a[1], exit_code=a[2],
                                 elapsed_seconds=int(a[3]), nodes=int(a[4]),
                                 node_hours=int(a[3]) * int(a[4]) / 3600))
    return dict(rows=rows, actual_node_hours=sum(r["node_hours"] for r in rows),
                final=bool(rows) and set(ids) == {r["job_id"] for r in rows}
                and all(r["state"].split("+")[0] in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "PREEMPTED", "REVOKED", "DEADLINE"} for r in rows),
                note="Running rows are provisional; refresh status after Slurm finishes.")


def latest_checkpoint(out, rootlevel):
    good, failures = [], []
    for p in sorted(out.rglob("*.rst")):
        try:
            good.append(checkpoint(p, hash_file=False, expected_root=rootlevel))
        except (ValueError, OSError, struct.error) as exc:
            failures.append(dict(path=str(p), error=str(exc)))
    if not good:
        raise ValueError(f"no complete native checkpoint: {failures}")
    newest = max(good, key=lambda r: (r["cycle"], r["time"]))
    row = checkpoint(newest["path"], expected_root=rootlevel)
    row["rejected_candidates"] = failures
    return row


def verify_restart(row, rootlevel):
    current = checkpoint(row["path"], expected_root=rootlevel)
    if any(current[k] != row[k] for k in ("sha256", "bytes", "time", "cycle", "blocks", "particles", "removed")):
        raise ValueError("restart checkpoint no longer matches its receipt")
    return current


def job_end_epoch():
    value = os.environ.get("SLURM_JOB_END_TIME", "")
    if value.isdigit() and int(value) > time.time():
        return int(value)
    text = subprocess.check_output(["scontrol", "show", "job", "-o", os.environ["SLURM_JOB_ID"]], text=True, timeout=20)
    match = re.search(r"\bEndTime=(\S+)", text)
    if not match or match[1] in ("Unknown", "None"):
        raise RuntimeError("cannot determine actual Slurm end time")
    # scontrol timestamps use the node's local timezone, as does naive.timestamp().
    return int(dt.datetime.fromisoformat(match[1]).timestamp())


def promoted_state(old, cp, reason, chunk):
    if old.get("checkpoint"):
        prior = old["checkpoint"]
        if cp["cycle"] <= prior["cycle"] or cp["time"] <= prior["time"]:
            raise ValueError("native checkpoint did not advance")
    # Resource/user stops promote verified progress before marking terminal.
    result = dict(old, checkpoint=cp, status=reason, updated_epoch=time.time())
    result["chunks"] = old.get("chunks", []) + [chunk]
    return result


def native_particles(cp):
    import numpy as np
    offset = cp["bytes"] - cp["particles"] * 10 * 8
    return np.memmap(cp["path"], mode="r", dtype="<f8", offset=offset,
                     shape=(10, cp["particles"]))


def audit_native(cp, fields=False):
    import numpy as np
    p = native_particles(cp)
    if not np.isfinite(p).all():
        raise ValueError("nonfinite native particles")
    # Pinned ParticlesIndex: mass, energy, x,vx,y,vy,z,vz, then GID, tag.
    gids, tags = p[8], p[9]
    if not np.array_equal(tags, tags.astype(np.int64)) or np.any(tags < 0) or np.any(tags >= 5_000_000):
        raise ValueError("invalid immutable tags")
    if len(np.unique(tags)) != cp["particles"] or np.any(p[0] <= 0):
        raise ValueError("duplicate tags or nonpositive weights")
    if not np.array_equal(gids, gids.astype(np.int64)) or np.any(gids < 0) or np.any(gids >= cp["blocks"]):
        raise ValueError("native GIDs outside leaf mesh")
    if sum(cp["removed"]) == 0 and not np.array_equal(np.sort(tags), np.arange(5_000_000)):
        raise ValueError("initial native tag census incomplete")
    if fields:
        remaining = cp["blocks"] * 25 * 40**3
        grid_start = cp["bytes"] - cp["particles"] * 10 * 8 - 32 - remaining * 8
        with Path(cp["path"]).open("rb") as f:
            f.seek(grid_start)
            while remaining:
                n = min(remaining, 4_000_000)
                values = np.fromfile(f, dtype="<f8", count=n)
                if len(values) != n or not np.isfinite(values).all():
                    raise ValueError("nonfinite or truncated full native Z4c state")
                remaining -= n
    return dict(particles=cp["particles"], unique_tags=True, finite=True,
                positive_weights=True, gids_in_leaf_range=True, full_grid_finite_checked=fields)


def initial_gate(out, log, extent):
    import numpy as np
    from validate_initial_s7 import written_particle_stats, parse_ledger, ledger_agreement
    written, ledger = written_particle_stats(out, domain_half_extent=extent)
    logged = parse_ledger(log)
    agree, detail = ledger_agreement(logged, ledger)
    checks = [written[k] for k in ("finite", "integer_tags", "exact_unique_tag_range", "positive_rest_weights", "inside_domain")]
    components = written["components"]
    checks += [written["time"] == 0, [r["count"] for r in components] == [3_000_000, 1_000_000, 1_000_000], agree,
               logged[1]["P_hat"][1] < 0 < logged[2]["P_hat"][1],
               logged[1]["J_origin_cov"][2] > 0 and logged[2]["J_origin_cov"][2] > 0]
    for i, x in ((1, -3), (2, 3)):
        c = components[i]
        checks += [np.max(np.abs(np.array(c["position_mean"]) - [x, 0, 0])) < .03,
                   all(.6 < s < .8 for s in c["position_std"]),
                   all(.019 < s < .021 for s in c["rest_frame_thermal_std"])]
    if not all(checks):
        raise ValueError("full initialization census/model gate failed")
    return dict(written=written, ledger_comparison=detail, passed=True)


def binary_reader():
    sys.path.insert(0, str(SOURCE / "vis/python"))
    import bin_convert
    return bin_convert


def mesh_info(path):
    data = binary_reader().read_coarsened_binary(str(path))
    return dict(blocks=len(data["mb_logical"]), cycle=int(data["cycle"]), time=float(data["time"]))


def eligible_amr(log):
    text = Path(log).read_text(errors="replace")
    return any(int(a) + int(b) > 0 for a, b in re.findall(r"(\d+) MeshBlocks created,\s*(\d+) deleted by AMR", text))


def checkpoint_matches_exit(cp, log):
    rows = re.findall(r"\btime=([-+0-9.eE]+)\s+cycle=(\d+)", Path(log).read_text(errors="replace"))
    if not rows:
        raise ValueError("no native final progress line; complete checkpoint alone cannot certify this exit")
    physical_time, cycle = float(rows[-1][0]), int(rows[-1][1])
    if cp["cycle"] != cycle or not math.isclose(cp["time"], physical_time, rel_tol=5e-6, abs_tol=1e-10):
        raise ValueError("checkpoint does not match native exit progress")


def seal_receipt(result):
    """Compatibility with the handoff's allocated seal_run.py; no hashing/transfer here."""
    cp = result.get("checkpoint") or {}
    return dict(exit_code=result["exit_code"],
                checkpoint_complete=bool(cp and result["exit_code"] == 0 and not result.get("verification_error")),
                checkpoint=cp.get("path"), checkpoint_sha256=cp.get("sha256"),
                time=cp.get("time"), cycle=cp.get("cycle"), native_checkpoint=cp,
                tracker_coverage_separate=True, scientific_completion=False)


def output_gate(out):
    import numpy as np
    reader = binary_reader()
    checks = {}
    for plane in ("xy", "xz", "yz"):
        for kind in ("tmunu", "con", "z4c", "weyl"):
            files = sorted(out.rglob(f"*.{kind}_{plane}.*.bin"))
            if not files:
                raise ValueError(f"missing {kind}/{plane} output")
            data = reader.read_binary(str(files[-1]))
            if not all(np.isfinite(v).all() for v in data["mb_data"].values()):
                raise ValueError(f"nonfinite {kind}/{plane} output")
            checks[kind + "_" + plane] = dict(time=float(data["time"]), cycle=int(data["cycle"]))
    real = np.loadtxt(out / "waveforms/rpsi4_real_0040.txt", ndmin=2)
    imag = np.loadtxt(out / "waveforms/rpsi4_imag_0040.txt", ndmin=2)
    if real.shape != imag.shape or real.shape[1] != 78 or not np.array_equal(real[:, 0], imag[:, 0]) or not np.isfinite(real).all() or not np.isfinite(imag).all():
        raise ValueError("raw R40 complex multipole contract failed")
    return dict(planes=checks, waveform_samples=len(real), waveform_start=float(real[0, 0]), waveform_end=float(real[-1, 0]), strain_certified=False)


def stop_reason(samples, cfg):
    for row in samples:
        if row.get("error"):
            return "telemetry_failure"
        if row["host_used_bytes"] / row["host_total_bytes"] >= cfg["memory_stop_fraction"]:
            return "resource_limit_host_memory"
        if any(g["used_mib"] / g["total_mib"] >= cfg["memory_stop_fraction"] for g in row["gpus"]):
            return "resource_limit_gpu_memory"
    return None


def sample(telemetry, stop, seconds):
    telemetry.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname()
    history = telemetry / (host + ".jsonl")
    while not stop.exists():
        row = dict(host=host, epoch=time.time())
        try:
            mem = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                k, v = line.split(":", 1)
                mem[k] = int(v.split()[0]) * 1024
            row.update(host_total_bytes=mem["MemTotal"], host_used_bytes=mem["MemTotal"] - mem["MemAvailable"])
            p = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True, timeout=15)
            row["gpus"] = [dict(uuid=a[0].strip(), used_mib=float(a[1]), total_mib=float(a[2])) for a in (l.split(",") for l in p.stdout.splitlines())]
            if len(row["gpus"]) != 1 or any(not all(math.isfinite(g[k]) for k in ("used_mib", "total_mib")) or g["total_mib"] <= 0 or g["used_mib"] < 0 for g in row["gpus"]):
                raise ValueError("expected one measurable H200 per GH node")
        except Exception as exc:
            row["error"] = str(exc)
        atomic(telemetry / (host + ".json"), row)
        with history.open("a") as f:
            f.write(json.dumps(row) + "\n")
        time.sleep(seconds)


def sampler_rows(telemetry, nodes, cfg):
    rows = [read(p) for p in telemetry.glob("*.json")]
    if len(rows) != nodes or any(time.time() - r["epoch"] > cfg["telemetry_stale_seconds"] for r in rows):
        raise RuntimeError("missing/stale per-node memory telemetry")
    return rows


def finalization_reserve(cfg, state):
    times = [c.get("finalization_seconds", 0) for c in state.get("chunks", [])]
    return max(cfg["finalization_reserve_seconds"], math.ceil(2 * max(times, default=0)))


def launch_seconds(end_epoch, cfg, state, now=None):
    remaining = end_epoch - (time.time() if now is None else now)
    return max(0, int(min(cfg["checkpoint_wall_seconds"], remaining - finalization_reserve(cfg, state))))


def memory_summary(telemetry):
    peaks, rates, host_peaks = {}, {}, {}
    for p in telemetry.glob("*.jsonl"):
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        valid = [r for r in rows if not r.get("error")]
        if not valid:
            raise ValueError("no valid memory history")
        host_peaks[p.stem] = max(r["host_used_bytes"] / r["host_total_bytes"] for r in valid)
        g = [r["gpus"][0] for r in valid]
        peaks[g[0]["uuid"]] = max(x["used_mib"] / x["total_mib"] for x in g)
        if len(valid) > 2:
            # Exclude startup; this is residual-growth evidence, not a bounded-memory proof.
            a, b = valid[len(valid) // 3], valid[-1]
            minutes = (b["epoch"] - a["epoch"]) / 60
            rates[g[0]["uuid"]] = (b["gpus"][0]["used_mib"] - a["gpus"][0]["used_mib"]) / minutes if minutes > 0 else None
    return dict(gpu_peak_fractions=peaks, host_peak_fractions=host_peaks,
                gpu_endpoint_growth_mib_per_minute=rates, long_time_growth_certified=False)


def launch_athena(run, label, deck, cfg, ranks, seconds, restart=None, overrides=(), rootlevel=3):
    if seconds < cfg["minimum_launch_seconds"]:
        raise RuntimeError("insufficient wall time for launch and finalization reserve")
    case = run / label
    case.mkdir(parents=True, exist_ok=False)
    out = case / "out"
    out.mkdir()
    telemetry = case / "telemetry"
    telemetry.mkdir()
    stop = case / "STOP_SAMPLERS"
    if restart:
        verify_restart(restart, rootlevel)
    args = ["ibrun", "-n", str(ranks), str(EXE), "--kokkos-map-device-id-by=mpi_rank"]
    args += ["-r", restart["path"]] if restart else ["-i", str(deck)]
    args += ["-d", str(out), "-t", f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}",
             "time/stop_file=" + str(out / "REQUEST_STOP"), "time/tlim=1e30", "time/nlim=-1",
             "mesh_refinement/max_nmb_per_rank=" + str(cfg["max_nmb_per_rank"])] + list(overrides)
    if restart:
        with Path(restart["path"]).open("rb") as f:
            header = f.read(1024**2).split(b"<par_end>", 1)[0].decode("utf-8")
        validate_overrides(header, args)
    else:
        validate_overrides(deck.read_text(), args)
    nodes = int(os.environ["SLURM_NNODES"])
    sampler = ["srun", "--overlap", "--nodes=" + str(nodes), "--ntasks=" + str(nodes), "--ntasks-per-node=1", "--cpus-per-task=1", sys.executable, str(HERE / "s10.py"), "sample", str(telemetry), str(stop), "--seconds=" + str(cfg["sample_seconds"])]
    atomic(case / "launch.json", dict(command=args, sampler=sampler, source=read(HANDOFF / "PROVENANCE.json"), input_sha256=digest(deck), executable_sha256=digest(EXE), host=socket.gethostname(), job_id=os.environ["SLURM_JOB_ID"], started_epoch=time.time(), restart=restart))
    reason = None
    started = time.time()
    end_epoch = job_end_epoch()
    proc = None
    with (case / "sampler.log").open("w") as sf, (case / "athena.log").open("w") as log:
        sam = subprocess.Popen(sampler, stdout=sf, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.time() + cfg["telemetry_stale_seconds"]
            while True:
                try:
                    rows = sampler_rows(telemetry, nodes, cfg)
                    reason = stop_reason(rows, cfg)
                    break
                except (RuntimeError, ValueError, OSError):
                    if sam.poll() is not None or time.time() >= deadline:
                        raise RuntimeError("per-node GPU/host sampler failed startup")
                    time.sleep(1)
            if reason:
                raise RuntimeError("memory/telemetry guard failed before launch: " + reason)
            # Sampler setup consumes allocation time; recompute before starting MPI.
            safe_seconds = min(seconds, end_epoch - int(time.time()) - cfg["finalization_reserve_seconds"])
            if safe_seconds < cfg["minimum_launch_seconds"]:
                raise RuntimeError("insufficient wall time after sampler startup")
            seconds = safe_seconds
            args[args.index("-t") + 1] = f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
            atomic(case / "actual_command.json", dict(command=args, launched_epoch=time.time()))
            proc = subprocess.Popen(args, cwd=out, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            seen_mesh = None
            mesh = None
            finalizing = False
            finalizing_epoch = None
            while proc.poll() is None:
                try:
                    rows = sampler_rows(telemetry, nodes, cfg)
                    reason = reason or stop_reason(rows, cfg)
                except (RuntimeError, ValueError, OSError):
                    reason = reason or "telemetry_failure"
                if (run / "STOP_REQUESTED").exists():
                    reason = reason or "user_stop"
                cfiles = sorted(out.rglob("*.mesh.*.cbin"))
                if cfiles:
                    candidate = cfiles[-1]
                    identity = (str(candidate), candidate.stat().st_size, candidate.stat().st_mtime_ns)
                    if identity != seen_mesh:
                        try:
                            mesh = mesh_info(candidate)
                            seen_mesh = identity
                        except (ValueError, OSError, EOFError, struct.error, TypeError, KeyError, IndexError):
                            pass  # May be observing an in-progress write.
                    if rootlevel == 3 and mesh and mesh["blocks"] >= cfg["capacity_stop_fraction"] * ranks * cfg["max_nmb_per_rank"]:
                        reason = reason or "resource_limit_mesh_capacity"
                blocks = mesh["blocks"] if mesh else (4936 if rootlevel == 3 else 792)
                # Native state plus forced bulk outputs, another checkpoint, and a 2% filesystem cushion.
                storage_needed = 6 * blocks * 25 * 40**3 * 8 + 2 * 5_000_000 * 80
                usage = shutil.disk_usage(out)
                if usage.free < storage_needed + cfg["storage_reserve_fraction"] * usage.total:
                    reason = reason or "resource_limit_storage"
                elapsed = time.time() - started
                if elapsed > cfg["startup_grace_seconds"] and not finalizing:
                    paths = [sorted(out.glob(f"*.co_{i}.txt")) for i in (0, 1)]
                    if any(not ps or time.time() - ps[-1].stat().st_mtime > cfg["diagnostic_stale_seconds"] for ps in paths):
                        reason = reason or "diagnostic_stale"
                log.flush()
                with (case / "athena.log").open("rb") as f:
                    f.seek(max(0, (case / "athena.log").stat().st_size - 65536))
                    tail = f.read().decode(errors="replace")
                if "leaving the evolution loop" in tail or elapsed >= seconds:
                    finalizing = True
                    finalizing_epoch = finalizing_epoch or time.time()
                if "### FATAL ERROR" in tail or "PARTICLE LEDGER FAIL" in tail:
                    reason = reason or "numerical_failure"
                if reason:
                    (out / "REQUEST_STOP").touch(exist_ok=True)
                    finalizing_epoch = finalizing_epoch or time.time()
                atomic(case / "telemetry.json", dict(updated_epoch=time.time(), samples=rows, mesh=mesh,
                      capacity_measurement="global occupancy proxy; does not bound instantaneous per-rank AMR jumps",
                      memory_warning=any(r.get("error") or r["host_used_bytes"] / r["host_total_bytes"] >= cfg["memory_warning_fraction"] or any(g["used_mib"] / g["total_mib"] >= cfg["memory_warning_fraction"] for g in r.get("gpus", [])) for r in rows),
                      stop_reason=reason, finalizing=finalizing, storage_free_bytes=usage.free, storage_required_bytes=storage_needed))
                time.sleep(cfg["sample_seconds"])
            rc = proc.returncode
            if rc != 0:
                reason = reason or "application_failure"
        finally:
            if proc is not None and proc.poll() is None:
                # An unexpected wrapper exception must not leave MPI unattended.
                (out / "REQUEST_STOP").touch(exist_ok=True)
                try:
                    proc.wait(timeout=max(1, min(300, end_epoch - int(time.time()) - 30)))
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait(timeout=10)
            stop.touch(exist_ok=True)
            try:
                sam.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(sam.pid, signal.SIGTERM)
                try:
                    sam.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(sam.pid, signal.SIGKILL)
                    sam.wait(timeout=10)
    result = dict(label=label, output=str(out), exit_code=rc, stop_reason=reason,
                  elapsed_seconds=time.time() - started, checkpoint=None)
    try:
        cp = latest_checkpoint(out, rootlevel)
        result["checkpoint"] = cp
        if rc == 0:
            checkpoint_matches_exit(cp, case / "athena.log")
        result["memory"] = memory_summary(telemetry)
        result["eligible_amr_change"] = eligible_amr(case / "athena.log")
        result["finalization_seconds"] = time.time() - (finalizing_epoch or (started + min(seconds, result["elapsed_seconds"])))
    except (ValueError, OSError, struct.error) as exc:
        result["verification_error"] = str(exc)
        result["stop_reason"] = reason or "checkpoint_failure"
    # Include all output sizes; hashing the native checkpoint certifies the restart separately.
    inventory = [dict(path=str(p.relative_to(case)), bytes=p.stat().st_size) for p in sorted(out.rglob("*")) if p.is_file()]
    atomic(case / "output_inventory.json", inventory)
    atomic(case / "result.json", result)
    atomic(case / "SEAL_RESULT.json", seal_receipt(result))
    return result


def require_success(result):
    if result["exit_code"] or result["stop_reason"] or not result["checkpoint"]:
        raise RuntimeError(f"case failed: {result['label']}: {result.get('stop_reason')}")
    return result["checkpoint"]


def comparison(reference, restarted, output):
    subprocess.run([sys.executable, str(HANDOFF / "scripts/compare_restart_s7.py"),
                    "--reference", reference["output"], "--restarted", restarted["output"],
                    "--athenak", str(SOURCE), "--output", str(output)], check=True)
    return read(output)


def debug_job(run, cfg):
    verify_package()
    base = run / "debug"
    base.mkdir(exist_ok=False)
    deck = run / "compact.athinput"
    common = ["z4c/tracker_walk_every=1", "z4c/co_0_out_every=1", "z4c/co_1_out_every=1"]
    def allowance(requested):
        return min(requested, launch_seconds(job_end_epoch(), cfg, {}))
    ref = launch_athena(run, "debug/reference", deck, cfg, 2, allowance(1200), overrides=common + ["time/nlim=2"], rootlevel=1)
    require_success(ref)
    init = initial_gate(Path(ref["output"]), base / "reference/athena.log", 64)
    seed = launch_athena(run, "debug/seed", deck, cfg, 2, allowance(1200), overrides=common + ["time/nlim=1"], rootlevel=1)
    cp = require_success(seed)
    resumed = launch_athena(run, "debug/resumed", deck, cfg, 3, allowance(1200), cp, common + ["time/nlim=2"], rootlevel=1)
    require_success(resumed)
    comp = comparison(ref, resumed, base / "restart_comparison.json")
    amr = launch_athena(run, "debug/amr", deck, cfg, 3, allowance(1800), resumed["checkpoint"], common + ["time/nlim=50"], rootlevel=1)
    cp = require_success(amr)
    if not amr["eligible_amr_change"] or cp["cycle"] < 34:
        raise RuntimeError("compact restart gate did not cross an actual eligible adaptive change")
    native = audit_native(cp, fields=True)
    marker = base / "clean_stop.marker"
    marker.touch(exist_ok=False)
    clean = launch_athena(run, "debug/clean_stop", deck, cfg, 3, allowance(1200), cp,
                          common + ["time/stop_file=" + str(marker), "time/stop_check_every=1"], rootlevel=1)
    clean_cp = require_success(clean)
    if clean_cp["cycle"] != cp["cycle"] or clean_cp["time"] != cp["time"] or "Terminating on clean stop-file request" not in (base / "clean_stop/athena.log").read_text():
        raise RuntimeError("native clean-stop checkpoint gate failed")
    atomic(run / "DEBUG_VERIFIED.json", dict(passed=True, initialization=init, restart=comp, native=native, amr=amr, clean_stop=clean, executable_sha256=digest(EXE), input_sha256=digest(deck), job_id=os.environ["SLURM_JOB_ID"]))


def production_job(run, cfg, segment):
    verify_package()
    debug = read(run / "DEBUG_VERIFIED.json")
    if not debug["passed"] or debug["executable_sha256"] != digest(EXE):
        raise RuntimeError("debug receipt executable mismatch")
    if segment > 1:
        previous = read(run / f"segment_{segment - 1:03d}.json")
        if previous["status"] != "verified_segment_end":
            raise RuntimeError("predecessor did not release continuation")
        full = read(run / "FULL_SIZE_VERIFIED.json")
        if not full["passed"] or full["executable_sha256"] != digest(EXE):
            raise RuntimeError("full-size release gate missing or changed executable")
    statepath = run / "state.json"
    state = read(statepath)
    if state["status"] not in ("prepared", "verified_segment_end", "running"):
        raise RuntimeError("sticky stop blocks production")
    if state.get("checkpoint"):
        verify_restart(state["checkpoint"], 3)
    starting_cycle = state["checkpoint"]["cycle"] if state.get("checkpoint") else -1
    end = job_end_epoch()
    deck = run / "production.athinput"
    root = run / f"segments/segment_{segment:03d}"
    root.mkdir(parents=True, exist_ok=False)
    state["status"] = "running"
    save_state(run, state)
    full_evidence = None
    if segment == 1:
        common = ["z4c/tracker_walk_every=1", "z4c/co_0_out_every=1", "z4c/co_1_out_every=1"]
        ref = launch_athena(run, "validation_full/reference", deck, cfg, 16,
                            min(1200, launch_seconds(end, cfg, state)), overrides=common + ["time/nlim=2"])
        require_success(ref)
        initial = initial_gate(Path(ref["output"]), run / "validation_full/reference/athena.log", 512)
        coarse = sorted(Path(ref["output"]).rglob("*.mesh.*.cbin"))
        if not coarse or mesh_info(coarse[0])["blocks"] != 4936:
            raise RuntimeError("full production mesh is not 4,936 initial blocks")
        seed = launch_athena(run, "segments/segment_001/prefix_seed", deck, cfg, 16,
                             min(1200, launch_seconds(end, cfg, state)), overrides=common + ["time/nlim=1"])
        if seed["checkpoint"]:
            state = promoted_state(state, seed["checkpoint"], seed["stop_reason"] or "running", seed)
            save_state(run, state)
        cp = require_success(seed)
        resumed = launch_athena(run, "segments/segment_001/prefix_resumed", deck, cfg, 16,
                                min(1200, launch_seconds(end, cfg, state)), cp,
                                common + ["time/nlim=2"])
        if resumed["checkpoint"]:
            state = promoted_state(state, resumed["checkpoint"], resumed["stop_reason"] or "running", resumed)
            save_state(run, state)
        cp = require_success(resumed)
        comp = comparison(ref, resumed, run / "validation_full/restart_comparison.json")
        # The two-step restart precedes the next waveform sample (.025M).
        # Validate initial wave output on the continuous reference, and evolved
        # production wave coverage on a later uninterrupted application chunk.
        full_evidence = dict(initialization=initial, restart=comp, prefix_native=audit_native(cp), initial_outputs=output_gate(Path(ref["output"])))
    n = 0
    while True:
        seconds = launch_seconds(end, cfg, state)
        if seconds < cfg["minimum_launch_seconds"]:
            break
        if (run / "STOP_REQUESTED").exists():
            state["status"] = "user_stop"
            save_state(run, state)
            break
        n += 1
        result = launch_athena(run, f"segments/segment_{segment:03d}/chunk_{n:03d}", deck, cfg, 16, seconds,
                               state.get("checkpoint"), ["z4c/tracker_walk_every=16", "z4c/co_0_out_every=20", "z4c/co_1_out_every=20"])
        if result["checkpoint"]:
            state = promoted_state(state, result["checkpoint"], result["stop_reason"] or "running", result)
            save_state(run, state)
        if result["exit_code"] or result["stop_reason"] or not result["checkpoint"]:
            state["status"] = result["stop_reason"] or "application_failure"
            save_state(run, state)
            break
    if state["status"] != "running":
        raise RuntimeError("segment stopped: " + state["status"])
    if not state.get("checkpoint"):
        raise RuntimeError("job ended without a verified checkpoint")
    if state["checkpoint"]["cycle"] <= starting_cycle:
        raise RuntimeError("no evolution progress in this allocation; continuation held")
    if segment == 1:
        chunks = [c for c in state["chunks"] if "/chunk_" in c["label"]]
        if not chunks or not any(c["eligible_amr_change"] for c in chunks):
            raise RuntimeError("full-size AMR change not observed; preserve progress and hold long jobs")
        full_evidence.update(passed=True, executable_sha256=digest(EXE), production_input_sha256=digest(deck),
                             final_native=audit_native(state["checkpoint"], fields=True), chunks=chunks,
                             evolved_outputs=output_gate(Path(chunks[-1]["output"])),
                             job_id=os.environ["SLURM_JOB_ID"], timestamp=time.time())
        atomic(run / "FULL_SIZE_VERIFIED.json", full_evidence)
    state.update(status="verified_segment_end", segment=segment, accounting=accounting(run))
    save_state(run, state)
    atomic(run / f"segment_{segment:03d}.json", state)


def save_state(run, state):
    atomic(run / "state.json", state)
    atomic(run / "PRODUCTION_OUTPUTS.json", dict(
            ordered_run_directories=[c["output"] for c in state["chunks"]],
            excluded_validation_roots=[str(run / "validation_full"), str(run / "debug")],
            checkpoint=state["checkpoint"], scientific_completion=False,
            full_size_gate_passed=(run / "FULL_SIZE_VERIFIED.json").exists(),
            coverage_note="Native checkpoint time is evolution progress; tracker and waveform gaps remain separate."))


def job(stage, run, segment):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("compute allocation required")
    if os.environ.get("SLURM_JOB_PARTITION") != ("gh-dev" if stage == "debug" else "gh"):
        raise RuntimeError("incorrect compute partition")
    try:
        cfg = read(run / "config.json")
        validate_config(cfg)
        for name, sha in read(run / "INPUT_SHA256.json").items():
            if digest(run / name) != sha:
                raise RuntimeError("reviewed configuration/input changed: " + name)
        for name, sha in read(run / "LAUNCHER_SHA256.json").items():
            if digest(HERE / name) != sha:
                raise RuntimeError("launcher changed after submission: " + name)
        if int(os.environ["SLURM_NNODES"]) != (cfg["debug_nodes"] if stage == "debug" else 16):
            raise RuntimeError("allocation node count mismatch")
        if (run / "STOP_REQUESTED").exists():
            state = read(run / "state.json")
            state["status"] = "user_stop"
            save_state(run, state)
            raise RuntimeError("user stop precedes this job; continuation held")
        if stage == "debug":
            debug_job(run, cfg)
        else:
            production_job(run, cfg, segment)
    except BaseException as exc:
        sp = run / "state.json"
        state = read(sp)
        state.update(status=state["status"] if state["status"] not in ("running", "prepared", "verified_segment_end") else "verification_or_application_failure", failure=str(exc), failed_stage=stage, failed_segment=segment)
        with contextlib.suppress(Exception):
            state["accounting"] = accounting(run)
        save_state(run, state)
        atomic(run / f"FAILURE_{stage}_{segment}.json", state)
        raise


def prepare_run(run, cfg):
    run.mkdir(parents=True, exist_ok=False)
    (run / "logs").mkdir()
    atomic(run / "config.json", cfg)
    pins = read(HANDOFF / "PROVENANCE.json")
    atomic(run / "PROVENANCE.json", pins)
    original = (HANDOFF / "inputs/gi_cluster_s10.athinput").read_text()
    production = (HERE / "production.athinput").read_text()
    if production != production_deck(original):
        raise ValueError("reviewed production input differs from approved derivation")
    for nlim in (-1, 1, 2):
        validate_overrides(production, production_overrides(nlim=nlim, tracker_every=1 if nlim > 0 else 20, walk_every=1 if nlim > 0 else 16))
    (run / "production.athinput").write_text(production)
    compact = (HANDOFF / "evidence/VALIDATION_ONLY_compact.athinput").read_text()
    # Remove the artificial large-header fixture comment; retain all model and geometry.
    compact = re.sub(r"\n<comment>\nvalidation_padding =[^\n]*\n?", "\n", compact).rstrip() + "\n"
    if compact != (HERE / "compact.athinput").read_text():
        raise ValueError("compact fixture differs from reviewed validation geometry")
    (run / "compact.athinput").write_text(compact)
    atomic(run / "state.json", dict(status="prepared", checkpoint=None, chunks=[], scientific_completion=False))
    atomic(run / "INPUT_SHA256.json", {p.name: digest(p) for p in (run / "production.athinput", run / "compact.athinput", run / "config.json", run / "PROVENANCE.json")})
    atomic(run / "LAUNCHER_SHA256.json", {name: digest(HERE / name) for name in ("s10.py", "job.sbatch", "build.sh", "submit.sh", "submit_gh.sh", "production.athinput", "compact.athinput")})


def prepare_gh_retry(run, cfg, previous):
    if not build_ready():
        raise RuntimeError("verified native build required; GH-only submission will not queue a build")
    gate_path = previous / "DEBUG_VERIFIED.json"
    gate = read(gate_path)
    if not gate.get("passed") or gate.get("executable_sha256") != digest(EXE) or gate.get("input_sha256") != digest(HERE / "compact.athinput"):
        raise RuntimeError("previous compact gate does not match this executable/fixture")
    parser = read(HANDOFF / "runtime_s10/build_parser_check/evidence/PARSER_VERIFIED.json")
    if not parser["passed"] or parser["production_input_sha256"] != digest(HERE / "production.athinput"):
        raise RuntimeError("exact native parser preflight is missing or stale")
    prepare_run(run, cfg)
    atomic(run / "DEBUG_VERIFIED.json", gate)
    atomic(run / "REUSED_DEBUG.json", dict(source=str(gate_path), source_sha256=digest(gate_path),
            job_id=gate["job_id"], executable_sha256=gate["executable_sha256"],
            copied_sha256=digest(run / "DEBUG_VERIFIED.json"), repeated_debug_job=False))
    atomic(run / "NATIVE_PARSER_VERIFIED.json", parser)
    atomic(run / "RETRY_PROVENANCE.json", dict(previous_run=str(previous), reason="missing explicit unlimited nlim key", fresh_t0=True,
            previous_production_job="1069970", previous_production_checkpoint=None,
            direct_gh_wall_hours=[8, 24, 24], native_executable_unchanged=True))


def submit_ready_gh(resume=False, dry_run=False, runner=subprocess.run):
    ready = HANDOFF / "runtime_s10/GH_READY_RUN.json"
    run = Path(read(ready)["run"])
    cfg = read(run / "config.json")
    validate_config(cfg)
    specs = gh_submission_specs(cfg)
    if dry_run:
        print(json.dumps(dict(submitted=False, run=str(run), jobs=[dict(stage=s[0], partition=s[1], nodes=s[2], wall_hours=s[3]) for s in specs],
                commands=[shlex.join(sbatch_args(s, run, "PREDECESSOR") if i else sbatch_args(s, run)) for i, s in enumerate(specs)]), indent=2))
        return
    if not re.match(r"^login\d+(?:\.|$)", socket.gethostname().lower()) or os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("submit manually from a Vista login node, outside an allocation")
    with (ready.parent / "submit.lock").open("a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        for name, sha in read(run / "INPUT_SHA256.json").items():
            if digest(run / name) != sha:
                raise RuntimeError("prepared GH input/config changed: " + name)
        for name, sha in read(run / "LAUNCHER_SHA256.json").items():
            if digest(HERE / name) != sha:
                raise RuntimeError("prepared GH launcher changed: " + name)
        if not build_ready(full=False):
            raise RuntimeError("native build missing; no debug/build jobs will be submitted")
        pointer = HANDOFF / "runtime_s10/ACTIVE_RUN.json"
        previous = read(pointer)
        if previous["run"] != str(run):
            old_run = Path(previous["run"])
            # Never silently replace an active or ambiguously submitted chain.
            ledger = read(old_run / "submission.json")
            ids = [j["job_id"] for j in ledger["jobs"]]
            import pwd
            user = pwd.getpwuid(os.getuid()).pw_name
            live = subprocess.run(["squeue", "-h", "-u", user, "-o", "%i|%T"], text=True, capture_output=True, timeout=20, check=True)
            old_ids = set(ids)
            if any(line.split("|", 1)[0].strip() in old_ids for line in live.stdout.splitlines()) or ledger.get("pending"):
                raise RuntimeError("previous chain has live/uncertain jobs; no duplicate GH submission")
            if not (run / "PREVIOUS_ACTIVE_RUN.json").exists():
                atomic(run / "PREVIOUS_ACTIVE_RUN.json", previous)
            atomic(pointer, dict(run=str(run), created_epoch=time.time(), previous_run=previous["run"]))
        result = submit_chain(run, specs, runner, resume=resume)
        print(json.dumps(dict(run=str(run), **result), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("dry-run")
    submission = sub.add_parser("submit")
    submission.add_argument("--resume", action="store_true", help="recover a confirmed preparation-stage submission and queue only missing dependent jobs")
    gh = sub.add_parser("submit-gh")
    gh.add_argument("--dry-run", action="store_true")
    gh.add_argument("--resume", action="store_true")
    status = sub.add_parser("status"); status.add_argument("run", type=Path, nargs="?")
    stop = sub.add_parser("stop"); stop.add_argument("run", type=Path, nargs="?")
    jp = sub.add_parser("job"); jp.add_argument("stage", choices=("debug", "production")); jp.add_argument("run", type=Path); jp.add_argument("--segment", type=int, default=0)
    sampler = sub.add_parser("sample"); sampler.add_argument("telemetry", type=Path); sampler.add_argument("stop", type=Path); sampler.add_argument("--seconds", type=int, default=10)
    args = p.parse_args()
    if args.command == "sample":
        sample(args.telemetry, args.stop, args.seconds)
        return
    cfg = config(); validate_config(cfg)
    pointer = HANDOFF / "runtime_s10/ACTIVE_RUN.json"
    if args.command == "dry-run":
        run = HANDOFF / "runtime_s10/runs/NEW_RUN"
        specs = submission_specs(cfg, build_ready(full=False))
        print(json.dumps(dict(submitted=False, jobs=[dict(stage=s[0], nodes=s[2], wall_hours=s[3], segment=s[4]) for s in specs], production_node_hours=896, commands=[shlex.join(sbatch_args(s, run, "PREDECESSOR") if i else sbatch_args(s, run)) for i, s in enumerate(specs)]), indent=2))
    elif args.command == "submit-gh":
        submit_ready_gh(resume=args.resume, dry_run=args.dry_run)
    elif args.command == "submit":
        if not re.match(r"^login\d+(?:\.|$)", socket.gethostname().lower()) or os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("submit manually from a Vista login node, outside an allocation")
        pointer.parent.mkdir(parents=True, exist_ok=True)
        with (pointer.parent / "submit.lock").open("a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            if pointer.exists():
                if not args.resume:
                    raise RuntimeError("an active run already exists; inspect status or use submit.sh --resume to recover preparation")
                run = Path(read(pointer)["run"])
            else:
                if args.resume:
                    raise RuntimeError("no existing submission to resume")
                stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                run = HANDOFF / "runtime_s10/runs" / stamp
                prepare_run(run, cfg)
                atomic(pointer, dict(run=str(run), created_epoch=time.time()))
            result = submit_chain(run, submission_specs(read(run / "config.json"), build_ready(full=False)), resume=args.resume)
            print(json.dumps(dict(run=str(run), **result), indent=2))
    elif args.command in ("status", "stop"):
        if args.run is None and not pointer.exists():
            if args.command == "stop":
                raise RuntimeError("no Session010 chain has been submitted")
            print(json.dumps(dict(status="not_submitted", native_build_ready=build_ready(full=False), full_size_gates="pending user-submitted GH jobs"), indent=2))
            return
        run = args.run or Path(read(pointer)["run"])
        if args.command == "stop":
            (run / "STOP_REQUESTED").touch(exist_ok=True)
            print("Graceful stop requested:", run)
        else:
            state = read(run / "state.json")
            state["accounting"] = accounting(run)
            print(json.dumps(state, indent=2))
    else:
        job(args.stage, args.run, args.segment)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("SESSION010:", exc, file=sys.stderr)
        sys.exit(42)
