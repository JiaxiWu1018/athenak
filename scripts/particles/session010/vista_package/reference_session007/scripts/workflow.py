#!/usr/bin/env python3
"""Bounded, persistent Slurm controller for GI Session 007 on Vista.

No daemon or agent is required after submission. State changes are flock-protected and
atomically replaced. A full workflow is pre-submitted as a finite dependency chain, so
batch-to-batch submission support is not assumed.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

TERMINAL = {
    "complete_physical_endpoint",
    "authorized_scientific_stop",
    "user_stop",
    "resource_limit",
    "numerical_failure",
    "infrastructure_failure",
    "configuration_test_failure",
    "diagnostic_stale",
    "no_progress",
    "segment_limit",
    "emergency_cancel",
}
INFRA_STATES = {"BOOT_FAIL", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "REVOKED"}
SUCCESS_STATES = {"COMPLETED"}
MINIMUM_PRODUCTION_CHECKPOINT_BYTES = 8 * 1024**3
MAX_PRODUCTION_RESTART_HEADER_BYTES = 1024 * 1024


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    return value


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def atomic_json(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextlib.contextmanager
def state_lock(control: pathlib.Path) -> Iterator[None]:
    control.mkdir(parents=True, exist_ok=True)
    lock = control / "workflow.lock"
    with lock.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def restart_parameter_end_offset(path: pathlib.Path) -> int:
    """Return the production restart's <par_end> offset, bounded by the reader limit."""
    with path.open("rb") as stream:
        header = stream.read(MAX_PRODUCTION_RESTART_HEADER_BYTES + 1)
    return header.find(b"<par_end>")


def wall_hours(value: str) -> float:
    parts = [int(x) for x in value.split(":")]
    if len(parts) != 3 or any(x < 0 for x in parts):
        raise ValueError(f"invalid HH:MM:SS wall time: {value}")
    return parts[0] + parts[1] / 60.0 + parts[2] / 3600.0


def validate_config(cfg: Dict[str, Any], full: bool = False) -> List[str]:
    errors: List[str] = []
    required = [
        "campaign_id", "account", "handoff_dir", "work_root", "scratch_root",
        "source_commit", "kokkos_commit", "input_relpath", "input_sha256",
        "control_input_relpath", "control_input_sha256", "restart_test_ranks",
        "quicktest_gpu_memory_fraction_limit", "quicktest_storage_stop_bytes",
        "production_nodes", "production_max_nmb_per_rank", "startup_nodes",
        "max_segments", "segment_wall",
        "athena_wall", "node_hour_cap", "storage_warn_bytes", "storage_stop_bytes",
        "checkpoint_reserve_bytes", "monitor_seconds", "telemetry_grace_seconds",
        "startup_telemetry_grace_seconds", "max_infrastructure_retries_per_segment",
        "target_time", "modules", "ucx_ib_gpu_direct_rdma", "quicktest_wall",
        "quicktest_nlim", "build_jobs",
        "production_authorized", "minimum_checkpoint_bytes",
        "gpu_partition", "dev_gpu_partition", "cpu_partition",
        "verify_wall", "build_wall", "inspect_wall",
    ]
    for key in required:
        if key not in cfg:
            errors.append(f"missing config key: {key}")
    if errors:
        return errors
    cid = str(cfg["campaign_id"])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,39}", cid):
        errors.append("campaign_id must be 3-40 safe characters")
    if "REPLACE" in str(cfg["account"]).upper() or not str(cfg["account"]).strip():
        errors.append("account is still a placeholder")
    if "RECHECK" in str(cfg["modules"]).upper() or not str(cfg["modules"]).strip():
        errors.append("Vista modules must be rechecked and explicitly configured")
    if str(cfg["ucx_ib_gpu_direct_rdma"]) not in ("no", "try", "yes"):
        errors.append("ucx_ib_gpu_direct_rdma must be no, try, or yes")
    if not isinstance(cfg["build_jobs"], int) or int(cfg["build_jobs"]) < 1:
        errors.append("build_jobs must be a positive integer")
    for key in ("handoff_dir", "work_root", "scratch_root"):
        if not pathlib.Path(str(cfg[key])).is_absolute():
            errors.append(f"{key} must expand to an absolute path")
    if not re.fullmatch(r"[0-9a-f]{40}", str(cfg["source_commit"])):
        errors.append("source_commit must be a full 40-hex commit")
    if not re.fullmatch(r"[0-9a-f]{40}", str(cfg["kokkos_commit"])):
        errors.append("kokkos_commit must be a full 40-hex commit")
    if not re.fullmatch(r"[0-9a-f]{64}", str(cfg["input_sha256"])):
        errors.append("input_sha256 must be 64 lowercase hex characters")
    if not re.fullmatch(r"[0-9a-f]{64}", str(cfg["control_input_sha256"])):
        errors.append("control_input_sha256 must be 64 lowercase hex characters")
    bundle_rel = cfg.get("source_bundle_relpath")
    bundle_sha = cfg.get("source_bundle_sha256")
    if bundle_rel is not None or bundle_sha is not None:
        bundle_path = pathlib.PurePosixPath(str(bundle_rel or ""))
        if (not bundle_rel or bundle_path.is_absolute() or ".." in bundle_path.parts):
            errors.append("source_bundle_relpath must be a safe relative path")
        if not re.fullmatch(r"[0-9a-f]{64}", str(bundle_sha or "")):
            errors.append("source_bundle_sha256 must be 64 lowercase hex characters")
        elif bundle_rel:
            source_bundle = pathlib.Path(str(cfg["handoff_dir"])) / str(bundle_rel)
            if not source_bundle.is_file():
                errors.append("configured source bundle does not exist")
            elif sha256(source_bundle) != bundle_sha:
                errors.append("configured source bundle checksum differs")
    for key in ("production_nodes", "startup_nodes", "max_segments"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            errors.append(f"{key} must be a positive integer")
    if cfg["production_nodes"] > 64 or cfg["startup_nodes"] > 64:
        errors.append("Vista gh jobs may not exceed 64 nodes")
    if cfg["production_nodes"] < 8 or cfg["startup_nodes"] < 8:
        errors.append("canonical Session 007 startup and production require at least 8 GH nodes")
    if (not isinstance(cfg["production_max_nmb_per_rank"], int)
            or not 338 <= cfg["production_max_nmb_per_rank"] <= 1600):
        errors.append("production_max_nmb_per_rank must be an integer in [338,1600]")
    if not isinstance(cfg["restart_test_ranks"], int) or not (
            1 <= cfg["restart_test_ranks"] <= cfg["startup_nodes"]):
        errors.append("restart_test_ranks must be between 1 and startup_nodes")
    elif cfg["restart_test_ranks"] != 4:
        errors.append("canonical Session 007 rank-change test is pinned to 8-to-4 ranks")
    try:
        segment_hours = wall_hours(str(cfg["segment_wall"]))
        exposure_hours = cfg["production_nodes"] * segment_hours * cfg["max_segments"]
        if exposure_hours > float(cfg["node_hour_cap"]) + 1.0e-12:
            errors.append(
                f"maximum segment exposure {exposure_hours:.3f} node-hours exceeds cap "
                f"{cfg['node_hour_cap']}"
            )
        if float(cfg["node_hour_cap"]) > 4608.0:
            errors.append("node_hour_cap exceeds the authorized 4608 node-hour ceiling")
        if wall_hours(str(cfg["athena_wall"])) >= segment_hours:
            errors.append("athena_wall must be shorter than segment_wall")
        if segment_hours > 48.0:
            errors.append("Vista gh segment_wall may not exceed 48 hours")
        if wall_hours(str(cfg["quicktest_wall"])) > 2.0:
            errors.append("Vista gh-dev quicktest_wall may not exceed 2 hours")
        if wall_hours(str(cfg["build_wall"])) > 2.0:
            errors.append("Vista gh-dev build_wall may not exceed 2 hours")
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    if int(cfg["monitor_seconds"]) < 300:
        errors.append("monitor_seconds must be at least the declared 300-second default")
    if int(cfg["telemetry_grace_seconds"]) < 2 * int(cfg["monitor_seconds"]):
        errors.append("telemetry_grace_seconds must span at least two monitor intervals")
    if int(cfg["startup_telemetry_grace_seconds"]) < int(cfg["telemetry_grace_seconds"]):
        errors.append("startup telemetry grace must not be shorter than steady-state grace")
    if not isinstance(cfg["quicktest_nlim"], int) or int(cfg["quicktest_nlim"]) < 1:
        errors.append("quicktest_nlim must be a positive integer")
    if not (0.0 < float(cfg["quicktest_gpu_memory_fraction_limit"]) <= 0.90):
        errors.append("quicktest GPU memory fraction limit must be in (0,0.90]")
    if int(cfg["quicktest_storage_stop_bytes"]) < 1:
        errors.append("quicktest storage stop must be positive")
    if not isinstance(cfg["minimum_checkpoint_bytes"], int) or int(
            cfg["minimum_checkpoint_bytes"]) < MINIMUM_PRODUCTION_CHECKPOINT_BYTES:
        errors.append(
            "minimum_checkpoint_bytes must be at least 8 GiB for this production"
        )
    if int(cfg["max_infrastructure_retries_per_segment"]) not in (0, 1):
        errors.append("infrastructure retries must be zero or one")
    initial_checkpoint = cfg.get("initial_checkpoint")
    if initial_checkpoint:
        checkpoint = pathlib.Path(str(initial_checkpoint))
        if not checkpoint.is_absolute():
            errors.append("initial_checkpoint must be an absolute path")
        if not re.fullmatch(r"[0-9a-f]{64}", str(
                cfg.get("initial_checkpoint_sha256", ""))):
            errors.append("initial_checkpoint_sha256 must be 64 lowercase hex characters")
        if not isinstance(cfg.get("initial_checkpoint_bytes"), int) or int(
                cfg.get("initial_checkpoint_bytes", 0)) < int(cfg["minimum_checkpoint_bytes"]):
            errors.append("initial_checkpoint_bytes is absent or below the checkpoint minimum")
        if not isinstance(cfg.get("initial_cycle"), int) or int(
                cfg.get("initial_cycle", 0)) < 1:
            errors.append("initial_cycle must be a positive integer")
        try:
            initial_time = float(cfg.get("initial_time", 0.0))
            if not 0.0 < initial_time < float(cfg["target_time"]):
                errors.append("initial_time must be between zero and target_time")
        except (TypeError, ValueError):
            errors.append("initial_time must be numeric")
        if checkpoint.exists():
            if not checkpoint.is_file():
                errors.append("initial_checkpoint is not a regular file")
            elif checkpoint.stat().st_size != int(cfg.get("initial_checkpoint_bytes", -1)):
                errors.append("initial_checkpoint size differs from the recorded size")
            elif bool(cfg.get("production_restart_smoke", False)):
                marker = restart_parameter_end_offset(checkpoint)
                expected_marker = cfg.get("initial_checkpoint_parameter_end_offset")
                if marker < 0:
                    errors.append("initial_checkpoint has no <par_end> within the 1 MiB reader limit")
                elif not isinstance(expected_marker, int) or marker != expected_marker:
                    errors.append("initial_checkpoint <par_end> offset differs from the recorded value")
        else:
            errors.append("initial_checkpoint does not exist")
        if full:
            if not bool(cfg.get("production_restart_smoke", False)):
                errors.append("full restart production requires production_restart_smoke=true")
            if not bundle_rel or not bundle_sha:
                errors.append("full restart production requires an explicitly hashed source bundle")
    elif any(key in cfg for key in (
            "initial_checkpoint_sha256", "initial_checkpoint_bytes",
            "initial_cycle", "initial_time")):
        errors.append("initial restart metadata requires initial_checkpoint")
    accounting_root = pathlib.Path(str(
        cfg.get("storage_accounting_root", cfg["scratch_root"])
    ))
    if not accounting_root.is_absolute():
        errors.append("storage_accounting_root must be an absolute path")
    else:
        campaign_root = pathlib.Path(str(cfg["scratch_root"])) / str(cfg["campaign_id"])
        try:
            campaign_root.relative_to(accounting_root)
        except ValueError:
            errors.append("storage_accounting_root must contain the campaign run root")
    warn = int(cfg["storage_warn_bytes"])
    stop = int(cfg["storage_stop_bytes"])
    reserve = int(cfg["checkpoint_reserve_bytes"])
    if not (0 < warn < stop and reserve > 0 and warn + reserve <= stop):
        errors.append("storage limits require 0 < warn and warn+reserve <= stop")
    if abs(float(cfg["target_time"]) - 64.0) > 1.0e-12:
        errors.append("canonical Session 007 target_time must remain 64.0")
    if full and not bool(cfg.get("production_authorized", False)):
        errors.append("production_authorized is false; use --prepare-only or authorize explicitly")
    return errors


def control_dir(cfg: Dict[str, Any]) -> pathlib.Path:
    return pathlib.Path(cfg["work_root"]) / "control"


def state_path(cfg: Dict[str, Any]) -> pathlib.Path:
    return control_dir(cfg) / "state.json"


def load_or_initialize_state(cfg: Dict[str, Any]) -> Dict[str, Any]:
    spath = state_path(cfg)
    if spath.exists():
        state = load_json(spath)
        if state.get("campaign_id") != cfg["campaign_id"]:
            raise RuntimeError("state campaign_id mismatch")
        return state
    initial_checkpoint = cfg.get("initial_checkpoint") or None
    state = {
        "schema_version": 1,
        "campaign_id": cfg["campaign_id"],
        "created_utc": utcnow(),
        "updated_utc": utcnow(),
        "status": "configured",
        "stop_requested": False,
        "emergency_cancel": False,
        "jobs": [],
        "pending_submission": None,
        "segments_completed": 0,
        "infrastructure_retries": {},
        "last_time": float(cfg.get("initial_time", 0.0)),
        "last_cycle": int(cfg.get("initial_cycle", 0)),
        "latest_checkpoint": initial_checkpoint,
        "history": [],
    }
    if initial_checkpoint:
        state["history"].append({
            "event": "initial_restart_configured",
            "checkpoint": initial_checkpoint,
            "checkpoint_sha256": cfg["initial_checkpoint_sha256"],
            "cycle": state["last_cycle"],
            "time": state["last_time"],
            "utc": utcnow(),
        })
    atomic_json(spath, state)
    return state


def scheduler(cfg: Dict[str, Any], command: str, args: Iterable[str],
              check: bool = True) -> subprocess.CompletedProcess[str]:
    bindir = cfg.get("scheduler_bin_dir", "")
    exe = str(pathlib.Path(bindir) / command) if bindir else command
    return subprocess.run(
        [exe, *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=check,
    )


def existing_job(cfg: Dict[str, Any], name: str) -> Optional[str]:
    proc = scheduler(
        cfg, "squeue",
        ["--noheader", "--name", name, "--user", os.environ.get("USER", ""),
         "--format=%i"],
        check=False,
    )
    ids = sorted({
        line.strip().split(";", 1)[0]
        for line in proc.stdout.splitlines()
        if line.strip().split(";", 1)[0].isdigit()
    })
    if not ids:
        # Recover the narrow crash window after sbatch accepted a job but before its ID
        # reached state.json.  -X omits steps; the campaign-prefixed name is unique.
        proc = scheduler(
            cfg, "sacct",
            ["-n", "-X", "--name", name, "-o", "JobIDRaw"], check=False,
        )
        ids = sorted({x.strip() for x in proc.stdout.splitlines()
                      if x.strip().isdigit()})
    if len(ids) > 1:
        raise RuntimeError(f"ambiguous recovery: multiple jobs named {name}: {ids}")
    return ids[0] if ids else None


def save_state(cfg: Dict[str, Any], state: Dict[str, Any]) -> None:
    state["updated_utc"] = utcnow()
    atomic_json(state_path(cfg), state)


def submit_one(cfg: Dict[str, Any], state: Dict[str, Any], *, kind: str,
               sequence: int, script: pathlib.Path, partition: str, nodes: int,
               wall: str, dependency: Optional[str], extra: List[str]) -> str:
    name = f"{cfg['campaign_id']}-{kind}-{sequence}"
    prior = [j for j in state["jobs"] if j["name"] == name]
    if prior:
        return str(prior[0]["job_id"])

    recovered = existing_job(cfg, name)
    state["pending_submission"] = {
        "name": name, "kind": kind, "sequence": sequence, "prepared_utc": utcnow()
    }
    save_state(cfg, state)
    submission_evidence: Dict[str, Any] = {}
    if recovered is None:
        logdir = control_dir(cfg) / "logs"
        logdir.mkdir(parents=True, exist_ok=True)
        args = [
            "--parsable", "--kill-on-invalid-dep=yes", "-A", str(cfg["account"]),
            "-p", partition, "-N", str(nodes), "-n", str(nodes), "-t", wall,
            "-J", name, "-o", str(logdir / f"{name}.%j.log"),
        ]
        if dependency:
            args.extend(["--dependency", dependency])
        args.extend([str(script), *extra])
        attempts = control_dir(cfg) / "submission_attempts"
        attempts.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        attempt = attempts / f"{stamp}-{os.getpid()}-{kind}-{sequence}"
        attempt.mkdir()
        proc = scheduler(cfg, "sbatch", args, check=False)
        stdout_path = attempt / "sbatch.stdout"
        stderr_path = attempt / "sbatch.stderr"
        status_path = attempt / "sbatch.status"
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        status_path.write_text(f"{proc.returncode}\n", encoding="utf-8")
        ids = sorted({
            line.strip().split(";", 1)[0]
            for line in proc.stdout.splitlines()
            if line.strip().split(";", 1)[0].isdigit()
        })
        submission_evidence = {
            "sbatch_stdout": str(stdout_path),
            "sbatch_stderr": str(stderr_path),
            "sbatch_status": str(status_path),
        }
        if proc.returncode == 0 and len(ids) == 1:
            recovered = ids[0]
        else:
            recovered = existing_job(cfg, name)
            if recovered is None:
                raise RuntimeError(
                    f"sbatch failed or returned an ambiguous ID for {name}; "
                    f"see {stdout_path}, {stderr_path}, and {status_path}"
                )
    state["jobs"].append({
        "name": name, "job_id": recovered, "kind": kind, "sequence": sequence,
        "dependency": dependency, "script": str(script), "submitted_utc": utcnow(),
        **submission_evidence,
    })
    state["pending_submission"] = None
    state["history"].append({"event": "job_recorded", "name": name,
                             "job_id": recovered, "utc": utcnow()})
    save_state(cfg, state)
    return recovered


def exposure(cfg: Dict[str, Any]) -> Dict[str, Any]:
    segment = cfg["production_nodes"] * wall_hours(cfg["segment_wall"])
    return {
        "production_nodes": cfg["production_nodes"],
        "maximum_segments": cfg["max_segments"],
        "segment_node_hours": segment,
        "maximum_node_hours": segment * cfg["max_segments"],
        "configured_node_hour_cap": cfg["node_hour_cap"],
        "storage_warning_bytes": cfg["storage_warn_bytes"],
        "storage_stop_bytes": cfg["storage_stop_bytes"],
        "physical_target_time": cfg["target_time"],
        "automatic_numerical_retries": 0,
        "maximum_infrastructure_retries_per_segment":
            cfg["max_infrastructure_retries_per_segment"],
        "maximum_preparation_node_hours":
            wall_hours(str(cfg["verify_wall"])) + wall_hours(str(cfg["build_wall"]))
            + cfg["startup_nodes"] * wall_hours(str(cfg["quicktest_wall"])),
    }


def cmd_dry_run(cfg: Dict[str, Any], full: bool) -> int:
    errors = validate_config(cfg, full=full)
    result = {"valid": not errors, "errors": errors, "exposure": exposure(cfg)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 2


def cmd_submit(cfg: Dict[str, Any], config_path: pathlib.Path, prepare_only: bool,
               checks_only: bool) -> int:
    errors = validate_config(cfg, full=not prepare_only and not checks_only)
    if errors:
        raise RuntimeError("; ".join(errors))
    if prepare_only:
        ctl = control_dir(cfg)
        with state_lock(ctl):
            state = load_or_initialize_state(cfg)
            if state.get("jobs"):
                raise RuntimeError("prepare-only refuses an already submitted campaign")
            state["status"] = "prepared_not_submitted"
            state["history"].append({"event": "prepared_not_submitted", "utc": utcnow()})
            save_state(cfg, state)
        command = f"python3 {pathlib.Path(__file__).resolve()} --config {config_path} submit"
        print(json.dumps({
            "campaign_id": cfg["campaign_id"], "status": "NOT SUBMITTED",
            "maximum_exposure": exposure(cfg), "one_remaining_command": command,
            "state": str(state_path(cfg)),
        }, indent=2, sort_keys=True))
        return 0
    scripts = pathlib.Path(__file__).resolve().parent
    ctl = control_dir(cfg)
    with state_lock(ctl):
        state = load_or_initialize_state(cfg)
        if state["status"] in TERMINAL:
            raise RuntimeError(f"campaign already terminal: {state['status']}")
        verify = submit_one(
            cfg, state, kind="verify", sequence=0,
            script=scripts / "vista_verify.sbatch", partition=str(cfg["dev_gpu_partition"]),
            nodes=1, wall=str(cfg["verify_wall"]), dependency=None,
            extra=[str(config_path)],
        )
        build = submit_one(
            cfg, state, kind="build", sequence=0,
            script=scripts / "vista_build.sbatch", partition=str(cfg["dev_gpu_partition"]),
            nodes=1, wall=str(cfg["build_wall"]), dependency=f"afterok:{verify}",
            extra=[str(config_path)],
        )
        quick = submit_one(
            cfg, state, kind="quicktest", sequence=0,
            script=scripts / "vista_quicktest.sbatch",
            partition=str(cfg["dev_gpu_partition"]),
            nodes=int(cfg["startup_nodes"]), wall=str(cfg["quicktest_wall"]),
            dependency=f"afterok:{build}", extra=[str(config_path)],
        )
        state["status"] = "preparation_queued"
        if not checks_only:
            predecessor = quick
            for seg in range(1, int(cfg["max_segments"]) + 1):
                sjob = submit_one(
                    cfg, state, kind="segment", sequence=seg,
                    script=scripts / "vista_segment.sbatch",
                    partition=str(cfg["gpu_partition"]),
                    nodes=int(cfg["production_nodes"]), wall=str(cfg["segment_wall"]),
                    dependency=f"afterok:{predecessor}",
                    extra=[str(config_path), str(seg)],
                )
                ijob = submit_one(
                    cfg, state, kind="inspect", sequence=seg,
                    script=scripts / "vista_inspect.sbatch",
                    partition=str(cfg["cpu_partition"]), nodes=1,
                    wall=str(cfg["inspect_wall"]), dependency=f"afterany:{sjob}",
                    extra=[str(config_path), str(seg), sjob],
                )
                predecessor = ijob
            state["status"] = "production_chain_queued"
        save_state(cfg, state)
        print(json.dumps({
            "campaign_id": cfg["campaign_id"],
            "checks_only": checks_only,
            "status": state["status"],
            "jobs": state["jobs"],
            "state": str(state_path(cfg)),
        }, indent=2, sort_keys=True))
    return 0


def read_result(cfg: Dict[str, Any], segment: int) -> Optional[Dict[str, Any]]:
    segment_root = pathlib.Path(cfg["scratch_root"]) / cfg["campaign_id"] / (
        "segments") / f"segment_{segment:03d}"
    canonical = segment_root / "result.json"
    if canonical.exists():
        return load_json(canonical)
    legacy = segment_root / "out" / "result.json"
    return load_json(legacy) if legacy.exists() else None


def infrastructure_retry_key(state: Dict[str, Any]) -> str:
    checkpoint = state.get("latest_checkpoint") or "initial_configuration"
    return f"{checkpoint}|{state.get('last_cycle', 0)}|{state.get('last_time', 0.0)}"


def decision(cfg: Dict[str, Any], state: Dict[str, Any],
             result: Optional[Dict[str, Any]], scheduler_state: str,
             stop_requested: bool = False, segment: Optional[int] = None
             ) -> Tuple[str, str]:
    if state.get("emergency_cancel"):
        return "emergency_cancel", "sticky emergency cancellation"
    if stop_requested or state.get("stop_requested"):
        if result and result.get("checkpoint_complete"):
            return "user_stop", "sticky graceful stop with verified checkpoint"
        return "user_stop", "sticky graceful stop; last verified checkpoint retained"
    if result is None:
        if scheduler_state in INFRA_STATES:
            retry_key = infrastructure_retry_key(state)
            used = int(state.get("infrastructure_retries", {}).get(retry_key, 0))
            if used < int(cfg["max_infrastructure_retries_per_segment"]):
                return "continue", f"recognized infrastructure state {scheduler_state}"
            return "infrastructure_failure", "infrastructure retry limit exhausted"
        return "configuration_test_failure", "missing segment result for non-infrastructure exit"
    if not isinstance(result.get("time"), (int, float)) or not isinstance(
            result.get("cycle"), int):
        return "diagnostic_stale", "malformed progress telemetry"
    reasons = set(result.get("stop_reasons", []))
    if "malformed_telemetry" in reasons or "diagnostic_stale" in reasons:
        return "diagnostic_stale", "essential progress telemetry was malformed or stale"
    if "resource_limit_storage" in reasons:
        return "resource_limit", "monitor reached the declared storage stop limit"
    if "numerical_failure" in reasons:
        return "numerical_failure", "monitor observed a fatal numerical/pathology pattern"
    if int(result.get("exit_code", 1)) != 0 or bool(result.get("fatal_pattern", False)):
        return "numerical_failure", "application returned nonzero or emitted a fatal pattern"
    if not result.get("checkpoint_complete", False):
        return "configuration_test_failure", "successful process exit without verified checkpoint"
    if float(result["time"]) >= float(cfg["target_time"]) - 1.0e-12:
        return "complete_physical_endpoint", "finite physical-time endpoint reached"
    if bool(cfg.get("authorized_early_stop", False)) and bool(
            result.get("authorized_common_horizon_stop", False)):
        return "authorized_scientific_stop", (
            "launch-authorized accepted common horizon enclosing both trackers"
        )
    if float(result["time"]) <= float(state.get("last_time", -1.0)) or int(
            result["cycle"]) <= int(state.get("last_cycle", -1)):
        return "no_progress", "time and cycle did not strictly increase"
    if int(result.get("storage_bytes", 0)) >= int(cfg["storage_stop_bytes"]):
        return "resource_limit", "storage stop limit reached"
    next_exposure = (
        (int(state.get("segments_completed", 0)) + 1)
        * int(cfg["production_nodes"]) * wall_hours(str(cfg["segment_wall"]))
    )
    if next_exposure > float(cfg["node_hour_cap"]) + 1.0e-12:
        return "resource_limit", "next segment would exceed node-hour cap"
    if int(state.get("segments_completed", 0)) + 1 >= int(cfg["max_segments"]):
        return "segment_limit", "finite segment limit reached before physical endpoint"
    return "continue", "verified checkpoint and strictly increasing progress"


def scheduler_state(cfg: Dict[str, Any], job_id: str) -> str:
    proc = scheduler(
        cfg, "sacct", ["-n", "-X", "-j", job_id, "-o", "State"],
        check=False,
    )
    token = proc.stdout.strip().splitlines()
    return token[0].split()[0].split("+")[0] if token else "UNKNOWN"


def cancel_future(cfg: Dict[str, Any], state: Dict[str, Any], segment: int) -> None:
    ids = [
        str(j["job_id"]) for j in state["jobs"]
        if int(j.get("sequence", 0)) > segment
    ]
    if ids:
        scheduler(cfg, "scancel", ids, check=False)


def cmd_inspect(cfg: Dict[str, Any], segment: int, parent: str) -> int:
    ctl = control_dir(cfg)
    with state_lock(ctl):
        state = load_or_initialize_state(cfg)
        result = read_result(cfg, segment)
        sched = scheduler_state(cfg, parent)
        stop_marker = ctl / "STOP_REQUESTED"
        action, reason = decision(
            cfg, state, result, sched, stop_marker.exists(), segment=segment
        )
        if result and result.get("checkpoint_complete"):
            state["latest_checkpoint"] = result.get("checkpoint")
            state["last_time"] = float(result["time"])
            state["last_cycle"] = int(result["cycle"])
        if action == "continue":
            if result:
                state["segments_completed"] = max(
                    int(state["segments_completed"]), segment
                )
            elif sched in INFRA_STATES:
                retries = state.setdefault("infrastructure_retries", {})
                key = infrastructure_retry_key(state)
                retries[key] = int(retries.get(key, 0)) + 1
            state["status"] = "continuation_released"
        else:
            state["status"] = action
            cancel_future(cfg, state, segment)
        state["history"].append({
            "event": "segment_inspected", "segment": segment, "parent_job": parent,
            "scheduler_state": sched, "decision": action, "reason": reason,
            "utc": utcnow(),
        })
        save_state(cfg, state)
        atomic_json(ctl / "status.json", {
            "campaign_id": cfg["campaign_id"], "status": state["status"],
            "reason": reason, "segment": segment, "latest_checkpoint":
            state.get("latest_checkpoint"), "time": state.get("last_time"),
            "cycle": state.get("last_cycle"), "updated_utc": utcnow(),
        })
        print(json.dumps({"decision": action, "reason": reason}, sort_keys=True))
        return 0 if action == "continue" else 42


def recorded_active_ids(cfg: Dict[str, Any], state: Dict[str, Any]) -> List[str]:
    return [str(j["job_id"]) for j in state.get("jobs", [])]


def cmd_stop(cfg: Dict[str, Any], emergency: bool) -> int:
    ctl = control_dir(cfg)
    with state_lock(ctl):
        state = load_or_initialize_state(cfg)
        state["stop_requested"] = True
        marker = ctl / ("EMERGENCY_CANCEL" if emergency else "STOP_REQUESTED")
        marker.write_text(utcnow() + "\n", encoding="utf-8")
        if emergency:
            state["emergency_cancel"] = True
            state["status"] = "emergency_cancel"
            ids = recorded_active_ids(cfg, state)
            if ids:
                scheduler(cfg, "scancel", ids, check=False)
        else:
            state["status"] = "user_stop_requested"
        state["history"].append({
            "event": "emergency_cancel" if emergency else "graceful_stop_requested",
            "utc": utcnow(),
        })
        save_state(cfg, state)
    print(f"{'EMERGENCY CANCEL' if emergency else 'GRACEFUL STOP'} recorded for "
          f"{cfg['campaign_id']}; state={state_path(cfg)}")
    return 0


def cmd_status(cfg: Dict[str, Any]) -> int:
    state = load_or_initialize_state(cfg)
    ids = recorded_active_ids(cfg, state)
    queue = ""
    if ids:
        proc = scheduler(
            cfg, "squeue", ["--jobs", ",".join(ids),
                            "--format=%i|%T|%M|%R|%j"], check=False
        )
        queue = proc.stdout
    print(json.dumps(state, indent=2, sort_keys=True))
    if queue:
        print("\nRECORDED JOBS\n" + queue)
    return 0


def cmd_config_get(cfg: Dict[str, Any], key: str) -> int:
    value: Any = cfg
    for part in key.split("."):
        value = value[part]
    if isinstance(value, (dict, list)):
        print(json.dumps(value, separators=(",", ":")))
    else:
        print(value)
    return 0


def binding(cfg: Dict[str, Any]) -> Dict[str, str]:
    encoded = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    result = {
        "source_commit": str(cfg["source_commit"]),
        "kokkos_commit": str(cfg["kokkos_commit"]),
        "input_sha256": str(cfg["input_sha256"]),
        "control_input_sha256": str(cfg["control_input_sha256"]),
        "expanded_configuration_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    if cfg.get("source_bundle_sha256"):
        result["source_bundle_sha256"] = str(cfg["source_bundle_sha256"])
    return result


def cmd_mark(cfg: Dict[str, Any], stage: str, status: str,
             details: Optional[pathlib.Path]) -> int:
    ctl = control_dir(cfg)
    payload: Dict[str, Any] = {
        "stage": stage, "status": status, "utc": utcnow(), "binding": binding(cfg)
    }
    if details:
        payload["details"] = str(details)
        payload["details_sha256"] = sha256(details)
    atomic_json(ctl / f"{stage}_receipt.json", payload)
    return 0


def cmd_gate_check(cfg: Dict[str, Any], stages: List[str]) -> int:
    ctl = control_dir(cfg)
    expected = binding(cfg)
    for stage in stages:
        receipt = load_json(ctl / f"{stage}_receipt.json")
        if receipt.get("status") != "pass" or receipt.get("binding") != expected:
            raise RuntimeError(f"{stage} receipt is absent, failed, or bound to other inputs")
        details = receipt.get("details")
        if details and sha256(pathlib.Path(details)) != receipt.get("details_sha256"):
            raise RuntimeError(f"{stage} details checksum changed after receipt")
    if "build" in stages:
        checksum_file = ctl / "executable.sha256"
        fields = checksum_file.read_text(encoding="utf-8").split()
        exe = pathlib.Path(cfg["work_root"]) / "build_cuda_gi" / "src" / "athena"
        if len(fields) < 1 or not exe.is_file() or sha256(exe) != fields[0]:
            raise RuntimeError("built executable does not match its recorded checksum")
    for label, relkey, hashkey in (
            ("canonical", "input_relpath", "input_sha256"),
            ("control", "control_input_relpath", "control_input_sha256")):
        deck = pathlib.Path(cfg["handoff_dir"]) / cfg[relkey]
        if not deck.is_file() or sha256(deck) != cfg[hashkey]:
            raise RuntimeError(f"{label} input no longer matches configured checksum")
    print(json.dumps({"gate": "pass", "stages": stages, "binding": expected},
                     sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=pathlib.Path)
    sub = parser.add_subparsers(dest="command", required=True)
    dry = sub.add_parser("dry-run")
    dry.add_argument("--full", action="store_true")
    submit = sub.add_parser("submit")
    modes = submit.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true",
                       help="validate and leave one command; submit no jobs")
    modes.add_argument("--checks-only", action="store_true",
                       help="submit verify/build/quicktest but not production")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("segment", type=int)
    inspect.add_argument("parent_job")
    sub.add_parser("status")
    sub.add_parser("graceful-stop")
    sub.add_parser("emergency-cancel")
    get = sub.add_parser("config-get")
    get.add_argument("key")
    mark = sub.add_parser("mark")
    mark.add_argument("stage")
    mark.add_argument("status", choices=("pass", "fail"))
    mark.add_argument("--details", type=pathlib.Path)
    gate = sub.add_parser("gate-check")
    gate.add_argument("stages", nargs="+", choices=("verify", "build", "quicktest"))

    args = parser.parse_args()
    cfg = expand(load_json(args.config.resolve()))
    try:
        if args.command == "dry-run":
            return cmd_dry_run(cfg, args.full)
        if args.command == "submit":
            return cmd_submit(
                cfg, args.config.resolve(), args.prepare_only, args.checks_only
            )
        if args.command == "inspect":
            return cmd_inspect(cfg, args.segment, args.parent_job)
        if args.command == "status":
            return cmd_status(cfg)
        if args.command == "graceful-stop":
            return cmd_stop(cfg, False)
        if args.command == "emergency-cancel":
            return cmd_stop(cfg, True)
        if args.command == "config-get":
            return cmd_config_get(cfg, args.key)
        if args.command == "mark":
            return cmd_mark(cfg, args.stage, args.status, args.details)
        if args.command == "gate-check":
            return cmd_gate_check(cfg, args.stages)
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
