#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "submit_probe_from_login_20260908T1959Z.sh"
RETRY_WRAPPER = ROOT / "submit_probe_retry_r2_from_login_20260908T2300Z.sh"
TRANSPORT_RETRY_WRAPPER = (
    ROOT / "submit_probe_transport_retry_r3_from_login_20260909T0020Z.sh"
)
RESUME_WRAPPER = ROOT / "submit_probe_transport_retry_r4_from_login_20260909T0020Z.sh"
GHDEV_RETRY_WRAPPER = (
    ROOT / "submit_probe_ghdev_retry_r5_from_login_20260909T0031Z.sh"
)
FRESH_GATE_WRAPPER = (
    ROOT / "submit_probe_ghdev_retry_r6_from_login_20260909T0538Z.sh"
)
EXACT_VALIDATION_WRAPPER = (
    ROOT / "submit_exact_validation_from_login_20260909T0552Z.sh"
)
BASE_CONFIG = ROOT / "workflow_probe_config_20260908T1959Z.json"


FAKE_SCHEDULER = r'''#!/usr/bin/env python3
import json
import os
import pathlib
import sys

state_path = pathlib.Path(os.environ["FAKE_SCHED_STATE"])
if state_path.exists():
    state = json.loads(state_path.read_text())
else:
    state = {"next_id": 1001, "jobs": {}, "calls": [], "failed_build_once": False}

command = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
state["calls"].append({"command": command, "args": args})

def separate(flag):
    try:
        return args[args.index(flag) + 1]
    except (ValueError, IndexError):
        return None

def save():
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

if command == "squeue":
    name = separate("--name")
    for job_id, job in state["jobs"].items():
        if job["name"] == name and job["state"] in {
            "PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"
        }:
            print(job_id)
    save()
    raise SystemExit(0)

if command == "sacct":
    job_id = separate("-j")
    if job_id is not None:
        job = state["jobs"].get(job_id)
        if job:
            print(f'{job["state"]}|{job["exit_code"]}')
    else:
        name = separate("--name")
        for existing_id, job in state["jobs"].items():
            if job["name"] == name:
                print(existing_id)
    save()
    raise SystemExit(0)

if command != "sbatch":
    save()
    raise SystemExit(90)

name = separate("-J")
mode = os.environ.get("FAKE_SCHED_MODE", "success")
if mode == "partial" and name and name.endswith("-build-r1") and not state["failed_build_once"]:
    state["failed_build_once"] = True
    save()
    print("sbatch: error: injected partial failure", file=sys.stderr)
    raise SystemExit(1)

if mode == "ambiguous" and name and name.endswith("-verify-r1"):
    save()
    print("1001")
    print("1002;vista")
    raise SystemExit(0)

job_id = str(state["next_id"])
state["next_id"] += 1
state["jobs"][job_id] = {
    "name": name,
    "state": "PENDING",
    "exit_code": "0:0",
}
save()
if mode == "unexpected":
    print("NOTIFICATION: harmless site-wrapper message")
print(f"{job_id};vista")
'''


class ProbeSubmissionWrapper(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        fake = self.bin / "fake_scheduler.py"
        fake.write_text(FAKE_SCHEDULER)
        fake.chmod(0o700)
        for name in ("sbatch", "squeue", "sacct"):
            (self.bin / name).symlink_to(fake)

        self.state_path = self.root / "scheduler.json"
        self.config_path = self.root / "probe.json"
        self.work = self.root / "work"
        self.scratch = self.root / "scratch"
        config = json.loads(BASE_CONFIG.read_text())
        config.update(
            campaign_id="probe-wrapper-test",
            account="AST21006",
            handoff_dir=str(ROOT),
            work_root=str(self.work),
            scratch_root=str(self.scratch),
            production_authorized=False,
        )
        self.config_path.write_text(json.dumps(config, indent=2) + "\n")

    def tearDown(self):
        self.temp.cleanup()

    def run_wrapper(
        self, mode: str, wrapper: pathlib.Path = WRAPPER
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(
            USER="probe-test-user",
            S7_PROBE_TEST_MODE="1",
            S7_PROBE_CONFIG=str(self.config_path),
            S7_SBATCH_BIN=str(self.bin / "sbatch"),
            S7_SQUEUE_BIN=str(self.bin / "squeue"),
            S7_SACCT_BIN=str(self.bin / "sacct"),
            FAKE_SCHED_STATE=str(self.state_path),
            FAKE_SCHED_MODE=mode,
        )
        return subprocess.run(
            ["/bin/bash", str(wrapper)], text=True, capture_output=True, env=env
        )

    def scheduler_state(self):
        return json.loads(self.state_path.read_text())

    def records(self):
        root = self.work / "control" / "probe_submission_state"
        return {path.stem: path.read_text() for path in root.glob("*.record")}

    def test_notice_is_preserved_but_only_numeric_id_enters_dependencies(self):
        result = self.run_wrapper("unexpected")
        self.assertEqual(result.returncode, 0, result.stderr)
        records = self.records()
        self.assertIn("job_id=1001\n", records["verify"])
        self.assertIn("dependency=afterok:1001\n", records["build"])
        self.assertIn("job_id=1002\n", records["build"])
        self.assertIn("dependency=afterok:1002\n", records["probe"])

        attempts = self.work / "control" / "probe_submission_attempts"
        verify_stdout = next(attempts.rglob("verify.sbatch.stdout")).read_text()
        verify_stderr = next(attempts.rglob("verify.sbatch.stderr")).read_text()
        verify_status = next(attempts.rglob("verify.sbatch.status")).read_text()
        self.assertIn("NOTIFICATION:", verify_stdout)
        self.assertEqual(verify_stderr, "")
        self.assertEqual(verify_status, "0\n")

    def test_partial_failure_records_verify_and_retry_does_not_duplicate_it(self):
        first = self.run_wrapper("partial")
        self.assertEqual(first.returncode, 2)
        self.assertIn("sbatch failed for build", first.stderr)
        records = self.records()
        self.assertIn("verify", records)
        self.assertNotIn("build", records)

        second = self.run_wrapper("success")
        self.assertEqual(second.returncode, 0, second.stderr)
        state = self.scheduler_state()
        accepted_names = [job["name"] for job in state["jobs"].values()]
        self.assertEqual(accepted_names.count("probe-wrapper-test-verify-r1"), 1)
        self.assertEqual(accepted_names.count("probe-wrapper-test-build-r1"), 1)
        self.assertEqual(accepted_names.count("probe-wrapper-test-capacity-r1"), 1)

        third = self.run_wrapper("success")
        self.assertEqual(third.returncode, 0, third.stderr)
        final = self.scheduler_state()
        accepted_names = [job["name"] for job in final["jobs"].values()]
        self.assertEqual(len(accepted_names), 3)

    def test_ambiguous_output_stops_without_dependency_then_retries_safely(self):
        first = self.run_wrapper("ambiguous")
        self.assertEqual(first.returncode, 2)
        self.assertIn("IDs: 1001 1002", first.stderr)
        self.assertEqual(self.scheduler_state()["jobs"], {})
        self.assertEqual(self.records(), {})

        second = self.run_wrapper("success")
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(len(self.scheduler_state()["jobs"]), 3)

    def test_revisioned_retry_chain_is_idempotent_and_keeps_r1_jobs(self):
        first = self.run_wrapper("success")
        self.assertEqual(first.returncode, 0, first.stderr)

        retry = self.run_wrapper("unexpected", RETRY_WRAPPER)
        self.assertEqual(retry.returncode, 0, retry.stderr)
        state = self.scheduler_state()
        names = [job["name"] for job in state["jobs"].values()]
        for stage in ("verify", "build", "capacity"):
            self.assertEqual(names.count(f"probe-wrapper-test-{stage}-r1"), 1)
            self.assertEqual(names.count(f"probe-wrapper-test-{stage}-r2"), 1)

        repeated = self.run_wrapper("success", RETRY_WRAPPER)
        self.assertEqual(repeated.returncode, 0, repeated.stderr)
        self.assertEqual(len(self.scheduler_state()["jobs"]), 6)

        retry_state = self.work / "control" / "probe_submission_state_r2"
        self.assertEqual(len(list(retry_state.glob("*.record"))), 3)
        self.assertTrue((self.work / "control" / "probe_submission_r2.txt").is_file())

    def test_transport_retry_is_revisioned_and_idempotent(self):
        first = self.run_wrapper("unexpected", TRANSPORT_RETRY_WRAPPER)
        self.assertEqual(first.returncode, 0, first.stderr)
        names = [job["name"] for job in self.scheduler_state()["jobs"].values()]
        for stage in ("verify", "build", "capacity"):
            self.assertEqual(names.count(f"probe-wrapper-test-{stage}-r3"), 1)

        repeated = self.run_wrapper("success", TRANSPORT_RETRY_WRAPPER)
        self.assertEqual(repeated.returncode, 0, repeated.stderr)
        self.assertEqual(len(self.scheduler_state()["jobs"]), 3)

    def test_existing_campaign_directory_does_not_block_unique_probe(self):
        old_probe = self.scratch / "probe-wrapper-test" / "resource_probe_older"
        old_probe.mkdir(parents=True)
        marker = old_probe / "preserved.txt"
        marker.write_text("old probe evidence\n")

        result = self.run_wrapper("success", RESUME_WRAPPER)
        self.assertEqual(result.returncode, 0, result.stderr)
        names = [job["name"] for job in self.scheduler_state()["jobs"].values()]
        self.assertIn("probe-wrapper-test-capacity-r4", names)
        self.assertEqual(marker.read_text(), "old probe evidence\n")

    def test_ghdev_retry_reuses_r4_gates_and_is_idempotent(self):
        first = self.run_wrapper("success", RESUME_WRAPPER)
        self.assertEqual(first.returncode, 0, first.stderr)
        state = self.scheduler_state()
        state["jobs"]["1001"].update(state="COMPLETED", exit_code="0:0")
        state["jobs"]["1002"].update(state="COMPLETED", exit_code="0:0")
        state["jobs"]["1003"].update(state="CANCELLED", exit_code="0:0")
        self.state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

        retry = self.run_wrapper("success", GHDEV_RETRY_WRAPPER)
        self.assertEqual(retry.returncode, 0, retry.stderr)
        state = self.scheduler_state()
        names = [job["name"] for job in state["jobs"].values()]
        self.assertNotIn("probe-wrapper-test-verify-r5", names)
        self.assertNotIn("probe-wrapper-test-build-r5", names)
        self.assertEqual(names.count("probe-wrapper-test-capacity-r5"), 1)
        probe_call = next(
            call
            for call in state["calls"]
            if call["command"] == "sbatch"
            and "probe-wrapper-test-capacity-r5" in call["args"]
        )
        self.assertEqual(probe_call["args"][probe_call["args"].index("-p") + 1], "gh-dev")
        self.assertIn("--dependency=afterok:1002", probe_call["args"])

        repeated = self.run_wrapper("success", GHDEV_RETRY_WRAPPER)
        self.assertEqual(repeated.returncode, 0, repeated.stderr)
        names = [job["name"] for job in self.scheduler_state()["jobs"].values()]
        self.assertEqual(names.count("probe-wrapper-test-capacity-r5"), 1)

    def test_fresh_gate_retry_reuses_build_and_depends_on_new_verify(self):
        first = self.run_wrapper("success", RESUME_WRAPPER)
        self.assertEqual(first.returncode, 0, first.stderr)
        state = self.scheduler_state()
        state["jobs"]["1001"].update(state="COMPLETED", exit_code="0:0")
        state["jobs"]["1002"].update(state="COMPLETED", exit_code="0:0")
        state["jobs"]["1003"].update(state="CANCELLED", exit_code="0:0")
        self.state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

        retry = self.run_wrapper("success", FRESH_GATE_WRAPPER)
        self.assertEqual(retry.returncode, 0, retry.stderr)
        state = self.scheduler_state()
        names = [job["name"] for job in state["jobs"].values()]
        self.assertEqual(names.count("probe-wrapper-test-verify-r6"), 1)
        self.assertNotIn("probe-wrapper-test-build-r6", names)
        self.assertEqual(names.count("probe-wrapper-test-capacity-r6"), 1)
        probe_call = next(
            call
            for call in state["calls"]
            if call["command"] == "sbatch"
            and "probe-wrapper-test-capacity-r6" in call["args"]
        )
        self.assertEqual(probe_call["args"][probe_call["args"].index("-p") + 1], "gh-dev")
        self.assertIn("--dependency=afterok:1004", probe_call["args"])

        repeated = self.run_wrapper("success", FRESH_GATE_WRAPPER)
        self.assertEqual(repeated.returncode, 0, repeated.stderr)
        names = [job["name"] for job in self.scheduler_state()["jobs"].values()]
        self.assertEqual(names.count("probe-wrapper-test-verify-r6"), 1)
        self.assertEqual(names.count("probe-wrapper-test-capacity-r6"), 1)

    def test_exact_validation_uses_ghdev_and_afterok_build(self):
        result = self.run_wrapper("success", EXACT_VALIDATION_WRAPPER)
        self.assertEqual(result.returncode, 0, result.stderr)
        state = self.scheduler_state()
        names = [job["name"] for job in state["jobs"].values()]
        self.assertEqual(names.count("probe-wrapper-test-quicktest-r1"), 1)
        call = next(
            item
            for item in state["calls"]
            if item["command"] == "sbatch"
            and "probe-wrapper-test-quicktest-r1" in item["args"]
        )
        self.assertEqual(call["args"][call["args"].index("-p") + 1], "gh-dev")
        self.assertEqual(call["args"][call["args"].index("-N") + 1], "8")
        self.assertIn("--dependency=afterok:1002", call["args"])
        self.assertTrue(call["args"][-2].endswith("vista_quicktest.sbatch"))


class SlurmScriptPaths(unittest.TestCase):
    def test_batch_scripts_resolve_helpers_from_configured_handoff(self):
        for path in sorted((ROOT / "scripts").glob("vista_*.sbatch")):
            if path.name == "vista_progress_movie.sbatch":
                continue
            text = path.read_text()
            self.assertNotIn('dirname "$0"', text, path.name)
            self.assertIn('json.load(open(sys.argv[1]))["handoff_dir"]', text, path.name)

    def test_probe_account_and_authorization(self):
        config = json.loads(BASE_CONFIG.read_text())
        self.assertEqual(config["account"], "AST21006")
        self.assertFalse(config["production_authorized"])
        self.assertEqual(config["ucx_ib_gpu_direct_rdma"], "no")
        self.assertEqual(config["production_nodes"], 16)
        self.assertEqual(config["production_max_nmb_per_rank"], 400)

    def test_build_loads_pinned_kokkos_arm_sve_cmake_compatibility(self):
        build = (ROOT / "scripts" / "vista_build.sbatch").read_text()
        verify = (ROOT / "scripts" / "vista_verify.sbatch").read_text()
        shim = ROOT / "scripts" / "kokkos_4_7_2_arm_sve_compat.cmake"
        self.assertIn('CMAKE_PROJECT_INCLUDE="$CMAKE_COMPAT"', build)
        self.assertIn('verify_details.${SLURM_JOB_ID:?}.txt', verify)
        self.assertEqual(shim.read_text().splitlines()[-1], "include(CheckIncludeFileCXX)")

    def test_transport_fallback_is_consistent_across_gpu_launchers(self):
        for name in (
            "vista_resource_probe.sbatch",
            "vista_quicktest.sbatch",
            "vista_segment.sbatch",
        ):
            text = (ROOT / "scripts" / name).read_text()
            self.assertIn("UCX_IB_GPU_DIRECT_RDMA", text, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
