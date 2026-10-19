"""Meaningful controller tests. Scheduler and CUDA execution are mocked."""
import json
import contextlib
import io
import os
import re
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import s10


class SubmissionTests(unittest.TestCase):
    def test_gh_only_script_submits_exactly_three_dependent_jobs(self):
        root = Path(self.tmp.name)
        (root / 'runtime_s10').mkdir()
        s10.atomic(root / 'runtime_s10/GH_READY_RUN.json', {'run': str(self.run)})
        s10.atomic(root / 'runtime_s10/ACTIVE_RUN.json', {'run': str(self.run)})
        s10.atomic(self.run / 'INPUT_SHA256.json', {})
        s10.atomic(self.run / 'LAUNCHER_SHA256.json', {})
        calls = []
        def scheduler(args, **kw):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, f'Welcome to Vista\n{2000 + len(calls)}\n', '')
        with patch.object(s10, 'HANDOFF', root), patch.object(s10, 'build_ready', return_value=True), patch.object(s10.socket, 'gethostname', return_value='login2.vista.tacc.utexas.edu'), patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()):
            s10.submit_ready_gh(runner=scheduler)
        self.assertEqual(len(calls), 3)
        self.assertEqual([next(a for a in c if a.startswith('--time=')) for c in calls], ['--time=08:00:00', '--time=24:00:00', '--time=24:00:00'])
        self.assertTrue(all('--partition=gh' in c and '--nodes=16' in c for c in calls))
        self.assertFalse(any(a.startswith('--dependency=') for a in calls[0]))
        self.assertIn('--dependency=afterok:2001', calls[1])
        self.assertIn('--dependency=afterok:2002', calls[2])

    def test_gh_only_specs_never_contain_debug_or_build(self):
        specs = s10.gh_submission_specs(s10.config())
        self.assertEqual(specs, [('production','gh',16,8,1), ('production','gh',16,24,2), ('production','gh',16,24,3)])

    def test_tacc_banner_before_parsable_id_is_accepted(self):
        text = '\nWelcome to Vista\n--> Checking allocation...OK\n1069961\n'
        self.assertEqual(s10.submission_id(text), '1069961')
        self.assertEqual(s10.submission_id('1069961;vista\n'), '1069961')
        self.assertIsNone(s10.submission_id('1069961\n1069962\n'))
        self.assertIsNone(s10.submission_id('allocation check failed\n'))

    def test_recovery_adopts_existing_debug_and_submits_only_missing_jobs(self):
        s10.atomic(self.run / 'LAUNCHER_SHA256.json', {'s10.py': s10.digest(s10.HERE / 's10.py')})
        s10.atomic(self.run / 'submission.json', dict(status='submission_incomplete_or_uncertain', jobs=[], pending=['sbatch'], created_epoch=0, last_attempt=dict(command=['sbatch'], returncode=0, stdout='Welcome to Vista\n1069961\n', stderr='')))
        calls = []
        def scheduler(args, **kw):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, f'Welcome to Vista\n{1069961 + len(calls)}\n', '')
        with patch.object(s10, 'confirm_job', return_value={'confirmed': True}):
            state = s10.submit_chain(self.run, s10.submission_specs(s10.config(), True), scheduler, resume=True)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all('--partition=gh' in c for c in calls))
        self.assertIn('--dependency=afterok:1069961', calls[0])
        self.assertEqual(state['jobs'][0]['job_id'], '1069961')
        self.assertEqual(state['jobs'][0]['stage'], 'debug')
        self.assertTrue((self.run / 'LAUNCHER_SHA256.before_recovery.json').exists())

    def test_submit_entrypoint_refuses_compute_node(self):
        with patch.object(s10.sys, "argv", ["s10.py", "submit"]), patch.object(s10.socket, "gethostname", return_value="i614-112.vista.tacc.utexas.edu"), patch.object(s10, "submit_chain") as call:
            with self.assertRaisesRegex(RuntimeError, "login node"):
                s10.main()
            call.assert_not_called()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name) / "run"
        self.run.mkdir()
        s10.atomic(self.run / "config.json", s10.config())
        self.addCleanup(self.tmp.cleanup)

    def test_finite_chain_and_notifications(self):
        calls = []
        def scheduler(args, **kw):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, f"{100 + len(calls)};vista\n", "")
        state = s10.submit_chain(self.run, s10.submission_specs(s10.config(), True), scheduler)
        self.assertEqual([j["wall_hours"] for j in state["jobs"]], [2, 8, 24, 24])
        self.assertEqual([j["nodes"] for j in state["jobs"]], [4, 16, 16, 16])
        for i, call in enumerate(calls):
            self.assertIn("--mail-user=jiaxiwu@caltech.edu", call)
            self.assertIn("--mail-type=BEGIN,END,FAIL", call)
            self.assertIn("--kill-on-invalid-dep=yes", call)
            self.assertIn("--no-requeue", call)
            if i:
                self.assertIn(f"--dependency=afterok:{100 + i}", call)
            else:
                self.assertFalse(any(x.startswith("--dependency") for x in call))

    def test_missing_native_build_adds_gg_build(self):
        specs = s10.submission_specs(s10.config(), False)
        self.assertEqual(specs[0], ("build", "gg", 1, 2, 0))
        self.assertEqual(len(specs), 5)

    def test_duplicate_submission_is_refused(self):
        runner = lambda args, **kw: subprocess.CompletedProcess(args, 0, "123\n", "")
        s10.submit_chain(self.run, [s10.submission_specs(s10.config(), True)[0]], runner)
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            s10.submit_chain(self.run, [], runner)

    def test_ambiguous_response_is_retained_without_retry(self):
        calls = []
        def runner(args, **kw):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, "unexpected output\n", "warning")
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            s10.submit_chain(self.run, s10.submission_specs(s10.config(), True), runner)
        state = s10.read(self.run / "submission.json")
        self.assertEqual(len(calls), 1)
        self.assertIsNotNone(state["pending"])
        self.assertEqual(state["last_attempt"]["stderr"], "warning")

    def test_partial_chain_submission_preserves_prior_ids(self):
        n = 0
        def runner(args, **kw):
            nonlocal n
            n += 1
            return subprocess.CompletedProcess(args, 0 if n == 1 else 1, "123\n" if n == 1 else "", "" if n == 1 else "queue error")
        with self.assertRaises(RuntimeError):
            s10.submit_chain(self.run, s10.submission_specs(s10.config(), True), runner)
        self.assertEqual(s10.read(self.run / "submission.json")["jobs"][0]["job_id"], "123")
        self.assertEqual(n, 2)


class PolicyTests(unittest.TestCase):
    def test_production_cli_overrides_require_existing_nlim(self):
        original = (s10.HANDOFF / "inputs/gi_cluster_s10.athinput").read_text()
        fixed = s10.production_deck(original)
        for nlim in (-1, 1, 2):
            s10.validate_overrides(fixed, s10.production_overrides(nlim=nlim))
        broken = re.sub(r"^nlim = -1\n", "", fixed, flags=re.MULTILINE)
        with self.assertRaisesRegex(ValueError, "time/nlim"):
            s10.validate_overrides(broken, s10.production_overrides())

    def test_resource_stop_blocks_dependent_production(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            s10.atomic(run / "DEBUG_VERIFIED.json", dict(passed=True, executable_sha256="hash"))
            s10.atomic(run / "segment_001.json", dict(status="resource_limit_gpu_memory"))
            with patch.object(s10, "verify_package"), patch.object(s10, "digest", return_value="hash"), patch.object(s10, "launch_athena") as launch:
                with self.assertRaisesRegex(RuntimeError, "predecessor"):
                    s10.production_job(run, s10.config(), 2)
                launch.assert_not_called()

    def test_compact_gate_does_not_release_long_jobs(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            s10.atomic(run / "DEBUG_VERIFIED.json", dict(passed=True, executable_sha256="hash"))
            s10.atomic(run / "segment_001.json", dict(status="verified_segment_end"))
            with patch.object(s10, "verify_package"), patch.object(s10, "digest", return_value="hash"), patch.object(s10, "launch_athena") as launch:
                with self.assertRaises(FileNotFoundError):
                    s10.production_job(run, s10.config(), 2)
                launch.assert_not_called()

    def test_sticky_stop_is_not_overwritten_by_first_job(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            s10.atomic(run / "DEBUG_VERIFIED.json", dict(passed=True, executable_sha256="hash"))
            s10.atomic(run / "state.json", dict(status="user_stop", checkpoint=None, chunks=[]))
            with patch.object(s10, "verify_package"), patch.object(s10, "digest", return_value="hash"), patch.object(s10, "launch_athena") as launch:
                with self.assertRaisesRegex(RuntimeError, "sticky stop"):
                    s10.production_job(run, s10.config(), 1)
                launch.assert_not_called()

    def test_checkpoint_must_match_application_exit(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "athena.log"
            log.write_text("Terminating on wall clock limit\ntime=9.33125e+00 cycle=2256\n")
            s10.checkpoint_matches_exit(dict(time=9.33125, cycle=2256), log)
            with self.assertRaises(ValueError):
                s10.checkpoint_matches_exit(dict(time=9.0, cycle=2200), log)

    def test_missing_end_time_uses_live_slurm_metadata(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "123"}, clear=True), patch.object(s10.subprocess, "check_output", return_value="JobId=123 EndTime=2026-10-14T12:00:00"):
            self.assertGreater(s10.job_end_epoch(), 0)

    def test_failed_run_output_list_keeps_only_production_chunks(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            state = dict(status="resource_limit_gpu_memory", checkpoint=dict(time=9.33125, cycle=2256), chunks=[dict(output="production/chunk/out")])
            s10.save_state(run, state)
            result = s10.read(run / "PRODUCTION_OUTPUTS.json")
            self.assertEqual(result["ordered_run_directories"], ["production/chunk/out"])
            self.assertFalse(result["scientific_completion"])
            self.assertFalse(result["full_size_gate_passed"])

    def test_production_changes_only_reviewed_time_and_outputs(self):
        original = (s10.HANDOFF / "inputs/gi_cluster_s10.athinput").read_text()
        text = s10.production_deck(original)
        d = s10.parse_deck(text)
        self.assertEqual(d["time"]["nlim"], "-1")
        self.assertEqual(d["time"]["tlim"], "1e30")
        self.assertEqual(d["problem"]["gi_clump1_bulk_vy"], "-0.133215")
        self.assertEqual(d["mesh"]["nx1"], "256")
        self.assertEqual(d["output5"]["dt"], "2.0")
        self.assertEqual(d["output2"]["dt"], "0.5")
        self.assertEqual(d["output19"]["dt"], "25.0")
        self.assertEqual(d["z4c"]["waveform_dt"], "0.025")
        self.assertEqual(d["fastflow"]["stop_time_2"], "1e30")

    def test_reserve_grows_with_measured_finalization(self):
        cfg = s10.config()
        self.assertEqual(s10.launch_seconds(10000, cfg, {}, now=0), 7200)
        self.assertEqual(s10.launch_seconds(10000, cfg, {}, now=6000), 1600)
        state = dict(chunks=[dict(finalization_seconds=1800)])
        self.assertEqual(s10.finalization_reserve(cfg, state), 3600)
        self.assertEqual(s10.launch_seconds(10000, cfg, state, now=7000), 0)

    def test_resource_stop_promotes_real_progress(self):
        old = dict(checkpoint=dict(time=1, cycle=100), chunks=[])
        cp = dict(time=2, cycle=200, sha256="a" * 64)
        new = s10.promoted_state(old, cp, "resource_limit_gpu_memory", dict(output="out"))
        self.assertEqual(new["checkpoint"]["cycle"], 200)
        self.assertEqual(new["status"], "resource_limit_gpu_memory")
        self.assertEqual(old["checkpoint"]["cycle"], 100)

    def test_regressing_or_stalled_checkpoint_rejected(self):
        old = dict(checkpoint=dict(time=2, cycle=200))
        for cp in (dict(time=1, cycle=100), dict(time=2, cycle=200)):
            with self.assertRaises(ValueError):
                s10.promoted_state(old, cp, "running", {})

    def test_worst_device_and_host_memory_stop(self):
        cfg = s10.config()
        good = dict(host_total_bytes=100, host_used_bytes=60, gpus=[dict(used_mib=60, total_mib=100)])
        bad = dict(good, gpus=[dict(used_mib=85, total_mib=100)])
        self.assertEqual(s10.stop_reason([good, bad], cfg), "resource_limit_gpu_memory")
        self.assertEqual(s10.stop_reason([dict(good, host_used_bytes=85)], cfg), "resource_limit_host_memory")
        self.assertEqual(s10.stop_reason([dict(error="NVML unavailable")], cfg), "telemetry_failure")
        self.assertIsNone(s10.stop_reason([good], cfg))

    def test_eligible_amr_requires_actual_topology_change(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "log"
            p.write_text("0 MeshBlocks created, 0 deleted by AMR\n")
            self.assertFalse(s10.eligible_amr(p))
            p.write_text("0 MeshBlocks created, 224 deleted by AMR\n")
            self.assertTrue(s10.eligible_amr(p))

    def test_missing_or_stale_per_node_telemetry_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(RuntimeError):
                s10.sampler_rows(root, 16, s10.config())
            s10.atomic(root / "node.json", dict(epoch=0))
            with self.assertRaises(RuntimeError):
                s10.sampler_rows(root, 1, s10.config())

    def test_prepare_records_hashes_and_separates_validation(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            s10.prepare_run(run, s10.config())
            for name, sha in s10.read(run / "INPUT_SHA256.json").items():
                self.assertEqual(s10.digest(run / name), sha)
            self.assertNotIn("validation_padding", (run / "compact.athinput").read_text())
            self.assertEqual(s10.read(run / "state.json")["checkpoint"], None)


class CheckpointTests(unittest.TestCase):
    def test_clean_resource_stop_can_be_sealed_but_failed_output_cannot(self):
        cp = dict(path="out/final.rst", sha256="a" * 64, time=9.33125, cycle=2256)
        result = dict(exit_code=0, checkpoint=cp, stop_reason="resource_limit_gpu_memory")
        receipt = s10.seal_receipt(result)
        self.assertTrue(receipt["checkpoint_complete"])
        self.assertFalse(receipt["scientific_completion"])
        self.assertEqual(receipt["checkpoint"], "out/final.rst")
        self.assertFalse(s10.seal_receipt(dict(result, exit_code=1))["checkpoint_complete"])
        self.assertFalse(s10.seal_receipt(dict(result, verification_error="bad exit progress"))["checkpoint_complete"])

    def native(self, path, cycle=2, root=3):
        # Sparse, exact layout fixture; all five million particles are recorded removed.
        header = b"<par_end>\n"
        offset = len(header)
        block_bytes = 25 * 40**3 * 8
        block_start = offset + 252 + 20 + 56 + 8
        particle_start = block_start + block_bytes
        with path.open("wb") as f:
            f.write(header)
            f.write(struct.pack("<ii", 1, root))
            f.seek(offset + 232)
            f.write(struct.pack("<ddi", cycle * .00625, .00625, cycle))
            f.seek(block_start - 8)
            f.write(struct.pack("<Q", block_bytes))
            f.seek(particle_start)
            f.write(struct.pack("<4d", 0, 5_000_000, 0, 0))

    def test_native_layout_and_checksum(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); p = root / "valid.rst"; self.native(p)
            cp = s10.latest_checkpoint(root, 3)
            self.assertEqual(cp["cycle"], 2)
            self.assertEqual(cp["sha256"], s10.digest(p))
            self.assertEqual(s10.verify_restart(cp, 3)["time"], .0125)

    def test_same_size_corruption_rejected_by_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "valid.rst"; self.native(p)
            cp = s10.checkpoint(p)
            with p.open("r+b") as f:
                f.seek(1000); f.write(b"bad")
            with self.assertRaisesRegex(ValueError, "receipt"):
                s10.verify_restart(cp, 3)

    def test_truncated_newer_checkpoint_does_not_replace_complete(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.native(root / "complete.rst", cycle=2)
            self.native(root / "newer.rst", cycle=3)
            with (root / "newer.rst").open("r+b") as f:
                f.truncate(4096)
            cp = s10.latest_checkpoint(root, 3)
            self.assertEqual(cp["cycle"], 2)
            self.assertEqual(len(cp["rejected_candidates"]), 1)

    def test_compact_checkpoint_rejected_as_production(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "compact.rst"; self.native(p, root=1)
            with self.assertRaises(ValueError):
                s10.checkpoint(p, expected_root=3)

    def test_particle_audit_uses_pinned_native_indices(self):
        import numpy as np
        a = np.zeros((10, 3)); a[0] = 1; a[8] = [0, 1, 2]; a[9] = [100, 200, 300]
        cp = dict(particles=3, removed=[4_999_997, 0, 0], blocks=3)
        with patch.object(s10, "native_particles", return_value=a):
            self.assertTrue(s10.audit_native(cp)["gids_in_leaf_range"])
            a[8, 2] = 3
            with self.assertRaises(ValueError):
                s10.audit_native(cp)


if __name__ == "__main__":
    unittest.main()
