#!/usr/bin/env python3
from __future__ import annotations
import contextlib
import csv
import importlib.util
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
if (ROOT/"vista_handoff/scripts/workflow.py").is_file():
    HANDOFF = ROOT/"vista_handoff"
    SOURCE = ROOT/"code"
else:
    HANDOFF = ROOT
    SOURCE = pathlib.Path(os.environ.get("S7_SOURCE_TREE", ROOT/"source/athenak"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


wf = load_module("s7_workflow", HANDOFF/"scripts/workflow.py")
monitor = load_module("s7_monitor", HANDOFF/"scripts/monitor.py")


def config(root: pathlib.Path):
    handoff = root/"handoff"; work = root/"work"; scratch = root/"scratch"
    (handoff/"inputs").mkdir(parents=True)
    deck = handoff/"inputs/canonical"
    deck.write_text("input\n")
    control = handoff/"inputs/control"
    control.write_text("control\n")
    cfg = {
        "campaign_id":"test-campaign", "account":"acct", "handoff_dir":str(handoff),
        "work_root":str(work), "scratch_root":str(scratch), "source_commit":"1"*40,
        "kokkos_commit":"2"*40, "input_relpath":"inputs/canonical",
        "input_sha256":wf.sha256(deck), "control_input_relpath":"inputs/control",
        "control_input_sha256":wf.sha256(control), "restart_test_ranks":4,
        "quicktest_gpu_memory_fraction_limit":.85,
        "quicktest_storage_stop_bytes":800,
        "production_nodes":8, "production_max_nmb_per_rank":400, "startup_nodes":8,
        "max_segments":3, "segment_wall":"01:00:00", "athena_wall":"00:45:00",
        "node_hour_cap":24, "storage_warn_bytes":1000, "storage_stop_bytes":2000,
        "checkpoint_reserve_bytes":500, "monitor_seconds":300,
        "telemetry_grace_seconds":600, "startup_telemetry_grace_seconds":900,
        "max_infrastructure_retries_per_segment":1, "target_time":64.0,
        "modules":"gcc cuda openmpi cmake", "ucx_ib_gpu_direct_rdma":"no",
        "quicktest_wall":"01:00:00",
        "quicktest_nlim":2, "build_jobs":2, "production_authorized":True,
        "minimum_checkpoint_bytes":8*1024**3, "gpu_partition":"gh",
        "dev_gpu_partition":"gh-dev", "cpu_partition":"gg",
        "verify_wall":"00:10:00", "build_wall":"01:00:00", "inspect_wall":"00:10:00",
        "authorized_early_stop":False,
    }
    return cfg


def state():
    return {"segments_completed":0, "last_time":0.0, "last_cycle":0,
            "stop_requested":False, "emergency_cancel":False,
            "infrastructure_retries":{}, "jobs":[]}


def result(**changes):
    value = {"time":1.0, "cycle":1, "exit_code":0, "checkpoint_complete":True,
             "storage_bytes":10, "fatal_pattern":False, "stop_reasons":[]}
    value.update(changes)
    return value


class Decisions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = config(pathlib.Path(self.tmp.name))
        self.st = state()
    def tearDown(self): self.tmp.cleanup()
    def action(self, res, sched="COMPLETED", **kw):
        return wf.decision(self.cfg, self.st, res, sched, **kw)[0]
    def test_continue_and_physical_endpoint(self):
        self.assertEqual(self.action(result()), "continue")
        self.assertEqual(self.action(result(time=64.0)), "complete_physical_endpoint")
    def test_authorized_early_stop_only_when_enabled(self):
        r=result(authorized_common_horizon_stop=True)
        self.assertEqual(self.action(r), "continue")
        self.cfg["authorized_early_stop"]=True
        self.assertEqual(self.action(r), "authorized_scientific_stop")
    def test_sticky_stops(self):
        self.assertEqual(self.action(result(), stop_requested=True), "user_stop")
        self.st["emergency_cancel"]=True
        self.assertEqual(self.action(result()), "emergency_cancel")
    def test_numerical_and_diagnostic_failures(self):
        self.assertEqual(self.action(result(exit_code=4)), "numerical_failure")
        self.assertEqual(self.action(result(fatal_pattern=True)), "numerical_failure")
        self.assertEqual(self.action(result(stop_reasons=["numerical_failure"])),
                         "numerical_failure")
        self.assertEqual(self.action({"time":"bad", "cycle":1}), "diagnostic_stale")
        self.assertEqual(self.action(result(stop_reasons=["diagnostic_stale"])),
                         "diagnostic_stale")
    def test_checkpoint_no_progress_and_storage(self):
        self.assertEqual(self.action(result(checkpoint_complete=False)),
                         "configuration_test_failure")
        self.assertEqual(self.action(result(time=0.0, cycle=0)), "no_progress")
        self.assertEqual(self.action(result(storage_bytes=2000)), "resource_limit")
    def test_caps(self):
        self.cfg["node_hour_cap"]=1
        self.assertEqual(self.action(result()), "resource_limit")
        self.cfg["node_hour_cap"]=24; self.cfg["max_segments"]=1
        self.assertEqual(self.action(result()), "segment_limit")
    def test_infrastructure_retry_and_ambiguous_exit(self):
        self.assertEqual(self.action(None, "NODE_FAIL", segment=1), "continue")
        self.st["infrastructure_retries"]={wf.infrastructure_retry_key(self.st):1}
        self.assertEqual(self.action(None, "NODE_FAIL", segment=1),
                         "infrastructure_failure")
        self.assertEqual(self.action(None, "FAILED", segment=1),
                         "configuration_test_failure")


class ConfigAndSubmission(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=pathlib.Path(self.tmp.name)
        self.cfg=config(self.root); self.path=self.root/"config.json"
        self.path.write_text(json.dumps(self.cfg))
    def tearDown(self): self.tmp.cleanup()
    def test_config_and_prepare_only_submits_nothing(self):
        self.assertEqual(wf.validate_config(self.cfg, full=True), [])
        unsafe=dict(self.cfg); unsafe["minimum_checkpoint_bytes"]=1024**2
        self.assertTrue(any("at least 8 GiB" in value
                            for value in wf.validate_config(unsafe, full=True)))
        unsafe=dict(self.cfg); unsafe["production_max_nmb_per_rank"]=337
        self.assertTrue(any("production_max_nmb_per_rank" in value
                            for value in wf.validate_config(unsafe, full=True)))
        unsafe=dict(self.cfg); unsafe["node_hour_cap"]=4609
        self.assertTrue(any("authorized 4608" in value
                            for value in wf.validate_config(unsafe, full=True)))
        with mock.patch.object(wf, "scheduler", side_effect=AssertionError("scheduler called")):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(wf.cmd_submit(self.cfg, self.path, True, False), 0)
        self.assertIn("NOT SUBMITTED", out.getvalue())
        self.assertEqual(wf.load_json(wf.state_path(self.cfg))["jobs"], [])
    def test_restart_config_initializes_progress_and_checks_checkpoint(self):
        checkpoint=self.root/"restart.rst"
        with checkpoint.open("wb") as stream:
            stream.write(b"x"*53207 + b"<par_end>\n")
            stream.truncate(8*1024**3)
        bundle=self.root/"handoff/source/fixed.bundle"
        bundle.parent.mkdir(parents=True)
        bundle.write_bytes(b"bundle")
        self.cfg.update(
            initial_checkpoint=str(checkpoint),
            initial_checkpoint_sha256="a"*64,
            initial_checkpoint_bytes=checkpoint.stat().st_size,
            initial_cycle=12, initial_time=1.25,
            initial_checkpoint_parameter_end_offset=53207,
            production_restart_smoke=True,
            source_bundle_relpath="source/fixed.bundle",
            source_bundle_sha256=wf.sha256(bundle),
        )
        self.assertEqual(wf.validate_config(self.cfg, full=True), [])
        initialized=wf.load_or_initialize_state(self.cfg)
        self.assertEqual(initialized["latest_checkpoint"], str(checkpoint))
        self.assertEqual(initialized["last_cycle"], 12)
        self.assertEqual(initialized["last_time"], 1.25)
        with checkpoint.open("r+b") as stream:
            stream.truncate(checkpoint.stat().st_size-1)
        self.assertTrue(any("size differs" in value
                            for value in wf.validate_config(self.cfg, full=True)))
    def test_full_chain_dependencies_and_duplicate_prevention(self):
        calls=[]; next_id=iter(range(100,200))
        def fake(cfg, command, args, check=True):
            calls.append((command,list(args)))
            if command in ("squeue","sacct"):
                return subprocess.CompletedProcess([],0,"","")
            return subprocess.CompletedProcess([],0,str(next(next_id))+"\n","")
        with mock.patch.object(wf,"scheduler",side_effect=fake):
            with contextlib.redirect_stdout(io.StringIO()):
                wf.cmd_submit(self.cfg,self.path,False,False)
                wf.cmd_submit(self.cfg,self.path,False,False)
        sbatch=[args for command,args in calls if command=="sbatch"]
        self.assertEqual(len(sbatch),9)  # verify/build/quick + 3 segment/inspect pairs
        self.assertIn("afterok:100", sbatch[1])
        self.assertIn("afterany:103", sbatch[4])
        self.assertEqual(len(wf.load_json(wf.state_path(self.cfg))["jobs"]),9)
    def test_site_wrapper_output_yields_one_numeric_job_id(self):
        st=wf.load_or_initialize_state(self.cfg)
        def fake(cfg, command, args, check=True):
            if command in ("squeue", "sacct"):
                return subprocess.CompletedProcess([],0,"","")
            return subprocess.CompletedProcess(
                [],0,"Welcome to Vista\nNo reservation for this job\n321;vista\n","notice\n")
        kwargs=dict(kind="verify",sequence=0,script=HANDOFF/"fake.sbatch",
                    partition="gg",nodes=1,wall="00:10:00",dependency=None,extra=[])
        with mock.patch.object(wf,"scheduler",side_effect=fake):
            self.assertEqual(wf.submit_one(self.cfg,st,**kwargs),"321")
        job=wf.load_json(wf.state_path(self.cfg))["jobs"][0]
        self.assertEqual(pathlib.Path(job["sbatch_status"]).read_text(),"0\n")
        self.assertIn("Welcome to Vista", pathlib.Path(job["sbatch_stdout"]).read_text())
    def test_interrupted_state_update_recovers_from_sacct(self):
        st=wf.load_or_initialize_state(self.cfg); accepted={"value":False}; saves=0
        original=wf.save_state
        def fake_sched(cfg, command, args, check=True):
            if command=="squeue": return subprocess.CompletedProcess([],0,"","")
            if command=="sacct":
                return subprocess.CompletedProcess([],0,"777\n" if accepted["value"] else "","")
            if command=="sbatch":
                accepted["value"]=True
                return subprocess.CompletedProcess([],0,"777\n","")
            raise AssertionError(command)
        def crashing_save(cfg, value):
            nonlocal saves
            saves += 1
            if saves == 2: raise RuntimeError("injected crash after scheduler acceptance")
            original(cfg,value)
        kwargs=dict(kind="segment",sequence=1,script=HANDOFF/"fake.sbatch",
                    partition="gh",nodes=2,wall="01:00:00",dependency=None,extra=[])
        with mock.patch.object(wf,"scheduler",side_effect=fake_sched), \
             mock.patch.object(wf,"save_state",side_effect=crashing_save):
            with self.assertRaises(RuntimeError): wf.submit_one(self.cfg,st,**kwargs)
        recovered=wf.load_or_initialize_state(self.cfg)
        with mock.patch.object(wf,"scheduler",side_effect=fake_sched):
            self.assertEqual(wf.submit_one(self.cfg,recovered,**kwargs),"777")
        self.assertEqual(len(wf.load_json(wf.state_path(self.cfg))["jobs"]),1)
    def test_gate_binding_pass_and_tamper_fail(self):
        ctl=wf.control_dir(self.cfg); ctl.mkdir(parents=True)
        exe=pathlib.Path(self.cfg["work_root"])/"build_cuda_gi/src/athena"
        exe.parent.mkdir(parents=True); exe.write_bytes(b"executable")
        (ctl/"executable.sha256").write_text(wf.sha256(exe)+"  athena\n")
        detail=ctl/"detail"; detail.write_text("ok")
        for stage in ("verify","build","quicktest"):
            wf.cmd_mark(self.cfg,stage,"pass",detail)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(wf.cmd_gate_check(self.cfg,["verify","build","quicktest"]),0)
        self.cfg["monitor_seconds"]=600
        with self.assertRaises(RuntimeError):
            wf.cmd_gate_check(self.cfg,["verify"])
        self.cfg["monitor_seconds"]=300
        detail.write_text("tampered")
        with self.assertRaises(RuntimeError): wf.cmd_gate_check(self.cfg,["verify"])

    def test_stops_are_sticky_and_cancel_only_recorded_jobs(self):
        st=wf.load_or_initialize_state(self.cfg)
        st["jobs"]=[{"job_id":"101"},{"job_id":"202"}]
        wf.save_state(self.cfg,st)
        calls=[]
        def fake(cfg,command,args,check=True):
            calls.append((command,list(args)))
            return subprocess.CompletedProcess([],0,"","")
        with mock.patch.object(wf,"scheduler",side_effect=fake):
            self.assertEqual(wf.cmd_stop(self.cfg,False),0)
            self.assertTrue((wf.control_dir(self.cfg)/"STOP_REQUESTED").exists())
            self.assertFalse(calls)
            self.assertEqual(wf.cmd_stop(self.cfg,True),0)
        self.assertEqual(calls,[("scancel",["101","202"])])
        saved=wf.load_json(wf.state_path(self.cfg))
        self.assertTrue(saved["stop_requested"])
        self.assertTrue(saved["emergency_cancel"])


class HorizonAndCheckpointFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=pathlib.Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()
    def write_tracker(self,index,x):
        (self.root/f"case.co_{index}.txt").write_text(
            "# tracker\n10 1.0 "+str(x)+" 0 0 0 0 0 7 .01 .01 .01 2 0 0 .01 1\n")
    def test_common_horizon_requires_persistence_time_and_enclosure(self):
        self.write_tracker(0,-.1); self.write_tracker(1,.1)
        fields=["cycle","time","capability_mode","center_x","center_y","center_z",
                "rmin","rmax","area","hrel","local_dx","rmin_cells",
                "quality_geometry_ok","quality_streak","quality_persist_ok",
                "capability_finite_positive","capability_on_grid",
                "capability_center_jump_ok","capability_geometry_ok","association_ok",
                "published_this_candidate"]
        path=self.root/"case.horizon_consumer_2.csv"
        with path.open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
            for cycle in (8,9,10):
                row={key:1 for key in fields}; row.update(
                    cycle=cycle,time=1.0,center_x=0,center_y=0,center_z=0,
                    rmin=.2,rmax=.3,area=1,hrel=.01)
                writer.writerow(row)
        cfg={"common_horizon_persistence_records":3,
             "common_horizon_tracker_time_tolerance":.2}
        self.assertTrue(monitor.accepted_common_horizon(self.root,cfg)["qualified"])
        text=path.read_text().replace(",0.2,0.3,",",0.05,0.3,")
        path.write_text(text)
        self.assertFalse(monitor.accepted_common_horizon(self.root,cfg)["qualified"])
    def test_truncated_checkpoint_is_rejected(self):
        run=self.root/"run"; run.mkdir(); (run/"tiny.rst").write_bytes(b"short")
        self.write_tracker(0,-.1); self.write_tracker(1,.1)
        for path in self.root.glob("case.co_*.txt"):
            path.rename(run/path.name)
        telemetry=self.root/"telemetry.json"; telemetry.write_text('{"stop_reasons":[]}')
        log=self.root/"log"; log.write_text("clean")
        cmd=[sys.executable,str(HANDOFF/"scripts/segment_finalize.py"),
             "--run",str(run),"--campaign-root",str(self.root),"--segment","1",
             "--exit-code","0","--nodes","1","--started-epoch","0","--log",str(log),
             "--telemetry",str(telemetry),"--minimum-checkpoint-bytes","16"]
        proc=subprocess.run(cmd,text=True,capture_output=True)
        self.assertEqual(proc.returncode,3)
        self.assertFalse(json.loads((run/"result.json").read_text())["checkpoint_complete"])


class SourceContracts(unittest.TestCase):
    def test_quicktest_requires_all_participating_node_memory(self):
        batch=(HANDOFF/"scripts/vista_quicktest.sbatch").read_text()
        summary=(HANDOFF/"scripts/summarize_quicktest.py").read_text()
        self.assertIn("srun --overlap",batch)
        self.assertIn("mesh_refinement/max_nmb_per_rank=1600",batch)
        self.assertIn("PROD_CAP=$(get production_max_nmb_per_rank)",batch)
        self.assertIn('mesh_refinement/max_nmb_per_rank=$PROD_CAP',batch)
        self.assertIn("production_output",batch)
        self.assertIn("production_restart",batch)
        self.assertIn("PRODUCTION_RESTART_NLIM=$((INITIAL_CYCLE + 1))",batch)
        self.assertIn("production_restart_checks",summary)
        self.assertIn("production_outputs",summary)
        self.assertIn("waveform_real",summary)
        self.assertIn('cd "$RUN"',batch)
        self.assertIn('mkdir -p "$RUN/$label"',batch)
        self.assertIn('cd "$RUN/$label"',batch)
        self.assertIn('-d "$RUN/unboosted" "time/nlim=$NLIM"',batch)
        self.assertIn("VALIDATION_gi_cluster_s7_two_clump_ccw_con3d.athinput",batch)
        self.assertIn("e7b3930a25ef556951bea31aa82fd941b2fc163f97b168a5060362e142d729e3",batch)
        self.assertIn("missing_memory_nodes",summary)
        self.assertIn("not missing_memory_nodes",summary)
        initial=(HANDOFF/"scripts/validate_initial_s7.py").read_text()
        restart=(HANDOFF/"scripts/compare_restart_s7.py").read_text()
        self.assertIn('latest(run, "*.con3d.*.bin")',initial)
        self.assertIn('default=1e-5',restart)
        self.assertIn('"reacquisition_gap": not valid_b',restart)

    def test_segment_outputs_and_result_are_kept_under_segment(self):
        wrapper=(HANDOFF/"scripts/vista_segment.sbatch").read_text()
        self.assertIn('cd "$RUN/out"', wrapper)
        self.assertIn('--run "$RUN"', wrapper)
        self.assertIn('--campaign-root "$ACCOUNTING_ROOT"', wrapper)

    def test_inspector_accepts_canonical_and_legacy_result_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp); cfg={"scratch_root":str(root),
                                         "campaign_id":"campaign"}
            segment=root/"campaign/segments/segment_001"
            (segment/"out").mkdir(parents=True)
            (segment/"out/result.json").write_text('{"value":"legacy"}')
            self.assertEqual(wf.read_result(cfg,1)["value"],"legacy")
            (segment/"result.json").write_text('{"value":"canonical"}')
            self.assertEqual(wf.read_result(cfg,1)["value"],"canonical")

    @classmethod
    def setUpClass(cls):
        if not (SOURCE/"src").is_dir():
            raise unittest.SkipTest("set S7_SOURCE_TREE to the reconstructed AthenaK tree")

    def test_three_components_and_clean_stop_are_wired(self):
        code=SOURCE
        pack=(code/"src/mesh/meshblock_pack.hpp").read_text()
        self.assertIn("MeshBlock* pmb=nullptr",pack)
        self.assertIn("Coordinates* pcoord=nullptr",pack)
        self.assertIn("destroy_component_split_tag_2",
                      (code/"src/particles/particles.cpp").read_text())
        bvals=(code/"src/bvals/bvals_part.cpp").read_text()
        self.assertIn("? 2 : 1",bvals)
        driver=(code/"src/driver/driver.cpp").read_text()
        self.assertIn("Clean stop requested by",driver)
        self.assertIn("Finalize writes all outputs",driver)
        wrapper=(HANDOFF/"scripts/vista_segment.sbatch").read_text()
        self.assertIn('time/stop_file=$RUN/out/REQUEST_STOP',wrapper)
    def test_restart_state_is_non_authoritative_until_live_acceptance(self):
        tracker=(SOURCE/"src/z4c/compact_object_tracker.cpp").read_text()
        flow=(SOURCE/"src/z4c/fastflow.cpp").read_text()
        self.assertIn("co_\" + nstr + \"_state_valid",tracker)
        self.assertIn("ah_surf_valid = false",flow)
        self.assertIn("consumer_exclude_other_punctures",flow)
        self.assertIn("association_ok",flow)

    def test_restart_reader_accepts_large_bounded_parameter_headers(self):
        reader=(SOURCE/"src/parameter_input.cpp").read_text()
        self.assertIn("kMaxHeaderSize = 1024 * 1024", reader)
        self.assertIn("header > kMaxHeaderSize", reader)
        self.assertNotIn("header > kBufSize*10", reader)


if __name__ == "__main__":
    unittest.main(verbosity=2)
