"""Small metadata fixtures; the raw-hash command is explicitly mocked."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

DIRECTORY = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('endpoint_manifest', DIRECTORY/'endpoint_manifest.py')
ep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ep)


class EndpointManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)/'yc0500_long700'
        self.c = dict(id='yc0500', key='yc0500_long700')
        self.checkpoint = dict(sha256='a'*64, time=700)
        self.context = patch.multiple(ep, ROOT=self.root, EXT=self.root.parent, MACHINE='amd')
        self.context.start()
        self.env = patch.dict(os.environ, SLURM_JOB_ID='unit_fixture', ST8X_START_EPOCH=str(time.time()), ST8X_WALL_SECONDS='900')
        self.env.start()
        self.write('runs/yc0500/state.json', dict(complete=True, latest=self.checkpoint))
        self.write('evidence/CASE_NUMERICAL_ACCEPTANCE.json', dict(numerical_checks_passed=True, checkpoint=self.checkpoint))
        self.write('runs/yc0500/FINAL_FIELDS_OK.json', dict(no_evolution=True, checkpoint=self.checkpoint))
        self.write('review_packet/MANIFEST.json', dict(checkpoint=self.checkpoint))
        self.deployment = dict(source_commit='b'*40, python_files={name: ep.sha(DIRECTORY/name) for name in ['endpoint_manifest.py','trajectory.py','closed_output_manifest.py']})
        p = self.root.parent/'evidence/ENDPOINT_MANIFEST_DEPLOYMENT.json'
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(self.deployment))
        self.calls = []
        self.command = patch.object(ep, 'cmd', self.fake_hash)
        self.command.start()
        self.du = patch.object(ep.subprocess, 'check_output', return_value='1024 fixture\n')
        self.du.start()

    def write(self, name, data):
        p = self.root/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data))

    def fake_hash(self, args):
        self.calls.append(args)
        self.write('evidence/CLOSED_OUTPUT_MANIFEST.json', dict(stable_files_verified=True, checkpoint=self.checkpoint, file_count=3, logical_bytes=123))

    def tearDown(self):
        self.du.stop()
        self.command.stop()
        self.env.stop()
        self.context.stop()
        self.tmp.cleanup()

    def test_complete_endpoint_hashes_then_records_retention(self):
        ep.finish(self.c)
        self.assertEqual(len(self.calls), 1)
        r = ep.load(self.root/'evidence/ENDPOINT_RETENTION.json')
        self.assertFalse(r['reused_existing_inventory'])
        self.assertTrue(r['no_native_evolution_here'])
        self.assertEqual(r['checkpoint'], self.checkpoint)

    def test_short_allocation_defers_without_hashing(self):
        with patch.dict(os.environ, ST8X_WALL_SECONDS='60'):
            ep.finish(self.c)
        self.assertEqual(self.calls, [])
        self.assertTrue((self.root/'evidence/jobs/unit_fixture/ENDPOINT_MANIFEST_DEFERRED.json').exists())
        self.assertFalse((self.root/'evidence/ENDPOINT_RETENTION.json').exists())

    def test_existing_inventory_is_reused(self):
        self.fake_hash([])
        self.calls.clear()
        ep.finish(self.c)
        self.assertEqual(self.calls, [])
        self.assertTrue(ep.load(self.root/'evidence/ENDPOINT_RETENTION.json')['reused_existing_inventory'])

    def test_incomplete_science_is_refused(self):
        self.write('runs/yc0500/state.json', dict(complete=False, latest=self.checkpoint))
        with self.assertRaises(AssertionError):
            ep.finish(self.c)
        self.assertEqual(self.calls, [])

    def test_changed_source_is_refused(self):
        self.deployment['python_files']['trajectory.py'] = 'c'*64
        p = self.root.parent/'evidence/ENDPOINT_MANIFEST_DEPLOYMENT.json'
        p.write_text(json.dumps(self.deployment))
        with self.assertRaises(AssertionError):
            ep.finish(self.c)
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
