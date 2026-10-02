#!/usr/bin/env python3
"""Small synthetic analysis fixtures; run through Slurm, never science outputs."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import analyze_s8 as analysis


class AnalysisChecks(unittest.TestCase):
    def test_circular_motion_and_small_denominator(self):
        time = np.linspace(0, 4, 81)
        vector = 6 * np.c_[np.cos(.2*time), np.sin(.2*time), np.zeros(len(time))]
        d, phase, omega, radial, ratio, groups = analysis.orbital_series(time, vector, np.ones(len(time), bool))
        np.testing.assert_allclose(d, 6)
        np.testing.assert_allclose(omega, .2, atol=1e-12)
        np.testing.assert_allclose(radial, 0, atol=1e-12)
        np.testing.assert_allclose(ratio, 0, atol=1e-12)
        _, _, _, _, zero_ratio, _ = analysis.orbital_series(time, np.tile([6., 0., 0.], (len(time), 1)), np.ones(len(time), bool))
        self.assertTrue(np.isnan(zero_ratio).all())

    def test_no_derivative_bridges_diagnostic_gap(self):
        time = np.arange(9, dtype=float)
        angle = np.r_[.1*np.arange(4), 2., -2. + .1*np.arange(4)]
        vector = np.c_[np.cos(angle), np.sin(angle), np.zeros(9)]
        live = np.ones(9, bool); live[4] = False
        d, phase, omega, radial, ratio, groups = analysis.orbital_series(time, vector, live)
        self.assertEqual([len(g) for g in groups], [4, 4])
        self.assertTrue(np.isnan(omega[4]))
        np.testing.assert_allclose(omega[live], .1, atol=1e-12)
        np.testing.assert_allclose(radial[live], 0, atol=1e-12)

    def test_small_rendering_fixture_and_mode_contract(self):
        ffmpeg = shutil.which('ffmpeg')
        self.assertIsNotNone(ffmpeg, 'movie encoder required for this allocated fixture check')
        with tempfile.TemporaryDirectory(prefix='s8_analysis_fixture_') as temp:
            root = Path(temp); run = root/'runs/gate_output'; run.mkdir(parents=True)
            (run/'ARCHIVE_VERIFIED.json').write_text('{}')
            (root/'evidence').mkdir()
            (root/'evidence/amd_state.json').write_text(json.dumps(dict(status='segment_cap', time=.25)))
            time = np.linspace(0, .25, 6)
            for index, sign in ((0, -1), (1, 1)):
                track = np.zeros((len(time), 19)); track[:, 0] = np.arange(len(time)); track[:, 1] = time
                track[:, 2] = sign*3*np.cos(time); track[:, 3] = sign*3*np.sin(time)
                track[:, 13] = 2; track[:, 15] = 1
                np.savetxt(run/f'fixture.co_{index}.txt', track)
            history = np.ones((6, 11)); history[:, 0] = time
            np.savetxt(run/'fixture.z4c.user.hst', history)
            (run/'waveforms').mkdir()
            for radius in (40, 50, 60, 70):
                real = np.zeros((6, 78)); imag = real.copy(); real[:, 0] = imag[:, 0] = time
                real[:, 1] = radius*1e-8; imag[:, 1] = -radius*1e-9
                real[:, 5] = radius*2e-8; imag[:, 5] = radius*2e-9
                np.savetxt(run/f'waveforms/rpsi4_real_{radius:04d}.txt', real)
                np.savetxt(run/f'waveforms/rpsi4_imag_{radius:04d}.txt', imag)
            for index in range(2):
                consumer = run/f'fixture.horizon_consumer_{index}.csv'
                consumer.write_text('published_this_candidate,association_ok,cycle,time,center_x,center_y,center_z,rmin\n1,1,1,0.05,0,0,0,0.2\n0,1,2,0.1,0,0,0,0.2\n')
                row = np.ones(21); row[0] = 1; row[1] = .05
                rejected = row.copy(); rejected[0] = 2; rejected[1] = .1
                np.savetxt(run/f'fixture.horizon_summary_{index}.txt', np.vstack([row, rejected]))
            for tm in (0., .25):
                (run/f'fixture_{tm}.part.vtk').write_text(f'time= {tm}\n')
            particles = dict(tag_float=np.array([0, 1, 3000000, 3000001, 4000000, 4000001.]),
                             position=np.array([[0, 0, 0], [.2, .3, 0], [-3, 0, 0], [-3, .1, 0], [3, 0, 0], [3, -.1, 0.]]),
                             momentum=np.zeros((6, 3)), mass=np.ones(6)*1e-5)
            encoder = types.SimpleNamespace(get_ffmpeg_exe=lambda: ffmpeg)
            with patch.object(sys, 'argv', ['analyze_s8.py', '--root', str(root)]), patch.dict(sys.modules, {'imageio_ffmpeg': encoder}), patch.object(analysis, 'read_particles', return_value=particles):
                analysis.main()
            out = next((root/'analysis').iterdir())
            summary = json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['accepted_horizon_rows'], {'0': 1, '1': 1, '2': 0})
            minus = np.loadtxt(out/'rpsi4_2-2_r40.csv', delimiter=',')
            plus = np.loadtxt(out/'rpsi4_22_r40.csv', delimiter=',')
            np.testing.assert_allclose(minus[:, 2], 40e-8)
            np.testing.assert_allclose(plus[:, 2], 80e-8)
            for name in ('orbit.png', 'radial_tangential.png', 'envelope_relative.png', 'raw_waveform.png', 'central.mp4', 'context.mp4', 'density.mp4'):
                self.assertGreater((out/name).stat().st_size, 100)


if __name__ == '__main__':
    unittest.main()
