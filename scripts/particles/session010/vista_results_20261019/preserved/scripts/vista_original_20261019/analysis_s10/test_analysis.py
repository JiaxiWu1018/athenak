import sys
import unittest
import os
from unittest.mock import patch
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from science import contiguous_groups, live, merge_rows, movie_indices, raster
import run


class Integrity(unittest.TestCase):
    def test_cli_refuses_login_before_writing(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, 'argv', ['run.py']), patch.object(run, 'freeze') as freeze:
            with self.assertRaises(RuntimeError):
                run.main()
            freeze.assert_not_called()

    def test_cli_refuses_gh_analysis_before_writing(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'SLURM_JOB_PARTITION': 'gh'}, clear=True), patch.object(sys, 'argv', ['run.py']), patch.object(run, 'freeze') as freeze:
            with self.assertRaises(RuntimeError):
                run.main()
            freeze.assert_not_called()
    def test_restart_same_observation_later_case_wins(self):
        a = np.array([[1, .5, 3, 0]])
        b = np.array([[1, .5, 3, 1]])
        result = merge_rows([a, b], check_cols=[2])
        self.assertEqual(result.shape, (1, 4))
        self.assertEqual(result[0, -1], 1)

    def test_conflicting_restart_rejected(self):
        with self.assertRaises(ValueError):
            merge_rows([[[1, .5, 3]], [[1, .5, 4]]], check_cols=[2])

    def test_live_horizon_tracker_without_particle_core(self):
        row = np.zeros(18); row[15] = .01; row[16] = 1
        self.assertTrue(live(row))
        row[16] = 0
        self.assertFalse(live(row))
        row[13] = 10; row[15] = np.nan
        self.assertFalse(live(row))

    def test_invalid_rows_and_restarts_split_arcs(self):
        groups = contiguous_groups(np.arange(6.), [0, 20, 40, 60, 80, 100],
                                    [True, True, False, True, True, True], [0, 0, 0, 0, 1, 1])
        self.assertEqual([g.tolist() for g in groups], [[0, 1], [3], [4, 5]])

    def test_amr_time_step_change_not_a_gap(self):
        groups = contiguous_groups([0, .125, .15625], [0, 20, 40], [True]*3, [0]*3)
        self.assertEqual(len(groups), 1)


class Rendering(unittest.TestCase):
    def fixture(self):
        return dict(mb_geometry=np.array([[-1, 1, -1, 1, 0, 0]]),
                    mb_logical=np.array([[0, 0, 0, 3]]),
                    mb_data={"E": [np.array([[[1, 2], [3, 4]]])]})

    def test_pixel_centers_and_orientation(self):
        values, level = raster(self.fixture(), "E", "xy", 1, cells=2)
        np.testing.assert_array_equal(values, [[1, 2], [3, 4]])
        np.testing.assert_array_equal(level, np.zeros((2, 2)))

    def test_uncovered_pixels_rejected(self):
        with self.assertRaises(ValueError):
            raster(self.fixture(), "E", "xy", 2, cells=4)

    def test_xz_orientation(self):
        fixture = self.fixture(); fixture['mb_geometry'] = np.array([[-1, 1, 0, 0, -1, 1]])
        fixture['mb_data']['E'] = [np.array([[[1, 2]], [[3, 4]]])]
        values, _ = raster(fixture, "E", "xz", 1, cells=2)
        np.testing.assert_array_equal(values, [[1, 2], [3, 4]])

    def test_sparse_movie_holds_old_snapshot(self):
        indices = movie_indices([0, 6, 8], 8, fps=2, speed=2)
        np.testing.assert_array_equal(indices, [0, 0, 0, 0, 0, 0, 1, 1, 2])

    def test_movie_uses_final_snapshot_at_end(self):
        indices = movie_indices([0, 1, 2.1], 2.1, fps=12, speed=2)
        self.assertEqual(indices[-1], 2)
        self.assertTrue(np.all(np.diff(indices) >= 0))

    def test_final_vtk_printing_precision_is_not_a_missing_frame(self):
        indices = movie_indices([0, 36., 36.55], 36.54999999998345)
        self.assertEqual(indices[-1], 2)


if __name__ == '__main__':
    unittest.main()
