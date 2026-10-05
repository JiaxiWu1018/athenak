import unittest
from horizon_review_20261005 import accepted_measurements


class HorizonJoinChecks(unittest.TestCase):
    def setUp(self):
        self.candidate = dict(cycle='3962', time='11.99687500000123', area='0.62104631851252368',
                              rmin='0.077227061466297819', center_x='-2.802734375',
                              center_y='-1.189453125', center_z='-0.005859375',
                              published_this_candidate='1', association_ok='1',
                              quality_geometry_ok='1', quality_persist_ok='1')
        self.row = [4., 11.9969, .1111546432554873, 0., 0., 0., 0., .6210463185125237,
                    .1, .1, .078, .07722706146629782, .1111545047896687, .0031568344,
                    float('nan'), float('nan'), float('nan'), float('nan'),
                    -2.802734375, -1.189453125, -.005859375]

    def test_actual_iteration_and_time_format_with_optional_nan_momentum(self):
        rows, counts = accepted_measurements([self.candidate], [self.row])
        self.assertEqual(counts['matched'], 1)
        self.assertEqual(rows[0][0]['cycle'], '3962')
        self.assertEqual(rows[0][1][0], 4.)

    def test_rejected_and_different_surface_are_excluded(self):
        rejected = dict(self.candidate, published_this_candidate='0')
        wrong = self.row.copy(); wrong[7] *= 1.01
        rows, counts = accepted_measurements([rejected], [self.row])
        self.assertEqual(counts['published'], 0)
        rows, counts = accepted_measurements([self.candidate], [wrong])
        self.assertFalse(rows); self.assertEqual(counts['unmatched'], 1)

    def test_ambiguous_match_is_not_a_measurement(self):
        rows, counts = accepted_measurements([self.candidate], [self.row, self.row])
        self.assertFalse(rows); self.assertEqual(counts['ambiguous'], 1)


if __name__ == '__main__':
    unittest.main()
