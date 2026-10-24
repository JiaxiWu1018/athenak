"""Preserve real endpoint timesteps rather than rounding their timestamps."""
import unittest
from products import merged


class NativeTimeMergeTests(unittest.TestCase):
    def test_tiny_final_timestep_keeps_both_native_records(self):
        before = dict(time=299.99999999999585, cycle=1576., dt=.1, m_alive=1.030106481999273)
        final = dict(time=300., cycle=1577., dt=4.149569576838985e-12, m_alive=1.030106481999273)
        self.assertEqual(merged([before, final]), [before, final])

    def test_exact_duplicate_coalesces(self):
        row = dict(time=300., cycle=1577., dt=4.149569576838985e-12, m_alive=1.030106481999273)
        self.assertEqual(merged([row, row.copy()]), [row])

    def test_incompatible_exact_duplicate_is_refused(self):
        before = dict(time=300., cycle=1577., m_alive=1.)
        after = dict(time=300., cycle=1577., m_alive=.99)
        with self.assertRaises(AssertionError):
            merged([before, after])

    def test_actual_times_remain_ordered(self):
        rows = [dict(time=300., cycle=1577.), dict(time=299.99999999999585, cycle=1576.), dict(time=0.,cycle=0.)]
        self.assertEqual([r['cycle'] for r in merged(rows)], [0.,1576.,1577.])


if __name__ == '__main__':
    unittest.main()
