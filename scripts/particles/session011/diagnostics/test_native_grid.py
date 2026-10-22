import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
from check_native_grid import scan

class NativeGridChecks(unittest.TestCase):
    def test_full_hash_and_nonfinite_grid_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'state';prefix=b'not8aligned';grid=np.array([np.nan,1.,2.,np.inf,-np.inf,3.,4.,np.nan],dtype='<f8')
            # A nonfinite native particle diagnostic in the suffix is not a grid value.
            suffix=np.array([np.nan],dtype='<f8').tobytes();data=prefix+grid.tobytes()+suffix;p.write_bytes(data)
            result=scan(p,len(prefix),grid.nbytes,chunk_bytes=16)
            self.assertEqual(result['sha256'],hashlib.sha256(data).hexdigest())
            self.assertEqual(result['grid_values'],8);self.assertEqual(result['nonfinite_grid_values'],4)
            self.assertEqual([r['flat_index'] for r in result['first_nonfinite']],[0,3,4,7])
            self.assertEqual(result['first_nonfinite'][-1]['byte_offset'],len(prefix)+7*8)
    def test_finite_and_truncated_grid(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'state';p.write_bytes(b'x'+np.array([0.,1.,-2.,3.],dtype='<f8').tobytes())
            self.assertEqual(scan(p,1,32,chunk_bytes=8)['nonfinite_grid_values'],0)
            with self.assertRaisesRegex(ValueError,'truncated grid'):scan(p,1,40,chunk_bytes=8)
            with self.assertRaises(ValueError):scan(p,1,31)

if __name__=='__main__':unittest.main()
