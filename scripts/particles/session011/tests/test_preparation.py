import sys
from pathlib import Path
import unittest
import numpy as np
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from prepare import parse
from analysis_utils import discrete_jz_terms,orbit,ffi,removal_corrected_jz

class PreparationTests(unittest.TestCase):
 def test_frozen_physics(self):
  a=parse((HERE/'inputs/session010_reference.athinput').read_text())
  b=parse((HERE/'inputs/production.athinput').read_text())
  for block in ['mesh','meshblock','mesh_refinement','z4c_amr','problem']:
   self.assertEqual(a[block],b[block])
  p=dict(b['particles']);p.pop('orbit_diagnostic_dt');self.assertEqual(a['particles'],p)
  self.assertEqual(b['time']['tlim'],'150.0');self.assertEqual(b['time']['nlim'],'-1')
  self.assertEqual(b['output5']['dt'],'0.5');self.assertEqual(b['output7']['dt'],'10.0')
  self.assertEqual([b['z4c'][f'extraction_radius_{i}'] for i in range(3)],['40.0','50.0','60.0'])

 def test_jz_exact_product_rule(self):
  rng=np.random.default_rng(4001);x0,u0,x1,u1=rng.normal(size=(4,100,3));mass=rng.uniform(size=100)
  m,p=discrete_jz_terms(x0,u0,x1,u1,mass)
  direct=mass*((x1[:,0]*u1[:,1]-x1[:,1]*u1[:,0])-(x0[:,0]*u0[:,1]-x0[:,1]*u0[:,0]))
  np.testing.assert_allclose(m+p,direct,rtol=1e-12,atol=1e-14)

 def test_no_bridging_orbit_gap(self):
  t=np.r_[np.arange(4)*.025,1+np.arange(4)*.025];right=np.c_[np.cos(t),np.sin(t),np.zeros(len(t))]
  d=orbit(t,np.zeros_like(right),right,np.ones(len(t),dtype=bool))
  self.assertAlmostEqual(d['observed_arc'][-1],.15)

 def test_ffi_plus_cross_sign(self):
  t=np.arange(1024)*.01;f=5/(len(t)*.01);h=np.exp(-2j*np.pi*f*t)
  reconstructed=ffi(t,-(2*np.pi*f)**2*h,f/4,taper=False)
  np.testing.assert_allclose(reconstructed,h,atol=1e-12)

 def test_ffi_rejects_gaps(self):
  with self.assertRaises(ValueError):ffi(np.r_[np.arange(8),20],np.ones(9),.1)

 def test_removal_not_torque(self):
  np.testing.assert_allclose(removal_corrected_jz([10,7],[0,3],[0,0],[0,0]),[0])

 def test_common_guess_path_and_persistence(self):
  src=(HERE.parents[2]/'src/z4c/fastflow.cpp').read_text()
  self.assertEqual(src.count('if (LoadSearchGuess()) { return; }'),2)
  self.assertIn('if (!quality_geometry_ok || !capability_on_grid) { last_a0 = -1.0; }',src)
  self.assertIn('if (quality_geometry_ok && capability_on_grid) { CacheSearchGuess(); }',src)
  self.assertIn('ah_surf_valid = false;',src)
  self.assertIn('Kokkos::fabs(hmean)/area <= ah_surf_hrel_limit',src)

 def test_no_production_submission_in_build(self):
  s=(HERE/'build.sh').read_text();self.assertNotIn('sbatch ',s);self.assertNotIn('ibrun ',s)

if __name__=='__main__':unittest.main()
