#!/usr/bin/env python3
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import products

class ProductChecks(unittest.TestCase):
    def test_missing_metric_energy_is_unknown_and_removal_J_is_exact(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'runs/segment_0000';run.mkdir(parents=True)
            out=root/'out';out.mkdir()
            pd.DataFrame([dict(time=.1,kind=1,component=0,horizon=0,old_inside=0,
                              advected_inside=1,new_inside=1,mass=2.,old_Jz_origin=3.,
                              En_trilinear=0.,Ecoord_trilinear=0.,metric_valid=0)]).to_csv(run/'x.gi_crossings.rank0.csv',index=False)
            pd.DataFrame([dict(time=.1,component=0,reason='lapse',mass=2.,x1=3.,x2=4.,
                              u1_cov=5.,u2_cov=6.,En_trilinear=7.,Ecoord_trilinear=8.,
                              metric_valid=1)]).to_csv(run/'x.prtcl_destroy.csv',index=False)
            accepted=pd.DataFrame(columns=['horizon','state_time','state_cycle','x','y','Px','Py','mass'])
            orbit=pd.DataFrame(dict(time=[0.,.25,.5],cycle=[0,1,2],separation=[6.,5.,4.],
                                   Omega=[np.nan]*3,radial_over_tangential=[np.nan]*3))
            result=products.diagnose(root,out,dict(strict_common_time=None,cycle=2),accepted,orbit)
            self.assertIsNone(result['median_radial_ratio'])
            flux=pd.read_csv(out/'crossing_fluxes.csv');removed=pd.read_csv(out/'removal_ledgers.csv')
            self.assertTrue(np.isnan(flux.advected_En.iloc[0]))
            self.assertEqual(flux.advected_rest_mass.iloc[0],2.)
            self.assertEqual(removed.Jz.iloc[0],-4.)
            self.assertEqual(removed.Ecoord_trilinear.iloc[0],8.)
    def test_full_synthetic_report_and_movies(self):
        import subprocess,sys
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'fixture';out=root/'products';scripts=Path(__file__).parent
            subprocess.run([sys.executable,str(scripts/'product_fixture.py'),str(scripts),str(root)],check=True)
            subprocess.run([sys.executable,str(scripts/'products.py'),'--root',str(root),'--out',str(out),'--movies'],check=True)
            metrics=json.loads((out/'metrics.json').read_text())
            self.assertEqual(metrics['endpoint'],75)
            self.assertEqual(metrics['strict_common_time'],.75)
            snapshot=json.loads((out/'snapshot_manifest.json').read_text())
            self.assertEqual(snapshot['selected_cycles'],[0,1,2,3])
            self.assertTrue(all(snapshot['events_observed']))
            import imageio_ffmpeg
            for name in ('wide','close'):
                path=out/f'{name}_1080p.mp4';self.assertGreater(path.stat().st_size,1000)
                subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(path),'-f','null','-'],check=True)
            for row in json.loads((out/'manifest.json').read_text()):
                self.assertEqual(products.sha256(out/row['path']),row['sha256'])
    def test_native_harmonic_convention(self):
        theta=np.linspace(0,np.pi,41);phi=np.linspace(-np.pi,np.pi,41)
        np.testing.assert_allclose(products.harmonic(0,0,theta,phi),1/np.sqrt(4*np.pi),atol=1e-14)
        np.testing.assert_allclose(products.harmonic(1,0,theta,phi),np.sqrt(3/(4*np.pi))*np.cos(theta),atol=1e-14)
        np.testing.assert_allclose(products.harmonic(1,1,theta,phi),-np.sqrt(3/(8*np.pi))*np.sin(theta)*np.exp(1j*phi),atol=1e-14)
    def test_ffi_reconstructs_complex_periodic_signal(self):
        time=np.arange(4096)/128;omega=2*np.pi*.25
        expected=np.exp(1j*omega*time)
        reconstructed,check=products.ffi(time,-omega**2*expected,.04,taper=False)
        np.testing.assert_allclose(reconstructed,expected,rtol=1e-12,atol=1e-12)
        self.assertLess(check['derivative_filtered_relative_rms'],1e-4)
    def test_ffi_refuses_nonuniform_time(self):
        with self.assertRaises(ValueError):products.ffi(np.array([0.,1.,3.]),np.ones(3),.04)
    def test_exact_ids_and_horizon_geometry(self):
        path=Path(os.environ['PRODUCT_QA'])
        path.mkdir(parents=True,exist_ok=True)
        dtype=np.dtype([('tag','<i8'),('q','<f8',(8,))]);data=np.zeros(3,dtype=dtype)
        data['tag']=[0,3_000_000,4_000_000];data['q'][:,0]=1;data['q'][:,1]=1
        data['q'][:,2]=[0,1,-1]
        frame=path/'fixture.trk2'
        header=b'ATHENAK_TRK2 version=2 endian=native record_bytes=72 count=3 cycle=0 time=0\n'
        frame.write_bytes(header.ljust(512,b'\0')+data.tobytes())
        cycle,time,loaded=products.trk(frame)
        self.assertEqual(cycle,0);self.assertEqual(time,0)
        np.testing.assert_array_equal(loaded['tag'],data['tag'])
        row=dict(horizon=0,lmax=0,x=0.,y=0.,z=0.,coefficients_a0_ac_as=f'{.5*np.sqrt(4*np.pi)};0;0;')
        np.testing.assert_allclose(products.radius(row,np.array([0.,1.,2.]),0),.5)
        fig=plt.figure(figsize=(16,9),dpi=120,facecolor=products.BG)
        ax=fig.add_axes([0,0,1,1],projection='3d')
        result=products.draw_state(ax,loaded,[row],2,time)
        fig.savefig(path/'geometry_qa_1080p.png',facecolor=products.BG);plt.close(fig)
        self.assertEqual(result['visible'],[0,1,1])
        with Image.open(path/'geometry_qa_1080p.png') as image:
            self.assertEqual(image.size,(1920,1080))
        (path/'geometry_qa.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID'),'run checks in Slurm'
    unittest.main()
