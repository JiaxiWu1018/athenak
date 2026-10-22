"""Regressions exposed by completed native data, without evolving a trajectory."""
import os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import products

class RealDataRegressions(unittest.TestCase):
    def test_zero_tangential_motion_does_not_make_finite_ratio(self):
        t=np.arange(5.)*.025;d=6.+t
        for omega in (np.zeros(5),np.full(5,1e-17)):
            radial,tangential,resolved,ratio=products.radial_motion(t,d,omega)
            np.testing.assert_allclose(radial,1.)
            self.assertFalse(resolved.any());self.assertTrue(np.isnan(ratio).all())

    def test_resolved_radial_motion_retains_analytic_ratio(self):
        t=np.arange(5.)*.025;d=6.+t
        radial,tangential,resolved,ratio=products.radial_motion(t,d,np.full(5,.2))
        self.assertTrue(resolved.all());np.testing.assert_allclose(ratio,1/(.2*d))

    def test_cutoff_legend_and_colors_after_short_initial_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';wave=run/'waveforms';wave.mkdir(parents=True)
            out=root/'out';out.mkdir()
            time=np.r_[np.arange(3)*.025,1+np.arange(128)*.025,6+np.arange(128)*.025]
            data=np.zeros((len(time),6));data[:,0]=time;data[:,5]=np.cos(2*np.pi*.25*time)
            np.savetxt(wave/'rpsi4_real_0040.txt',data)
            data[:,5]=np.sin(2*np.pi*.25*time);np.savetxt(wave/'rpsi4_imag_0040.txt',data)
            pd.DataFrame(dict(legacy_time=time,state_time=time,state_cycle=np.arange(len(time)))).to_csv(wave/'state_times.csv',index=False)
            saved=products.savefig;observed=[]
            def inspect(fig,path):
                if path.name=='provisional_strain':
                    handles,labels=fig.axes[0].get_legend_handles_labels()
                    observed.append(labels)
                    self.assertEqual(labels,['f0=0.01','f0=0.02','f0=0.04'])
                    self.assertEqual([line.get_color() for line in fig.axes[0].lines],['C0','C1','C2']*2)
                saved(fig,path)
            with patch.object(products,'savefig',side_effect=inspect):products.waves([run],out)
            self.assertEqual(len(observed),1)

if __name__=='__main__':
    products.require_compute()
    unittest.main()
