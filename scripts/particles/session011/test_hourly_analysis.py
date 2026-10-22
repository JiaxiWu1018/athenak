import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import hourly_analysis

class HourlyAnalysisChecks(unittest.TestCase):
    def test_missing_measurements_stay_unknown(self):
        self.assertIsNone(hourly_analysis.finite(np.nan))
        self.assertIsNone(hourly_analysis.finite(np.inf))
        self.assertEqual(hourly_analysis.finite(0.),0.)
    def test_actual_pair_domains_and_resolved_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            metrics=dict(endpoint=2.,endpoint_checkpoint=dict(time=1.),first_individual_times=[0.,1.])
            (out/'metrics.json').write_text(json.dumps(metrics))
            grid=[]
            for cycle in range(3):
                for mask in (0,1):
                    value=(cycle+1.)**2*4 if mask==0 else [4.,np.nan,.25][cycle]
                    grid.append(dict(cycle=cycle,time=float(cycle),mask=mask,shell=0,
                                     H2_volume=value,M2_volume=value/4,proper_volume=1. if np.isfinite(value) else np.nan))
            pd.DataFrame(grid).to_csv(out/'metric_shells.csv',index=False)
            pd.DataFrame(dict(cycle=[0,1,2],time=[0.,1.,2.],wake_quadrupole_real=[1.,2.,3.],
                              wake_quadrupole_imag=[0.,np.nan,.3])).to_csv(out/'envelope_wake_moments.csv',index=False)
            pd.DataFrame(dict(cycle=[0,1,2],time=[0.,1.,2.],component=[0,0,0],mask=[1,1,1],Jz=[0.,.1,.2])).to_csv(out/'component_ledgers.csv',index=False)
            pd.DataFrame(dict(cycle=[0,1,2],time=[0.,1.,2.],evolving_env_omega2_proxy=[1.,.8,.6],
                frozen_env_omega2_proxy=[1.,1.,1.],measured_omega2=[1.,.4,.2])).to_csv(out/'envelope_support_proxy.csv',index=False)
            pd.DataFrame(dict(cycle=[0,1,2],time=[0.,1.,2.],tangential_speed_resolved=[True,False,True])).to_csv(out/'orbit.csv',index=False)
            pd.DataFrame([dict(state_cycle=0,state_time=0.,horizon=0),dict(state_cycle=1,state_time=1.,horizon=1),
                          dict(state_cycle=2,state_time=2.,horizon=0),dict(state_cycle=2,state_time=2.,horizon=1)]).to_csv(out/'accepted_horizons.csv',index=False)
            before={p.name:p.read_bytes() for p in out.iterdir()}
            result=hourly_analysis.analyze(out)
            self.assertEqual(result['first_pair_time'],2.)
            self.assertEqual(result['constraint_domains']['0']['H_RMS'],6.)
            self.assertEqual(result['constraint_domains']['1']['H_RMS'],.5)
            self.assertEqual(result['resolved_measured_support_samples'],2)
            self.assertEqual(result['native_audited_time'],1.)
            self.assertEqual(result['diagnostic_time'],2.)
            for name,data in before.items():self.assertEqual((out/name).read_bytes(),data)
            for name in ('constraint_domain_comparison','envelope_wake_and_Jz','envelope_support_comparison'):
                self.assertTrue((out/(name+'.png')).exists());self.assertTrue((out/(name+'.pdf')).exists())
            self.assertFalse(result['production_modified'])
            self.assertIn('not a flux-corrected torque',(out/'ANALYSIS.md').read_text())

if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID'),'run in Slurm'
    unittest.main()
