import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import hourly

class HourlySafety(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def test_snapshot_excludes_future_and_incomplete_records(self):
        source=self.root/'source.csv';out=self.root/'out.csv'
        source.write_bytes(b'cycle,time,count\n10,1.0,7\n11,1.1,8\n12,1.2,9\n13,1.3,')
        hourly.csv_snapshot(source,out,dict(time=1.1,cycle=11))
        with out.open() as f:rows=list(csv.DictReader(f))
        self.assertEqual([int(r['cycle']) for r in rows],[10,11])
        self.assertTrue(source.read_bytes().endswith(b'13,1.3,'))
    def test_event_state_time_controls_cutoff(self):
        source=self.root/'source.csv';out=self.root/'out.csv'
        source.write_text('cycle,time,state_cycle,state_time\n9,.9,10,1.0\n10,1.,11,1.1\n')
        hourly.csv_snapshot(source,out,dict(time=1.0,cycle=10))
        with out.open() as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['state_cycle'],'10')
    def test_schedule_is_hour_aligned_and_idempotent(self):
        with patch('hourly.time.time',return_value=3601.),patch('hourly.subprocess.run',return_value=__import__('subprocess').CompletedProcess([],0,'42\n','')) as call:
            value=hourly.schedule(self.root)
            self.assertEqual(value['due_epoch'],7200.)
            self.assertIn('--begin=now+3599seconds',call.call_args.args[0])
        with patch('hourly.active_job_state',return_value='PENDING'),patch('hourly.subprocess.run') as call:
            self.assertEqual(hourly.schedule(self.root)['job_id'],'42');call.assert_not_called()
    def test_frame_cache_reuses_physics_across_snapshot_paths(self):
        import products
        p=self.root/'cohort';p.write_bytes(b'immutable cohort')
        item=dict(path=str(p),cycle=10,time=1.,horizons=[dict(x=1.,coefficients_a0_ac_as='1;0;0;',source='hour1/events.csv',segment='segment_1')])
        before=products.frame_cache_key(item,4)
        item['horizons'][0]['source']='hour2/events.csv'
        self.assertEqual(before,products.frame_cache_key(item,4))
        item['horizons'][0]['x']=2.
        self.assertNotEqual(before,products.frame_cache_key(item,4))
        item['horizons'][0]['x']=1.;p.write_bytes(b'changed physical cohort')
        self.assertNotEqual(before,products.frame_cache_key(item,4))

    def test_cached_movie_frames_reuse_then_invalidate_geometry(self):
        import numpy as np
        import pandas as pd
        import products
        run=self.root/'run';(run/'trk').mkdir(parents=True)
        data=np.zeros(3,dtype=[('tag','<i8'),('q','<f8',(8,))])
        data['tag']=[0,3_000_000,4_000_000];data['q'][:,0:2]=1.
        header=b'ATHENAK_TRK2 version=2 endian=native record_bytes=72 count=3 cycle=0 time=0\n'.ljust(512,b'\0')
        (run/'trk/x.trk2').write_bytes(header+data.tobytes())
        events=pd.DataFrame([dict(state_cycle=0,state_time=0.,horizon=0,published=1,lmax=0,
            x=0.,y=0.,z=0.,coefficients_a0_ac_as=f'{.5*np.sqrt(4*np.pi)};0;0;',source='hour1')])
        cache=self.root/'cache'
        first=self.root/'first';second=self.root/'second';third=self.root/'third'
        for p in (first,second,third):p.mkdir()
        with patch.object(products,'draw_state',wraps=products.draw_state) as draw:
            products.render([run],events,first,None,True,cache)
            self.assertEqual(draw.call_count,6)
        events['source']='hour2'
        with patch.object(products,'draw_state',wraps=products.draw_state) as draw:
            products.render([run],events,second,None,True,cache)
            self.assertEqual(draw.call_count,4) # only the four-panel figure redraws
        self.assertEqual(len(list(cache.glob('*.png'))),2)
        events['x']=.1
        with patch.object(products,'draw_state',wraps=products.draw_state) as draw:
            products.render([run],events,third,None,True,cache)
            self.assertEqual(draw.call_count,6)
        self.assertEqual(len(list(cache.glob('*.png'))),4)

    def test_live_reduction_preserves_the_native_audit_anchor(self):
        import subprocess
        import sys
        import products
        scripts=Path(products.__file__).parent
        view=self.root/'view';out=self.root/'out';out.mkdir()
        subprocess.run([sys.executable,str(scripts/'product_fixture.py'),str(scripts),str(view)],check=True,stdout=subprocess.DEVNULL)
        native=dict(time=.75,cycle=3,particles=3,removed=[0,0,0,0])
        (view/'control/latest_checkpoint.json').write_text(json.dumps(native))
        sample=dict(time=75.,cycle=4,particles=3,removed=[0,0,0,0])
        (view/'control/HOURLY_CONTEXT.json').write_text(json.dumps(dict(sample=sample,source_root=str(view))))
        _,_,_,metrics=products.reduce(view,out)
        self.assertEqual(metrics['endpoint_checkpoint'],native)
        self.assertEqual(metrics['endpoint'],75.)
        self.assertEqual(metrics['state_kind'],'sampled live diagnostics')

if __name__=='__main__':unittest.main()
