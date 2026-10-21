"""Small correctness fixture, not an extra production/convergence trajectory."""
import numpy as np
from common import *
from checkpoint import validate
from run_segment import cohort

def main():
    job();work=ROOT/'tests'/f'native_{job()}';work.mkdir()
    atomic(ROOT/'evidence/active.json',dict(work=str(work),phase='native_checks',job=job()))
    results=[]
    def launch(label,args):
        command=['prun',str(ROOT/'protected/athena'),'--kokkos-map-device-id-by=mpi_rank','-d',str(work)]+args
        atomic(work/f'{label}.launch.json',dict(command=command,utc=now(),job=job()))
        with (work/f'{label}.log').open('w') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
        assert r.returncode==0,(label,r.returncode)
        text=(work/f'{label}.log').read_text();assert '### FATAL' not in text
        checkpoints=sorted((work/'rst').glob('*.rst'))
        d=validate(checkpoints[-1]);d.pop('parameters');results.append(dict(label=label,checkpoint=d))
        return d,text
    a,log=launch('capacity',['-i',str(ROOT/'inputs/frozen_ah.athinput'),
      'problem/production_frozen_ah_check=false','fastflow/num_horizons=0',
      'mesh_refinement/refinement=adaptive','mesh_refinement/num_levels=4',
      'mesh_refinement/ncycle_check=1','mesh_refinement/refinement_interval=1',
      'mesh_refinement/capacity_stop_fraction=0.05','z4c_amr/loehner_threshold=0.000001',
      'time/tlim=1','time/nlim=3','time/ndiag=1'])
    assert 'CLEAN_STOP: amr_capacity' in log and a['particle_count']==2000
    # Small 16^3-cell fixture only: allow its deliberately forced all-block split.
    b,log=launch('amr_restart',['-r',a['path'],'mesh_refinement/max_nmb_per_rank=512',
      'mesh_refinement/capacity_stop_fraction=0.9','time/nlim=2'])
    assert b['cycle']==2 and b['nblocks']>a['nblocks'] and b['particle_count']==2000
    (work/'REQUEST_STOP').write_text('requested_fixture_stop\n')
    c,log=launch('stop_restart',['-r',b['path'],'time/nlim=3'])
    assert c['cycle']==b['cycle'] and c['time']==b['time'] and 'CLEAN_STOP: requested_fixture_stop' in log
    (work/'REQUEST_STOP').unlink()
    d,log=launch('removal_restart',['-r',c['path'],'particles/excise_lapse=1.1',
      'mesh_refinement/ncycle_check=1000000','time/nlim=3'])
    assert d['particle_count']==0 and sum(d['removal_counts'])==2000 and d['cycle']==3
    t,cy,mask,values=cohort(sorted((work/'cohort').glob('*.s8c'))[-1])
    assert cy==3 and not mask.any()
    e,log=launch('zero_population_restart',['-r',d['path'],'time/nlim=4'])
    assert e['particle_count']==0 and sum(e['removal_counts'])==2000 and e['cycle']==4
    atomic(ROOT/'evidence/NATIVE_CHECKS_OK.json',dict(utc=now(),job=job(),results=results,
           checks=['AMR capacity clean stop','AMR growth and migration','restart','stop file',
                   'complete removal ledger','cohort death masks','zero-population restart']))
if __name__=='__main__':main()
