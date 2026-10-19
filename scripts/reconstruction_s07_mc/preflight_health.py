"""Allocated full-size short-run lapse/constraint review inputs, on AMD."""
import numpy as np
from cfg import *
import analyze as A

def main():
    require_job();summaries={}
    for case in CASES:
        work=ROOT/'runs'/case['id']/'preflight';receipt=load(ROOT/'evidence/preflight'/(case['id']+'.json'))
        assert receipt['passed']
        h=A.history(work/'out'/(case['id']+'.user.hst'))
        assert len(h)==11 and h[0]['time']==0 and abs(h[-1]['time']-20)<1e-7
        assert np.isfinite([[v for k,v in row.items()if k!='constraint_usable']for row in h]).all()
        assert all(row['core_lapse']>0 for row in h)
        valid=[r for r in h if r['constraint_usable']]
        rows=[json.loads(v)for v in (work/'memory_samples.jsonl').read_text().splitlines()]
        gpus=[v for r in rows for v in r['vram_used_bytes'].values()]
        summaries[case['id']]=dict(lapse_initial=h[0]['core_lapse'],lapse_final=h[-1]['core_lapse'],
            lapse_min=min(r['core_lapse']for r in h),lapse_max=max(r['core_lapse']for r in h),
            H_initial=valid[0]['H_rms_r10'],H_final=valid[-1]['H_rms_r10'],
            M_initial=valid[0]['M_rms_r10'],M_final=valid[-1]['M_rms_r10'],
            initial_E=valid[0]['E_mean_r10'],final_E=valid[-1]['E_mean_r10'],
            max_host_RSS=max(r['aggregate_host_rss_bytes']for r in rows),peak_gpu_bytes=max(gpus)if gpus else None,
            forecast_node_hours=receipt['forecast_1000M_node_hours'],restart_verified=receipt['restart_probe_passed'],
            input_sha256=sha(ROOT/'inputs'/(case['id']+'.athinput')),
            preflight_receipt_sha256=sha(ROOT/'evidence/preflight'/(case['id']+'.json')))
    atomic(ROOT/'evidence/PREFLIGHT_HEALTH_QA.json',dict(utc=now(),job=os.environ['SLURM_JOB_ID'],source=source(),
        passed=True,measurements=summaries,actual_agent_review_pending=True,native_data_host='AMD only'))
    print(json.dumps(summaries,indent=2),flush=True)
if __name__=='__main__':main()
