"""Allocated early physical-diagnostic review inputs; no new science thresholds."""
import csv
import subprocess
import numpy as np
from cfg import *
from os_gate import require_release
import analyze as A

def measurements(rows):
    use=[r for r in rows if r['constraint_usable']]
    result=dict(rows=len(rows),valid_constraint_rows=len(use),
        endpoint_M=rows[-1]['time'],last_completed_dt_M=rows[-1]['dt_completed'],
        core_lapse_initial=rows[0]['core_lapse'],core_lapse_final=rows[-1]['core_lapse'],
        core_lapse_min=min(r['core_lapse']for r in rows),core_lapse_max=max(r['core_lapse']for r in rows))
    for key in ['H_rms_r10','M_rms_r10','H_rms_r16','M_rms_r16','E_mean_r10']:
        result[key]=dict(initial=use[0][key],last_valid=use[-1][key],
            minimum=min(r[key]for r in use),maximum=max(r[key]for r in use))
    return result

def main():
    require_job();require_release(ROOT)
    dest=ROOT/'preflight_health';dest.mkdir(exist_ok=False)
    summaries={};sources=[]
    for case in CASES:
        name=case['id'];base=ROOT/'evidence/amd/runs'/name/'preflight'
        assert load(ROOT/'evidence/amd/evidence/preflight'/(name+'.json'))['passed']
        manifest=load(base/'OUTPUT_MANIFEST.json')
        selected=[f for f in manifest['files'] if f['path']in ['out/'+name+'.user.hst','out/'+name+'.z4c.user.hst']]
        assert len(selected)==2
        local=dest/name;local.mkdir();paths=dest/(name+'.files.txt')
        paths.write_text(''.join(f['path']+'\n'for f in selected))
        subprocess.run(['rsync','-a','--files-from='+str(paths),f'hpcfund.amd.com:{AMD}/runs/{name}/preflight/',str(local)+'/'],check=True)
        for f in selected:
            p=local/f['path'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
            sources.append(dict(case=name,**f))
        h=A.history(local/'out'/(name+'.user.hst'))
        assert len(h)==11 and h[0]['time']==0 and abs(h[-1]['time']-20)<1e-7
        assert np.isfinite([[r[k]for k in r if k!='constraint_usable']for r in h]).all()
        assert all(r['core_lapse']>0 for r in h)
        ga=np.atleast_2d(np.loadtxt(local/'out'/(name+'.z4c.user.hst')))
        assert np.isfinite(ga).all() and np.array_equal(ga[:,0],[r['time']for r in h])
        summaries[name]=measurements(h)
    with (BASE/'reduced/health.csv').open()as stream:
        prefix=[r for r in csv.DictReader(stream)if float(r['time'])<=20.25]
    h=[{k:(str(v).lower()=='true'if k=='constraint_usable'else float(v))for k,v in r.items()}for r in prefix]
    summaries['ordinary_1p5m_s1']=measurements(h)
    atomic(ROOT/'evidence/PREFLIGHT_HEALTH_QA.json',dict(utc=now(),job=os.environ['SLURM_JOB_ID'],
        source=source(),passed=True,source_history_hashes_verified=True,measurements=summaries,files=sources,
        actual_agent_review_pending=True,
        scope='Finite native histories and existing diagnostics. No new physical stability pass threshold; compare actual short-run lapse/constraints before production.',
        constraint_rule='Keep t0; otherwise completed dt>1e-6. Tiny endpoint-step constraint rows excluded.'))
    print(json.dumps(summaries,indent=2),flush=True)
    print('PREFLIGHT_HEALTH_QA_OK: actual agent review still required',flush=True)

if __name__=='__main__':main()
