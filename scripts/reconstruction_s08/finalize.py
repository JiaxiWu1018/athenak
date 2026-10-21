"""Allocated final numeric checks and provenance; visual approval remains explicit."""
import re
import numpy as np
from common import *
from reduce import csv_rows


def main():
    job()
    results=[]
    for case in CASES:
        work=ROOT/'runs'/case['id'];dest=ROOT/'reduced'/case['id']
        state=load(work/'state.json');summary=load(dest/'summary.json')
        assert state['complete'] and summary['complete']
        assert (dest/'REDUCTION_OK.json').exists()
        end=summary['endpoint'];onset=state['latest']['ah_onset']
        expected=min(300,onset+20) if case['unstable'] and onset>=0 else 300
        mass=csv_rows(dest/'mass.csv');history=csv_rows(dest/'history.csv')
        shells=csv_rows(dest/'shells.csv')
        # Native stop is evaluated after a completed step.
        dt=max(row['dt'] for row in mass)
        assert expected-1e-8<=end<=expected+dt+1e-8,(case['id'],end,expected)
        for name,rows in [('mass',mass),('history',history)]:
            assert abs(rows[0]['time'])<1e-8 and abs(rows[-1]['time']-end)<1e-7,(case['id'],name)
        assert abs(shells[0]['time'])<1e-8
        if summary['first_removal'] is not None:
            assert max(r['time'] for r in shells)<summary['first_removal']
        radii=np.array([[r[f'r{q}'] for q in range(10,100,10)] for r in shells])
        assert np.isfinite(radii).all() and (radii>0).all() and (np.diff(radii,axis=1)>=0).all()
        fallback=0;fatal=[]
        for segment in state['segments']:
            logfile=work/(str(segment['job'])+'.log')
            if not logfile.exists():continue
            with logfile.open(errors='replace') as stream:
                for line in stream:
                    match=re.search(r'cycle \d+: (\d+) of \d+ particles fell back',line)
                    if match:fallback+=int(match[1])
                    if '### FATAL' in line or re.search(r'search_fail=[1-9]',line):fatal.append(line.strip())
        assert not fatal,fatal[:5]
        result=dict(case=case['id'],endpoint=end,expected_endpoint=expected,
          max_mass_closure=summary['max_closure'],fallback_particle_steps=fallback,
          shell_radius_min=float(radii.min()),shell_radius_max=float(radii.max()),
          core_lapse_min=min(r['core_lapse'] for r in history),
          core_lapse_max=max(r['core_lapse'] for r in history),
          # Constraint values at a near-zero final dt are retained, not silently hidden.
          max_H_rms_r10=max(r['H_rms_r10'] for r in history),
          max_M_rms_r10=max(r['M_rms_r10'] for r in history),
          final_checkpoint=state['latest'])
        results.append(result)
    movies=load(ROOT/'evidence/MOVIES_RENDERED.json')
    assert len(movies['products'])==6 and all(p['full_decode'] for p in movies['products'])
    assert (ROOT/'evidence/PLOTS_RENDERED.json').exists()
    atomic(ROOT/'evidence/FINAL_NUMERIC_QA.json',dict(job=job(),utc=now(),cases=results,
       visual_review='pending; requires inspection of actual exported figures and movie frames'))
    # Do not hash mutable scheduler logs or active telemetry as immutable evidence.
    manifest=[]
    for directory in ['inputs','samples','runs','reduced','plots','movies','frames','reports']:
        for path in sorted((ROOT/directory).rglob('*')):
            if path.is_file():manifest.append(dict(**receipt(path),relative=str(path.relative_to(ROOT))))
    atomic(ROOT/'evidence/FINAL_DATA_MANIFEST.json',dict(job=job(),utc=now(),files=manifest,
       retention=f'Retained at {ROOT} for review; no deletion authorized by this manifest.',
       machine=os.environ.get('S8_MACHINE','amd')))
    du=lambda path:int(subprocess.check_output(['du','-s','-B1',str(path)],text=True).split()[0])
    storage=dict(job=job(),utc=now(),raw_root=str(ROOT),session_bytes=du(ROOT),
                 raw_on_perseus=os.environ.get('S8_MACHINE')=='perseus')
    if storage['raw_on_perseus']:
        import shutil
        storage['filesystem_free_bytes']=shutil.disk_usage(ROOT).free
        storage['amd_import_manifest']=receipt(ROOT/'evidence/PERSEUS_EXPORT.json')
        storage['amd_import_originals_retained']=True
    else:
        storage['whole_user_bytes']=du(ROOT.parent)
    atomic(ROOT/'evidence/FINAL_STORAGE.json',storage)
    # Deliberately do not create COMPLETE: scientific interpretation, visual review,
    # thin-mirror verification and final accounting are still agent obligations.


if __name__=='__main__':main()
