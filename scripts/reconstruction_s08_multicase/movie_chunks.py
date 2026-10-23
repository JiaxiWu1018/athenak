"""Resumable, source-bound movie chunks and independent final full decode."""
import fcntl
import inspect
import sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from common import *
from movies import layout
from movie_helpers import panel,load_case,update,spatial_window,outline,equator_basis,review_indices,encode_frames
from publication_style import style,array,shell_limits,time_axis
from run_segment import cohort
from extension_resources import admission

def prepare(mode):
    job();admission();name,fig,states,times=layout(mode)
    assembly=load(ROOT/'evidence/ASSEMBLY_OK.json');keys={d['case']['id'] for d,p in states}
    bindings=[r for r in assembly['sources'] if r['case'] in keys]
    renderer='\n'.join(inspect.getsource(f) for f in [layout,panel,load_case,update,spatial_window,outline,equator_basis,style,array,shell_limits,time_axis,cohort])
    reductions={key:{f:sha(ROOT/'reduced'/key/f) for f in ['summary.json','shells.csv','history.csv','cohort_index.csv','ah.csv']} for key in keys}
    plan=dict(name=name,mode=mode,times=list(map(float,times)),source_bindings=bindings,reductions=reductions,
      renderer_sha256=hashlib.sha256(renderer.encode()).hexdigest(),matplotlib=matplotlib.__version__,numpy=np.__version__,dpi=120,fps=15)
    fingerprint=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    frames=EXT/'movie_frames'/f'{name}_{fingerprint[:20]}';frames.mkdir(parents=True,exist_ok=True)
    if (frames/'PLAN.json').exists():assert load(frames/'PLAN.json')['fingerprint']==fingerprint
    else:atomic(frames/'PLAN.json',dict(fingerprint=fingerprint,plan=plan,utc=now(),job=job()))
    expected={}
    for source in bindings:
        for field in ['packet','prior500']:
            value=source.get(field)
            if not value:continue
            manifest=Path(value['path']);assert sha(manifest)==value['sha256']
            for row in load(manifest)['files']:
                if row['relative'].startswith('cohort/'):
                    expected[str(manifest.parent/row['relative'])]=row['sha256']
    return name,fig,states,times,frames,fingerprint,expected

def render(mode,chunk,size=100):
    assert chunk>=0 and 1<=size<=150
    name,fig,states,times,frames,fingerprint,expected=prepare(mode)
    start=chunk*size;end=min(start+size,len(times));assert start<end
    (frames/'chunks').mkdir(exist_ok=True);(frames/'receipts').mkdir(exist_ok=True)
    with (frames/f'.chunk_{chunk:04d}.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        checked={}
        for d,p in states:
            for t in times[start:end]:
                if t>d['summary']['endpoint']+1e-8:continue
                i=max(0,np.searchsorted(d['times'],t+1e-10,side='right')-1);path=d['index'][i]['file']
                if path not in checked:
                    assert sha(path)==expected[path],path
                    checked[path]=expected[path]
        rows=[]
        for i in range(start,end):
            assert not (ROOT/'evidence/RESOURCE_STOP.json').exists(),'Resource guard stopped rendering; frames retained'
            target=frames/f'{i:06d}.png';receipt_path=frames/'receipts'/f'{i:06d}.json'
            if receipt_path.exists():
                old=load(receipt_path);assert old['fingerprint']==fingerprint and old['time']==float(times[i])
                assert target.stat().st_size==old['bytes'] and sha(target)==old['sha256']
                rows.append(old);continue
            if target.exists():
                archive=frames/'unreceipted';archive.mkdir(exist_ok=True)
                atomic(archive/f'{i:06d}_{job()}.json',receipt(target))
                target.rename(archive/f'{i:06d}_{job()}.png')
            for d,p in states:update(d,p,float(times[i]),mode=='comparison')
            if mode=='comparison':fig.suptitle(f'Session 8 extensions • synchronized time t/Mref = {times[i]:.1f}',fontsize=12)
            temporary=frames/f'{i:06d}.{job()}.tmp.png';fig.savefig(temporary,dpi=120,facecolor='white')
            r=receipt(temporary);temporary.replace(target);r['path']=str(target)
            r.update(frame=i,time=float(times[i]),fingerprint=fingerprint,job=job());atomic(receipt_path,r);rows.append(r)
        atomic(frames/'chunks'/f'{chunk:04d}.json',dict(utc=now(),job=job(),fingerprint=fingerprint,start=start,end=end,frames=rows,source_cohorts_verified=checked))
    plt.close(fig)
    print(json.dumps(dict(name=name,chunk=chunk,start=start,end=end,fingerprint=fingerprint,frames_directory=str(frames))))

def encode(mode):
    name,fig,states,times,frames,fingerprint,expected=prepare(mode);plt.close(fig)
    rows=[]
    for i,t in enumerate(times):
        r=load(frames/'receipts'/f'{i:06d}.json');target=frames/f'{i:06d}.png'
        assert r['fingerprint']==fingerprint and r['time']==float(t) and r['frame']==i
        assert target.stat().st_size==r['bytes'] and sha(target)==r['sha256'];rows.append(r)
    frame_manifest=frames/'INPUT_MANIFEST.json'
    if not frame_manifest.exists():atomic(frame_manifest,dict(utc=now(),job=job(),name=name,fingerprint=fingerprint,frames=rows,fps=15))
    else:assert load(frame_manifest)['fingerprint']==fingerprint
    marker=ROOT/'evidence'/f'MOVIE_{name}_OK.json'
    if marker.exists():
        prior=load(marker);assert prior['fingerprint']==fingerprint and prior['full_decode'] and sha(prior['path'])==prior['sha256'];return prior
    path=ROOT/'movies'/f'{name}.mp4'
    partial=[p for p in [path,*list((ROOT/'plots').glob(name+'_decoded_*.png')),*list((ROOT/'plots').glob(name+'_review_*.png'))] if p.exists()]
    if partial:
        archive=ROOT/'interrupted'/f'movie_{name}_{job()}_{time.time_ns()}';archive.mkdir(parents=True)
        atomic(archive/'BEFORE.json',[receipt(p) for p in partial])
        for p in partial:p.rename(archive/p.name)
    previews=review_indices(states,times,mode=='comparison')
    record=encode_frames(name,frames,states,times,previews,frame_manifest)
    record.update(utc=now(),job=job(),fingerprint=fingerprint,source_assembly=receipt(ROOT/'evidence/ASSEMBLY_OK.json'),actual_visual_acceptance=False)
    atomic(marker,record)
    return record

if __name__=='__main__':
    if sys.argv[1]=='render':render(sys.argv[2],int(sys.argv[3]),int(sys.argv[4]) if len(sys.argv)>4 else 100)
    else:
        assert sys.argv[1]=='encode';encode(sys.argv[2])
