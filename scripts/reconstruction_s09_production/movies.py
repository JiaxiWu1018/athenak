"""Individual or synchronized production movie; all rendering remains in Slurm."""
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from common import *
from movie_helpers import load_case,panel,encode
from publication_style import style,time_axis

def layout(mode):
    style();assert load(ROOT/'evidence/ASSEMBLY_OK.json')['source_files_SHA256_verified']
    assert mode=='comparison' or mode in [c['id'] for c in PLOT_CASES]
    name='s09_comparison' if mode=='comparison' else mode
    cases=[load(ROOT/'reduced'/r['id']/'summary.json')['case'] for r in PLOT_CASES if mode=='comparison' or r['id']==mode]
    data=[load_case(c) for c in cases];states=[]
    if mode=='comparison':
        end=max(d['summary']['endpoint'] for d in data);fig=plt.figure(figsize=(5*len(data),8),layout='constrained');grid=fig.add_gridspec(3,len(data),height_ratios=[2,1,1]);spatial=[]
        for j,d in enumerate(data):
            axes=[fig.add_subplot(grid[k,j]) for k in range(3)];p=panel(fig,axes,d);axes[0].set_title(d['case']['title']);states.append((d,p));spatial.append(axes[0])
            for axis in axes[1:]:time_axis(axis,end)
        fig.colorbar(states[0][1]['scatter'],ax=spatial,fraction=.02,pad=.02,label=r'Initial radius $r_0/R_{\rm iso}(0)$')
        times=np.arange(0,end+1e-8,.5)
        if times[-1]<end-1e-8:times=np.r_[times,end]
    else:
        d=data[0];fig=plt.figure(figsize=(10,6),layout='constrained');grid=fig.add_gridspec(2,2,width_ratios=[1.1,1]);axes=[fig.add_subplot(grid[:,0]),fig.add_subplot(grid[0,1]),fig.add_subplot(grid[1,1])]
        p=panel(fig,axes,d);axes[0].set_title(d['case']['title']);states=[(d,p)];times=d['times']
        fig.suptitle('Session 9 • fixed initial particle IDs • accepted AH contours',fontsize=11)
        fig.colorbar(p['scatter'],ax=axes[0],fraction=.035,pad=.025,label=r'Initial radius $r_0/R_{\rm iso}(0)$')
    return name,fig,states,times

def main():
    job();style();assert load(ROOT/'evidence/ASSEMBLY_OK.json')['source_files_SHA256_verified']
    mode=sys.argv[1];assert mode=='comparison' or mode in [c['id'] for c in PLOT_CASES]
    name='s09_comparison' if mode=='comparison' else mode
    marker=ROOT/'evidence'/f'MOVIE_{name}_OK.json'
    if marker.exists():
        prior=load(marker);assert sha(prior['path'])==prior['sha256'] and prior['full_decode'];return
    # Preserve any failed encoded attempt and its representative PNGs before a
    # rerender. Existing completed accepted movies are returned above unchanged.
    prior=[p for p in [ROOT/'movies'/f'{name}.mp4',*list((ROOT/'plots').glob(name+'_*review_*.png')),*list((ROOT/'plots').glob(name+'_decoded_*.png'))] if p.exists()]
    if prior:
        archive=ROOT/'interrupted'/f'movie_{name}_{job()}_{time.time_ns()}';archive.mkdir(parents=True)
        atomic(archive/'BEFORE.json',[receipt(p) for p in prior])
        for p in prior:p.rename(archive/p.name)
    name,fig,states,times=layout(mode)
    record=encode(name,fig,states,times,sync=mode=='comparison')
    record.update(utc=now(),job=job(),source_assembly=receipt(ROOT/'evidence/ASSEMBLY_OK.json'),actual_visual_acceptance=False)
    atomic(marker,record)

if __name__=='__main__':main()
