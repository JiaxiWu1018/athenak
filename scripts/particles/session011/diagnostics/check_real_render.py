"""Exact-cycle, fixed-ID render integration check on audited production data."""
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import proj3d
root=Path(os.environ['SESSION_ROOT'])
tooling=Path(os.environ.get('TOOLING_DIR',str(root/'code/scripts/particles/session011')))
sys.path.insert(0,str(tooling))
import products
products.require_compute()
build=json.loads((root/'evidence/analysis_build_473989/RESULT.json').read_text())
assert build['status']=='build_and_python_tests_pass'
expected=os.environ.get('RENDERER_SHA256',build['tooling_sha256']['products.py'])
assert products.sha256(tooling/'products.py')==expected
run=root/'runs/segment_0003';audit=json.loads((run/'population_audit.json').read_text())
assert audit['exact_partition'] and audit['cycle']==1902
out=root/'evidence'/f'real_render_{os.environ["SLURM_JOB_ID"]}';out.mkdir()
events=products.load_csvs([run],'*.ah_events_*.csv',['state_cycle','horizon'])
events=events[events.published==1]
selected=None
for frame in sorted((run/'trk').glob('*.trk2'),reverse=True):
    cycle,time,data=products.trk(frame)
    hs=events[(events.state_cycle==cycle)&events.horizon.isin([0,1])]
    if set(hs.horizon)=={0,1}:
        assert cycle<=audit['cycle'] and abs(float(hs.state_time.max())-time)<1.e-10
        assert np.all(np.abs(hs.state_time.to_numpy()-time)<1.e-10)
        selected=(frame,cycle,time,data,hs);break
assert selected is not None,'no exact-cycle cohort with both accepted horizons'
frame,cycle,time,data,hs=selected
initial_path=sorted((root/'runs/segment_0000/trk').glob('*.trk2'))[0]
initial_cycle,initial_time,initial=products.trk(initial_path)
assert initial_cycle==0 and initial_time==0
initial=initial[np.argsort(initial['tag'])]
where=np.searchsorted(initial['tag'],data['tag']);assert np.all(where<len(initial))
assert np.array_equal(initial['tag'][where],data['tag'])
assert np.array_equal(initial['q'][where,0],data['q'][:,0]),'cohort rest weights changed'
before=hashlib.sha256(data.tobytes()).hexdigest()
horizons=hs.to_dict('records');hs.to_csv(out/'accepted_horizons.csv',index=False)
results=[]
for name,limit in [('wide',24),('close',4)]:
    fig=plt.figure(figsize=(16,9),dpi=120,facecolor=products.BG)
    ax=fig.add_axes([0,0,1,1],projection='3d')
    details=products.draw_state(ax,data,horizons,limit,time)
    ax.text2D(.025,.035,'Validation preview; exact AH geometry overlaid above particles',transform=ax.transAxes,color='white',fontsize=10)
    p=out/f'{name}_1080p.png';fig.savefig(p,facecolor=products.BG)
    with Image.open(p) as im:
        assert im.size==(1920,1080)
        pixels=np.asarray(im.convert('RGB'))
    black=np.max(pixels,axis=2)<12
    regions=np.zeros(black.shape,dtype=bool);visibility=[]
    theta,phi=np.meshgrid(np.linspace(0,np.pi,73),np.linspace(0,2*np.pi,145),indexing='ij')
    for h in horizons:
        r=products.radius(h,theta,phi)
        xyz=(h['x']+r*np.sin(theta)*np.cos(phi),h['y']+r*np.sin(theta)*np.sin(phi),h['z']+r*np.cos(theta))
        xp,yp,_=proj3d.proj_transform(*(a.ravel() for a in xyz),ax.get_proj())
        screen=ax.transData.transform(np.column_stack([xp,yp]));screen[:,1]=1080-screen[:,1]
        lo=np.floor(screen.min(axis=0)).astype(int)-2;hi=np.ceil(screen.max(axis=0)).astype(int)+2
        assert np.all(lo>=0) and hi[0]<1920 and hi[1]<1080
        patch=black[lo[1]:hi[1]+1,lo[0]:hi[0]+1];count=int(patch.sum())
        if name=='close':assert count>0,f'accepted AH {h["horizon"]} invisible in close view'
        regions[lo[1]:hi[1]+1,lo[0]:hi[0]+1]=True
        visibility.append(dict(horizon=int(h['horizon']),near_black_pixels=count,projected_bounds=[lo.tolist(),hi.tolist()]))
    assert not np.any(black&~regions),'black pixels outside exact projected surface bounds'
    plt.close(fig)
    assert details['accepted_horizons']==[0,1]
    results.append(dict(view=name,surface_visibility=visibility,**details))
assert hashlib.sha256(data.tobytes()).hexdigest()==before
record=dict(status='pass',job_id=os.environ['SLURM_JOB_ID'],evolution_advances=0,cycle=cycle,time=time,
 initial_cohort=str(initial_path),initial_cohort_sha256=products.sha256(initial_path),
 source_cohort=str(frame),source_cohort_sha256=products.sha256(frame),
 exact_initial_ID_subset=True,exact_rest_weights=True,render_did_not_mutate_records=True,
 paired_horizons_at_exact_cycle=True,renderer_sha256=expected,source=os.environ.get('SOURCE_COMMIT',build['source']),views=results)
(out/'RESULT.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
manifest=[dict(path=p.name,bytes=p.stat().st_size,sha256=products.sha256(p)) for p in sorted(out.iterdir()) if p.is_file()]
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(record,indent=2))
