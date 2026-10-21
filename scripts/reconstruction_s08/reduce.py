"""Streaming production reductions and cross-checks against native ID/death records."""
import argparse,csv,re
import numpy as np
from common import *
from run_segment import cohort
from reduce_run import read_pvtk

def write_csv(path,rows):
    assert rows
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def unique_rows(rows,key='time'):
    out={}
    for r in rows:
        t=r[key]
        if t in out:
            for k in r:
                if isinstance(r[k],(int,float)):
                    assert np.isclose(r[k],out[t][k],rtol=1e-10,atol=1e-12,equal_nan=True),(t,k)
        out[t]=r
    return [out[k] for k in sorted(out)]

def csv_rows(path):
    with Path(path).open() as f:return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]

def reduce_case(c,require_complete=True):
    job();work=ROOT/'runs'/c['id'];state=load(work/'state.json')
    if require_complete:assert state['complete']
    dest=ROOT/'reduced'/c['id'];dest.mkdir(parents=True,exist_ok=True)
    n0=2000000;death_cycle=np.full(n0,np.iinfo(np.int64).max,dtype=np.int64)
    death_time=np.full(n0,np.inf);reason=np.zeros(n0,dtype=np.uint8)
    names=['exit','sphere','lapse','horizon'];log=work/f"{c['id']}.prtcl_destroy.csv"
    if log.exists():
        with log.open() as f:
            for line in f:
                if line.startswith('#'):continue
                r=line.split(',');tag=int(r[2]);assert 0<=tag<n0 and not reason[tag]
                death_cycle[tag]=int(r[0]);death_time[tag]=float(r[1]);reason[tag]=names.index(r[3])+1
    first_removal=float(death_time.min());counts=[int(np.sum(reason==i)) for i in range(1,5)]
    ledger=unique_rows(csv_rows(work/f"{c['id']}.mass.csv"))
    for row in ledger:
        dead=death_cycle<int(row['cycle']);assert row['n_alive']==n0-int(dead.sum())
        for i,name in enumerate(names,1):assert row['n_'+name]==int(np.sum(dead&(reason==i)))
        assert abs(row['closure'])<1e-10 and row['weight_error']<1e-10
    write_csv(dest/'mass.csv',ledger)
    shells=[];snapshots=[]
    for file in sorted((work/'pvtk').glob('*.vtk')):
        t,cycle,n,fields=read_pvtk(str(file));tags=fields['ptag'].astype(np.int64)
        assert np.array_equal(tags,fields['ptag']) and len(np.unique(tags))==n
        expected=np.flatnonzero(death_cycle>=cycle);assert np.array_equal(np.sort(tags),expected)
        assert all(np.isfinite(v).all() for v in fields.values())
        mass=fields['prtcl_mass'];m0=ledger[0]['m_initial'];weight=m0/n0
        assert n==0 or np.all(mass==np.float32(weight))
        snapshots.append(dict(file=str(file),time=t,cycle=cycle,n=n,bytes=file.stat().st_size))
        if t>=first_removal or n==0:continue
        pos=np.column_stack([fields[a] for a in 'xyz']);com=np.average(pos,axis=0,weights=mass)
        rad=np.linalg.norm(pos-com,axis=1);order=np.argsort(rad)
        cumulative=np.cumsum(mass[order]);cumulative/=cumulative[-1]
        q=rad[order[np.searchsorted(cumulative,np.arange(1,10)/10)]]
        row=dict(time=t,cycle=cycle,n=n,com_x=com[0],com_y=com[1],com_z=com[2])
        row.update({f'r{i*10}':v for i,v in enumerate(q,1)});shells.append(row)
    shells=unique_rows(shells);write_csv(dest/'shells.csv',shells)
    hist=np.atleast_2d(np.loadtxt(work/f"{c['id']}.user.hst"));assert hist.shape[1]==19
    histories=[]
    for row in hist:
        assert np.isfinite(row).all() and row[14]>0
        histories.append(dict(time=row[0],dt=row[1],core_lapse=row[15]/row[14],
             outer_lapse=row[17]/row[16],H_rms_r10=np.sqrt(max(row[3],0)/row[2]),
             M_rms_r10=np.sqrt(max(row[4],0)/row[2]),dt_completed=row[18]))
    # A restart recomputes constraints, so duplicate histories may legitimately
    # differ in constraint values; require lapse continuity and retain the last.
    hmap={}
    for row in histories:
        if row['time'] in hmap:assert abs(row['core_lapse']-hmap[row['time']]['core_lapse'])<1e-9
        hmap[row['time']]=row
    histories=[hmap[t] for t in sorted(hmap)];write_csv(dest/'history.csv',histories)
    cohort_index=[]
    for file in sorted((work/'cohort').glob('*.s8c')):
        t,cycle,mask,values=cohort(file)
        assert np.array_equal(mask.astype(bool),death_cycle[::100]>=cycle)
        matching=[s for s in snapshots if s['cycle']==cycle]
        if matching:
            _,_,_,fields=read_pvtk(matching[-1]['file']);order=np.argsort(fields['ptag'])
            ids=fields['ptag'][order].astype(int);use=ids%100==0;ids=ids[use]
            for k,col in [('x',1),('y',2),('z',3),('ux',4),('uy',5),('uz',6),('prtcl_mass',7)]:
                assert np.array_equal(fields[k][order][use],values[ids//100,col].astype(np.float32))
        cohort_index.append(dict(time=t,cycle=cycle,file=str(file),alive=int(mask.sum())))
    write_csv(dest/'cohort_index.csv',cohort_index)
    summary=dict(case=c,endpoint=state['latest']['time'],complete=state['complete'],
       first_removal=None if not np.isfinite(first_removal) else first_removal,
       removals=dict(zip(names,counts)),rest_mass_initial=ledger[0]['m_initial'],
       max_closure=max(abs(r['closure']) for r in ledger),snapshots=len(snapshots),
       cohorts=len(cohort_index),ah_first=None,ah_endpoint_window=None)
    if c['unstable']:
        ah=csv_rows(work/f"{c['id']}.ah_quality_0.csv");write_csv(dest/'ah.csv',ah)
        good=[r for r in ah if r['accepted']==1]
        for r in good:
            assert r['missing']==0 and r['streak']>=3 and r['hrel']<=.05
            assert r['rmin']>=4*r['dx'] and r['area']>0 and np.isfinite([r['mass'],r['mirr']]).all()
            assert np.isclose(r['mirr'],np.sqrt(r['area']/(16*np.pi)),rtol=1e-12)
        if good:
            summary['ah_first']=good[0]['time'];tail=[r for r in good if r['time']>=summary['endpoint']-5]
            if tail:
                summary['ah_endpoint_window']={key:dict(median=float(np.median([r[key] for r in tail])),
                  min=min(r[key] for r in tail),max=max(r[key] for r in tail)) for key in ['mirr','mass']}
                summary['ah_endpoint_window'].update(start=tail[0]['time'],end=tail[-1]['time'],samples=len(tail))
    atomic(dest/'summary.json',summary);atomic(dest/'snapshots.json',snapshots)
    atomic(dest/'REDUCTION_OK.json',dict(utc=now(),job=job(),population_ids=True,removal_ids=True,
             mass_ledger=True,cohort_vs_full_snapshots=True,shell_cutoff='first removal'))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--partial',action='store_true');a=p.parse_args()
    print(json.dumps(reduce_case(next(c for c in CASES if c['id']==a.case),not a.partial),indent=2))
