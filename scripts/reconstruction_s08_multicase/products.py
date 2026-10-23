"""Verified source packets -> production standalone PDFs in the Session8 style."""
import csv,shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from common import *
from reduce import csv_rows,write_csv
from publication_style import style,save,array,shell_limits,time_axis

def merged(rows):
    out={}
    for row in rows:
        key=round(float(row['time']),8)
        if key in out:
            for column,value in row.items():
                if isinstance(value,(int,float)) and column in out[key]:
                    assert np.isclose(value,out[key][column],rtol=1e-9,atol=2e-7,equal_nan=True),(key,column)
        out[key]=row
    return [out[t] for t in sorted(out)]

def verify(packet):
    manifest=load(packet/'MANIFEST.json')
    for entry in manifest['files']:
        p=packet/entry['relative'];assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
    return manifest

def assemble():
    job();data=[];bindings=[]
    for directory in ['reduced','plots','movies','reports','runs','evidence/samples']:(ROOT/directory).mkdir(parents=True,exist_ok=True)
    for raw in ALL_CASES:
        c=case_config(raw['id']);key=c['key'];packet=EXT/'imported'/key
        if not packet.exists():packet=EXT/key/'review_packet'
        manifest=verify(packet);accept=load(packet/'numerical_acceptance.json')
        assert accept['numerical_checks_passed'] and manifest['case']['key']==key
        c.update(id=key,mode=key,title=rf"$y_c={c['yc']:.3f}$, $U={c['U']:.3f}$")
        if c.get('inherited_time'):c['title']=r'$y_c=0.500$, unperturbed'
        destination=ROOT/'reduced'/key;destination.mkdir(exist_ok=True)
        prefixes=[]
        if c.get('inherited_time'):
            legacy=EXT/'imported/legacy500_packet';legacy_manifest=verify(legacy)
            assert legacy_manifest['anchor']['sha256']=='7309c1b16b29d5c3de300dadce3ab37be2ba73796ae92b78ed45681134ff2e64'
            prefixes=[legacy]
        parts=prefixes+[packet];table={}
        for name in ['shells','mass','history','ah']:
            table[name]=merged([row for part in parts for row in csv_rows(part/'reduced'/(name+'.csv'))])
            write_csv(destination/(name+'.csv'),table[name])
        index=[]
        for part in parts:
            with (part/'reduced/cohort_index.csv').open() as f:
                for row in csv.DictReader(f):
                    origin=row['file'];row.update(time=float(row['time']),cycle=int(row['cycle']),alive=int(row['alive']),source_file=origin,file=str(part/'cohort'/Path(origin).name));assert Path(row['file']).exists();index.append(row)
        table['cohort_index']=merged(index);write_csv(destination/'cohort_index.csv',table['cohort_index'])
        summary=load(packet/'reduced/summary.json');summary['case']=c
        assert summary['complete'] and table['shells'][0]['time']==0
        assert abs(table['history'][-1]['time']-summary['endpoint'])<1e-7
        atomic(destination/'summary.json',summary)
        shutil.copy2(packet/'sample.json',ROOT/'evidence/samples'/f'{key}.json')
        native=ROOT/'runs'/key;native.mkdir(exist_ok=True)
        original_id=manifest['case']['id'];shape=packet/'native'/f'{original_id}.horizon_shape_0.txt'
        target=native/f'{key}.horizon_shape_0.txt'
        if shape.exists() and not target.exists():target.symlink_to(shape)
        table.update(summary=summary,case=c,state=load(packet/'state.json'),source=packet,acceptance=accept)
        data.append(table);bindings.append(dict(case=key,packet=receipt(packet/'MANIFEST.json'),prior500=receipt(prefixes[0]/'MANIFEST.json') if prefixes else None,numerical_acceptance=receipt(packet/'numerical_acceptance.json')))
    atomic(ROOT/'evidence/ASSEMBLY_OK.json',dict(utc=now(),job=job(),sources=bindings,source_files_SHA256_verified=True,previous_acceptance_reused_only_for_0_to500=True))
    return data

def events(axes,s):
    for ax in axes:
        if s['first_removal'] is not None:ax.axvline(s['first_removal'],ls=':',c='.45',lw=.9)
        if s['ah_first'] is not None:ax.axvline(s['ah_first'],ls='--',c='.15',lw=.9)
        ax.grid(alpha=.15,lw=.5)

def figures(data):
    style();lim=shell_limits(*(d['shells'] for d in data));cm=plt.get_cmap('cividis');norm=Normalize(.1,.9)
    fig,ax=plt.subplots(2,len(data),figsize=(5*len(data),6),squeeze=False,layout='constrained',sharey='row')
    for j,d in enumerate(data):
        s=d['summary'];t=array(d['shells'],'time')
        for q in range(10,100,10):ax[0,j].plot(t,array(d['shells'],f'r{q}'),c=cm(norm(q/100)),lw=1)
        ax[0,j].set(yscale='log',ylim=lim,title=d['case']['title']);ax[1,j].plot(array(d['history'],'time'),array(d['history'],'core_lapse'),c='#275d7f',lw=1.2)
        ax[1,j].set(ylim=(0,1),xlabel=r'$t/M_{\rm ref}$')
        for a in ax[:,j]:time_axis(a,s['endpoint'])
        if d['case'].get('inherited_time'):
            for a in ax[:,j]:
                a.axvline(300,c='.55',ls='-.',lw=.8);a.axvline(500,c='.55',ls=':',lw=.8)
        events(ax[:,j],s)
    ax[0,0].set_ylabel(r'COM rest-mass shell radius $r_q/M_{\rm ref}$');ax[1,0].set_ylabel(r'Core lapse $\langle\alpha\rangle_{r<0.75M}$')
    fig.colorbar(ScalarMappable(norm=norm,cmap=cm),ax=ax[0,:],fraction=.025,pad=.02,ticks=[.1,.5,.9],label='Enclosed rest-mass fraction')
    save(fig,'s08_extension_shells_lapse')
    fig,ax=plt.subplots(2,len(data),figsize=(5*len(data),5.5),squeeze=False,layout='constrained',sharey='row')
    extent=max(1e-15,1.2*max(abs(r['closure']) for d in data for r in d['mass']))
    for j,d in enumerate(data):
        m=d['mass'];t=array(m,'time');m0=m[0]['m_initial']
        ax[0,j].plot(t,array(m,'m_alive')/m0,c='#275d7f',label='Alive')
        ax[0,j].plot(t,sum(array(m,'n_'+k) for k in ['sphere','lapse','horizon'])/2000000,c='#b35c26',label='Removed')
        ax[0,j].plot(t,array(m,'n_exit')/2000000,c='.45',ls=':',label='Escaped')
        ax[0,j].plot(t,1+array(m,'closure'),c='.1',ls='--',lw=.9,label='Accounted total')
        ax[0,j].set(title=d['case']['title'],ylim=(-.03,1.04));ax[1,j].plot(t,array(m,'closure'),c='#275d7f',lw=.9)
        ax[1,j].set(ylim=(-extent,extent),xlabel=r'$t/M_{\rm ref}$');ax[1,j].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        for a in ax[:,j]:time_axis(a,d['summary']['endpoint'])
        events(ax[:,j],d['summary'])
    ax[0,0].set_ylabel(r'Rest mass / initial $M_0$');ax[1,0].set_ylabel(r'$M_{0,{\rm accounted}}/M_0(0)-1$');ax[0,-1].legend(frameon=False,fontsize=8,loc='center')
    save(fig,'s08_extension_mass_conservation')
    fig,axes=plt.subplots(1,len(data),figsize=(5*len(data),3.6),squeeze=False,layout='constrained',sharey=True)
    accepted_masses=[r[k] for d in data for r in d['ah'] if r['accepted']==1 for k in ['mirr','mass']]
    ah_upper=max(1.15,1.08*max(accepted_masses)) if accepted_masses else 1.15
    for ax,d in zip(axes[0],data):
        ah=d['ah'];good=array(ah,'accepted')==1
        for key,c,ls,label in [('mirr','#275d7f','-',r'$M_{\rm irr}$'),('mass','#b35c26','--',r'$M_{\rm CH}$')]:
            values=array(ah,key);values[~good]=np.nan;ax.plot(array(ah,'time'),values,c=c,ls=ls,label=label)
        if not good.any():ax.text(.5,.5,'No accepted horizon',ha='center',transform=ax.transAxes)
        ax.set(title=d['case']['title'],xlabel=r'$t/M_{\rm ref}$',ylim=(0,ah_upper));time_axis(ax,d['summary']['endpoint']);events([ax],d['summary'])
    axes[0,0].set_ylabel(r'AH mass / $M_{\rm ref}$');axes[0,-1].legend(frameon=False)
    save(fig,'s08_extension_AH_masses')
    atomic(ROOT/'evidence/PLOTS_RENDERED.json',dict(job=job(),utc=now(),products=[receipt(p) for p in sorted((ROOT/'plots').glob('*.pdf'))],visual_review='pending actual inspection',x_axes='Each column uses its actual endpoint; time units common; do not compare column widths as durations'))

if __name__=='__main__':figures(assemble())
