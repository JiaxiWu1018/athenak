"""Static initialization evidence; run only in a Perseus Slurm allocation."""
import csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from health import require_window,read_rows
ROOT=Path(sys.argv[1]);run=Path(sys.argv[2]);require_window(run,allow_stopped=False)
physical=read_rows(next((run/'out').glob('*.plummer_physical.csv')))
physical={int(r['bin']):{k:float(v) for k,v in r.items()} for r in physical if float(r['time'])==0}
reference=list(csv.reader((ROOT/'initial_data/cpp_profile.csv').open()))
shells=np.array([list(map(float,r[2:])) for r in reference if r[0]=='shell' and float(r[1])==1000])
F=np.array([list(map(float,r[2:])) for r in reference if r[0]=='F' and float(r[1])==1000])
mom=np.array([list(map(float,r[2:])) for r in reference if r[0]=='moment' and float(r[1])==1000])
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
colors={'M0':'#17649a','E':'#9a671b','Sr':'#17649a','Stheta':'#be6d28','Sphi':'#726182'}
fig,axes=plt.subplots(2,2,figsize=(10.5,7.5),layout='constrained')
ax=axes[0,0]
ax.plot(mom[:,0],mom[:,1],'-o',label=r'$\psi$');ax.plot(mom[:,0],mom[:,2],'--s',label=r'$\alpha$')
ax.set_xscale('symlog',linthresh=.1)
ax.set(xlabel='Isotropic radius r / M',ylabel='Analytic metric function');ax.legend()
ax=axes[0,1];ax.loglog(F[:,0],F[:,1],color=colors['M0'])
ax.set(xlabel=r'Binding energy $1-e$',ylabel=r'$\mu F(e)/\epsilon_*$',title='Regularized Abel reconstruction')
ax=axes[1,0]
radius=np.sqrt(np.maximum(shells[:,1],.01)*shells[:,2])
for quantity,column in [('M0',3),('E',4)]:
    expected=shells[:,column]/shells[:,6]
    actual=np.array([physical[b][quantity] for b in range(64)])/shells[:,6]
    valid=expected>0
    ax.loglog(radius[valid],expected[valid],color=colors[quantity],label=quantity+' continuum')
    ax.loglog(radius[valid],actual[valid],'.',color=colors[quantity])
ax.set(xlabel='Areal shell radius / M',ylabel='Proper-volume shell density',title='Full-N initial density and energy');ax.legend()
ax=axes[1,1]
for quantity,marker in [('Sr','o'),('Stheta','s'),('Sphi','^')]:
    vals=np.array([physical[b][quantity] for b in range(64)]);N=np.array([physical[b]['N'] for b in range(64)])
    squares=np.array([physical[b][quantity+'_square'] for b in range(64)])
    sem=np.sqrt(np.maximum(0,2*(squares-vals*vals/np.maximum(N,1))))
    keep=(N>=200)&(shells[:,5]>0)
    ax.errorbar(radius[keep],100*(vals[keep]/shells[keep,5]-1),100*sem[keep]/shells[keep,5],
       fmt=marker,ms=3,alpha=.8,color=colors[quantity],label=quantity)
ax.axhline(0,color='.25',lw=.8);ax.axhspan(-2,2,color='.85',alpha=.5,label='±2% reference')
ax.set(xscale='log',xlabel='Areal shell radius / M',ylabel='Stress minus continuum / %',title='Pair-aware 1σ sampling uncertainty');ax.legend(ncol=2,fontsize=8)
fig.suptitle('Isotropic Plummer initialization · η = 0.10 · N = 2,113,536 · seed 1985')
ROOT.joinpath('figures').mkdir(exist_ok=True)
for extension in ('png','pdf'):fig.savefig(ROOT/'figures'/('initialization_fidelity.'+extension))
plt.close(fig)
(ROOT/'figures/initialization_fidelity.json').write_text(json.dumps(dict(source_run=str(run),
   continuum=str(ROOT/'initial_data/cpp_profile.csv'),time=0,N=2113536,uncertainty='1 sigma; N/2 correlated initial pairs',
   scope='Initialization only; no equilibrium or stability claim',shell_density_volume='analytic continuum proper shell volume'),indent=2)+'\n')
