"""Review already-audited partial data without promoting final products."""
import argparse,csv
import numpy as np
from common import *
from reduce import csv_rows
from run_segment import cohort
import movies
from plots import style,array
import matplotlib.pyplot as plt


def main():
    job();p=argparse.ArgumentParser();p.add_argument('case');a=p.parse_args()
    case=next(c for c in CASES if c['id']==a.case)
    reduced=ROOT/'reduced'/a.case;summary=load(reduced/'summary.json')
    assert (reduced/'REDUCTION_OK.json').exists()
    with (reduced/'cohort_index.csv').open() as f:index=list(csv.DictReader(f))
    times=np.array([float(r['time']) for r in index]);assert np.all(np.diff(times)>0)
    assert times[-1]<=summary['endpoint']+1e-8
    _,_,mask,initial=cohort(index[0]['file']);assert mask.all()
    d=dict(case=case,summary=summary,index=index,times=times,
        radius=np.linalg.norm(initial[:,1:4],axis=1),
        R=load(ROOT/'evidence/samples'/f'{a.case}.json')['Riso'],
        shell=csv_rows(reduced/'shells.csv'),history=csv_rows(reduced/'history.csv'),
        ah=csv_rows(reduced/'ah.csv') if case['unstable'] else [],
        shapes=movies.shapes(ROOT/'runs'/a.case/f'{a.case}.horizon_shape_0.txt'))
    out=ROOT/'review_previews'/f'{a.case}_{os.environ.get("SLURM_STEP_ID","batch")}_{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%S}'
    (out/'plots').mkdir(parents=True);(out/'movies').mkdir()
    style();fig,axes=plt.subplots(2,1,figsize=(6,5),sharex=True,layout='constrained')
    for q in range(10,100,10):axes[0].plot(array(d['shell'],'time'),array(d['shell'],f'r{q}'),color=plt.get_cmap('cividis')((q-10)/80),lw=1)
    axes[0].set(yscale='log',ylabel=r'COM shell radius $r_q/M$')
    axes[1].plot(array(d['history'],'time'),array(d['history'],'core_lapse'),color='#275d7f')
    axes[1].set(xlim=(0,summary['endpoint']),ylim=(0,1),xlabel=r'$t/M$',ylabel='Core lapse')
    fig.suptitle(f'PRELIMINARY: {a.case}, audited through {summary["endpoint"]:.1f}M')
    for ax in axes:ax.grid(alpha=.15)
    for ext in ['png','pdf']:fig.savefig(out/'plots'/f'pilot_shells_lapse.{ext}',dpi=150)
    plt.close(fig)
    # Render the actual exported PDF, in addition to checking its PNG counterpart.
    import pypdfium2
    pdf=pypdfium2.PdfDocument(str(out/'plots/pilot_shells_lapse.pdf'))
    pdf[0].render(scale=1.5).to_pil().save(out/'plots/pilot_pdf_render.png');pdf.close()
    fig=plt.figure(figsize=(10,6),layout='constrained');grid=fig.add_gridspec(2,2,width_ratios=[1.1,1])
    axes=[fig.add_subplot(grid[:,0]),fig.add_subplot(grid[0,1]),fig.add_subplot(grid[1,1])]
    panel=movies.panel(fig,axes,d)
    for ax in axes[1:]:ax.set_xlim(0,summary['endpoint'])
    fig.suptitle('PRELIMINARY pilot: fixed IDs and initial-radius colors')
    movies.ROOT=out
    product=movies.encode(a.case,fig,[(d,panel)],times)
    atomic(out/'PREVIEW.json',dict(job=job(),step=os.environ.get('SLURM_STEP_ID'),utc=now(),
      input_audit=load(reduced/'REDUCTION_OK.json'),endpoint=summary['endpoint'],movie=product,
      files=[receipt(f) for f in sorted((out/'plots').glob('*'))],visual_review='pending'))
    print(out)


if __name__=='__main__':main()
