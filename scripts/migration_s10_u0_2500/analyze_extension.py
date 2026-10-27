"""Allocated final particle audit and updated0-endpoint shell/central-lapse figures."""
import csv,shutil
import numpy as np
from common import *
from reduce import reduce_case,csv_rows,write_csv
from case_review import review
from publication_style import style

def joined(prefix,tail):
    older=csv_rows(prefix);newer=csv_rows(tail);assert older[-1]['time']==1500 and newer[0]['time']>=1500-1e-7;rows={v['time']:v for v in older}
    for row in newer:
        if row['time'] in rows:
            for key in ['core_lapse'] if 'core_lapse' in row else [k for k in row if k.startswith('r')]:assert abs(row[key]-rows[row['time']][key])<1e-9,(key,row['time'])
        rows[row['time']]=row
    return [rows[t] for t in sorted(rows)]

def main():
    job();state=load(ROOT/'runs/yc0500/state.json');assert state['complete'];summary=reduce_case(CASE);checked=review(CASE);assert checked['numerical_checks_passed'];red=ROOT/'reduced/yc0500';out=EXT/'delivery'/('control_to2500_'+job());assert not out.exists();out.mkdir(parents=True)
    prefix=EXT/'reference/control_0_1500';pm=load(prefix/'SOURCE_MANIFEST.json')
    for x in pm['copied']:assert receipt(x['path'])==x
    shell=joined(prefix/'shells.csv',red/'shells.csv');history=joined(prefix/'history.csv',red/'history.csv');write_csv(out/'shells_0_endpoint.csv',shell);write_csv(out/'lapse_history_0_endpoint.csv',history)
    from origin_lapse import extract
    paths={}
    for folder in [BASE/'runs/yc0500/rst',DONOR/'runs/yc0500/rst',PARENT/'yc0500_control/runs/yc0500/rst',ROOT/'runs/yc0500/rst']:
        for p in sorted(folder.glob('*.rst')):
            # Header-only eligibility pass; extract/hash selected scientific states in Slurm.
            with p.open('rb') as f:
                data=b''
                while not data.endswith(b'<par_end>\n'):
                    line=f.readline();assert line and len(data)<2**20;data+=line
                off=f.tell();f.seek(off+232);import struct;t,dt,cycle=struct.unpack('<ddi',f.read(20))
            if t<=state['latest']['time']+1e-7 and (abs(t/100-round(t/100))<1e-7 or abs(t-state['latest']['time'])<1e-7):paths[t]=p
    centers=[extract(paths[t]) for t in sorted(paths)];assert any(abs(x['time']-1500)<1e-7 for x in centers) and abs(centers[-1]['time']-state['latest']['time'])<1e-7
    atomic(out/'CENTRAL_LAPSE_SOURCES.json',dict(utc=now(),job=job(),source_enum=receipt(EXT/'src/athenak/src/z4c/z4c.hpp'),measurements=centers));write_csv(out/'central_lapse_checkpoints.csv',[dict(time=x['time'],cycle=x['cycle'],central_lapse=x['central_lapse']) for x in centers])
    style();import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    for q,color in zip(range(10,100,10),plt.cm.viridis(np.linspace(.08,.92,9))):ax.plot([v['time'] for v in shell],[v[f'r{q}'] for v in shell],color=color,label=f'{q}%')
    ax.axvline(1500,color='.4',ls=':',lw=1,label='1500M continuation');ax.set(yscale='log',xlim=(0,2500),xlabel='t / M_ref',ylabel='COM coordinate rest-mass shell radius [M]',title=f"Unperturbed corrected yc=0.500: actual endpoint {summary['endpoint']:g}M");ax.legend(ncol=5,fontsize=8);ax.grid(alpha=.18)
    for ext in ['png','pdf']:fig.savefig(out/('mass_shells_to2500.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,7),layout='constrained',sharex=True)
    axes[0].plot([v['time'] for v in centers],[v['central_lapse'] for v in centers],'-o',ms=3,label='3D origin interpolation; scientific checkpoints')
    axes[0].set(ylabel='Lapse at coordinate origin',title='Central lapse and original core-average history');axes[0].legend(fontsize=8)
    axes[1].plot([v['time'] for v in history],[v['core_lapse'] for v in history],label='Coordinate-volume mean inside r<0.75M; native0.25M history');axes[1].set(xlabel='t / M_ref',ylabel='Mean core lapse');axes[1].legend(fontsize=8)
    for ax in axes:ax.set_xlim(0,2500);ax.axvline(1500,color='.4',ls=':',lw=1);ax.grid(alpha=.18)
    for ext in ['png','pdf']:fig.savefig(out/('central_lapse_to2500.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)
    from extension_resources import storage
    disk=storage();record=dict(utc=now(),job=job(),case=CASE,summary=summary,checkpoint=state['latest'],numerical_acceptance=receipt(ROOT/'evidence/CASE_NUMERICAL_ACCEPTANCE.json'),plots=[receipt(p) for p in sorted(out.glob('*')) if p.suffix in ['.png','.pdf']],tables=[receipt(p) for p in sorted(out.glob('*.csv'))],central_lapse_sources=receipt(out/'CENTRAL_LAPSE_SOURCES.json'),final_storage=disk,actual_visual_review_pending=True,history_reused_through_M=1500,original700M_and1500M_preserved=True,limitations='Coordinates and gauge diagnostics. Native core average is coordinate-volume weighted r<0.75M, not alpha(0) or proper-volume weighting. Historical local constraint history likewise uses coordinate volume; inherited solver unchanged. Boundary effects and absent convergence control remain unresolved. No stability claim.')
    cp=state['latest'];last=shell[-1]
    text=['# Unperturbed corrected yc=0.500 continuation', '',
        f"The accepted1500M trajectory reached {summary['endpoint']:.9g}M. Numerical particle, mass, stream and constraint checks passed; actual visual review is pending.", '',
        f"Final particles: {cp['particle_count']:,}; removal counts: {cp['removal_counts']}; maximum mass-ledger closure: {summary['max_closure']:.6g}.",
        f"Qualified apparent-horizon onset: {summary['ah_first']}. Active finest spacing: {cp['finest_active_dx']:g}M.", '',
        f"At the last full-particle snapshot ({last['time']:.9g}M), r50={last['r50']:.9g}M and r90={last['r90']:.9g}M.",
        f"At the final scientific checkpoint, coordinate-origin lapse={centers[-1]['central_lapse']:.12g}; dense native core-average lapse at {history[-1]['time']:.9g}M is {history[-1]['core_lapse']:.12g}.", '',
        'The core history is a coordinate-volume average inside r<0.75M. Origin lapse is trilinearly interpolated from8native3D cell centres; source hashes and weights are in CENTRAL_LAPSE_SOURCES.json.', '',
        '| Time window [M] | r50 peak-to-peak / median |', '| --- | --- |']
    for lo,hi in [(1000,1500),(1500,2000),(2000,2500)]:
        values=[v['r50'] for v in shell if lo<=v['time']<=hi]
        if len(values)>1:text.append(f'| {lo}–{hi} | {(max(values)-min(values))/np.median(values):.6g} |')
    text += ['', 'These finite-window excursions do not establish asymptotic stability. Boundary effects, gauge dependence, inherited source coupling and absent convergence controls remain limitations.', '',
        f"Final checkpoint: `{cp['path']}`; SHA256 `{cp['sha256']}`.",
        f"Native delivery: `{out}`. Original700M/1500M anchors and accepted0–1500M products are preserved.",
        f"Combined AMD-session storage plus reserved Perseus allowance: {disk['own']/GIB:.6f}GiB; whole-user WORK: {disk['whole']/TIB:.6f}TiB. The user authorized available WORK usage without a per-campaign ceiling. Whole-user and filesystem safeguards remain active. Final allocated-node accounting and verified review-copy locations are recorded after this allocation terminates.", '',
        'Products: mass_shells_to2500.png/.pdf; central_lapse_to2500.png/.pdf; REPORT_continuation.md/.pdf; shell/lapse CSV tables; numerical acceptance and source manifests.', '']
    report=out/'REPORT_continuation.md';report.write_text('\n'.join(text));record['report']=receipt(report)
    from pdf_report import render_pdf
    layout=render_pdf(text,out/'REPORT_continuation.pdf',
        [out/'mass_shells_to2500.png',out/'central_lapse_to2500.png'])
    record.update(report_pdf=layout['pdf'],pdf_layout=receipt(out/'REPORT_LAYOUT.json'),
        pdf_previews=layout['actual_pdf_previews'])
    atomic(out/'GENERATION_OK.json',record);atomic(EXT/'evidence/FINAL_PRODUCTS_READY.json',dict(utc=now(),job=job(),delivery=str(out),generation=receipt(out/'GENERATION_OK.json'),actual_visual_review_pending=True));print('UPDATED_PLOTS_READY',out)
