"""Allocated compact readout and report; final delivery still requires actual visual review."""
import subprocess
from cfg import *
from compare_cases import read,NAMES,LABELS,PRIMARY

def main():
    require_job();assert load(ROOT/'evidence/RESULTS_ARITHMETIC_QA.json')['passed']
    dest=ROOT/'reports';dest.mkdir(exist_ok=True)
    windows=read(ROOT/'comparison/metric_windows.csv');norms=read(ROOT/'comparison/residual_norms.csv');ratios=read(ROOT/'comparison/residual_ratios.csv')
    summary={};lines=['# ST session 7: MC particle-count comparison','',
        'Four fresh proper-profile yc=0.819 runs reached 1000M. Same mesh, gauge, pusher, executable and output cadences as session6. Public seed1/internal PCG64(3001); nested positions and covariant momenta, equal masses M0/N.','',
        'All evolved raw particle, field and checkpoint data remain on AMD. This report and compact review products may be copied to Perseus.','',
        '## Early and late evolution','',
        'Windows are P..2P and 4P..5P, with P=197.625957866M. Shell amplitudes and shifts below use the common analytic equilibrium radius; lapse uses absolute units. These are coordinate radii and a fixed-origin r<0.75M mean lapse.','',
        '| Case | Quantity | Early raw half-range | Late raw half-range | Early detrended | Late detrended | Early mean shift | Late mean shift |','|---|---|---:|---:|---:|---:|---:|---:|']
    for name,label in zip(NAMES,LABELS):
        summary[name]={}
        for metric in PRIMARY:
            early=next(r for r in windows if r['case']==name and r['metric']==metric and r['family']=='P'and abs(float(r['start'])-P)<1e-8)
            late=next(r for r in windows if r['case']==name and r['metric']==metric and r['family']=='P'and abs(float(r['start'])-4*P)<1e-8)
            factor=1 if metric=='core_lapse'else 100
            vals=[factor*float(r[k])for k,r in [('half_range',early),('half_range',late),('detrended_half_range',early),('detrended_half_range',late),('mean_shift',early),('mean_shift',late)]]
            lines.append('| '+label+' | '+metric+(' [%]'if metric!='core_lapse'else'')+' | '+' | '.join(f'{v:.6g}'for v in vals)+' |')
            summary[name][metric]=dict(early=early,late=late)
    lines+=['','## Adjacent particle-count residuals','','Residuals are finer minus coarser. Rows report D(1m,2m), D(2m,4m), D(4m,8m), then both fine/coarse RMS ratios. Ratios below one indicate closer histories for the stated seed and window. They do not by themselves imply approach to equilibrium.','','| Quantity | Mode | Window | D12 | D24 | D48 | D24/D12 | D48/D24 |','|---|---|---|---:|---:|---:|---:|---:|']
    for metric in PRIMARY:
        for mode in ['absolute','change_from_initial']:
            for family in ['full','early_P','late_P']:
                nn=sorted([r for r in norms if r['metric']==metric and r['mode']==mode and r['family']==family],key=lambda r:int(r['level']))
                rr=sorted([r for r in ratios if r['metric']==metric and r['mode']==mode and r['family']==family],key=lambda r:int(r['coarse_level']))
                vals=[r['RMS']for r in nn]+[r['fine_over_coarse']for r in rr]
                lines.append('| '+metric+' | '+mode+' | '+family+' | '+' | '.join(f'{float(v):.6g}'if v!=''else'undefined'for v in vals)+' |')
    lines+=['','## Interpretation and review','','Actual agent scientific assessment and PNG/PDF/encoded-movie review remain required. No automatic stability classification or convergence order is assigned. Inspect all100M/P/2P windows, mean drifts and numerical health before deciding whether growth is persistent or saturating. This is one nested realization at fixed mesh, timestep and boundary.','','## Figures and movies','']
    figures=['shells_com','shells_origin','health_comparison','amplitude_and_drift','adjacent_residuals','residual_norms']
    for name in figures:lines.append(f'- [{name} PNG](../plots/{name}.png) · [vector PDF](../plots/{name}.pdf)')
    lines+=['','Movies follow20000 fixed IDs per case with immutable initial-radius colours, all-z x-y projection, common equal axes and501 native output slots at15fps.']
    for group in ['0','1','2','3','scaling']:lines.append(f'- [Movie {group}](../movies/s7_{group}_xy_initial_radius.mp4)')
    lines+=['','## Reproduction and retention','',f'Native root: `{AMD}/runs/CASE/production/out`. All raw outputs and preflight data remain there; no deletion is authorized. Exact inputs in inputs/, sample binaries on AMD in samples/, launch/environment/checkpoint/output manifests in runs/ and evidence/. Compiled physics5a5db71c, Kokkos6739bc62, executable SHA256'+EXE_SHA+'.',
        '', 'Analysis source: '+source()+'. All generation, simulation, reductions and rendering used Slurm on AMD. The independent Perseus source clone and thin review copy do not contain session7 evolved raw states.','']
    (dest/'REPORT_session7.md').write_text('\n'.join(lines))
    raw=[]
    for case in CASES:
        p=ROOT/'runs'/case['id']/'production/OUTPUT_MANIFEST.json';m=load(p)
        raw.append(dict(case=case['id'],path=str(p.parent/'out'),logical_bytes=m['logical_bytes'],manifest_sha256=sha(p),retained=True))
    allocated=int(subprocess.check_output(['du','-s','-B1',str(ROOT)],text=True).split()[0])
    whole=int(subprocess.check_output(['du','-s','-B1','/work1/eliasmost/jiaxiwu'],text=True).split()[0])
    atomic(ROOT/'evidence/SCIENCE_READOUT.json',dict(utc=now(),source=source(),cases=summary,raw=raw,review_pending=True))
    atomic(ROOT/'evidence/STORAGE.json',dict(utc=now(),job=os.environ['SLURM_JOB_ID'],session_allocated_bytes=allocated,whole_user_allocated_bytes=whole,raw=raw,
        native_data_policy='All raw remains AMD; compact reports/figures/movies/reductions only may transfer to Perseus'))
    (dest/'STORAGE.md').write_text('# Session7 retained storage\n\nAll raw remains on AMD.\n\n'+ '\n'.join(f'- {r["case"]}: `{r["path"]}`, {r["logical_bytes"]} logical bytes, retained.'for r in raw)+f'\n\nWhole-user AMD allocated bytes: {whole}; session allocated bytes: {allocated}. No deletion.\n')
    atomic(ROOT/'evidence/ANALYSIS_RENDER_COMPLETE.json',dict(utc=now(),source=source(),actual_agent_review_pending=True))
    print('REPORT_AND_READOUT_OK; ACTUAL_REVIEW_REQUIRED',flush=True)
if __name__=='__main__':main()
