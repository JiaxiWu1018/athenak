"""Source-backed final scientific readout; requires all four 500M runs and short controls."""
import argparse
import json
from pathlib import Path
from campaign import ROOT, CASES, resource_snapshot
from analyze import PRIMARY_SCHEME, PRIMARY_H_LABEL, BASE_H, completed, read_csv, window_average
from state_io import require_job, sha, write_json


def segment_sources(record):
    work=Path(record['work'])
    launch=json.loads((work/'LAUNCH.json').read_text())
    expected=json.loads((ROOT/'evidence/FIXED_FAST_V1_OK.json').read_text())
    assert expected['passed'] and launch['exe_sha256']==expected['hip_sha256']
    assert launch['configuration']==CASES[record['case']]
    assert sha(work/'input.athinput')==launch['input_sha256']
    return [work/name for name in ('COMPLETE.json','LAUNCH.json','input.athinput','OUTPUT_MANIFEST.json')]


def main():
    require_job()
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();assert PRIMARY_SCHEME=='fixed_fast_v1'
    sources=[];runs=[]
    tolerance_paths=sorted((ROOT/"evidence").glob("OPTIMIZATION_TOLERANCE_*.json"))
    tolerance_checks=[json.loads(p.read_text()) for p in tolerance_paths]
    assert any(r["passed"] for r in tolerance_checks), "No passed full-count tolerance sensitivity gate"
    sources.extend(tolerance_paths)
    for n in (1,2,4,8):
        case=f'{PRIMARY_SCHEME}_{n}m_h{PRIMARY_H_LABEL}'
        records=completed(case);assert records and records[-1]['target']==500
        for r in records:
            assert r['checkpoint']['finite'] and r['checkpoint']['original_ids_masses']
            assert r['checkpoint']['count']==n*1000000 and r['checkpoint']['removals']==0
            assert r['solver']['failures']==0 and r['solver']['max_residual']<=1.01e-11
            sources.extend(segment_sources(r))
        assert records[0]['initial']['exact_sample_match']
        runs.append(dict(particles=n*1000000,case=case,endpoint=500,segments=records))
    for label in ('010','005','0025'):
        records=completed(f'{PRIMARY_SCHEME}_1m_h{label}')
        assert any(r['target']==100 for r in records)
        assert records[0]['initial']['exact_sample_match']
        for r in records:
            assert r['checkpoint']['finite'] and r['checkpoint']['original_ids_masses']
            assert r['checkpoint']['count']==1000000 and r['checkpoint']['removals']==0
            assert r['solver']['failures']==0 and r['solver']['max_residual']<=1.01e-11
            sources.extend(segment_sources(r))
    order_path=a.data/'temporal_order.csv';window_path=a.data/'constraint_windows.csv'
    orders=read_csv(order_path);windows=read_csv(window_path);sources.extend([order_path,window_path,a.data/'TEMPORAL_SOURCES.json',a.data/'ANALYSIS.json'])
    particle_orders={}
    lines=['# Session10: fixed-node particle predictor + RK3','',
        'Four exact nested seed1 ST yc=0.819 realizations, with 1/2/4/8 million particles, reached 500M. The primary scheme uses nodes selected from the Euler-predicted midpoint, recomputes interpolation weights and derivatives each iteration, and requires scaled residual ≤ 1e-11. It accepts no Euler or interpolation fallback.','',
        f'The stable production timestep is nominal h={BASE_H:g}M, at finest grid spacing 0.5M. The original h=.2M fixed-node experiment developed nonfinite outer-field diagnostics at 18M and aborted at 18.4M. The earlier frozen-source RK3 control also failed at 18.4M. These failed outcomes are retained separately.','',
        '## Temporal self-convergence','',
        '| Scheme | Time [M] | Position p | Covariant momentum p | Energy p |',
        '|---|---:|---:|---:|---:|']
    for scheme in ('legacy',PRIMARY_SCHEME):
        values=[]
        for t in (20,50,100):
            rr=[next(r for r in orders if r['scheme']==scheme and r['kind']=='particles' and r['field']==field and float(r['time'])==t) for field in ('position','covariant_momentum','energy')]
            assert all(r['observed_p'] and r['above_roundoff_floor']=='True' for r in rr)
            v=[float(r['observed_p']) for r in rr];values.extend(v)
            lines.append('| '+('Legacy RK4, fallback allowed' if scheme=='legacy' else 'Fixed-node RK3, strict')+f' | {t} | '+ ' | '.join(f'{p:.4f}' for p in v)+' |')
        particle_orders[scheme]=dict(min=min(values),max=max(values))
    lines+=['',f"Measured particle orders span {particle_orders[PRIMARY_SCHEME]['min']:.3f}–{particle_orders[PRIMARY_SCHEME]['max']:.3f} for the new scheme and {particle_orders['legacy']['min']:.3f}–{particle_orders['legacy']['max']:.3f} for the legacy controls. First order corresponds to p=1 and second order to p=2.",'',
        'The new timestep triplet is .1/.05/.025M; the retained legacy triplet is .2/.1/.05M. Each order uses matched double-precision states at the same endpoint and the same particle IDs. Particle norms use rest-mass weights; grid norms use active-cell coordinate volume. Restart alignment can cause short timestep-growth transients, so these are nominal maximum-timestep refinements. Per-field grid orders are retained in reduced/temporal_order.csv; particle orders do not establish the order of every grid field.','',
        '![Temporal order](figures/temporal_order.png)','',
        '![Matched solution differences](figures/temporal_differences.png)','',
        '## Constraint comparison','',
        'The following ratios compare time-mean RMS inside coordinate r<10M. Values below 1 indicate smaller constraint RMS in the new run. The archived comparison tables extend to1000M; this report and its figures use their0–500M interval. The archived scheme uses h=.2M and lagged particle sources; the new runs use h=.1M and synchronized endpoint diagnostics. Thus long-run ratios include timestep and diagnostic-timing differences, as well as the coupling/stencil change.','',
        '| Particles [million] | Window [M] | Hamiltonian new/archived | Momentum new/archived |',
        '|---:|---:|---:|---:|']
    for n in (1,2,4,8):
        for lo,hi in ((20,100),(100,300),(300,500)):
            rr=[next(r for r in windows if int(r['N'])==n*1000000 and r['region']=='r10' and r['constraint']==m and int(r['start'])==lo and int(r['end'])==hi) for m in ('H','M')]
            lines.append(f'| {n} | {lo}–{hi} | '+ ' | '.join(f"{float(r['ratio_new_old']):.4f}" for r in rr)+' |')
    matched={}
    old=a.data/'legacy_1m_h010/health.csv';new=a.data/f'{PRIMARY_SCHEME}_1m_h010/health.csv';sources.extend([old,new])
    for metric in ('H','M'):
        matched[metric]=window_average(read_csv(new),f'{metric}_rms_r10',20,100)/window_average(read_csv(old),f'{metric}_rms_r10',20,100)
    resources=resource_snapshot()
    lines+=['',f"For the short comparison with both schemes at h=.1M and synchronized diagnostics, the 20–100M r<10M time-mean ratios are H={matched['H']:.4f} and momentum={matched['M']:.4f}. This comparison still includes the changed integrator, coupling, fixed-node force interpolation and strict solver acceptance.",'',
        '![Constraint evolution](figures/constraint_evolution.png)','',
        '![Constraint ratios](figures/constraint_ratios.png)','',
        '![Matched timestep short controls](figures/short_controls.png)','',
        'Core panels use coordinate-volume RMS; full-domain panels use native proper-volume RMS. Particle-number scaling uses one nested seed at fixed mesh and timestep. It measures sampling dependence, not temporal order or spatial convergence.','',
        '![Particle-number scaling](figures/particle_number_scaling.png)','',
        '## Fallback alternatives and provenance','',
        'The primary trajectories contain no accepted fallback steps. Preserved legacy RK4 controls allow the original Euler/geometry fallback behavior; audited Euler event totals are 135/33/5 for h=.2/.1/.05M. The failed frozen-source RK3 h=.2M control recorded 233484 Euler events before abort. These are events, not unique particles, and the aborted log may omit final events. Fallback counts alone do not establish the cause of first-order convergence. A new predictor-with-Euler-fallback variant was not mixed into the primary data.','',
        'The performance optimization reuses inverse-metric derivative results only when both field views are identical, guarded in native code. Native CPU/GPU/MPI/restart/crossing gates passed. The first full 1M-particle 20M comparison exceeded its 2e-12 absolute threshold (maximum particle difference 6.49e-11); that failed gate is retained. A separately recorded test with solver tolerance tightened from 1e-11 to 1e-13 is required to establish agreement within the original threshold and bound nominal differences below 10% of measured solver-tolerance sensitivity (or 2e-12). This is numerical equivalence relative to solver error, not bitwise identity; see OPTIMIZATION_TOLERANCE evidence. All raw runs, failed attempts, checkpoints and reproduction inputs remain under the AMD campaign root. No historical campaign was overwritten and no four legacy 500M reruns were performed.','',
        f'Allocated report snapshot: {resources.get("campaign_bytes",0):,} campaign bytes ({resources.get("campaign_bytes",0)/2**30:.2f} GiB), {resources.get("node_hours",0):.3f} allocated node-hours. All data retained; no cleanup authorized.','',
        f'Raw-data root: `{ROOT}`. Exact file hashes and completed segment records are in FINAL_SOURCES.json. Figure-source manifests are beside each PNG/PDF. Storage and allocated node-hour snapshot are in FINAL_SUMMARY.json. Data are retained; no cleanup is authorized.','']
    for name in ('FIXED_V1_OK.json','FIXED_FAST_V1_OK.json','FIXED_FAST_V1_SOURCE.json'):
        sources.append(ROOT/'evidence'/name)
    sources.extend(sorted((ROOT/'evidence').glob('FAST_EQUIVALENCE_*.json')))
    audit=ROOT/'analysis/control_audit_495421/AUDIT.json';sources.append(audit)
    (a.out/'REPORT.md').write_text('\n'.join(lines))
    write_json(a.out/'FINAL_SUMMARY.json',dict(primary_scheme=PRIMARY_SCHEME,runs=runs,particle_orders=particle_orders,matched_h010_core_ratios=matched,resources=resources))
    write_json(a.out/'FINAL_SOURCES.json',dict(script_sha256=sha(__file__),sources=[dict(path=str(p),sha256=sha(p)) for p in sorted(set(sources))],outputs=[dict(path=p.name,sha256=sha(p)) for p in (a.out/'REPORT.md',a.out/'FINAL_SUMMARY.json')]))

if __name__=='__main__':main()
