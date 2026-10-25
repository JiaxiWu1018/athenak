"""Allocated milestone audit and source-backed diagnostic figures."""
import csv
import re
import shutil
import numpy as np
from common import *
from reduce import reduce_case, csv_rows, write_csv
from publication_style import style

def plot_case(c, red, destination):
    import matplotlib.pyplot as plt
    style()
    destination.mkdir(parents=True, exist_ok=True)
    shell = csv_rows(red/'shells.csv')
    history = csv_rows(red/'history.csv')
    mass = csv_rows(red/'mass.csv')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    colors = plt.cm.viridis(np.linspace(.08, .92, 9))
    for q, color in zip(range(10, 100, 10), colors):
        axes[0, 0].plot([r['time'] for r in shell], [r[f'r{q}'] for r in shell], c=color, label=f'{q}%')
    axes[0, 0].set(yscale='log', ylabel='COM coordinate rest-mass shell radius [M]')
    axes[0, 0].legend(ncol=3, fontsize=7)
    axes[0, 1].plot([r['time'] for r in history], [r['core_lapse'] for r in history])
    axes[0, 1].set(ylabel='Volume-mean core lapse')
    for key in ['H_rms_r10', 'M_rms_r10']:
        axes[1, 0].semilogy([r['time'] for r in history], [r[key] for r in history], label=key)
    axes[1, 0].set(ylabel='Proper-volume RMS constraint, r<10M')
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot([r['time'] for r in mass], [r['closure'] for r in mass])
    axes[1, 1].set(ylabel='Rest-mass ledger fractional closure')
    for ax in axes.flat:
        ax.set_xlabel('t / M_ref')
        ax.grid(alpha=.18)
    fig.suptitle(f"Session10 {c['key']}: U={c['U']:+g}; actual endpoint {history[-1]['time']:g}M")
    for suffix in ['png', 'pdf']:
        fig.savefig(destination/f'diagnostics.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)

def main():
    job()
    from extension_resources import admission
    admission()
    work = ROOT/'runs/yc0500'
    state = load(work/'state.json')
    end = state['latest']['time']
    assert state.get('complete') or end >= float(os.environ['S10_TARGET'])-1e-8
    red = ROOT/'reduced/yc0500'
    if os.environ.get('S10_FIGURES_ONLY') == '1':
        prior_path = ROOT/'evidence'/f'MILESTONE_{round(end):05d}_NUMERICAL.json'
        prior = load(prior_path)
        assert prior['numerical_passed'] and abs(prior['endpoint']-end) < 1e-8
        source_review = Path(prior['figures'][0]['path']).parent/'NUMERICAL_ACCEPTANCE.json'
        source_receipt = receipt(source_review)
        assert load(source_review) == prior
        for entry in prior['science_trends']['source_inputs']:
            path = red/Path(entry['path']).name
            assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
        destination = ROOT/'reviews'/f'milestone_{round(end):05d}_figures_{job()}'
        assert not destination.exists()
        plot_case(CASE, red, destination)
        from milestone_review import science_trends, forecast
        marker = dict(prior, utc=now(), job=job(), figures_only=True,
            numerical_checks_reused_from=source_receipt,
            science_trends=science_trends(CASE, red, destination),
            milestone_forecast=forecast(destination),
            figures=[receipt(p) for p in sorted(destination.iterdir()) if p.suffix in ['.png', '.pdf']],
            actual_visual_review=False)
        atomic(destination/'NUMERICAL_ACCEPTANCE.json', marker)
        atomic(prior_path, marker)
        return
    if red.exists() and not (red/'REDUCTION_OK.json').exists():
        preserved = EXT/'evidence/allocations'/job()/'prior_incomplete_reduction'
        shutil.copytree(red, preserved)
        atomic(preserved/'PRESERVATION.json', dict(utc=now(), job=job(),
            source=str(red), files=[receipt(p) for p in sorted(preserved.rglob('*')) if p.is_file()],
            reason='Retain failed/partial analysis outputs before a registered retry'))
    summary = reduce_case(CASE, require_complete=state.get('complete', False))
    checks = []
    from reduce_run import read_pvtk
    shells = csv_rows(red/'shells.csv')
    snaps = [s for s in load(red/'snapshots.json') if summary['first_removal'] is None or s['time'] < summary['first_removal']]
    for snap in [snaps[0], snaps[len(snaps)//2], snaps[-1]]:
        t, cycle, n, fields = read_pvtk(snap['file'])
        pos = np.column_stack([fields[k] for k in 'xyz']).astype(np.float64)
        rr = np.sort(np.linalg.norm(pos-pos.mean(axis=0), axis=1))
        row = min(shells, key=lambda v: abs(v['time']-t))
        for q in range(10, 100, 10):
            rank = int(np.ceil(n*q/100))-1
            assert rr[max(0, rank-1)]-1e-10 <= row[f'r{q}'] <= rr[min(n-1, rank+1)]+1e-10
        checks.append(dict(time=t, source=receipt(Path(snap['file'])), quantiles='independent rank check within one particle'))
    logs = []
    fallbacks = 0
    rejected = 0
    transport_pointer_errors = 0
    for segment in state['segments']:
        if not segment.get('log'):
            continue
        p = Path(segment['log'])
        text = p.read_text(errors='replace')
        assert '### FATAL' not in text and not re.search(r'search_fail=[1-9]', text)
        fallbacks += sum(map(int, re.findall(r'cycle\s+\d+:\s+(\d+) of \d+ particles fell back', text)))
        rejected += sum(map(int, re.findall(r'rejected(?:_updates)?=(\d+)', text)))
        transport_pointer_errors += text.count('UCX  ERROR get pointer info fail (nil)')
        logs.append(receipt(p))
    grids = []
    for p in sorted((work/'checkpoint_receipts').glob('*.json')):
        d = load(p)
        if 'finest_active_dx' in d:
            grids.append(dict(time=d['time'], cycle=d['cycle'], nblocks=d['nblocks'], finest_active_dx=d['finest_active_dx']))
    if grids:
        write_csv(red/'checkpoint_mesh.csv', sorted(grids, key=lambda r: r['time']))
    from boundary import tail
    boundary = tail(CASE)
    final = None
    if state.get('complete'):
        from case_review import review
        final = review(CASE)
    destination = ROOT/'reviews'/f'milestone_{round(end):05d}_{job()}'
    plot_case(CASE, red, destination)
    from milestone_review import science_trends, forecast
    trends = science_trends(CASE, red, destination)
    milestone_forecast = forecast(destination)
    matched_pilot = None
    if abs(end-20) < 1e-8 and not CASE.get('inherited_time'):
        from pilot_review import review as pilot_review
        matched_pilot = pilot_review(CASE, destination)
    native_cost = sum(s['elapsed'] for s in state['segments'] if s.get('simulation_advanced', 0) > 0 and s.get('elapsed'))/3600
    advanced = end-CASE.get('inherited_time', 0)
    marker = dict(utc=now(), job=job(), case=CASE, endpoint=end, complete=state.get('complete', False),
        summary=summary, quantile_checks=checks, native_logs=logs, fallback_particle_steps=fallbacks,
        rejected_updates_from_recognized_counters=rejected, boundary=boundary,
        transport_pointer_info_error_lines=transport_pointer_errors,
        transport_note='These diagnostics also appear at successful20M pilot finalization. Native return codes, particle checks and constraints are assessed separately; no transport-correctness proof is implied.',
        native_node_hours=native_cost, native_node_hours_per_M=native_cost/advanced if advanced > 0 else None,
        final_numerical_acceptance=final, matched_pilot=matched_pilot,
        science_trends=trends, milestone_forecast=milestone_forecast,
        figures=[receipt(p) for p in sorted(destination.iterdir()) if p.suffix in ['.png', '.pdf']],
        actual_visual_review=False, numerical_passed=True)
    atomic(destination/'NUMERICAL_ACCEPTANCE.json', marker)
    atomic(ROOT/'evidence'/f'MILESTONE_{round(end):05d}_NUMERICAL.json', marker)
    if end >= 20 and not CASE.get('inherited_time') and not (EXT/'evidence/OUTWARD_PILOT_REVIEW.json').exists():
        atomic(EXT/'evidence/OUTWARD_PILOT_REVIEW.json', marker)
    print(json.dumps(dict(case=CASE['key'], endpoint=end, numerical_passed=True, review=str(destination)), indent=2))

if __name__ == '__main__':
    main()
