"""Independent arithmetic checks of actual native histories and completed reductions.

This does not re-read raw particle/3D snapshots or independently re-test the pusher.
Those native-state checks belong to each allocated run's completion gates.
"""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from common import ROOT, CASES, N, atomic, require_job, sha


def table(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Empty table: ' + str(path))
    def number(v):
        return float({'True': '1', 'False': '0', '': 'nan'}.get(v, v))
    return {k: np.array([number(row[k]) for row in rows]) for k in rows[0]}


def equal(a, b, label, atol=2e-13, rtol=2e-12):
    if not np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True):
        raise ValueError(label + ': maximum absolute difference ' + str(np.nanmax(np.abs(a-b))))
    return float(np.nanmax(np.abs(np.asarray(a)-b)))


def check_case(case, evidence):
    base = ROOT/'reduced'/case
    native = evidence/case
    if not (native/'RUN_OK').exists() or not (native/'INTERIM_100_OK').exists():
        raise ValueError('Completed native gates absent: ' + case)
    summary = json.loads((base/'summary.json').read_text())
    completion = json.loads((native/'completion.json').read_text())
    endpoint = completion['endpoint_M']
    assert endpoint == summary['endpoint_M'] == 300
    assert summary['initial_sample_sha256'] == completion['input_sample_sha256']
    assert len(summary['sources']) == endpoint//2+1
    assert len(completion['checkpoints']) == 4
    equal(np.array(sorted(s['time'] for s in completion['checkpoints'])),
          np.array([0, 100, 200, 300]), 'Checkpoint output times', atol=1e-7, rtol=0)
    assert all(s['particle_count'] == N and not any(s['removal_header']) and
               s['payload_verified'] for s in completion['checkpoints'])
    equal(np.array([s['rest_mass'] for s in completion['checkpoints']]),
          1.0301064819992727, 'DP rest-mass ledger', atol=1e-13, rtol=0)
    shells = table(base/'shells.csv')
    assert np.array_equal(shells['nominal_time'], np.arange(0, endpoint+1, 2))
    times = shells['time']
    assert np.all(np.diff(times)>0) and times[0] == 0 and times[-1] == endpoint
    for k, v in shells.items():
        assert np.isfinite(v).all(), k
    radii = np.column_stack([shells[f'r{q}'] for q in range(10, 100, 10)])
    origin = np.column_stack([shells[f'r{q}_origin'] for q in range(10, 100, 10)])
    assert np.all(radii>0) and np.all(np.diff(radii, axis=1)>0)
    assert np.all(origin>0) and np.all(np.diff(origin, axis=1)>0)
    equal(shells['rest_mass'], shells['rest_mass'][0], 'Snapshot mass constancy', atol=2e-14, rtol=0)
    com = np.column_stack([shells['com_'+a] for a in 'xyz'])
    equal(np.sqrt(np.einsum('ij,ij->i', com, com)), shells['com_mag'], 'COM vector norm')

    # Use explicit segment integrals and an independent weighted least-squares
    # solve, without importing the production window/helper functions.
    windows = table(base/'windows.csv')
    assert len(windows['start']) == 27
    window_error = 0.
    for low in (0, 100, 200):
        high = low+100
        nodes = np.unique(np.r_[low, times[(times>low)&(times<high)], high])
        yy = np.column_stack([np.interp(nodes, times, col) for col in radii.T])
        widths = np.diff(nodes)
        mean = np.sum(widths[:, None]*(yy[:-1]+yy[1:])/2, axis=0)/100
        weight = np.zeros(len(nodes))
        weight[:-1] += widths/2
        weight[1:] += widths/2
        design = np.column_stack([np.ones(len(nodes)), nodes-low-50])
        fit = np.linalg.lstsq(design*np.sqrt(weight)[:, None],
                              yy*np.sqrt(weight)[:, None], rcond=None)[0]
        residual = yy-design@fit
        independent = {
            'half_range_fraction': np.ptp(yy, axis=0)/(2*radii[0]),
            'mean_shift_fraction': (mean-radii[0])/radii[0],
            'detrended_half_range_fraction': np.ptp(residual, axis=0)/(2*radii[0]),
            'slope_fraction_per_M': fit[1]/radii[0],
        }
        use = (windows['start']==low)&(windows['end']==high)
        assert np.array_equal(windows['quantile'][use], np.arange(10, 100, 10))
        for key, value in independent.items():
            window_error = max(window_error, equal(value, windows[key][use], 'Window '+key))

    health = table(base/'health.csv')
    glob = table(base/'global_constraints.csv')
    custom = np.atleast_2d(np.loadtxt(native/'out'/(case+'.user.hst')))
    global_raw = np.atleast_2d(np.loadtxt(native/'out'/(case+'.z4c.user.hst')))
    assert custom.shape == (151, 19) and global_raw.shape == (151, 11)
    equal(custom[:, 0], health['time'], 'Custom history times', atol=0, rtol=0)
    equal(global_raw[:, 0], glob['time'], 'Global history times', atol=0, rtol=0)
    usable = (custom[:, 0]==0)|(custom[:, 18]>1e-6)
    assert usable[0] and np.array_equal(usable, health['constraint_usable'])
    assert np.array_equal(usable, glob['constraint_usable'])
    equal(custom[:, 18], health['dt_completed'], 'Completed timestep', atol=0, rtol=0)
    coords = np.arange(-15.75, 16, .5)
    rr2 = sum(x*x for x in np.meshgrid(coords, coords, coords, indexing='ij'))
    norm_error = 0.
    for radius, offset in ((10, 2), (16, 8)):
        vol = np.count_nonzero(rr2<radius**2)*.5**3
        equal(custom[:, offset], vol, 'Coordinate volume r'+str(radius), atol=1e-10, rtol=0)
        for kind, col in (('H', offset+1), ('M', offset+2)):
            assert np.all(custom[:, col]>=0)
            values = np.sqrt(custom[:, col]/custom[:, offset])
            norm_error = max(norm_error, equal(values, health[f'{kind}_rms_r{radius}'], 'Core '+kind))
            assert values[0]>0
            equal(values[0], summary['initial_constraint_crosscheck'][str(radius)][kind+'_rms'],
                  'Initial 3D/history '+kind, atol=0, rtol=2e-6)
    equal(custom[:, 15]/custom[:, 14], health['core_lapse'], 'Core lapse')
    equal(custom[:, 17]/custom[:, 16], health['outer_lapse'], 'Outer lapse')
    for kind, col in (('H', 3), ('M', 4)):
        norm_error = max(norm_error, equal(np.sqrt(global_raw[:, col]/global_raw[:, 10]),
                                          glob[kind+'_rms'], 'Global '+kind))

    # Reconstruct L2 numerators from radial RMS and exact Cartesian cell counts.
    fields = table(base/'field_errors.csv')
    rr = np.sqrt(rr2).ravel()
    bin_count = np.histogram(rr, bins=np.arange(0, 32.0001, .5))[0]
    rad = case.split('_')[0]
    reference_paths = [ROOT/'reference'/(m+'.npz') for m in (rad, 'proper')]
    norm2 = [float(np.sum(np.load(path)['E'][rr<10]**2)) for path in reference_paths]
    energy_error = 0.
    for record in summary['field_checks']:
        use = fields['time']==record['time']
        assert use.sum() == np.count_nonzero(bin_count)
        support = fields['r'][use]<10
        for suffix, denominator in zip(('target', 'equilibrium'), norm2):
            error = fields['E_'+suffix+'_error_rms'][use]
            numerator = np.sum(bin_count[bin_count>0][support]*error[support]**2)
            energy_error = max(energy_error, equal(np.sqrt(numerator/denominator),
                record['E_relative_L2_'+suffix], 'Projected E relative L2', atol=2e-12, rtol=2e-10))

    profiles = np.load(base/'profiles.npz')
    equal(profiles['mass_bins'].sum(1), shells['rest_mass']*(1-shells['outside_profile_mass_fraction']),
          'Radial histogram mass', atol=1e-10, rtol=1e-10)
    occupancy = table(base/'occupancy.csv')
    for record in summary['sampling_checks']:
        use = occupancy['time']==record['time']
        centered = np.sum(occupancy['cells'][use]*occupancy['count_mean'][use])
        assert 0<centered<=N+1e-7
        if record['outside_diagnostic_cube_mass_fraction']==0:
            equal(centered, N, 'Centre occupancy total', atol=1e-7, rtol=0)
        assert np.all(occupancy['neff_mean'][use]>=0)
        matter = occupancy['initial_matter_cells'][use]
        selected = matter>0
        empty = np.average(occupancy['initial_matter_empty_fraction'][use][selected], weights=matter[selected])
        equal(empty, record['initial_matter_empty_fraction'], 'Empty fraction')
    weights = json.loads((base/'weights.json').read_text())
    assert sum(weights['particle_count']) == N and weights['mass_min']>0
    assert 1<=weights['global_effective_count']<=N+1e-7
    equal(np.sum(weights['rest_mass_per_bin']), 1.0301064819992727, 'Weight histogram mass', atol=2e-10, rtol=0)
    mixing = table(base/'cohort_mixing.csv')
    for t in np.unique(mixing['time']):
        for cohort in range(1, 11):
            use = (mixing['time']==t)&(mixing['initial_cohort']==cohort)
            assert np.array_equal(mixing['current_mass_decile'][use], np.arange(1, 11))
            equal(mixing['mass_fraction'][use].sum(), 1, 'Cohort row mass', atol=1e-10, rtol=0)
    for name in ('shells', 'health', 'global_constraints', 'windows'):
        short = table(ROOT/'reduced_interim'/case/(name+'.csv'))
        full = table(base/(name+'.csv'))
        for key in short:
            equal(short[key], full[key][:len(short[key])], 'Interim prefix '+name+'/'+key, atol=0, rtol=0)
    events = 0
    cycles = set()
    for line in (native/'solver.log').read_text().splitlines():
        if '### gr_boris non-convergence:' in line:
            tokens = line.split()
            cycles.add(int(tokens[tokens.index('cycle')+1].rstrip(':')))
            events += int(tokens[tokens.index('of')-1])
    assert events == completion['boris_fallbacks'] and len(cycles) == completion['boris_fallback_cycles']
    return {'case': case, 'endpoint_M': endpoint, 'snapshots': len(times),
            'window_max_abs_error': window_error, 'constraint_norm_max_abs_error': norm_error,
            'energy_L2_max_abs_error': energy_error, 'boris_fallbacks': events,
            'boris_fallback_cycles': len(cycles), 'raw_bytes_retained': completion['raw_bytes'],
            'scope': 'Independent reduction arithmetic/native history reconciliation; native particle checks use run receipts'}


def main():
    require_job()
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', nargs='+', default=[c[0] for c in CASES])
    ap.add_argument('--evidence-root', type=Path, default=ROOT/'runs')
    args = ap.parse_args()
    results = [check_case(case, args.evidence_root) for case in args.cases]
    manifest = []
    for case in args.cases:
        paths = list((ROOT/'reduced'/case).glob('*'))+list((ROOT/'reduced_interim'/case).glob('*'))
        paths += list((args.evidence_root/case).glob('*.json'))
        paths += list((args.evidence_root/case/'out').glob('*.hst'))
        paths += [args.evidence_root/case/'solver.log']
        manifest += [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in paths if p.is_file()]
    repo = Path(__file__).resolve().parents[2]
    receipt = {'job': os.environ['SLURM_JOB_ID'], 'host': __import__('socket').gethostname(),
               'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'source_commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
               'script_sha256': sha(Path(__file__)), 'cases': results, 'manifest': manifest,
               'readiness': 'Ready within reviewed arithmetic scope; incomplete campaign and no visual acceptance implied'}
    dest = ROOT/'qa_reductions'/os.environ['SLURM_JOB_ID']/'QA_REDUCTIONS_OK.json'
    atomic(dest, receipt)
    print('QA_REDUCTIONS_OK', len(results), dest, flush=True)


if __name__ == '__main__':
    main()
