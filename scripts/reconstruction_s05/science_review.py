"""Extract seed-level comparison tables for scientific interpretation under Slurm.

A radial profile is included only when all its 12 native run gates have passed.
This describes short-time numerical responses; it does not select a sampler or
replace native-state QA, the independent arithmetic checks, or figure inspection.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from common import ROOT, CASES, atomic, require_job, sha
from qa_reductions import table


def rms(values):
    return float(np.linalg.norm(values)/np.sqrt(len(values)))


def quotient(value, baseline):
    return float(value/baseline) if abs(baseline)>1e-12 else None


def window_mean(times, values, low, high):
    """Time-weighted mean on the recorded usable span, without extrapolation."""
    left, right = max(low, times[0]), min(high, times[-1])
    assert right>left
    nodes = np.r_[left, times[(times>left)&(times<right)], right]
    return float(np.trapezoid(np.interp(nodes, times, values), nodes)/(right-left)), [float(left), float(right)]


def main():
    require_job()
    parser = argparse.ArgumentParser()
    parser.add_argument('--radial', choices=('legacy', 'proper'), action='append')
    args = parser.parse_args()
    radials = args.radial or ['legacy', 'proper']
    assert len(set(radials)) == len(radials)
    cases = [row for row in CASES if row[1] in radials]
    case_rows, windows, fields, mixing, manifest = [], [], [], [], []
    for case, radial, method, seed in cases:
        base, native = ROOT/'reduced'/case, ROOT/'runs'/case
        assert (native/'RUN_OK').exists(), 'Incomplete profile: '+case
        summary = json.loads((base/'summary.json').read_text())
        completion = json.loads((native/'completion.json').read_text())
        assert summary['endpoint_M'] == completion['endpoint_M'] == 300
        assert summary['initial_sample_sha256'] == completion['input_sample_sha256']
        shells, w, health, glob, mix = [table(base/(name+'.csv')) for name in
                                       ('shells', 'windows', 'health', 'global_constraints', 'cohort_mixing')]
        assert len(w['start']) == 27 and len(shells['time']) == 151
        info = {'case': case, 'radial': radial, 'method': method, 'seed': seed}
        usable = health['constraint_usable'].astype(bool)
        global_usable = glob['constraint_usable'].astype(bool)
        row = dict(info, initial_E_error=summary['initial_error'],
                   final_COM_M=float(shells['com_mag'][-1]), max_COM_M=float(shells['com_mag'].max()),
                   minimum_core_lapse=float(health['core_lapse'].min()), final_core_lapse=float(health['core_lapse'][-1]),
                   boris_fallbacks=completion['boris_fallbacks'],
                   max_outside_finest_mass_fraction=summary['max_outside_finest_mass_fraction'])
        for q in (10, 50, 90):
            row['r'+str(q)+'_initial_M'] = float(shells['r'+str(q)][0])
            row['r'+str(q)+'_final_fractional_shift'] = float(shells['r'+str(q)][-1]/shells['r'+str(q)][0]-1)
        for key in ('H_rms_r10', 'M_rms_r10', 'H_rms_r16', 'M_rms_r16'):
            row[key+'_initial'] = float(health[key][0])
            row[key+'_last_usable'] = float(health[key][usable][-1])
        row['last_usable_constraint_time_M'] = float(health['time'][usable][-1])
        case_rows.append(row)
        for low in (0, 100, 200):
            use = w['start']==low
            assert np.array_equal(w['quantile'][use], np.arange(10, 100, 10))
            item = dict(info, start_M=low, end_M=low+100)
            for source, target in (('half_range_fraction', 'raw_amplitude_rms'),
                                   ('detrended_half_range_fraction', 'detrended_amplitude_rms'),
                                   ('mean_shift_fraction', 'mean_shift_rms')):
                item[target] = rms(w[source][use])
            for q in (10, 50, 90):
                select = use&(w['quantile']==q)
                for key in ('half_range_fraction', 'detrended_half_range_fraction', 'mean_shift_fraction'):
                    item[f'r{q}_'+key] = float(w[key][select].item())
            for key in ('H_rms_r10', 'M_rms_r10', 'H_rms_r16', 'M_rms_r16'):
                item[key+'_time_mean'], support = window_mean(health['time'][usable], health[key][usable], low, low+100)
                item['core_constraint_support_M'] = support
            for key in ('H_rms', 'M_rms'):
                item['global_'+key+'_time_mean'], support = window_mean(glob['time'][global_usable], glob[key][global_usable], low, low+100)
                item['global_constraint_support_M'] = support
            windows.append(item)
        assert len(summary['field_checks']) == len(summary['sampling_checks']) == 4
        for slot, (field, sampling) in enumerate(zip(summary['field_checks'], summary['sampling_checks'])):
            assert abs(field['time']-100*slot)<1e-7 and abs(sampling['time']-100*slot)<1e-7
            fields.append(dict(info, nominal_time_M=100*slot,
                               E_relative_L2_target=field['E_relative_L2_target'],
                               E_relative_L2_equilibrium=field['E_relative_L2_equilibrium'],
                               initial_matter_empty_fraction=sampling['initial_matter_empty_fraction'],
                               outside_cube_mass_fraction=sampling['outside_diagnostic_cube_mass_fraction']))
            use = np.isclose(mix['time'], 100*slot, atol=1e-7, rtol=0)
            assert use.sum()==100
            diagonal = use&(mix['initial_cohort']==mix['current_mass_decile'])
            mixing.append(dict(info, nominal_time_M=100*slot,
                               mean_cohort_diagonal_fraction=float(mix['mass_fraction'][diagonal].mean())))
        for path in [base/(name+'.csv') for name in ('shells', 'windows', 'health', 'global_constraints', 'cohort_mixing')]+[base/'summary.json', native/'completion.json']:
            manifest.append({'path':str(path.relative_to(ROOT)), 'bytes':path.stat().st_size, 'sha256':sha(path)})

    compared = ('raw_amplitude_rms', 'detrended_amplitude_rms', 'mean_shift_rms',
                'H_rms_r10_time_mean', 'M_rms_r10_time_mean', 'H_rms_r16_time_mean',
                'M_rms_r16_time_mean', 'global_H_rms_time_mean', 'global_M_rms_time_mean')
    baseline = {(r['radial'], r['seed'], r['start_M']):r for r in windows if r['method']=='mc'}
    for row in windows:
        mc = baseline[(row['radial'], row['seed'], row['start_M'])]
        assert np.allclose(row['core_constraint_support_M'], mc['core_constraint_support_M'], atol=1e-7, rtol=0)
        assert np.allclose(row['global_constraint_support_M'], mc['global_constraint_support_M'], atol=1e-7, rtol=0)
        for metric in compared:
            row[metric+'_ratio_to_mc'] = quotient(row[metric], mc[metric])
            if row['method']=='mc': assert row[metric+'_ratio_to_mc'] == 1
    ensemble = []
    lines = ['# Seed-level scientific review tables', '',
             'Profiles covered: '+', '.join(radials)+'. All included profiles have12 validated300M runs.', '',
             'Values below are RMS over r10 through r90, normalized by each run’s initial radius. Each ratio uses the same profile and seed label’s Monte Carlo run. Seeds1/2/3 are separate realizations; saved times are not trials. Raw excursions include drift; detrending can remove slow oscillation.', '',
             '| Profile | Sampler | Window M | Raw amplitude / MC: seeds1,2,3 | Detrended amplitude / MC: seeds1,2,3 | Mean-shift RMS / MC: seeds1,2,3 |',
             '|---|---|---|---|---|---|']
    for radial in radials:
        for method in ('mc', 'stratified', 'cell', 'ablation'):
            for low in (0, 100, 200):
                subset = sorted([r for r in windows if (r['radial'], r['method'], r['start_M'])==(radial, method, low)], key=lambda r:r['seed'])
                assert [r['seed'] for r in subset] == [1, 2, 3]
                group = dict(radial=radial, method=method, start_M=low, end_M=low+100)
                texts = []
                for metric in compared:
                    values = [r[metric+'_ratio_to_mc'] for r in subset]
                    group[metric+'_ratio_to_mc'] = {'seeds':values,
                        'mean':float(np.mean(values)) if all(v is not None for v in values) else None,
                        'range':[min(values),max(values)] if all(v is not None for v in values) else None,
                        'all_three_lower':all(v is not None and v<1 for v in values)}
                    if metric in compared[:3]: texts.append(', '.join('undefined' if v is None else f'{v:.5g}' for v in values))
                lines.append(f'| {radial} | {method} | {low}–{low+100} | '+' | '.join(texts)+' |')
                ensemble.append(group)
    lines += ['', 'Constraint means use usable completed steps only, with actual-time trapezoidal weighting on the common recorded support. The tiny final300M step is excluded; the final valid endpoint is recorded, with no extrapolation. Cohort retention is the arithmetic mean of the10 row-normalized diagonal fractions, not particle-identity preservation. Late continuum departures and empty fractions include coherent motion. These tables support interpretation and do not establish physical instability, convergence or Plummer stability.']
    destination = ROOT/'science_review'/os.environ['SLURM_JOB_ID']/('_'.join(radials))
    repo = Path(__file__).resolve().parents[2]
    atomic(destination/'METRICS.json', {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'job':os.environ['SLURM_JOB_ID'], 'source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
           'script_sha256':sha(Path(__file__)), 'radials':radials, 'case_rows':case_rows,
           'window_rows':windows, 'field_rows':fields, 'mixing_rows':mixing,
           'ensemble_windows':ensemble, 'source_manifest':manifest})
    atomic(destination/'TABLES.md', '\n'.join(lines)+'\n')
    print('SCIENCE_TABLES', destination, flush=True)


if __name__ == '__main__':
    main()
