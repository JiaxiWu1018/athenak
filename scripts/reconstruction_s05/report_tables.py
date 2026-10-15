"""Summarize actual complete-ensemble review evidence for report authoring.

Three seeds are the realizations. Quantiles, windows and saved times are not
additional trials. This supplies descriptive tables, not sampler-selection rules.
"""
import argparse
import csv
import datetime
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

from common import ROOT, atomic, require_job, sha


def summary(values):
    assert len(values) == 3
    if any(v is None for v in values):
        return {'seeds':values, 'mean':None, 'range':None}
    assert all(math.isfinite(v) for v in values)
    return {'seeds':values, 'mean':statistics.fmean(values), 'range':[min(values), max(values)]}


def quotient(a, b):
    return a/b if abs(b)>1e-12 else None


def groups(rows, keys, metrics):
    result = []
    for values, group in itertools.groupby(sorted(rows, key=lambda r:tuple(r[k] for k in keys)),
                                           key=lambda r:tuple(r[k] for k in keys)):
        selected = sorted(group, key=lambda r:r['seed'])
        assert [r['seed'] for r in selected] == [1, 2, 3]
        result.append(dict(zip(keys, values), **{m:summary([r[m] for r in selected]) for m in metrics}))
    return result


def text(value, factor=1):
    if value['mean'] is None:
        return 'undefined'
    low, high = value['range']
    return f"{factor*value['mean']:.5g} ({factor*low:.5g}–{factor*high:.5g})"


def main():
    require_job()
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion', type=Path, required=True)
    parser.add_argument('--science', type=Path, required=True)
    args = parser.parse_args()
    motion = json.loads(args.motion.read_text())
    science = json.loads(args.science.read_text())
    assert science['radials'] == ['legacy', 'proper']
    assert len(science['case_rows']) == 24 and len(science['window_rows']) == 72
    assert len(motion['rows']) == len(science['field_rows']) == len(science['mixing_rows']) == 96
    for row in science['source_manifest']:
        path = (ROOT/row['path']).resolve()
        assert path.is_relative_to(ROOT.resolve())
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']

    def key(row):
        return row['radial'], row['method'], row['seed'], row['nominal_time_M']
    fields = {key(r):r for r in science['field_rows']}
    assert len(fields) == 96 and len({key(r) for r in motion['rows']}) == 96
    noise = ('E_target_relative_L2', 'E_angular_residual_relative_L2', 'E_radial_component_relative_L2',
             'J_coordinate_rms', 'stress_coordinate_anisotropy_rms', 'neff_mean_r10')
    mc = {(r['radial'], r['seed'], r['nominal_time_M']):r.copy()
          for r in motion['rows'] if r['method']=='mc'}
    for row in motion['rows']:
        reference = fields[key(row)]
        assert math.isclose(row['E_target_relative_L2'], reference['E_relative_L2_target'], rel_tol=2e-12, abs_tol=2e-13)
        assert math.isclose(row['E_target_relative_L2']**2,
                            row['E_angular_residual_relative_L2']**2+row['E_radial_component_relative_L2']**2,
                            rel_tol=2e-12, abs_tol=2e-13)
        baseline = mc[(row['radial'], row['seed'], row['nominal_time_M'])]
        for name in noise:
            row[name+'_ratio_to_mc'] = quotient(row[name], baseline[name])

    window_metrics = ('raw_amplitude_rms', 'detrended_amplitude_rms', 'mean_shift_rms',
                      'H_rms_r10_time_mean', 'M_rms_r10_time_mean', 'H_rms_r16_time_mean',
                      'M_rms_r16_time_mean', 'global_H_rms_time_mean', 'global_M_rms_time_mean')
    window_groups = groups(science['window_rows'], ('radial', 'method', 'start_M'),
                           window_metrics+tuple(m+'_ratio_to_mc' for m in window_metrics))
    motion_groups = groups(motion['rows'], ('radial', 'method', 'nominal_time_M'),
                           noise+('count_mean_r10', 'initial_matter_empty_fraction_r10')+
                           tuple(m+'_ratio_to_mc' for m in noise))
    field_groups = groups(science['field_rows'], ('radial', 'method', 'nominal_time_M'),
                          ('E_relative_L2_target', 'E_relative_L2_equilibrium', 'initial_matter_empty_fraction'))
    case_metrics = ('final_COM_M', 'max_COM_M', 'minimum_core_lapse', 'final_core_lapse', 'boris_fallbacks',
                    'r10_final_fractional_shift', 'r50_final_fractional_shift', 'r90_final_fractional_shift',
                    'H_rms_r10_initial', 'M_rms_r10_initial', 'H_rms_r10_last_usable', 'M_rms_r10_last_usable')
    case_groups = groups(science['case_rows'], ('radial', 'method'), case_metrics)
    mixing_groups = groups(science['mixing_rows'], ('radial', 'method', 'nominal_time_M'),
                           ('mean_cohort_diagonal_fraction',))
    weights, resources, mass_errors, extra_inputs = [], [], [], []
    for row in science['case_rows']:
        case = row['case']
        weight_path = ROOT/'reduced'/case/'weights.json'
        memory_path = ROOT/'runs'/case/'memory_samples.jsonl'
        completion = json.loads((ROOT/'runs'/case/'completion.json').read_text())
        memory = [json.loads(line) for line in memory_path.read_text().splitlines()]
        recorded = next(r for r in completion['manifest'] if r['path']==str(memory_path.relative_to(ROOT)))
        assert sha(memory_path) == recorded['sha256'] and memory_path.stat().st_size == recorded['bytes']
        assert max(r['athena_ranks'] for r in memory) == 4
        info = {k:row[k] for k in ('case', 'radial', 'method', 'seed')}
        weight = json.loads(weight_path.read_text())
        weights.append(dict(info, **{k:weight[k] for k in ('mass_min', 'mass_max', 'global_effective_count')}))
        cdf_path = ROOT/'reduced'/case/'initial_mass_errors.csv'
        with cdf_path.open() as stream:
            cdf_rows = list(csv.DictReader(stream))
        cdf_keys = ('point_error_target', 'point_error_equilibrium', 'CIC_error_target', 'CIC_error_equilibrium')
        mass_errors.append(dict(info, **{k+'_max_abs':max(abs(float(r[k])) for r in cdf_rows) for k in cdf_keys}))
        resources.append(dict(info, solver_wall_seconds=completion['solver_wall_seconds'],
                              raw_manifest_bytes=completion['raw_bytes'],
                              sampled_max_aggregate_host_rss_bytes=max(r['aggregate_host_rss_bytes'] for r in memory),
                              sampled_max_device_vram_bytes=max((v for r in memory for v in r['vram_used_bytes'].values()), default=0)))
        extra_inputs.extend({'path':str(p.relative_to(ROOT)), 'bytes':p.stat().st_size, 'sha256':sha(p)}
                            for p in (weight_path, memory_path, cdf_path))
    weight_groups = groups(weights, ('radial', 'method'), ('mass_min', 'mass_max', 'global_effective_count'))
    mass_error_groups = groups(mass_errors, ('radial', 'method'), tuple(k+'_max_abs' for k in cdf_keys))
    resource_summary = {name:{'minimum':min(r[name] for r in resources), 'maximum':max(r[name] for r in resources),
                              'mean':statistics.fmean(r[name] for r in resources)}
                        for name in ('solver_wall_seconds', 'raw_manifest_bytes',
                                     'sampled_max_aggregate_host_rss_bytes', 'sampled_max_device_vram_bytes')}
    resource_summary['total_production_raw_manifest_bytes'] = sum(r['raw_manifest_bytes'] for r in resources)
    old_windows = {(r['method'], r['seed'], r['start_M']):r for r in science['window_rows'] if r['radial']=='legacy'}
    contrasts = []
    for row in science['window_rows']:
        if row['radial'] != 'proper':
            continue
        baseline = old_windows[(row['method'], row['seed'], row['start_M'])]
        contrasts.append(dict(method=row['method'], seed=row['seed'], start_M=row['start_M'],
                              **{m+'_proper_to_legacy':quotient(row[m], baseline[m]) for m in window_metrics[:3]}))
    contrast_groups = groups(contrasts, ('method', 'start_M'), tuple(m+'_proper_to_legacy' for m in window_metrics[:3]))

    lines = ['# Complete-ensemble descriptive tables', '',
             'Entries are mean (minimum–maximum) over seeds1/2/3. Saved times and shell quantiles are not independent trials. See source JSON for every seed value and constraint ratio. Ratios across methods/profiles use the same seed label but generally independent random streams.', '',
             '## Shell excursions and shifts', '',
             'RMS across r10…r90 of fractional measures relative to each run’s own initial shell radii, shown in percent. Raw half-range includes drift; detrending can remove slow oscillations.', '',
             '| Profile | Sampler | Window M | Raw amplitude % | Detrended amplitude % | Mean-shift RMS % |',
             '|---|---|---|---:|---:|---:|']
    for r in window_groups:
        lines.append(f"| {r['radial']} | {r['method']} | {r['start_M']}–{r['start_M']+100} | "+' | '.join(text(r[m], 100) for m in window_metrics[:3])+' |')
    lines += ['', '## Density and sampling after motion', '',
              'Energy errors are relative L2 on fixed origin-r<10M, normalized by the chosen initial CIC continuum. Angular residuals subtract the mean at each exact Cartesian radius; radial components are orthogonal complements. Late departures include ordered motion, asphericity and translation. Counts and Neff are coordinate-volume means, not independent velocity counts.', '',
              '| Profile | Sampler | Time M | Total E error % | Angular E error % | Radial component % | Mean Neff | Initially intersecting cells empty % |',
              '|---|---|---|---:|---:|---:|---:|---:|']
    for r in motion_groups:
        lines.append(f"| {r['radial']} | {r['method']} | {r['nominal_time_M']} | "+
                     ' | '.join(text(r[m], 100) for m in noise[:3])+' | '+text(r['neff_mean_r10'])+' | '+text(r['initial_matter_empty_fraction_r10'], 100)+' |')
    lines += ['', '## Final centre motion and radius shifts', '',
              '| Profile | Sampler | Final COM / M | Final r10 shift % | Final r50 shift % | Final r90 shift % | Fallback events |',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in case_groups:
        lines.append(f"| {r['radial']} | {r['method']} | {text(r['final_COM_M'])} | "+
                     ' | '.join(text(r[f'r{q}_final_fractional_shift'], 100) for q in (10, 50, 90))+' | '+text(r['boris_fallbacks'])+' |')
    lines += ['', '## Radial-measure contrast', '',
              'Proper/historical ratios compare the same sampler and seed label at fixed N, mesh, metric and endpoint. These descriptive ratios do not establish physical stability or convergence.', '',
              '| Sampler | Window M | Raw amplitude ratio | Detrended amplitude ratio | Mean-shift RMS ratio |',
              '|---|---|---:|---:|---:|']
    for r in contrast_groups:
        lines.append(f"| {r['method']} | {r['start_M']}–{r['start_M']+100} | "+
                     ' | '.join(text(r[m+'_proper_to_legacy']) for m in window_metrics[:3])+' |')
    destination = ROOT/'report_tables'/os.environ['SLURM_JOB_ID']
    repo = Path(__file__).resolve().parents[2]
    atomic(destination/'STATS.json', {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'job':os.environ['SLURM_JOB_ID'], 'host':__import__('socket').gethostname(),
           'source_commit':subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
           'script_sha256':sha(Path(__file__)), 'motion_input':{'path':str(args.motion.resolve()), 'sha256':sha(args.motion)},
           'science_input':{'path':str(args.science.resolve()), 'sha256':sha(args.science)},
           'source_files_verified':len(science['source_manifest']), 'window_groups':window_groups,
           'motion_groups':motion_groups, 'field_groups':field_groups, 'case_groups':case_groups,
           'mixing_groups':mixing_groups, 'radial_measure_contrast':contrast_groups,
           'weight_groups':weight_groups, 'initial_mass_error_groups':mass_error_groups,
           'resource_rows':resources, 'resource_summary':resource_summary,
           'extra_input_manifest':extra_inputs,
           'resource_definition':'Launch-to-exit wall includes in-run output; sampled host RSS sums matching Athena processes. VRAM is the largest sampled whole-device DRM counter, not exact solver-only memory. Global mass-weight effective count differs from per-cell CIC Neff.'})
    atomic(destination/'TABLES.md', '\n'.join(lines)+'\n')
    print('REPORT_TABLES_READY', destination, flush=True)


if __name__ == '__main__':
    main()
