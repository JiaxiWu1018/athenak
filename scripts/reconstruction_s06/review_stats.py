"""Allocated descriptive review of compact results; no new diagnostic definitions."""
from session import *
import numpy as np
from plot_results import column, usable, weighted_mean
import csv

def extrema(rows, key):
    values = column(rows, key); times = column(rows, 'time')
    lo, hi = int(np.argmin(values)), int(np.argmax(values))
    return {'min': float(values[lo]), 'min_time': float(times[lo]),
            'max': float(values[hi]), 'max_time': float(times[hi]), 'final': float(values[-1])}

def main():
    require_job()
    dest = ROOT / 'reduced'
    shells, health, global_h, windows = [read_csv(dest / name) for name in
                                         ('shells.csv', 'health.csv', 'global_constraints.csv', 'windows.csv')]
    central = read_csv(dest / 'central_energy.csv')
    summary = load(dest / 'summary.json')
    result = {'utc': now(), 'job': os.environ['SLURM_JOB_ID'], 'shells': {}, 'constraint_ratios': {},
              'lapse': extrema(health, 'core_lapse'), 'outer_lapse': extrema(health, 'outer_lapse'),
              'central_E_slice': extrema(central, 'E_central_slice_r075'), 'field_checks': summary['field_checks'],
              'outside_finest_max_fraction': summary['max_outside_finest_mass_fraction'],
              'max_mass_outside_diagnostic_cube_fraction': max(r['outside_diagnostic_cube_mass_fraction'] for r in summary['sampling_checks']),
              'particle_restart_comparison': summary['smoke_production_interior']}
    for q in (10, 50, 90):
        primary = column(shells, f'r{q}'); origin = column(shells, f'r{q}_origin')
        com_change = primary / primary[0] - 1; origin_change = origin / origin[0] - 1
        result['shells'][str(q)] = {'radius_extrema': extrema(shells, f'r{q}'),
                                   'max_abs_COM_vs_origin_change_pp': float(100 * np.max(np.abs(com_change - origin_change))),
                                   'windows': [r for r in windows if int(r['quantile']) == q]}
    for family, rows, mask, keys in [('core', health, usable(health), ['H_rms_r10', 'M_rms_r10', 'H_rms_r16', 'M_rms_r16']),
                                    ('global', global_h, usable(global_h), ['H_rms', 'M_rms'])]:
        for key in keys:
            early = weighted_mean(rows, key, P, 2 * P, mask)
            late = weighted_mean(rows, key, 4 * P, 5 * P, mask)
            result['constraint_ratios'][key] = {'family': family, 'early': early, 'late': late,
                                               'late_over_early': late['mean'] / early['mean']}
    profiles = np.load(dest / 'profiles.npz')
    density = profiles['mass_bins'][:, 0] / (4 * np.pi / 3 * np.diff(profiles['edges']**3)[0])
    result['central_coordinate_mass_density'] = {'initial': float(density[0]), 'min': float(density.min()),
        'max': float(density.max()), 'max_time': float(profiles['time'][np.argmax(density)]),
        'min_over_initial': float(density.min() / density[0]), 'max_over_initial': float(density.max() / density[0]),
        'definition': 'Same COM-centred0..0.5M coordinate-rest-mass profile bin as session5'}
    raw = load(ROOT / 'evidence/OUTPUT_MANIFEST.json')
    stage = load(ROOT / 'evidence/PERSEUS_ANALYSIS_STAGE.json')
    result['storage'] = {'amd_production_raw_logical_bytes': raw['production_raw_logical_bytes'],
                          'perseus_production_raw_logical_bytes': raw['local_production_raw_logical_bytes'],
                          'perseus_all_staged_run_files_logical_bytes': sum(r['bytes'] for r in stage['files']),
                          'final_checkpoint_bytes': 2372815454}
    atomic(ROOT / 'evidence/SCIENCE_REVIEW.json', result)
    table = ['# Descriptive shell-window review', '', '|Shell|Family|Start/M|End/M|Raw excursion %|Detrended excursion %|Mean shift %|',
             '|---|---|---:|---:|---:|---:|---:|']
    for q in (10, 50, 90):
        for r in result['shells'][str(q)]['windows']:
            table.append(f'|r{q}|{r["family"]}|{float(r["start"]):.3f}|{float(r["end"]):.3f}|'
                         f'{100*float(r["half_range_fraction"]):.4f}|{100*float(r["detrended_half_range_fraction"]):.4f}|'
                         f'{100*float(r["mean_shift_fraction"]):+.4f}|')
    (ROOT / 'reports/WINDOW_TABLES.md').write_text('\n'.join(table) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'shells'}, indent=2))

if __name__ == '__main__':
    main()
