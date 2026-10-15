"""Verify full-ensemble report arithmetic and file integrity; collect motion metrics.

All calculations use actual completed cases. Image decoding/header checks do not
constitute visual acceptance, and ensemble flags do not select a winning sampler.
"""
import datetime
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
from common import ROOT, CASES, atomic, require_job, sha
from qa_reductions import table, equal

FIGURES = (
    'f1_initial_errors', 'f1_initial_mass_projection', 'f1_radial_and_angular_errors',
    'f2_occupancy_profiles', 'f2_effective_count_profiles', 'f2_empty_cells_and_weights',
    'f3_shell_histories', 'f3_equilibrium_departures', 'f3_mass_fraction_heatmaps',
    'f3_initial_cohorts', 'f3_cohort_mixing', 'f4_constraints_lapse_com',
    'f4_native_global_constraints', 'f5_amplitude_ratios',
    'f5_detrended_amplitude_ratios', 'f5_mean_shell_shifts', 'f5_drift_ratios')


def quotient(value, baseline):
    return float(value/baseline) if abs(baseline)>1e-12 else None


def same(value, expected, label):
    if expected is None:
        assert value is None, label
    else:
        assert value is not None, label
        assert np.shape(value) == np.shape(expected), label+' shape'
        equal(np.asarray(value), np.asarray(expected), label)


def report_check(interim):
    endpoint = 100 if interim else 300
    data = ROOT/('reduced_interim' if interim else 'reduced')
    out = ROOT/'plots'/('interim100' if interim else 'final')
    windows = {}
    summaries = {}
    for case, rad, method, seed in CASES:
        gate = ROOT/'runs'/case/('INTERIM_100_OK' if interim else 'RUN_OK')
        assert gate.exists(), str(gate)
        summary = json.loads((data/case/'summary.json').read_text())
        assert summary['endpoint_M'] == endpoint
        summaries[(rad, method, seed)] = summary
        w = table(data/case/'windows.csv')
        entries = {}
        for i in range(len(w['start'])):
            key = (int(w['start'][i]), int(w['quantile'][i]))
            assert key not in entries
            entries[key] = {k: float(v[i]) for k, v in w.items()}
        assert len(entries) == 9*(endpoint//100)
        windows[(rad, method, seed)] = entries

    # Verify every plotted individual ratio against the actual same-profile,
    # same-label MC realization; no production ratio/window helper is used.
    ratios = json.loads((out/'amplitude_and_drift_ratios.json').read_text())
    assert len(ratios) == 2*4*3*3*(endpoint//100)*2
    seen = set()
    for row in ratios:
        rad, method, seed = row['radial'], row['method'], row['seed']
        low, q, definition = int(row['window_start']), int(row['quantile']), row['amplitude_definition']
        key = (rad, method, seed, low, q, definition)
        assert key not in seen
        seen.add(key)
        value = windows[(rad, method, seed)][(low, q)]
        base = windows[(rad, 'mc', seed)][(low, q)]
        assert q in (10, 50, 90) and definition in ('half_range_fraction', 'detrended_half_range_fraction')
        same(row['amplitude_ratio_to_mc'], quotient(value[definition], base[definition]), 'Individual amplitude')
        drift = quotient(value['mean_shift_fraction'], base['mean_shift_fraction'])
        same(row['signed_mean_shift_ratio_to_mc'], drift, 'Signed shift ratio')
        same(row['absolute_mean_shift_ratio_to_mc'], abs(drift) if drift is not None else None, 'Absolute shift ratio')
        same(row['mean_shift_fraction'], value['mean_shift_fraction'], 'Mean shift')
        same(row['mc_mean_shift_fraction'], base['mean_shift_fraction'], 'MC mean shift')

    findings = json.loads((out/'findings.json').read_text())
    assert len(findings) == 8
    seen = set()
    last = endpoint-100
    for finding in findings:
        rad, method = finding['radial'], finding['method']
        assert (rad, method) not in seen
        seen.add((rad, method))
        initial = [summaries[(rad, method, s)]['initial_error']['E_relative_L2_target'] for s in (1, 2, 3)]
        amps, shifts = [], []
        for seed in (1, 2, 3):
            def vector(m, key):
                return np.array([windows[(rad, m, seed)][(last, q)][key] for q in range(10, 100, 10)])
            denominator = np.linalg.norm(vector('mc', 'half_range_fraction'))
            assert denominator>0
            amps.append(float(np.linalg.norm(vector(method, 'half_range_fraction'))/denominator))
            # Nine-shell RMS: norm/sqrt(9), including the declared near-zero rule.
            denominator = np.linalg.norm(vector('mc', 'mean_shift_fraction'))/3
            shifts.append(quotient(np.linalg.norm(vector(method, 'mean_shift_fraction'))/3, denominator))
        same(finding['initial_E_error'], initial, 'Initial ensemble errors')
        same(finding['final_window_rms_shell_amplitude_ratio'], amps, 'Nine-shell RMS amplitude ratios')
        for actual, expected in zip(finding['final_window_rms_shell_mean_shift_ratio'], shifts):
            same(actual, expected, 'Nine-shell RMS shift ratio')
        assert finding['all_three_amplitudes_lower_than_mc'] == all(v<1 for v in amps)
        assert finding['all_three_drift_magnitudes_lower_than_mc'] == all(v is not None and v<1 for v in shifts)

    # Check the evidence identities that actually supplied these figures.
    source = json.loads((out/'SOURCE_MANIFEST.json').read_text())
    for row in source:
        path = (ROOT/row['path']).resolve()
        assert path.is_relative_to(ROOT.resolve())
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    artifact_manifest = []
    for stem in FIGURES:
        png, pdf = out/(stem+'.png'), out/(stem+'.pdf')
        with Image.open(png) as picture:
            assert picture.width>=1500 and picture.height>=800
            picture.verify()
        raw = pdf.read_bytes()
        assert raw.startswith(b'%PDF-') and b'%%EOF' in raw[-1024:] and len(raw)>5000
        artifact_manifest.extend({'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
                                  'sha256': sha(p)} for p in (png, pdf))
    return {'endpoint_M': endpoint, 'individual_ratios_checked': len(ratios),
            'ensemble_findings_checked': len(findings), 'figure_pairs': len(FIGURES),
            'source_files_checked': len(source), 'artifacts': artifact_manifest,
            'scope': 'Arithmetic, evidence hashes, PNG decoding and PDF signatures; human visual review still required'}


def motion_metrics():
    """Comparable origin-centred r<10M diagnostics at four common saved slots."""
    axis = np.arange(-15.75, 16, .5)
    xyz = np.meshgrid(axis, axis, axis, indexing='ij')
    rr = np.sqrt(sum(x*x for x in xyz)).ravel()
    edges = np.arange(0, 32.0001, .5)
    counts = np.histogram(rr, edges)[0]
    centers = (edges[:-1]+edges[1:])/2
    selected = centers[counts>0]<10
    cell_count = counts[counts>0][selected]
    norm = {rad: float(np.sum(np.load(ROOT/'reference'/(rad+'.npz'))['E'][rr<10]**2))
            for rad in ('legacy', 'proper')}
    rows = []
    for case, radial, method, seed in CASES:
        fields = table(ROOT/'reduced'/case/'field_errors.csv')
        occupancy = table(ROOT/'reduced'/case/'occupancy.csv')
        times = np.unique(fields['time'])
        assert np.array_equal(np.rint(times/100), np.arange(4))
        for slot, time in enumerate(times):
            use = fields['time']==time
            equal(fields['r'][use], centers[counts>0], 'Native radial grid', atol=0, rtol=0)
            total2 = np.sum(cell_count*fields['E_target_error_rms'][use][selected]**2)
            angular2 = np.sum(cell_count*fields['E_target_angular_rms'][use][selected]**2)
            assert angular2<=total2*(1+1e-10)
            row = {'case': case, 'radial': radial, 'method': method, 'seed': seed,
                   'nominal_time_M': slot*100, 'actual_field_time_M': float(time),
                   'E_target_relative_L2': float(np.sqrt(total2/norm[radial])),
                   'E_angular_residual_relative_L2': float(np.sqrt(angular2/norm[radial])),
                   'E_radial_component_relative_L2': float(np.sqrt(max(total2-angular2, 0)/norm[radial]))}
            for key in ('J_coordinate_rms', 'stress_coordinate_anisotropy_rms'):
                row[key] = float(np.sqrt(np.average(fields[key][use][selected]**2, weights=cell_count)))
            use = (np.rint(occupancy['time']/100)==slot)&(occupancy['r']<10)
            assert use.any()
            for key in ('count_mean', 'neff_mean'):
                row[key+'_r10'] = float(np.average(occupancy[key][use], weights=occupancy['cells'][use]))
            matter = use&(occupancy['initial_matter_cells']>0)
            row['initial_matter_empty_fraction_r10'] = float(np.average(
                occupancy['initial_matter_empty_fraction'][matter], weights=occupancy['initial_matter_cells'][matter]))
            rows.append(row)
    return {'rows': rows, 'definitions': {
        'support': 'Fixed coordinate origin r<10M, equal dx0.5M coordinate-volume cells',
        'energy_normalization': 'Chosen initial continuum reference L2 projected through the same CIC',
        'angular': 'Error residual after subtracting each exact Cartesian-radius group mean',
        'radial': 'Orthogonal radial-group component from total squared error minus angular squared error',
        'occupancy': 'Coordinate-volume mean over fixed r<10M cells, including empty cells',
        'interpretation': 'Late angular departures include asphericity/translation. Momentum/stress moments may include ordered flow; they are not pure shot-noise estimators.'}}


def main():
    require_job()
    assert (ROOT/'evidence/REPORT_READY').exists()
    prior = json.loads((ROOT/'qa_reductions/463822/QA_REDUCTIONS_OK.json').read_text())
    assert {r['case'] for r in prior['cases']} == {
        'legacy_mc_s1', 'legacy_stratified_s1', 'legacy_cell_s1'}
    for row in prior['manifest']:
        path = Path(row['path']).resolve()
        assert path.is_relative_to(ROOT.resolve())
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    results = [report_check(interim) for interim in (True, False)]
    dest = ROOT/'qa_final'/os.environ['SLURM_JOB_ID']
    atomic(dest/'MOTION_METRICS.json', motion_metrics())
    # Separate du processes are essential: one invocation containing a nested
    # campaign and its parent excludes already-counted files from the parent.
    storage = {'measured_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'whole_user_allocated_bytes': int(subprocess.check_output(
                   ['du', '-s', '-B1', '/work1/eliasmost/jiaxiwu'], text=True).split()[0]),
               'session_allocated_bytes': int(subprocess.check_output(
                   ['du', '-s', '-B1', str(ROOT)], text=True).split()[0]),
               'retention': 'All raw retained on AMD; no deletion. Anta capacity check must be reported separately.',
               'note': 'Authoritative separate-call measurements; report batch FINAL_STORAGE.txt uses nested du arguments.'}
    atomic(dest/'STORAGE.json', storage)
    repo = Path(__file__).resolve().parents[2]
    atomic(dest/'QA_FINAL_OK.json', {
        'job': os.environ['SLURM_JOB_ID'], 'host': __import__('socket').gethostname(),
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'script_sha256': sha(Path(__file__)), 'reports': results,
        'readiness': 'Ready within checked arithmetic/file scope; human scientific interpretation and PNG/PDF visual review remain required'})
    print('QA_FINAL_OK', dest, flush=True)


if __name__ == '__main__':
    main()
