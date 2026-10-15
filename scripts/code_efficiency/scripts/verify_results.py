#!/usr/bin/env python3
"""Independently verify retained measurements and final artifacts inside Slurm."""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

ROOT = Path(os.environ.get('EFFICIENCY_SESSION', Path(__file__).resolve().parents[1]))
FIGURES = ['01_particle_count', '02_meshblock_layout', '03_strong_scaling',
           '04_dynamic_particle_mesh_cost', '05_sampling_and_rank_imbalance',
           '06_kernel_cost_and_interpolation','07_rank_phase_work_and_waits']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Final artifact checks require a Slurm allocation')
    failures, missing = [], []
    counts = {}

    def require(condition, message):
        if not condition:
            failures.append(message)

    def near(actual, expected):
        return math.isfinite(actual) and math.isclose(actual, expected, rel_tol=2e-10, abs_tol=1e-10)

    def read(path):
        return json.loads(path.read_text())

    def digest(path):
        return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()

    matrix = read(ROOT / 'matrix.json')
    require(len(matrix) == 60 and len({c['name'] for c in matrix}) == 60,
            'Matrix must contain 60 distinct throughput cases')
    profile_names = {c['name'] for c in matrix if not c['mesh_only'] and
                     (c['test'] >= 4 or c['name'] in {
                         'count_st_dep0_N1000000', 'count_st_dep1_N1000000',
                         'count_flat_dep0_N1000000', 'count_flat_dep1_N1000000',
                         'blocks_st_dep1_B16', 'blocks_st_dep1_B64',
                         'ranks_st_dep1_R2', 'ranks_st_dep1_R4'})}
    device_names = {'dynamic_ppc6', 'dynamic_ppc24', 'dynamic_ppc96', 'dynamic_ppc384',
                    'count_st_dep0_N1000000', 'count_flat_dep0_N1000000'}
    require(len(profile_names) == 27, 'Profile plan must contain 27 cases')
    manifests = read(ROOT / 'provenance/executables/manifest.json')
    modes = {record['kind']: key for key, record in manifests.items()}
    all_receipts = []
    primary = {}
    for kind, names in [('throughput', {c['name'] for c in matrix}),
                        ('profile', profile_names), ('device', device_names)]:
        counts[kind] = 0
        for name in sorted(names):
            path = ROOT / 'results' / kind / name / 'receipt.json'
            if not path.exists() or not read(path).get('valid'):
                missing.append(f'{kind}/{name}')
                continue
            r = read(path)
            c = r['case']
            label = f'{kind}/{name}'
            counts[kind] += 1
            all_receipts.append(r)
            if kind == 'throughput':
                primary[name] = r
            require(not r.get('superseded_reason') and r['exit_code'] == 0, label + ': promoted status')
            ranks = c['ranks']
            for part in ['timings', 'initial', 'final']:
                require(len(r[part]) == ranks and {x['rank'] for x in r[part]} == set(range(ranks)),
                        label + ': missing/duplicate ranks in ' + part)
            if not all(len(r[part]) == ranks for part in ['timings', 'initial', 'final']):
                continue
            n = c['n']
            expected_endpoint = (1 if n >= 100000000 else 5) if kind == 'throughput' else 1.5
            require(near(r['endpoint'], expected_endpoint), label + ': approved target endpoint')
            modulus = (1 << 64) - 1
            expected_sum = n * (n - 1) // 2 & modulus
            expected_square = n * (n - 1) * (2 * n - 1) // 6 & modulus
            for part in ['initial', 'final']:
                require(sum(x['local_count'] for x in r[part]) == n, label + ': global count in ' + part)
                for x in r[part]:
                    require(x['valid'] and x['count'] == n and x['tag_sum'] == expected_sum
                            and x['tag_square'] == expected_square, label + ': count/tag audit in ' + part)
                    require(sum(x['block_counts']) == x['local_count'], label + ': block census in ' + part)
            maximum = max(t['elapsed_local'] for t in r['timings'])
            for t in r['timings']:
                rank = t['rank']
                duration = t['final_time'] - t['start_time']
                require(t['cycles'] >= 10 and duration > 0 and maximum > 0, label + ': nonempty timing window')
                require(near(t['elapsed_max'], maximum), label + ': maximum elapsed over ranks')
                require(near(t['dt'], .05) and near(duration, t['cycles'] * .05), label + ': fixed timestep interval')
                require(near(t['start_time'], .5) and t['final_time'] <= expected_endpoint + 1e-8,
                        label + ': physical-time limits')
                require(t['physical_cells'] == c['grid'] ** 3, label + ': physical-cell denominator')
                require(t.get('device_timing_compiled') == (kind == 'device'), label + ': compiled mode')
                require(t['profile'] == (kind != 'throughput'), label + ': phase timer mode')
                require(near(t['particles_per_second'], (0 if c['mesh_only'] else n) * t['cycles'] / maximum),
                        label + ': particle updates/s')
                require(near(t['cells_per_second'], c['grid'] ** 3 * t['cycles'] / maximum),
                        label + ': cell cycles/s')
                require(near(t['seconds_per_M'], maximum / duration), label + ': wall seconds/M')
                if c['frozen']:
                    a = next(x for x in r['initial'] if x['rank'] == rank)
                    b = next(x for x in r['final'] if x['rank'] == rank)
                    require(a['geometry_hash'] == b['geometry_hash'], label + ': frozen arrays')
            folder = ROOT / r['attempt_path']
            provenance = read(folder / 'provenance.json')
            require(provenance['exe_sha256'] == modes['profile' if kind == 'device' else 'release'],
                    label + ': immutable executable')
            require(provenance['input_sha256'] == digest(folder / 'input.athinput'), label + ': exact input checksum')
            if r.get('trace_requested'):
                require(r.get('trace_captured') and any(p.stat().st_size > 0 for p in folder.glob('timeline*.nsys-rep')),
                        label + ': retained Nsight timeline')
            for item in read(folder / 'manifest.json')['files']:
                retained = folder / item['path']
                require(retained.exists() and retained.stat().st_size == item['bytes']
                        and digest(retained) == item['sha256'], label + ': checksum ' + item['path'])
    # Compare identical frozen trajectories across rank decompositions independently.
    for metric in ['st', 'flat']:
        for deposition in [0, 1]:
            baseline = primary.get(f'blocks_{metric}_dep{deposition}_B32')
            for ranks in [2, 4]:
                parallel = primary.get(f'ranks_{metric}_dep{deposition}_R{ranks}')
                if baseline is None or parallel is None:
                    continue
                label = parallel['case']['name'] + ': matched one-rank control'
                for t in parallel['timings']:
                    require(t['cycles'] == baseline['timings'][0]['cycles']
                            and near(t['final_time'], baseline['timings'][0]['final_time']), label + ' window')
                for part in ['initial', 'final']:
                    require(all(x['payload_hash'] == baseline[part][0]['payload_hash']
                                for x in parallel[part]), label + ' ' + part + ' payload')
    # Reconcile the human-facing numeric table with the original receipts.
    table = list(csv.DictReader((ROOT / 'results/throughput.csv').open()))
    table = {row['name']: row for row in table if row['valid'] == 'True'}
    require(set(table) == set(primary), 'Throughput CSV case coverage disagrees with receipts')
    for name, r in primary.items():
        row = table.get(name)
        if row is None:
            continue
        t = r['timings'][0]
        for key, original in [('particle_updates_per_second', t['particles_per_second']),
                              ('cell_cycles_per_second', t['cells_per_second']),
                              ('seconds_per_M', t['seconds_per_M']),
                              ('M_per_second', 1 / t['seconds_per_M']),
                              ('actual_endpoint', t['final_time'])]:
            require(near(float(row[key]), original), name + ': CSV ' + key)
    numerical = read(ROOT / 'provenance/numerical_gate.json')
    require(numerical.get('passed') and numerical.get('sampling_balanced')
            and numerical.get('executables') == modes, 'Executable-bound numerical/sample gate')
    validation = read(ROOT / 'results/VALIDATION.json')
    for key in ['initial_realizations_match', 'source_moments_match', 'compiled_modes', 'scaling_states_match']:
        require(validation.get(key), 'Analysis validation: ' + key)
    # Check hashes independently of the report writer, including PDF/PNG/table outputs.
    report_manifest = read(ROOT / 'provenance/report_manifest.json')
    plotting_commit = report_manifest.get('plotting_source_commit')
    plotting_sources = report_manifest.get('plotting_sources', [])
    require(bool(plotting_commit) and len(plotting_sources) >= 2, 'Recorded rendering source versions')
    if plotting_commit:
        for item in plotting_sources:
            archived = subprocess.run(['git', '-C', str(ROOT / 'src/athenak'), 'show',
                                       f'{plotting_commit}:scripts/code_efficiency/{item["path"]}'],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            require(archived.returncode == 0 and hashlib.sha256(archived.stdout).hexdigest() == item['sha256'],
                    'Committed rendering source: ' + item['path'])
    for item in report_manifest['files']:
        path = ROOT / item['path']
        require(path.exists() and path.stat().st_size == item['bytes']
                and digest(path) == item['sha256'], 'Report artifact checksum: ' + item['path'])
    if not missing:
        require(validation.get('complete'), 'Complete coverage was not validated by analysis')
        for name in FIGURES:
            for extension in ['pdf', 'png']:
                path = ROOT / 'figures' / f'{name}.{extension}'
                require(path.exists() and path.stat().st_size > 1000, 'Missing final figure: ' + str(path))
    report = dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                  job_id=os.environ['SLURM_JOB_ID'], counts=counts, missing=missing,
                  failures=failures, available_measurements_passed=not failures,
                  complete=not missing and not failures,
                  checks='Rates recomputed from rank timing windows; exact plan/rank/count/tag coverage; immutable executable and input hashes; timing modes; freeze audits; numerical gate; retained per-attempt and report artifact checksums; CSV reconciliation. Visual inspection is recorded separately.')
    (ROOT / 'results/FINAL_QA.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ['counts', 'available_measurements_passed', 'complete', 'failures']}), flush=True)
    return 0 if not failures and (args.allow_partial or not missing) else 1


if __name__ == '__main__':
    raise SystemExit(main())
