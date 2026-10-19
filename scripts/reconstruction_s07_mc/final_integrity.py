"""Allocated audit of closed session7 native files and generated products.

This is staged separately from the frozen simulation runtime. It writes a
snapshot receipt, never COMPLETE, and does not replace scientific/visual review.
"""
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

ROOT = Path('/work1/eliasmost/jiaxiwu/st_reconstruction_s07_mc_20261019')
RUNTIME = '9e8a0614469cc449734b9804b62c92dd85605311'
PHYSICS = '5a5db71c9d02498bfe3402c73c8a2b4b3180e377'
KOKKOS = '6739bc623081648af9e752b616d9671527922cbf'
EXE = '5fb4fb9f2131984854b3dd2eef455e8acc6808152e800698fcf5349e5bc35482'
CASES = {f'mc_{n}m_s1': n*1000000 for n in (1, 2, 4, 8)}
FIGURES = ('shells_com', 'shells_origin', 'health_comparison',
           'amplitude_and_drift', 'adjacent_residuals', 'residual_norms')
PRODUCT_ROOTS = {'reports', 'plots', 'comparison', 'reduced', 'movies', 'review'}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    tmp.replace(path)


def digest(path):
    before = path.stat()
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            value.update(block)
    after = path.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_ino, after.st_size, after.st_mtime_ns), str(path)+' changed while read'
    return value.hexdigest()


def relative(path):
    path = Path(path)
    if path.is_absolute():
        path = path.relative_to(ROOT)
    assert path.parts and '..' not in path.parts, path
    return path.as_posix()


def bound(index, entry, base=None):
    if base is None:
        base = ROOT
    path = relative(base/entry['path'])
    row = index[path]
    assert row['bytes'] == entry['bytes'], path
    assert row['sha256'] == entry['sha256'], path
    return path


def listing(excluded):
    rows = {}
    for parent, dirs, files in os.walk(ROOT, followlinks=False):
        base = Path(parent)
        dirs[:] = sorted(d for d in dirs if relative(base/d) not in excluded)
        for name in sorted(files+ [d for d in dirs if (base/d).is_symlink()]):
            path = base/name
            rel = relative(path)
            if rel in excluded:
                continue
            stat = path.lstat()
            if path.is_symlink():
                rows[rel] = dict(path=rel, kind='symlink', target=os.readlink(path))
            else:
                assert path.is_file(), path
                rows[rel] = dict(path=rel, kind='file', bytes=stat.st_size,
                                 allocated_bytes=stat.st_blocks*512,
                                 mtime_ns=stat.st_mtime_ns, inode=stat.st_ino)
    return rows


def inventory(excluded):
    before = listing(excluded)
    result = []
    for i, (rel, row) in enumerate(sorted(before.items())):
        entry = dict(row)
        if row['kind'] == 'file':
            entry['sha256'] = digest(ROOT/rel)
        result.append(entry)
        if i % 500 == 0:
            print('INVENTORY', i, rel, flush=True)
    assert before == listing(excluded), 'Closed tree changed during audit'
    return result


def closed_accounting(required, destination):
    assert required and len(set(required)) == len(required)
    command = ['sacct', '-X', '-j', ','.join(required), '-n', '-P',
               '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocNodes,NodeList']
    text = subprocess.check_output(command, text=True)
    rows = [line.split('|') for line in text.splitlines() if line.strip()]
    assert len(rows) == len(required) and {r[0] for r in rows} == set(required)
    assert all(r[1] == 'COMPLETED' and r[2] == '0:0' and int(r[4]) == 1 for r in rows), rows
    result = dict(utc=now(), command=command, rows=rows,
                  node_hours=sum(int(r[3])*int(r[4]) for r in rows)/3600,
                  scope='Closed production/reduction/comparison/movie/report jobs only; current integrity job excluded')
    atomic(destination, result)
    return result


def verify_cases(index):
    checked = []
    for name, n in CASES.items():
        initial = read(ROOT/'evidence/samples'/f'{name}.json')
        assert index[f'samples/{name}.bin']['sha256'] == initial['sha256']
        for role in ('preflight', 'restart_probe', 'production'):
            work = ROOT/'runs'/name/role
            if role == 'restart_probe' and n != 8000000:
                assert not work.exists()
                continue
            launch, done, manifest = [read(work/f) for f in
                                      ('launch.json', 'completion.json', 'OUTPUT_MANIFEST.json')]
            assert launch['executable_sha256'] == EXE and launch['compiled_source'] == PHYSICS
            assert launch['kokkos'] == KOKKOS and launch['scripts_source'] == RUNTIME
            assert launch['sample_sha256'] == initial['sha256']
            assert launch['input_sha256'] == index[f'inputs/{name}.athinput']['sha256']
            assert launch['case']['n'] == done['case']['n'] == n
            assert manifest['source'] == RUNTIME and manifest['case'] == name
            endpoint = {'preflight': 20, 'restart_probe': 22, 'production': 1000}[role]
            assert done['endpoint_M'] == manifest['endpoint_M'] == endpoint
            expected = {relative(work/r['path']) for r in manifest['files']}
            actual = {p for p in index if p.startswith(relative(work/'out')+'/')}
            assert expected == actual and len(expected) == len(manifest['files']), (name, role)
            assert manifest['logical_bytes'] == sum(r['bytes'] for r in manifest['files'])
            for row in manifest['files']:
                bound(index, row, work)
            for state in done['checkpoints']:
                assert state['particle_count'] == n and not any(state['removal_header'])
                assert index[relative(state['path'])]['sha256'] == state['sha256']
            assert abs(max(s['time'] for s in done['checkpoints'])-endpoint) < 1e-7
            all_rows = [r for p, r in index.items() if p.startswith(relative(work)+'/')]
            sidecars = [r for r in all_rows if not r['path'].startswith(relative(work/'out')+'/')]
            checked.append(dict(case=name, role=role, endpoint_M=endpoint,
                                path=str(work), out_files=len(expected),
                                out_logical_bytes=manifest['logical_bytes'],
                                sidecar_files=len(sidecars),
                                sidecar_logical_bytes=sum(r['bytes'] for r in sidecars),
                                retained=True))
            if role == 'production':
                assert (work/'RUN_OK').exists()
                assert len(done['checkpoints']) == 11
                assert all(abs(t-100*j) < 1e-7 for j, t in
                           enumerate(sorted(s['time'] for s in done['checkpoints'])))
                summary = read(ROOT/'reduced'/name/'summary.json')
                assert summary['source'] == RUNTIME and summary['states'] == 501
                assert abs(summary['endpoint_M']-1000) < 1e-7
                assert len(summary['field_checks']) == 11
                assert len(summary['preflight_production_interior']) == 2
                assert len(summary['source_particle_outputs']) == 501
                for row in summary['source_particle_outputs']:
                    bound(index, row, work)
                particle_paths = {relative(work/r['path']) for r in summary['source_particle_outputs']}
                assert particle_paths == {p for p in expected if p.endswith('.part.vtk')}
    return checked


def verify_products(index):
    qa = read(ROOT/'evidence/RESULTS_ARITHMETIC_QA.json')
    assert qa['passed'] and qa['source'] == RUNTIME
    assert (qa['residual_rows'], qa['ratio_rows'], qa['metric_windows']) == (780, 520, 680)
    assert len(qa['native_checks']) == 12
    assert qa['old_shell_windows'] == {name: 153 for name in CASES}
    assert read(ROOT/'evidence/ANALYSIS_RENDER_COMPLETE.json')['actual_agent_review_pending']
    for filename, count in [('residual_norms', 780), ('residual_ratios', 520),
                            ('metric_windows', 680), ('window_differences', 510)]:
        with (ROOT/'comparison'/f'{filename}.csv').open() as stream:
            assert sum(1 for _ in csv.DictReader(stream)) == count, filename
    for name in FIGURES:
        assert index[f'plots/{name}.png']['bytes'] > 0
        assert index[f'plots/{name}.pdf']['bytes'] > 0
    assert len(list((ROOT/'review/figures').glob('*.png'))) == 6
    movies = []
    for group in ('0', '1', '2', '3', 'scaling'):
        receipt = read(ROOT/'movie_render'/group/'RENDER_OK.json')
        assert receipt['source'] == RUNTIME and receipt['full_decode_passed']
        assert receipt['frames'] == receipt['ffprobe']['decoded_frames'] == 501
        assert receipt['fps'] == receipt['ffprobe']['fps'] == 15
        assert receipt['display_count_per_panel'] == 20000 and receipt['clipping'] == 0
        count = 4 if group == 'scaling' else 1
        assert receipt['ffprobe']['size'] == [1200*count, 1080]
        assert abs(receipt['ffprobe']['duration']-501/15) < .02
        path = bound(index, receipt['movie'])
        bound(index, receipt['contact_sheet'])
        for row in receipt['data']:
            bound(index, row)
        movies.append(dict(group=group, path=path, sha256=receipt['movie']['sha256'],
                           decoded_frames=501, actual_visual_review_required=True))
    assert index['reports/REPORT_session7.md']['bytes'] > 0
    return movies


def main():
    job = os.environ['SLURM_JOB_ID']
    assert Path(os.environ['ROOT']) == ROOT and ROOT.is_dir()
    assert not (ROOT/'evidence/COMPLETE').exists()
    assert not (ROOT/'evidence/NATIVE_INTEGRITY_QA.json').exists(), 'Preserve previous audit; use a new attempt'
    pin_path = Path(__file__).resolve().parent/'POSTPROCESSING_SOURCE_PIN.json'
    pin = read(pin_path)
    assert pin['native_runtime_source'] == RUNTIME
    for row in pin['files']:
        assert digest(pin_path.parent/row['path']) == row['sha256'], row['path']
    native = read(ROOT/'evidence/SOURCE_PIN.json')
    assert native['head'] == RUNTIME
    for row in native['runtime_files']:
        assert digest(ROOT/'src/athenak'/row['path']) == row['sha256'], row['path']
    assert digest(ROOT/'protected/athena') == EXE
    destination = ROOT/'evidence/integrity'/job
    destination.mkdir(parents=True, exist_ok=False)
    accounting_path = destination/'CLOSED_ACCOUNTING.json'
    accounting = closed_accounting(pin['required_closed_jobs'], accounting_path)
    report_path = ROOT/'reports'/f'STORAGE_INTEGRITY_{job}.md'
    exclusions = {'evidence/integrity', 'evidence/NATIVE_INTEGRITY_QA.json',
                  f'evidence/jobs/{job}_single', f'logs/st_s07_native_integrity.{job}.log',
                  relative(report_path)}
    rows = inventory(exclusions)
    index = {r['path']: r for r in rows if r['kind'] == 'file'}
    cases = verify_cases(index)
    movies = verify_products(index)
    groups = {}
    for row in index.values():
        parts = Path(row['path']).parts
        key = '/'.join(parts[:3]) if parts[0] == 'runs' else parts[0]
        group = groups.setdefault(key, dict(files=0, logical_bytes=0, allocated_bytes=0))
        group['files'] += 1
        group['logical_bytes'] += row['bytes']
        group['allocated_bytes'] += row['allocated_bytes']
    products = [r for r in rows if Path(r['path']).parts[0] in PRODUCT_ROOTS]
    native_rows = [r for r in rows if Path(r['path']).parts[0] not in PRODUCT_ROOTS]
    manifests = []
    for name, entries in [('CLOSED_NATIVE_MANIFEST', native_rows), ('GENERATED_PRODUCT_SNAPSHOT', products)]:
        path = destination/f'{name}.json'
        atomic(path, dict(utc=now(), job=job, files=entries, excluded_current_audit_paths=sorted(exclusions),
                          scope='Closed native bytes and current generated products; final scientific/visual review pending'))
        manifests.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path)))
    sizes = {name: int(subprocess.check_output(['du', '-s', '-B1', str(path)], text=True).split()[0])
             for name, path in [('session_allocated_bytes', ROOT),
                                ('whole_user_allocated_bytes', ROOT.parent)]}
    result = dict(utc=now(), job=job, host=socket.gethostname(), passed=True,
                  audit_source=pin['head'], native_runtime_source=RUNTIME,
                  compiled_physics=PHYSICS, kokkos=KOKKOS, executable_sha256=EXE,
                  closed_accounting_path=str(accounting_path),
                  closed_accounting_sha256=digest(accounting_path),
                  closed_jobs=len(accounting['rows']), manifests=manifests,
                  inventory_groups=groups, retained_runs_including_sidecars=cases,
                  movies=movies, sizes=sizes, all_inventory_files_rehashed=True,
                  native_data_host='AMD only; no raw or movie frame/cache transfer',
                  final_scientific_visual_review_pending=True,
                  current_audit_log_environment_storage_and_terminal_accounting_to_freeze_after_exit=True)
    lines = ['# Session7 closed native integrity and retained storage', '',
             'All raw data, samples and movie caches/frames remain on AMD. No deletion.', '',
             '| Retained root | Files | Logical bytes | Allocated bytes |', '|---|---:|---:|---:|']
    lines += [f'| `{ROOT/key}` | {value["files"]} | {value["logical_bytes"]} | {value["allocated_bytes"]} |'
              for key, value in sorted(groups.items())]
    lines += ['', 'All recorded native output hashes were checked again. This inventory includes sidecars outside out/.',
              'The generated-product manifest is a snapshot before actual scientific/visual review.',
              'Current audit log/environment/storage and terminal job accounting must be frozen after this job exits.',
              '', f'Session allocated bytes: {sizes["session_allocated_bytes"]}. Whole-user allocated bytes: {sizes["whole_user_allocated_bytes"]}.', '']
    report_path.write_text('\n'.join(lines))
    atomic(ROOT/'evidence/NATIVE_INTEGRITY_QA.json', result)
    print('CLOSED_NATIVE_INTEGRITY_OK; SCIENTIFIC_VISUAL_CLOSEOUT_STILL_REQUIRED', len(index), flush=True)


if __name__ == '__main__':
    main()
