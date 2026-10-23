"""Shared preparation and evidence checks. Never launches a native application."""
import csv
import json
import math
import struct
import subprocess
from pathlib import Path

from prepare import CODE, HERE, SESSION, parse, sha, source_hash

BRANCH = 'project/GI-in-cluster'
KINDS = ('gi', 'os', 'spin')
CASES = {
    'compact_gi': ('gi', 'compact_validation.athinput', (), 600),
    'os_reference': ('os', 'os_validation.athinput', (), 3300),
    'os_bad_fallback_restart': ('os', 'os_validation.athinput',
                                ('fastflow/initial_radius_0=0.01',), 1500),
    'spin_coarse': ('spin', 'spin_validation.athinput',
                    ('fastflow/ntheta=10', 'fastflow/lmax=6'), 300),
    'spin_fine': ('spin', 'spin_validation.athinput',
                  ('fastflow/ntheta=20', 'fastflow/lmax=8'), 300),
}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=CODE, text=True,
                                   timeout=30).strip()


def check_inputs():
    for path in sorted((HERE / 'inputs').glob('*validation.athinput')):
        deck = parse(path.read_text())
        for block in ('mesh', 'meshblock'):
            for key in ('nx1', 'nx2', 'nx3'):
                if int(deck.get(block, {}).get(key, '0')) < 1:
                    raise RuntimeError(f'{path.name}: missing/invalid {block}/{key}')
    before = parse((HERE / 'inputs/session010_reference.athinput').read_text())
    after = parse((HERE / 'inputs/production.athinput').read_text())
    allowed = {('job', 'basename'), ('time', 'tlim'), ('time', 'nlim'),
               ('fastflow', 'verbose'), ('fastflow', 'diagnostic_dt'),
               ('particles', 'orbit_diagnostic_dt'), ('output7', 'last_time'),
               ('z4c', 'nrad_wave_extraction'), ('z4c', 'extraction_radius_1'),
               ('z4c', 'extraction_radius_2'), ('z4c', 'co_0_out_every'),
               ('z4c', 'co_1_out_every')}
    allowed.update(('fastflow', f'stop_time_{i}') for i in range(3))
    allowed.update((f'output{i}', 'dt') for i in range(1, 22))
    for block in set(before) | set(after):
        for key in set(before.get(block, {})) | set(after.get(block, {})):
            if (before.get(block, {}).get(key) != after.get(block, {}).get(key)
                    and (block, key) not in allowed):
                raise RuntimeError(f'Unapproved production change: {block}/{key}')
    if float(after['time']['tlim']) != 150 or int(after['time']['nlim']) != -1:
        raise RuntimeError('Production must be fresh t0 with strict tlim=150, nlim=-1')
    required = {('z4c', 'nrad_wave_extraction'): 3,
                ('z4c', 'extraction_radius_0'): 40, ('z4c', 'extraction_radius_1'): 50,
                ('z4c', 'extraction_radius_2'): 60, ('z4c', 'waveform_dt'): .025,
                ('particles', 'orbit_diagnostic_dt'): .025,
                ('fastflow', 'diagnostic_dt'): .025, ('output7', 'last_time'): -1}
    for i in range(1, 22):
        required[f'output{i}', 'dt'] = (.025 if i in (1, 8) else .5 if i == 5
                                       else 10 if i in (6, 7, 18, 19, 20, 21) else .25)
    for i in range(3):
        required['fastflow', f'stop_time_{i}'] = 150
    for (block, key), expected in required.items():
        if float(after[block][key]) != expected:
            raise RuntimeError(f'Wrong production output/stop contract: {block}/{key}')


def manifest():
    check_inputs()
    native = source_hash()
    kokkos = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE / 'kokkos',
                                     text=True, timeout=15).strip()
    if kokkos != '6739bc623081648af9e752b616d9671527922cbf':
        raise RuntimeError('Unexpected Kokkos revision')
    executables = {}
    for kind in KINDS:
        receipt = json.loads((SESSION / f'evidence/build/{kind}.json').read_text())
        exe = SESSION / f'build/{kind}/src/athena'
        if (receipt.get('compiled') is not True
                or receipt.get('native_source_sha256') != native
                or Path(receipt['executable']).resolve() != exe.resolve()
                or receipt.get('executable_sha256') != sha(exe)
                or receipt.get('kokkos_commit') != kokkos):
            raise RuntimeError(f'Stale native build: {kind}')
        executables[kind] = receipt['executable_sha256']
    paths = sorted([*HERE.glob('*.py'), *HERE.glob('*.sh'), *HERE.glob('*.sbatch'),
                    *HERE.glob('inputs/*.athinput')])
    return dict(schema_version=2, native_source_sha256=native,
                kokkos_commit=kokkos, executable_sha256=executables,
                files={str(p.relative_to(CODE)): sha(p) for p in paths},
                cases={label: dict(kind=k, input=d, overrides=list(e), limit_seconds=s)
                       for label, (k, d, e, s) in CASES.items()})


def validated(current=None):
    current = manifest() if current is None else current
    path = SESSION / 'evidence/VALIDATION_PASS.json'
    receipt = json.loads(path.read_text())
    if receipt.get('passed') is not True or receipt.get('manifest') != current:
        raise RuntimeError('Missing, legacy, or stale native GPU validation')
    job = str(receipt['validation_job'])
    if not job.isdigit():
        raise RuntimeError('Invalid validation job ID')
    root = SESSION / 'runtime' / ('validation_' + job)
    if (receipt.get('outputs') != str(root)
            or json.loads((root / 'PASS.json').read_text()) != receipt
            or (root / 'FAILURE.json').exists()):
        raise RuntimeError('Validation run evidence is missing or failed')
    for label in CASES:
        result = json.loads((root / label / 'RESULT.json').read_text())
        if result.get('passed') is not True or result.get('job_id') != job:
            raise RuntimeError('Incomplete native validation: ' + label)
    artifacts = receipt.get('artifacts', {})
    required = {str(Path(label) / name) for label in CASES
                for name in ('COMMAND.json', 'RESULT.json', 'athena.log')}
    if not required.issubset(artifacts):
        raise RuntimeError('Incomplete validation artifact manifest')
    for relative, digest in artifacts.items():
        p = (root / relative).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p) != digest:
            raise RuntimeError('Changed/missing validation artifact: ' + relative)
    cp = Path(receipt['checkpoint'])
    if not cp.resolve().is_relative_to(root.resolve()) or sha(cp) != receipt['checkpoint_sha256']:
        raise RuntimeError('Changed/missing validated restart checkpoint')
    return receipt


def ready(verify_remote=False):
    current = manifest()
    validation = validated(current)
    push = json.loads((SESSION / 'evidence/PUSH_VERIFIED.json').read_text())
    if (push.get('passed') is not True or push.get('manifest') != current
            or push.get('validation_receipt_sha256') != sha(SESSION / 'evidence/VALIDATION_PASS.json')
            or push.get('validation_job') != validation['validation_job']
            or push.get('commit') != git('rev-parse', 'HEAD')
            or git('branch', '--show-current') != BRANCH or git('status', '--porcelain')):
        raise RuntimeError('Tested clean fast-forward push has not been verified for this checkout')
    if verify_remote:
        remote = git('ls-remote', 'origin', 'refs/heads/' + BRANCH).split()
        if not remote or remote[0] != push['commit']:
            raise RuntimeError('Remote campaign branch no longer matches the verified commit')
    return current


def checkpoint_header(path):
    """Read only the native ARM/double mesh header, common to all three builds."""
    with Path(path).open('rb') as stream:
        data = stream.read(1024 ** 2)
        end = data.index(b'<par_end>') + 10
        deck = parse(data[:end - 10].decode())
        stream.seek(end + 232)
        tm, dt, cycle = struct.unpack('<ddi', stream.read(20))
    if not math.isfinite(tm) or not math.isfinite(dt) or dt <= 0 or cycle < 0:
        raise RuntimeError('Invalid native checkpoint state')
    return dict(time=tm, dt=dt, cycle=cycle, deck=deck, path=str(path))


def endpoint(out, *, target_time=None, target_cycle=None):
    states = [checkpoint_header(p) for p in Path(out).glob('rst/*.rst')]
    if not states:
        raise RuntimeError('Missing native final checkpoint')
    final = max(states, key=lambda r: (r['time'], r['cycle']))
    if ((target_time is not None and abs(final['time'] - target_time) > 1.e-10)
            or (target_cycle is not None and final['cycle'] != target_cycle)):
        raise RuntimeError('Validation stopped before its required endpoint')
    return {k: final[k] for k in ('time', 'dt', 'cycle', 'path')}


def read_rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def horizon_quality(rows):
    accepted = [r for r in rows if r['published'] == '1']
    if not accepted:
        raise RuntimeError('No accepted horizon history')
    for row in accepted:
        values = [float(row[k]) for k in ('mass', 'spin', 'area', 'hmean', 'rmin', 'rmax', 'local_dx')]
        m, spin, area, hmean, rmin, rmax, dx = values
        if (not all(map(math.isfinite, values)) or m <= 0 or area <= 0 or dx <= 0
                or row['found'] != '1' or int(row['streak']) < 3
                or abs(hmean) / area > .2 or rmin < 4 * dx or rmax > 4):
            raise RuntimeError('Published horizon violates the unchanged validation gates')
    return accepted


def fresh_persistence(path):
    candidates = read_rows(path)
    good = 0
    for row in candidates:
        streak = int(row['quality_streak'])
        if row['quality_geometry_ok'] == '1':
            # Failed finds are absent from this legacy table and may reset the streak.
            good = good + 1 if streak != 1 else 1
            if streak != good:
                raise RuntimeError('Restart persistence did not start with fresh finds')
        else:
            good = 0
        if row['published_this_candidate'] == '1' and (good < 3 or streak < 3):
            raise RuntimeError('Restart published before three fresh successful finds')
    if not any(r['published_this_candidate'] == '1' for r in candidates):
        raise RuntimeError('Restart never published a fresh horizon')


def startup_census(rows):
    """Require all three actual t0 model rows, independent of file row order."""
    initial = list(rows)[:3]
    if len(initial) < 3:
        raise RuntimeError('Missing full-size initial census')
    initial.sort(key=lambda r: int(r['component']))
    if ([int(r['component']) for r in initial] != [0, 1, 2]
            or [int(r['count']) for r in initial] != [3000000, 1000000, 1000000]
            or any(int(r['cycle']) != 0 or float(r['time']) != 0
                   or int(r['leaf_blocks']) != 4936 for r in initial)
            or any(not math.isfinite(float(r['rest_mass'])) or float(r['rest_mass']) <= 0
                   for r in initial)):
        raise RuntimeError('Full-size initial census/model/mesh does not match the frozen input')
    # Native rest mass is distinct from the source masses 0.76/0.12/0.12.
    return dict(initial_counts=[int(r['count']) for r in initial], leaf_blocks=4936,
                component_rest_masses=[float(r['rest_mass']) for r in initial])
