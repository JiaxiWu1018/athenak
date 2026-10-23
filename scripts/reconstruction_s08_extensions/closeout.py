"""Hash closed scientific outputs and assemble source-bound review provenance.

Run inside the final Slurm review allocation. Actual visual acceptance and the
terminal scheduler accounting are recorded separately, after this job closes.
"""
import socket

from common import *
from extension_resources import EXT, storage

BASE = Path(os.environ['ST8X_BASE'])
LIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED'}


def files_under(root):
    # Do not traverse directory links or move any raw data.
    return sorted(p for p in root.rglob('*') if p.is_file())


def bound(path):
    result = receipt(path)
    result['relative_path'] = str(path.relative_to(EXT))
    if path.is_symlink():
        result['link_target'] = os.readlink(path)
        result['resolved_path'] = str(path.resolve())
    return result


def scheduler():
    records = load(EXT / 'evidence/jobs.json')
    ids = {str(r['id']) for r in records}
    assert job() in ids, 'Register the review allocation before closing evidence'
    fields = ['JobIDRaw', 'JobName', 'Partition', 'State', 'ExitCode',
              'ElapsedRaw', 'AllocNodes', 'NTasks', 'Start', 'End', 'NodeList']
    text = subprocess.check_output(
        ['sacct', '-X', '-j', ','.join(sorted(ids)), '-n', '-P',
         '--format=' + ','.join(fields)], text=True, timeout=60)
    rows = [dict(zip(fields, line.split('|'))) for line in text.splitlines()
            if line.strip()]
    assert ids == {r['JobIDRaw'] for r in rows}, 'Missing scheduler records'
    active = [r for r in rows if r['State'].split('+')[0] in LIVE]
    assert all(r['JobIDRaw'] == job() for r in active), active
    for r in rows:
        r['allocated_node_hours'] = (
            int(r['ElapsedRaw'] or 0) * int(r['AllocNodes'] or 0) / 3600)
        r['terminal'] = r['State'].split('+')[0] not in LIVE
    result = dict(utc=now(), job=job(), host=socket.gethostname(), rows=rows,
                  observed_allocated_node_hours=sum(r['allocated_node_hours'] for r in rows),
                  terminal_accounting=False,
                  caveat='Review job is still running. Refresh sacct after its terminal state; node-hours are allocation time, not weighted billing units.')
    atomic(ROOT / 'evidence/ACCOUNTING_OBSERVED.json', result)
    return result


def main():
    job()
    assert load(ROOT / 'evidence/PRODUCTS_READY.json')['numeric']
    assert load(ROOT / 'evidence/PDF_RASTERS.json')['products']
    sources = load(ROOT / 'evidence/SOURCE_MAP.json')['cases']
    raw = []
    closed = []
    controls = set()
    for source in sources:
        mode = source['mode']
        root = EXT / mode
        state = load(root / 'runs/yc0500/state.json')
        assert state['complete']
        assert state['latest']['sha256'] == sha(Path(state['latest']['path']))
        final = load(root / 'runs/yc0500/FINAL_FIELDS_OK.json')
        assert final, ('Missing final native 3D export', mode)
        initial = load(root / 'runs/yc0500/INITIAL_OK.json')
        if mode != 'long':
            assert initial['actual_full_particle_output_checked']
            assert initial['actual_fixed_ID_cohort_checked']
            controls.add('control_' + str(initial['ranks']))
        closed.append(dict(mode=mode, state=bound(root / 'runs/yc0500/state.json'),
                           initial=initial, admission=load(root / 'evidence/PRODUCTION_ADMISSION.json'),
                           final_fields=final, executable=bound(root / 'protected/athena'),
                           input=bound(root / 'inputs/yc0500.athinput'),
                           sample=bound(root / 'samples/yc0500.bin')))
        for subdir in ['runs/yc0500', 'reduced/yc0500', 'inputs']:
            raw.extend(bound(p) for p in files_under(root / subdir))
    for mode in sorted(controls):
        raw.extend(bound(p) for p in files_under(EXT / mode / 'runs/yc0500'))
    raw.extend(bound(p) for p in files_under(ROOT / 'reduced'))
    assert len({p['relative_path'] for p in raw}) == len(raw)
    current_files = {p['path']: p for p in raw}
    for case in closed:
        for expected in case['final_fields']['files']:
            actual = current_files[expected['path']]
            assert actual['sha256'] == expected['sha256'] and actual['bytes'] == expected['bytes'], expected['path']
    atomic(ROOT / 'evidence/CLOSED_DATA_MANIFEST.json', dict(
        utc=now(), job=job(), files=raw, files_count=len(raw),
        logical_bytes=sum(p['bytes'] for p in raw),
        scope='Closed production raw, canonical and assembled review reductions, exact inputs, final 3D fields, matched initial controls. No deletion or relocation. Per-case executable/sample links are bound separately.'))
    # Bind every original reduced prefix used by the continuation without copying
    # or rehashing the entire immutable baseline raw campaign.
    baseline = [bound(p) if EXT in p.parents else receipt(p)
                for p in sorted((BASE / 'reduced/yc0500').glob('*')) if p.is_file()]
    archives = [receipt(EXT / name) for name in ['source.tar', 'kokkos.tar']]
    job_evidence = []
    for mode in [s['mode'] for s in sources] + [ROOT.name]:
        for p in files_under(EXT / mode / 'evidence/jobs'):
            if p.name not in ['telemetry.jsonl', 'LATEST.json', 'watchdog.log']:
                job_evidence.append(bound(p))
    versions = dict(utc=now(), job=job(), native_source_archive=load(EXT / 'evidence/SOURCE_PIN.json'),
                    active_native=load(EXT / 'evidence/NATIVE_ACTIVE.json'),
                    native_revision=load(EXT / 'evidence/NATIVE_REVISION.json'),
                    archives=archives, baseline_reduced_prefix=baseline,
                    cases=closed, job_evidence=job_evidence)
    atomic(ROOT / 'evidence/PROVENANCE.json', versions)
    account = scheduler()
    disk = storage()
    atomic(ROOT / 'evidence/STORAGE_REVIEW.json', dict(
        utc=now(), job=job(), **disk, allocated_bytes_from_du=True,
        retention='All current raw data, full lossless movie frames, logs, scripts, source snapshots and final checkpoints remain on AMD. Only verified superseded extension recoveries may have rotated; their per-case receipts are retained. No earlier-session or Jeans files removed.',
        review_allowance_bytes=200 * GIB, final_combined_storage_pending=True))
    assert disk['extension'] < 199 * GIB and account['observed_allocated_node_hours'] < 39
    (ROOT / 'reports/METHODS_AND_PROVENANCE.md').write_text('''# Session 8 extension methods and provenance

The exact unperturbed continuation reads the original 300M checkpoint (SHA256 `9b8ee147663dfb9ee74571660324268cc971ea4ca783b3877de335b843c6e28c`) and uses the original executable (SHA256 `41411a2e024cb910d2313e9b30ec4370fb37cc9a9dea60cc015c3df3682f1e71`). Its endpoint is 500M or the first accepted AH plus 20M, whichever is earlier. The initial four-to-eight-rank change passed a zero-step identity check on actual particle data and metric interiors. Later topology changes, if any, require the same gate.

Fresh inward cases read the preserved original seed-1 proper-volume stratified realization with 2,000,000 equal-rest-mass particles (sample SHA256 `7c031edd772dd42b3823a0fb26f9566d35732c939c169e7b20271b91f96c4339`). Positions, IDs and rest masses are unchanged. With x equal to equilibrium areal radius divided by the equilibrium surface radius, the coherent Eulerian physical radial velocity increment is U(3x-x^3)/2. Tangential physical velocity is retained and covariant momenta are reconstructed with the new Lorentz factor and corrected spherical conformal factor. U=-0.02 is the primary case. U=-0.05 is required only if the primary reaches 300M without an accepted AH. Each inward endpoint is 300M or first accepted AH plus 20M, whichever is earlier.

The spherical momentum and incremental Hamiltonian initialization use the same proper-volume density as the original sample. An independent angular quadrature and radial Green-function calculation check the continuum correction. Production admission requires actual full-population native output and fixed-ID cohort agreement plus matched U=0 central/global Hamiltonian and momentum L2 ratios no greater than 1.005. The first fixed-mesh preflight failed this unchanged gate. Native revision `db942607bd251dab36bbd4a022ec58ee62317d01` removes spurious interpolation of the exact background and interpolates only its perturbation. The failed artifacts and repaired passing preflight are retained; see INITIALIZATION_REPAIR.md. The native inward executable SHA256 is `fa261c0bd46ea96cc57e8df593946a05f2ea72ffc53863c7d55f902f69c0cafc`.

The baseline mesh, +/-512M domain, gauge, Z4c, RK4, CFL=0.4, GR Boris particle update and conservative deposition remain unchanged. The protected finest initial spacing is 0.25M, with AMR allowed to 0.0625M. Recovery output is nominally every 50M plus segment, AH and final boundaries. Histories use 0.25M, full particle and equatorial slice output 1M, and the fixed 20,000-ID cohort 0.5M, subject to completed-step timing. Inputs and native logs retain the exact settings. Native AH quality, on-grid, residual, resolution and three-search persistence gates are unchanged. Apparent-horizon mass is reported separately from the particle rest-mass ledger.

Every job freezes its deployed Python/shell workflow, native source pin, active executable selector, toolchain, host, Slurm allocation and launch commands. The independent source clone is on branch `experiment/ST-reconstruction-s08-yc0500-extensions` of JiaxiWu1018/athenak. The source archive is commit `9ffe5c2ea45c198d40e74453e7525e8b5289c2c5`, plus the separately preserved native revision; Kokkos is `6739bc623081648af9e752b616d9671527922cbf`. The actual per-job workflow hashes, not an assumed common final workflow, identify executed scripts. Builds use GNU12/OpenMPI4.1.8/CMake3.25.2/prun2.3/ROCm6.4.1, HIP gfx90a, MPI and -O3. OpenMPI vader single-copy remains disabled. PROVENANCE.json and CLOSED_DATA_MANIFEST.json bind exact inputs, source archives, executables, final checkpoints, native outputs, reductions and reused baseline tables.

The inherited particle stress is frozen over the metric RK stages and redeposited after the full step. This source-time coupling remains unchanged and limits claims about the coupled scheme's accuracy. No new convergence study or larger-domain control is included. A finite-time expanding or surviving cluster is not established to be a stable migrated equilibrium. Coordinate shell/lapse dynamics alone do not establish the physical cause of expansion or boundary isolation. Particle rest-mass closure is bookkeeping, not ADM-energy conservation; the inward mode deliberately changes initial energy and angular momentum slightly.

All heavy work runs in Slurm. The additional extension allowance is 40 allocated node-hours and 200GiB, with a one-node-hour finalization reserve; up to four nodes total and two per trajectory. Whole-user AMD disk safeguards and a robust five-minute disk watchdog remain active. Current raw data, reductions, source/build records, logs and lossless frames remain on the cluster. The one-way thin review mirror contains reports, exact inputs, final PDFs/PNGs/MP4s and compact verification receipts. Final terminal accounting, mirror checksum verification and actual visual acceptance are separate closing receipts and must exist before COMPLETE.
''')
    (ROOT / 'reports/RETENTION.md').write_text(
        '# Extension data retention\n\n'
        f'Authoritative AMD extension root: `{EXT}`. Original baseline root: `{BASE}`. '
        'No original-session or Jeans data were deleted or changed. Closed production data and matched controls '
        'are enumerated and SHA256-bound in CLOSED_DATA_MANIFEST.json; baseline reduced-prefix hashes are in PROVENANCE.json. '
        'Full lossless movie frames remain in review/frames. Failed preflights, native revisions and diagnostic logs remain retained. '
        'Only verified superseded extension recovery files may have been rotated under the approved policy; latest two, AH, '
        'final checkpoints and the original 300M anchor are protected. No post-campaign cleanup has been authorized or performed.\n')
    products = []
    for directory in ['plots', 'movies', 'inputs', 'reports']:
        products.extend(bound(p) for p in files_under(ROOT / directory))
    atomic(ROOT / 'evidence/REVIEW_PRODUCT_MANIFEST.json', dict(
        utc=now(), job=job(), products=products,
        bytes=sum(p['bytes'] for p in products),
        actual_visual_acceptance=False,
        pending='Actual scientific/visual review, terminal sacct, combined storage and verified one-way thin mirror'))
    print(json.dumps(dict(closed_files=len(raw), closed_bytes=sum(p['bytes'] for p in raw),
                          review_files=len(products), storage=disk,
                          observed_node_hours=account['observed_allocated_node_hours'])))


if __name__ == '__main__':
    main()
