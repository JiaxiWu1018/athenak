"""Allocated source, executable, original sample and 700M-anchor preparation."""
import shutil
import socket
from common import *
from extension_resources import admission
from checkpoint import validate

def preserve(source, destination, expected=None):
    before = receipt(source)
    if expected:
        assert before['sha256'] == expected
    if not destination.exists():
        shutil.copy2(source, destination)
    after = receipt(destination)
    assert before['sha256'] == after['sha256']
    return dict(source=before, destination=after)

def main():
    job()
    disk = admission()
    protected = EXT/'protected'
    protected.mkdir(exist_ok=True)
    records = {}
    records['original_executable'] = preserve(DONOR/'protected/athena', protected/'athena_original', OLD_EXE_SHA)
    perturbation = Path('/work1/eliasmost/jiaxiwu/st_s08_yc0500_extensions_20261023/protected/athena_db942607bd25')
    records['perturbation_executable'] = preserve(perturbation, protected/'athena_perturbation')
    meta = load(BASE/'evidence/samples/yc0500.json')
    records['sample'] = preserve(DONOR/'samples/yc0500.bin', protected/'yc0500.bin')
    old = validate(DONOR/'runs/yc0500/rst/yc0500.00058.rst')
    assert old['sha256'] == ANCHOR_SHA and old['time'] == 700
    assert old['particle_count'] == 2000000 and sum(old['removal_counts']) == 0
    assert old['nblocks'] == 344
    old.pop('parameters')
    repo = EXT/'src/athenak'
    assert not subprocess.check_output(['git', '-C', str(repo), 'diff', '2d1f2525', '--', 'src', 'CMakeLists.txt', 'cmake', 'kokkos'], text=True)
    records.update(utc=now(), job=job(), host=socket.gethostname(), native_source_unchanged=True,
        source_commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        original_700M=old, original_sample_metadata=meta,
        kokkos='6739bc623081648af9e752b616d9671527922cbf', storage=disk)
    atomic(EXT/'evidence/PREPARATION_OK.json', records)
    # Full 2300M new evolution at2M cadence, 23 scientific checkpoints, controls,
    # zero-step copies, latest recoveries, final fields, frames and clean-stop reserve.
    forecast = dict(particles=86., scientific_checkpoints=110., recovery_and_gates=40.,
                    cohorts_slices=24., figures_movies_source=20., reserve=48.)
    assert sum(forecast.values()) < 350
    atomic(EXT/'evidence/INITIAL_RESOURCE_FORECAST.json', dict(utc=now(), job=job(),
        components_GiB=forecast, total_GiB=sum(forecast.values()), limit_GiB=350,
        production_node_hours_estimate=33.5113, node_hour_limit=50,
        basis='Measured179.877M/9435s MI2508x tail; refresh after20M pilot; not a guaranteed cost'))

if __name__ == '__main__':
    main()
