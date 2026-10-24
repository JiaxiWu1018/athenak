"""Preserve the actual donor build settings alongside the validated executables."""
import shutil
from common import *

def audit():
    job()
    assert sha(EXT/'protected/athena_original') == OLD_EXE_SHA
    assert sha(EXT/'protected/athena_perturbation') == PERTURBED_EXE_SHA
    marker = EXT/'evidence/NATIVE_ARTIFACT_LINKS.json'
    if marker.exists():
        return load(marker)
    records = []
    for name, root in [('original', BASE), ('perturbation', Path('/work1/eliasmost/jiaxiwu/st_s08_yc0500_extensions_20261023'))]:
        cache = root/'build/CMakeCache.txt'
        if cache.exists():
            destination = EXT/'protected'/(name+'_CMakeCache.txt')
            shutil.copy2(cache, destination)
            assert sha(cache) == sha(destination)
            flags = [line for line in cache.read_text().splitlines() if line.startswith((
                'CMAKE_CXX_COMPILER:', 'CMAKE_C_COMPILER:', 'CMAKE_CXX_FLAGS:', 'CMAKE_C_FLAGS:',
                'Kokkos_ENABLE_HIP:', 'Kokkos_ARCH_AMD_GFX90A:', 'Athena_ENABLE_MPI:', 'PROBLEM:'))]
            records.append(dict(kind=name, cache=receipt(destination), original=receipt(cache), flags=flags))
    r = dict(utc=now(), job=job(), original_executable=receipt(EXT/'protected/athena_original'),
        perturbation_executable=receipt(EXT/'protected/athena_perturbation'), build_caches=records,
        original_native_commit='12633257d601d53c8af7a35d77519d681c900f75',
        corrected_initializer_commit='db942607bd25', workflow_base='2d1f2525',
        known_recipe='HIP/GFX90A; gnu12/12.2.0 OpenMPI4/4.1.8 CMake3.25.2 prun2.3 ROCm6.4.1; MPI ON; -O3; PROBLEM=particles/relativistic_cluster',
        scope='Reused validated executable artifacts. No new native compilation or solver change; exact source/input/executable/launch hashes recorded by allocation.')
    atomic(marker, r)
    return r
