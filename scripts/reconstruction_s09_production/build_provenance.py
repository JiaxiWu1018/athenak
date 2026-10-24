"""Bind reused native executable to preserved source revision and build context."""
import shutil
from common import *


def main():
    job()
    prior = Path('/work1/eliasmost/jiaxiwu/st_s08_yc0500_extensions_20261023')
    native = load(prior / 'evidence/NATIVE_REVISION.json')
    assert native['head'] == 'db942607bd251dab36bbd4a022ec58ee62317d01'
    cpp = prior / native['cpp_relative']
    assert sha(cpp) == native['cpp_sha256']
    current = EXT / 'src/athenak/src/pgen/particles/relativistic_cluster.cpp'
    assert sha(current) == sha(cpp), 'The reused executable pgen and current source differ'
    destination = EXT / 'evidence/native_build'
    destination.mkdir(exist_ok=True)
    files = [prior / 'build/CMakeCache.txt', prior / 'evidence/NATIVE_REVISION.json', cpp]
    entries = []
    for source in files:
        target = destination / source.name
        if not target.exists():
            shutil.copy2(source, target)
        assert sha(target) == sha(source)
        entries.append(dict(source=receipt(source), destination=receipt(target)))
    cache = (destination / 'CMakeCache.txt').read_text()
    flags = [line for line in cache.splitlines()
             if line and not line.startswith(('#', '//')) and
             any(line.startswith(prefix) for prefix in
                 ['CMAKE_C_COMPILER:', 'CMAKE_CXX_COMPILER:', 'CMAKE_CXX_FLAGS:',
                  'CMAKE_C_FLAGS:', 'Kokkos_ENABLE_HIP:', 'Kokkos_ARCH_AMD_GFX90A:',
                  'Athena_ENABLE_MPI:', 'PROBLEM:'])]
    settings = {line.split(':', 1)[0]: line.split('=', 1)[1]
                for line in cache.splitlines()
                if '=' in line and not line.startswith(('#', '//'))}
    assert settings['Kokkos_ENABLE_HIP'].upper() == 'ON'
    assert settings['Kokkos_ARCH_AMD_GFX90A'].upper() == 'ON'
    executable = receipt(EXT / 'protected/athena')
    assert executable['sha256'] == 'fa261c0bd46ea96cc57e8df593946a05f2ea72ffc53863c7d55f902f69c0cafc'
    atomic(EXT / 'evidence/NATIVE_BUILD_PROVENANCE.json',
           dict(utc=now(), job=job(), native_revision=native,
                executable=executable, pgen_source_matches=True,
                original_build_context=entries, flags=flags,
                scope='Reused immutable executable. No new build; original machine/toolchain context and incremental pgen revision are retained separately.'))


if __name__ == '__main__':
    main()
