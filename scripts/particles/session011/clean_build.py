"""Compile in fresh trees, retaining all previous builds and evidence. No science."""
import datetime as dt
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid

from prepare import CODE, HERE, SESSION, atomic, sha, source_hash
from abi_check import inspect

PROBLEMS = {'gi': 'particles/gi_cluster', 'os': 'particles/nr_pic_os',
            'spin': 'z4c/kerr_schild'}
KOKKOS = '6739bc623081648af9e752b616d9671527922cbf'


def main(kinds):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'gg':
        raise RuntimeError('Native compilation requires allocated GG')
    if not kinds or len(set(kinds)) != len(kinds) or any(k not in PROBLEMS for k in kinds):
        raise RuntimeError('Specify distinct kinds: gi os spin')
    kokkos = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE/'kokkos', text=True).strip()
    if kokkos != KOKKOS:
        raise RuntimeError('Unexpected Kokkos revision')
    native = source_hash()
    token = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    fresh = SESSION/'build'/('clean_' + token)
    evidence = SESSION/'evidence'/('clean_build_' + token)
    retained = SESSION/'build'/('retained_' + token)
    fresh.mkdir(parents=True, exist_ok=False)
    evidence.mkdir(parents=True, exist_ok=False)
    previous = SESSION/'evidence/build'
    if previous.exists():
        shutil.copytree(previous, evidence/'previous_receipts')
    record = dict(epoch=time.time(), job_id=os.environ['SLURM_JOB_ID'],
                  host=socket.gethostname(), native_source_sha256=native,
                  build_directory=str(fresh), retained_directory=str(retained),
                  compiled=False, simulation_tested=False, kinds=kinds,
                  checkout_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip(),
                  loaded_modules=os.environ.get('LOADEDMODULES', ''))
    atomic(evidence/'BUILD.json', record)
    try:
        for kind in kinds:
            bdir = fresh/kind
            args = ['cmake', '-S', str(CODE), '-B', str(bdir),
                    '-DCMAKE_PROJECT_INCLUDE='+str(HERE/'kokkos_4_7_2_arm_sve_compat.cmake'),
                    '-DKokkos_ENABLE_CUDA=On', '-DKokkos_ARCH_HOPPER90=On',
                    '-DKokkos_ARCH_ARMV9_GRACE=On', '-DCMAKE_C_COMPILER=gcc',
                    '-DCMAKE_CXX_COMPILER='+str(CODE/'kokkos/bin/nvcc_wrapper'),
                    '-DAthena_ENABLE_MPI=ON', '-DCMAKE_BUILD_TYPE=Release',
                    '-DPROBLEM='+PROBLEMS[kind]]
            print('Clean compilation:', kind, 'logs:', evidence, flush=True)
            with (evidence/(kind+'.configure.log')).open('w') as out:
                subprocess.run(args, check=True, stdout=out, stderr=subprocess.STDOUT)
            with (evidence/(kind+'.build.log')).open('w') as out:
                subprocess.run(['cmake', '--build', str(bdir), '-j',
                                os.environ.get('S11_BUILD_JOBS', '16')],
                               check=True, stdout=out, stderr=subprocess.STDOUT)
            if source_hash() != native:
                raise RuntimeError('Native sources changed during compilation; builds not promoted')
            layout = inspect(bdir/'src/athena')
            atomic(evidence/(kind+'.abi.json'), layout)
            if not layout['consistent']:
                raise RuntimeError('Particle energy routines disagree on the compiled class layout')
        # All requested builds must succeed before changing any launch path.
        retained.mkdir(exist_ok=False)
        receipts = SESSION/'evidence/build'
        receipts.mkdir(exist_ok=True)
        for kind in kinds:
            target = SESSION/'build'/kind
            if target.exists() or target.is_symlink():
                target.rename(retained/kind)
            target.symlink_to((fresh/kind).relative_to(target.parent), target_is_directory=True)
            exe = target/'src/athena'
            receipt = dict(kind=kind, executable=str(exe), executable_sha256=sha(exe),
                           native_source_sha256=native, job_id=os.environ['SLURM_JOB_ID'],
                           host=socket.gethostname(), epoch=time.time(), compiled=True,
                           simulation_tested=False, kokkos_commit=KOKKOS,
                           clean_build=True, build_directory=str(fresh/kind),
                           build_evidence=str(evidence), previous_build=str(retained/kind))
            atomic(receipts/(kind+'.json'), receipt)
        record.update(compiled=True, executable_sha256={
            k: sha(SESSION/'build'/k/'src/athena') for k in kinds})
        print('Clean builds complete; no native simulation performed.', flush=True)
    except BaseException as exc:
        record['error'] = repr(exc)
        raise
    finally:
        atomic(evidence/'BUILD.json', record)


if __name__ == '__main__':
    main(sys.argv[1:])
