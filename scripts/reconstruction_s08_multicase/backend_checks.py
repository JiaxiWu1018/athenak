"""Qualify the same CUDA/Kokkos backend with the initialization-only repair."""
import fcntl
from common import *
from setup_case import section

def main():
    job();assert MACHINE=='perseus'
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'qualification.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        marker=EXT/'evidence/CUDA_BACKEND_CHECKS_OK.json'
        exe=EXT/'protected/athena_inward_db942'
        if marker.exists():
            assert load(marker)['executable']['sha256']==sha(exe)
            return
        for name in ['inputs','samples','protected','tests','evidence']:(ROOT/name).mkdir(exist_ok=True)
        (ROOT/'protected/athena').symlink_to(exe)
        source=SESSION/'perseus/samples/frozen_ah.bin'
        (ROOT/'samples/frozen_ah.bin').symlink_to(source)
        text=(SESSION/'perseus/inputs/frozen_ah.athinput').read_text()
        text=section(text,'problem',dict(cluster_sample_file=ROOT/'samples/frozen_ah.bin'))
        inp=ROOT/'inputs/frozen_ah.athinput';inp.write_text(text)
        work=ROOT/'tests'/('frozen_'+job());work.mkdir()
        args=native_launcher()+[str(exe),'--kokkos-map-device-id-by=mpi_rank','-i',str(inp),'-d',str(work),'problem/cluster_t0_dump='+str(work/'initial_particles.bin')]
        atomic(work/'launch.json',dict(command=args,utc=now(),job=job(),executable=receipt(exe)))
        with (work/'native.log').open('w') as log:cmd(args,stdout=log,stderr=subprocess.STDOUT)
        assert (work/'FROZEN_AH_OK').exists()
        from native_checks import main as checks
        checks()
        atomic(marker,dict(utc=now(),job=job(),executable=receipt(exe),
          frozen_horizon=receipt(work/'FROZEN_AH_OK'),native_checks=load(ROOT/'evidence/NATIVE_CHECKS_OK.json'),
          scope='Preserved CUDA12.9/Kokkos4.7/AMPERE80 backend; full-sample case-specific initialization and matched constraints remain separate gates'))

if __name__=='__main__':main()
