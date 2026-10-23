"""Materialize borrowed submodule objects in the new clone, in Slurm only."""
import fcntl
from common import *

def main():
    job();assert MACHINE=='perseus'
    repo=LOCAL/'src/athenak';out=EXT/'evidence/source_objects';out.mkdir(parents=True,exist_ok=True)
    with (out/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        alt=Path(subprocess.check_output(['git','-C',str(repo/'kokkos'),'rev-parse','--path-format=absolute','--git-path','objects/info/alternates'],text=True).strip())
        if alt.exists():
            original=alt.read_text();atomic(out/'BEFORE.json',dict(utc=now(),job=job(),path=str(alt),value=original))
            cmd(['git','-C',repo/'kokkos','-c','pack.threads=1','repack','-a','-d'])
            preserved=out/'kokkos.original_alternates.txt';assert not preserved.exists()
            alt.rename(preserved)
            try:cmd(['git','-C',repo/'kokkos','fsck','--connectivity-only'],stdout=subprocess.DEVNULL)
            except BaseException:
                preserved.rename(alt);raise
        assert not alt.exists()
        atomic(out/'INDEPENDENT_OK.json',dict(utc=now(),job=job(),kokkos_head=subprocess.check_output(['git','-C',str(repo/'kokkos'),'rev-parse','HEAD'],text=True).strip(),independent_object_store=True,source_tree_modified=False))

if __name__=='__main__':main()
