"""Rebuild only a provenance-bound native revision; retain the prior executable."""
import hashlib,shutil,tarfile
from common import *
from extension_resources import EXT,admission

def main():
    job();admission();pin=load(EXT/'evidence/NATIVE_REVISION.json');rev=EXT/'evidence/native_revisions'/pin['head']
    rev.mkdir(parents=True,exist_ok=True);repo=EXT/'src/athenak';relative='src/pgen/particles/relativistic_cluster.cpp';target=repo/relative
    candidate=EXT/pin['cpp_relative'];assert sha(candidate)==pin['cpp_sha256']
    other=[]
    with tarfile.open(EXT/'source.tar') as archive:
        for member in archive.getmembers():
            if not member.isfile() or not (member.name.startswith('src/') or member.name=='CMakeLists.txt') or member.name==relative:continue
            expected=hashlib.sha256(archive.extractfile(member).read()).hexdigest();actual=receipt(repo/member.name)
            assert actual['sha256']==expected,member.name
            other.append(actual)
    old=receipt(target);before=rev/'before.cpp'
    if not before.exists():shutil.copy2(target,before)
    assert sha(before)==old['sha256'] or old['sha256']==pin['cpp_sha256']
    shutil.copy2(candidate,target)
    with (rev/'build.log').open('w') as out:cmd(['cmake','--build',EXT/'build','-j',os.environ.get('SLURM_CPUS_PER_TASK','16')],stdout=out,stderr=subprocess.STDOUT)
    exe=EXT/'protected'/('athena_'+pin['head'][:12]);built=EXT/'build/src/athena'
    if exe.exists():assert sha(exe)==sha(built)
    else:shutil.copy2(built,exe)
    shutil.copy2(EXT/'build/CMakeCache.txt',rev/'CMakeCache.txt')
    record=dict(utc=now(),job=job(),head=pin['head'],path=str(exe),executable=receipt(exe),cpp_before=old,cpp_after=receipt(target),all_other_native_files_match_original_archive=other,base_source_pin=load(EXT/'evidence/SOURCE_PIN.json'),original_executable_retained=receipt(EXT/'protected/athena'))
    atomic(rev/'BUILD_OK.json',record);atomic(EXT/'evidence/NATIVE_ACTIVE.json',record)
    # Preserve the failed preflight receipt before the latest-preflight pointer changes.
    prior=EXT/'evidence/STATIC_PREFLIGHT.json'
    if prior.exists():
        archived=EXT/'evidence'/('STATIC_PREFLIGHT_'+load(prior)['job']+'.json')
        if not archived.exists():shutil.copy2(prior,archived)
    print(json.dumps(dict(head=pin['head'],executable=record['executable']),indent=2))
if __name__=='__main__':main()
