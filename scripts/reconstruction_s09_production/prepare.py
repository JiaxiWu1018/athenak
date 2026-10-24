"""Compute-node provenance, sample hashes and conservative full-cap forecast."""
import shutil, socket
from common import *
from extension_resources import admission

def main():
    job();disk=admission();base=Path(os.environ['ST8X_BASE'])
    (EXT/'protected').mkdir(exist_ok=True)
    source=Path('/work1/eliasmost/jiaxiwu/st_s08_yc0500_extensions_20261023/protected/athena_db942607bd25')
    exe=EXT/'protected/athena'
    if not exe.exists():shutil.copy2(source,exe)
    assert sha(source)==sha(exe)
    records=[]
    for c in ALL_CASES:
        ident=f"yc{round(c['yc']*1000):04d}"
        f=base/'samples'/f'{ident}.bin';meta=load(base/'evidence/samples'/f'{ident}.json')
        r=receipt(f)
        records.append(dict(case=c,sample=r,metadata=meta))
    repo=EXT/'src/athenak'
    native_diff=subprocess.check_output(['git','-C',str(repo),'diff','2d1f2525','--','src','CMakeLists.txt','cmake','kokkos'],text=True)
    assert native_diff==''
    source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    native_source_unchanged=True
    atomic(EXT/'evidence/PREPARATION_OK.json',dict(utc=now(),job=job(),host=socket.gethostname(),native_source_unchanged=True,source_commit=source_commit,native_base='2d1f2525',executable=receipt(exe),original_executable=receipt(source),samples=records,storage=disk,kokkos='6739bc623081648af9e752b616d9671527922cbf'))
    # All four full300M caps, controls, independent restart copies, endpoint fields,
    # four protected checkpoints per case, frames/products and the48GiB clean-stop reserve.
    forecast=dict(streams=4*300*107016371/GIB,controls=20.,restart_identity_copies=24.,protected_checkpoints=96.,fields=12.,source_dependencies=8.,reference_packet=3.,frames_products=20.,reserve=48.)
    assert sum(forecast.values())<400
    atomic(EXT/'evidence/INITIAL_RESOURCE_FORECAST.json',dict(utc=now(),job=job(),components_GiB=forecast,total_GiB=sum(forecast.values()),limit_GiB=400,node_hour_limit=40,basis='Previous measured stream rate, all four full caps, conservative retained checkpoints; refresh after actual initialization/earlyAMR'))
    deployment=dict(source_commit=source_commit,python_files={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')})
    atomic(EXT/'evidence/ENDPOINT_MANIFEST_DEPLOYMENT.json',deployment)
if __name__=='__main__':main()
