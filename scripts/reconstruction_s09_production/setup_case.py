"""Preserve original realizations and model-specific initial hierarchies."""
import re
import shutil
from common import *

def section(text,name,entries):
    pattern=r'(?ms)^<'+re.escape(name)+r'>\n(.*?)(?=^<|\Z)'
    match=re.search(pattern,text);body=match.group(1) if match else ''
    for key,value in entries.items():
        body=re.sub(r'(?m)^'+re.escape(key)+r'\s*=.*\n?','',body)
        body+=f'{key} = {value}\n'
    block=f'<{name}>\n'+body.rstrip()+'\n\n'
    return text[:match.start()]+block+text[match.end():] if match else text+'\n'+block

def setup(c,U=None,control=False):
    job();U=c['U'] if U is None else U
    for name in ['inputs','samples','protected','evidence','runs/'+c['id']]:
        (ROOT/name).mkdir(parents=True,exist_ok=True)
    base=Path(os.environ['ST8X_BASE'])
    sample=base/'samples'/f"{c['id']}.bin"
    exe=EXT/'protected/athena'
    assert sample.exists() and exe.exists(),(str(sample),str(exe))
    for destination,source in [(ROOT/'samples'/f"{c['id']}.bin",sample),(ROOT/'protected/athena',exe)]:
        if destination.exists():assert destination.resolve()==source.resolve()
        else:destination.symlink_to(source)
    text=(base/'inputs'/f"{c['id']}.athinput").read_text()
    text=section(text,'time',dict(tlim=c['target'],stop_check_every=1))
    text=section(text,'mesh_refinement',dict(num_levels=c['num_levels']))
    text=section(text,'particles',dict(excise_lapse=.1,excise_ah='false'))
    source=(base/'inputs/yc0500.athinput').read_text()
    horizon=re.search(r'(?ms)^<fastflow>\n(.*?)(?=^<|\Z)',source).group(1)
    values={line.split('=',1)[0].strip():line.split('=',1)[1].strip() for line in horizon.splitlines() if '=' in line}
    text=section(text,'fastflow',values)
    text=section(text,'problem',dict(cluster_sample_file=ROOT/'samples'/f"{c['id']}.bin",cluster_t0_dump=ROOT/'runs'/c['id']/'initial_particles.bin',cluster_radial_mode_u=U,cluster_solve_momentum_constraint='true' if U!=0 or control else 'false',cluster_solve_hamiltonian_constraint='true' if U!=0 or control else 'false',production_history_dt=.25,production_cohort_dt=.5))
    text=section(text,'output1',dict(dt=.25))
    for index in [2,3,4,5]:text=section(text,'output'+str(index),dict(dt=1))
    text=section(text,'mesh_refinement',dict(max_nmb_per_rank=rank_capacity()))
    if control:
        for index in [6,7]:text=section(text,'output'+str(index),dict(dt=0))
    path=ROOT/'inputs'/f"{c['id']}.athinput"
    if path.exists():
        original=path.read_text()
        if original!=text:
            # Preserve the initialization deck. A topology continuation changes
            # only the runtime capacity, explicitly passed on every native CLI.
            anchor=load(ROOT/'evidence/EXACT_SETUP.json')
            assert sha(path)==anchor['input']['sha256'],'Original input changed'
            normalized=section(original,'mesh_refinement',dict(max_nmb_per_rank=rank_capacity()))
            # section() can add empty lines when called on an already rendered
            # deck. Every nonempty line must still match exactly and the stored
            # original bytes must retain the initialization checksum above.
            assert [line for line in normalized.splitlines() if line.strip()]==[line for line in text.splitlines() if line.strip()],'Existing exact input differs beyond runtime capacity and empty lines'
            atomic(ROOT/'evidence'/f'INPUT_CAPACITY_OVERRIDE_{job()}.json',dict(
                utc=now(),job=job(),unchanged_initial_input=receipt(path),
                runtime_max_nmb_per_rank=rank_capacity(),
                reason='Allocated topology changed; original deck retained, capacity overridden explicitly in zero-step gate and evolution CLI.'))
    else:path.write_text(text)
    metadata=ROOT/'evidence/samples'/f"{c['id']}.json";metadata.parent.mkdir(exist_ok=True)
    old=base/'evidence/samples'/f"{c['id']}.json"
    if not metadata.exists():
        if old.exists():shutil.copy2(old,metadata)
        else:
            from st_migration_lib import cluster_profile
            p=cluster_profile(c['yc'],dx=1e-4)
            atomic(metadata,dict(yc=c['yc'],Riso=float(p['riso_over_m'][-1]),R=float(p['R_over_M']),M0=float(p['M0_over_M'])))
    record=dict(utc=now(),job=job(),case=c,U=U,control=control,input=receipt(path),sample=receipt(sample),executable=receipt(exe))
    import time
    atomic(ROOT/'evidence/setup'/f'{job()}_{time.time_ns()}.json',record)
    anchor=ROOT/'evidence/EXACT_SETUP.json'
    if not anchor.exists():atomic(anchor,record)
    return ROOT/'runs'/c['id']
