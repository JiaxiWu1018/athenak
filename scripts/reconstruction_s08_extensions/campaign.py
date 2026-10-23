"""Isolated extension initialization, validation and bounded evolution."""
import sys,time,shutil
import numpy as np
from common import *
from extension_resources import admission, EXT
from checkpoint import validate
from run_segment import run
BASE=Path(os.environ['ST8X_BASE']);CASE=CASES[0]

def section(text,name,entries):
    import re
    pattern=r'(?ms)^<'+re.escape(name)+r'>\n(.*?)(?=^<|\Z)';m=re.search(pattern,text);body=m.group(1) if m else ''
    for k,v in entries.items():
        body=re.sub(r'(?m)^'+re.escape(k)+r'\s*=.*\n?','',body);body+=f'{k} = {v}\n'
    block=f'<{name}>\n'+body.rstrip()+'\n\n'
    return text[:m.start()]+block+text[m.end():] if m else text+'\n'+block

def setup(root,U,exe):
    for d in ['inputs','samples','protected','evidence','runs/yc0500']: (root/d).mkdir(parents=True,exist_ok=True)
    for dest,source in [(root/'samples/yc0500.bin',BASE/'samples/yc0500.bin'),(root/'protected/athena',exe)]:
        if not dest.exists():dest.symlink_to(source)
    text=(BASE/'inputs/yc0500.athinput').read_text()
    text=section(text,'time',dict(tlim=500 if U==0 else 300))
    text=section(text,'problem',dict(cluster_sample_file=root/'samples/yc0500.bin',cluster_t0_dump=root/'runs/yc0500/initial_particles.bin',cluster_radial_mode_u=U,cluster_solve_momentum_constraint='true' if U!=0 or root.name=='control' else 'false',cluster_solve_hamiltonian_constraint='true' if U!=0 or root.name=='control' else 'false'))
    (root/'inputs/yc0500.athinput').write_text(text)


def build():
    exe=EXT/'protected/athena'
    if exe.exists():return exe
    repo=EXT/'src/athenak';build=EXT/'build';out=EXT/'evidence'/('build_'+job());out.mkdir(parents=True,exist_ok=True)
    admission()
    for p in ['source.tar','kokkos.tar']:
        atomic(out/(p+'.json'),receipt(EXT/p))
    if not repo.exists():
        repo.mkdir(parents=True)
        cmd(['tar','-xf',EXT/'source.tar','-C',repo]);cmd(['tar','-xf',EXT/'kokkos.tar','-C',repo/'kokkos'])
    args=['cmake','-S',repo,'-B',build,'-DKokkos_ENABLE_HIP=On','-DKokkos_ARCH_AMD_GFX90A=On','-DCMAKE_C_COMPILER=cc','-DCMAKE_CXX_COMPILER=hipcc','-DAthena_ENABLE_MPI=ON','-DCMAKE_CXX_FLAGS=-O3','-DCMAKE_C_FLAGS=-O3','-DPROBLEM=particles/relativistic_cluster']
    with (out/'configure.log').open('w') as f:cmd(args,stdout=f,stderr=subprocess.STDOUT)
    with (out/'build.log').open('w') as f:cmd(['cmake','--build',build,'-j',os.environ.get('SLURM_CPUS_PER_TASK','16')],stdout=f,stderr=subprocess.STDOUT)
    exe.parent.mkdir(exist_ok=True);shutil.copy2(build/'src/athena',exe);shutil.copy2(build/'CMakeCache.txt',out/'CMakeCache.txt')
    atomic(out/'BUILD_OK.json',dict(utc=now(),job=job(),executable=receipt(exe),source=load(EXT/'evidence/SOURCE_PIN.json')))
    return exe


def constraint_reference():
    repo=EXT/'src/athenak';text=(repo/'src/pgen/particles/relativistic_cluster.cpp').read_text()
    parts=text[text.index('struct ClusterState {'):text.index('std::size_t CDFIndex(')]
    parts+=text[text.index('struct CTTMomentumProfile {'):text.index('struct BiasStats {')]
    header='#include <algorithm>\n#include <cmath>\n#include <cstdlib>\n#include <iostream>\n#include <iomanip>\n#include <limits>\n#include <vector>\nusing Real=double;\n'
    main=r"""
int main(int argc,char**argv){
  double U=std::atof(argv[1]);auto p=ConstructProfile(.5,1e-4);
  auto c=ConstructIncrementalConstraintProfile(p,1.,U,true);
  std::cerr<<"U="<<U<<" iterations="<<c.iterations<<" rel_update="<<c.final_relative_update<<" delta_adm="<<c.delta_adm_mass<<"\n";
  std::cout<<std::setprecision(17);
  for(std::size_t i=0;i<c.radius.size();++i)std::cout<<c.radius[i]<<" "<<c.conformal_a[i]<<" "<<c.a[i]<<"\n";
}
"""
    # The reference source is preserved; momentum transformation is independently
    # reconstructed in Python, with the solver profile bound to this executable.
    f=EXT/'evidence/constraint_reference.cpp';f.write_text(header+parts+main)
    cmd(['g++','-O3','-std=c++17',f,'-o',EXT/'evidence/constraint_reference'])
    for U in [-.02,-.05]:
        with (EXT/'evidence'/f'constraint_{abs(U):.3f}.txt').open('w') as out, (EXT/'evidence'/f'constraint_{abs(U):.3f}.log').open('w') as err:
            cmd([EXT/'evidence/constraint_reference',str(U)],stdout=out,stderr=err)


def initialize(mode,U):
    root=EXT/mode;exe=EXT/'protected/athena';setup(root,U,exe)
    if (root/'runs/yc0500/INITIAL_OK.json').exists():return
    env=os.environ.copy();env.update(ROOT=str(root),ST8X_U=str(U))
    cmd(['python3',EXT/'scripts/run_segment.py','yc0500','--initialize','--target','300','--wall-seconds','900'],env=env)


def constraint_gate(mode):
    control=EXT/'control/runs/yc0500';work=EXT/mode/'runs/yc0500'
    a=np.atleast_2d(np.loadtxt(control/'yc0500.user.hst'))[0];b=np.atleast_2d(np.loadtxt(work/'yc0500.user.hst'))[0]
    ratios={name:float(np.sqrt(b[k]/a[k])) for name,k in [('H_r10',3),('M_r10',4),('H_r16',9),('M_r16',10)]}
    ga=np.atleast_2d(np.loadtxt(control/'yc0500.z4c.user.hst'))[0];gb=np.atleast_2d(np.loadtxt(work/'yc0500.z4c.user.hst'))[0]
    ratios.update(H_global=float(np.sqrt(gb[3]/ga[3])),M_global=float(np.sqrt(gb[4]/ga[4])))
    assert np.isfinite(list(ratios.values())).all() and max(ratios.values())<=1.005,ratios
    record=dict(job=job(),utc=now(),case='yc0500',constraint_norm_ratios=ratios,tolerance=1.005,control=receipt(control/'yc0500.user.hst'),perturbed=receipt(work/'yc0500.user.hst'))
    atomic(ROOT/'evidence/PRODUCTION_ADMISSION.json',record)
    return record


def long():
    setup(ROOT,0,BASE/'protected/athena');work=ROOT/'runs/yc0500';state=work/'state.json'
    if not state.exists():
        source=BASE/'runs/yc0500/rst/yc0500.00008.rst';v=validate(source)
        assert v['sha256']=='9b8ee147663dfb9ee74571660324268cc971ea4ca783b3877de335b843c6e28c' and v['time']==300
        (work/'rst').mkdir(exist_ok=True);destination=work/'rst'/source.name;shutil.copy2(source,destination)
        v=validate(destination);v.pop('parameters')
        atomic(state,dict(segments=[],latest=v,complete=False))
        atomic(work/'INITIAL_OK.json',dict(inherited_original=True,source=receipt(source),utc=now(),job=job()))
        from restart_identity import gate
        gate(destination)
    if not (ROOT/'evidence/PRODUCTION_ADMISSION.json').exists() or load(ROOT/'evidence/PRODUCTION_ADMISSION.json').get('ranks')!=int(os.environ['SLURM_NTASKS']):
        from restart_identity import gate
        gate(load(state)['latest']['path'])
    state=run(CASE,target=500,wall_seconds=120,suppress_segment_3d=True)
    rate=state['segments'][-1].get('native_node_hours_per_M')
    atomic(ROOT/'evidence/BENCHMARK.json',dict(job=job(),utc=now(),native_node_hours_per_M=rate,remaining_M=500-state['latest']['time']))
    # A queued four-GPU node was at least two hours away at submission. Avoid a
    # slow single-GPU long run when its measured forecast loses that advantage.
    if rate and rate<=.027 and not (work/'REQUEST_STOP').exists():
        remain=int(os.environ['ST8X_WALL_SECONDS'])-(time.time()-float(os.environ['ST8X_START_EPOCH']))-240
        if remain>180:state=run(CASE,target=500,wall_seconds=int(remain),suppress_segment_3d=True)
    from reduce import reduce_case
    reduce_case(CASE,require_complete=state['complete'])


def inward(mode):
    U=-.02 if mode=='inward0020' else -.05
    if mode=='inward0050':
        prior=load(EXT/'inward0020/runs/yc0500/state.json')
        if not prior['complete'] or prior['latest']['time']<300-1e-8 or prior['latest']['ah_onset']>=0:
            atomic(ROOT/'evidence/FALLBACK_NOT_REQUIRED.json',dict(utc=now(),job=job(),prior=prior['latest']));return
    build()
    if not (EXT/'evidence/constraint_0.020.txt').exists():constraint_reference()
    initialize('control',0);initialize(mode,U);constraint_gate(mode)
    work=ROOT/'runs/yc0500';state=load(work/'state.json')
    if not state['complete']:
        remain=int(os.environ['ST8X_WALL_SECONDS'])-(time.time()-float(os.environ['ST8X_START_EPOCH']))-600
        state=run(CASE,target=300,wall_seconds=max(120,int(remain)),suppress_segment_3d=True)
    from reduce import reduce_case
    reduce_case(CASE,require_complete=state['complete'])

if __name__=='__main__':
    job();mode=sys.argv[1]
    if mode=='long':long()
    elif mode=='initialize':initialize(sys.argv[2],float(sys.argv[3]))
    else:inward(mode)
