"""Preserve the old physical mesh and sample, changing only approved controls."""
import re
import shutil
from common import *

def section(text, name, entries):
    pattern = r'(?ms)^<'+re.escape(name)+r'>\n(.*?)(?=^<|\Z)'
    match = re.search(pattern, text)
    body = match.group(1) if match else ''
    for key, value in entries.items():
        body = re.sub(r'(?m)^'+re.escape(key)+r'\s*=.*\n?', '', body)
        body += f'{key} = {value}\n'
    block = f'<{name}>\n'+body.rstrip()+'\n\n'
    return text[:match.start()]+block+text[match.end():] if match else text+'\n'+block

def setup(c, U=None, control=False):
    job()
    U = c['U'] if U is None else U
    for name in ['inputs', 'samples', 'protected', 'evidence', 'runs/'+c['id']]:
        (ROOT/name).mkdir(parents=True, exist_ok=True)
    sample = EXT/'protected/yc0500.bin'
    exe = EXT/'protected'/('athena_original' if c.get('inherited_time') else 'athena_perturbation')
    for destination, source in [(ROOT/'samples/yc0500.bin', sample), (ROOT/'protected/athena', exe)]:
        if destination.exists():
            assert destination.resolve() == source.resolve()
        else:
            destination.symlink_to(source)
    text = (BASE/'inputs/yc0500.athinput').read_text()
    text = section(text, 'time', dict(tlim=20 if control else c['target'], stop_check_every=1))
    text = section(text, 'mesh_refinement', dict(num_levels=8, max_nmb_per_rank=rank_capacity()))
    text = section(text, 'particles', dict(excise_lapse=.1, excise_ah='false'))
    text = section(text, 'problem', dict(cluster_sample_file=ROOT/'samples/yc0500.bin',
        cluster_t0_dump=ROOT/'runs/yc0500/initial_particles.bin', cluster_radial_mode_u=U,
        cluster_solve_momentum_constraint='true' if U != 0 or control else 'false',
        cluster_solve_hamiltonian_constraint='true' if U != 0 or control else 'false',
        production_history_dt=.25, production_cohort_dt=.5))
    text = section(text, 'output1', dict(dt=.25))
    for i in [2, 3, 4, 5]:
        text = section(text, 'output'+str(i), dict(dt=2))
    for i in [6, 7]:
        text = section(text, 'output'+str(i), dict(dt=0 if control else '1e30'))
    text = section(text, 'output8', dict(dt=100))
    path = ROOT/'inputs/yc0500.athinput'
    if path.exists():
        assert path.read_text() == text, 'Existing exact input differs'
    else:
        path.write_text(text)
    metadata = ROOT/'evidence/samples/yc0500.json'
    if not metadata.exists():
        metadata.parent.mkdir(exist_ok=True)
        shutil.copy2(BASE/'evidence/samples/yc0500.json', metadata)
    record = dict(utc=now(), job=job(), case=c, U=U, control=control,
                  input=receipt(path), sample=receipt(sample), executable=receipt(exe))
    import time
    atomic(ROOT/'evidence/setup'/f'{job()}_{time.time_ns()}.json', record)
    if not (ROOT/'evidence/EXACT_SETUP.json').exists():
        atomic(ROOT/'evidence/EXACT_SETUP.json', record)
    return ROOT/'runs/yc0500'
