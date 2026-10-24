"""Preflight contract checks, including the positive mode and unchanged old mesh."""
import ast
from common import *
from checkpoint import parameters
from setup_case import section

def check():
    for p in Path(__file__).parent.glob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
    assert len(ALL_CASES) == 2
    assert [(c['U'], c['start'], c['target']) for c in ALL_CASES] == [(.005, 0., 1500.), (0., 700., 1500.)]
    original = (EXT/'templates/yc0500.athinput').read_text()
    before = parameters(original)
    changed = section(section(original, 'time', {'tlim': 1500}), 'output8', {'dt': 100})
    after = parameters(changed)
    for name in ['mesh', 'meshblock', 'z4c', 'z4c_amr']+[n for n in before if n.startswith('refined_region')]:
        assert before[name] == after[name], name
    assert before['mesh']['x1min'] == '-512' and before['mesh']['x1max'] == '512'
    for c in ALL_CASES:
        assert (2*c['L']/128)/2**(c['num_levels']-1) == .0625
        assert c['initial_dx'] == .25
    record = dict(utc=now(), job=job(), contract_checked=True, positive_mode=.005,
        targets=[1500, 1500], original_physical_mesh_preserved=True)
    atomic(EXT/'evidence/CONFIGURATION_OK.json', record)
    return record

if __name__ == '__main__':
    print(check())
