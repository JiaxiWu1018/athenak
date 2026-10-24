"""Configuration regression for the previously hardcoded case/ceiling assumptions."""
import ast,json
from pathlib import Path
p=Path(__file__).parent
for f in p.glob('*.py'):ast.parse(f.read_text(),filename=str(f))
cases=json.loads((p/'cases.json').read_text());assert len(cases)==4
assert {c['yc']:c['finest_dx'] for c in cases}=={.283:.0625,.605:.0625,.819:.125,.9:.125}
for c in cases:
    assert c['U']==-.005 and c['target']==300 and c['num_levels']==8
    assert (2*c['L']/128)/(2**(c['num_levels']-1))==c['finest_dx']
    assert c['initial_dx']==(c['L']/2048)
print('PASS: four explicit cases, exact AMR ceilings and native base; Python syntax valid')
