import argparse
from pathlib import Path
from native_gates import deck, run
from state_io import require_job, read_rst, final_rst, compare_states, write_json, sha

require_job()
ap=argparse.ArgumentParser();ap.add_argument('--exe',required=True);ap.add_argument('--root',required=True)
a=ap.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=False)
work=run(Path(a.exe),root,'np1',deck(),ranks=1)
run(Path(a.exe),root,'bad_solver',deck(limit=1),ranks=1,fail='particle solve failed')
cpu=root.parent/root.name.replace('hip_', 'cpu_')/'np1'
comparison=compare_states(read_rst(final_rst(cpu)),read_rst(final_rst(work)))
assert comparison['grid_max_abs']<1.e-9 and comparison['particle_max_abs']<1.e-9,comparison
write_json(root/'GATES_OK.json',dict(cpu_gpu=comparison,exe_sha256=sha(a.exe)))
