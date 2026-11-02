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

# Compare new legacy mode against the immutable original HIP executable.
old_exe=Path('/work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102/reference/session7/protected/athena')
legacy_text=deck(coupling='legacy',integrator='rk4').replace('synchronized_constraints=true','synchronized_constraints=false')
old=run(old_exe,root,'archived_legacy',legacy_text,ranks=1)
new=run(Path(a.exe),root,'new_legacy',legacy_text,ranks=1)
control=compare_states(read_rst(final_rst(old)),read_rst(final_rst(new)))
assert control['grid_max_abs']<1.e-11 and control['particle_max_abs']<1.e-11,control
write_json(root/'GATES_OK.json',dict(cpu_gpu=comparison,legacy_vs_original=control,exe_sha256=sha(a.exe)))
