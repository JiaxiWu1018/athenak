from pathlib import Path
import json
import shutil
from state_io import require_job, sha, write_json
require_job()
root=Path('/work1/eliasmost/jiaxiwu/st_reconstruction_s10_rk3_20261102')
manifest=json.loads((root/'evidence/TRANSFER_MANIFEST.json').read_text())
for row in manifest['files']:
    p=root/'reference/session7'/row['path']
    assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
source=json.loads((root/'evidence/WORKING_SOURCE.json').read_text())
for path,value in source['files'].items():
    if path.startswith('src/'):
        assert sha(root/'src/athenak'/path)==value,path
(root/'protected').mkdir(exist_ok=True)
exe=root/'build_hip/src/athena';out=root/'protected/athena.rk3'
if out.exists():assert sha(out)==sha(exe)
else:shutil.copy2(exe,out)
write_json(root/'evidence/INPUTS_OK.json',dict(files=len(manifest['files']),
    samples='Exact archived Session7 samples; per-file SHA256 verified',
    native_source_matches_compiled_manifest=True,hip_sha256=sha(exe),
    source_commit=(root/'evidence/SOURCE_COMMIT.txt').read_text().strip()))
