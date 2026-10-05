import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from health import require_window
run=Path(sys.argv[1]);health=require_window(run,allow_stopped=False)
state=json.loads((run/'segment_state.json').read_text())
if state['completed']:raise RuntimeError('endpoint already completed; restart forbidden')
if state['returncode'] and not state.get('resource_failure',False):
    raise RuntimeError('non-resource failure requires diagnosis before retry')
checkpoint=Path(state['checkpoint'])
if not checkpoint.is_file():raise RuntimeError('verified checkpoint missing')
if state['checkpoint_time']>health['last_healthy_time']+1.e-10:
    raise RuntimeError('checkpoint is newer than the healthy window')
digest=hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else None
if digest is None:
    h=hashlib.sha256()
    with checkpoint.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    digest=h.hexdigest()
if digest!=state['checkpoint_sha256']:raise RuntimeError('checkpoint checksum mismatch')
print(json.dumps(dict(checkpoint=str(checkpoint),health=health)))
