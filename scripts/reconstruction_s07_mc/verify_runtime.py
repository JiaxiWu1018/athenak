"""Refuse detached/changed runtime scripts on allocated AMD nodes."""
import hashlib
import json
import os
from pathlib import Path
root=Path(os.environ['ROOT']);pin=json.loads((root/'evidence/SOURCE_PIN.json').read_text())
for row in pin['runtime_files']:
    path=root/'src/athenak'/row['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],str(path)
print('RUNTIME_SOURCE_OK',pin['head'],flush=True)
