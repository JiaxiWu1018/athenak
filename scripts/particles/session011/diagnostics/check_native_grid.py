"""Hash a saved native state while checking every stored grid Real, including ghosts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import checkpoint,require_compute


def scan(path,grid_start,grid_bytes,chunk_bytes=8*1024**2):
    if grid_start<0 or grid_bytes<0 or grid_bytes%8 or chunk_bytes<=0 or chunk_bytes%8:
        raise ValueError('invalid Real64 scan bounds')
    h=hashlib.sha256();invalid=0;seen=0;samples=[]
    with Path(path).open('rb') as f:
        remaining=grid_start
        while remaining:
            data=f.read(min(chunk_bytes,remaining))
            if not data:raise ValueError('truncated header')
            h.update(data);remaining-=len(data)
        remaining=grid_bytes
        while remaining:
            length=min(chunk_bytes,remaining);data=f.read(length)
            if len(data)!=length:raise ValueError('truncated grid')
            h.update(data);values=np.frombuffer(data,dtype='<f8');bad=np.flatnonzero(~np.isfinite(values))
            invalid+=len(bad)
            for i in bad[:max(0,10-len(samples))]:
                samples.append(dict(flat_index=seen+int(i),byte_offset=grid_start+(seen+int(i))*8,
                                    value=repr(float(values[i]))))
            seen+=len(values);remaining-=len(data)
        for data in iter(lambda:f.read(chunk_bytes),b''):h.update(data)
    return dict(sha256=h.hexdigest(),grid_values=seen,nonfinite_grid_values=invalid,first_nonfinite=samples)


def main():
    require_compute()
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--expected-sha256',required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir()
    started=time.time();before=args.checkpoint.stat();state=checkpoint(args.checkpoint,hashing=False)
    grid_bytes=state['blocks']*25*40**3*8;grid_start=state['particle_offset']-32-grid_bytes
    result=scan(args.checkpoint,grid_start,grid_bytes)
    after=args.checkpoint.stat()
    unchanged=(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    passed=result['sha256']==args.expected_sha256 and result['nonfinite_grid_values']==0 and unchanged
    result.update(status='pass' if passed else 'fail',job_id=os.environ['SLURM_JOB_ID'],
        checkpoint=state,expected_sha256=args.expected_sha256,source_file_unchanged=unchanged,
        grid_start=grid_start,grid_bytes=grid_bytes,grid_fields=25,ghosts_included=True,
        evolution_advances=0,production_modified=False,elapsed_seconds=time.time()-started,
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits='Finiteness check only; not positivity, positive-definiteness, constraint convergence or AH angular validation')
    (args.out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.out/'REPORT.md').write_text(f'''# Native saved-grid finiteness check

Status: {result['status']}. Checkpoint time {state['time']:.12g}, cycle {state['cycle']}.
All {result['grid_values']:,} stored grid Real values in 25 fields, including ghost
cells, were checked. Nonfinite values: {result['nonfinite_grid_values']}.
The full checkpoint SHA256 {'matches' if result['sha256']==args.expected_sha256 else 'DOES NOT MATCH'}
the audited manifest; source size/mtime unchanged: {unchanged}.

No simulation was evolved, stopped or modified. This checks finiteness only;
it does not establish metric positivity, constraint convergence, or saved-state
AH angular validation. Exact offsets, hashes, job and method are in RESULT.json.
''')
    print(json.dumps(result,indent=2))
    if not passed:sys.exit(1)


if __name__=='__main__':main()
