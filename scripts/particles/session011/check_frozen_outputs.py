#!/usr/bin/env python3
"""Readback assertions for the zero-advance analytic fixture. Run under Slurm."""
import csv
import json
import math
import os
import re
import struct
import sys
from pathlib import Path
from runtime import checkpoint

def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def main():
    assert os.environ.get('SLURM_JOB_ID')
    root=Path(sys.argv[1]);fresh=root/'fresh'
    text=(root/'fresh.log').read_text()
    assert 'FROZEN_AH_CHECKS_PASS' in text and 'evolution_advances=0' in text
    count=int(re.search(r'z4c_one_puncture: placed (\d+)',text)[1])
    frames=list((fresh/'trk').glob('*.trk2'));assert frames
    for frame in frames:
        data=frame.read_bytes();header=data[:512].rstrip(b'\x00').decode()
        n=int(re.search(r'count=(\d+)',header)[1]);assert len(data)==512+72*n
        assert n==8 and 'cycle=0 ' in header and 'time=0' in header
        records=list(struct.iter_unpack('=q8d',data[512:]));tags=[r[0] for r in records]
        assert sorted(tags)==list(range(8))
        assert all(math.isfinite(x) for r in records for x in r[1:])
        assert all(abs(r[1]-.001)<1e-15 for r in records)
    ledger=rows(fresh/'frozen_ah.gi_shell_transfer.csv')
    # No particle displacement or momentum change in a frozen-state check.
    for r in ledger:
        for k in ('mass_in','mass_out','Jz_adv_in','Jz_adv_out','Jz_force_delta'):
            assert float(r[k])==0,(k,r)
    profile=rows(fresh/'frozen_ah.gi_particles.csv')
    # Repeated time=0 samples are explicit; use the first sample's rows up to
    # the repeated first shell key, rather than silently adding duplicate states.
    first=[];seen=set()
    for r in profile:
        key=(r['component'],r['shell'],r['mask'])
        if key in seen:break
        first.append(r);seen.add(key)
    all_rows=[r for r in first if r['mask']=='0']
    assert sum(float(r['count']) for r in all_rows)==count
    assert abs(sum(float(r['rest_mass']) for r in all_rows)-.001*count)<1e-12
    for h in (0,1):
        events=rows(fresh/f'frozen_ah.ah_events_{h}.csv')
        assert any(r['reason']=='pending_persistence' and r['seed_valid']=='1' and r['published']=='0' for r in events)
        assert any(r['published']=='1' for r in events)
        assert all(float(r['state_time'])==0 for r in events)
    checkpoints=[checkpoint(p,expected_particles=count,ntrackers=1,block_n=24) for p in sorted((fresh/'rst').glob('*.rst'))]
    assert checkpoints and all(r['cycle']==0 and r['time']==0 for r in checkpoints)
    (root/'readback.json').write_text(json.dumps(dict(particles=count,frames=len(frames),checkpoints=checkpoints),indent=2)+'\n')
    print('FROZEN_OUTPUT_READBACK_PASS')

if __name__=='__main__':main()
