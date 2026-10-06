#!/usr/bin/env python3
"""Verify every startup/continuation override exists before allocating full physics."""
import argparse,re
from pathlib import Path
GATE_COMMON=('output7/last_time=0','time/ndiag=1','z4c/co_0_out_every=1','z4c/co_1_out_every=1','fastflow/start_time_0=0','fastflow/start_time_1=0','fastflow/start_time_2=0')
SLIM=tuple(f'output{i}/dt=0' for i in (6,18,19,20,21))
OTHER=('time/nlim=2','time/stop_file=REQUEST_STOP','time/stop_check_every=1','time/tlim=12')
def keys(path):
 block=None;result=set()
 for line in Path(path).read_text().splitlines():
  line=line.split('#',1)[0].strip()
  match=re.fullmatch(r'<([^>]+)>',line)
  if match:block=match[1]
  elif '=' in line and block:result.add(block+'/'+line.split('=',1)[0].strip())
 return result
def validate(path,overrides=GATE_COMMON+SLIM+OTHER):
 present=keys(path);missing=[x.split('=',1)[0] for x in overrides if x.split('=',1)[0] not in present]
 if missing:raise ValueError('startup override absent from input: '+','.join(missing))
 return len(overrides)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('input',type=Path);a=p.parse_args();print('All',validate(a.input),'startup/continuation overrides have explicit input keys.')
