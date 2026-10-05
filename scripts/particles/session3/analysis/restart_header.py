"""Read the pinned AthenaK little-endian, float64 restart header without its payload.

Layout from outputs/restart.cpp, RegionSize=9 doubles, RegionIndcs=19 int32.
Reject unsupported precision/endianness rather than guessing. Loaded continuity
is checked separately by the pilot.
"""
import math,struct,sys,json
from pathlib import Path
def read_header(path):
    with Path(path).open('rb') as f:prefix=f.read(65536)
    end=prefix.find(b'<par_end>\n')
    if end<0:raise ValueError('restart parameter terminator missing')
    offset=end+len(b'<par_end>\n')
    nmb,rootlevel=struct.unpack_from('<ii',prefix,offset)
    offset+=8+9*8+2*19*4
    time,dt,cycle=struct.unpack_from('<ddi',prefix,offset)
    if not (0<nmb<10000000 and 0<=rootlevel<32 and math.isfinite(time) and time>=0 and
            math.isfinite(dt) and dt>=0 and cycle>=0):raise ValueError('invalid/unsupported restart layout')
    return dict(time=time,dt=dt,cycle=cycle,nmb=nmb,root_level=rootlevel)
if __name__=='__main__':print(json.dumps(read_header(sys.argv[1]),indent=2))
