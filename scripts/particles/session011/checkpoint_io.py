"""Pinned GI double/MPI collective restart format; verification never deletes files."""
import hashlib,json,math,os,struct
from pathlib import Path
def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()

def checkpoint(path, hash_file=True, expected_root=3):
    """Exact native double/MPI single-file layout of the frozen GI source."""
    path = Path(path)
    st = path.stat()
    with path.open('rb') as f:
        header = f.read(1024**2)
        end = header.find(b'<par_end>')
        if end < 0: raise ValueError('missing bounded parameter header')
        offset = end + 10
        f.seek(offset)
        nmb, rootlevel = struct.unpack('<ii', f.read(8))
        if not 0 < nmb <= st.st_size//(25*40**3*8) or rootlevel != expected_root:
            raise ValueError('unexpected mesh size/root level')
        f.seek(offset + 232)
        simtime, dt, cycle = struct.unpack('<ddi', f.read(20))
        if not all(math.isfinite(v) for v in (simtime,dt)) or simtime < 0 or dt <= 0 or cycle < 0:
            raise ValueError('invalid checkpoint time/cycle')
        f.seek(offset + 252 + nmb*20 + 56)
        block_bytes = struct.unpack('<Q', f.read(8))[0]
        if block_bytes != 25*40**3*8:
            raise ValueError('unexpected Z4c block layout')
        particle_offset = f.tell() + block_bytes*nmb
        f.seek(particle_offset)
        counts = struct.unpack('<4d', f.read(32))
        if any(not math.isfinite(x) or x < 0 or x != int(x) for x in counts):
            raise ValueError('invalid particle census')
        if sum(counts) != 5_000_000: raise ValueError('particle conservation failed')
        expected = particle_offset + 32 + int(counts[0])*10*8
        if st.st_size != expected: raise ValueError('truncated or incompatible checkpoint')
    row = dict(path=str(path), bytes=st.st_size, header_bytes=offset,
               time=simtime, cycle=cycle, dt=dt, blocks=nmb,
               particles=int(counts[0]), removed=list(map(int,counts[1:])))
    if hash_file: row['sha256'] = digest(path)
    if path.stat().st_size != st.st_size or path.stat().st_mtime_ns != st.st_mtime_ns:
        raise ValueError('checkpoint changed during verification')
    return row
