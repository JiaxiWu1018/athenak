"""Double-precision reader for this campaign's Z4c+dust shared restart format."""
from pathlib import Path
import hashlib
import json
import os
import re
import struct
import numpy as np


def require_job():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Scientific I/O and analysis require a Slurm allocation')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.writing')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def parse_input(text):
    out, block = {}, None
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip()
        if line.startswith('<'):
            block = line.strip('<>')
            out.setdefault(block, {})
        elif '=' in line and block:
            key, value = line.split('=', 1)
            out[block][key.strip()] = value.strip()
    return out


def read_rst(path):
    require_job()
    path = Path(path)
    with path.open('rb') as f:
        header = bytearray()
        while True:
            line = f.readline()
            if not line or len(header) > 1 << 20:
                raise ValueError('Invalid restart parameter header')
            header.extend(line)
            if line.strip() == b'<par_end>':
                break
        params = parse_input(header.decode())
        assert 'z4c' in params and 'particles' in params
        assert not set(params).intersection(('hydro', 'mhd', 'radiation', 'turb_driving'))
        assert not any(k.startswith('z4c_tracker') for k in params)
        nmb, level = struct.unpack('<2i', f.read(8))
        box = struct.unpack('<9d', f.read(72))
        root = struct.unpack('<19i', f.read(76))
        block = struct.unpack('<19i', f.read(76))
        t, dt, cycle = struct.unpack('<2di', f.read(20))
        locations = np.frombuffer(f.read(16*nmb), dtype='<i4').reshape(nmb, 4).copy()
        costs = f.read(4*nmb)
        last_output, = struct.unpack('<d', f.read(8))
        # ST campaign has no puncture trackers or turbulence state.
        size, = struct.unpack('<Q', f.read(8))
        grid_offset = f.tell()
        ng, nx, ny, nz = block[:4]
        shape = (nmb, 25, nz+2*ng, ny+2*ng, nx+2*ng)
        assert size == np.prod(shape[1:])*8, (size, shape)
        f.seek(grid_offset+nmb*size)
        census = np.frombuffer(f.read(32), dtype='<f8').copy()
        count = int(census[0])
        assert count == census[0] and np.all(census[1:] == 0), census
        particle_offset = f.tell()
        assert path.stat().st_size == particle_offset+10*count*8
    grid = np.memmap(path, mode='r', dtype='<f8', offset=grid_offset, shape=shape)
    particles = np.memmap(path, mode='r', dtype='<f8', offset=particle_offset,
                          shape=(10, count))
    assert np.all(np.isfinite(particles))
    tags = particles[9].astype(np.int64)
    assert np.array_equal(tags, particles[9]) and len(np.unique(tags)) == count
    order = np.argsort(tags)
    return dict(path=str(path), time=t, dt=dt, cycle=cycle, params=params,
                count=count, census=census, grid=grid, particles=particles, order=order,
                ng=ng, box=box, locations=locations, root_level=level, root=root,
                block=block, grid_offset=grid_offset, particle_offset=particle_offset)


def interior(state):
    ng = state['ng']
    return state['grid'][:, :, ng:-ng, ng:-ng, ng:-ng]


def compare_states(a, b):
    assert a['count'] == b['count'] and abs(a['time']-b['time']) < 1.e-10
    assert np.array_equal(a['locations'], b['locations'])
    pa, pb = a['particles'][:, a['order']], b['particles'][:, b['order']]
    assert np.array_equal(pa[9], pb[9]) and np.array_equal(pa[0], pb[0])
    gmax = max(float(np.max(np.abs(ga-gb))) for ga, gb in zip(interior(a), interior(b)))
    return dict(grid_max_abs=gmax, particle_max_abs=float(np.max(np.abs(pa[2:8]-pb[2:8]))),
                energy_max_abs=float(np.max(np.abs(pa[1]-pb[1]))))


def final_rst(work):
    files = list(Path(work).glob('rst/*.rst'))
    assert files, str(work)
    return max(files, key=lambda p: read_rst(p)['time'])
