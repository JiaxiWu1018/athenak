#!/usr/bin/env python3
"""Reader for AthenaK `file_type = cart` (CartesianGridOutput) binary dumps.

Session 2 uses this output so the movie density/constraint panels live on a FIXED
UNIFORM Cartesian visualisation grid whose physical extent and pixel size are
identical in every frame, independent of the AMR layout.  Session 1's movies used
`bin` AMR-patch slices instead, which the Session-2 specification forbids for the
density panel.

On-disk layout, from src/outputs/cartgrid.cpp:90-138 and the MetaData struct in
src/outputs/outputs.hpp:399-407:

    struct MetaData {          offset  bytes   note
      int   cycle;                  0      4
      float time;                   4      4   SINGLE precision
      float center[3];              8     12
      float extent[3];             20     12   HALF-widths
      int   numpoints[3];          32     12
      bool  is_cheb;               44      1
      <3 padding bytes>            45      3   C struct alignment for the next int
      int   noutvars;              48      4
    };                                   52   = sizeof(MetaData)

    int   label_len                52      4
    char  labels[label_len]        56          space-separated variable names
    float data[noutvars][nz][ny][nx]           written k, j, i (x is the fast axis)

Grid geometry (src/utils/cart_grid.cpp:45-62):
    min_x = center_x - extent_x,  max_x = center_x + extent_x,
    d_x   = (max_x - min_x)/(nx - 1)            -> nx must be >= 2
so the node coordinates are  x_i = min_x + i*d_x,  i = 0 .. nx-1 (endpoints included).
"""
import glob
import os
import re
import struct

import numpy as np

HEADER_FMT = '<i f 3f 3f 3i'      # cycle, time, center[3], extent[3], numpoints[3]
HEADER_SIZE = 52                  # includes the bool + 3 pad bytes + noutvars


def read_cart(path):
    """Read one cart dump.  Returns a dict with the metadata, the axes, and the data."""
    with open(path, 'rb') as fh:
        raw = fh.read(HEADER_SIZE)
        if len(raw) < HEADER_SIZE:
            raise IOError('%s: truncated header (%d bytes)' % (path, len(raw)))
        cycle, time = struct.unpack_from('<i f', raw, 0)
        center = np.array(struct.unpack_from('<3f', raw, 8), dtype=float)
        extent = np.array(struct.unpack_from('<3f', raw, 20), dtype=float)
        numpoints = np.array(struct.unpack_from('<3i', raw, 32), dtype=int)
        is_cheb = bool(struct.unpack_from('<?', raw, 44)[0])
        noutvars = struct.unpack_from('<i', raw, 48)[0]

        (label_len,) = struct.unpack('<i', fh.read(4))
        labels = fh.read(label_len).decode('ascii').split()

        nx, ny, nz = (int(v) for v in numpoints)
        count = noutvars * nx * ny * nz
        data = np.fromfile(fh, dtype='<f4', count=count)
        if data.size != count:
            raise IOError('%s: expected %d floats, got %d' % (path, count, data.size))

    if len(labels) != noutvars:
        raise IOError('%s: %d labels for %d vars: %r' % (path, len(labels), noutvars, labels))

    # written k, j, i with x fastest -> (nvar, nz, ny, nx)
    data = data.reshape(noutvars, nz, ny, nx)

    lo = center - extent
    hi = center + extent
    # d = (hi-lo)/(n-1); n == 1 would divide by zero in the code, and extent == 0
    # with n == 2 gives d == 0 and both nodes on the plane (the exact-equatorial trick).
    with np.errstate(divide='ignore', invalid='ignore'):
        d = np.where(numpoints > 1, (hi - lo) / np.maximum(numpoints - 1, 1), 0.0)
    axes = [lo[a] + d[a] * np.arange(numpoints[a]) for a in range(3)]

    return dict(path=path, cycle=cycle, time=float(time), center=center, extent=extent,
                numpoints=numpoints, is_cheb=is_cheb, noutvars=noutvars, labels=labels,
                dx=d, lo=lo, hi=hi, x=axes[0], y=axes[1], z=axes[2], data=data)


def equatorial(rec, var, atol=None):
    """The equatorial plane of one variable as a (ny, nx) array, with (x, y) axes.

    Session-2 decks sample the equator as TWO planes at z = +/- pixel/2 rather than one
    plane at z = 0, because z = 0 is a MeshBlock face at every level of this
    origin-centred mesh: cart_grid.cpp owns a node with a bounds test inclusive at both
    ends and cartgrid.cpp MPI_SUMs, so a node on a shared face is DOUBLED.  The two
    planes straddle the equator symmetrically, so their mean is the z = 0 field to
    O(pixel^2) and neither plane sits on the face.

    Accepted geometries, in order:
      * z nodes symmetric about 0  -> return their mean (the Session-2 case, and also
        the degenerate all-at-zero case);
      * a node lying on z = 0      -> return that plane;
      * anything else              -> refuse rather than silently mis-slice.
    """
    iv = rec['labels'].index(var)
    z = np.asarray(rec['z'], dtype=float)
    tol = atol if atol is not None else 1.0e-9 * max(1.0, float(np.abs(z).max()))
    if len(z) == 2 and abs(z[0] + z[1]) <= tol:
        plane = rec['data'][iv].mean(axis=0)
    elif np.all(np.abs(z) <= tol):
        plane = rec['data'][iv].mean(axis=0)
    elif np.abs(z).min() <= tol:
        plane = rec['data'][iv][int(np.argmin(np.abs(z)))]
    else:
        raise ValueError('%s: z nodes %s are neither symmetric about 0 nor on it'
                         % (rec['path'], z))
    return plane, rec['x'], rec['y']


def series(rundir, file_id, basename=None):
    """All cart frames for one output id, sorted by frame index."""
    pat = os.path.join(rundir, 'cart', '%s.%s.*.bin'
                       % (basename or '*', file_id))
    files = sorted(glob.glob(pat),
                   key=lambda p: int(re.search(r'\.(\d+)\.bin$', p).group(1)))
    return files


def check_fixed_grid(files):
    """Assert every frame shares the same center, extent and numpoints.

    This is the guarantee the Session-2 specification demands of the movie grid: the
    same physical extent and the same uniform pixel spacing in every frame, so visual
    change cannot be an artefact of the mesh.  Returns the common geometry.
    """
    ref = None
    for p in files:
        r = read_cart(p)
        key = (tuple(r['center']), tuple(r['extent']), tuple(r['numpoints']),
               tuple(r['labels']))
        if ref is None:
            ref = key
        elif key != ref:
            raise ValueError('%s: grid changed between frames\n  first %r\n  this  %r'
                             % (p, ref, key))
    return ref


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+')
    ap.add_argument('--check-fixed', action='store_true',
                    help='verify all frames share one grid geometry')
    a = ap.parse_args()
    if a.check_fixed:
        ref = check_fixed_grid(a.files)
        print('grid is FIXED across %d frames: center=%s extent=%s numpoints=%s vars=%s'
              % (len(a.files), ref[0], ref[1], ref[2], ref[3]))
    for p in a.files[:5]:
        r = read_cart(p)
        print('%s  cycle %d  t = %.6f  n = %s  extent = %s  dx = %s  vars = %s'
              % (os.path.basename(p), r['cycle'], r['time'], tuple(r['numpoints']),
                 tuple(r['extent']), tuple(np.round(r['dx'], 8)), r['labels']))
        for v in r['labels']:
            pl, x, y = equatorial(r, v)
            print('    %-10s z=0 plane %s  min %.6e  max %.6e  finite %d/%d'
                  % (v, pl.shape, np.nanmin(pl), np.nanmax(pl),
                     int(np.isfinite(pl).sum()), pl.size))
