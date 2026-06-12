"""Shared HDF5 writer for NRPIC particle initial-condition files.

This encodes the on-disk contract expected by the C++ reader
``src/particles/read_particle.cpp`` in ONE place, so the generators below and the reader
cannot drift apart.

Required 1-D datasets (all the same length N):
  x, y, z      particle positions
  ux, uy, uz   COVARIANT spatial 4-velocity components u_i  (NOT coordinate velocity\
 dx/dt)
Optional 1-D dataset:
  mass         per-particle rest mass; if absent, the scalar ``<particles> mass`` is used.

The reader reads with ``H5T_NATIVE_DOUBLE`` (or NATIVE_FLOAT for a single-precision\
 build),
so datasets are written as NATIVE-endian float64 -- do NOT byte-swap to big-endian the way
AthenaK's ``.athdf`` mesh output does.
"""
import numpy as np
import h5py


def write_particle_table(fname, x, y, z, ux, uy, uz, mass=None):
    """Write the six required arrays (+ optional mass) to an HDF5 file `fname`."""
    cols = {"x": x, "y": y, "z": z, "ux": ux, "uy": uy, "uz": uz}
    arrs = {}
    n = None
    for name, col in cols.items():
        a = np.ascontiguousarray(col, dtype=np.float64).ravel()
        if n is None:
            n = a.size
        elif a.size != n:
            raise ValueError(f"dataset '{name}' has length {a.size}, expected {n}")
        arrs[name] = a
    if mass is not None:
        m = np.ascontiguousarray(mass, dtype=np.float64).ravel()
        if m.size != n:
            raise ValueError(f"dataset 'mass' has length {m.size}, expected {n}")
        arrs["mass"] = m

    with h5py.File(fname, "w") as f:
        for name, a in arrs.items():
            f.create_dataset(name, data=a)        # native-endian float64
        f.attrs["n_particles"] = int(n)
        f.attrs["note"] = "NRPIC particle IC: ux,uy,uz are the covariant 4-velocity u_i"
    print(f"wrote {n} particles to '{fname}'"
          + (" (with per-particle mass)" if mass is not None else ""))
    return n
