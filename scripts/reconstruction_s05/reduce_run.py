#!/usr/bin/env python3
"""Reduce one Session-04 coherent-mode run to a compact, durable CSV/NPZ set.

Designed to run ON THE AMD SIDE inside a Slurm job (numpy only, no scipy), so that only
the reduced products ever cross to Perseus.

Products, all written into <rundir>/analysis/:
  cycles.csv        per-cycle wall clock / dt from the log (cost accounting)
  hstnorms.csv      time, dt, and the Hamiltonian/momentum constraint norms
  shells.csv        Lagrangian shell radii (enclosed-rest-mass quantiles) vs time,
                    plus bound/unbound mass fractions and the radial-flow diagnostics
  particles.npz     per-dump radial profiles of density, radial current and <v_r>,
                    plus histograms of -u_t and |v|
  fields.csv        alpha_c(t), central deposited energy density, from the z=0 slices
  profiles.npz      radial profiles of alpha and the deposited energy at selected times
  census.csv        the particle destruction ledger per cycle
  summary.json      the scalars a report table needs
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import sys

import numpy as np

# number of physical-level columns written to fields_*.csv (Session 5 hardcoded 8)
N_LEVEL_COLS = 9

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import st_migration_lib as L    # noqa: E402


# ----------------------------------------------------------------------------------
def _open(path):
    return gzip.open(path, "rt", errors="replace") if path.endswith(".gz") \
        else open(path, "r", errors="replace")


def parse_log(path):
    """cycle / time / dt / elapsed from the ndiag lines, plus the ledger and warnings.

    The AthenaK per-ndiag line is
        elapsed=<float> cycle=<int> time=<float> dt=<float>
    (a restart segment restarts `elapsed` from ~0, so the wall clock is accumulated
    per segment and the segment boundary is detected by `elapsed` going backwards).
    """
    cyc, tim, dts, ela = [], [], [], []
    ledger, warn = {}, []
    boris_fallbacks, boris_cycles = 0, set()
    banner, bias = {}, []
    rx = re.compile(r"elapsed=([-\d.eE+]+)\s+cycle=(\d+)\s+time=([-\d.eE+]+)"
                    r"\s+dt=([-\d.eE+]+)")
    with _open(path) as fh:
        for line in fh:
            m = rx.search(line)
            if m:
                ela.append(float(m.group(1)))
                cyc.append(int(m.group(2)))
                tim.append(float(m.group(3)))
                dts.append(float(m.group(4)))
                continue
            if "Relativistic cluster initialized" in line:
                for tok in line.split(":", 1)[1].split(","):
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        try:
                            banner[k.strip()] = float(v)
                        except ValueError:
                            banner[k.strip()] = v.strip()
            elif "### gr_boris non-convergence:" in line:
                mfb = re.search(r"cycle\s+(\d+):\s+(\d+)\s+of", line)
                if mfb:
                    boris_cycles.add(int(mfb.group(1)))
                    boris_fallbacks += int(mfb.group(2))
            elif ("Cluster outward bias" in line or "Cluster tangential bias" in line
                  or "Cluster perturbation summary" in line
                  or "Cluster coherent radial mode" in line):
                bias.append(line.strip())
            elif line.startswith("particles: initial="):
                for tok in re.findall(r"(\w+)=(\d+)", line):
                    ledger[tok[0]] = int(tok[1])
                ledger["conservation_ok"] = "conservation OK" in line
            elif ("### FATAL" in line or "### WARNING" in line
                  or "rejected" in line.lower() or "search_fail=[1-9]" in line
                  or "nan" in line.lower() or "trilinear" in line.lower()
                  or "excision" in line.lower()):
                warn.append(line.rstrip()[:300])
            elif "zone-cycles/cpu_second" in line or "Terminating on" in line:
                banner.setdefault("tail", []).append(line.strip())
                if "Terminating on" in line:
                    banner["termination"] = line.strip()
    # non-zero search_fail is the one migration counter that must never fire
    with _open(path) as fh:
        for line in fh:
            m = re.search(r"search_fail=(\d+)", line)
            if m and int(m.group(1)) > 0:
                warn.append(line.rstrip()[:300])
    return dict(cycle=np.array(cyc), time=np.array(tim), dt=np.array(dts),
                elapsed=np.array(ela), banner=banner, ledger=ledger,
                warnings=warn[:200], bias=bias, boris_fallbacks=boris_fallbacks,
                boris_fallback_cycles=len(boris_cycles))


# ----------------------------------------------------------------------------------
def read_pvtk(path):
    """Read an AthenaK legacy-VTK particle dump.

    Layout written by src/outputs/vtk_prtcl.cpp:
        POINTS <n> float                 -> 3n big-endian float32 (x, y, z)
        POINT_DATA <n>
        SCALARS gid float / ptag float   -> n float32 each  (the integer fields)
        VECTORS prtcl_vel float          -> 3n float32      (covariant u_i)
        SCALARS prtcl_energy float       -> n float32       (-u_t)
        SCALARS prtcl_mass float         -> n float32
    Returns (time, cycle, npart, dict). Keys: x,y,z, gid, ptag, ux,uy,uz,
    prtcl_energy, prtcl_mass.
    """
    with open(path, "rb") as fh:
        raw = fh.read()

    def line_at(off):
        end = raw.index(b"\n", off)
        return raw[off:end].decode("ascii", "replace"), end + 1

    _, off = line_at(0)
    l1, off = line_at(off)
    time, cycle = float("nan"), -1
    mt = re.search(r"time=\s*([-\d.eE+]+)", l1)
    mc = re.search(r"cycle=\s*(\d+)", l1)
    if mt:
        time = float(mt.group(1))
    if mc:
        cycle = int(mc.group(1))
    while True:
        line, off = line_at(off)
        if line.startswith("POINTS"):
            npart = int(line.split()[1])
            break
    pts = np.frombuffer(raw, dtype=">f4", count=3*npart, offset=off).astype(np.float64)
    off += 3*npart*4
    out = {"x": pts[0::3], "y": pts[1::3], "z": pts[2::3]}
    while off < len(raw):
        nl = raw.find(b"\n", off)
        if nl < 0:
            break
        line = raw[off:nl].decode("ascii", "replace").strip()
        off = nl + 1
        if line.startswith("SCALARS"):
            name = line.split()[1]
            nl = raw.find(b"\n", off)          # LOOKUP_TABLE default
            off = nl + 1
            out[name] = np.frombuffer(raw, dtype=">f4", count=npart,
                                      offset=off).astype(np.float64)
            off += npart*4
        elif line.startswith("VECTORS"):
            name = line.split()[1]
            v = np.frombuffer(raw, dtype=">f4", count=3*npart,
                              offset=off).astype(np.float64)
            off += 3*npart*4
            if name == "prtcl_vel":
                out["ux"], out["uy"], out["uz"] = v[0::3], v[1::3], v[2::3]
            else:
                out[name + "_x"], out[name + "_y"], out[name + "_z"] = \
                    v[0::3], v[1::3], v[2::3]
    return time, cycle, npart, out


# ----------------------------------------------------------------------------------
def reduce_particles(pvtk_files, prof, mass, nbins, nsub, out):
    rows = []
    prof_r, prof_rho, prof_sr, prof_mu, prof_coord_vr, prof_L, prof_t = (
        [], [], [], [], [], [], [])
    hist_e, hist_v, hist_L, hist_t = [], [], [], []
    prev_r_by_tag, prev_time = None, None
    e_edges = np.linspace(0.5, 1.5, 201)
    v_edges = np.linspace(0.0, 1.0, 201)
    # |L| is an EXACT invariant in spherical symmetry, so its histogram is both a
    # physics diagnostic and a hard accuracy check
    L_edges = np.linspace(0.0, 8.0, 201)
    for k, f in enumerate(pvtk_files):
        try:
            t, cyc, npart, d = read_pvtk(f)
        except Exception as exc:              # noqa: BLE001
            print(f"  skip {os.path.basename(f)}: {exc}")
            continue
        if npart == 0:
            # a fully excised (collapsed) cluster: emit a COMPLETE row so the CSV keeps
            # a rectangular schema and downstream readers see 0, not a missing field
            rows.append(dict(time=t, cycle=cyc, npart=0, mass_total=0.0,
                             r10=np.nan, r25=np.nan, r50=np.nan, r75=np.nan,
                             r90=np.nan, r99=np.nan, r_mean=np.nan, r_max=np.nan,
                             frac_outward=np.nan, mean_mu=np.nan, mean_abs_mu=np.nan,
                             mean_mu2=np.nan, mean_mu4=np.nan, beta_aniso=np.nan,
                             mean_ur_cov=np.nan, sigma_ur2_cov=np.nan,
                             sigma_ut2_cov=np.nan, beta_u_cov_proxy=np.nan,
                             mean_coord_vr=np.nan, coord_vr_persistent_fraction=0.0,
                             mean_umag=np.nan,
                             mean_absL=np.nan, rms_L=np.nan, median_absL=np.nan,
                             Ltot_x=0.0, Ltot_y=0.0, Ltot_z=0.0,
                             energy_mean=np.nan, energy_std=np.nan,
                             energy_nonfinite=0, frac_energy_gt1=np.nan,
                             mass_outside_r24=0.0, mass_outside_r32=0.0,
                             mass_outside_r64=0.0, mass_outside_r128=0.0,
                             mass_outside_r256=0.0, mass_outside_r500=0.0))
            continue
        r = np.sqrt(d["x"]**2 + d["y"]**2 + d["z"]**2)
        m = d.get("prtcl_mass", np.full(npart, np.nan))
        m = np.where(np.isfinite(m), m, 0.0)
        mt = float(np.sum(m))
        # Coordinate radial velocity from matched particle radii in adjacent dumps. This
        # is dr_iso/dt, not an orthonormal velocity; unlike the momentum-direction proxy
        # below, it directly measures coherent shell motion without assuming a metric.
        tags_f = d.get("ptag", np.arange(npart, dtype=np.float64))
        tags = np.rint(tags_f).astype(np.int64)
        if (not np.all(np.isfinite(tags_f)) or np.any(tags < 0)
                or np.max(np.abs(tags_f-tags)) > 0.0 or np.unique(tags).size != tags.size):
            raise RuntimeError(f"invalid/non-unique particle tags in {f}")
        coord_vr = np.full(npart, np.nan)
        if prev_r_by_tag is not None and prev_time is not None and t > prev_time:
            within = tags < prev_r_by_tag.size
            persistent = within & np.isfinite(prev_r_by_tag[np.minimum(
                tags, prev_r_by_tag.size-1)])
            coord_vr[persistent] = (r[persistent]-prev_r_by_tag[tags[persistent]])/(t-prev_time)
        # radial velocity from the stored covariant u_i and the CURRENT positions
        ok = r > 0
        nx = np.where(ok, d["x"]/np.where(ok, r, 1.0), 0.0)
        ny = np.where(ok, d["y"]/np.where(ok, r, 1.0), 0.0)
        nz = np.where(ok, d["z"]/np.where(ok, r, 1.0), 0.0)
        ux, uy, uz = d["ux"], d["uy"], d["uz"]
        udotn = ux*nx + uy*ny + uz*nz
        umag = np.sqrt(ux*ux + uy*uy + uz*uz)
        mu = udotn/np.maximum(umag, 1e-300)
        # The legacy directional beta is retained for comparison. It is exact only for
        # monoenergetic samples. The coherent mode changes |u|, so also report a centered
        # radial/tangential second-moment beta made from covariant u_i. In conformally
        # flat regions the local conformal factor cancels within a shell; globally this is
        # explicitly a momentum-space proxy, not an exact Eulerian-frame dispersion.
        mu2 = float(np.average(mu*mu, weights=m)) if mt > 0 else np.nan
        beta_aniso = (1.0 - (1.0 - mu2)/(2.0*mu2)) if (mt > 0 and mu2 > 0) else np.nan
        mean_ur = float(np.average(udotn, weights=m)) if mt > 0 else np.nan
        sigma_ur2 = float(np.average((udotn-mean_ur)**2, weights=m))             if mt > 0 else np.nan
        ut2 = np.maximum(umag*umag-udotn*udotn, 0.0)
        sigma_ut2 = float(np.average(ut2, weights=m)) if mt > 0 else np.nan
        beta_u_cov = 1.0-sigma_ut2/(2.0*sigma_ur2)             if mt > 0 and sigma_ur2 > 0 else np.nan
        finite_coord = np.isfinite(coord_vr)
        mean_coord_vr = float(np.average(coord_vr[finite_coord], weights=m[finite_coord]))             if finite_coord.any() and np.sum(m[finite_coord]) > 0 else np.nan
        # |L| = |x cross u| per unit rest mass.  In exact spherical symmetry this is an
        # EXACT constant of the motion for every particle, so <|L|> is a hard numerical
        # invariant: any drift is discretisation/finite-N asphericity, not physics.
        Lx = d["y"]*uz - d["z"]*uy
        Ly = d["z"]*ux - d["x"]*uz
        Lz = d["x"]*uy - d["y"]*ux
        Lmag = np.sqrt(Lx*Lx + Ly*Ly + Lz*Lz)
        en = d.get("prtcl_energy", np.full(npart, np.nan))
        finite_e = np.isfinite(en)
        # One sort supplies all requested shells.  At 2e6 particles and ~301 dumps a
        # second argsort solely for r99 is a material waste in the scheduled reducer.
        q = (L.mass_quantile_radii(r, m, qs=(0.1, 0.25, 0.5, 0.75, 0.9, 0.99))
             if mt > 0 else np.full(6, np.nan))
        q99 = q[5]
        rows.append(dict(
            time=t, cycle=cyc, npart=npart, mass_total=mt,
            r10=q[0], r25=q[1], r50=q[2], r75=q[3], r90=q[4], r99=q99,
            r_mean=float(np.average(r, weights=m)) if mt > 0 else np.nan,
            r_max=float(np.max(r)),
            frac_outward=float(np.average(udotn > 0, weights=m)) if mt > 0 else np.nan,
            mean_vr_proxy=float(np.average(mu, weights=m)) if mt > 0 else np.nan,
            mean_mu=float(np.average(mu, weights=m)) if mt > 0 else np.nan,
            mean_abs_mu=float(np.average(np.abs(mu), weights=m)) if mt > 0 else np.nan,
            mean_mu2=mu2,
            mean_mu4=float(np.average(mu**4, weights=m)) if mt > 0 else np.nan,
            beta_aniso=beta_aniso,
            mean_ur_cov=mean_ur, sigma_ur2_cov=sigma_ur2, sigma_ut2_cov=sigma_ut2,
            beta_u_cov_proxy=beta_u_cov,
            mean_coord_vr=mean_coord_vr,
            coord_vr_persistent_fraction=float(np.mean(finite_coord)),
            mean_umag=float(np.average(umag, weights=m)) if mt > 0 else np.nan,
            mean_absL=float(np.average(Lmag, weights=m)) if mt > 0 else np.nan,
            rms_L=float(np.sqrt(np.average(Lmag*Lmag, weights=m)))
            if mt > 0 else np.nan,
            median_absL=float(np.median(Lmag)) if mt > 0 else np.nan,
            Ltot_x=float(np.sum(m*Lx)) if mt > 0 else np.nan,
            Ltot_y=float(np.sum(m*Ly)) if mt > 0 else np.nan,
            Ltot_z=float(np.sum(m*Lz)) if mt > 0 else np.nan,
            energy_mean=float(np.average(en[finite_e], weights=m[finite_e]))
            if finite_e.any() and np.sum(m[finite_e]) > 0 else np.nan,
            energy_std=float(np.sqrt(np.average(
                (en[finite_e] - np.average(en[finite_e], weights=m[finite_e]))**2,
                weights=m[finite_e]))) if finite_e.any() and np.sum(m[finite_e]) > 0
            else np.nan,
            energy_nonfinite=int(np.sum(~finite_e)),
            frac_energy_gt1=float(np.average(en[finite_e] > 1.0,
                                             weights=m[finite_e]))
            if finite_e.any() and np.sum(m[finite_e]) > 0 else np.nan,
            mass_outside_r24=float(np.sum(m[r > 24.0])),
            mass_outside_r32=float(np.sum(m[r > 32.0])),
            mass_outside_r64=float(np.sum(m[r > 64.0])),
            mass_outside_r128=float(np.sum(m[r > 128.0])),
            mass_outside_r256=float(np.sum(m[r > 256.0])),
            mass_outside_r500=float(np.sum(m[r > 500.0])),
        ))
        if k % max(nsub, 1) == 0:
            rc, sw, mmu = L.radial_bin(r, m, values=udotn/np.maximum(umag, 1e-300),
                                       nbins=nbins, rmax=24.0)
            edges = np.linspace(0.0, 24.0, nbins+1)
            vol = (4.0*np.pi/3.0)*(edges[1:]**3 - edges[:-1]**3)
            _, scur, _ = L.radial_bin(r, m*udotn, nbins=nbins, rmax=24.0)
            _, _, mL = L.radial_bin(r, m, values=Lmag, nbins=nbins, rmax=24.0)
            if finite_coord.any():
                _, _, mcoord = L.radial_bin(
                    r[finite_coord], m[finite_coord], values=coord_vr[finite_coord],
                    nbins=nbins, rmax=24.0)
            else:
                mcoord = np.full(nbins, np.nan)
            prof_t.append(t); prof_r.append(rc)
            prof_rho.append(sw/vol); prof_sr.append(scur/vol); prof_mu.append(mmu)
            prof_coord_vr.append(mcoord); prof_L.append(mL)
            he, _ = np.histogram(en[finite_e], bins=e_edges,
                                 weights=m[finite_e]) if finite_e.any() \
                else (np.zeros(len(e_edges)-1), None)
            hv, _ = np.histogram(umag, bins=v_edges*4.0, weights=m)
            hL, _ = np.histogram(Lmag, bins=L_edges, weights=m)
            hist_e.append(he); hist_v.append(hv); hist_L.append(hL)
            hist_t.append(t)
        new_size = max((int(np.max(tags))+1),
                       (prev_r_by_tag.size if prev_r_by_tag is not None else 0))
        current_r_by_tag = np.full(new_size, np.nan)
        current_r_by_tag[tags] = r
        prev_r_by_tag, prev_time = current_r_by_tag, t
    np.savez_compressed(out, times=np.array(prof_t),
                        r=np.array(prof_r[0]) if prof_r else np.zeros(0),
                        rho=np.array(prof_rho), Sr=np.array(prof_sr),
                        mean_mu=np.array(prof_mu),
                        mean_coord_vr=np.array(prof_coord_vr),
                        mean_absL=np.array(prof_L),
                        hist_times=np.array(hist_t),
                        energy_edges=e_edges, energy_hist=np.array(hist_e),
                        umag_edges=v_edges*4.0, umag_hist=np.array(hist_v),
                        L_edges=L_edges, L_hist=np.array(hist_L))
    return rows


# ----------------------------------------------------------------------------------
# variables reduced per output type.  A trailing "@r" means "project the vector
# (X, Xy, Xz) onto the radial direction" -- that is how S_r and M_r are obtained.
FIELD_VARS = {
    "z4c":   ["z4c_alpha", "z4c_chi"],
    "tmunu": ["tmunu_E", "tmunu_Sx@r"],
    "con2d": ["con_H", "con_M", "con_Mx@r"],
    "adm":   ["adm_psi4"],
}


def _radial_project(fd, base, slice_x3):
    """(r, S_i n^i, w) for a vector stored as <base>, <base[:-1]+y>, <base[:-1]+z>."""
    stem = base[:-1]
    r, vx, w = L.bin_flatten(fd, stem + "x", slice_x3=slice_x3)
    _, vy, _ = L.bin_flatten(fd, stem + "y", slice_x3=slice_x3)
    _, vz, _ = L.bin_flatten(fd, stem + "z", slice_x3=slice_x3)
    # rebuild the direction cosines from the same coordinate construction
    xs, ys, zs = [], [], []
    for mb in range(fd["n_mbs"]):
        x1, x2, x3, _, _, _ = L.bin_cell_coords(fd, mb)
        if slice_x3 is not None and x3.size == 1:
            x3 = np.array([slice_x3])
        X3, X2, X1 = np.meshgrid(x3, x2, x1, indexing="ij")
        xs.append(X1.ravel()); ys.append(X2.ravel()); zs.append(X3.ravel())
    X = np.concatenate(xs); Y = np.concatenate(ys); Z = np.concatenate(zs)
    rr = np.sqrt(X*X + Y*Y + Z*Z)
    ok = rr > 0
    inv = np.where(ok, 1.0/np.where(ok, rr, 1.0), 0.0)
    return r, (vx*X + vy*Y + vz*Z)*inv, w


def reduce_fields(bin_files, tag, nbins, nsub, out_npz, vis_python, slice_x3=0.0):
    """Central values + radial profiles for the variables of one output type."""
    want = FIELD_VARS.get(tag, [])
    rows = []
    prof = {}
    prof_t, prof_r = [], None
    for k, f in enumerate(sorted(bin_files)):
        try:
            fd = L.read_binary(f, vis_python=vis_python)
        except Exception as exc:                        # noqa: BLE001
            print(f"  skip {os.path.basename(f)}: {exc}")
            continue
        t = float(fd["time"])
        row = {"time": t, "cycle": int(fd.get("cycle", -1))}
        # Each slice retains per-block logical levels and geometry. This is a compact,
        # time-resolved audit of whether adaptive refinement follows the central region.
        logical = np.asarray(fd.get("mb_logical", []))
        if logical.ndim == 2 and logical.shape[1] >= 4:
            levels = logical[:, 3].astype(int)
            row["n_mbs_slice"] = int(levels.size)
            row["level_min"] = int(np.min(levels))
            row["level_max"] = int(np.max(levels))
            # Session 6 runs with <mesh_refinement> num_levels = 9, i.e. physical levels
            # 0..8, so the histogram must not be capped at 8 columns as it was through
            # Session 5.  N_LEVEL_COLS is set from the deck's num_levels.
            for lev in range(N_LEVEL_COLS):
                row[f"nmb_level{lev}"] = int(np.sum(levels == lev))
            geom = np.asarray(fd.get("mb_geometry", []))
            if geom.ndim == 2 and geom.shape[0] == levels.size:
                sample = next(iter(fd["mb_data"].values()))
                nx = np.asarray(sample).shape[-1]
                dx = (geom[:, 1] - geom[:, 0])/nx
                row["dx_min_slice"] = float(np.min(dx))
                row["dx_max_slice"] = float(np.max(dx))
        keep = (k % max(nsub, 1) == 0)
        for spec in want:
            radial = spec.endswith("@r")
            name = spec[:-2] if radial else spec
            if name not in fd["mb_data"]:
                continue
            if radial:
                r, v, w = _radial_project(fd, name, slice_x3)
                key = name[:-1] + "r"
            else:
                r, v, w = L.bin_flatten(fd, name, slice_x3=slice_x3)
                key = name
            central = float("nan")
            rcore = float("nan")
            for rc_try in (0.75, 1.5, 3.0, 6.0):
                core = r < rc_try
                if int(core.sum()) >= 4 and float(np.sum(w[core])) > 0.0:
                    central = float(np.average(v[core], weights=w[core]))
                    rcore = rc_try
                    break
            row[f"{key}_central"] = central
            row[f"{key}_rcore"] = rcore
            row[f"{key}_min"] = float(np.min(v))
            row[f"{key}_max"] = float(np.max(v))
            if tag == "con2d":
                squared_on_disk = name in ("con_C", "con_M", "con_Z")
                for rcut in (8.0, 16.0, 32.0):
                    inside = r < rcut
                    if inside.any() and float(np.sum(w[inside])) > 0.0:
                        v2 = np.abs(v[inside]) if squared_on_disk else v[inside]**2
                        row[f"{key}_rms_r{int(rcut)}"] = float(np.sqrt(
                            np.sum(v2*w[inside])/np.sum(w[inside])))
            if name == "adm_psi4":
                # far-field ADM-mass proxy: psi = 1 + M/(2 rbar) for a conformally flat
                # exterior, so M = 2 rbar (psi4^(1/4) - 1).  Read as a trend, not as a
                # precise mass: it assumes near-conformal flatness outside the matter and
                # it is contaminated by the outgoing gauge wave.
                for lo, hi_ in ((10.0, 16.0), (16.0, 22.0)):
                    row[f"M_adm_{int(lo)}_{int(hi_)}"] = L.adm_mass_from_psi4(
                        r, v, w, rmin=lo, rmax=hi_)
            if keep:
                rcs, sw, mv = L.radial_bin(r, w, values=v, nbins=nbins, rmax=24.0)
                prof.setdefault(key, []).append(mv)
                prof_r = rcs
        if keep:
            prof_t.append(t)
        rows.append(row)
    if out_npz and prof_r is not None and prof:
        np.savez_compressed(out_npz, times=np.array(prof_t), r=prof_r,
                            **{k: np.array(v) for k, v in prof.items()})
    return rows


def write_csv(path, rows, cols=None):
    if not rows:
        open(path, "w").write("")
        return
    if cols is None:
        cols = []
        for r in rows:                      # union, preserving first-seen order
            for k in r:
                if k not in cols:
                    cols.append(k)
    with open(path, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join("" if r.get(c) is None else str(r.get(c, ""))
                              for c in cols) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--outdir", default=None,
                    help="output directory; default is <rundir>/analysis")
    ap.add_argument("--log", default=None)
    ap.add_argument("--yc", type=float, default=0.605)
    ap.add_argument("--mass", type=float, default=1.0)
    ap.add_argument("--nbins", type=int, default=120)
    ap.add_argument("--nsub", type=int, default=5,
                    help="keep radial profiles every nsub-th dump")
    ap.add_argument("--vis-python", default=None)
    args = ap.parse_args()

    out = args.outdir or os.path.join(args.rundir, "analysis")
    os.makedirs(out, exist_ok=True)
    prof = L.cluster_profile(args.yc)
    summary = {"label": args.label, "rundir": args.rundir,
               "equilibrium": {k: float(prof[k]) for k in
                               ("R_over_M", "Riso_over_M", "M0_over_M", "zc",
                                "alpha_c", "alpha_s", "binding_fraction")}}

    if args.log and os.path.exists(args.log):
        lg = parse_log(args.log)
        write_csv(os.path.join(out, "cycles.csv"),
                  [dict(cycle=int(c), time=float(t), dt=float(d), elapsed=float(e))
                   for c, t, d, e in zip(lg["cycle"], lg["time"], lg["dt"],
                                         lg["elapsed"])])
        if lg["elapsed"].size > 2:
            de = np.diff(lg["elapsed"])
            dc = np.diff(lg["cycle"])
            ok = (de > 0) & (dc > 0)
            summary["s_per_cycle_median"] = float(np.median(de[ok]/dc[ok])) \
                if ok.any() else None
            summary["elapsed_last"] = float(lg["elapsed"][-1])
            summary["cycle_last"] = int(lg["cycle"][-1])
            summary["time_last"] = float(lg["time"][-1])
        summary["banner"] = lg["banner"]
        summary["ledger"] = lg["ledger"]
        summary["bias_lines"] = lg["bias"]
        summary["warnings"] = lg["warnings"]
        summary["boris_fallbacks"] = lg["boris_fallbacks"]
        summary["boris_fallback_cycles"] = lg["boris_fallback_cycles"]
        n_banner = lg["banner"].get("N")
        if n_banner and summary.get("cycle_last"):
            summary["boris_fallback_fraction_per_particle_update"] = float(
                lg["boris_fallbacks"]/(float(n_banner)*summary["cycle_last"]))

    hst = glob.glob(os.path.join(args.rundir, "out", "*.hst"))
    if hst:
        h = L.read_hst(hst[0])
        rows = []
        for i in range(len(h["time"])):
            vol = h["Volume"][i]
            rows.append(dict(time=h["time"][i], dt=h["dt"][i],
                             H_L2=np.sqrt(h["H-norm2"][i]),
                             M_L2=np.sqrt(h["M-norm2"][i]),
                             Mx_L2=np.sqrt(h["Mx-norm2"][i]),
                             My_L2=np.sqrt(h["My-norm2"][i]),
                             Mz_L2=np.sqrt(h["Mz-norm2"][i]),
                             C_L2=np.sqrt(h["C-norm2"][i]),
                             Z_L2=np.sqrt(h["Z-norm2"][i]),
                             Volume=vol,
                             H_rms=np.sqrt(h["H-norm2"][i]/vol) if vol > 0 else np.nan,
                             M_rms=np.sqrt(h["M-norm2"][i]/vol) if vol > 0 else np.nan))
        write_csv(os.path.join(out, "hstnorms.csv"), rows)
        summary["hst_file"] = os.path.basename(hst[0])
        summary["hst_rows"] = len(rows)

    pv = sorted(glob.glob(os.path.join(args.rundir, "out", "pvtk", "*.part.vtk")))
    if pv:
        print(f"reducing {len(pv)} particle dumps")
        rows = reduce_particles(pv, prof, args.mass, args.nbins, args.nsub,
                                os.path.join(out, "particles.npz"))
        write_csv(os.path.join(out, "shells.csv"), rows)
        summary["n_particle_dumps"] = len(pv)
        if rows:
            summary["first"] = rows[0]
            summary["last"] = rows[-1]

    for tag, pat, npz in (("z4c", "*.z4c.*.bin", "profiles_z4c.npz"),
                          ("tmunu", "*.tmunu.*.bin", "profiles_tmunu.npz"),
                          ("con2d", "*.con2d.*.bin", "profiles_con.npz"),
                          ("adm", "*.adm.*.bin", "profiles_adm.npz")):
        files = sorted(glob.glob(os.path.join(args.rundir, "out", "bin", pat)))
        if not files:
            continue
        print(f"reducing {len(files)} {tag} slices")
        rows = reduce_fields(files, tag, args.nbins, args.nsub,
                             os.path.join(out, npz), args.vis_python)
        write_csv(os.path.join(out, f"fields_{tag}.csv"), rows)

    cens = glob.glob(os.path.join(args.rundir, "out", "*prtcl_destroy*.csv"))
    if cens:
        summary["destroy_csv"] = os.path.basename(cens[0])

    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
