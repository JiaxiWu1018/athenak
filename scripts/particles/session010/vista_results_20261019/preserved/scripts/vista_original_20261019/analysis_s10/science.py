"""Reviewed reductions and fixed-scale renderers for the closed Session010 branch."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time

from run import BASE, OUT, SOURCE, atomic, check_time, digest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from PIL import Image
import imageio_ffmpeg

sys.path.insert(0, str(SOURCE / "vis/python"))
import bin_convert
from horizon_review_20261005 import accepted_measurements
from validate_initial_s7 import read_particles
from render_progress_movie import choose_tags

COLORS = ("#777777", "#2875B0", "#DB7C26")
LABELS = ("Envelope", "Original left clump", "Original right clump")
LIMITS = ((0, 3_000_000), (3_000_000, 4_000_000), (4_000_000, 5_000_000))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.facecolor": "white", "axes.grid": False})


def savefig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def csvwrite(name, rows, columns):
    np.savetxt(OUT / name, np.asarray(rows), delimiter=",", header=columns, comments="")


def numeric(path, columns=None):
    if not any(l.strip() and not l.startswith("#") for l in Path(path).read_text().splitlines()):
        return np.empty((0, columns or 0))
    return np.loadtxt(path, ndmin=2)


def merge_rows(parts, time_col=1, check_cols=None):
    """Later checkpoint observation wins; persistent conflicts are rejected."""
    result = {}
    for rows in parts:
        for row in rows:
            row = np.asarray(row)
            key = round(float(row[time_col]), 11)
            if key in result and check_cols is not None:
                if not np.allclose(row[check_cols], result[key][check_cols], rtol=3e-6, atol=3e-8, equal_nan=True):
                    raise ValueError("Conflicting restart overlap at t=" + str(key))
            result[key] = row
    return np.asarray([result[k] for k in sorted(result)])


def live(row):
    return bool(np.isfinite(row[2:5]).all() and np.isfinite(row[15]) and
                (row[13] > 0 or row[16] > 0))


def contiguous_groups(times, cycles, valid, cases):
    groups, active = [], []
    for k in range(len(times)):
        gap = active and (cases[k] != cases[active[-1]] or cycles[k] - cycles[active[-1]] > 20)
        if not valid[k] or gap:
            if active:
                groups.append(np.asarray(active)); active = []
        if valid[k]:
            active.append(k)
    if active:
        groups.append(np.asarray(active))
    return groups


def file_time(path):
    with Path(path).open("rb") as stream:
        match = re.search(rb"time=\s*([-+0-9.eE]+)", stream.read(256))
    if not match:
        raise ValueError("Missing output time: " + str(path))
    return float(match[1])


def primary_files(manifest, pattern):
    by_time = {}
    for root in manifest["ordered_outputs"]:
        for path in sorted(Path(root).rglob(pattern)):
            tm = file_time(path)
            if tm <= manifest["cutoff"] + 5.1e-5:
                by_time[round(tm, 9)] = path
    return [by_time[k] for k in sorted(by_time)]


def binary_header(stream):
    if stream.readline().strip() != b"Athena binary output version=1.1":
        raise ValueError("Unsupported native binary header")
    count = int(stream.readline().split(b"=")[-1]); meta = {}
    for _ in range(count - 1):
        key, value = stream.readline().decode().split("=", 1); meta[key.strip()] = value.strip()
    nvars = int(stream.readline().split(b"=")[-1])
    variables = stream.readline().decode().split()[1:]
    size = int(stream.readline().split(b"=")[-1]); header = stream.read(size)
    return meta, nvars, variables, header


def mesh_count(path):
    """Read fixed-size coarsened mesh records without allocating field arrays."""
    with Path(path).open("rb") as stream:
        meta, nvars, _, _ = binary_header(stream)
        offset = stream.tell(); indexes = np.frombuffer(stream.read(24), dtype="<i4")
        cells = int(np.prod(indexes[1::2] - indexes[::2] + 1))
        record = 40 + 6 * int(meta["size of location"]) + cells * nvars * int(meta["size of variable"])
    size = Path(path).stat().st_size - offset
    if size % record:
        raise ValueError("Partial coarsened mesh output")
    return [float(meta["time"]), int(meta["cycle"]), size // record]


def reduce(manifest):
    check_time(); roots = [Path(p) for p in manifest["ordered_outputs"]]
    track_parts = [[], []]
    for case_id, root in enumerate(roots):
        for i in (0, 1):
            rows = numeric(root / f"s10_orbit_gw.co_{i}.txt", 17)
            track_parts[i].append(np.c_[rows, np.full(len(rows), case_id)])
    tracks = [merge_rows(parts, check_cols=list(range(2, 13)) + [16]) for parts in track_parts]
    for i, tr in enumerate(tracks):
        np.save(OUT / f"track_{i}.npy", tr)
        csvwrite(f"tracker_{i}.csv", tr, "cycle,time,x,y,z,vx,vy,vz,logical_level,dx,dy,dz,source,core_count,core_rest_mass,lapse_min,AH_tracking,case")
    times = np.intersect1d(tracks[0][:, 1], tracks[1][:, 1])
    a, b = [tr[np.searchsorted(tr[:, 1], times)] for tr in tracks]
    valid = np.array([live(x) and live(y) for x, y in zip(a, b)])
    vec = b[:, 2:5] - a[:, 2:5]; distance = np.linalg.norm(vec, axis=1)
    phase = np.full(len(times), np.nan); arc = phase.copy(); omega = phase.copy()
    groups = contiguous_groups(times, a[:, 0], valid, a[:, -1]); offset = 0.
    for g in groups:
        ph = np.unwrap(np.arctan2(vec[g, 1], vec[g, 0])); phase[g] = ph
        arc[g] = offset + ph - ph[0]; offset = arc[g[-1]]
        if len(g) >= 3:
            omega[g] = np.gradient(ph, times[g])
    distance[~valid] = np.nan
    csvwrite("orbit.csv", np.c_[times, distance, phase, arc, omega, valid],
             "time,coordinate_separation,phase_within_valid_interval,observed_arc_excluding_gaps,coordinate_Omega,valid")
    np.savez(OUT / "orbit.npz", time=times, distance=distance, arc=arc, valid=valid)
    horizons = []; quality = {}
    for i in (0, 1, 2):
        merged = []
        for case_id, root in enumerate(roots):
            with (root / f"s10_orbit_gw.horizon_consumer_{i}.csv").open() as f:
                consumers = list(csv.DictReader(f))
            summaries = numeric(root / f"s10_orbit_gw.horizon_summary_{i}.txt", 21)
            matched, stats = accepted_measurements(consumers, sorted(summaries.tolist(), key=lambda r: r[1]))
            if stats["unmatched"] or stats["ambiguous"]:
                raise ValueError("Accepted horizon lacks unique geometry match")
            quality[f"{case_id}:{i}"] = stats
            for c, row in matched:
                row[0], row[1] = int(c["cycle"]), float(c["time"])
                merged.append(row + [float(c["rmin_cells"]), case_id])
        rows = merge_rows([merged]) if merged else np.empty((0, 23))
        horizons.append(rows); np.save(OUT / f"horizon_{i}.npy", rows)
        csvwrite(f"accepted_horizon_{i}.csv", rows,
                 "cycle,time,M_BH,Sx,Sy,Sz,S,area,hrms,hmean,meanradius,rmin,M_irr,spin,Px,Py,Pz,P,center_x,center_y,center_z,rmin_cells,case")
    atomic(OUT / "horizon_joins.json", quality)
    waves = []
    for k, root in enumerate(roots):
        rp, ip = root / "waveforms/rpsi4_real_0040.txt", root / "waveforms/rpsi4_imag_0040.txt"
        if not rp.exists() and not ip.exists():
            continue
        reals, imag = numeric(rp), numeric(ip)
        if reals.shape != imag.shape or reals.shape[1] != 78 or not np.array_equal(reals[:, 0], imag[:, 0]):
            raise ValueError("Raw complex waveform contract failed")
        if not np.isfinite(reals).all() or not np.isfinite(imag).all():
            raise ValueError("Nonfinite waveform")
        waves.append(np.c_[reals, imag[:, 1:], np.full(len(reals), k)])
    wave = merge_rows(waves, time_col=0, check_cols=slice(1, 155)); np.save(OUT / "raw_wave.npy", wave)
    csvwrite("raw_quadrupole.csv", wave[:, [0, 1, 78, 3, 80, 5, 82]], "time,re_2m2,im_2m2,re_20,im_20,re_22,im_22")
    history = merge_rows([numeric(root / "s10_orbit_gw.z4c.user.hst") for root in roots], time_col=0)
    np.save(OUT / "history.npy", history)
    mesh = merge_rows([[mesh_count(p) for p in root.rglob("*.mesh.*.cbin")] for root in roots], time_col=0)
    np.save(OUT / "mesh.npy", mesh); csvwrite("amr_history.csv", mesh, "time,cycle,leaf_blocks")
    # One array per removal event; tags must be globally unique on this primary branch.
    removed = []; seen = np.zeros(5_000_000, dtype=bool)
    for root in roots:
        check_time()
        p = root / "s10_orbit_gw.prtcl_destroy.csv"
        if not p.exists():
            continue
        with p.open() as f:
            columns = f.readline().lstrip("# ").strip().split(",")
        use = [columns.index(k) for k in ("time", "tag", "x1", "x2", "u1_cov", "u2_cov", "component", "mass")]
        if p.stat().st_size < 200:
            continue
        arr = np.loadtxt(p, delimiter=",", skiprows=1, usecols=use, ndmin=2)
        if not len(arr):
            continue
        # The approved run uses low-lapse removal only, with no AH/exit removal.
        if p.read_bytes().count(b",lapse,") != len(arr):
            raise ValueError("Unexpected removal reason; do not equate it with low-lapse excision")
        tags = arr[:, 1].astype(np.int64)
        if not np.isfinite(arr).all() or np.any(arr[:, 7] <= 0) or np.any(tags < 0) or np.any(tags >= len(seen)):
            raise ValueError("Invalid removal ledger")
        if np.any(seen[tags]) or len(np.unique(tags)) != len(tags):
            raise ValueError("Duplicate removal tag")
        seen[tags] = True
        expected_component = np.where(tags < 3_000_000, 0, np.where(tags < 4_000_000, 1, 2))
        if not np.array_equal(expected_component, arr[:, 6]):
            raise ValueError("Removal component/tag mismatch")
        jz = arr[:, 7] * (arr[:, 2] * arr[:, 5] - arr[:, 3] * arr[:, 4])
        removed.append(np.c_[arr[:, 0], arr[:, 6], arr[:, 7], jz, tags])
    deaths = np.concatenate(removed) if removed else np.empty((0, 5))
    deaths = deaths[np.argsort(deaths[:, 0], kind="stable")]
    np.save(OUT / "removals.npy", deaths)
    if len(deaths) != sum(manifest["checkpoint"]["removed"]):
        raise ValueError("Removal event count does not match final native checkpoint")
    cache = OUT / "cache/particles"; cache.mkdir(parents=True, exist_ok=True)
    particle_files = primary_files(manifest, "*.part.vtk"); ledger, shell, coverage = [], [], []
    chosen = None; cohort_details = None
    for index, p in enumerate(particle_files):
        check_time(); print("PARTICLES", index + 1, "/", len(particle_files), p.name, flush=True)
        pt = read_particles(p); tm = pt["time"]; tags = pt["tag_float"].astype(np.int64)
        if (not np.array_equal(tags, pt["tag_float"]) or len(np.unique(tags)) != len(tags)
                or np.any(tags < 0) or np.any(tags >= 5_000_000)):
            raise ValueError("Invalid particle identities")
        xyz = pt["position"].astype(np.float64); mu = pt["mass"].astype(np.float64); mom = pt["momentum"].astype(np.float64)
        if (not np.isfinite(xyz).all() or not np.isfinite(mu).all() or not np.isfinite(mom).all()
                or not np.isfinite(pt["energy"]).all() or np.any(mu <= 0)):
            raise ValueError("Invalid particle snapshot")
        if chosen is None:
            if tm != 0 or len(tags) != 5_000_000:
                raise ValueError("Fixed cohort must be selected from the full fresh t0 state")
            chosen, cohort_details = choose_tags(tags); chosen.sort(); np.save(OUT / "cohort_tags.npy", chosen)
        selected = np.isin(tags, chosen); group = np.where(tags < 3_000_000, 0, np.where(tags < 4_000_000, 1, 2))
        np.savez(cache / f"{index:04d}.npz", time=tm, xyz=xyz[selected].astype(np.float32),
                 tag=tags[selected], group=group[selected], alive=len(tags))
        for comp, (low, high) in enumerate(LIMITS):
            q = group == comp; mass = mu[q].sum(); jz = np.sum(mu[q] * np.cross(xyz[q], mom[q])[:, 2])
            center = np.sum(mu[q, None] * xyz[q], axis=0) / mass if mass else np.full(3, np.nan)
            dead = deaths[(deaths[:, 1] == comp) & (deaths[:, 0] <= tm + 5.1e-5)]
            row = [tm, comp, q.sum(), mass, *center, jz, len(dead), dead[:, 2].sum(), dead[:, 3].sum()]
            ledger.append(row); balanced = int(q.sum()) + len(dead) == high - low
            coverage.append(dict(time=tm, component=comp, count_balanced=balanced))
            if not balanced:
                raise ValueError(f"Particle/removal census mismatch: t={tm}, component={comp}")
        rad = np.linalg.norm(xyz, axis=1); q = (rad >= 38) & (rad <= 42)
        shell.append([tm, q.sum(), mu[q].sum(), mu.sum()])
    ledger = np.asarray(ledger); np.save(OUT / "matter.npy", ledger)
    csvwrite("matter_components.csv", ledger, "time,component,alive_count,alive_rest_mass,center_x,center_y,center_z,alive_Jz_cov,removed_count,removed_rest_mass,removed_Jz_cov")
    csvwrite("matter_near_r40.csv", shell, "time,count_r38_to42,rest_mass_r38_to42,all_alive_rest_mass")
    for comp in range(3):
        q = ledger[ledger[:, 1] == comp]; total_mass = q[:, 3] + q[:, 9]
        if not np.allclose(total_mass, total_mass[0], rtol=3e-6, atol=1e-10):
            raise ValueError("Alive plus removed rest mass does not balance")
    summary = dict(cutoff=manifest["cutoff"], tracker_end=float(times[-1]),
                   coordinate_separation_final=float(distance[-1]),
                   observed_arc_degrees_excluding_gaps=float(np.degrees(offset)),
                   endpoint_orientation_change_degrees=float(np.degrees(np.arctan2(vec[-1, 1], vec[-1, 0]) - np.arctan2(vec[0, 1], vec[0, 0]))),
                   horizons={str(i): dict(accepted=len(h), first_time=float(h[0, 1]) if len(h) else None,
                              last_mass=float(h[-1, 2]) if len(h) else None,
                              last_spin=float(h[-1, 13]) if len(h) else None) for i, h in enumerate(horizons)},
                   removal_events=len(deaths), wave_samples=len(wave), wave_end=float(wave[-1, 0]),
                   waveform_max_spacing=float(np.diff(wave[:, 0]).max()),
                   particle_snapshots=len(particle_files), particle_times=sorted(set(ledger[:, 0].tolist())),
                   census_balanced=True, rest_mass_balanced=True, cohort=cohort_details,
                   constraints_mask="proper-volume chi>=0.0625; NOT an apparent-horizon exterior mask",
                   waveform_status="Raw finite-radius rPsi4 only. Strain, spectrum and ringdown deferred.")
    atomic(OUT / "REDUCTIONS.json", summary)


def raster(data, variable, plane, limit, cells=512):
    """Sample the native leaf slice at pixel centers. No smoothing or norm claims."""
    axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane]
    grid = -limit + (np.arange(cells) + .5) * (2 * limit / cells)
    result = np.full((cells, cells), np.nan, dtype=np.float32)
    levels = np.full_like(result, -1)
    for k, geom in enumerate(data["mb_geometry"]):
        x0, x1 = geom[2 * axes[0]:2 * axes[0] + 2]; y0, y1 = geom[2 * axes[1]:2 * axes[1] + 2]
        ix = np.where((grid >= x0) & (grid < x1))[0]; iy = np.where((grid >= y0) & (grid < y1))[0]
        if not len(ix) or not len(iy):
            continue
        values = np.squeeze(data["mb_data"][variable][k])
        if values.ndim != 2:
            raise ValueError("Expected a native 2D slice")
        bx = np.minimum(((grid[ix] - x0) / (x1 - x0) * values.shape[1]).astype(int), values.shape[1] - 1)
        by = np.minimum(((grid[iy] - y0) / (y1 - y0) * values.shape[0]).astype(int), values.shape[0] - 1)
        level = int(data["mb_logical"][k, 3]) - 3
        old = levels[np.ix_(iy, ix)]; take = level >= old
        sample = values[np.ix_(by, bx)]
        result[np.ix_(iy, ix)] = np.where(take, sample, result[np.ix_(iy, ix)])
        levels[np.ix_(iy, ix)] = np.where(take, level, old)
    if np.any(levels < 0) or not np.isfinite(result).all():
        raise ValueError("Uncovered or nonfinite raster pixels")
    return result, levels


def fields(manifest):
    directory = OUT / "cache/fields"; directory.mkdir(parents=True, exist_ok=True)
    files = primary_files(manifest, "*.tmunu_xy.*.bin"); frames = []
    for index, path in enumerate(files):
        check_time(); target = directory / f"{index:04d}.npz"
        print("FIELDS", index + 1, "/", len(files), flush=True)
        if not target.exists():
            name = path.name; root = path.parent
            data = bin_convert.read_binary(str(path)); tm, cycle = data["time"], data["cycle"]
            ec, levels = raster(data, "tmunu_E", "xy", 6)
            ew, _ = raster(data, "tmunu_E", "xy", 35)
            vertical = bin_convert.read_binary(str(root / name.replace("tmunu_xy", "tmunu_xz")))
            exz, _ = raster(vertical, "tmunu_E", "xz", 35)
            metric = bin_convert.read_binary(str(root / name.replace("tmunu_xy", "z4c_xy")))
            alpha, _ = raster(metric, "z4c_alpha", "xy", 6)
            con = bin_convert.read_binary(str(root / name.replace("tmunu_xy", "con_xy")))
            h, _ = raster(con, "con_H", "xy", 6)
            for d in (vertical, metric, con):
                if d["cycle"] != cycle or d["time"] != tm:
                    raise ValueError("Unsynchronized native field slices")
            np.savez(target, time=tm, cycle=cycle, energy_c=ec, energy_w=ew, energy_xz=exz,
                     alpha=alpha, H=h, level=levels)
        with np.load(target) as z:
            frames.append(dict(path=str(target), time=float(z["time"]), cycle=int(z["cycle"]), source=str(path)))
    positives = []; conmax = 0.; alpha_max = 1.
    for row in frames:
        with np.load(row["path"]) as z:
            for name in ("energy_c", "energy_w", "energy_xz"):
                a = z[name]; positives.append(a[a > 0][::16])
            conmax = max(conmax, float(np.max(np.abs(z["H"]))))
            alpha_max = max(alpha_max, float(np.max(z["alpha"])))
    positive = np.concatenate(positives)
    scale = dict(energy_min=max(1e-14, float(positive.min())), energy_max=float(positive.max()),
                 H_min=1e-8, H_max=max(1e-7, conmax), alpha_max=alpha_max,
                 note="Fixed global scales. E extrema sampled every 16th positive raster pixel; colorbar extends both ends.")
    atomic(OUT / "FIELDS.json", dict(frames=frames, scales=scale,
             resampling="512-square pixel-center native-leaf sampling; not a proper-volume reduction",
             missing_data="No state interpolation; native timestamps retained"))


def decorations(ax, ylabel, restarts, formation):
    ax.set(xlabel=r"$t/M_{ref}$", ylabel=ylabel)
    ax.grid(alpha=.16)
    for t in restarts:
        ax.axvline(t, color=".65", lw=.6, ls=":")
    for t in formation:
        ax.axvline(t, color=".4", lw=.7, ls="--")


def static_plots(manifest):
    tracks = [np.load(OUT / f"track_{i}.npy") for i in (0, 1)]
    horizons = [np.load(OUT / f"horizon_{i}.npy") for i in (0, 1, 2)]
    formation = [h[0, 1] for h in horizons[:2] if len(h)]
    restarts = [json.loads((Path(p) / "result.json").read_text())["checkpoint"]["time"] for p in manifest["cases"][:-1]]
    orbit = np.load(OUT / "orbit.npz")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    for i, tr in enumerate(tracks):
        good = np.array([live(r) for r in tr]); xyz = tr[:, 2:5].copy(); xyz[~good] = np.nan
        for case in np.unique(tr[:, -1]):
            q = tr[:, -1] == case
            axes[0].plot(xyz[q, 0], xyz[q, 1], color=COLORS[i + 1], ls=("-" if i == 0 else "--"), label=LABELS[i + 1] if case == 0 else None)
        axes[0].scatter(xyz[-1, 0], xyz[-1, 1], color=COLORS[i + 1], s=22)
    axes[0].set(xlabel=r"$x/M_{ref}$", ylabel=r"$y/M_{ref}$", aspect="equal"); axes[0].legend(fontsize=8)
    for case in np.unique(tracks[0][:, -1]):
        q = (orbit["time"] >= tracks[0][tracks[0][:, -1] == case, 1].min()) & (orbit["time"] <= tracks[0][tracks[0][:, -1] == case, 1].max())
        axes[1].plot(orbit["time"][q], orbit["distance"][q], color=COLORS[1])
        axes[2].plot(orbit["time"][q], np.degrees(orbit["arc"][q]), color=COLORS[1])
    decorations(axes[1], "Coordinate separation / Mref", restarts, formation)
    decorations(axes[2], "Observed rotation [degrees]", restarts, formation)
    fig.suptitle("Session010 — coordinate motion; rotation excludes unobserved gaps")
    savefig(fig, "01_orbit")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    for i, h in enumerate(horizons[:2]):
        for ax, col in zip(axes, (2, 13, 21)):
            ax.plot(h[:, 1], h[:, col], ".", ms=.7, color=COLORS[i + 1], label=LABELS[i + 1])
    for ax, lab in zip(axes, ("Accepted horizon mass / Mref", "Spin (coordinate prescription)", "Minimum radius / local cell size")):
        decorations(ax, lab, restarts, formation); ax.legend(fontsize=8)
    fig.suptitle("Session010 — uniquely matched, accepted individual horizons")
    savefig(fig, "02_horizons")
    matter = np.load(OUT / "matter.npy")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    figj, aj = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for i in range(3):
        q = matter[matter[:, 1] == i]; color, label = COLORS[i], LABELS[i]
        axes[0].plot(q[:, 0], q[:, 2] / 1e6, "o-", ms=3, color=color, label=label)
        axes[1].plot(q[:, 0], q[:, 3], "o-", ms=3, color=color)
        axes[1].plot(q[:, 0], q[:, 9], "--", color=color)
        axes[2].plot(q[:, 0], q[:, 3] + q[:, 9], "o-", ms=3, color=color)
        aj[0].plot(q[:, 0], q[:, 7], "o-", ms=3, color=color, label=label)
        aj[1].plot(q[:, 0], q[:, 10], "o-", ms=3, color=color)
    axes[0].legend(fontsize=8)
    for axis in axes:
        axis.set_ylim(bottom=0)
    for ax, lab in zip(axes, ("Active particles [millions]", "Rest mass: active solid, removed dashed", "Active + removed rest mass")):
        decorations(ax, lab, restarts, formation)
    for ax, lab in zip(aj, ("Surviving matter Jz (covariant u)", "Removed matter Jz (separate ledger)")):
        decorations(ax, lab, restarts, formation)
    aj[0].legend(fontsize=8)
    fig.suptitle("Session010 — particle and rest-mass accounting, not black-hole accretion")
    figj.suptitle("Session010 — matter angular momentum; not total-system angular momentum")
    savefig(fig, "03_matter_accounting"); savefig(figj, "04_matter_angular_momentum")
    history, mesh = np.load(OUT / "history.npy"), np.load(OUT / "mesh.npy")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    valid = (history[:, 0] > 0) & (history[:, 10] > 0) & np.isfinite(history).all(axis=1)
    for col, label, color in ((3, "Hamiltonian", COLORS[1]), (4, "Momentum", COLORS[2])):
        axes[0].semilogy(history[valid, 0], np.sqrt(history[valid, col] / history[valid, 10]), color=color, label=label)
    axes[0].legend(fontsize=8); axes[1].plot(mesh[:, 0], mesh[:, 2], color=COLORS[1])
    axes[1].axhline(5760, color=".4", ls="--", lw=.8, label="Capacity stop proxy"); axes[1].legend(fontsize=8)
    for i, tr in enumerate(tracks):
        axes[2].step(tr[:, 1], tr[:, 9], where="post", color=COLORS[i + 1], ls=("-" if i == 0 else "--"), label=LABELS[i + 1])
    axes[2].set_yscale("log", base=2); axes[2].legend(fontsize=8)
    for ax, lab in zip(axes, ("Proper-volume RMS (chi >= 0.0625)", "Leaf mesh blocks", "Cell spacing at tracker / Mref")):
        decorations(ax, lab, restarts, formation)
    fig.suptitle("Session010 — numerical diagnostics; constraint mask is NOT AH exterior")
    savefig(fig, "06_health_amr")
    wave = np.load(OUT / "raw_wave.npy"); t = wave[:, 0]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), layout="constrained", sharex=True)
    axes[0].plot(t, wave[:, 5], color=COLORS[1], label="Real (2,2)")
    axes[0].plot(t, wave[:, 82], color=COLORS[2], ls="--", label="Imaginary (2,2)")
    for j, label, color in ((1, "(2,-2)", "#777777"), (3, "(2,0)", COLORS[2]), (5, "(2,2)", COLORS[1])):
        axes[1].semilogy(t, np.hypot(wave[:, j], wave[:, j + 77]), color=color, label=label)
    axes[2].plot(t[1:], np.diff(t), ".", ms=2, color=COLORS[1]); axes[2].axhline(.025, ls="--", color=".4", lw=.8)
    for ax, lab in zip(axes, ("Raw r Psi4(2,2)", "Raw multipole amplitude", "Saved sample spacing / Mref")):
        decorations(ax, lab, restarts, formation)
    axes[0].legend(); axes[1].legend(); fig.suptitle("Session010 — raw extraction at r=40 Mref; no strain/ringdown interpretation")
    savefig(fig, "07_raw_gw")


def marker(ax, tm, plane="xy"):
    dims = {"xy": (2, 3), "xz": (2, 4)}[plane]
    for i in (0, 1):
        tr = np.load(OUT / f"track_{i}.npy"); j = np.searchsorted(tr[:, 1], tm + 1e-10) - 1
        if j >= 0 and tm - tr[j, 1] <= .15 and live(tr[j]):
            ax.plot(tr[j, dims[0]], tr[j, dims[1]], "+", color=COLORS[i + 1], ms=7, mew=1.4)


def image_panel(fig, ax, value, limit, norm, cmap, label, tm, plane="xy", markers=True):
    im = ax.imshow(value, origin="lower", extent=(-limit, limit, -limit, limit),
                   norm=norm, cmap=cmap, interpolation="nearest", rasterized=True)
    ax.set(xlabel="x / Mref", ylabel=("y" if plane == "xy" else "z") + " / Mref", aspect="equal")
    if markers:
        marker(ax, tm, plane)
    cb = fig.colorbar(im, ax=ax, shrink=.82, pad=.025, extend="both" if isinstance(norm, LogNorm) else "neither")
    cb.set_label(label, fontsize=8); cb.ax.tick_params(labelsize=8)
    return im


def make_field_frame(row, scales, kind, target):
    with np.load(row["path"]) as z:
        tm = float(z["time"]); en = LogNorm(scales["energy_min"], scales["energy_max"])
        if kind == "matter":
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=100, layout="constrained")
            for ax, key, limit, plane in zip(axes, ("energy_w", "energy_c", "energy_xz"), (35, 6, 35), ("xy", "xy", "xz")):
                image_panel(fig, ax, np.ma.masked_less_equal(z[key], 0), limit, en, "magma", "Eulerian matter energy E", tm, plane)
            title = "Wide orbital plane | Central orbital plane | Wide vertical plane"
        else:
            fig, axes = plt.subplots(2, 2, figsize=(10, 9), dpi=100, layout="constrained")
            specs = [("energy_c", en, "magma", "Eulerian matter energy E"),
                     ("alpha", Normalize(0, scales["alpha_max"]), "viridis", "Lapse alpha"),
                     ("H", LogNorm(scales["H_min"], scales["H_max"]), "cividis", "Absolute Hamiltonian constraint |H|"),
                     ("level", Normalize(0, 10), "viridis", "AMR level above root")]
            for ax, (key, norm, cmap, label) in zip(axes.flat, specs):
                value = np.abs(z[key]) if key == "H" else z[key]
                if isinstance(norm, LogNorm): value = np.ma.masked_less_equal(value, 0)
                image_panel(fig, ax, value, 6, norm, cmap, label, tm)
            title = "Central matter, lapse, raw constraint and AMR"
        fig.suptitle(f"Session010 — {title}\nSaved t/Mref = {tm:.5f}; fixed scales; '+' = valid tracker")
        fig.savefig(target, dpi=100); plt.close(fig)


def movie_indices(times, cutoff, fps=12, speed=2.):
    times = np.asarray(times, dtype=float)
    # Particle VTK writes six significant digits; the verified native checkpoint
    # preserves the same final time at double precision. Snap only that endpoint.
    times = np.where(np.abs(times - cutoff) <= 5.1e-5, cutoff, times)
    playback = np.arange(int(math.ceil(cutoff / speed * fps)) + 1) * speed / fps
    return np.maximum(0, np.searchsorted(times, np.minimum(playback, cutoff), side="right") - 1)


def encode_frames(paths, times, target, cutoff):
    """Hold last recorded state on a uniform playback timeline, never interpolate."""
    indices = movie_indices(np.asarray(times), cutoff)
    if indices[-1] != len(paths) - 1:
        raise ValueError("Movie timeline omits the final saved state")
    concat = target.with_suffix(".ffconcat")
    with concat.open("w") as stream:
        stream.write("ffconcat version 1.0\n")
        for j in indices:
            escaped = str(paths[int(j)]).replace("'", "'\\''")
            stream.write(f"file '{escaped}'\nduration {1 / 12:.12f}\n")
        stream.write(f"file '{paths[int(indices[-1])]}'\n")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-threads", "4", "-f", "concat", "-safe", "0",
                    "-i", str(concat), "-vf", "fps=12,pad=ceil(iw/2)*2:ceil(ih/2)*2", "-frames:v", str(len(indices)),
                    "-c:v", "libx264", "-threads", "4", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)], check=True)
    return dict(source_frames=len(paths), encoded_frames=len(indices), fps=12, physical_M_per_playback_second=2,
                source_times=list(map(float, times)), final_source_frame_included=True,
                note="Zero-order hold of saved states; no interpolated motion; actual snapshot time displayed. Final VTK timestamp is matched within its printing precision.")


def render(manifest):
    check_time(); static_plots(manifest)
    field = json.loads((OUT / "FIELDS.json").read_text()); rows, scales = field["frames"], field["scales"]
    movies = {}
    # A representative snapshot montage uses exactly the same scales as the movies.
    desired = (0., 8.5, 18., manifest["cutoff"])
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), layout="constrained")
    selected = []
    for col, tm in enumerate(desired):
        row = min(rows, key=lambda r: abs(r["time"] - tm)); selected.append(row["time"])
        with np.load(row["path"]) as z:
            for axis, key, norm, cmap, label in ((axes[0, col], "energy_c", LogNorm(scales["energy_min"], scales["energy_max"]), "magma", "Matter energy E"),
                                               (axes[1, col], "alpha", Normalize(0, scales["alpha_max"]), "viridis", "Lapse alpha")):
                values = np.ma.masked_less_equal(z[key], 0) if key == "energy_c" else z[key]
                image_panel(fig, axis, values, 6, norm, cmap, label, row["time"])
                axis.set_title(f"t/Mref = {row['time']:.4f}")
    fig.suptitle("Session010 — collapse snapshots; fixed central window and scales")
    savefig(fig, "05_collapse_snapshots")
    for kind in ("matter", "health"):
        directory = OUT / f"frames/{kind}"; directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for k, row in enumerate(rows):
            check_time(); target = directory / f"{k:04d}.png"; paths.append(target)
            if not target.exists(): make_field_frame(row, scales, kind, target)
            if k % 10 == 0: print("RENDER", kind, k + 1, "/", len(rows), flush=True)
        target = OUT / ("matter_evolution.mp4" if kind == "matter" else "collapse_amr_health.mp4")
        movies[target.name] = encode_frames(paths, [r["time"] for r in rows], target, manifest["cutoff"])
    files = sorted((OUT / "cache/particles").glob("*.npz")); paths, times = [], []
    directory = OUT / "frames/particles3d"; directory.mkdir(parents=True, exist_ok=True)
    for k, source in enumerate(files):
        check_time(); target = directory / f"{k:04d}.png"; paths.append(target)
        with np.load(source) as z:
            tm, xyz, group = float(z["time"]), z["xyz"], z["group"]; times.append(tm)
            if target.exists(): continue
            fig = plt.figure(figsize=(13, 6), dpi=100, layout="constrained")
            for panel, limit in enumerate((35, 6)):
                ax = fig.add_subplot(1, 2, panel + 1, projection="3d")
                for i in range(3):
                    q = (group == i) & (np.max(np.abs(xyz), axis=1) <= limit)
                    ax.scatter(*xyz[q].T, s=.7 if i == 0 else 1., alpha=.12 if i == 0 else .55,
                               c=COLORS[i], label=LABELS[i], rasterized=True, depthshade=False)
                ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit), xlabel="x / Mref", ylabel="y / Mref", zlabel="z / Mref", box_aspect=(1, 1, 1))
                ax.view_init(elev=25, azim=-60)
                ax.set_title("Envelope context" if panel == 0 else "Central close-up")
                if panel == 0: ax.legend(fontsize=8, markerscale=4, loc="upper left")
            fig.suptitle(f"Session010 — fixed-identity 3D particle sample; saved t/Mref = {tm:.5f}\nFixed cameras; frames held between sparse snapshots; removed particles are not replaced")
            fig.savefig(target, dpi=100); plt.close(fig)
        print("RENDER particles3d", k + 1, "/", len(files), flush=True)
    target = OUT / "particles_3d.mp4"
    movies[target.name] = encode_frames(paths, times, target, manifest["cutoff"])
    atomic(OUT / "MOVIES.json", movies)


def verify(manifest):
    import pypdfium2
    figures = sorted(OUT.glob("[0-9][0-9]_*.png")); movies = sorted(OUT.glob("*.mp4")); results = []
    if len(figures) != 7 or len(movies) != 3:
        raise ValueError("Expected seven static figures and three movies")
    pdf_previews = OUT / "qa/pdf_previews"; pdf_previews.mkdir(parents=True, exist_ok=True)
    for path in figures:
        with Image.open(path) as im:
            im.verify()
        if not path.with_suffix(".pdf").exists(): raise ValueError("Missing PDF companion")
        doc = pypdfium2.PdfDocument(str(path.with_suffix(".pdf")))
        if len(doc) != 1:
            raise ValueError("Figure PDF should contain exactly one page")
        bitmap = doc[0].render(scale=1.4)
        bitmap.to_pil().save(pdf_previews / path.name)
        bitmap.close(); doc.close()
    for path in movies:
        check_time()
        decoder = imageio_ffmpeg.read_frames(str(path), input_params=["-threads", "2"], output_params=["-threads", "2"])
        metadata = next(decoder); count = sum(1 for _ in decoder)
        expected = json.loads((OUT / "MOVIES.json").read_text())[path.name]["encoded_frames"]
        if count != expected:
            raise ValueError("Decoded movie frame count mismatch")
        results.append(dict(path=path.name, sha256=digest(path), bytes=path.stat().st_size,
                            decoded_frames=count, metadata=metadata))
    # This is machine verification only; the agent's visual inspection is recorded separately.
    atomic(OUT / "VERIFICATION.json", dict(passed=True, png_files=len(figures), pdf_files=len(figures), movies=results,
             full_movie_decode=True, all_pdfs_rasterized=True, visual_review_pending=True, scientific_convergence_claim=False))
    import importlib.metadata
    provenance = dict(python=sys.version, job=os.environ["SLURM_JOB_ID"],
                      versions={p: importlib.metadata.version(p) for p in ("numpy", "matplotlib", "pillow", "h5py", "imageio-ffmpeg", "pypdfium2")},
                      ffmpeg_path=imageio_ffmpeg.get_ffmpeg_exe(), ffmpeg_sha256=digest(imageio_ffmpeg.get_ffmpeg_exe()),
                      analysis_scripts={str(p.relative_to(BASE)): digest(p) for p in sorted((BASE / "analysis_s10").glob("*.py"))},
                      reviewed_helpers={name: digest(BASE / "analysis_support" / name) for name in
                                        ("horizon_review_20261005.py", "validate_initial_s7.py", "render_progress_movie.py")},
                      binary_reader_sha256=digest(SOURCE / "vis/python/bin_convert.py"),
                      source_commit=subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip(),
                      source_status=subprocess.check_output(["git", "-C", str(SOURCE), "status", "--porcelain"], text=True),
                      simulation_changed=False, submitted_jobs=False, copied_or_deleted_raw_data=False)
    atomic(OUT / "PROVENANCE.json", provenance)
    summary = json.loads((OUT / "REDUCTIONS.json").read_text())
    text = f"""# Session010 — first-production analysis\n\nClosed evolution branch through t={manifest['cutoff']:.5f} Mref. Native checkpoint progress and diagnostic coverage remain separate.\n\n## Deliverables\n\nSeven PNG/PDF figures: orbit, accepted horizons, matter accounting, matter angular momentum, collapse snapshots, numerical health/AMR, and raw GW. Three MP4 movies: matter evolution, collapse/AMR health, and fixed-identity 3D particles. Source manifests and reduced tables are retained alongside them.\n\n## Findings and limits\n\n- Latest tracker separation: {summary['coordinate_separation_final']:.5f} Mref; endpoint orientation change: {summary['endpoint_orientation_change_degrees']:.3f} degrees. Rotation is coordinate dependent; no radiation-driven inspiral claim.\n- Accepted individual horizons: first records at {summary['horizons']['0']['first_time']:.5f} and {summary['horizons']['1']['first_time']:.5f} Mref. Latest masses: {summary['horizons']['0']['last_mass']:.6f}, {summary['horizons']['1']['last_mass']:.6f} Mref. Spin uses the inherited coordinate prescription.\n- {summary['removal_events']:,} removal events; every available particle snapshot balances against the exact component removal ledger. Rest mass also balances within float32 VTK precision. Removal is not an independently measured horizon-accretion flux.\n- {summary['wave_samples']} paired finite raw wave samples through {summary['wave_end']:.5f} Mref. Largest sampling interval: {summary['waveform_max_spacing']:.5f} Mref. No strain, spectrum, ringdown or precision-waveform claim.\n- Constraint history uses proper-volume weighting with chi>=0.0625, not an AH-exterior mask. AMR activity is not a convergence demonstration.\n- Field movies use pixel-center resampling of native leaf slices, fixed global scales and no smoothing. The 3D movie is a fixed t0 cohort, not all particles; removed tags are never replenished. Sparse snapshots are held, not interpolated.\n- All units use inherited Mref=1, distinct from source masses, particle rest masses and measured horizon masses. Initial local momentum constraints are unsolved.\n\n## Integrity and reproduction\n\nInputs are frozen in INPUTS.json and checksum receipts are indexed in SEALED.json. Native source and executable are unchanged. Scripts are in analysis_s10/. Run `bash analysis_s10/run.sh` inside a GG allocation to resume; it never submits a job. Raw data/checkpoints are preserved, with no transfer or deletion. Machine checks fully decoded each movie; see VERIFICATION.json and the separate visual-review receipt.\n"""
    (OUT / "RESULTS.md").write_text(text)
    artifacts = [dict(path=str(p.relative_to(OUT)), bytes=p.stat().st_size, sha256=digest(p))
                 for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in ("ARTIFACTS.json", "ATTEMPT.json", "PROGRESS.json", "LAST_FAILURE.json")]
    atomic(OUT / "ARTIFACTS.json", dict(files=artifacts, scientific_completion=False))
