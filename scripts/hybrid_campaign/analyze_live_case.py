#!/usr/bin/env python3
"""Per-run analysis for one live sampler-causality case.

Produces a single JSON + CSV pair per run so that the ensemble stage can work
purely on small summaries.  All quantities are sampler-agnostic: radial
diagnostics use actual particle radii (mass-radius percentiles) rather than tag
groups, because a tag group is an exact radial shell only for the deterministic
and angular-random samplers and is merely an equal-rest-mass label for the
Monte Carlo variants.

Measured per particle-dump time:
  * individual |L| drift relative to t=0, median / p95 / max
  * particle-density multipoles through l=8, and the l=4 amplitude and axis
  * torque projection: the component of dL/dt that is coherent with the l=4
    density mode, which is the quantity the 2026-07-28 diagnostic found growing
  * mass radii R10/R25/R50/R90 and shell spreading
  * particle count, nonfinite states
History-file quantities (constraints, ADM, rest mass, pusher health) are read
from the .hst ledgers written by the solver.
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from initial_realization_diagnostics import (real_sph_harm_matrix,
                                             angular_multipoles, l4_orientation)


def read_vtk(path):
    blob = path.read_bytes()
    time = float(re.search(rb"time=\s*([-+0-9.eE]+)", blob).group(1))
    m = re.search(rb"POINTS\s+(\d+)\s+float", blob)
    n = int(m.group(1))

    def block(marker, count, required=True):
        off = blob.find(marker)
        if off < 0:
            if required:
                raise ValueError(f"{path}: missing {marker!r}")
            return None
        start = blob.find(b"\n", off + len(marker)) + 1
        return np.frombuffer(blob[start:start + 4*count], dtype=">f4").astype(float)

    pos = block(m.group(0), 3*n).reshape(n, 3)
    vel = block(b"VECTORS prtcl_vel float", 3*n).reshape(n, 3)
    tag = block(b"SCALARS ptag float\nLOOKUP_TABLE default", n).astype(np.int64)
    energy = block(b"SCALARS prtcl_energy float\nLOOKUP_TABLE default", n)
    mass = block(b"SCALARS prtcl_mass float\nLOOKUP_TABLE default", n)
    dL = block(b"VECTORS prtcl_dL_dt float", 3*n, required=False)
    order = np.argsort(tag)
    return dict(time=time, pos=pos[order], vel=vel[order], tag=tag[order],
                energy=energy[order], mass=mass[order],
                dL=None if dL is None else dL.reshape(n, 3)[order])


def read_hst(path):
    if not path.exists():
        return None
    labels, rows = None, []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            if "=" in line and "[" in line:
                labels = re.findall(r"\[\d+\]=(\S+)", line)
            continue
        if line.strip():
            rows.append([float(v) for v in line.split()])
    if not rows:
        return None
    arr = np.array(rows)
    if labels and len(labels) == arr.shape[1]:
        return {lab: arr[:, i] for i, lab in enumerate(labels)}
    return {f"c{i}": arr[:, i] for i in range(arr.shape[1])}


def l4_field(unit_vectors):
    """Return the 9 real l=4 coefficients of the angular particle density."""
    theta = np.arccos(np.clip(unit_vectors[:, 2], -1.0, 1.0))
    phi = np.arctan2(unit_vectors[:, 1], unit_vectors[:, 0])
    names, Y = real_sph_harm_matrix(theta, phi, 4)
    coeffs = Y.mean(axis=1)
    a00 = coeffs[0]
    sel = [i for i, (l, _) in enumerate(names) if l == 4]
    return coeffs[sel] / a00


def analyze(run_dir, meta, lmax=8):
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("pvtk/*.part.vtk"))
    if not files:
        return None
    dumps = sorted((read_vtk(f) for f in files), key=lambda d: d["time"])
    d0 = dumps[0]
    tag0 = d0["tag"]
    L0 = np.cross(d0["pos"], d0["vel"])
    L0abs = np.linalg.norm(L0, axis=1)
    r0 = np.linalg.norm(d0["pos"], axis=1)
    index0 = {t: i for i, t in enumerate(tag0)}
    m_p = float(np.median(d0["mass"]))
    a4_0 = l4_field(d0["pos"] / np.maximum(r0, 1e-30)[:, None])
    a4_0_hat = a4_0 / max(np.linalg.norm(a4_0), 1e-300)

    rows = []
    for d in dumps:
        n = len(d["tag"])
        finite = np.isfinite(d["pos"]).all(axis=1) & np.isfinite(d["vel"]).all(axis=1)
        # Post-collapse dumps in the terminal cascade are mostly or entirely
        # nonfinite; they carry no physical content and destabilize the
        # multipole/orientation solves.  Skip dumps that lost >50% of states.
        if finite.sum() < 0.5 * max(n, 1):
            continue
        idx = np.array([index0.get(t, -1) for t in d["tag"]])
        matched = finite & (idx >= 0)
        L = np.cross(d["pos"][matched], d["vel"][matched])
        Labs = np.linalg.norm(L, axis=1)
        base = L0abs[idx[matched]]
        ok = base > 0
        dL_rel = np.abs(Labs[ok] - base[ok]) / base[ok]
        # Radius-matched drift.  The deterministic sampler's innermost midpoint
        # quantile sits at u=0.1596, while the Monte Carlo variants sample down
        # to u~0.012.  Because |L| ~ u^2 near the centre, those few very small
        # radii carry hugely inflated *relative* drift and would bias a naive
        # comparison of the tail.  Restricting to the radial range all samplers
        # populate (r0 above the deterministic minimum) keeps 99.6% of every
        # realization and makes the statistic exactly comparable.
        r0_matched = r0[idx[matched]][ok]
        band = r0_matched >= meta["r_match_min"]
        dL_band = dL_rel[band]
        r = np.linalg.norm(d["pos"][matched], axis=1)
        rs = np.sort(r)

        nvec = d["pos"][matched] / np.maximum(r, 1e-30)[:, None]
        power, _, coeff = angular_multipoles(nvec, lmax)
        try:
            axis4, eig4, l4n = l4_orientation(coeff)
        except np.linalg.LinAlgError:
            axis4 = np.full(3, np.nan)
            eig4, l4n = float("nan"), float("nan")
        a4 = l4_field(nvec)
        a4n = np.linalg.norm(a4)
        # Alignment of the current l=4 density mode with the t=0 mode: 1 means
        # the pattern is locked to the initial realization, 0 means it has been
        # replaced by an unrelated pattern.
        align4 = float(np.dot(a4, a4_0_hat) / max(a4n, 1e-300))

        # Torque coherent with the l=4 density mode.  For every particle build
        # the l=4 reconstruction at its own direction and correlate it with the
        # magnitude of dL/dt; a coherent mode gives a nonzero correlation.
        torque_l4 = float("nan")
        torque_rms = float("nan")
        if d["dL"] is not None:
            tq = d["dL"][matched]
            tq_norm = np.linalg.norm(tq, axis=1)
            torque_rms = float(np.sqrt(np.mean(tq_norm ** 2)))
            theta = np.arccos(np.clip(nvec[:, 2], -1.0, 1.0))
            phi = np.arctan2(nvec[:, 1], nvec[:, 0])
            names, Y = real_sph_harm_matrix(theta, phi, 4)
            f4 = np.zeros(len(nvec))
            for i, (l, mm) in enumerate(names):
                if l == 4:
                    f4 += coeff[(l, mm)] * Y[i]
            if f4.std() > 0 and tq_norm.std() > 0:
                torque_l4 = float(np.corrcoef(f4, tq_norm)[0, 1])

        rows.append({
            "time": d["time"], "t_over_P": d["time"] / meta["period"],
            "n_alive": int(n), "n_finite": int(finite.sum()),
            "n_matched": int(matched.sum()),
            "n_nonfinite": int((~finite).sum()),
            "dL_median": float(np.median(dL_rel)) if dL_rel.size else float("nan"),
            "dL_p95": float(np.percentile(dL_rel, 95)) if dL_rel.size else float("nan"),
            "dL_max": float(dL_rel.max()) if dL_rel.size else float("nan"),
            "dL_mean": float(dL_rel.mean()) if dL_rel.size else float("nan"),
            "dLm_median": float(np.median(dL_band)) if dL_band.size else float("nan"),
            "dLm_p95": (float(np.percentile(dL_band, 95))
                        if dL_band.size else float("nan")),
            "dLm_max": float(dL_band.max()) if dL_band.size else float("nan"),
            "dLm_frac_kept": (float(band.mean()) if band.size else float("nan")),
            "R10": float(np.percentile(rs, 10)), "R25": float(np.percentile(rs, 25)),
            "R50": float(np.percentile(rs, 50)), "R90": float(np.percentile(rs, 90)),
            "r_max": float(rs[-1]), "r_min": float(rs[0]),
            "L_scalar": float(m_p * Labs.sum()),
            "Jx": float(m_p * L[:, 0].sum()), "Jy": float(m_p * L[:, 1].sum()),
            "Jz": float(m_p * L[:, 2].sum()),
            **{f"P{l}": power[l] for l in range(lmax + 1)},
            "l4_amp": float(a4n), "l4_align_t0": align4,
            "l4_axis_x": float(axis4[0]), "l4_axis_y": float(axis4[1]),
            "l4_axis_z": float(axis4[2]),
            "torque_rms": torque_rms, "torque_l4_corr": torque_l4,
        })

    hst = read_hst(run_dir / f"{meta['name']}.user.hst")
    z4c = read_hst(run_dir / f"{meta['name']}.z4c.user.hst")
    last = rows[-1]
    first = rows[0]
    summary = {
        **{k: meta[k] for k in ("name", "model", "sampler", "seed", "period",
                                "q", "npart") if k in meta},
        "n_dumps": len(rows),
        "t_final": last["time"], "t_final_over_P": last["t_over_P"],
        "dL_median_final": last["dL_median"], "dL_p95_final": last["dL_p95"],
        "dL_max_final": last["dL_max"],
        "dLm_median_final": last["dLm_median"], "dLm_p95_final": last["dLm_p95"],
        "dLm_max_final": last["dLm_max"],
        "dLm_frac_kept": last["dLm_frac_kept"],
        "r_match_min": meta["r_match_min"],
        "R10_ratio": last["R10"] / first["R10"],
        "R25_ratio": last["R25"] / first["R25"],
        "R50_ratio": last["R50"] / first["R50"],
        "R90_ratio": last["R90"] / first["R90"],
        "P4_initial": first["P4"], "P4_final": last["P4"],
        "P4_growth": last["P4"] / first["P4"] if first["P4"] > 0 else float("nan"),
        "l4_amp_initial": first["l4_amp"], "l4_amp_final": last["l4_amp"],
        "l4_align_final": last["l4_align_t0"],
        # The l=4 amplitude oscillates rather than growing monotonically for the
        # coherent samplers, so the endpoint value alone is misleading: it
        # depends on where 1P falls in the oscillation.  Report the peak, the
        # time average, and the peak-to-endpoint ratio as well.
        "l4_amp_peak": float(np.max([r["l4_amp"] for r in rows])),
        "l4_amp_peak_time_over_P": float(
            rows[int(np.argmax([r["l4_amp"] for r in rows]))]["t_over_P"]),
        "l4_amp_timemean": float(np.mean([r["l4_amp"] for r in rows])),
        "l4_peak_over_initial": (float(np.max([r["l4_amp"] for r in rows])
                                       / first["l4_amp"])
                                 if first["l4_amp"] > 0 else float("nan")),
        "l4_align_absmean": float(np.mean(np.abs(
            [r["l4_align_t0"] for r in rows]))),
        "torque_l4_corr_peak": float(np.nanmax(
            [abs(r["torque_l4_corr"]) for r in rows])),
        "l4_axis_initial": [first["l4_axis_x"], first["l4_axis_y"], first["l4_axis_z"]],
        "l4_axis_final": [last["l4_axis_x"], last["l4_axis_y"], last["l4_axis_z"]],
        "torque_l4_corr_final": last["torque_l4_corr"],
        "torque_rms_final": last["torque_rms"],
        "n_final": last["n_alive"], "n_lost": first["n_alive"] - last["n_alive"],
        "n_nonfinite_final": last["n_nonfinite"],
        "J_norm_final": float(np.linalg.norm([last["Jx"], last["Jy"], last["Jz"]])),
        "J_norm_initial": float(np.linalg.norm([first["Jx"], first["Jy"], first["Jz"]])),
    }
    if hst:
        def g(k, default=float("nan")):
            return float(hst[k][-1]) if k in hst else default
        m0 = hst.get("M0_alive")
        summary.update({
            "M0_initial": float(m0[0]) if m0 is not None else float("nan"),
            "M0_final": float(m0[-1]) if m0 is not None else float("nan"),
            "M0_rel_change": (float(abs(m0[-1] - m0[0]) / m0[0])
                              if m0 is not None and m0[0] != 0 else float("nan")),
            "geo_fallbacks_final": g("geo_fbacks"),
            "alpha_min_final": g("alpha_min"),
            "rho_max_final": g("rho_max"),
            "vr_inner10_final": g("vr_inner10"),
            "N_alive_hst_final": g("N_alive"),
        })
    if z4c:
        # Rows with non-positive diagnostic volume are invalid, not zero
        # constraints; mask them rather than reporting them as good.
        vol = z4c.get("Volume")
        valid = np.isfinite(vol) & (vol > 0) if vol is not None else None
        for label, tag in (("H-norm2", "ham"), ("M-norm2", "mom"),
                           ("C-norm2", "cnorm"), ("Z-norm2", "znorm")):
            if label not in z4c:
                continue
            series = z4c[label]
            if valid is not None:
                series = series[valid]
            series = series[np.isfinite(series)]
            if series.size == 0:
                continue
            summary[f"{tag}_initial"] = float(series[0])
            summary[f"{tag}_final"] = float(series[-1])
            summary[f"{tag}_max"] = float(series.max())
            summary[f"{tag}_growth"] = (float(series[-1] / series[0])
                                        if series[0] > 0 else float("nan"))
        summary["z4c_rows_valid"] = int(valid.sum()) if valid is not None else -1
        summary["z4c_rows_total"] = int(len(z4c["Volume"])) if vol is not None else -1
    return summary, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--r-match-min", type=float, default=None,
                    help="override the radius-matching threshold (isotropic M)")
    args = ap.parse_args()

    manifest = json.loads((args.root / "inputs" / "input_manifest.json").read_text())
    meta = next(c for c in manifest if c["name"] == args.case)
    # The radius-matching threshold is the deterministic sampler's innermost
    # midpoint quantile for this model and N_r, expressed as an isotropic
    # radius, so it is identical for every sampler in the comparison.
    if args.r_match_min is not None:
        meta["r_match_min"] = args.r_match_min
    else:
        import cluster_sampler_reference as ref
        det = ref.realize("shell_fibonacci", meta["seed"],
                          radius_over_mass=meta["q"], nradial=meta["nradial"],
                          nangular=meta["nangular"], octahedral=True)
        meta["r_match_min"] = float(det["riso"].min())
    res = analyze(args.root / "runs" / args.case, meta)
    if res is None:
        print(f"{args.case}: no particle dumps")
        raise SystemExit(2)
    summary, rows = res
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{args.case}_summary.json").write_text(json.dumps(summary, indent=2))
    with (args.output / f"{args.case}_timeseries.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{args.case}: t={summary['t_final']:.2f} "
          f"({summary['t_final_over_P']:.3f}P) "
          f"dL_p95={summary['dL_p95_final']:.4e} "
          f"matched={summary['dLm_p95_final']:.4e} "
          f"P4 {summary['P4_initial']:.3e}->{summary['P4_final']:.3e} "
          f"align4={summary['l4_align_final']:+.3f} "
          f"R50ratio={summary['R50_ratio']:.4f} lost={summary['n_lost']}")


if __name__ == "__main__":
    main()
