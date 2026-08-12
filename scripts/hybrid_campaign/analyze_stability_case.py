#!/usr/bin/env python3
"""Stability analysis for one long-duration hybrid-sampler run.

Built around the Lagrangian radial cohorts that the stratified_antithetic
sampler provides natively: the immutable tag group ir = tag/(4*nangular) is a
contiguous equal-rest-mass band of the initial radial CDF (stratum index ==
pair index == tag/2), so the solver's per-group shell ledger
(*.lagrangian_shells.csv, cadence 1M) already tracks 128 fixed-membership
cohorts by particle ID.  No re-binning is ever done after t=0.

Produces per run:
  * cohort_trajectories.csv   median areal/isotropic radius per cohort vs time
  * stability_timeseries.csv  mass radii, lapse, density, constraints, J, ...
  * events.json               automatically detected regime changes
  * summary.json              scalar endpoint + fit summary

Event definitions (marked, never silently classified):
  secular_contraction   R50 smoothed over a 1P window decreases monotonically
                        for >= 1P AND the total drop exceeds 3x the early
                        oscillation amplitude (measured over [0.5P, 2P]).
  unbounded_oscillation the peak-to-peak envelope of R50 in a 1P window
                        exceeds 2x its early value, sustained >= 0.5P.
  cohort_departure      an outer cohort's |median radius - initial| exceeds
                        5x the early scatter of that cohort AND a positive
                        exponential rate fits with R^2 > 0.9 over >= 0.5P.
  boundary_contact      sqrt(2), v=1.7, and v=1 conservative estimates from
                        the input manifest (marked on every plot; nothing
                        after the sqrt(2) mark is used for physics claims).

The ST orbit-class label per cohort uses x = C u^2 at the cohort's initial
areal radius: unstable if x > 1/6, 'near' if x > 0.9/6.
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


def read_csv_rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, skipinitialspace=True))


def read_shell_ledger(path, ncohort):
    """time -> arrays over cohorts of (median_riso, median_rareal, mean_vrad, count)."""
    times, blocks = [], {}
    with open(path) as fh:
        header = None
        for line in fh:
            if line.startswith("#"):
                continue
            if header is None:
                header = line.strip().split(",")
                idx = {k: i for i, k in enumerate(header)}
                continue
            p = line.strip().split(",")
            if len(p) < len(header):
                continue
            t = float(p[idx["time"]])
            sh = int(p[idx["shell"]])
            blocks.setdefault(t, {})[sh] = (
                float(p[idx["median_riso"]]), float(p[idx["median_rareal"]]),
                float(p[idx["mean_vrad"]]), int(p[idx["count"]]))
    times = sorted(blocks)
    riso = np.full((len(times), ncohort), np.nan)
    rareal = np.full((len(times), ncohort), np.nan)
    vrad = np.full((len(times), ncohort), np.nan)
    count = np.zeros((len(times), ncohort), int)
    for i, t in enumerate(times):
        for sh, (a, b, c, n) in blocks[t].items():
            if 0 <= sh < ncohort:
                riso[i, sh], rareal[i, sh], vrad[i, sh], count[i, sh] = a, b, c, n
    return np.array(times), riso, rareal, vrad, count


def read_hst(path):
    labels, rows = None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            if "=" in line and "[" in line:
                labels = re.findall(r"\[\d+\]=(\S+)", line)
            continue
        if line.strip():
            rows.append([float(v) for v in line.split()])
    if not rows or not labels:
        return None
    arr = np.array(rows)
    return {lab: arr[:, i] for i, lab in enumerate(labels[:arr.shape[1]])}


def smooth(y, t, window):
    """centered moving average over a time window (non-uniform safe)."""
    out = np.full_like(y, np.nan, dtype=float)
    for i, ti in enumerate(t):
        sel = (t >= ti - window / 2) & (t <= ti + window / 2)
        if sel.sum() >= 3:
            out[i] = np.nanmean(y[sel])
    return out


def envelope(y, t, window):
    """peak-to-peak amplitude of (y - smooth) within a sliding window."""
    base = smooth(y, t, window)
    r = y - base
    out = np.full_like(y, np.nan, dtype=float)
    for i, ti in enumerate(t):
        sel = (t >= ti - window / 2) & (t <= ti + window / 2)
        if sel.sum() >= 3:
            out[i] = np.nanmax(r[sel]) - np.nanmin(r[sel])
    return out


def detect_events(t, R50, cohort_t, cohort_r, cohort_meta, period, tc_marks):
    events = {"boundary_contact": tc_marks}
    good = np.isfinite(R50)
    t, R50 = t[good], R50[good]
    if len(t) < 10:
        return events

    early = (t >= 0.5 * period) & (t <= 2.0 * period)
    base_env = envelope(R50, t, period)
    early_env = np.nanmedian(base_env[early]) if early.sum() > 3 else np.nan
    events["early_oscillation_amplitude_R50"] = float(early_env)

    # secular contraction
    sm = smooth(R50, t, period)
    dsm = np.gradient(sm, t)
    run_start = None
    events["secular_contraction"] = None
    for i in range(len(t)):
        if np.isfinite(dsm[i]) and dsm[i] < 0:
            if run_start is None:
                run_start = i
            if (t[i] - t[run_start] >= period
                    and np.isfinite(early_env)
                    and sm[run_start] - sm[i] > 3.0 * early_env):
                events["secular_contraction"] = {
                    "onset_t": float(t[run_start]),
                    "onset_t_over_P": float(t[run_start] / period),
                    "drop_at_detection": float(sm[run_start] - sm[i]),
                }
                break
        else:
            run_start = None

    # unbounded oscillation
    events["unbounded_oscillation"] = None
    if np.isfinite(early_env) and early_env > 0:
        exceed = base_env > 2.0 * early_env
        run = None
        for i in range(len(t)):
            if exceed[i] and np.isfinite(base_env[i]):
                run = i if run is None else run
                if t[i] - t[run] >= 0.5 * period:
                    events["unbounded_oscillation"] = {
                        "onset_t": float(t[run]),
                        "onset_t_over_P": float(t[run] / period)}
                    break
            else:
                run = None

    # cohort exponential departure (outer 16 cohorts + any ST-unstable ones)
    departures = []
    ncoh = cohort_r.shape[1]
    watch = sorted(set(range(ncoh - 16, ncoh)) |
                   {c["cohort"] for c in cohort_meta if c["st_class"] != "stable"})
    for ci in watch:
        r = cohort_r[:, ci]
        ok = np.isfinite(r)
        if ok.sum() < 10:
            continue
        tt, rr = cohort_t[ok], r[ok]
        r0 = np.nanmedian(rr[tt <= 0.25 * period])
        disp = rr - r0
        early_sc = np.nanstd(disp[(tt >= 0.5 * period) & (tt <= 2.0 * period)])
        if not np.isfinite(early_sc) or early_sc == 0:
            continue
        big = np.abs(disp) > 5.0 * early_sc
        onset = None
        for i in range(len(tt)):
            if big[i]:
                onset = i if onset is None else onset
                if tt[i] - tt[onset] >= 0.5 * period:
                    seg = (tt >= tt[onset]) & (tt <= tt[i])
                    x, y = tt[seg], np.abs(disp[seg])
                    pos = y > 0
                    rate, r2 = float("nan"), float("nan")
                    if pos.sum() >= 5:
                        cfit = np.polyfit(x[pos], np.log(y[pos]), 1)
                        pred = np.polyval(cfit, x[pos])
                        ss = 1 - (np.sum((np.log(y[pos]) - pred) ** 2)
                                  / max(np.sum((np.log(y[pos])
                                                - np.log(y[pos]).mean()) ** 2), 1e-30))
                        rate, r2 = float(cfit[0]), float(ss)
                    if np.isfinite(rate) and rate > 0 and r2 > 0.9:
                        departures.append({
                            "cohort": int(ci),
                            "st_class": cohort_meta[ci]["st_class"],
                            "u_initial": cohort_meta[ci]["u_initial"],
                            "onset_t": float(tt[onset]),
                            "onset_t_over_P": float(tt[onset] / period),
                            "growth_rate_per_M": rate,
                            "efold_per_P": rate * period,
                            "fit_r2": r2,
                        })
                    break
            else:
                onset = None
    events["cohort_departures"] = departures
    return events


def psi4_jz_flux(waveform_dir, radius_tag, period):
    """Cumulative GW angular-momentum estimate from the rpsi4 mode files.

    dJz/dt = (r^2/16pi) sum_lm m * Im[ h_lm * conj(dh_lm/dt) ], with
    dh/dt = int psi4 dt and h = int int psi4.  Fixed-frequency-free plain time
    integration with linear detrending of each integral; adequate as a
    diagnostic of the ORDER of GW J loss, not a precision budget.
    """
    real_f = Path(waveform_dir) / f"rpsi4_real_{radius_tag}.txt"
    imag_f = Path(waveform_dir) / f"rpsi4_imag_{radius_tag}.txt"
    if not (real_f.exists() and imag_f.exists()):
        return None
    re_dat = np.loadtxt(real_f, comments="#")
    im_dat = np.loadtxt(imag_f, comments="#")
    if re_dat.ndim != 2 or re_dat.shape[0] < 10:
        return None
    with open(real_f) as fh:
        head = [next(fh) for _ in range(3)]
    cols = None
    for h in head:
        if "l=" in h or "2 2" in h:
            cols = h
    t = re_dat[:, 0]
    n = min(len(re_dat), len(im_dat))
    t = t[:n]
    # columns after t: modes in AthenaK order (l from 2..lmax, m from -l..l)
    nmodes = re_dat.shape[1] - 1
    lmax = int(math.isqrt(nmodes + 4)) - 1  # sum_{l=2}^{L}(2l+1) = (L+1)^2-4
    mlist = []
    for l in range(2, lmax + 1):
        for m in range(-l, l + 1):
            mlist.append(m)
    if len(mlist) != nmodes:
        mlist = mlist[:nmodes] + [0] * max(0, nmodes - len(mlist))
    psi = re_dat[:n, 1:] + 1j * im_dat[:n, 1:]
    dt = np.gradient(t)
    hdot = np.cumsum(psi * dt[:, None], axis=0)
    hdot -= hdot.mean(axis=0)
    h = np.cumsum(hdot * dt[:, None], axis=0)
    h -= h.mean(axis=0)
    r_ext = float(radius_tag)  # tag like 0080 -> 80
    flux = (r_ext ** 2 / (16 * math.pi)) * np.sum(
        np.array(mlist)[None, :] * np.imag(h * np.conj(hdot)), axis=1)
    dJz = np.cumsum(flux * dt)
    return {"t": t.tolist()[::4], "dJz_cum": dJz.tolist()[::4],
            "final_dJz": float(dJz[-1]), "lmax": lmax, "r_ext": r_ext}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--manifest", default="inputs_long/long_manifest.json")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ncohort", type=int, default=128)
    args = ap.parse_args()

    manifest = json.loads((args.root / args.manifest).read_text())
    meta = next(c for c in manifest if c["name"] == args.case)
    period = float(meta["period"])
    C = 1.0 / float(meta["q"])
    run = args.root / "runs" / args.case
    args.output.mkdir(parents=True, exist_ok=True)

    # cohort metadata from the analytic stratified construction: cohort ci
    # spans strata [ci, ci+1)/ncohort of the rest-mass CDF
    import cluster_sampler_reference as ref
    prof = ref.Profile(C, 1.0, meta["nradial"])
    cohort_meta = []
    for ci in range(args.ncohort):
        p_mid = (ci + 0.5) / args.ncohort
        u = float(prof.invert(p_mid))
        x = C * u * u
        st = "unstable" if x > 1.0 / 6.0 else ("near" if x > 0.9 / 6.0 else "stable")
        cohort_meta.append({"cohort": ci, "u_initial": u, "x_initial": x,
                            "st_class": st})

    ct, riso, rareal, vrad, count = read_shell_ledger(
        run / f"{args.case}.lagrangian_shells.csv", args.ncohort)
    mass_rows = read_csv_rows(run / f"{args.case}.mass_radii.csv")
    hst = read_hst(run / f"{args.case}.user.hst")
    z4c = read_hst(run / f"{args.case}.z4c.user.hst")

    tm = np.array([float(r["time"]) for r in mass_rows])
    RQ = {k: np.array([float(r[k]) for r in mass_rows])
          for k in ("R10", "R25", "R50", "R75", "R90")}

    tc_marks = {}
    for name, key in (("sqrt2", "boundary_contact_sqrt2_M"),
                      ("v1.7", "boundary_contact_v1p7_M"),
                      ("light", "boundary_contact_light_M")):
        val = meta.get(key)
        if val is not None and math.isfinite(val):
            tc_marks[name] = {"t": val, "t_over_P": val / period}
    if "sqrt2" not in tc_marks:
        # fall back to the small-domain convention: half-width 176M, sqrt(2)
        tc = (176.0 - 5.05) / math.sqrt(2.0)
        tc_marks["sqrt2"] = {"t": tc, "t_over_P": tc / period}
    events = detect_events(tm, RQ["R50"], ct, rareal, cohort_meta, period, tc_marks)

    # per-cohort |L| drift and endpoint displacement
    cohort_rows = []
    for ci in range(args.ncohort):
        ok = np.isfinite(rareal[:, ci])
        if ok.sum() < 2:
            continue
        r0 = rareal[ok, ci][0]
        cohort_rows.append({
            **cohort_meta[ci],
            "r_areal_initial": float(r0),
            "r_areal_final": float(rareal[ok, ci][-1]),
            "displacement_final": float(rareal[ok, ci][-1] - r0),
            "rel_displacement_final": float((rareal[ok, ci][-1] - r0) / r0),
            "min_count": int(count[:, ci][count[:, ci] > 0].min())
            if (count[:, ci] > 0).any() else 0,
        })

    gw = psi4_jz_flux(run / "waveforms", "0080", period)

    ts_path = args.output / f"{args.case}_stability_timeseries.csv"
    with ts_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "t_over_P", "R10", "R25", "R50", "R75", "R90"])
        for i, t in enumerate(tm):
            w.writerow([t, t / period] + [RQ[k][i] for k in
                                          ("R10", "R25", "R50", "R75", "R90")])
    with (args.output / f"{args.case}_cohorts.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "t_over_P"] +
                   [f"c{ci:03d}" for ci in range(args.ncohort)])
        for i, t in enumerate(ct):
            w.writerow([t, t / period] + list(rareal[i]))

    summary = {
        "case": args.case, "model": meta["model"], "seed": meta["seed"],
        "period": period, "t_final": float(tm[-1]) if len(tm) else float("nan"),
        "t_final_over_P": float(tm[-1] / period) if len(tm) else float("nan"),
        "events": events, "cohorts": cohort_rows,
        "gw_dJz_cum_final": None if gw is None else gw["final_dJz"],
        "boundary_marks": tc_marks,
    }
    if hst is not None:
        for k, lab in (("alpha_min", "alpha_min"), ("alpha_ctr", "alpha_ctr"),
                       ("rho_max", "rho_max"), ("N_alive", "N_alive"),
                       ("geo_fbacks", "geo_fbacks"), ("N_nonfinit", "N_nonfinite")):
            if k in hst:
                summary[f"{lab}_final"] = float(hst[k][-1])
                summary[f"{lab}_min"] = float(np.nanmin(hst[k]))
    if z4c is not None and "H-norm2" in z4c:
        vol = z4c.get("Volume")
        valid = np.isfinite(vol) & (vol > 0) if vol is not None else slice(None)
        summary["ham_final"] = float(z4c["H-norm2"][valid][-1])
        summary["ham_max"] = float(np.nanmax(z4c["H-norm2"][valid]))
        summary["mom_final"] = float(z4c["M-norm2"][valid][-1])
    if gw is not None:
        (args.output / f"{args.case}_gw_jz.json").write_text(json.dumps(gw))

    (args.output / f"{args.case}_events.json").write_text(
        json.dumps(events, indent=2))
    (args.output / f"{args.case}_summary.json").write_text(
        json.dumps(summary, indent=2))
    dep = events.get("cohort_departures") or []
    print(f"{args.case}: t={summary['t_final_over_P']:.2f}P "
          f"secular={events.get('secular_contraction') is not None} "
          f"unbounded={events.get('unbounded_oscillation') is not None} "
          f"departures={len(dep)} "
          f"alpha_min={summary.get('alpha_min_min', float('nan')):.4f}")


if __name__ == "__main__":
    main()
