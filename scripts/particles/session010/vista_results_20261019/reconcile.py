#!/usr/bin/env python3
"""Verify the supplied Vista archive, promote review products, and recheck results."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def groups(cycles, cases, valid):
    active = []
    for k in range(len(valid)):
        boundary = active and (
            cases[k] != cases[active[-1]]
            or cycles[k] - cycles[active[-1]] > 20
        )
        if not valid[k] or boundary:
            if active:
                yield np.asarray(active)
                active = []
        if valid[k]:
            active.append(k)
    if active:
        yield np.asarray(active)


def live(rows):
    return (
        np.isfinite(rows[:, 2:5]).all(axis=1)
        & np.isfinite(rows[:, 15])
        & ((rows[:, 13] > 0) | (rows[:, 16] > 0))
    )


def measure(data):
    archived = json.loads((data / "REDUCTIONS.json").read_text())
    checkpoint = json.loads((data / "RESTART_HEALTH.json").read_text())["checkpoint"]
    health = json.loads((data / "SIMULATION_HEALTH.json").read_text())
    tracks = [np.load(data / f"track_{i}.npy", allow_pickle=False) for i in (0, 1)]
    horizons = [np.load(data / f"horizon_{i}.npy", allow_pickle=False) for i in (0, 1, 2)]
    indexes = []
    for tr in tracks:
        ix = {}
        for row in tr:
            key = (int(row[-1]), int(row[0]))
            assert key not in ix, ("Ambiguous tracker identity", key)
            ix[key] = row
        indexes.append(ix)
    enclosed = []
    for h in horizons[2]:
        pair = [ix.get((int(h[-1]), int(h[0]))) for ix in indexes]
        if any(row is None for row in pair):
            continue
        pair = np.asarray(pair)
        if not live(pair).all() or np.max(np.abs(pair[:, 1] - h[1])) >= 1e-10:
            continue
        radius = np.linalg.norm(pair[:, 2:5] - h[18:21], axis=1).max()
        if radius < h[11]:
            enclosed.append((float(h[1]), int(h[0]), int(h[-1])))
    assert enclosed
    common = enclosed[0][0]
    assert abs(common - archived["first_common_enclosure_time"]) < 1e-10
    strict = json.loads((data / "common_enclosure.json").read_text())
    assert len(enclosed) == len(strict["accepted_enclosing"])
    times = np.intersect1d(tracks[0][:, 1], tracks[1][:, 1])
    pair = [tr[np.searchsorted(tr[:, 1], times)] for tr in tracks]
    a, b = pair
    assert np.array_equal(a[:, -1], b[:, -1])
    assert np.array_equal(a[:, 0], b[:, 0])
    valid = live(a) & live(b) & (times < common)
    separation = np.linalg.norm(b[:, 2:5] - a[:, 2:5], axis=1)
    vec = b[:, 2:5] - a[:, 2:5]
    arc = np.full(times.shape, np.nan)
    offset = 0.0
    segments = list(groups(a[:, 0], a[:, -1], valid))
    for g in segments:
        phase = np.unwrap(np.arctan2(vec[g, 1], vec[g, 0]))
        arc[g] = offset + phase - phase[0]
        offset = arc[g[-1]]
    degrees = float(np.degrees(offset))
    assert abs(degrees - archived["observed_arc_degrees_excluding_gaps"]) < 1e-7
    both_horizons = max(h[0, 1] for h in horizons[:2])
    after_degrees = 0.0
    for g in groups(a[:, 0], a[:, -1], valid & (times >= both_horizons)):
        phase = np.unwrap(np.arctan2(vec[g, 1], vec[g, 0]))
        after_degrees += float(np.degrees(phase[-1] - phase[0]))
    wave = np.load(data / "raw_wave.npy", allow_pickle=False)
    assert wave.shape == (archived["wave_samples"], 156)
    assert np.isfinite(wave).all() and (np.diff(wave[:, 0]) > 0).all()
    # l=2, m=-2..2 are the first five modes; (2,2) has zero-based mode index 4.
    amplitude = np.hypot(wave[:, 5], wave[:, 82])
    peak = int(np.argmax(amplitude))
    removals = np.load(data / "removals.npy", allow_pickle=False, mmap_mode="r")
    assert removals.shape[1] == 6 and removals.shape[0] == archived["removal_events"]
    tags = np.asarray(removals[:, 4], dtype=np.int64)
    assert np.array_equal(tags, removals[:, 4])
    assert np.unique(tags).size == tags.size and tags.min() >= 0 and tags.max() < 5_000_000
    for start in range(0, len(removals), 500_000):
        assert np.isfinite(removals[start:start + 500_000]).all()
    matter = np.load(data / "matter.npy", allow_pickle=False)
    initial_counts = np.array([3_000_000, 1_000_000, 1_000_000])
    final_components = []
    for i in range(3):
        rows = matter[matter[:, 1] == i]
        assert np.array_equal(rows[:, 2] + rows[:, 8], np.full(len(rows), initial_counts[i]))
        assert np.allclose(rows[:, 3] + rows[:, 9], rows[0, 3], rtol=1e-6, atol=1e-10)
        final_components.append(dict(
            component=i, time=float(rows[-1, 0]), alive=int(rows[-1, 2]),
            removed=int(rows[-1, 8]), alive_rest_mass=float(rows[-1, 3]),
        ))
    assert sum(x["alive"] for x in final_components) == checkpoint["particles"]
    assert sum(x["removed"] for x in final_components) == len(removals)
    assert checkpoint["particles"] + len(removals) == 5_000_000
    amr = json.loads((data / "AMR_HEALTH.json").read_text())
    assert sum(amr["level_counts"].values()) == checkpoint["blocks"]
    assert amr["wave_floor_satisfied"] and amr["finest_dx"] == 1 / 256
    saved_h = health["constraint_norms"]
    metrics = dict(
        native_time=checkpoint["time"], cycle=checkpoint["cycle"],
        source_commit="386870f88f60e25e1a8fac76fe04a071b87a01cf",
        first_individual_horizons=[float(h[0, 1]) for h in horizons[:2]],
        first_raw_accepted_common=float(horizons[2][0, 1]),
        first_strict_common_enclosure=common, strict_simultaneous_enclosure_samples=len(enclosed),
        last_binary_time=float(times[valid][-1]),
        initial_coordinate_separation=float(separation[valid][0]),
        last_binary_coordinate_separation=float(separation[valid][-1]),
        observed_rotation_degrees_excluding_gaps=degrees, observed_turns=degrees / 360,
        post_individual_horizon_observed_turns=after_degrees / 360,
        valid_orbit_intervals=len(segments),
        last_accepted_common=dict(time=float(horizons[2][-1, 1]),
                                  mass=float(horizons[2][-1, 2]),
                                  coordinate_spin=float(horizons[2][-1, 13])),
        final_candidate_hrel=float(health["horizon_quality"]["2"]["latest_candidate"]["hrel"]),
        configured_hrel_limit=health["horizon_quality"]["2"]["configured_hrel_limit"],
        particles_alive=checkpoint["particles"], low_lapse_removals=len(removals),
        removed_fraction=len(removals) / 5_000_000, final_components=final_components,
        wave_samples=len(wave), wave_end=float(wave[-1, 0]),
        max_wave_spacing=float(np.diff(wave[:, 0]).max()),
        raw_22_peak_time=float(wave[peak, 0]), raw_22_peak_amplitude=float(amplitude[peak]),
        post_peak_coverage=float(wave[-1, 0] - wave[peak, 0]),
        leaf_blocks=checkpoint["blocks"], finest_dx=amr["finest_dx"],
        wave_floor_preserved=True, gpu_peak_fraction=health["gpu_peak_fraction"],
        host_peak_fraction=health["host_peak_fraction"],
        H_RMS=saved_h["last_H_RMS"], M_RMS=saved_h["last_M_RMS"],
        constraint_mask=saved_h["mask"],
        local_rechecks=["common enclosure on exact case/cycle/time",
                        "orbit arc with original gap/case rules", "raw mode (2,2) peak",
                        "all removal tags unique and finite", "component census/rest mass closure"],
        native_checkpoint_audit="Preserved October 18 Vista audit; bulk checkpoint absent locally",
    )
    return metrics, (times, separation, arc, valid, segments, a, b, horizons, wave, amplitude)


def overview(path, metrics, arrays):
    times, separation, arc, valid, segments, a, b, horizons, wave, amplitude = arrays
    common = metrics["first_strict_common_enclosure"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))
    colors = ("#1768ac", "#dc661b", "#333333")
    for g in segments:
        for tr, color in zip((a, b), colors):
            axes[0, 0].plot(tr[g, 2], tr[g, 3], color=color, lw=1)
        axes[0, 1].plot(times[g], separation[g], color=colors[0], lw=1)
    axes[0, 0].set(xlabel="x / Mref", ylabel="y / Mref", aspect="equal",
                   title=f"Inward motion: {metrics['observed_turns']:.2f} observed turns")
    axes[0, 0].plot([], [], color=colors[0], label="Original left")
    axes[0, 0].plot([], [], color=colors[1], label="Original right")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].axvline(common, ls="--", color="black", lw=1, label="First strict common enclosure")
    axes[0, 1].set(xlabel="t / Mref", ylabel="Coordinate separation / Mref",
                   title="Binary curves end at common enclosure")
    axes[0, 1].legend(fontsize=8)
    for i, h in enumerate(horizons):
        keep = h[:, 1] < common if i < 2 else h[:, 1] >= common
        h = h[keep]
        # Separate case boundaries and unsampled cycles; never bridge restart gaps.
        for g in groups(h[:, 0], h[:, -1], np.ones(len(h), dtype=bool)):
            axes[1, 0].plot(h[g, 1], h[g, 2], color=colors[i], lw=1)
        axes[1, 0].plot([], [], color=colors[i], label=("Left", "Right", "Common")[i])
    last = metrics["last_accepted_common"]["time"]
    axes[1, 0].axvspan(last, metrics["native_time"], color="0.92",
                       label="No accepted late common horizon")
    axes[1, 0].set(xlabel="t / Mref", ylabel="Accepted horizon mass / Mref",
                   title="Late remnant properties unresolved", xlim=(0, metrics["native_time"]))
    axes[1, 0].legend(fontsize=7)
    wave_breaks = np.r_[0, np.where((np.diff(wave[:, -1]) != 0)
                                   | (np.diff(wave[:, 0]) > .0375))[0] + 1, len(wave)]
    for start, end in zip(wave_breaks[:-1], wave_breaks[1:]):
        axes[1, 1].plot(wave[start:end, 0], amplitude[start:end], color=colors[0], lw=1)
    axes[1, 1].axvline(metrics["raw_22_peak_time"], color="black", ls="--", lw=1)
    axes[1, 1].set(xlabel="t / Mref", ylabel="Absolute raw rPsi4 (2,2) at R40",
                   title=f"Main raw burst peaks at t={metrics['raw_22_peak_time']:.2f}")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle("Session010: collapse, short plunge and common horizon", fontsize=14)
    fig.text(.5, .018, "Coordinate diagnostics; gaps excluded. Raw rPsi4 is not strain. "
             "No converged inspiral or validated ringdown claim.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .96))
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--replot-to", type=Path,
                    help="Recheck reduced data and render to a new output prefix; preserve the release")
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Allocated compute is required for hashing, reductions and plotting")
    s = args.session.resolve()
    source = s / "session010_summary_20261019"
    workspace = source / "workspace"
    if args.replot_to is not None:
        target = args.replot_to.resolve()
        outputs = [target.with_suffix(ext) for ext in (".png", ".pdf", ".json")]
        if any(path.exists() for path in outputs):
            raise RuntimeError("Replot destination exists; choose a new output prefix")
        metrics, arrays = measure(workspace / "runtime_s10/analysis/through_segment002_20261018")
        target.parent.mkdir(parents=True, exist_ok=True)
        overview(target, metrics, arrays)
        atomic(target.with_suffix(".json"), metrics)
        print("REDUCED-DATA REPLOT PASSED", target)
        return
    evidence = s / "evidence/vista_reconciliation_20261019"
    evidence.mkdir(parents=True, exist_ok=True)
    receipt = evidence / "COMPLETE.json"
    if receipt.exists():
        raise RuntimeError("Completed reconciliation exists; preserve it and choose a new dated release")
    result = subprocess.run([sys.executable, str(source / "tools/verify_summary.py"), str(source)],
                            text=True, capture_output=True)
    (evidence / "source_verification.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    records = []

    def copy(src, dest):
        src, dest = Path(src), Path(dest)
        assert src.is_file() and not src.is_symlink()
        src_hash = digest(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert digest(dest) == src_hash, ("Refuse to replace a different file", str(dest))
        else:
            shutil.copy2(src, dest)
        assert digest(dest) == src_hash
        records.append(dict(source=str(src.relative_to(s)), destination=str(dest.relative_to(s)),
                            bytes=dest.stat().st_size, sha256=src_hash))

    for name in ("README.md", "REPORT_AGENT.md", "REPORT_Jeans10.md", "CLOSEOUT.md"):
        copy(s / name, s / "history/perseus_preparation_20261013" / name)
    current = workspace / "runtime_s10/analysis/through_segment002_20261018"
    historical = workspace / "runtime_s10/analysis/first_production_20261015"
    for label, release in (("through_segment002_20261018", current),
                           ("first_production_20261015", historical)):
        prefix = s if release == current else s / "archive/vista_first_production_20261015"
        for f in release.iterdir():
            if not f.is_file():
                continue
            if f.suffix in (".png", ".pdf"):
                copy(f, prefix / "plots" / label / f.name)
            elif f.suffix == ".mp4":
                copy(f, prefix / "movies" / label / f.name)
            elif release == current and f.suffix in (".csv", ".npy", ".npz"):
                copy(f, s / "reduced_data/vista_20261018" / f.name)
            elif release == current and f.suffix in (".json", ".md"):
                copy(f, evidence / "vista_analysis" / f.name)
        if release == current:
            for f in (release / "qa").glob("*decoded.png"):
                copy(f, evidence / "movie_qa" / f.name)
    for f in (source / "reports").iterdir():
        copy(f, s / "reports/vista_original_20261019" / f.name)
    for f in (source / "inventory").iterdir():
        if f.is_file():
            copy(f, evidence / "source_inventory" / f.name)
    for name in ("BUILD_VERIFIED.json",):
        copy(workspace / "runtime_s10/build" / name, evidence / name)
    copy(workspace / "runtime_s10/SESSION_LOG.md", evidence / "VISTA_SESSION_LOG.md")
    for name in ("analysis_s10", "analysis_s10_update_20261018",
                 "launch_s10", "launch_s10_recovery"):
        for f in (workspace / name).rglob("*"):
            if f.is_file():
                copy(f, s / "scripts/vista_original_20261019" / name / f.relative_to(workspace / name))
    for run in ("20261013T200009Z", "20261014_gh_retry_01", "20261016_gh_resume_01"):
        r = workspace / "runtime_s10/runs" / run
        for name in ("config.json", "production.athinput", "compact.athinput"):
            copy(r / name, s / "config/vista_original_20261019" / run / name)
    copy(workspace / "runtime_s10/runs/20261016_gh_resume_01/production.athinput",
         s / "inputs/vista_production/production.athinput")
    metrics, arrays = measure(current)
    atomic(evidence / "SCIENTIFIC_RECHECK.json", metrics)
    overview(s / "plots/through_segment002_20261018/00_scientific_summary", metrics, arrays)
    # Decode the promoted copies with a native x86 executable, never the archived ARM executable.
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    decoded = []
    for row in records:
        if not row["destination"].endswith(".mp4"):
            continue
        target = s / row["destination"]
        log = evidence / ("decode_" + target.parent.name + "_" + target.stem + ".log")
        with log.open("w") as stream:
            p = subprocess.run([ffmpeg, "-v", "error", "-i", str(target),
                                "-map", "0:v:0", "-f", "null", "-"],
                               stdout=stream, stderr=subprocess.STDOUT)
        if p.returncode:
            raise RuntimeError("Movie decode failed: " + str(target))
        decoded.append(row["destination"])
    atomic(evidence / "PROMOTION_MANIFEST.json", records)
    atomic(receipt, dict(passed=True, job=os.environ["SLURM_JOB_ID"],
                         host=os.uname().nodename, python=sys.version,
                         source_summary=str(source), source_sha_manifest=digest(source / "SHA256SUMS.txt"),
                         verified_summary_stdout=result.stdout.strip(),
                         copied_files=len(records), copied_bytes=sum(r["bytes"] for r in records),
                         all_promoted_hashes_match=True, decoded_movies=decoded,
                         original_summary_modified=False, production_launched=False,
                         raw_transfer_performed=False,
                         finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                         finished_local=dt.datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()))
    print(json.dumps(metrics, indent=2))
    print("VISTA RECONCILIATION PASSED", receipt)


if __name__ == "__main__":
    main()
