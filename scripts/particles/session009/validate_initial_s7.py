#!/usr/bin/env python3
"""Validate exact Session 007 boosted/unboosted initialization and written particles."""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
import numpy as np

LEDGER = re.compile(
    r"ledger\s+(envelope|clump)\s+(\d+):\s+N=(\d+)\s+"
    r"rest_mass=([-+0-9.eE]+)\s+sum\(mW\)=([-+0-9.eE]+).*?"
    r"P_cov=\(([-+0-9.eE]+),([-+0-9.eE]+),([-+0-9.eE]+)\).*?"
    r"P_hat=\(([-+0-9.eE]+),([-+0-9.eE]+),([-+0-9.eE]+)\).*?"
    r"J_origin_cov=\(([-+0-9.eE]+),([-+0-9.eE]+),([-+0-9.eE]+)\).*?"
    r"J_internal_cov=\(([-+0-9.eE]+),([-+0-9.eE]+),([-+0-9.eE]+)\)"
)
EXPECTED_COUNTS = (3_000_000, 1_000_000, 1_000_000)
CENTERS = ((0., 0., 0.), (-3., 0., 0.), (3., 0., 0.))
BOOSTS = (0., -0.133215, 0.133215)
CLUMP_MASS, CLUMP_SIGMA = 0.12, 0.70

def parse_ledger(path: Path) -> list[dict]:
    found = []
    for match in LEDGER.finditer(path.read_text(errors="replace")):
        g = match.groups()
        values = [float(v) for v in g[3:]]
        found.append({
            "name": g[0], "component": int(g[1]), "count": int(g[2]),
            "rest_mass": values[0], "sum_mW": values[1],
            "P_cov": values[2:5], "P_hat": values[5:8],
            "J_origin_cov": values[8:11], "J_internal_cov": values[11:14],
        })
    if [row["component"] for row in found] != [0, 1, 2]:
        raise ValueError(f"expected envelope and two clump ledger rows in {path}")
    return found

def area_stats(raw: dict, variable: str, window=None) -> dict:
    """Area-weighted norms on the recorded z=0 leaf-block slice."""
    geometry = np.asarray(raw["mb_geometry"], dtype=float)
    values = np.asarray(raw["mb_data"][variable], dtype=float)[:, 0]
    nx, ny = int(raw["nx1_out_mb"]), int(raw["nx2_out_mb"])
    area = square = absolute = 0.
    maximum = 0.
    for block, geom in zip(values, geometry):
        dx, dy = (geom[1]-geom[0])/nx, (geom[3]-geom[2])/ny
        x = geom[0] + (np.arange(nx)+.5)*dx
        y = geom[2] + (np.arange(ny)+.5)*dy
        selected = np.ones((ny, nx), dtype=bool)
        if window:
            xmin, xmax, ymin, ymax = window
            selected = ((x[None, :] >= xmin) & (x[None, :] < xmax)
                        & (y[:, None] >= ymin) & (y[:, None] < ymax))
        data = block[selected]
        area += data.size*dx*dy
        square += float(np.sum(data*data))*dx*dy
        absolute += float(np.sum(np.abs(data)))*dx*dy
        if data.size:
            maximum = max(maximum, float(np.max(np.abs(data))))
    if area == 0.:
        raise ValueError(f"empty mask for {variable}: {window}")
    return {"area": area, "rms": float(np.sqrt(square/area)),
            "mean_abs": absolute/area, "max_abs": maximum}

def volume_stats(raw: dict, variable: str, window=None) -> dict:
    """Volume-weighted norms on the full recorded 3-D leaf-block hierarchy."""
    geometry = np.asarray(raw["mb_geometry"], dtype=float)
    values = np.asarray(raw["mb_data"][variable], dtype=float)
    nx, ny, nz = (int(raw["nx1_out_mb"]), int(raw["nx2_out_mb"]),
                  int(raw["nx3_out_mb"]))
    volume = square = absolute = 0.
    maximum = 0.
    for block, geom in zip(values, geometry):
        dx, dy, dz = ((geom[1]-geom[0])/nx, (geom[3]-geom[2])/ny,
                      (geom[5]-geom[4])/nz)
        x = geom[0] + (np.arange(nx)+.5)*dx
        y = geom[2] + (np.arange(ny)+.5)*dy
        z = geom[4] + (np.arange(nz)+.5)*dz
        selected = np.ones((nz, ny, nx), dtype=bool)
        if window:
            xmin, xmax, ymin, ymax, zmin, zmax = window
            selected = ((x[None, None, :] >= xmin) & (x[None, None, :] < xmax)
                        & (y[None, :, None] >= ymin) & (y[None, :, None] < ymax)
                        & (z[:, None, None] >= zmin) & (z[:, None, None] < zmax))
        data = block[selected]
        volume += data.size*dx*dy*dz
        square += float(np.sum(data*data))*dx*dy*dz
        absolute += float(np.sum(np.abs(data)))*dx*dy*dz
        if data.size:
            maximum = max(maximum, float(np.max(np.abs(data))))
    if volume == 0.:
        raise ValueError(f"empty 3-D mask for {variable}: {window}")
    return {"volume": volume, "rms": float(np.sqrt(square/volume)),
            "mean_abs": absolute/volume, "max_abs": maximum}

def first(run: Path, pattern: str) -> Path:
    matches = sorted(run.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"{pattern} under {run}")
    return matches[0]

def latest(run: Path, pattern: str) -> Path:
    matches = sorted(run.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"{pattern} under {run}")
    return matches[-1]

def vtk_block(data: bytes, marker: bytes, count: int) -> np.ndarray:
    start = data.find(marker)
    if start < 0:
        raise ValueError(f"missing VTK marker {marker!r}")
    start = data.find(b"\n", start+len(marker))+1
    if start <= 0:
        raise ValueError(f"unterminated VTK marker {marker!r}")
    return np.frombuffer(data, dtype=">f4", count=count, offset=start)

def read_particles(path: Path) -> dict:
    data = path.read_bytes()
    point = re.search(rb"POINTS\s+(\d+)\s+float", data)
    tm = re.search(rb"time=\s*([-+0-9.eE]+)", data)
    if point is None or tm is None:
        raise ValueError(f"incomplete particle output {path}")
    count = int(point.group(1))
    return {
        "path": str(path), "time": float(tm.group(1)), "count": count,
        "position": vtk_block(data, point.group(0), 3*count).reshape(count, 3),
        "momentum": vtk_block(data, b"VECTORS prtcl_vel float", 3*count).reshape(count, 3),
        "tag_float": vtk_block(data, b"SCALARS ptag float\nLOOKUP_TABLE default", count),
        "energy": vtk_block(data, b"SCALARS prtcl_energy float\nLOOKUP_TABLE default", count),
        "mass": vtk_block(data, b"SCALARS prtcl_mass float\nLOOKUP_TABLE default", count),
    }

def erf_nonnegative(x: np.ndarray) -> np.ndarray:
    """Vectorized erf approximation; maximum absolute error is about 1.5e-7."""
    t = 1./(1.+0.3275911*x)
    poly = (((((1.061405429*t-1.453152027)*t)+1.421413741)*t
             -0.284496736)*t+0.254829592)*t
    return 1.-poly*np.exp(-x*x)

def clump_term(radius: np.ndarray) -> np.ndarray:
    out = np.empty_like(radius, dtype=np.float64)
    nz = radius > 1.e-12
    out[nz] = (CLUMP_MASS/(2.*radius[nz])
               * erf_nonnegative(radius[nz]/(np.sqrt(2.)*CLUMP_SIGMA)))
    out[~nz] = CLUMP_MASS/(CLUMP_SIGMA*np.sqrt(2.*np.pi))
    return out

def full_psi(position: np.ndarray, profile: np.ndarray) -> np.ndarray:
    pos = np.asarray(position, dtype=np.float64)
    radius = np.linalg.norm(pos, axis=1)
    psi = np.interp(radius, profile[:, 0], profile[:, 2],
                    left=profile[0, 2], right=profile[-1, 2])
    for center in CENTERS[1:]:
        psi += clump_term(np.linalg.norm(pos-np.asarray(center), axis=1))
    return psi

def vector3(values: np.ndarray) -> list[float]:
    return [float(x) for x in values]

def written_particle_stats(run: Path) -> tuple[dict, list[dict]]:
    particles = read_particles(first(run, "*.part.vtk"))
    profile_path = first(run, "gi_profile_M076_two_clump_s7.txt")
    profile = np.loadtxt(profile_path)
    xyz, mom = particles["position"], particles["momentum"]
    tag_float, mass = particles["tag_float"], particles["mass"]
    tags = np.rint(tag_float).astype(np.int64)
    finite = bool(np.isfinite(xyz).all() and np.isfinite(mom).all()
                  and np.isfinite(mass).all() and np.isfinite(particles["energy"]).all()
                  and np.isfinite(tag_float).all())
    integer_tags = bool(np.array_equal(tag_float, tags.astype(tag_float.dtype)))
    ordered = np.sort(tags)
    exact_tags = bool(ordered.size == 5_000_000
                      and np.array_equal(ordered, np.arange(5_000_000)))
    masks = (tags < 3_000_000,
             (tags >= 3_000_000) & (tags < 4_000_000),
             (tags >= 4_000_000) & (tags < 5_000_000))
    component_rows, ledger_rows = [], []
    for component, (mask, center, beta) in enumerate(zip(masks, CENTERS, BOOSTS)):
        pos = np.asarray(xyz[mask], dtype=np.float64)
        cov = np.asarray(mom[mask], dtype=np.float64)
        mu = np.asarray(mass[mask], dtype=np.float64)
        psi = full_psi(pos, profile)
        qhat = cov/(psi[:, None]*psi[:, None])
        what = np.sqrt(1.+np.sum(qhat*qhat, axis=1))
        center_array = np.asarray(center)
        ledger_rows.append({
            "component": component, "count": int(mask.sum()),
            "rest_mass": float(np.sum(mu, dtype=np.float64)),
            "sum_mW": float(np.sum(mu*what, dtype=np.float64)),
            "P_cov": vector3(np.sum(mu[:, None]*cov, axis=0)),
            "P_hat": vector3(np.sum(mu[:, None]*qhat, axis=0)),
            "J_origin_cov": vector3(np.sum(mu[:, None]*np.cross(pos, cov), axis=0)),
            "J_internal_cov": vector3(
                np.sum(mu[:, None]*np.cross(pos-center_array, cov), axis=0)),
        })
        gamma = 1./np.sqrt(1.-beta*beta)
        thermal = qhat.copy()
        thermal[:, 1] = gamma*(qhat[:, 1]-beta*what)
        component_rows.append({
            "component": component, "count": int(mask.sum()),
            "tag_min": int(tags[mask].min()), "tag_max": int(tags[mask].max()),
            "rest_weight_min": float(mu.min()), "rest_weight_max": float(mu.max()),
            "position_mean": vector3(np.mean(pos, axis=0)),
            "position_std": vector3(np.std(pos, axis=0)),
            "orthonormal_momentum_mean": vector3(np.mean(qhat, axis=0)),
            "rest_frame_thermal_mean": vector3(np.mean(thermal, axis=0)),
            "rest_frame_thermal_std": vector3(np.std(thermal, axis=0)),
        })
    return {
        "path": particles["path"], "profile": str(profile_path),
        "time": particles["time"], "count": particles["count"],
        "finite": finite, "integer_tags": integer_tags,
        "exact_unique_tag_range": exact_tags,
        "positive_rest_weights": bool(np.all(mass > 0.)),
        "inside_domain": bool(np.all(xyz >= -1024.) and np.all(xyz <= 1024.)),
        "components": component_rows,
    }, ledger_rows

def ledger_agreement(logged: list[dict], written: list[dict]) -> tuple[bool, list[dict]]:
    comparisons, passed = [], True
    for left, right in zip(logged, written):
        row = {"component": left["component"], "fields": {}}
        for key in ("rest_mass", "sum_mW", "P_cov", "P_hat",
                    "J_origin_cov", "J_internal_cov"):
            a = np.atleast_1d(np.asarray(left[key], dtype=float))
            b = np.atleast_1d(np.asarray(right[key], dtype=float))
            delta = np.abs(a-b)
            scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.e-9)
            ok = bool(np.all(delta <= 2.e-7+2.e-5*scale))
            passed &= ok
            row["fields"][key] = {"pass": ok, "max_abs": float(delta.max()),
                                  "max_scaled": float((delta/scale).max())}
        row["count_pass"] = left["count"] == right["count"]
        passed &= row["count_pass"]
        comparisons.append(row)
    return passed, comparisons

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boosted-run", required=True, type=Path)
    parser.add_argument("--unboosted-run", required=True, type=Path)
    parser.add_argument("--boosted-log", required=True, type=Path)
    parser.add_argument("--unboosted-log", required=True, type=Path)
    parser.add_argument("--athenak", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.athenak.resolve()/"vis"/"python"))
    import bin_convert

    runs = {"boosted": args.boosted_run, "unboosted": args.unboosted_run}
    logs = {"boosted": args.boosted_log, "unboosted": args.unboosted_log}
    variables = ("con_H", "con_M", "con_Mx", "con_My", "con_Mz")
    windows = {
        "left": (-4.5, -1.5, -1.5, 1.5, -1.5, 1.5),
        "right": (1.5, 4.5, -1.5, 1.5, -1.5, 1.5),
        "common_central": (-6., 6., -6., 6., -6., 6.),
    }
    slice_windows = {key: (value[0], value[1], value[2], value[3])
                     for key, value in windows.items()}
    result = {
        "schema_version": 2, "cases": {}, "windows": windows,
        "constraint_norm":
            "cell-volume-weighted on the preparation-only full 3-D leaf-block output; no field-value mask",
        "momentum_density_norm":
            "cell-area-weighted on the canonical written z=0 leaf-block slice",
    }
    agreements = {}
    for name, run in runs.items():
        # The t=0 constraint array is emitted before the first ADM-constraint task and
        # is therefore zero by construction.  Use the latest matched bounded-startup
        # field/source dumps (t=0.0125 for the two-step gate); particles remain the
        # actual first t=0 snapshot selected in written_particle_stats().
        con = bin_convert.read_binary(str(latest(run, "*.con3d.*.bin")))
        tm = bin_convert.read_binary(str(latest(run, "*.tmunu.*.bin")))
        written, written_ledger = written_particle_stats(run)
        logged = parse_ledger(logs[name])
        agree, detail = ledger_agreement(logged, written_ledger)
        agreements[name] = agree
        result["cases"][name] = {
            "constraint_time": float(con["time"]), "source_time": float(tm["time"]),
            "logged_ledger": logged, "written_ledger": written_ledger,
            "ledger_comparison": detail, "written_particles": written,
            "global": {v: volume_stats(con, v) for v in variables},
            **{region: {v: volume_stats(con, v, window) for v in variables}
               for region, window in windows.items()},
            "momentum_density": {
                region: {v: area_stats(tm, v, window)
                         for v in ("tmunu_Sx", "tmunu_Sy", "tmunu_Sz")}
                for region, window in slice_windows.items()
            },
        }

    boosted, unboosted = result["cases"]["boosted"], result["cases"]["unboosted"]
    bl, ul = boosted["logged_ledger"], unboosted["logged_ledger"]
    clump_rows = [row for case in (boosted, unboosted)
                  for row in case["written_particles"]["components"]
                  if row["component"] in (1, 2)]
    assertions = {
        "exact_five_million_particle_census": all(
            case["written_particles"]["count"] == 5_000_000
            for case in (boosted, unboosted)),
        "three_exact_component_counts": all(
            [row["count"] for row in case["written_particles"]["components"]]
            == list(EXPECTED_COUNTS) for case in (boosted, unboosted)),
        "tags_unique_complete_and_integer": all(
            case["written_particles"]["integer_tags"]
            and case["written_particles"]["exact_unique_tag_range"]
            for case in (boosted, unboosted)),
        "finite_positive_and_contained": all(
            case["written_particles"]["finite"]
            and case["written_particles"]["positive_rest_weights"]
            and case["written_particles"]["inside_domain"]
            for case in (boosted, unboosted)),
        "intended_clump_centers": all(
            abs(case["written_particles"]["components"][i]["position_mean"][0]
                - CENTERS[i][0]) < .03
            and abs(case["written_particles"]["components"][i]["position_mean"][1]) < .03
            and abs(case["written_particles"]["components"][i]["position_mean"][2]) < .03
            for case in (boosted, unboosted) for i in (1, 2)),
        "clump_widths_near_configured_value": all(
            all(.60 < value < .80
                for value in case["written_particles"]["components"][i]["position_std"])
            for case in (boosted, unboosted) for i in (1, 2)),
        "rest_frame_velocity_spreads_near_0p02": all(
            all(.019 < value < .021 for value in row["rest_frame_thermal_std"])
            for row in clump_rows),
        "independent_ledgers_match_written_particles": all(agreements.values()),
        "counterclockwise_boost_signs": bl[1]["P_hat"][1] < 0 < bl[2]["P_hat"][1],
        "positive_clump_orbital_Lz":
            bl[1]["J_origin_cov"][2] > 0 and bl[2]["J_origin_cov"][2] > 0,
        "unboosted_bulk_momentum_small":
            abs(ul[1]["P_hat"][1]) < 1.e-6 and abs(ul[2]["P_hat"][1]) < 1.e-6,
        "matched_initial_output_times":
            abs(boosted["constraint_time"]-unboosted["constraint_time"]) < 1.e-12
            and abs(boosted["written_particles"]["time"]
                    - unboosted["written_particles"]["time"]) < 1.e-12,
    }
    result["assertions"] = assertions
    result["boosted_over_unboosted"] = {
        region: {
            variable: {
                metric: boosted[region][variable][metric]
                        / unboosted[region][variable][metric]
                if unboosted[region][variable][metric] != 0 else None
                for metric in ("rms", "mean_abs", "max_abs")
            } for variable in variables
        } for region in ("global", "left", "right", "common_central")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(assertions, indent=2, sort_keys=True))
    return 0 if all(assertions.values()) else 3

if __name__ == "__main__":
    raise SystemExit(main())
