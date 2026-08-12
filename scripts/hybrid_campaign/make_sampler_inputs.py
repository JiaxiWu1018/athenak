#!/usr/bin/env python3
"""Generate matched athinput files for the sampler-causality campaign.

Every run in a given family differs ONLY in ``cluster_sampler`` and
``cluster_seed``; grid, particle count, boundaries, damping parameters, output
cadence, and diagnostics are byte-identical to the preserved reduced-cost
R/M=6.1 and R/M=5.9 comparison inputs.  The evolution limit is set from the
model surface period so that "one period" means the same physical fraction of
the orbit in each model.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

PERIODS = {"q6p1": 94.661770035504048, "q5p9": 90.044644089281221}
Q = {"q6p1": 6.1, "q5p9": 5.9}

TEMPLATE = """# Sampler-causality campaign, {model} (R/M={q}), sampler={sampler}, seed={seed}.
# Grid, particle count, boundaries, damping, and diagnostics are identical to the
# preserved reduced-cost comparison input; only <problem>/cluster_sampler and
# <problem>/cluster_seed differ within a family.  tlim = {nperiod} x surface period.

<comment>
problem = NRPIC sampler-causality homogeneous cluster R/M={q} {sampler} seed {seed}

<job>
basename = {basename}

<mesh>
nghost = 4
nx1    = 128
x1min  = -176.0
x1max  =  176.0
ix1_bc = outflow
ox1_bc = outflow
nx2    = 128
x2min  = -176.0
x2max  =  176.0
ix2_bc = outflow
ox2_bc = outflow
nx3    = 128
x3min  = -176.0
x3max  =  176.0
ix3_bc = outflow
ox3_bc = outflow

<meshblock>
nx1 = 32
nx2 = 32
nx3 = 32

<mesh_refinement>
refinement       = static
max_nmb_per_rank = 1600

<refined_region1>
level = 1
x1min = -87.9
x1max =  87.9
x2min = -87.9
x2max =  87.9
x3min = -87.9
x3max =  87.9

<refined_region2>
level = 2
x1min = -43.9
x1max =  43.9
x2min = -43.9
x2max =  43.9
x3min = -43.9
x3max =  43.9

<refined_region3>
level = 3
x1min = -21.9
x1max =  21.9
x2min = -21.9
x2max =  21.9
x3min = -21.9
x3max =  21.9

<refined_region4>
level = 4
x1min = -10.9
x1max =  10.9
x2min = -10.9
x2max =  10.9
x3min = -10.9
x3max =  10.9

<refined_region5>
level = 5
x1min = -5.4
x1max =  5.4
x2min = -5.4
x2max =  5.4
x3min = -5.4
x3max =  5.4

<time>
evolution  = dynamic
integrator = rk4
cfl_number = 0.25
nlim       = 100000000
tlim       = {tlim!r}
ndiag      = 100

<z4c>
lapse_harmonic = 0.0
lapse_oplog    = 2.0
shift_eta      = 2.0
diss           = 0.1
chi_div_floor  = 0.00001
damp_kappa1    = 0.02
damp_kappa2    = 0.0
nrad_wave_extraction = 3
extraction_nlev = 10
extraction_radius_0 = 20.0
extraction_radius_1 = 40.0
extraction_radius_2 = 80.0
waveform_dt = 0.5

<particles>
particle_type       = dust
pusher              = gr_boris
gr_boris_diagnostics = true
gr_boris_freeze_metric = false
gr_boris_live_monopole = false
init                = pgen
feedback            = {feedback}
mass                = 1.0
debug               = 0
destroy_log         = true
excise_radius       = 0.0
excise_x1           = 0.0
excise_x2           = 0.0
excise_x3           = 0.0
excise_lapse        = 0.08
cross_level_deposit = conservative

<problem>
user_hist                       = true
cluster_mass                    = 1.0
cluster_radius_over_mass        = {q}
cluster_xi                      = 1.0
cluster_nradial                 = {nradial}
cluster_nangular                = {nangular}
cluster_octahedral_quiet_start  = true
cluster_rotation_enable         = false
cluster_rotation_axis_x         = 1.0
cluster_rotation_axis_y         = 2.0
cluster_rotation_axis_z         = 3.0
cluster_rotation_angle          = 0.37
cluster_sampler                 = {sampler}
cluster_seed                    = {seed}
cluster_center_x1               = 0.0
cluster_center_x2               = 0.0
cluster_center_x3               = 0.0

<output1>
file_type = hst
dt        = 1.0
data_format = %.17e

<output2>
file_type = pvtk
variable  = prtcl_all
dt        = {pvtk_dt!r}

<output3>
file_type = bin
variable  = con
slice_x3  = 0.0
dt        = 2.0

<output4>
file_type = bin
variable  = tmunu
slice_x3  = 0.0
dt        = 2.0

<output5>
file_type = bin
variable  = z4c
slice_x3  = 0.0
dt        = 2.0

<output6>
file_type       = cbin
variable        = adm
coarsen_factor  = 1
dt              = {cbin_dt!r}

<output7>
file_type = rst
dt        = {rst_dt!r}
"""

# The fixed-background smoke input replaces <z4c> with <adm> and feedback=false.
FIXED_TEMPLATE = """# Fixed-background pusher smoke test, {model} (R/M={q}), sampler={sampler}, seed={seed}.
# Analytic frozen metric with feedback=false: any drift here is a pusher or
# coordinate-conversion error, not live-field feedback.

<comment>
problem = NRPIC sampler fixed-background smoke R/M={q} {sampler}

<job>
basename = {basename}

<mesh>
nghost = 4
nx1    = 64
x1min  = -44.0
x1max  =  44.0
ix1_bc = outflow
ox1_bc = outflow
nx2    = 64
x2min  = -44.0
x2max  =  44.0
ix2_bc = outflow
ox2_bc = outflow
nx3    = 64
x3min  = -44.0
x3max  =  44.0
ix3_bc = outflow
ox3_bc = outflow

<meshblock>
nx1 = 32
nx2 = 32
nx3 = 32

<mesh_refinement>
refinement       = static
max_nmb_per_rank = 1600

<refined_region1>
level = 1
x1min = -22.0
x1max =  22.0
x2min = -22.0
x2max =  22.0
x3min = -22.0
x3max =  22.0

<refined_region2>
level = 2
x1min = -11.0
x1max =  11.0
x2min = -11.0
x2max =  11.0
x3min = -11.0
x3max =  11.0

<time>
evolution  = dynamic
integrator = rk4
cfl_number = 0.25
nlim       = 100000000
tlim       = {tlim!r}
ndiag      = 50

<adm>

<particles>
particle_type       = dust
pusher              = gr_boris
gr_boris_diagnostics = true
gr_boris_freeze_metric = false
gr_boris_live_monopole = false
init                = pgen
feedback            = false
mass                = 1.0
debug               = 0
destroy_log         = true
excise_radius       = 0.0
excise_x1           = 0.0
excise_x2           = 0.0
excise_x3           = 0.0
excise_lapse        = 0.08

<problem>
user_hist                       = true
cluster_mass                    = 1.0
cluster_radius_over_mass        = {q}
cluster_xi                      = 1.0
cluster_nradial                 = {nradial}
cluster_nangular                = {nangular}
cluster_octahedral_quiet_start  = true
cluster_rotation_enable         = false
cluster_sampler                 = {sampler}
cluster_seed                    = {seed}
cluster_center_x1               = 0.0
cluster_center_x2               = 0.0
cluster_center_x3               = 0.0

<output1>
file_type = hst
dt        = 0.5
data_format = %.17e

<output2>
file_type = pvtk
variable  = prtcl_all
dt        = {pvtk_dt!r}
"""


def write_input(path, kind, **kw):
    text = (FIXED_TEMPLATE if kind == "fixed" else TEMPLATE).format(**kw)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--nradial", type=int, default=128)
    ap.add_argument("--nangular", type=int, default=1032)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    plan = json.loads(Path(args.manifest).read_text())
    rows = []
    for entry in plan:
        model = entry["model"]
        kind = entry.get("kind", "live")
        nperiod = entry.get("nperiod", 1.0)
        tlim = (entry["tlim"] if "tlim" in entry
                else PERIODS[model] * nperiod)
        basename = entry["name"]
        nradial = entry.get("nradial", args.nradial)
        nangular = entry.get("nangular", args.nangular)
        path = args.output / f"{basename}.athinput"
        digest = write_input(
            path, kind, model=model, q=Q[model], sampler=entry["sampler"],
            seed=entry["seed"], basename=basename, tlim=tlim, nperiod=nperiod,
            nradial=nradial, nangular=nangular,
            feedback="true",
            pvtk_dt=entry.get("pvtk_dt", 2.0 if kind == "fixed" else 5.0),
            cbin_dt=entry.get("cbin_dt", 10.0),
            rst_dt=entry.get("rst_dt", 20.0))
        rows.append({**entry, "input": str(path), "input_sha256": digest,
                     "tlim": tlim, "nradial": nradial, "nangular": nangular,
                     "npart": 4 * nradial * nangular,
                     "period": PERIODS[model], "q": Q[model]})
        print(f"{basename}: sampler={entry['sampler']} seed={entry['seed']} "
              f"tlim={tlim:.6f} sha={digest[:12]}")

    out = args.output / "input_manifest.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(rows)} inputs; manifest {out}")


if __name__ == "__main__":
    main()
