#!/usr/bin/env python3
"""Generate the Stage-4 long-duration inputs on the causally clean domain.

Causal-domain design
--------------------
The reduced comparison domain ([-176M,176M]^3) admitted boundary influence at
1.28P/1.34P, contaminating every late-time feature of the 2026-07-30/31
campaigns.  This mesh keeps the *entire* refinement structure and spacing of
that validated configuration inside |x| <= 176M (finest dx = 0.0859375M around
the cluster, identical dt = 0.021484375M) and adds three coarser outer shells:

    level  half-width/M   dx/M                 role
    root       1408       22        new coarse exterior
    1           704       11        new coarse exterior
    2           352       5.5       new coarse exterior
    3           176       2.75      == old root domain
    4..8    87.9..5.4  1.375..0.0859375  == old levels 1..5 (unchanged)

512 MeshBlocks of 32^3 (vs 344 before, x1.49 cost); the timestep is set by the
unchanged finest level, so all particle diagnostics remain directly comparable.

Boundary-contact estimates (distance from boundary to the cluster surface at
r_iso = 5.05M/4.85M, divided by the coordinate characteristic speed):

    speed model                          v      t_contact(6.1)   t_contact(5.9)
    coordinate light speed (alpha/A^2)  <= 1      1403M=14.8P      1403M=15.6P
    conservative Z4c bound (sqrt(2))     1.414     992M=10.48P      992M=11.02P
    worst-case gauge sqrt(2/alpha_min)   1.70      825M= 8.72P      825M= 9.16P

The analysis window is [0, 8P].  Against the primary conservative bound
(sqrt(2), the same convention as every previous campaign) the margin is
2.48P / 3.02P, satisfying the >= max(1P, 20%) = 1.6P requirement.  Even under
the unrealistically pessimistic 1.70 bound, boundary contact stays outside the
window (0.72P / 1.16P margin), and a [0, 7P] sub-window satisfies the full
margin requirement.  The predicted contact times are stamped on every
time-series plot by the analysis pipeline.

Everything else -- particle count (528,384), physics, Z4c damping, CFL,
integrator, diagnostics -- is identical to the validated Stage-2/2026-08-01
configuration; only the compactness differs between the two models.  GW
extraction gains a fourth radius at 160M, now far from the boundary.
"""
import argparse
import hashlib
import json
from pathlib import Path

PERIODS = {"q6p1": 94.661770035504048, "q5p9": 90.044644089281221}
Q = {"q6p1": 6.1, "q5p9": 5.9}
SEEDS = [1985, 424242, 20260801]
NPERIOD = 8.0

TEMPLATE = """# Hybrid-sampler long-duration stability run, {model} (R/M={q}), seed={seed}.
# Causally clean domain: [-1408M,1408M]^3, 8 static levels, finest dx and the
# whole structure inside |x|<=176M identical to the validated reduced grid.
# Conservative (sqrt2) boundary contact at 992M = {tc_over_p:.2f}P; tlim = 8P.
# period = {period!r}

<comment>
problem = NRPIC hybrid-sampler stability run R/M={q} stratified_antithetic seed {seed}

<job>
basename = {basename}

<mesh>
nghost = 4
nx1    = 128
x1min  = -1408.0
x1max  =  1408.0
ix1_bc = outflow
ox1_bc = outflow
nx2    = 128
x2min  = -1408.0
x2max  =  1408.0
ix2_bc = outflow
ox2_bc = outflow
nx3    = 128
x3min  = -1408.0
x3max  =  1408.0
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
x1min = -704.0
x1max =  704.0
x2min = -704.0
x2max =  704.0
x3min = -704.0
x3max =  704.0

<refined_region2>
level = 2
x1min = -352.0
x1max =  352.0
x2min = -352.0
x2max =  352.0
x3min = -352.0
x3max =  352.0

<refined_region3>
level = 3
x1min = -176.0
x1max =  176.0
x2min = -176.0
x2max =  176.0
x3min = -176.0
x3max =  176.0

<refined_region4>
level = 4
x1min = -87.9
x1max =  87.9
x2min = -87.9
x2max =  87.9
x3min = -87.9
x3max =  87.9

<refined_region5>
level = 5
x1min = -43.9
x1max =  43.9
x2min = -43.9
x2max =  43.9
x3min = -43.9
x3max =  43.9

<refined_region6>
level = 6
x1min = -21.9
x1max =  21.9
x2min = -21.9
x2max =  21.9
x3min = -21.9
x3max =  21.9

<refined_region7>
level = 7
x1min = -10.9
x1max =  10.9
x2min = -10.9
x2max =  10.9
x3min = -10.9
x3max =  10.9

<refined_region8>
level = 8
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
nrad_wave_extraction = 4
extraction_nlev = 10
extraction_radius_0 = 20.0
extraction_radius_1 = 40.0
extraction_radius_2 = 80.0
extraction_radius_3 = 160.0
waveform_dt = 0.5

<particles>
particle_type       = dust
pusher              = gr_boris
gr_boris_diagnostics = true
gr_boris_freeze_metric = false
gr_boris_live_monopole = false
init                = pgen
feedback            = true
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
cluster_nradial                 = 128
cluster_nangular                = 1032
cluster_octahedral_quiet_start  = true
cluster_rotation_enable         = false
cluster_sampler                 = stratified_antithetic
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
dt        = 5.0

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
coarsen_factor  = 2
dt              = {cbin_dt!r}

<output7>
file_type = rst
dt        = {rst_dt!r}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in ("q6p1", "q5p9"):
        P = PERIODS[model]
        for seed in SEEDS:
            name = f"long_{model}_stratified_antithetic_s{seed}"
            text = TEMPLATE.format(
                model=model, q=Q[model], seed=seed, basename=name,
                period=P, tlim=NPERIOD * P, tc_over_p=992.0 / P,
                cbin_dt=round(P / 8.0, 9), rst_dt=round(P / 4.0, 9))
            path = args.output / f"{name}.athinput"
            path.write_text(text)
            digest = hashlib.sha256(text.encode()).hexdigest()
            rows.append({
                "name": name, "model": model, "q": Q[model], "seed": seed,
                "sampler": "stratified_antithetic", "kind": "live",
                "stage": f"long_{model}", "nperiod": NPERIOD,
                "tlim": NPERIOD * P, "period": P,
                "rst_dt": round(P / 4.0, 9), "cbin_dt": round(P / 8.0, 9),
                "pvtk_dt": 5.0, "nradial": 128, "nangular": 1032,
                "npart": 528384, "input": str(path), "input_sha256": digest,
                "boundary_contact_sqrt2_M": 992.0,
                "boundary_contact_sqrt2_P": 992.0 / P,
                "boundary_contact_v1p7_M": 825.3,
                "boundary_contact_light_M": 1402.95,
            })
            print(f"{name}: tlim={NPERIOD*P:.3f} sha={digest[:12]}")

    manifest = args.output / "long_manifest.json"
    manifest.write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(rows)} inputs; manifest {manifest}")


if __name__ == "__main__":
    main()
