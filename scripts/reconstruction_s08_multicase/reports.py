"""Scientific review text generated exclusively from completed source packets."""
from common import *
from reduce import csv_rows,write_csv

def fmt(value,digits=6):return '—' if value is None else f'{value:.{digits}g}'

def main():
    job();assembly=load(ROOT/'evidence/ASSEMBLY_OK.json');rows=[];details=[]
    for raw in ALL_CASES:
        key=raw['id'];packet=EXT/'imported'/key
        if not packet.exists():packet=EXT/key/'review_packet'
        a=load(packet/'numerical_acceptance.json');s=load(ROOT/'reduced'/key/'summary.json');shell=csv_rows(ROOT/'reduced'/key/'shells.csv');h=csv_rows(ROOT/'reduced'/key/'history.csv')
        assert a['numerical_checks_passed'] and s['complete']
        window=a['AH_endpoint_window'];mirr=window['mirr']['median'] if window else None;mass=window['mass']['median'] if window else None
        row=dict(case=key,yc=raw['yc'],U=raw['U'],endpoint_M=a['endpoint_M'],first_accepted_AH_M=a['first_accepted_AH_M'],first_converged_AH_M=a['first_converged_AH_M'],first_removal_M=a['first_removal_M'],final_particles=a['final_population'],maximum_rest_mass_closure=a['maximum_rest_mass_closure'],final5M_Mirr_median=mirr,final5M_MCH_median=mass,r50_initial_M=shell[0]['r50'],r50_last_pre_removal_M=shell[-1]['r50'],last_shell_time_M=shell[-1]['time'],final_core_lapse=h[-1]['core_lapse'])
        rows.append(row)
        outcome=(f"Accepted AH at {fmt(row['first_accepted_AH_M'])} M_ref; the run ends at {fmt(row['endpoint_M'])} M_ref after the approved tail or time cap.") if row['first_accepted_AH_M'] is not None else f"Reached {fmt(row['endpoint_M'])} M_ref with no accepted AH."
        details += ['',f"### {key}",'',outcome,f"The median COM rest-mass shell is {fmt(row['r50_initial_M'])} M_ref initially and {fmt(row['r50_last_pre_removal_M'])} M_ref at its last unremoved full-population sample, t={fmt(row['last_shell_time_M'])} M_ref. Final surviving population: {row['final_particles']:,}."]
        if window:
            details.append(f"Accepted samples over {fmt(window['start'])}–{fmt(window['end'])} M_ref ({window['samples']} searches) give median Mirr={fmt(mirr)} and MCH={fmt(mass)} M_ref. These are qualitative late-window masses; no settling or precision claim follows.")
        geometry=a['removal_geometry']
        if geometry:
            details.append(f"{geometry['events_before_first_converged']:,} removal events precede the first qualified converged surface and have no contemporaneous AH assigned. {geometry['events_with_recent_preceding_surface']:,} events have a recent preceding resolved surface; {geometry['events_outside_preceding_inscribed_radius']:,} lie outside its sampled inscribed coordinate radius. This is a coordinate check, without a worldtube-containment proof.")
    write_csv(ROOT/'reports/extension_summary.csv',rows)
    lines=['# Session 8: 700M continuation and inward perturbation scan','',
      'Five requested trajectories use the original proper-volume stratified seed-1 samples with 2,000,000 equal-rest-mass particles. The three yc=.500 variants share the same exact .500 realization; yc=.605 and .819 reuse their respective originals. The unperturbed case continues the accepted 500 M state. All times, radii and AH masses use the original unperturbed equilibrium mass M_ref=1. The inward constraint corrections intentionally change ADM mass slightly and are not renormalized away.','',
      '| Case | Endpoint [M_ref] | Accepted AH [M_ref] | Final particles | Max rest-mass closure |',
      '|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['case']} | {fmt(r['endpoint_M'])} | {fmt(r['first_accepted_AH_M'])} | {r['final_particles']:,} | {r['maximum_rest_mass_closure']:.3g} |")
    lines += details+['','## Interpretation and validation','',
      'An accepted persistent AH establishes black-hole formation. Finite-time contraction, expansion or survival alone does not establish stable migration. Shell curves are coordinate radii about the instantaneous center of mass and stop strictly before the first removal; no post-removal quantile is interpreted as an original Lagrangian shell.','',
      'Rest-mass closure accounts for alive, removed and escaped particle rest mass. AH mass is displayed separately and is not added to that ledger. The ledger does not measure ADM-energy conservation.','',
      'Full particle/ID/removal/cohort checks, three independent full-population quantile spot checks per new trajectory segment, stream coverage, global and local constraint definitions, native-log health, AH residual/resolution/persistence, and finite final 3D fields are bound to each numerical acceptance receipt. Previously accepted 0–500 M results are reused with their exact source manifest; they are not presented as a new convergence test.','',
      'The inherited particle stress-energy source remains frozen across the metric RK stages. Its evolution-accuracy limitations remain unresolved by this experiment. No new convergence or asymptotic-stability claim is made.','',
      'The 700 M continuation retains the original +/-512M box. The separate annular-lapse assessment documents outer-domain gauge behavior; a larger-domain control and proof of causal isolation are unavailable. The .819 kick retains +/-1024M and its initial hierarchy, with allowable AMR to 0.0625 M.','',
      'Numerical acceptance and actual exported-PDF/encoded-frame visual acceptance are separate records. See evidence/ASSEMBLY_OK.json, per-source numerical_acceptance.json, and the final visual acceptance record.']
    (ROOT/'reports/SCIENCE_REPORT.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'reports/FIGURES.md').write_text('''# Production figure conventions

The shell/lapse fiducial shows nine 10–90% COM rest-mass deciles and the native coordinate-volume core lapse within r<.75M_ref. Shells stop before first removal. Columns use their actual endpoints with common physical units and common shell-radius limits; column width does not encode duration. The unperturbed panel marks300M and the500M extension boundary. Dashed lines mark first accepted AH; dotted lines mark first particle removal.

The mass figure displays alive/removed/escaped rest mass and the accounted total, followed by the signed closure residual. The AH figure shows accepted Mirr and Christodoulou mass; rejected or missing surfaces remain gaps. Qualitative AH masses use unchanged residual, on-grid, resolution and three-search persistence gates.

Source-backed figures: s08_extension_shells_lapse.pdf, s08_extension_mass_conservation.pdf, s08_extension_AH_masses.pdf and the 700 M outer-domain annular-lapse PDF. Actual raster inspection is recorded separately.
''')
    (ROOT/'reports/MOVIES.md').write_text('''# Production movies

Five individual and one synchronized 15 fps MP4 use the fixed 20,000 initial particle IDs divisible by 100. Colors remain fixed at initial coordinate radius divided by the initial isotropic surface radius, on a common 0..1 scale. The central slice is |z|<.05R_iso(0). Each spatial window stays fixed for the entire movie and includes every displayed cohort particle with a 5% margin; scales are labeled and may differ. The initial surface is dotted. Black AH equators require a recent contemporaneous accepted resolved native harmonic surface.

Frames use completed cohort samples and print their actual times. Synchronized movies clear the particles and label ended runs, with no invented evolution beyond an endpoint. Nominal comparison cadence is 0.5 M, with the actual fractional final endpoint appended. Every encoded frame must decode, the frame count must match, and representative encoded frames including horizon and lineage/endpoint transitions must be visually inspected. Lossless PNG frames and source arrays remain on the clusters.
''')
    (ROOT/'reports/TECHNICAL_REPORT.md').write_text('''# Numerical implementation and provenance

The evolution source and integrator remain unchanged. Fresh inward initial data use native db942607's exact-background incremental spherical constraint correction; the preserved Kokkos/CUDA 12.9 and ROCm 6.4.1/GFX90A backends are retained. No MI350/backend upgrade is used. The unperturbed restart retains its original evolution executable. Actual build, executable hashes, module versions, commands, exact inputs, workflow snapshots and checkpoint payload receipts are retained by job.

Original particle positions, tags and equal rest-mass weights are preserved. The physical coherent radial mode uses U(3x-x^3)/2 with x the initial areal radius divided by the surface radius. An independent 192-point angular/Green calculation checks the continuum correction, and actual full 2M initialization is checked against an identical native U=0 control with Hamiltonian/momentum norm ratios<=1.005. Machine/rank/capacity changes require byte-verified checkpoint payload and actual zero-step particle/interior-grid identity.

Histories and the mass ledger use 0.25 M, full particles and equatorial slices 1 M, fixed-ID cohorts 0.5 M and AH searches 0.5 M. Native segments checkpoint cleanly at bounded wall times. Interrupted original bytes remain inventoried outside the accepted canonical prefix. Exact native stopping is accepted AH + 20 M capped at each requested limit; the endpoint audit permits at most one actual timestep of the native AH-tail crossing, subject to the hard time cap.

Raw data, reductions, long logs, code, lossless movie frames, final figures/movies and source manifests remain on clusters. A one-way thin review mirror is verified separately. Accounting separates exact terminal scheduler evidence from observed or bounded unknown allocations; exit traps alone do not prove completion.
''')
    atomic(ROOT/'evidence/REPORTS_READY.json',dict(utc=now(),job=job(),source_assembly=receipt(ROOT/'evidence/ASSEMBLY_OK.json'),science=rows,actual_visual_acceptance=False,resource_accounting_and_thin_mirror_pending=True))

if __name__=='__main__':main()
