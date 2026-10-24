"""Source-bound scientific reports; actual visual acceptance stays explicit."""
import csv
import shutil
from common import *
from extension_resources import storage, allocated_bytes


def main():
    job()
    assembly = load(ROOT / 'evidence/ASSEMBLY_OK.json')
    assert assembly['source_files_SHA256_verified']
    assert [r['case'] for r in assembly['sources']] == [c['id'] for c in PLOT_CASES]
    rows, sources, retention = [], [], []
    (ROOT / 'inputs').mkdir(exist_ok=True)
    for raw in PLOT_CASES:
        key = raw['id']
        summary = load(ROOT / 'reduced' / key / 'summary.json')
        reused = raw.get('reused_session8', False)
        packet = EXT / 'imported' / key if reused else EXT / key / 'review_packet'
        acceptance = load(packet / 'numerical_acceptance.json')
        manifest = load(packet / 'MANIFEST.json')
        assert acceptance['numerical_checks_passed'] and summary['complete']
        assert manifest['checkpoint']['sha256'] == acceptance['checkpoint']['sha256']
        rows.append(dict(case=key, yc=raw['yc'], U=raw['U'],
                         endpoint_M=summary['endpoint'],
                         first_accepted_AH_M=summary['ah_first'],
                         final_alive=acceptance['final_population'],
                         first_removal_M=summary['first_removal'],
                         rest_mass_closure=summary['max_closure'],
                         removals=summary['removals'],
                         late_AH_window=summary['ah_endpoint_window'],
                         reused_session8=reused))
        sources.append(dict(case=key, manifest=receipt(packet / 'MANIFEST.json'),
                            acceptance=receipt(packet / 'numerical_acceptance.json'),
                            original_source_root=manifest['source_root'],
                            source_machine=manifest['machine']))
        destination = ROOT / 'inputs' / f'{key}.athinput'
        shutil.copy2(packet / 'exact.athinput', destination)
        assert sha(destination) == sha(packet / 'exact.athinput')
        if not reused:
            inventory = EXT / key / 'evidence/CLOSED_OUTPUT_MANIFEST.json'
            closed = load(inventory)
            assert closed['stable_files_verified']
            assert closed['checkpoint']['sha256'] == manifest['checkpoint']['sha256']
            retention.append(dict(case=key, raw_root=str(EXT / key),
                                  allocated_bytes=allocated_bytes(EXT / key),
                                  closed_output_manifest=receipt(inventory),
                                  retention='Retained on AMD; recorded recovery rotations only'))

    text = (
        '# Session 9: uniform inward perturbation\n\n'
        'Four new trajectories use U=-0.005 and the original proper-volume, '
        'stratified seed-1 samples of 2,000,000 equal-rest-mass particles. '
        'The completed session 8 yc=0.500/U=-0.005 result is reused without '
        'evolution. Times and AH masses use the original equilibrium M_ref=1.\n\n'
        '| yc | Endpoint [M_ref] | First accepted AH [M_ref] | Final alive | Origin |\n'
        '|---|---:|---:|---:|---|\n'
    )
    for row in rows:
        onset = row['first_accepted_AH_M']
        onset_text = 'none' if onset is None else f'{onset:.6g}'
        origin = 'Session 8 reference' if row['reused_session8'] else 'Session 9 AMD'
        text += (f"| {row['yc']:.3f} | {row['endpoint_M']:.6g} | {onset_text} | "
                 f"{row['final_alive']:,} | {origin} |\n")
    text += (
        '\nA persistent qualified apparent horizon establishes black-hole formation. '
        'Absence of an accepted horizon through a finite endpoint does not '
        'establish asymptotic stability or stable migration. Outcomes are measured '
        'without enforcing predetermined fates.\n\n'
        'The stopping rule is min(300M, accepted AH + 20M). The reused 0.500 '
        'trajectory originally had a 200M cap; AH onset at 73M and its 93M '
        'endpoint also satisfy this rule without extension.\n\n'
        'Shells are COM-centered rest-mass deciles and stop strictly before first '
        'removal. Core lapse is the coordinate-volume average inside 0.75M_ref. '
        'All static comparison columns use a common 0–300M_ref time axis; curves '
        'end at their actual endpoints. Dotted markers show first removal; '
        'dashed markers show persistent accepted AH onset.\n\n'
        'The rest-mass ledger tests particle bookkeeping, separately from ADM '
        'energy and AH mass. Irreducible and Christodoulou AH masses are reported '
        'independently. Final-five-M measurements are qualitative, without a '
        'precision or settling claim; missing or rejected AH intervals stay explicit.\n\n'
        'Lapse removal uses threshold 0.1; AH particle excision is disabled. '
        'Per-case geometry audits record removals preceding the first qualified '
        'AH and make no worldtube-containment proof. The inherited source-time '
        'coupling is unchanged. No new convergence or causal-isolation claim is made.\n\n'
        '[Shells and lapse](../plots/s09_shells_lapse.pdf), '
        '[rest-mass ledger](../plots/s09_mass_conservation.pdf), '
        '[AH masses](../plots/s09_AH_masses.pdf), and '
        '[synchronized movie](../movies/s09_comparison.mp4).\n'
    )
    (ROOT / 'reports/SCIENCE_REPORT.md').write_text(text)
    (ROOT / 'reports/TECHNICAL_REPORT.md').write_text(
        '# Numerical implementation and provenance\n\n'
        'Native source is unchanged from 2d1f2525, including db942607 incremental '
        'initial constraint correction. The physical mode U(3x-x^3)/2 is applied '
        'in the local Eulerian orthonormal frame; x is initial areal radius divided '
        'by surface areal radius. Original positions, IDs and rest-mass weights '
        'are preserved. Particle energies change; M_ref is not renormalized.\n\n'
        'Domains, initial hierarchies, 128^3 root cells and 32^3 MeshBlocks are '
        'preserved. Cases 0.283/0.605 use ±512M, initial finest 0.25M and allowed '
        '0.0625M; 0.819/0.900 use ±1024M, initial finest 0.5M and allowed 0.125M. '
        'All four use explicit num_levels=8. AMR criterion and hierarchy floors, '
        'Z4c, gauge, RK4/CFL=0.4, particle pusher and deposition are unchanged.\n\n'
        'Admission verifies actual full samples, transforms, outputs/cohorts, '
        '344-block hierarchies, matched U=0 constraint norm ratios ≤1.005 and '
        'zero-step restart identity. Independent 192-point angular/Green '
        'references check continuum corrections. Preserved Kokkos and '
        'ROCm 6.4.1/GFX90A backends are fixed.\n\n'
        'Histories at 0.25M, full particles/equatorial fields at 1M and fixed '
        '20,000-ID cohorts/AH searches at 0.5M retain actual times. AH acceptance '
        'requires positive, on-grid geometry, hrel≤0.05, minimum radius of four '
        'local cells and three consecutive qualifications. Final 3D exports '
        'do not evolve the accepted checkpoint.\n\n'
        'Endpoint audits check population IDs/counts/weights, removals, ledger, '
        'cadence/endpoint coverage, constraints/native health, AH geometry/masses '
        'and independent full-population COM quantiles. Actual PDF and encoded '
        'frame inspection is recorded separately from automated validation.\n'
    )
    columns = ['case', 'yc', 'U', 'endpoint_M', 'first_accepted_AH_M',
               'final_alive', 'first_removal_M', 'rest_mass_closure', 'reused_session8']
    with (ROOT / 'reports/OUTCOMES.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in columns} for row in rows])
    atomic(ROOT / 'evidence/REPORT_SOURCE_BINDINGS.json',
           dict(utc=now(), job=job(), outcomes=rows, sources=sources,
                raw_retention=retention, storage=storage(), actual_visual_acceptance=False))
    (ROOT / 'reports/CAPTIONS.md').write_text(
        '# Figure and movie captions\n\n'
        'Shell/lapse: five seed-1, proper-volume stratified 2M-particle realizations '
        'with physical U=-0.005 in ascending yc. Nine COM rest-mass deciles; '
        'curves stop before first removal. Core lapse averages coordinate volume '
        'inside 0.75M_ref. Common 0–300M_ref axes retain actual endpoints. '
        'Dotted: first removal; dashed: persistent accepted AH.\n\n'
        'Ledger: alive, removed and escaped rest-mass fractions, accounted total '
        'and closure error. Initial rest mass is the normalization. This '
        'bookkeeping does not measure ADM-energy conservation.\n\n'
        'AH mass: Mirr=sqrt(A/16pi) and Christodoulou mass in original M_ref '
        'units. Only accepted resolved samples appear; gaps are not interpolated. '
        'Last-five-M statistics are qualitative.\n\n'
        'Movies: fixed 20,000 original IDs in a central xy slice of fixed '
        'thickness. Colors encode initial radius/Riso(0). Spatial bounds are '
        'fixed over each case’s history and may differ between cases. AH '
        'contours require contemporaneous accepted quality. Panels show actual '
        'cohort time/count and explicitly mark ended runs. Synchronized display '
        'uses common physical time; encoding is 15fps.\n'
    )
    (ROOT / 'README.md').write_text(
        '# Reconstruction ST session 9\n\n'
        '[Science](reports/SCIENCE_REPORT.md), '
        '[technical provenance](reports/TECHNICAL_REPORT.md), '
        '[outcomes](reports/OUTCOMES.csv), [captions](reports/CAPTIONS.md).\n\n'
        'Consult evidence/FINAL_ACCEPTANCE.json for delivery status. Numerical '
        'acceptance and actual visual acceptance are separate. New raw data, '
        'reductions, logs, source and frames stay on AMD. The original 0.500 '
        'reference raw data remain preserved on Perseus.\n'
    )


if __name__ == '__main__':
    main()
