"""Source-bound scientific narrative for the existing static MD/PDF deliverable."""
import textwrap
from common import *
from reduce import csv_rows


def number(value, precision=8):
    return 'unavailable' if value is None else f'{value:.{precision}g}'


def build_lines(data, statistics, profile_coverage, profile_drift, out, validation_only=False):
    """Read accepted evidence, preserve metric scope, and freeze report provenance."""
    job()
    if not validation_only:
        assert {d['case']['id'] for d in data} == {'yc0500_outward', 'yc0500_control'}
    sources = []
    bound = []
    for d in data:
        root, summary = d['root'], d['summary']
        visual_path = root/'evidence/CASE_VISUAL_ACCEPTANCE.json'
        visual = load(visual_path)
        assert visual['actual_visual_review']
        assert abs(visual['endpoint']-summary['endpoint']) < 1e-7
        marker_path = root/'evidence'/f"MILESTONE_{round(summary['endpoint']):05d}_NUMERICAL.json"
        marker = load(marker_path)
        assert marker['numerical_passed']
        assert abs(marker['endpoint']-summary['endpoint']) < 1e-7
        state_path = root/'runs/yc0500/state.json'
        state = load(state_path)
        assert state['complete'] and summary['complete']
        assert state['latest']['sha256'] == d['acceptance']['checkpoint']['sha256']
        assert abs(d['acceptance']['endpoint_M']-summary['endpoint']) < 1e-7
        sources.extend(receipt(p) for p in [visual_path, marker_path, state_path,
            root/'evidence/CASE_NUMERICAL_ACCEPTANCE.json', d['red']/'summary.json'])
        for entry in marker['science_trends']['source_inputs']:
            path = d['red']/Path(entry['path']).name
            assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
            sources.append(receipt(path))
        bound.append((d, marker))
    title = '# Session 10: outward collapse and original unperturbed continuation'
    lines = [title, '']
    if validation_only:
        lines += ['REPORT WORKFLOW VALIDATION ONLY. The accepted outward case exercises the actual narrative/PDF code; this is not the final two-case report or a new scientific acceptance.', '']
    for d, marker in bound:
        s, a = d['summary'], d['acceptance']
        label = 'Outward U=+0.005' if not d['case']['start'] else 'Original U=0 control'
        if s['ah_first'] is not None:
            lines += [f"{label} formed a persistence-qualified apparent horizon at {number(s['ah_first'])}M and ended at {number(s['endpoint'])}M under the agreed horizon-plus20M rule. A bound non-black-hole remnant was not obtained in this trajectory.", '']
        else:
            lines += [f"{label} reached {number(s['endpoint'])}M with no accepted apparent horizon. The final alive population is {a['final_population']:,}. Survival alone does not establish stability; the finite-window shell and profile comparisons below describe the available settling evidence.", '']
    lines += ['## Experiment and validity gates', '',
        'Both cases use corrected-profile yc=0.500 data on the original +/-512M mesh, initial finest spacing M/4 and AMR permission to M/16. Root128^3/block32^3, refinement regions/criterion, gauge/Z4c, RK4/CFL0.4, deposition, GR-Boris pusher and source-time coupling are preserved. Each original realization has2,000,000 proper-volume stratified seed1 particles, with unchanged positions, IDs and rest-mass weights.', '',
        'The outward case applies the physical radial increment delta v^rhat=0.005(3x-x^3)/2, x=initial areal radius/surface radius, followed by the validated momentum and Hamiltonian corrections. M_ref=1 remains the original equilibrium mass; no renormalization or amplitude scan was performed. The calculated mean increment is0.00390723754 and continuum ADM increase1.419462437e-5M (about0.060% of initial binding energy). This is a justified single trial, not a demonstrated optimal damping amplitude.', '',
        'The preserved matched initialization/U0/outward20M checks passed before production; the largest initial H/M norm ratio was1.0000490181, below1.005. The control imports the original700M donor, with actual zero-step identity under the preserved original executable/eight ranks before advancement. Accepted control history and fixed-ID cohorts through700M are reused; that interval was not evolved again.', '',
        f'Original donor SHA256: {ANCHOR_SHA}. Original executable SHA256: {OLD_EXE_SHA}. Perturbed initializer SHA256: {PERTURBED_EXE_SHA}.', '',
        '## Endpoint measurements and qualifications', '',
        '| Case | End [M] | Accepted AH [M] | Alive particles | First removal [M] |',
        '|---|---:|---:|---:|---:|']
    for d, marker in bound:
        s, a = d['summary'], d['acceptance']
        lines.append(f"| {d['case']['id']} | {number(s['endpoint'])} | {number(s['ah_first']) if s['ah_first'] is not None else 'none'} | {a['final_population']} | {number(s['first_removal']) if s['first_removal'] is not None else 'none'} |")
    for d, marker in bound:
        s, a, trends = d['summary'], d['acceptance'], marker['science_trends']
        h, mass = d['curves']['history'][-1], d['curves']['mass'][-1]
        cp = a['checkpoint']
        lines += ['', f"### {d['case']['id']}", '',
            f"The last uncensored whole-population shell sample is {number(trends['last_uncensored_shell_time'])}M: r50={number(trends['final_r50'])}M and r90={number(trends['final_r90'])}M. The first available profile baseline is {number(trends['profile_baseline_time'])}M; redistribution at {number(trends['last_profile_time'])}M is {number(trends['final_profile_redistribution'])}, defined as sum(abs(delta mass bins))/(2 M0_initial). This baseline statistic is distinct from comparisons between late time-mean profiles.", '',
            f"Endpoint volume-mean core lapse={number(h['core_lapse'])}; local proper-volume H/M RMS inside r<10M={number(h['H_rms_r10'])}/{number(h['M_rms_r10'])}. Active checkpoint mesh: {cp['nblocks']} blocks, finest dx={number(cp['finest_active_dx'])}M. Temporal changes and finite norms are not a convergence result; global exterior-diluted norms do not replace local diagnostics.", '',
            f"Final alive rest mass={number(mass['m_alive'])}M of initial M0={number(s['rest_mass_initial'])}M. Native removal counters: "+', '.join(f'{k}={v}' for k, v in s['removals'].items())+'. Alive matter after lapse excision is not an escaped/unbound mass fraction. Rest-mass ledger closure measures bookkeeping, not ADM-energy conservation; horizon mass and particle rest mass cannot simply be added into an ADM budget.', '',
            f"Audit coverage: {s['snapshots']} full-particle files and {s['cohorts']} fixed-ID cohorts, with independent whole-population rank checks, particle/removal IDs, original weights, ledger and cohort/full-output agreement. Maximum ledger closure={number(s['max_closure'])}. Source-bound finite final fields and stream coverage are recorded in CASE_NUMERICAL_ACCEPTANCE.json.", '',
            f"Recognized native diagnostics: {marker['fallback_particle_steps']} fallback particle-steps; {marker['rejected_updates_from_recognized_counters']} rejected updates; {marker['transport_pointer_info_error_lines']} UCX null-pointer-info lines. Accepted logs contain no positive search-failure counters or fatal marker. Native return codes, inventories and constraints provide separate evidence; these checks do not independently prove transport correctness.", '']
        if s['first_removal'] is not None:
            lines += [f"Whole-population shell/profile curves stop before the first removal at {number(s['first_removal'])}M; the last uncensored radii are not survivor radii at the endpoint.", '']
        if s['ah_first'] is not None:
            ah = s['ah_endpoint_window']
            lines += [f"AH acceptance requires streak>=3, no missing samples, relative residual<=0.05, minimum coordinate radius>=4dx and finite positive area/mass. There are {a['accepted_AH_samples']} accepted and {a['converged_AH_samples']} converged samples; the first converged sample is {number(a['first_converged_AH_M'])}M. Saved restarts reset the persistence counter: converged/resolved low-residual samples can await streak3 without being accepted at every output.", '',
                f"The final {ah['samples']} accepted samples span {number(ah['start'])}-{number(ah['end'])}M, with median irreducible mass {number(ah['mirr']['median'])}M and range {number(ah['mirr']['min'])}-{number(ah['mirr']['max'])}M. This describes the saved window, not a demonstrated asymptotic black-hole mass.", '']
            geometry = a.get('removal_geometry')
            if geometry:
                lines += [f"Removal geometry: {geometry['events_before_first_converged']} events precede the first converged surface by at most {number(geometry['maximum_early_lead_M'])}M. Their maximum radius/first-following-inscribed-radius is {number(geometry['early_max_radius_over_first_following_rmin'])}; this later comparison does not assign a contemporaneous horizon. Of {geometry['events_with_recent_preceding_surface']} events with recent preceding qualified surfaces, {geometry['events_outside_preceding_inscribed_radius']} are outside sampled inscribed radii and {geometry['events_without_recent_preceding_surface']} lack a recent surface. Maximum preceding-radius ratio={number(geometry['maximum_radius_over_preceding_rmin'])}. This is a coordinate sanity check without angular interpolation or exact worldtube containment.", '']
        flow_path = d['red']/'coordinate_radial_flow.csv'
        if flow_path.exists():
            f = csv_rows(flow_path)[-1]
            lines += [f"Last fixed-ID radial-flow interval is centered at {number(f['time'])}M: mean coordinate dr/dt={number(f['mean_coordinate_dr_dt'])}, RMS={number(f['rms_coordinate_dr_dt'])}, outgoing fraction={number(f['outgoing_fraction'])}, over {int(f['fixed_ids'])} IDs. These COM-relative coordinate finite differences are not locally measured physical velocities; a small mean can reflect oscillation phase.", '']
        windows = {}
        for row in trends['shell_windows']:
            windows.setdefault((row['start'], row['end']), {})[row['shell']] = row['fractional_peak_to_peak']
        if windows:
            lines += ['Complete100M windows in the newly audited trajectory use (maximum-minimum)/sample mean. These measured excursions include phase and drift, rather than isolated oscillation amplitudes.', '',
                '| Window [M] | r10 excursion [%] | r50 excursion [%] | r90 excursion [%] |',
                '|---|---:|---:|---:|']
            for (lo, hi), values in sorted(windows.items()):
                lines.append(f"| {lo:g}-{hi:g} | {100*values[10]:.6g} | {100*values[50]:.6g} | {100*values[90]:.6g} |")
            lines.append('')
    lines += ['## Finite-window settling evidence', '',
        'Shell means integrate piecewise-linear native snapshots over time; peak-to-peak excursions include interpolated window boundaries. The requested500-1000M and1000-1500M windows are compared only when fully covered. Excursions combine phase, oscillation and secular motion, and do not isolate damping.', '',
        '| Case | Shell [%] | Window [M] | Complete | Mean [M] | Peak-to-peak [M] |',
        '|---|---:|---|---|---:|---:|']
    for r in statistics:
        lines.append(f"| {r['case']} | {r['shell']} | {r['start']:g}-{r['end']:g} | {r['complete']} | {number(r['mean'])} | {number(r['peak_to_peak'])} |")
    lines += ['', '| Case | Profile window [M] | Complete | Covered duration [M] | Max native gap [M] |',
        '|---|---|---|---:|---:|']
    for r in profile_coverage:
        lines.append(f"| {r['case']} | {r['start']:g}-{r['end']:g} | {r['complete']} | {number(r['covered_duration'])} | {number(r['max_native_gap'])} |")
    lines += ['', 'Profile means use equal750-1000M,1000-1250M and1250-1500M windows. Missing or censored windows remain unavailable, never zero; no extrapolation is used. Density is rest mass per COM-coordinate shell volume, not proper-volume energy density. Profile drift compares consecutive time-mean mass-bin profiles: total variation=sum(abs(delta bins))/(2 M0_initial); enclosed change=max(abs(cumulative delta bins))/M0_initial.', '',
        '| Case | Adjacent windows [M] | Complete | Total variation | Max enclosed change |',
        '|---|---|---|---:|---:|']
    for r in profile_drift:
        # Explicit field names are the public window_metrics output contract.
        pair = f"{r['earlier_start']:g}-{r['earlier_end']:g} / {r['later_start']:g}-{r['later_end']:g}"
        lines.append(f"| {r['case']} | {pair} | {r['complete']} | {number(r['total_variation_M0'])} | {number(r['max_enclosed_mass_change_M0'])} |")
    for d, marker in bound:
        key = d['case']['id']
        selected = [r for r in statistics if r['case']==key and r['shell']==50]
        if len(selected)==2 and all(r['complete'] for r in selected):
            amplitudes = [r['peak_to_peak']/r['mean'] for r in selected]
            trend = 'decreased' if amplitudes[1]<amplitudes[0] else 'did not decrease'
            lines += ['', f"{key}: fractional r50 excursion {trend} between the requested windows ({100*amplitudes[0]:.4g}% to {100*amplitudes[1]:.4g}%). This finite comparison mixes phase and drift; it is not a stability determination."]
        else:
            lines += ['', f'{key}: both requested late shell windows are not fully available, so no late-window damping comparison is assigned.']
        drifts = [r for r in profile_drift if r['case']==key and r['complete']]
        if len(drifts)==2:
            trend = 'decreased' if drifts[1]['total_variation_M0']<drifts[0]['total_variation_M0'] else 'did not decrease'
            lines += [f"Consecutive equal-window profile total variation {trend} ({number(drifts[0]['total_variation_M0'])} to {number(drifts[1]['total_variation_M0'])}); two finite comparisons do not establish asymptotic settling."]
        else:
            lines += ['Two complete adjacent-profile comparisons are unavailable; a decreasing late profile-drift trend is not established.']
    lines += ['', 'The outward case initially expanded and recollapsed; its small excursions near the expanded turning region are followed by renewed contraction and collapse. This one trial cannot establish an optimal kick or locate a stable branch. No extra amplitude trial or endpoint extension is inferred.', '',
        '## Resources, provenance and exact locations', '']
    ledger_path = EXT/'evidence/allocation_cost_ledger.json'
    assert ledger_path.exists(), 'Preserve the current all-allocation ledger before generating the report'
    if ledger_path.exists():
        ledger = load(ledger_path)
        sources.append(receipt(ledger_path))
        lines += [f"Preserved allocation ledger observed {ledger['scheduler_observed_utc']}: closed {ledger['closed_allocated_node_seconds']} node-seconds ({ledger['closed_allocated_node_hours']:.8g}nodeh); active observed {ledger['active_observed_node_seconds']} node-seconds; accrued {ledger['total_accrued_node_hours']:.8g}nodeh. Original requested reservations total {ledger['original_requested_node_seconds']} node-seconds. This snapshot is not final billing: later/current analysis, rendering, review and transfer accounting must be reconciled before final delivery. Failed/check/analysis/product allocations remain included; final supplemental review allocations must also be included.", '']
    lines += ['The hard50 allocated node-hours/350GiB new-storage caps include gates, failures, analysis and products. Whole-user AMD watchdog measurements warn at1.5TiB and stop growth at1.7TiB or projected1.9TiB; session checkpoint reserve48GiB and GPU-memory guards remain in force. Fresh allocated storage and final all-job accounting are required at delivery, not inferred from an old forecast.', '',
        'All production uses exclusive MI2508x nodes/eightMPI ranks, GNU12/OpenMPI4.1.8/CMake3.25.2/prun2.3/ROCm6.4.1/GFX90A, vader single-copy disabled, UCX_TLS=^rocm_ipc and UCX_MEMTYPE_CACHE=n. Native sources match corrected-profile2d1f2525; Kokkos6739bc623081648af9e752b616d9671527922cbf. Each allocation freezes its own workflow, source commit, environment, exact input, executable hashes and launch; newer report code does not relabel earlier numerical sources.', '',
        f'AMD campaign root: {EXT}', f'Perseus thin review root: {LOCAL}', f'This generation: {out}',
        f'Allocation provenance/logs: {EXT}/evidence/allocations/JOB_ID/', '',
        'Raw particles/cohorts, reduced scientific data, native logs, source and render frames remain on AMD. Only reports, figures/PDFs, movies, exact inputs and compact review/acceptance artifacts enter the one-way SHA-verified Perseus thin copy. Original700M and earlier evidence remain protected. Only inventoried/SHA/replacement-verified superseded Session10 intermediate recovery files rotate; scientific100M, initial, final and first-qualified-AH anchors are retained.', '']
    for d, marker in bound:
        cp = d['acceptance']['checkpoint']
        lines += [f"{d['case']['id']} raw/checkpoints: {d['root']}/runs/yc0500/", f"Reduced diagnostics: {d['red']}/", f"Exact endpoint checkpoint: {cp['path']}", f"Checkpoint bytes={cp['bytes']}; SHA256={cp['sha256']}", '']
    lines += ['Tables: window_statistics.csv, profile_window_coverage.csv and profile_drift.csv. REPORT_SOURCES.json binds the exact numerical/visual/summary and frozen metric inputs used by this narrative. Native stream definitions, local/global constraints, independent particle audits and final-field receipts remain in each CASE_NUMERICAL_ACCEPTANCE.json.', '',
        'Individual and synchronized movies show20,000 fixed original IDs in an xy projection, with colors fixed by initial coordinate radius and actual native-time labels. They are not full-population density movies. The synchronized movie holds a finished case at its last native state with an ENDED label; its time range does not extend that trajectory. An accepted individual movie may reuse identical encoded bytes only after source/checkpoint/movie/manifest hashes match.', '',
        '## Limits and required final acceptance', '',
        'Boundary effects in the retained+/-512M box remain unresolved. Source-time coupling is unchanged and there is no larger-box or resolution-convergence comparison. COM-coordinate profiles and flow, particle excision, growing horizon mass and late constraints limit interpretation. Neither survival nor a finite-window decrease alone proves asymptotic stability.', '',
        'This generated report is reviewable evidence, not final delivery acceptance. Numerical acceptance, actual PNG inspection, complete movie decoding, actual encoded-frame inspection, raster inspection of EVERY exported final PDF page, final costs/storage/source checks and one-way SHA-verified thin-copy acceptance must all be recorded. Automatic supervision remains open until those products are accepted.']
    atomic(out/'REPORT_SOURCES.json', dict(utc=now(), job=job(), validation_only=validation_only,
        sources=sources, case_keys=[d['case']['id'] for d in data],
        final_visual_accounting_storage_transfer_acceptance_pending=True))
    return lines


def render_pdf(lines, path, figures=()):
    """Bound line length and page height; actual PDF raster review stays separate."""
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    wrapped = '\n'.join('\n'.join(textwrap.wrap(line, 94)) if line else '' for line in lines).splitlines()
    pages = []
    with PdfPages(path) as pdf:
        for start in range(0, len(wrapped), 62):
            page = wrapped[start:start+62]
            assert all(len(line)<=94 for line in page)
            fig = plt.figure(figsize=(8.3, 11.7))
            fig.text(.055, .95, '\n'.join(page), va='top', fontsize=8, family='monospace')
            pdf.savefig(fig)
            plt.close(fig)
            pages.append(dict(page=len(pages)+1, kind='narrative', lines=len(page)))
        for source in figures:
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            ax.imshow(plt.imread(source))
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            pages.append(dict(page=len(pages)+1, kind='figure', source=receipt(source)))
        assert pdf.get_pagecount()==len(pages)
    atomic(path.with_name('REPORT_LAYOUT.json'), dict(utc=now(), job=job(),
        pdf=receipt(path), pages=pages, maximum_text_columns=94,
        maximum_text_lines_per_page=62, actual_PDF_raster_review_pending=True))

