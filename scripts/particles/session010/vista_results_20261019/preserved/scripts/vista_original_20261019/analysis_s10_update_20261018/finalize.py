#!/usr/bin/env python3
"""Finalize the local index only after explicit review of the bound QA images."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from run import BASE, OUT, SOURCE, atomic, check_time, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reviewed-by', required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'gg':
        raise RuntimeError('Finalize hashes and review records inside an allocated GG job')
    check_time()
    verification = json.loads((OUT / 'VERIFICATION.json').read_text())
    sources = json.loads((OUT / 'qa/REVIEW_SOURCES.json').read_text())
    if not verification['passed'] or not verification.get('all_pdfs_rasterized'):
        raise RuntimeError('Machine verification must pass before visual sign-off')
    for row in sources['pdf_previews']:
        if digest(OUT / row['pdf']) != row['pdf_sha256'] or digest(OUT / row['path']) != row['preview_sha256']:
            raise RuntimeError('PDF review images are stale')
    for row in sources['decoded_frames']:
        if digest(OUT / row['movie']) != row['movie_sha256'] or digest(OUT / row['path']) != row['preview_sha256']:
            raise RuntimeError('Decoded movie review images are stale')
    for row in verification['movies']:
        if digest(OUT / row['path']) != row['sha256']:
            raise RuntimeError('A movie changed after full decoding')
    reviewed = dict(reviewed_by=args.reviewed_by, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    passed=True, inspected=['All seven exported PDF figure previews',
                      'Original PNG figures', 'First, middle and final decoded frames of all three movies',
                      'Grayscale figure review sheet'],
                    checks=['Readable labels and no clipped annotations', 'Fixed camera and color scales',
                      'Correct final recorded timestamps', 'Original component identities and removal caveat',
                      'No strain or ringdown claim'],
                    limits=['Not a convergence study', 'Color-vision-deficiency simulation not performed'],
                    review_sources_sha256=digest(OUT / 'qa/REVIEW_SOURCES.json'))
    atomic(OUT / 'VISUAL_REVIEW.json', reviewed)
    verification['visual_review_pending'] = False
    verification['visual_review_receipt'] = 'VISUAL_REVIEW.json'
    atomic(OUT / 'VERIFICATION.json', verification)
    summary = json.loads((OUT / 'REDUCTIONS.json').read_text())
    titles = ['Orbit before common horizon', 'Accepted horizons', 'Matter accounting', 'Matter angular momentum',
              'Collapse snapshots', 'Numerical health and AMR', 'Raw gravitational waves']
    figures = sorted(OUT.glob('[0-9][0-9]_*.png'))
    lines = ['# Session010 analysis through the second segment', '',
             f'Completed data through **{summary["cutoff"]:.5f} Mref**. Seven figures and three movies are ready; input files are checksum-sealed, particle ledgers balance, and PDF/video checks passed.', '',
             '## Figures', '', '| Figure | PNG | PDF |', '|---|---|---|']
    for title, path in zip(titles, figures):
        lines.append(f'| {title} | [View]({path.name}) | [PDF]({path.with_suffix(".pdf").name}) |')
    lines += ['', '## Movies', '',
              '- [Matter evolution](matter_evolution.mp4): wide orbital, central orbital and vertical energy slices.',
              '- [Collapse and AMR health](collapse_amr_health.mp4): synchronized matter energy, lapse, raw Hamiltonian constraint and refinement level.',
              '- [3D particles](particles_3d.mp4): fixed original particle identities, fixed cameras and envelope/clump colors.', '',
              'Movies run at 2 Mref per playback second. They hold recorded snapshots between saved times; they do not invent intermediate motion. Each final saved state is included.', '',
              '## Reading the results', '',
              '[Interpretation and numerical tables](RESULTS.md) · [PDF overview](qa/all_pdf_figures.png) · [Decoded movie overview](qa/all_movies_decoded.png)', '',
              f'A common horizon enclosing both tracked objects is first accepted at {summary["first_common_enclosure_time"]:.5f} Mref. Binary orbit curves end there; coincident post-merger trackers are not two orbiting holes. The last accepted common-surface measurement is at {summary["horizons"]["2"]["last_time"]:.5f} Mref: later candidates fail the configured horizon-quality gate. Late remnant mass and spin are therefore not reliable measurements. Coordinate motion alone does not establish radiation-driven inspiral.', '',
              'White regions in the energy images are nonpositive sampled E, not missing geometry. Fine-grid particle deposition can look grainy. The 3D movie shows surviving particles only, so a disappearing clump sample does not mean its black hole has disappeared.', '',
              'Raw rPsi4 is saved at r=40 Mref. Strain, spectra, ringdown, equilibrium and convergence claims remain deferred. Dotted vertical guides mark application restarts; dashed guides mark the first accepted individual horizons. In particle-ledger plots, points are saved snapshots and connecting lines are guides.', '',
              '## Reduced tables', '',
              '[Orbit](orbit.csv) · [Original left surface](accepted_horizon_0.csv) · [Original right surface](accepted_horizon_1.csv) · [Common surface](accepted_horizon_2.csv) · [Common enclosure checks](common_enclosure.json) · [Matter components](matter_components.csv) · [Matter near R40](matter_near_r40.csv) · [Raw quadrupole modes](raw_quadrupole.csv) · [AMR history](amr_history.csv)', '',
              '## Integrity and reproduction', '',
              '[Input inventory](INPUTS.json) · [Checksum seals](SEALED.json) · [Verification](VERIFICATION.json) · [Visual review](VISUAL_REVIEW.json) · [Provenance](PROVENANCE.json)', '',
              '[Restart and simulation health](HEALTH_AND_CONTINUATION.md) · [Native checkpoint audit](RESTART_HEALTH.json)', '',
              'Run `bash analysis_s10_update_20261018/run.sh` inside an allocated Vista GG job to reproduce/resume the pipeline. No simulation was changed or submitted, and no raw data or checkpoints were deleted or transferred.']
    (OUT / 'INDEX.md').write_text('\n'.join(lines) + '\n')
    provenance = json.loads((OUT / 'PROVENANCE.json').read_text())
    provenance['analysis_scripts'] = {str(p.relative_to(BASE)): digest(p) for p in sorted(Path(__file__).parent.iterdir()) if p.is_file()}
    provenance['source_status'] = subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True)
    if provenance['source_status']:
        raise RuntimeError('Native source tree is no longer clean')
    atomic(OUT / 'PROVENANCE.json', provenance)
    promoted = [p for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in
                ('ARTIFACTS.json', 'ATTEMPT.json', 'PROGRESS.json', 'LAST_FAILURE.json')]
    promoted += [p for p in sorted((OUT / 'qa').rglob('*')) if p.is_file()]
    atomic(OUT / 'ARTIFACTS.json', dict(scientific_completion=False,
           files=[dict(path=str(p.relative_to(OUT)), bytes=p.stat().st_size, sha256=digest(p)) for p in promoted]))
    print('FINALIZED', OUT / 'INDEX.md')


if __name__ == '__main__':
    main()
