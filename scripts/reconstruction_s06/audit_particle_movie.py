"""Freeze the movie addendum without replacing the original science manifest."""
import re
import subprocess

from session import ROOT, PHYSICS, atomic, load, now, require_job, sha
from particle_movie import DATA, WORK, RECEIPT, TARGET, entry, identity, verified


def main():
    require_job()
    destination = ROOT / 'evidence/PARTICLE_MOVIE_ADDENDUM.json'
    assert not destination.exists(), 'Preserve existing final movie addendum'
    data = load(RECEIPT)
    render = load(WORK / 'RENDER_OK.json')
    review = load(ROOT / 'evidence/MOVIE_VISUAL_REVIEW.json')
    source = load(ROOT / 'evidence/MOVIE_SOURCE.json')
    repo = ROOT / 'src/athenak'
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    assert head == source['head'] == source['remote_head_verified']
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True)
    subprocess.run(['git', '-C', str(repo), 'diff', '--exit-code', PHYSICS, 'HEAD', '--', 'src'], check=True)
    assert data['frames'] == render['frames'] == 501 and render['full_decode_passed']
    assert review['accepted'] and review['source'] == 'actual frames decoded from the encoded MP4'
    assert data['clipped_display_particles'] == 0 and render['view_limit_M'] == 16
    verified(DATA, data['cache'])
    verified(TARGET, render['movie'])
    verified(RECEIPT, render['extraction_receipt'])
    verified(WORK / 'render_15019/contact_sheet.png', render['contact_sheet'])
    for job in ['15017', '15019']:
        terminal = (ROOT / f'evidence/jobs/{job}/scontrol_terminal.txt').read_text()
        assert 'JobState=COMPLETED' in terminal and 'ExitCode=0:0' in terminal
    frames = sorted((WORK / 'render_15019/frames').glob('frame_*.png'))
    assert len(frames) == 501
    assert [p.name for p in frames] == [f'frame_{k:04d}.png' for k in range(501)]
    assert len(list((WORK / 'render_15019/decoded').glob('decoded_*.png'))) == 6
    storage = []
    for path in [WORK, TARGET.parent]:
        files = [p for p in path.rglob('*') if p.is_file()]
        storage.append(dict(path=str(path), files=len(files),
                            logical_bytes=sum(p.stat().st_size for p in files),
                            allocated_bytes=int(subprocess.check_output(['du', '-s', '-B1', str(path)], text=True).split()[0])))
    storage_doc = ROOT / 'reports/MOVIE_STORAGE.md'
    rows = ['# Movie addendum storage', '', 'All movie products are retained on Perseus. No deletion or transfer occurred.', '',
            '| Path | Files | Logical bytes | Allocated bytes |', '|---|---:|---:|---:|']
    rows.extend(f"| `{r['path']}` | {r['files']} | {r['logical_bytes']} | {r['allocated_bytes']} |" for r in storage)
    rows.extend(['', 'The work directory retains the verified fixed-ID trajectory cache, extraction/render receipts, all501 PNG frames, six encoded-frame checks and preview/contact sheets. The movies directory retains the final H.264 MP4. Source native diagnostics and checkpoints remain at the paths recorded in REPORT_session6.md; this addendum did not change their retention.', ''])
    atomic(storage_doc, '\n'.join(rows))
    documents = [ROOT / p for p in ['README.md', 'reports/REPORT_session6.md', 'reports/FIGURES.md',
        'reports/MOVIE_session6.md', 'reports/MOVIE_STORAGE.md']]
    links = []
    for doc in documents:
        for href in re.findall(r'\]\(([^)]+)\)', doc.read_text()):
            if href.startswith(('https:', 'http:', '#')):
                continue
            target = (doc.parent / href.split('#')[0]).resolve()
            assert target == destination or target.exists(), (doc, href)
            links.append(dict(document=str(doc.relative_to(ROOT)), href=href))
    paths = documents + [ROOT / p for p in ['evidence/MOVIE_SOURCE.json',
        'evidence/MOVIE_PREVIEW_REVIEW.json', 'evidence/MOVIE_VISUAL_REVIEW.json']]
    paths += [p for p in WORK.rglob('*') if p.is_file()]
    paths += [TARGET]
    paths += [p for p in (ROOT / 'evidence/before_movie_addendum_20261017').rglob('*') if p.is_file()]
    for job in ['15017', '15019']:
        paths += [ROOT / f'evidence/jobs/{job}/environment.txt', ROOT / f'evidence/jobs/{job}/scontrol_terminal.txt',
                  ROOT / f'evidence/jobs/{job}/submitted.sbatch', ROOT / f'logs/st_s06_movie.{job}.log']
    paths += [repo / f'scripts/reconstruction_s06/{name}' for name in
              ['particle_movie.py', 'particle_movie.sbatch', 'audit_particle_movie.py', 'README.md']]
    files = [dict(entry(p), relative_path=str(p.relative_to(ROOT))) for p in sorted(set(paths))]
    atomic(destination, dict(identity(), status='complete', final_source=head,
        render_source=render['source_commit'], frames=501, fps=15, duration_s=33.4,
        original_science_manifest='evidence/FINAL_DELIVERY_MANIFEST.json',
        preservation='Original delivery documents/manifest frozen under evidence/before_movie_addendum_20261017; this addendum hashes updated documents and new movie products separately',
        source_physics_diff_empty=True, complete_native_hashes_verified_at_extraction=True,
        actual_encoded_frames_visually_accepted=True, full_decode_passed=True,
        storage=storage, links=links, files=files,
        no_new_evolution=True, no_deletion=True, no_paper_suite_launch=True))
    print('PARTICLE_MOVIE_ADDENDUM_OK', len(files), 'files', len(links), 'links', storage, flush=True)


if __name__ == '__main__':
    main()
