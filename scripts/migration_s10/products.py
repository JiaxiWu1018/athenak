"""Final comparison figures, fixed-ID movies and source-bound Markdown/PDF."""
import csv
import shutil
import numpy as np
from common import *
from reduce import csv_rows, write_csv
from run_segment import cohort
from publication_style import style
from window_metrics import time_window, coverage, profile_windows

def resolve_prefix():
    """Resolve accepted old cohorts by exact native time/cycle, never redraw them."""
    prefix = EXT/'reference/control_0_700'
    marker = prefix/'RESOLVED.json'
    manifest=load(prefix/'SOURCE_MANIFEST.json')
    for entry in manifest['files']:
        p=prefix/Path(entry['copy']).name
        assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
    if marker.exists():
        for entry in load(marker)['original_cohorts']:
            p=Path(entry['source']['path'])
            assert p.stat().st_size==entry['source']['bytes'] and sha(p)==entry['source']['sha256']
        return prefix
    with (prefix/'cohort_index.csv').open() as f:
        rows = list(csv.DictReader(f))
    checked = []
    for row in rows:
        name = Path(row['file']).name
        candidates = [Path(row.get('source_file', row['file'])),
            Path('/work1/eliasmost/jiaxiwu/st_s08_scan_20261023/legacy500_packet/cohort')/name,
            DONOR/'runs/yc0500/cohort'/name,
            Path('/work1/eliasmost/jiaxiwu/st_s08_yc0500_extensions_20261023/long/runs/yc0500/cohort')/name]
        valid = []
        for p in candidates:
            if not p.exists():
                continue
            with p.open('rb') as f:
                header = f.readline().split()
            if header[0] == b'S8COHORT1' and abs(float(header[1])-float(row['time'])) < 1e-7 and int(header[2]) == int(float(row['cycle'])):
                valid.append(p)
        assert valid, ('Missing accepted original cohort', row)
        p = valid[0]
        row['file'] = str(p)
        checked.append(dict(time=float(row['time']), cycle=int(float(row['cycle'])), source=receipt(p)))
    write_csv(prefix/'cohort_index_resolved.csv', rows)
    atomic(marker, dict(utc=now(), job=job(), accepted_compact_sources=load(prefix/'SOURCE_MANIFEST.json'),
        original_cohorts=checked, no_historical_evolution=True))
    return prefix

def joined(prefix, tail, field):
    older = csv_rows(prefix/field)
    newer = csv_rows(tail/field)
    assert max(r['time'] for r in older) <= 700+1e-7
    assert min(r['time'] for r in newer) >= 700-1e-7
    result = {r['time']: r for r in older}
    result.update({r['time']: r for r in newer})
    return [result[t] for t in sorted(result)]

def source_data(case_keys=None):
    selected = [c for c in ALL_CASES if case_keys is None or c['id'] in case_keys]
    assert selected
    prefix = resolve_prefix() if any(c['start'] for c in selected) else None
    data = []
    for raw in selected:
        key = raw['id']
        root = EXT/key
        red = root/'reduced/yc0500'
        acceptance = load(root/'evidence/CASE_NUMERICAL_ACCEPTANCE.json')
        assert acceptance['numerical_checks_passed']
        summary = load(red/'summary.json')
        assert summary['complete']
        curves = {name: (joined(prefix, red, name+'.csv') if raw['start'] else csv_rows(red/(name+'.csv')))
                  for name in ['shells', 'history', 'mass', 'ah']}
        with (red/'cohort_index.csv').open() as f:
            indices = list(csv.DictReader(f))
        if raw['start']:
            with (prefix/'cohort_index_resolved.csv').open() as f:
                indices = list(csv.DictReader(f))+indices
        bytime = {float(r['time']): r for r in indices}
        indices = [bytime[t] for t in sorted(bytime)]
        data.append(dict(case=raw, root=root, red=red, summary=summary, curves=curves,
            cohorts=indices, acceptance=acceptance))
    return data

def figures(data, out):
    import matplotlib.pyplot as plt
    files = []
    colors = plt.cm.viridis(np.linspace(.08, .92, 9))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, layout='constrained')
    for col, d in enumerate(data):
        curves = d['curves']
        for q, color in zip(range(10, 100, 10), colors):
            axes[0, col].plot([r['time'] for r in curves['shells']], [r[f'r{q}'] for r in curves['shells']], c=color, label=f'{q}%')
        axes[0, col].set(yscale='log', ylabel='COM coordinate shell radius [M]', title=f"U={d['case']['U']:+g}")
        axes[0, col].legend(ncol=3, fontsize=7)
        axes[1, col].plot([r['time'] for r in curves['history']], [r['core_lapse'] for r in curves['history']])
        axes[1, col].set(xlabel='t / M_ref', ylabel='Mean core lapse')
        for ax in axes[:, col]:
            ax.set_xlim(0, 1500)
            ax.grid(alpha=.18)
        if d['case']['start']:
            for ax in axes[:, col]:
                ax.axvline(700, c='.4', ls=':', lw=.8)
    fig.suptitle('Corrected-profile yc=0.500: outward kick and unperturbed continuation')
    for ext in ['png', 'pdf']:
        p = out/f'shells_lapse.{ext}'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        files.append(p)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    for d, color in zip(data, ['#245d88', '#bd5e28']):
        label = f"U={d['case']['U']:+g}"
        h = d['curves']['history']
        axes[0, 0].semilogy([r['time'] for r in h], [r['H_rms_r10'] for r in h], c=color, label=label)
        mass = d['curves']['mass']
        axes[0, 1].plot([r['time'] for r in mass], [r['m_alive']/r['m_initial'] for r in mass], c=color, label=label)
        flow = d['red']/'coordinate_radial_flow.csv'
        if flow.exists():
            rows = csv_rows(flow)
            axes[1, 0].plot([r['time'] for r in rows], [r['mean_coordinate_dr_dt'] for r in rows], c=color, label=label)
        grid = d['red']/'checkpoint_mesh.csv'
        if grid.exists():
            rows = csv_rows(grid)
            axes[1, 1].step([r['time'] for r in rows], [r['finest_active_dx'] for r in rows], where='post', c=color, label=label)
    for ax, ylabel in zip(axes.flat, ['Local Hamiltonian RMS, r<10M', 'Alive rest mass / initial rest mass',
            'Fixed-ID mean coordinate dr/dt', 'Checkpoint-sampled finest active dx [M]']):
        ax.set(xlabel='t / M_ref', ylabel=ylabel, xlim=(0, 1500))
        ax.legend(fontsize=8)
        ax.grid(alpha=.18)
    for ext in ['png', 'pdf']:
        p = out/f'health_flow_mesh.{ext}'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        files.append(p)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for ax, d in zip(axes, data):
        z = np.load(d['red']/'coordinate_rest_mass_profiles.npz')
        centers = (z['edges'][1:]+z['edges'][:-1])/2
        volumes = 4*np.pi/3*np.diff(z['edges']**3)
        profiles, drift = profile_windows(z['time'], z['mass_per_bin'], d['summary']['rest_mass_initial'])
        for result, color in zip(profiles, ['#245d88', '#bd5e28', '#38815b']):
            if result['complete']:
                density = result['mean']/volumes
                ax.loglog(centers, np.where(density > 0, density, np.nan), c=color,
                    label=f'{result["start"]:g}–{result["end"]:g}M time mean')
        ax.set(xlim=(.05, 512), xlabel='COM coordinate radius [M]',
            ylabel='dM0 / coordinate shell volume [M^-2]', title=f"U={d['case']['U']:+g}")
        if any(result['complete'] for result in profiles):
            ax.legend(fontsize=8)
        else:
            ax.text(.5, .5, 'No complete late profile window', ha='center', transform=ax.transAxes)
    for ext in ['png', 'pdf']:
        p = out/f'late_profiles.{ext}'
        fig.savefig(p, dpi=200, bbox_inches='tight')
        files.append(p)
    plt.close(fig)
    from collapse_figure import render
    for d in data:
        record = render(d['root'], d['case']['id'], out)
        if record:
            files.extend(Path(entry['path']) for entry in record['plots'])
    return files

def movie(data, out, name, targets=None):
    import matplotlib.pyplot as plt
    exe=shutil.which('ffmpeg')
    if not exe:
        import imageio_ffmpeg
        exe=imageio_ffmpeg.get_ffmpeg_exe()
    endpoint = max(d['summary']['endpoint'] for d in data)
    targets = np.unique(np.r_[np.arange(0, endpoint+1e-8, 2.), endpoint]) if targets is None else np.asarray(targets,dtype=float)
    selected = []
    max_xy = 0.
    for d in data:
        times = np.array([float(v['time']) for v in d['cohorts']])
        chosen = []
        for t in targets:
            capped = min(t, d['summary']['endpoint'])
            k = int(np.argmin(abs(times-capped)))
            assert abs(times[k]-capped) <= .65
            chosen.append(d['cohorts'][k])
        selected.append(chosen)
        for row in {r['file']: r for r in chosen}.values():
            t, cycle, mask, values = cohort(row['file'])
            if mask.any():
                max_xy = max(max_xy, float(abs(values[mask == 1, 1:3]).max()))
    limit = max(6., max_xy*1.05)
    with (EXT/'protected/yc0500.bin').open('rb') as f:
        f.readline()
        initial = np.fromfile(f, dtype='<f8').reshape(-1, 8)[::100]
    radius0 = np.linalg.norm(initial[:, 1:4], axis=1)
    fig, axes = plt.subplots(1, len(data), figsize=(6*len(data), 6), squeeze=False, layout='constrained')
    axes = axes[0]
    artists = []
    labels = []
    for ax, d in zip(axes, data):
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect='equal', xlabel='x / M_ref', ylabel='y / M_ref')
        artists.append(ax.scatter([], [], c=[], s=.8, cmap='viridis', vmin=0, vmax=float(radius0.max()), linewidths=0))
        labels.append(ax.text(.02, .98, '', transform=ax.transAxes, va='top', fontsize=9))
        ax.set_title(f"U={d['case']['U']:+g}: 20,000 fixed original IDs")
    frame_manifest = []
    previews = out/(name+'_previews')
    previews.mkdir(exist_ok=True)
    frames=out/(name+'_frames')
    frames.mkdir(exist_ok=False)
    for i, t in enumerate(targets):
        sources = []
        for ax, d, rows, artist, label in zip(axes, data, selected, artists, labels):
            row = rows[i]
            actual, cycle, mask, values = cohort(row['file'])
            active = mask == 1
            artist.set_offsets(values[active, 1:3])
            artist.set_array(radius0[active])
            ended = bool(t > d['summary']['endpoint']+1e-8)
            ah = d['summary']['ah_first']
            label.set_text(f'target {t:g}M; native {actual:g}M\n'+
                ('ENDED; last state held\n' if ended else '')+
                (f'Accepted AH at {ah:g}M\n' if ah is not None and actual >= ah else '')+
                f'alive cohort {int(active.sum())}/20000')
            sources.append(dict(file=row['file'], time=actual, cycle=cycle, ended=ended))
        frame=frames/f'{i:06d}.png'
        fig.savefig(frame,dpi=130,facecolor='white')
        if i in {0, len(targets)//2, len(targets)-1}:
            shutil.copy2(frame,previews/f'frame_{i:05d}.png')
        frame_manifest.append(dict(frame=i, target_time=float(t), sources=sources,encoder_input=receipt(frame)))
    plt.close(fig)
    path = out/(name+'.mp4')
    encode=[exe,'-v','error','-xerror','-framerate','15','-i',frames/'%06d.png',
        '-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart','-threads','4',path]
    cmd(encode)
    decoded=subprocess.run([exe,'-v','error','-xerror','-threads','1','-i',str(path),
        '-progress','pipe:1','-nostats','-f','null','-'],capture_output=True,text=True)
    assert decoded.returncode==0 and not decoded.stderr,decoded.stderr
    probe=dict(line.split('=',1) for line in decoded.stdout.splitlines() if '=' in line)
    assert probe.get('progress')=='end' and int(probe['frame'])==len(targets),probe
    selection = '+'.join(r'eq(n\,'+str(i)+')' for i in sorted({0, len(targets)//2, len(targets)-1}))
    cmd([exe, '-v', 'error','-xerror','-threads','1', '-i', path, '-vf', 'select='+selection,
         '-vsync', 'vfr','-threads','1', previews/'encoded_%03d.png'], stdout=subprocess.DEVNULL)
    atomic(out/(name+'_manifest.json'), dict(utc=now(), job=job(), movie=receipt(path), frames=frame_manifest,
        full_decode_passed=True, probe=probe,encoder=receipt(exe),encode_command=list(map(str,encode)),xy_limit=limit, color='Initial coordinate radius; fixed across all frames',
        actual_encoded_frame_visual_review=False, scope='Fixed-ID cohort projection; not a full-population density movie'))
    return path


def case_movie_only():
    """Create an independently reviewable movie after that case is accepted."""
    datum, = source_data([CASE['key']])
    state = load(datum['root']/'runs/yc0500/state.json')
    assert state['complete'] and state['latest']['payload_verified']
    assert state['latest']['sha256'] == datum['acceptance']['checkpoint']['sha256']
    assert abs(float(os.environ['S10_TARGET'])-datum['summary']['endpoint']) < 1e-7
    visual_path = datum['root']/'evidence/CASE_VISUAL_ACCEPTANCE.json'
    visual = load(visual_path)
    assert visual['actual_visual_review']
    assert abs(visual['endpoint']-datum['summary']['endpoint']) < 1e-7
    out = EXT/'delivery'/f"case_movie_{CASE['key']}_{job()}"
    assert not out.exists(), 'Preserve earlier product attempts'
    out.mkdir(parents=True)
    path = movie([datum], out, CASE['key'])
    manifest_path = out/(CASE['key']+'_manifest.json')
    manifest = load(manifest_path)
    assert manifest['full_decode_passed']
    used = sorted({row['file'] for frame in manifest['frames'] for row in frame['sources']})
    atomic(out/'CASE_MOVIE_GENERATION_OK.json', dict(utc=now(), job=job(),
        case=CASE['key'], endpoint=datum['summary']['endpoint'],
        checkpoint_sha256=datum['acceptance']['checkpoint']['sha256'],
        numerical_source=receipt(datum['root']/'evidence/CASE_NUMERICAL_ACCEPTANCE.json'),
        accepted_case_visual_source=receipt(visual_path), movie=receipt(path),
        manifest=receipt(manifest_path), cohort_sources=[receipt(Path(p)) for p in used],
        full_decode_passed=True, actual_encoded_frame_visual_review_pending=True,
        final_campaign_delivery_complete=False,
        scope='One accepted case movie only. Other case evolution and final comparisons remain separate.'))
    print(out)


def individual_movie(datum, out):
    """Reuse a source-matched, actually inspected case movie when available."""
    key = datum['case']['id']
    accepted_path = EXT/'delivery/accepted_case_movies'/(key+'.json')
    if not accepted_path.exists():
        return movie([datum], out, key)
    accepted = load(accepted_path)
    assert accepted['actual_encoded_frame_visual_review'] and accepted['full_decode_passed']
    assert accepted['checkpoint_sha256'] == datum['acceptance']['checkpoint']['sha256']
    current = receipt(datum['root']/'evidence/CASE_NUMERICAL_ACCEPTANCE.json')
    assert current['sha256'] == accepted['numerical_source']['sha256']
    for entry in [accepted['movie'], accepted['manifest']]:
        assert receipt(Path(entry['path']))['sha256'] == entry['sha256']
    manifest = load(Path(accepted['manifest']['path']))
    assert manifest['full_decode_passed']
    assert manifest['movie']['sha256'] == accepted['movie']['sha256']
    destination = out/(key+'.mp4')
    assert not destination.exists()
    shutil.copy2(accepted['movie']['path'], destination)
    assert receipt(destination)['sha256'] == accepted['movie']['sha256']
    shutil.copy2(accepted['manifest']['path'], out/(key+'_manifest.json'))
    atomic(out/(key+'_REUSED.json'), dict(utc=now(), job=job(),
        acceptance=receipt(accepted_path), copied_movie=receipt(destination),
        original_generation=accepted, no_rerender=True,
        scope='Identical encoded bytes with matching accepted numerical source and checkpoint.'))
    return destination

def window_tables(data, out):
    statistics = []
    profile_coverage = []
    profile_drift = []
    for d in data:
        for lo, hi in [(500, 1000), (1000, 1500)]:
            rows = d['curves']['shells']
            for q in [10, 50, 90]:
                result = time_window([r['time'] for r in rows], [r[f'r{q}'] for r in rows], lo, hi)
                statistics.append(dict(case=d['case']['id'], shell=q, **coverage(result),
                    mean=float(result['mean']) if result['complete'] else None,
                    peak_to_peak=float(result['peak_to_peak']) if result['complete'] else None))
        z = np.load(d['red']/'coordinate_rest_mass_profiles.npz')
        profiles, drift = profile_windows(z['time'], z['mass_per_bin'], d['summary']['rest_mass_initial'])
        profile_coverage.extend(dict(case=d['case']['id'], **coverage(r)) for r in profiles)
        profile_drift.extend(dict(case=d['case']['id'], **r) for r in drift)
    write_csv(out/'window_statistics.csv', statistics)
    write_csv(out/'profile_window_coverage.csv', profile_coverage)
    write_csv(out/'profile_drift.csv', profile_drift)
    return statistics, profile_coverage, profile_drift


def main():
    job()
    from extension_resources import admission
    admission()
    style()
    if os.environ.get('S10_CASE_ONLY') == '1':
        case_movie_only()
        return
    data = source_data()
    out = EXT/'delivery'/('generated_'+job())
    assert not out.exists(), 'Preserve prior products; use a new reviewed generation directory for a retry'
    out.mkdir(parents=True)
    files = figures(data, out)
    statistics, profile_coverage, profile_drift = window_tables(data, out)
    from scientific_report import build_lines, render_pdf
    lines = build_lines(data, statistics, profile_coverage, profile_drift, out)
    report = out/'REPORT_migration10.md'
    report.write_text('\n'.join(lines)+'\n')
    render_pdf(lines, out/'REPORT_migration10.pdf', [p for p in files if p.suffix == '.png'])
    for d in data:
        individual_movie(d, out)
    movie(data, out, 'comparison')
    atomic(out/'GENERATION_OK.json', dict(utc=now(), job=job(),
        sources=[d['acceptance'] for d in data], products=[receipt(p) for p in sorted(out.iterdir()) if p.is_file()],
        final_visual_review_pending=True, final_accounting_and_storage_pending=True,
        final_thin_mirror_pending=True))
    if shutil.which('pdftoppm'):
        cmd(['pdftoppm', '-png', '-r', '90', out/'REPORT_migration10.pdf', out/'report_page'], stdout=subprocess.DEVNULL)
    else:
        atomic(out/'PDF_RASTER_REVIEW_PENDING.json',dict(reason='pdftoppm absent from AMD PATH',
            required_next_step='Render this exact PDF in an allocated Perseus review job and inspect every exported page',pdf=receipt(out/'REPORT_migration10.pdf')))

if __name__ == '__main__':
    main()
