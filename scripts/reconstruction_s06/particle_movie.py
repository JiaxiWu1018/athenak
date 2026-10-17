"""ID-stable x-y particle movie, coloured by exact initial coordinate radius.

The session5 trajectory cache supplies 0..300M; native session6 PVTK supplies
302..1000M. All extraction, hashing, rendering and encoding require Slurm.
"""
import argparse
import csv
import math
import os
from pathlib import Path
import socket
import subprocess

import numpy as np

from session import ROOT, CASE, N, P, atomic, load, now, require_job, sample, sha
from reduce_run import read_pvtk

OLD = ROOT.parent / '05_sampling_shell_response_20261014'
WORK = ROOT / 'particle_movie_20261017'
DATA = WORK / 'trajectories.npz'
RECEIPT = WORK / 'DATA_OK.json'
FPS = 15
STEM = 's6_yc0819_0_1000_particles_xy_initial_radius'
TARGET = ROOT / 'movies' / (STEM + '.mp4')


def identity():
    repo = Path(__file__).resolve().parents[2]
    return dict(utc=now(), job=os.environ['SLURM_JOB_ID'], host=socket.gethostname(),
                source_commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
                script_sha256=sha(Path(__file__)))


def entry(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def verified(path, row):
    assert path.stat().st_size == row['bytes'], path
    assert sha(path) == row['sha256'], path


def extract():
    assert not RECEIPT.exists() and not DATA.exists(), 'Preserve existing extraction'
    WORK.mkdir(parents=True, exist_ok=True)
    old_receipt = OLD / 'particle_visualization_data/464699/DATA_OK.json'
    old = load(old_receipt)
    old_cache = OLD / old['cache']['path']
    verified(old_cache, old['cache'])
    initial_source = next(r for r in old['initial_sources'] if r['case'] == CASE)
    initial_path = ROOT / 'samples' / (CASE + '.bin')
    verified(initial_path, initial_source)
    _, initial = sample(initial_path)
    assert len(initial) == N and np.array_equal(initial[:, 0], np.arange(N))
    with np.load(old_cache, allow_pickle=False) as cache:
        assert cache['movie_cases'].tolist() == [CASE, 'legacy_stratified_s1']
        ids = cache['display_ids'].copy()
        expected_ids = np.sort(np.random.default_rng(2026101505).choice(N, 20000, replace=False))
        assert np.array_equal(ids, expected_ids)
        clouds = list(cache['movie_clouds'][0].copy())
        times = cache['times'][0].tolist()
    assert len(clouds) == 151 and np.array_equal(times, np.arange(0, 301, 2))
    assert np.array_equal(clouds[0], initial[ids, 1:4].astype(np.float32))
    r0 = np.linalg.norm(initial[ids, 1:4], axis=1)
    initial_surface = 8.936231120010849
    assert np.all((r0 > 0) & (r0 < initial_surface))
    manifest_path = ROOT / 'evidence/PERSEUS_ANALYSIS_STAGE.json'
    expected = {r['path']: r for r in load(manifest_path)['files']}
    paths = sorted((ROOT / 'runs/production/out/pvtk').glob('*.part.vtk'))
    assert len(paths) == 350
    sources = []
    for k, path in enumerate(paths):
        key = str(path.relative_to(ROOT))
        verified(path, expected[key])
        time, cycle, count, fields = read_pvtk(str(path))
        assert count == N and math.isfinite(time) and time > times[-1]
        tags_float = fields['ptag']
        tags = tags_float.astype(np.int64)
        assert np.array_equal(tags_float, tags)
        assert np.all((tags >= 0) & (tags < N))
        assert np.all(np.bincount(tags, minlength=N) == 1)
        assert np.array_equal(fields['prtcl_mass'], initial[tags, 7].astype(np.float32).astype(np.float64))
        positions = np.column_stack([fields[c] for c in 'xyz'])
        assert np.isfinite(positions).all()
        index = np.empty(N, dtype=np.int32)
        index[tags] = np.arange(N, dtype=np.int32)
        clouds.append(positions[index[ids]].astype(np.float32))
        times.append(time)
        sources.append(dict(expected[key], time_M=time, cycle=cycle))
        if k % 25 == 0:
            print('EXTRACT', k, time, flush=True)
    clouds = np.asarray(clouds)
    times = np.asarray(times)
    with open(ROOT / 'reduced/shells.csv') as stream:
        history_times = np.array([float(row['time']) for row in csv.DictReader(stream)])
    assert clouds.shape == (501, 20000, 3) and np.isfinite(clouds).all()
    assert np.allclose(times, history_times, atol=1e-8, rtol=0)
    assert times[0] == 0 and times[150] == 300 and times[-1] == 1000
    max_xy = float(np.abs(clouds[:, :, :2]).max())
    limit = max(12.0, 2 * math.ceil((max_xy + 0.5) / 2))
    np.savez_compressed(DATA, ids=ids, positions=clouds, times=times, initial_radius=r0)
    atomic(RECEIPT, dict(identity(), cache=entry(DATA), original_cache_receipt=entry(old_receipt),
        original_cache=entry(old_cache), initial_sample=entry(initial_path),
        native_manifest=entry(manifest_path), native_sources=sources,
        frames=501, display_count=20000, original_count=N, display_id_rng_seed=2026101505,
        time_start_M=0, time_end_M=1000, times_M=times.tolist(),
        xyz_component_max_M=np.abs(clouds).max(axis=(0, 1)).tolist(), max_abs_xy_M=max_xy,
        view_limit_M=limit, clipped_display_particles=0,
        initial_radius_min_M=float(r0.min()), initial_radius_max_M=float(r0.max()),
        color_range_M=[0, initial_surface],
        radius_definition='sqrt(x0^2+y0^2+z0^2) from exact t=0 sample; coordinate/isotropic radius about fixed origin',
        projection='x-y positions through all z; no slice, recentering, interpolation or time-dependent resampling',
        continuity='Identical original IDs throughout; exact t=0 position equality and all501 native history times checked',
        old_segment='151 source-hashed session5 fixed-ID trajectory states, 0..300M',
        new_segment='350 source-hashed native session6 PVTK states, after300..1000M'))
    print('DATA_OK', DATA, 'view', limit, 'max_xy', max_xy, flush=True)


def render(preview):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from PIL import Image

    receipt = load(RECEIPT)
    verified(DATA, receipt['cache'])
    with np.load(DATA, allow_pickle=False) as cache:
        positions = cache['positions']
        times = cache['times']
        r0 = cache['initial_radius']
        ids = cache['ids']
    assert positions.shape == (501, 20000, 3) and len(np.unique(ids)) == 20000
    limit = receipt['view_limit_M']
    assert np.abs(positions[:, :, :2]).max() < limit
    dest = WORK / ('preview_' if preview else 'render_')
    dest = dest.with_name(dest.name + os.environ['SLURM_JOB_ID'])
    dest.mkdir(parents=True, exist_ok=False)
    frames = dest / 'frames'
    frames.mkdir()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14,
                         'text.color': '#24313a', 'axes.labelcolor': '#24313a'})
    fig = plt.figure(figsize=(10, 9), dpi=120, facecolor='white')
    ax = fig.add_axes([0.12, 0.14, 0.70, 0.73])
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect='equal',
           xlabel=r'$x/M$', ylabel=r'$y/M$')
    ax.set_xticks(np.arange(-limit, limit + 0.1, 4))
    ax.set_yticks(np.arange(-limit, limit + 0.1, 4))
    ax.grid(color='#e8ebed', linewidth=0.6, zorder=0)
    for spine in ax.spines.values():
        spine.set_color('#7b858b')
    # A fixed random draw order avoids systematic occlusion by initial radial rank.
    order = np.random.default_rng(2026101706).permutation(len(ids))
    norm = Normalize(*receipt['color_range_M'])
    dots = ax.scatter(positions[0, order, 0], positions[0, order, 1], c=r0[order],
                      s=1.6, cmap='viridis', norm=norm, linewidths=0, alpha=1, zorder=2)
    cb = fig.colorbar(dots, cax=fig.add_axes([0.86, 0.25, 0.025, 0.50]))
    cb.set_label(r'Initial radius $r_0/M$', labelpad=12)
    cb.set_ticks([0, 2, 4, 6, 8, receipt['color_range_M'][1]])
    cb.set_ticklabels(['0', '2', '4', '6', '8', '8.94'])
    fig.text(0.5, 0.965, r'ST $y_c=0.819$ | proper profile, stratified seed 1',
             ha='center', fontsize=17)
    clock = fig.text(0.47, 0.912, '', ha='center', fontsize=16)
    fig.text(0.5, 0.061, '20,000 fixed particles out of 1,500,000 | x–y projection through all z',
             ha='center', fontsize=11)
    fig.text(0.5, 0.032, 'Colour stays fixed at the initial 3D radius | fixed origin and axes',
             ha='center', fontsize=11)
    picks = sorted(set(int(np.argmin(np.abs(times - t))) for t in [0, 300, 600, 800, 838, 1000]))
    use = picks if preview else range(len(times))
    for k in use:
        dots.set_offsets(positions[k, order, :2])
        clock.set_text(f't = {times[k]:.2f} M    |    {times[k]/P:.3f} surface periods')
        fig.savefig(frames / f'frame_{k:04d}.png', dpi=120)
        if k % 25 == 0:
            print('RENDER', k, times[k], flush=True)
    plt.close(fig)
    result = dict(identity(), preview_only=preview, extraction_receipt=entry(RECEIPT),
                  numpy=np.__version__, matplotlib=matplotlib.__version__,
                  frame_pixels=[1200, 1080], point_area_pt2=1.6, alpha=1,
                  draw_order_seed=2026101706, colormap='viridis',
                  color_range_M=receipt['color_range_M'], view_limit_M=limit,
                  frames=501, fps=FPS, nominal_duration_s=501/FPS)
    if not preview:
        TARGET.parent.mkdir(exist_ok=True)
        assert not TARGET.exists(), 'Preserve existing movie'
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-framerate', str(FPS),
                        '-i', str(frames / 'frame_%04d.png'), '-c:v', 'libx264',
                        '-threads', '2', '-preset', 'medium', '-crf', '18',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(TARGET)], check=True)
        import json
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
            '-select_streams', 'v:0', '-show_entries',
            'stream=codec_name,width,height,nb_read_frames,r_frame_rate,pix_fmt:format=duration',
            '-of', 'json', str(TARGET)], text=True))
        stream = probe['streams'][0]
        assert int(stream['nb_read_frames']) == 501 and stream['r_frame_rate'] == '15/1'
        assert (stream['width'], stream['height']) == (1200, 1080)
        assert stream['codec_name'] == 'h264' and stream['pix_fmt'] == 'yuv420p'
        assert abs(float(probe['format']['duration']) - 501/FPS) < 0.02
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '2', '-i', str(TARGET),
                        '-f', 'null', '-'], check=True)
        checks = dest / 'decoded'
        checks.mkdir()
        expr = '+'.join(f'eq(n\\,{k})' for k in picks)
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '2', '-i', str(TARGET),
                        '-vf', 'select=' + expr, '-vsync', '0', str(checks / 'decoded_%02d.png')], check=True)
        images = sorted(checks.glob('*.png'))
        assert len(images) == len(picks)
        result.update(movie=entry(TARGET), ffprobe=probe, full_decode_passed=True,
                      ffmpeg=subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0])
    else:
        images = [frames / f'frame_{k:04d}.png' for k in picks]
    sheet = Image.new('RGB', (1200, 1620), 'white')
    for j, path in enumerate(images):
        with Image.open(path) as im:
            sheet.paste(im.convert('RGB').resize((600, 540), Image.Resampling.LANCZOS),
                        (600*(j % 2), 540*(j // 2)))
    sheet.save(dest / 'contact_sheet.png')
    result['review_frames'] = [dict(frame=k, time_M=float(times[k])) for k in picks]
    result['contact_sheet'] = entry(dest / 'contact_sheet.png')
    atomic(dest / 'RENDER_OK.json', result)
    atomic(WORK / ('PREVIEW_OK.json' if preview else 'RENDER_OK.json'), result)
    print('RENDER_OK', dest, flush=True)


def main():
    require_job()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['extract', 'preview', 'render'])
    args = parser.parse_args()
    if args.mode == 'extract':
        extract()
    else:
        render(args.mode == 'preview')


if __name__ == '__main__':
    main()
