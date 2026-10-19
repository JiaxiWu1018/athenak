"""Allocated, isolated COM-shell preview from closed native output slots."""
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import analyze as reference
import reduce_run as native
from cfg import ROOT, AMD, P, PHYSICS, EXE_SHA, now, load, atomic, sha, require_job

PIN = Path(__file__).resolve().parent/'INTERIM_SOURCE_PIN.json'
QUANTILES = np.array([.1, .5, .9])
CASES = [(4, '#009e73', '-.'), (8, '#cc79a7', ':')]
RANKS = {}
WEIGHTS = {}


def initialize():
    for million, _, _ in CASES:
        n = million*1000000
        path = ROOT/'samples'/f'mc_{million}m_s1.bin'
        with path.open('rb') as stream:
            stream.readline()
            row = np.fromfile(stream, dtype='<f8', count=8)
        weight = float(np.float32(row[7]))
        cumulative = np.cumsum(np.full(n, weight))
        cumulative /= cumulative[-1]
        RANKS[n] = np.searchsorted(cumulative, QUANTILES)
        WEIGHTS[n] = weight


def reduce_frame(task):
    million, slot, filename = task
    n = million*1000000
    path = Path(filename)
    before = path.stat()
    t, cycle, count, data = native.read_pvtk(str(path))
    assert count == n and abs(t-2*slot) <= .25
    assert all(np.isfinite(v).all() for v in data.values())
    tags = data['ptag'].astype(np.int64)
    assert np.array_equal(tags, data['ptag']) and np.all((tags >= 0) & (tags < n))
    assert np.all(np.bincount(tags, minlength=n) == 1)
    mass = data['prtcl_mass']
    assert np.all(mass == WEIGHTS[n])
    pos = np.column_stack([data[c] for c in 'xyz'])
    center = np.average(pos, weights=mass, axis=0)
    radii = np.linalg.norm(pos-center, axis=1)
    # Equal native masses give exactly the reference cumulative-mass ranks.
    # Partition selects these order statistics without sorting unused ranks.
    radii.partition(RANKS[n])
    values = radii[RANKS[n]]
    row = dict(slot=slot, time=t, cycle=cycle, particles=n,
               com_x=float(center[0]), com_y=float(center[1]), com_z=float(center[2]),
               com_mag=float(np.linalg.norm(center)), rest_mass=float(mass.sum()),
               **{f'r{q}': float(v) for q, v in zip((10, 50, 90), values)})
    checked = slot in (0, 100, 200)
    if checked:
        original, _, _, _ = reference.sample_state(pos, mass, None, 'proper')
        for key in ('com_x', 'com_y', 'com_z', 'com_mag', 'r10', 'r50', 'r90'):
            assert original[key] == row[key], (million, slot, key, original[key], row[key])
    digest = sha(path)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    source = dict(path=str(path), bytes=after.st_size, sha256=digest, time_M=t,
                  slot=slot, reference_full_sort_checked=checked)
    return million, row, source


def plot(rows, dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pypdfium2 as pdfium
    plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': .18,
                         'savefig.dpi': 180})
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, layout='constrained')
    for j, q in enumerate((10, 50, 90)):
        for million, color, style in CASES:
            t = np.array([r['time'] for r in rows[million]])
            v = np.array([r[f'r{q}'] for r in rows[million]])
            axes[j, 0].plot(t, v, color=color, ls=style, lw=1.7,
                            label=f'MC {million} million')
            axes[j, 1].plot(t, 100*(v/v[0]-1), color=color, ls=style, lw=1.7)
        axes[j, 0].set_ylabel(f'$r_{{{q}}}/M$')
        axes[j, 1].set_ylabel('Change from initial [%]')
        axes[j, 1].axhline(0, color='#666666', lw=.6, zorder=0)
    axes[0, 0].set_title('Enclosed rest-mass radius about COM')
    axes[0, 1].set_title('Fractional shell change')
    axes[0, 0].legend(loc='best', framealpha=.9)
    for ax in axes.ravel():
        ax.set_xlim(0, 400)
        ax.set_xticks([0, 100, 200, 300, 400])
    for ax in axes[-1]:
        ax.set_xlabel('$t/M$')
    for ax in axes[0]:
        ax.secondary_xaxis('top', functions=(lambda t: t/P, lambda p: p*P)).set_xlabel('Surface periods')
    fig.suptitle('ST session 7 | COM-centred mass shells | interim 0–400M\n'
                 '$y_c=0.819$ · proper profile · nested MC seed 1 · fixed mesh', fontsize=14)
    fig.supxlabel('All particles; every native 2M output. 1M/2M runs not yet available. Each row has its own y-scale.', fontsize=10)
    png, pdf = dest/'shells_com_to400M.png', dest/'shells_com_to400M.pdf'
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    document = pdfium.PdfDocument(str(pdf))
    assert len(document) == 1
    document[0].render(scale=1.5).to_pil().save(dest/'shells_com_to400M_pdf.png')
    document.close()


def main():
    require_job()
    assert ROOT == AMD
    started = time.monotonic()
    pin = load(PIN)
    for item in pin['files']:
        assert sha(PIN.parent/item['path']) == item['sha256']
    assert load(ROOT/'evidence/SOURCE_PIN.json')['head'] == pin['native_runtime_source']
    dest = ROOT/'interim'/('com_shells_to400M_'+pin['head'][:8])
    dest.mkdir(parents=True, exist_ok=False)
    tasks = []
    for million, _, _ in CASES:
        paths = sorted((ROOT/'runs'/f'mc_{million}m_s1'/'production/out/pvtk').glob('*.part.vtk'))
        assert len(paths) >= 201
        tasks += [(million, slot, str(path)) for slot, path in enumerate(paths[:201])]
    rows = {million: [] for million, _, _ in CASES}
    sources = {million: [] for million, _, _ in CASES}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, initializer=initialize) as pool:
        for i, (million, row, source) in enumerate(pool.map(reduce_frame, tasks, chunksize=1)):
            rows[million].append(row)
            sources[million].append(source)
            if i % 10 == 0:
                print('INTERIM_SHELL', million, row['slot'], row['time'], flush=True)
    for million, _, _ in CASES:
        values = rows[million]
        assert len(values) == 201 and values[0]['time'] == 0
        assert abs(values[-1]['time']-400) <= .25
        assert np.all(np.diff([r['time'] for r in values]) > 0)
        with (dest/f'mc_{million}m_s1_shells_com.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    plot(rows, dest)
    atomic(dest/'SOURCES.json', dict(utc=now(), job=os.environ['SLURM_JOB_ID'], host=socket.gethostname(),
           source=pin['head'], native_runtime_source=pin['native_runtime_source'],
           compiled_physics=PHYSICS, executable_sha256=EXE_SHA, source_files=sources,
           full_count_unique_IDs_finite_fixed_masses_verified=True,
           reference_full_sort_checks=6, optimized_quantiles_exact_reference_match=True,
           center='Rest-mass-weighted moving COM', shell_definition='Reference cumulative rest-mass order statistic',
           interval_M=[0, 400], cadence_M=2, states_per_case=201,
           native_data_host='AMD only', elapsed_seconds=time.monotonic()-started))
    products = [dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p))
                for p in sorted(dest.iterdir()) if p.is_file()]
    atomic(dest/'PRODUCT_MANIFEST.json', dict(utc=now(), source=pin['head'], files=products))
    atomic(ROOT/'evidence/INTERIM400_READY.json', dict(utc=now(), job=os.environ['SLURM_JOB_ID'],
           source=pin['head'], directory=str(dest), product_manifest_sha256=sha(dest/'PRODUCT_MANIFEST.json'),
           actual_visual_review_pending=True, raw_transferred=False))
    print('INTERIM400_READY', dest, flush=True)


if __name__ == '__main__':
    main()
