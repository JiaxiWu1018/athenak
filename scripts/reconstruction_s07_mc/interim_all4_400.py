"""One four-run COM-shell figure after all production prefixes reach400M."""
import concurrent.futures
import csv
import importlib.util
import multiprocessing
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from cfg import ROOT, AMD, P, atomic, load, now, sha, require_job

CASES = [(1, '#0072b2', '-'), (2, '#d55e00', '--'),
         (4, '#009e73', '-.'), (8, '#cc79a7', ':')]


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def plot(rows, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pypdfium2 as pdfium
    plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': .18,
                         'savefig.dpi': 180})
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
    fig.subplots_adjust(left=.085, right=.98, bottom=.09, top=.78, hspace=.14, wspace=.21)
    for million, color, style in CASES:
        data = rows[million]
        t = np.array([float(r['time']) for r in data])
        assert len(t) == 201 and t[0] == 0 and abs(t[-1]-400) < 1e-7
        assert np.all(np.diff(t) > 0)
        for j, q in enumerate((10, 50, 90)):
            v = np.array([float(r[f'r{q}']) for r in data])
            axes[j, 0].plot(t, v, color=color, ls=style, lw=1.6, label=f'MC {million} million')
            axes[j, 1].plot(t, 100*(v/v[0]-1), color=color, ls=style, lw=1.6)
    for j, q in enumerate((10, 50, 90)):
        axes[j, 0].set_ylabel(f'$r_{{{q}}}/M$')
        axes[j, 1].set_ylabel('Change from initial [%]')
        axes[j, 1].axhline(0, color='#666666', lw=.6, zorder=0)
    axes[0, 0].set_title('Enclosed rest-mass coordinate radius about COM', pad=53)
    axes[0, 1].set_title('Fractional shell change', pad=53)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [0, 2, 1, 3]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc='upper center', bbox_to_anchor=(.5, .932), ncol=2, frameon=False)
    for ax in axes.ravel():
        ax.set_xlim(0, 400)
        ax.set_xticks([0, 100, 200, 300, 400])
    for ax in axes[-1]:
        ax.set_xlabel('$t/M$')
    for ax in axes[0]:
        ax.secondary_xaxis('top', functions=(lambda t: t/P, lambda p: p*P)).set_xlabel('Surface periods')
    fig.suptitle('ST session 7 | all four production runs | COM shells through400M\n'
                 '$y_c=0.819$ · proper profile · nested MC seed 1 · fixed mesh', y=.986, fontsize=14)
    fig.text(.5, .025, 'All particles; every native 2M output. Each row has its own y-scale.',
             ha='center', fontsize=10)
    fig.savefig(destination/'shells_com_all4_to400M.png')
    fig.savefig(destination/'shells_com_all4_to400M.pdf')
    plt.close(fig)
    document = pdfium.PdfDocument(str(destination/'shells_com_all4_to400M.pdf'))
    assert len(document) == 1
    document[0].render(scale=1.5).to_pil().save(destination/'shells_com_all4_to400M_pdf.png')
    document.close()


def main():
    require_job()
    assert ROOT == AMD
    pin_path = Path(__file__).resolve().parent/'ALL4_SOURCE_PIN.json'
    pin = load(pin_path)
    for entry in pin['files']:
        assert sha(pin_path.parent/entry['path']) == entry['sha256']
    for million, _, _ in CASES:
        # The next saved slot ensures that the400M file was closed previously.
        path = ROOT/'runs'/f'mc_{million}m_s1'/'production/out/pvtk'/f'mc_{million}m_s1.prtcl_all.00201.part.vtk'
        assert path.is_file() and path.stat().st_size > 0, str(path)
    original = ROOT/'interim/com_shells_to400M_af63c033'
    assert sha(original/'PRODUCT_MANIFEST.json') == pin['existing_reduction_manifest_sha256']
    for entry in load(original/'PRODUCT_MANIFEST.json')['files']:
        path = original/entry['path']
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
    helper_path = Path(pin['reducer_helper']['path'])
    assert sha(helper_path) == pin['reducer_helper']['sha256']
    spec = importlib.util.spec_from_file_location('session7_interim_af63', helper_path)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    # Configure the preserved analysis helper in memory; no source file changes.
    helper.CASES = CASES[:2]
    destination = ROOT/'interim'/('com_shells_all4_to400M_'+pin['head'][:8])
    destination.mkdir(parents=True, exist_ok=False)
    rows, sources = {}, {}
    inherited = load(original/'SOURCES.json')
    assert inherited['optimized_quantiles_exact_reference_match']
    for million in (4, 8):
        path = original/f'mc_{million}m_s1_shells_com.csv'
        rows[million] = read_rows(path)
        sources[million] = inherited['source_files'][str(million)]
        shutil.copyfile(path, destination/path.name)
    tasks = []
    for million in (1, 2):
        paths = sorted((ROOT/'runs'/f'mc_{million}m_s1'/'production/out/pvtk').glob('*.part.vtk'))
        assert len(paths) >= 202
        tasks.extend((million, slot, str(path)) for slot, path in enumerate(paths[:201]))
        rows[million], sources[million] = [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, initializer=helper.initialize,
            mp_context=multiprocessing.get_context('fork')) as pool:
        for j, (million, row, source) in enumerate(pool.map(helper.reduce_frame, tasks, chunksize=1)):
            rows[million].append(row)
            sources[million].append(source)
            if j % 20 == 0:
                print('ALL4_SHELLS', million, row['slot'], row['time'], flush=True)
    for million in (1, 2):
        with (destination/f'mc_{million}m_s1_shells_com.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[million][0]))
            writer.writeheader()
            writer.writerows(rows[million])
    assert all(len(rows[n]) == len(sources[n]) == 201 for n, _, _ in CASES)
    assert sum(s['reference_full_sort_checked'] for ss in sources.values() for s in ss) == 12
    plot(rows, destination)
    atomic(destination/'SOURCES.json', dict(utc=now(), job=os.environ['SLURM_JOB_ID'],
           source=pin['head'], native_runtime_source=pin['native_runtime_source'], source_files=sources,
           inherited_reduction_source=inherited['source'], all_sources_are_production=True,
           interval_M=[0, 400], states_per_case=201, all_particles_used=True,
           reference_full_sort_checks=12, full_count_unique_IDs_finite_fixed_masses_verified=True,
           actual_visual_review_pending=True, native_data_host='AMD only'))
    files = [dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p))
             for p in sorted(destination.iterdir()) if p.is_file()]
    atomic(destination/'PRODUCT_MANIFEST.json', dict(utc=now(), source=pin['head'], files=files))
    atomic(ROOT/'evidence/ALL4_400_READY.json', dict(utc=now(), job=os.environ['SLURM_JOB_ID'],
           source=pin['head'], directory=str(destination), product_manifest_sha256=sha(destination/'PRODUCT_MANIFEST.json'),
           actual_visual_review_pending=True, raw_transferred=False))
    print('ALL4_400_READY', destination, flush=True)


if __name__ == '__main__':
    main()
