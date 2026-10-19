"""Re-export the validated interim figure with a fixed unobstructive legend."""
import csv
from pathlib import Path
import os
import numpy as np
from cfg import ROOT, AMD, P, atomic, sha, now, load, require_job


def main():
    require_job()
    assert ROOT == AMD
    pin = load(Path(__file__).resolve().parent/'INTERIM_LAYOUT_SOURCE_PIN.json')
    for row in pin['files']:
        assert sha(Path(__file__).resolve().parent/row['path']) == row['sha256']
    original = ROOT/'interim/com_shells_to400M_af63c033'
    manifest = load(original/'PRODUCT_MANIFEST.json')
    assert sha(original/'PRODUCT_MANIFEST.json') == pin['original_manifest_sha256']
    for row in manifest['files']:
        p = original/row['path']
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256']
    qa = load(original/'SOURCES.json')
    assert qa['optimized_quantiles_exact_reference_match'] and qa['reference_full_sort_checks'] == 6
    dest = original/'layout_v2'
    dest.mkdir(exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pypdfium2 as pdfium
    plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': .18, 'savefig.dpi': 180})
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, layout='constrained')
    inputs = []
    for million, color, style in [(4, '#009e73', '-.'), (8, '#cc79a7', ':')]:
        path = original/f'mc_{million}m_s1_shells_com.csv'
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 201
        t = np.array([float(r['time']) for r in rows])
        assert t[0] == 0 and t[-1] == 400 and np.all(np.diff(t) > 0)
        for j, q in enumerate((10, 50, 90)):
            v = np.array([float(r[f'r{q}']) for r in rows])
            axes[j, 0].plot(t, v, color=color, ls=style, lw=1.7, label=f'MC {million} million')
            axes[j, 1].plot(t, 100*(v/v[0]-1), color=color, ls=style, lw=1.7)
        inputs.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
    for j, q in enumerate((10, 50, 90)):
        axes[j, 0].set_ylabel(f'$r_{{{q}}}/M$')
        axes[j, 1].set_ylabel('Change from initial [%]')
        axes[j, 1].axhline(0, color='#666666', lw=.6, zorder=0)
    axes[0, 0].set_title('Enclosed rest-mass coordinate radius about COM')
    axes[0, 1].set_title('Fractional shell change')
    axes[0, 0].legend(loc='upper left', framealpha=.9)
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
    fig.savefig(dest/'shells_com_to400M.png')
    fig.savefig(dest/'shells_com_to400M.pdf')
    plt.close(fig)
    document = pdfium.PdfDocument(str(dest/'shells_com_to400M.pdf'))
    document[0].render(scale=1.5).to_pil().save(dest/'shells_com_to400M_pdf.png')
    document.close()
    atomic(dest/'RENDER_QA.json', dict(utc=now(), job=os.environ['SLURM_JOB_ID'], source=pin['head'],
           reduction_source=qa['source'], input_manifest_sha256=pin['original_manifest_sha256'], inputs=inputs,
           data_unchanged=True, legend='Fixed upper left in both PNG and PDF', actual_visual_review_pending=True))
    files = [dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(dest.iterdir()) if p.is_file()]
    atomic(dest/'PRODUCT_MANIFEST.json', dict(utc=now(), source=pin['head'], files=files))
    atomic(ROOT/'evidence/INTERIM400_LAYOUT_READY.json', dict(utc=now(), job=os.environ['SLURM_JOB_ID'],
           source=pin['head'], directory=str(dest), product_manifest_sha256=sha(dest/'PRODUCT_MANIFEST.json')))
    print('INTERIM400_LAYOUT_READY', dest, flush=True)


if __name__ == '__main__':
    main()
