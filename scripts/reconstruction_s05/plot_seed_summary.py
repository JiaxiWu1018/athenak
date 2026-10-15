"""Plot all nine-shell RMS summaries with explicit individual seed symbols."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from common import ROOT, atomic, require_job, sha


def main():
    require_job()
    parser = argparse.ArgumentParser()
    parser.add_argument('stats', type=Path)
    args = parser.parse_args()
    data = json.loads(args.stats.read_text())
    assert len(data['window_groups']) == 24
    methods = ('mc', 'stratified', 'cell', 'ablation')
    labels = ('Monte Carlo', 'Stratified', 'Cell-aware', 'Velocity ablation')
    colors = ('#333333', '#0072B2', '#D55E00', '#009E73')
    metrics = ('raw_amplitude_rms', 'raw_amplitude_rms_ratio_to_mc', 'mean_shift_rms_ratio_to_mc')
    titles = ('Raw excursion / own initial radius (%)', 'Raw excursion / same-profile MC',
              'Mean-shift RMS / same-profile MC')
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), layout='constrained')
    for row, radial in enumerate(('legacy', 'proper')):
        for column, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[row, column]
            for method_index, (method, color) in enumerate(zip(methods, colors)):
                selected = sorted((r for r in data['window_groups']
                                   if r['radial']==radial and r['method']==method),
                                  key=lambda r:r['start_M'])
                assert [r['start_M'] for r in selected] == [0, 100, 200]
                factor = 100 if column==0 else 1
                x = np.arange(3)+(method_index-1.5)*0.14
                mean = np.array([r[metric]['mean'] for r in selected])*factor
                low = np.array([r[metric]['range'][0] for r in selected])*factor
                high = np.array([r[metric]['range'][1] for r in selected])*factor
                ax.errorbar(x, mean, yerr=[mean-low, high-mean], color=color,
                            lw=1, capsize=3, marker='_', markersize=10)
                for seed, marker in enumerate(('o', 's', '^')):
                    values = np.array([r[metric]['seeds'][seed] for r in selected])*factor
                    ax.scatter(x+(seed-1)*0.018, values, c=color, marker=marker,
                               s=22, alpha=0.75, zorder=3)
            ax.set(title=f'{radial}: {title}', xticks=[0, 1, 2],
                   xticklabels=['0–100', '100–200', '200–300'], xlabel='Window / M')
            ax.grid(alpha=0.2)
            ax.set_ylim(bottom=0)
            if column:
                ax.axhline(1, color='gray', ls='--', lw=0.7)
    handles = [Line2D([], [], color=c, lw=2, label=l) for c, l in zip(colors, labels)]
    handles += [Line2D([], [], color='gray', marker=m, ls='', label=f'Seed {i+1}')
                for i, m in enumerate(('o', 's', '^'))]
    fig.legend(handles=handles, loc='outside upper center', ncol=7, fontsize=9)
    destination = ROOT/'plots'/f'seed_review_{os.environ["SLURM_JOB_ID"]}'
    destination.mkdir(parents=True, exist_ok=False)
    stem = destination/'f5_seed_summary'
    for suffix in ('.png', '.pdf'):
        fig.savefig(stem.with_suffix(suffix), dpi=180)
    plt.close(fig)
    preview = destination/'pdf_preview'
    subprocess.run(['pdftoppm', '-png', '-r', '120', '-singlefile',
                    str(stem.with_suffix('.pdf')), str(preview)], check=True)
    paths = [stem.with_suffix('.png'), stem.with_suffix('.pdf'), preview.with_suffix('.png')]
    atomic(destination/'MANIFEST.json', {'job':os.environ['SLURM_JOB_ID'],
           'input':{'path':str(args.stats), 'sha256':sha(args.stats)},
           'script_sha256':sha(Path(__file__)),
           'source_commit':subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[2]),
                                                   'rev-parse', 'HEAD'], text=True).strip(),
           'artifacts':[{'path':str(p.relative_to(ROOT)), 'sha256':sha(p), 'bytes':p.stat().st_size} for p in paths]})
    print('SEED_FIGURE_READY', destination)


if __name__ == '__main__':
    main()
