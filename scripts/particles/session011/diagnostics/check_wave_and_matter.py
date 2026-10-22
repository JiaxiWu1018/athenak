"""Investigate finite-window FFI offsets and discrete matter-J accounting, without evolution."""
import argparse
import json
import os
import platform
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import products


def main():
    products.require_compute()
    ap = argparse.ArgumentParser()
    ap.add_argument('--review', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir()
    paths = [args.review / name for name in
             ('wave22_R40.csv', 'component_ledgers.csv', 'shell_transfers.csv')]
    deaths = sorted((args.review.parent / 'source_view/runs').glob('segment_*/*.prtcl_destroy.csv'))
    assert deaths
    paths += deaths
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
    hashes = {str(p): products.sha256(p) for p in paths}

    data = pd.read_csv(paths[0])
    t = data.time.to_numpy()
    psi = data.re.to_numpy() + 1j * data.im.to_numpy()
    splits = np.flatnonzero(np.diff(t) > .0375) + 1
    intervals = []
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    labelled = set()
    for index, ids in enumerate(np.split(np.arange(len(t)), splits)):
        if len(ids) < 100:
            continue
        uniform = np.arange(t[ids[0]], t[ids[-1]], .025)
        values = np.interp(uniform, t[ids], psi[ids].real) + 1j * np.interp(uniform, t[ids], psi[ids].imag)
        n = len(uniform)
        window = np.ones(n)
        edge = max(2, int(.1 * n))
        ramp = .5 * (1 - np.cos(np.linspace(0, np.pi, edge)))
        window[:edge] = ramp
        window[-edge:] = ramp[::-1]
        source = (values - np.mean(values)) * window
        fundamental = 1 / (n * (uniform[1] - uniform[0]))
        centered = []
        rows = []
        for f0 in (.01, .02, .04):
            h, check = products.ffi(uniform, values, f0)
            dc = -np.mean(source) / (2 * np.pi * f0)**2
            assert abs(np.mean(h) - dc) < 1e-12 * max(abs(dc), 1e-30)
            free = h - np.mean(h)
            centered.append(free)
            rows.append(dict(f0=f0, mean_rh_plus=float(h.real.mean()),
                             mean_rh_cross=float(-h.imag.mean()),
                             predicted_dc_rh_plus=float(dc.real),
                             centered_rms=float(np.sqrt(np.mean(abs(free)**2))),
                             cutoff_below_first_nonzero_bin=bool(f0 < fundamental),
                             derivative_check=check))
            label = f'f0={f0:.2f}' if f0 not in labelled else None
            axes[0].plot(uniform - 40, h.real, label=label)
            axes[1].plot(uniform - 40, free.real, label=label)
            labelled.add(f0)
        difference = max(float(np.max(abs(x - centered[0]))) for x in centered)
        intervals.append(dict(index=index, samples=n, first=float(uniform[0]), last=float(uniform[-1]),
                              first_nonzero_frequency=fundamental, cutoffs=rows,
                              max_centered_cutoff_difference=difference))
    assert intervals
    axes[0].set(ylabel='Original provisional r h+', title='Published cutoff-dependent offsets')
    axes[1].set(ylabel='r h+ minus interval mean', xlabel='Approximate coordinate retarded time t-R / Mref',
                title='Same data after removing the unconstrained constant')
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    products.savefig(fig, args.out / 'wave_zero_mode')

    ledger = pd.read_csv(paths[1])
    ledger = ledger[ledger['mask'] == 0].sort_values(['component', 'time'])
    transfer = pd.read_csv(paths[2]).groupby(['cycle', 'component']).sum(numeric_only=True)
    timelines = {c: d.reset_index(drop=True) for c, d in ledger.groupby('component')}
    increments = {c: np.zeros((len(d), 3)) for c, d in timelines.items()}
    for path in deaths:
        for chunk in pd.read_csv(path, usecols=['time', 'component', 'mass', 'x1', 'x2', 'u1_cov', 'u2_cov'], chunksize=200_000):
            assert np.isfinite(chunk.to_numpy()).all()
            for c, d in chunk.groupby('component'):
                times = timelines[c].time.to_numpy()
                # Native destruction cycle is pre-advance; its time is post-push.
                ids = np.searchsorted(times, d.time.to_numpy() - 1e-10)
                assert (ids < len(times)).all()
                j = d.mass * (d.x1 * d.u2_cov - d.x2 * d.u1_cov)
                for k, weights in enumerate((np.ones(len(d)), d.mass, j)):
                    increments[c][:, k] += np.bincount(ids, weights=weights, minlength=len(times))
    rows = []
    summaries = []
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for c, d in timelines.items():
        force = transfer.reindex(pd.MultiIndex.from_arrays([d.cycle, np.full(len(d), c)], names=['cycle', 'component']))
        assert force.Jz_force_delta.notna().all()
        removed = increments[c]
        delta_j = np.r_[0., np.diff(d.Jz)]
        residual = delta_j + removed[:, 2] - force.Jz_force_delta.to_numpy()
        mass_residual = np.r_[0., np.diff(d.rest_mass)] + removed[:, 1]
        frame = pd.DataFrame(dict(cycle=d.cycle, time=d.time, component=c, surviving_Jz=d.Jz,
                                  delta_surviving_Jz=delta_j, removed_count=removed[:, 0],
                                  removed_mass=removed[:, 1], removed_Jz=removed[:, 2],
                                  push_Jz_delta=force.Jz_force_delta.to_numpy(), closure_residual=residual,
                                  rest_mass_residual=mass_residual))
        rows.append(frame)
        selected = frame[frame.time.between(7., 8.)]
        axes[c].plot(selected.time, selected.delta_surviving_Jz, label='Change in surviving Jz')
        axes[c].plot(selected.time, selected.push_Jz_delta, label='Recorded push change')
        axes[c].plot(selected.time, -selected.removed_Jz, label='Removal contribution')
        axes[c].plot(selected.time, selected.closure_residual, 'k--', label='Accounting residual')
        axes[c].set_ylabel(f'Component {c}: delta Jz')
        extreme = selected.loc[selected.surviving_Jz.idxmax()]
        summaries.append(dict(component=int(c), removed_count=int(removed[:, 0].sum()),
                              max_abs_Jz_closure_residual=float(max(abs(residual))),
                              max_abs_rest_mass_residual=float(max(abs(mass_residual))),
                              max_abs_shell_advection_imbalance=float(max(abs(force.Jz_adv_in - force.Jz_adv_out))),
                              collapse_window_peak_time=float(extreme.time),
                              collapse_window_peak_surviving_Jz=float(extreme.surviving_Jz)))
    axes[0].legend(fontsize=9)
    axes[-1].set_xlabel('State time / Mref')
    fig.suptitle('Discrete coordinate matter-J accounting near collapse; no conserved spacetime-J claim')
    fig.tight_layout()
    products.savefig(fig, args.out / 'matter_Jz_closure')
    pd.concat(rows).to_csv(args.out / 'matter_Jz_closure.csv', index=False)
    unchanged = all((p.stat().st_size, p.stat().st_mtime_ns) == before[str(p)] for p in paths)
    assert unchanged
    result = dict(status='pass', job_id=os.environ['SLURM_JOB_ID'], host=platform.node(),
                  python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                  source=os.environ.get('SOURCE_COMMIT'), evolution_advances=0,
                  production_modified=False, input_metadata_unchanged=unchanged, input_sha256=hashes,
                  products_sha256=products.sha256(Path(products.__file__)), wave_intervals=intervals,
                  matter_components=summaries,
                  limits='Coordinate matter-J bookkeeping is not spacetime angular-momentum conservation or physical/numerical causal attribution; finite-window strain has no complete burst/ringdown claim')
    (args.out / 'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
