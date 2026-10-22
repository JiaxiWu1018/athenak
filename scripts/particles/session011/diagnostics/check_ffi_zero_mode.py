"""Check the FFI correction against old centered real-data strain; no evolution."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import products


def main():
    products.require_compute()
    ap = argparse.ArgumentParser()
    ap.add_argument('--review', type=Path, required=True)
    ap.add_argument('--legacy-products', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir()
    source_view = args.review.parent / 'source_view'
    runs = sorted((source_view / 'runs').glob('segment_*'))
    inputs = sorted((source_view / 'runs').glob('segment_*/waveforms/*'))
    inputs = [p for p in inputs if p.is_file()]
    metadata = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in inputs}
    hashes = {str(p): products.sha256(p) for p in inputs}
    env = dict(os.environ, PRODUCT_QA=str(args.out / 'synthetic'))
    with (args.out / 'tests.log').open('w') as log:
        subprocess.run([sys.executable, str(Path(products.__file__).with_name('test_products.py'))],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    spec = importlib.util.spec_from_file_location('legacy_products', args.legacy_products)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    data = pd.read_csv(args.review / 'wave22_R40.csv')
    time = data.time.to_numpy()
    psi = data.re.to_numpy() + 1j * data.im.to_numpy()
    comparisons = []
    for index, ids in enumerate(np.split(np.arange(len(time)), np.flatnonzero(np.diff(time) > .0375) + 1)):
        if len(ids) < 100:
            continue
        uniform = np.arange(time[ids[0]], time[ids[-1]], .025)
        values = np.interp(uniform, time[ids], psi[ids].real) + 1j * np.interp(uniform, time[ids], psi[ids].imag)
        for f0 in (.01, .02, .04):
            old, previous = legacy.ffi(uniform, values, f0)
            new, check = products.ffi(uniform, values, f0)
            expected = old - old.mean()
            np.testing.assert_allclose(new, expected, rtol=1e-10, atol=1e-20)
            assert abs(new.mean()) < 1e-20
            assert abs(check['derivative_filtered_relative_rms'] - previous['derivative_filtered_relative_rms']) < 1e-10
            assert abs(check['derivative_original_relative_rms'] - previous['derivative_original_relative_rms']) < 1e-10
            comparisons.append(dict(interval=index, f0=f0, **{k:v for k,v in check.items() if k != 'f0'},
                                    max_difference_from_legacy_minus_mean=float(max(abs(new - expected)))))
    assert comparisons
    plot = args.out / 'corrected_wave_products'
    plot.mkdir()
    wave = products.waves(runs, plot)
    for radius in (36, 40, 44):
        assert products.sha256(plot / f'wave22_R{radius}.csv') == products.sha256(args.review / f'wave22_R{radius}.csv')
    assert wave['zero_frequency'] and wave['frequency_resolution']
    assert all((p.stat().st_size, p.stat().st_mtime_ns) == metadata[str(p)] for p in inputs)
    result = dict(status='pass', job_id=os.environ['SLURM_JOB_ID'], source=os.environ['SOURCE_COMMIT'],
                  host=platform.node(), python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                  evolution_advances=0, production_modified=False, input_metadata_unchanged=True,
                  input_sha256=hashes, old_products_sha256=products.sha256(args.legacy_products),
                  new_products_sha256=products.sha256(Path(products.__file__)), comparisons=comparisons,
                  raw_wave_tables_unchanged=True, full_product_tests='passed; see tests.log',
                  qualification='Zero mode fixes an arbitrary integration constant; it does not recover memory, improve frequency resolution or validate burst/ringdown')
    (args.out / 'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    manifest = [dict(path=str(p.relative_to(args.out)), bytes=p.stat().st_size, sha256=products.sha256(p))
                for p in sorted(args.out.rglob('*')) if p.is_file()]
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
