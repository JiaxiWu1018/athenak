"""Extract small, source-verified particle visualization caches in AMD Slurm."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

import numpy as np

from common import ROOT, N, atomic, require_job, sample, sha
from reduce_run import read_pvtk

MOVIE_CASES = ('proper_stratified_s1', 'legacy_stratified_s1')
INITIAL_CASES = ('proper_mc_s1', 'proper_stratified_s1', 'proper_cell_s1', 'proper_ablation_s1')
LIMIT = 12.0
BINS = 256
DISPLAY_COUNT = 20000


def histogram(positions):
    counts = np.histogram2d(positions[:, 0], positions[:, 1], bins=BINS,
                            range=[[-LIMIT, LIMIT], [-LIMIT, LIMIT]])[0]
    assert int(counts.sum()) == N
    return counts.astype(np.uint32)


def main():
    require_job()
    destination = ROOT/'particle_visualization_data'/os.environ['SLURM_JOB_ID']
    destination.mkdir(parents=True, exist_ok=False)
    ids = np.sort(np.random.default_rng(2026101505).choice(N, DISPLAY_COUNT, replace=False))
    clouds, projections, times, sources = [], [], [], []
    for case in MOVIE_CASES:
        run = ROOT/'runs'/case
        completion = json.loads((run/'completion.json').read_text())
        assert completion['endpoint_M'] == 300 and (run/'RUN_OK').exists()
        expected = {r['path']:r for r in completion['manifest']}
        paths = sorted((run/'out/pvtk').glob('*.part.vtk'))
        assert len(paths) == 151
        cloud, projection, time = [], [], []
        for frame, path in enumerate(paths):
            row = expected[str(path.relative_to(ROOT))]
            assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
            t, cycle, count, fields = read_pvtk(str(path))
            tags = fields['ptag'].astype(np.int64)
            assert count == N and np.all((tags>=0)&(tags<N))
            assert np.all(np.bincount(tags, minlength=N)==1)
            positions = np.column_stack([fields[a] for a in 'xyz'])
            assert np.isfinite(positions).all() and np.max(np.abs(positions)) < LIMIT
            assert abs(t-2*frame)<1e-7
            index = np.empty(N, dtype=np.int32)
            index[tags] = np.arange(N, dtype=np.int32)
            cloud.append(positions[index[ids]].astype(np.float32))
            projection.append(histogram(positions))
            time.append(t)
            sources.append(dict(row, case=case, time_M=t, cycle=cycle))
            if frame % 25 == 0:
                print('EXTRACT', case, frame, t, flush=True)
        clouds.append(np.array(cloud));projections.append(np.array(projection));times.append(time)
    assert np.allclose(times[0], times[1], rtol=0, atol=1e-7)
    initial_clouds, initial_projections, initial_sources, position_hashes = [], [], [], []
    for case in INITIAL_CASES:
        path = ROOT/'samples'/f'{case}.bin'
        evidence = json.loads((ROOT/'evidence/samples'/f'{case}.json').read_text())
        assert path.stat().st_size == evidence['bytes'] and sha(path) == evidence['sha256']
        _, data = sample(path)
        assert len(data) == N and np.array_equal(data[:, 0], np.arange(N))
        positions = np.ascontiguousarray(data[:, 1:4])
        assert np.isfinite(positions).all() and np.max(np.abs(positions)) < LIMIT
        position_hashes.append(hashlib.sha256(positions.tobytes()).hexdigest())
        initial_clouds.append(positions[ids].astype(np.float32))
        initial_projections.append(histogram(positions))
        initial_sources.append({'path':str(path.relative_to(ROOT)), 'bytes':path.stat().st_size,
                                'sha256':evidence['sha256'], 'case':case})
    assert position_hashes[1] == position_hashes[3]
    assert np.array_equal(initial_clouds[1], initial_clouds[3])
    assert np.array_equal(initial_projections[1], initial_projections[3])
    assert np.array_equal(np.array(clouds)[0, 0], initial_clouds[1])
    cache = destination/'particle_views.npz'
    np.savez_compressed(cache, display_ids=ids, movie_cases=np.array(MOVIE_CASES),
                        initial_cases=np.array(INITIAL_CASES), times=np.array(times),
                        movie_clouds=np.array(clouds), movie_counts=np.array(projections),
                        initial_clouds=np.array(initial_clouds), initial_counts=np.array(initial_projections))
    repo = Path(__file__).resolve().parents[2]
    atomic(destination/'DATA_OK.json', {
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'job':os.environ['SLURM_JOB_ID'], 'host':socket.gethostname(),
        'source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'script_sha256':sha(Path(__file__)),
        'cache':{'path':str(cache.relative_to(ROOT)), 'bytes':cache.stat().st_size,'sha256':sha(cache)},
        'movie_cases':MOVIE_CASES, 'initial_cases':INITIAL_CASES, 'seed':1,
        'particle_count':N, 'display_count':DISPLAY_COUNT, 'display_id_rng_seed':2026101505,
        'limit_M':LIMIT, 'projection_bins':BINS, 'projection':'x-y, all z, unweighted particle counts',
        'frames':151, 'nominal_times_M':list(range(0,301,2)),
        'actual_time_arrays_M':times, 'movie_sources':sources, 'initial_sources':initial_sources,
        'initial_position_sha256':position_hashes,
        'ablation_positions_identical_to_stratified':True,
        'note':'Fixed original-ID display sample, shared across cases and all times. No COM recentering, interpolation, reweighting or new simulation.'})
    print('PARTICLE_VISUAL_DATA_OK', destination, flush=True)


if __name__ == '__main__':
    main()
