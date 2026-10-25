"""Allocated reader/render and accepted-movie reuse checks; no evolution."""
import csv
import numpy as np
from common import *

def parallel_scheduling_check(out):
    from submit_local import production_predecessor, active_conflict
    outward = dict(id='fixture_outward', role='production', case='yc0500_outward',
        target=500., scheduler='JobState=RUNNING')
    control = dict(id='fixture_control', role='production', case='yc0500_control',
        target=1000., scheduler='JobState=PENDING')
    assert production_predecessor([outward], 'production', 'yc0500_control') == ''
    assert production_predecessor([outward, control], 'production', 'yc0500_outward') == outward['id']
    assert production_predecessor([outward, control], 'production', 'yc0500_control') == control['id']
    assert not active_conflict(outward, 'production', 'yc0500_control', 1000.)
    assert active_conflict(control, 'production', 'yc0500_control', 1500.)
    # Exercise the exact frozen launcher lock commands in a disposable QA tree.
    # Never acquire or modify the live production lock files in this fixture.
    launch = EXT/'evidence/allocations'/job()/'launch_amd.sh'
    block = launch.read_text().split('if [ "$S10_CPU_ONLY" != 1 ]; then\n', 1)[1].split('\nfi', 1)[0]
    fixture = out/'concurrent_lock_fixture'
    (fixture/'evidence/locks').mkdir(parents=True)
    script = 'set -euo pipefail\nbatch="$1"\nST8_CASE="$2"\n'+block+'\nprintf "LOCKED\\n"\nread -r token\n'
    children = []
    try:
        import select
        for case in ['yc0500_outward', 'yc0500_control']:
            child = subprocess.Popen(['bash', '-c', script, 'lock_fixture', str(fixture), case],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            children.append(child)
            assert select.select([child.stdout], [], [], 10)[0], 'Lock fixture did not respond'
            assert child.stdout.readline().strip() == 'LOCKED'
        assert all(child.poll() is None for child in children), 'Both case locks must coexist'
        duplicate = subprocess.run(['bash', '-c', script, 'lock_fixture', str(fixture), 'yc0500_outward'],
            input='done\n', capture_output=True, text=True, timeout=10)
        assert duplicate.returncode == 1 and 'Another Session10 allocation' in duplicate.stdout
    finally:
        for child in children:
            child.communicate('done\n', timeout=10)
    assert all(child.returncode == 0 for child in children)
    record = dict(utc=now(), job=job(), frozen_launcher=receipt(launch),
        independent_case_dependencies=True, simultaneous_distinct_case_locks=True,
        duplicate_same_case_lock_rejected=True, same_case_active_job_any_target_rejected=True,
        fixture_only=True, no_production_lock_files_touched=True,
        actual_concurrent_production_not_established_by_this_fixture=True)
    atomic(out/'PARALLEL_SCHEDULING_OK.json', record)
    return receipt(out/'PARALLEL_SCHEDULING_OK.json')


def accepted_movie_reuse_check(out):
    from products import source_data, individual_movie
    index = EXT/'delivery/accepted_case_movies/yc0500_outward.json'
    if not index.exists():
        return None
    accepted = load(index)
    assert accepted['actual_encoded_frame_visual_review'] and accepted['full_decode_passed']
    destination = out/'accepted_movie_reuse'
    destination.mkdir()
    datum, = source_data(['yc0500_outward'])
    copied = individual_movie(datum, destination)
    result = receipt(copied)
    assert result['sha256'] == accepted['movie']['sha256']
    assert result['bytes'] == accepted['movie']['bytes']
    assert not list(destination.glob('*_frames')), 'Accepted movie must be copied without rendering'
    reused = destination/'yc0500_outward_REUSED.json'
    assert load(reused)['no_rerender']
    record = dict(utc=now(), job=job(), accepted_index=receipt(index),
        copied_movie=result, reuse_provenance=receipt(reused),
        source_matched_encoded_bytes=True, no_new_frame_directory=True,
        no_evolution=True, no_new_visual_acceptance_claim=True,
        scope='Actual final-workflow reuse function exercised on the already inspected movie; '
              'combined final generation and synchronized movie remain pending.')
    atomic(out/'ACCEPTED_MOVIE_REUSE_OK.json', record)
    return receipt(out/'ACCEPTED_MOVIE_REUSE_OK.json')

def main():
    from extension_resources import admission
    from extension_resources import request_horizon_checkpoint
    admission()
    from products import resolve_prefix, movie
    from analysis import plot_case
    from run_segment import cohort
    from run_segment import protected_checkpoint_paths
    from reduce import csv_rows
    prefix = resolve_prefix()
    fixture=[dict(path=f'fixture_{t}.rst',time=t,dt=.1,ah_onset=95.5 if t>=95.5 else -1,final=t==300)
             for t in [0,10,20,40,95.5,100,200,210,220,300]]
    assert protected_checkpoint_paths(fixture)=={f'fixture_{t}.rst' for t in [0,20,95.5,100,200,210,220,300]}
    with (prefix/'cohort_index_resolved.csv').open() as f:
        rows = list(csv.DictReader(f))
    for t in [0, 350, 700]:
        row = min(rows,key=lambda r:abs(float(r['time'])-t))
        actual,cycle,mask,values=cohort(row['file'])
        assert abs(actual-t)<.65 and mask.sum()==20000
        assert np.array_equal(values[mask==1,0],np.arange(20000)*100)
    out=EXT/'qa'/job()
    out.mkdir(parents=True)
    parallel = parallel_scheduling_check(out)
    reused_movie = accepted_movie_reuse_check(out)
    from resource_projection import analytic_checks as projection_checks
    projection = projection_checks()
    from milestone_review import forecast
    projection_dir = out/'resource_forecast'
    projection_dir.mkdir()
    actual_forecast = forecast(projection_dir)
    atomic(out/'RESOURCE_PROJECTION_OK.json', dict(utc=now(), job=job(), analytic=projection,
        actual_metadata_forecast=receipt(projection_dir/'RESOURCE_FORECAST.json'),
        cost_projection_within_cap=actual_forecast['cost_projection_within_cap'],
        storage_projection_within_cap=actual_forecast['storage_projection_within_cap'],
        conditional_not_a_guarantee=True, shared_caps_unchanged=True, no_new_evolution=True))
    from window_metrics import analytic_checks, time_window, coverage, profile_windows
    checks = analytic_checks()
    accepted = EXT/'yc0500_outward/reviews/milestone_00100_figures_478742'
    inputs = load(accepted/'SCIENCE_TRENDS.json')['source_inputs']
    for entry in inputs:
        p = accepted/'data'/Path(entry['path']).name
        assert p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256']
    profile_file = accepted/'data/coordinate_rest_mass_profiles.npz'
    z = np.load(profile_file)
    m0 = load(accepted/'data/summary.json')['rest_mass_initial']
    measured = []
    for lo, hi in [(10., 40.), (40., 70.), (70., 100.)]:
        result = time_window(z['time'], z['mass_per_bin'], lo, hi)
        assert result['complete'] and np.isclose(result['mean'].sum(), m0, rtol=1e-12)
        measured.append(dict(**coverage(result), mean_mass=float(result['mean'].sum())))
    unavailable, drift = profile_windows(z['time'], z['mass_per_bin'], m0)
    assert all(not r['complete'] and r['mean'] is None for r in unavailable)
    assert all(not r['complete'] and r['total_variation_M0'] is None for r in drift)
    atomic(out/'WINDOW_METRICS_OK.json', dict(utc=now(), job=job(), analytic=checks,
        accepted_100M_fixture=True, sources=[receipt(profile_file), receipt(accepted/'SCIENCE_TRENDS.json')],
        complete_test_windows=measured, missing_late_windows_remain_blank=True,
        no_new_evolution=True, no_new_scientific_endpoint=True))
    fixture_root=out/'horizon_checkpoint_fixture';work=fixture_root/'runs/yc0500'
    work.mkdir(parents=True)
    atomic(fixture_root/'evidence/active.json',dict(job=job(),work=str(work),case='yc0500',phase='evolution'))
    (work/'yc0500.ah_quality_0.csv').write_text('time,accepted,missing,streak,hrel,rmin,dx,mass,mirr,area\n1,1,0,3,0.01,0.5,0.0625,1,1,50.26548246\n')
    event=request_horizon_checkpoint(fixture_root)
    assert event['accepted_native_time']==1 and (work/'REQUEST_STOP').read_text().strip()=='qualified_AH_checkpoint'
    assert request_horizon_checkpoint(fixture_root) is None
    raw=dict(id='historical_control_renderer_test',key='historical_control_renderer_test',U=0.,start=0.)
    plot_case(raw,prefix,out)
    datum=dict(case=raw,summary=dict(endpoint=700.,ah_first=None),cohorts=rows)
    path=movie([datum],out,'historical_control_3frame_smoke',targets=[0.,350.,700.])
    h=csv_rows(prefix/'history.csv');assert h[0]['time']==0 and abs(h[-1]['time']-700)<1e-7
    atomic(out/'QA_OK.json',dict(utc=now(),job=job(),original_history=receipt(prefix/'SOURCE_MANIFEST.json'),
        resolved_cohorts=receipt(prefix/'RESOLVED.json'),movie=receipt(path),
        window_metrics=receipt(out/'WINDOW_METRICS_OK.json'),
        parallel_scheduling=parallel,
        resource_projection=receipt(out/'RESOURCE_PROJECTION_OK.json'),
        accepted_movie_reuse=reused_movie,
        validation_only=True,fixtures_only=reused_movie is None,
        no_new_evolution=True,no_session10_scientific_result=True,
        actual_encoded_visual_review_pending=True))
    print(out)

if __name__=='__main__':
    main()
