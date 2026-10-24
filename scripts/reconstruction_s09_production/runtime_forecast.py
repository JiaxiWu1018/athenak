"""Conservative full-cap forecast from current output metadata, inside Slurm."""
import math
from common import *
from extension_resources import storage, allocated_bytes, stop


def size(path):
    return allocated_bytes(path) if path.exists() else 0


def refresh():
    job()
    disk = storage()
    additions, cases = 0., []
    for raw in ALL_CASES:
        case = case_config(raw['id'])
        root = EXT / case['key']
        work = root / 'runs' / case['id']
        state_path = work / 'state.json'
        state = load(state_path) if state_path.exists() else None
        current = state['latest']['time'] if state else 0.
        onset = state['latest']['ah_onset'] if state else -1.
        target = min(300., onset + 20.) if onset >= 0 else 300.
        remaining = max(0., target - current)
        full = max([p.stat().st_size for p in (work / 'pvtk').glob('*.vtk')] + [80_000_000])
        fields = 0
        for name in ['z4c2d', 'tme2d', 'con2d']:
            fields += max([p.stat().st_size for p in (work / 'bin').glob('*.' + name + '.*.bin')] + [5_000_000])
        cohort = max([p.stat().st_size for p in (work / 'cohort').glob('*.s8c')] + [1_360_000])
        rate = full + 1.2 * fields + 2 * cohort + 1_000_000
        streams = rate * (math.ceil(remaining) + 1) if remaining else 0
        checkpoints = list((work / 'rst').glob('*.rst'))
        largest = max([p.stat().st_size for p in checkpoints] + [6 * GIB])
        checkpoint_reservation = max(24 * GIB, 4 * largest * 1.1)
        checkpoint_add = max(0., checkpoint_reservation - size(work / 'rst'))
        tests_add = max(0., 6 * GIB - size(root / 'tests'))
        controls = sum(size(p) for p in (EXT / 'controls').glob(case['id'] + '_amd_*'))
        controls_add = max(0., 5 * GIB - controls)
        fields_add = max(0., 3 * GIB - size(work / 'final_fields'))
        if state and state.get('complete'):
            streams = 0
            if (root / 'review_packet/MANIFEST.json').exists():
                checkpoint_add = tests_add = controls_add = fields_add = 0
        added = streams + checkpoint_add + tests_add + controls_add + fields_add
        additions += added
        rates = [row['native_node_hours_per_M'] for row in state['segments']
                 if row.get('native_node_hours_per_M') is not None] if state else []
        cases.append(dict(case=case['key'], current_M=current, target_M=target,
                          remaining_M=remaining, measured_stream_bytes_per_M=rate,
                          projected_additional_GiB=added / GIB,
                          maximum_checkpoint_bytes=largest,
                          observed_native_nodeh_per_M=rates[-1] if rates else None))
    # Sources/reference packets, review frames/products and a two-trajectory
    # checkpoint reserve remain pessimistic additions even when partly present.
    fixed = (8 + 3 + 20 + 48) * GIB
    forecast = disk['own'] + additions + fixed
    record = dict(utc=now(), job=job(), storage=disk, cases=cases,
                  projected_peak_GiB=forecast / GIB, limit_GiB=400,
                  fixed_source_review_checkpoint_reserve_GiB=fixed / GIB,
                  basis='Actual output/checkpoint metadata; all remaining full caps unless AH onset is accepted. Preserved fixtures and existing bytes included.')
    atomic(EXT / 'evidence' / ('FORECAST_' + CASE['key'] + '.json'), record)
    if forecast > 400 * GIB:
        stop('projected_session_storage_exceeds400GiB')
    return record


if __name__ == '__main__':
    print(json.dumps(refresh(), indent=2))
