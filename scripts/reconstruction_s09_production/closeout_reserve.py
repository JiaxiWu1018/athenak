"""Use a closed matching-size endpoint to forecast audit time, never acceptance."""
import datetime
import math


PHASE_FILES = ['reduce.py', 'final_fields.py', 'case_review.py',
               'review_packet.py', 'closed_output_manifest.py', 'endpoint_manifest.py']


def seconds(snapshots, reference_seconds=None, reference_snapshots=None):
    assert isinstance(snapshots, int) and snapshots >= 0
    if reference_seconds is None:
        return max(300, 180 + 3 * snapshots)
    assert math.isfinite(reference_seconds) and reference_seconds > 0
    assert isinstance(reference_snapshots, int) and reference_snapshots > 0
    # Retain a 50% timing margin plus two minutes for variable I/O/teardown.
    # Never scale the measured time down for a smaller set of snapshots.
    return max(300, math.ceil(1.5 * reference_seconds *
                             max(1., snapshots / reference_snapshots) + 120))


def estimate(c, work, state, ranks):
    from common import EXT, Path, load, receipt, sha
    count = len(list((work / 'pvtk').glob('*.vtk')))
    result = dict(seconds=seconds(count), full_particle_snapshots=count,
                  basis='Fallback:180+3seconds per full snapshot, minimum300seconds.')
    # This timing reference applies only to the final, unremoved, unrefined
    # 300M endpoint on the same eight-rank backend. Other endpoints retain
    # the original reserve. All numerical/field/packet/manifest checks follow.
    if c['key'] != 'yc0819_u0005' or ranks != 8:
        return result
    try:
        assert c['L'] == 1024 and c['initial_dx'] == .5 and c['finest_dx'] == .125
        last = state['latest']
        assert state['complete'] and abs(last['time'] - 300) < 1e-8
        assert last['particle_count'] == 2000000 and last['nblocks'] == 344
        assert last['ah_onset'] < 0 and last['removal_counts'] == [0, 0, 0, 0]
        root = EXT / 'yc0900_u0005'
        ref_work = root / 'runs/yc0900'
        ref_state = load(ref_work / 'state.json')
        ref = ref_state['latest']
        assert ref_state['complete'] and abs(ref['time'] - 300) < 1e-8
        assert ref['particle_count'] == last['particle_count'] and ref['nblocks'] == last['nblocks']
        assert ref['ah_onset'] < 0 and ref['removal_counts'] == last['removal_counts']
        record = root / 'evidence/jobs' / ref['job']
        end_path = record / 'ALLOCATION_END.json'
        end = load(end_path)
        assert end['exit_status'] == 0 and end['nodes'] == 1 and end['ranks'] == ranks
        phase_hashes = load(record / 'DEPLOYMENT.json')['python_files']
        assert all(sha(Path(__file__).with_name(name)) == phase_hashes[name] for name in PHASE_FILES)
        paths = [root / 'evidence/CASE_NUMERICAL_ACCEPTANCE.json',
                 ref_work / 'FINAL_FIELDS_OK.json', root / 'review_packet/MANIFEST.json',
                 root / 'evidence/CLOSED_OUTPUT_MANIFEST.json']
        audits = [load(path) for path in paths]
        assert audits[0]['numerical_checks_passed'] and audits[1]['no_evolution']
        assert audits[3]['stable_files_verified']
        assert all(a['checkpoint']['sha256'] == ref['sha256'] for a in audits)
        assert ref_state['segments'][-1]['executable']['sha256'] == state['segments'][-1]['executable']['sha256']
        elapsed = (datetime.datetime.fromisoformat(end['utc']) -
                   datetime.datetime.fromisoformat(ref['utc'])).total_seconds()
        assert 0 < elapsed <= end['elapsed_before_exit_seconds']
        ref_count = len(list((ref_work / 'pvtk').glob('*.vtk')))
        result.update(seconds=seconds(count, elapsed, ref_count),
                      basis='Completed matching344-block/2M-unremoved/300M/eight-rank endpoint;1.5x measured closeout plus120seconds, scaled up by snapshot count.',
                      measured_closeout_seconds=elapsed, reference_snapshots=ref_count,
                      reference=[receipt(end_path), receipt(ref_work / 'state.json'),
                                 *[receipt(path) for path in paths]],
                      matching_phase_source_sha256={name: phase_hashes[name] for name in PHASE_FILES})
    except (AssertionError, KeyError, ValueError, OSError) as error:
        result['reference_not_applicable'] = type(error).__name__ + ': ' + str(error)
    return result
