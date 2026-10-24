"""Classify actual native runtime messages without suppressing inherited diagnostics."""
import re
from common import *

KNOWN_NULL = 'UCX  ERROR get pointer info fail (nil)'
ACCEPTED_REFERENCE = Path('/work1/eliasmost/jiaxiwu/st_s08_scan_20261023/yc0605_u0020')


def classify(path):
    known, other = 0, []
    with path.open(errors='replace') as stream:
        for number, line in enumerate(stream, 1):
            if re.search(r'UCX\s+ERROR', line):
                if re.search(r'rocm_base\.c:\d+\s+UCX\s+ERROR get pointer info fail \(nil\)\s*$', line):
                    known += 1
                else:
                    other.append(dict(line=number, text=line.strip()))
            if re.search(r'### FATAL|search_fail=[1-9]|MPI_ABORT|MPI_ERR_|Segmentation fault|Memory access fault|hipError', line):
                other.append(dict(line=number, text=line.strip()))
    return dict(source=receipt(path), known_null_pointer_diagnostic_count=known,
                other_runtime_errors=other)


def check_reference():
    job()
    reference_path = ACCEPTED_REFERENCE / 'evidence/CASE_NUMERICAL_ACCEPTANCE.json'
    reference = load(reference_path)
    assert reference['numerical_checks_passed']
    reference_log = next(row for row in reference['accepted_native_logs']
                         if Path(row['path']).name == '476727_s001.log')
    assert sha(reference_log['path']) == reference_log['sha256']
    example = classify(Path(reference_log['path']))
    assert example['known_null_pointer_diagnostic_count'] > 0
    example['source_acceptance'] = receipt(reference_path)
    atomic(EXT / 'evidence/TRANSPORT_REFERENCE_CHECK.json', example)
    return example


def audit():
    job()
    example = check_reference()
    assert not example['other_runtime_errors']
    cases = []
    for raw in ALL_CASES:
        source = EXT / raw['id'] / 'evidence/CASE_NUMERICAL_ACCEPTANCE.json'
        accept = load(source)
        assert accept['numerical_checks_passed']
        logs = []
        for row in accept['accepted_native_logs']:
            path = Path(row['path'])
            assert sha(path) == row['sha256']
            result = classify(path)
            assert not result['other_runtime_errors'], result
            logs.append(result)
        assert logs
        cases.append(dict(case=raw['id'], source_acceptance=receipt(source), logs=logs,
                          known_null_pointer_diagnostic_count=sum(r['known_null_pointer_diagnostic_count'] for r in logs),
                          other_runtime_errors=0))
    result = dict(utc=now(), job=job(), cases=cases,
                  diagnostic_signature='rocm_base.c UCX ERROR get pointer info fail (nil)',
                  accepted_reference=example, accepted_reference_audit=example['source_acceptance'],
                  matching_diagnostic_present_in_previously_accepted_backend=True,
                  no_unclassified_runtime_error=True,
                  scope='The known message is retained and counted, not suppressed. Its occurrence in an accepted reference does not prove general harmlessness; acceptance also requires full native population/ID/weight/cohort/field/constraint checks. No native or transport-stack change.')
    atomic(ROOT / 'evidence/TRANSPORT_DIAGNOSTICS.json', result)
    text = '# Native transport diagnostics\n\n'
    text += ('The preserved UCX 1.18.1 backend prints `get pointer info fail (nil)` '
             'messages during evolution. The exact signature is also present in the '
             'previously accepted MI250 yc=0.605/U=-0.020 run. Raw messages remain '
             'in the source logs; no log-level suppression or backend change was made.\n\n'
             '| New case | Known null-pointer diagnostic messages | Other classified runtime errors |\n'
             '|---|---:|---:|\n')
    for row in cases:
        text += f"| {row['case']} | {row['known_null_pointer_diagnostic_count']} | 0 |\n"
    text += ('\nCounts cover the accepted native segment logs, including initialization. '
             'Interrupted originals remain separately preserved. The reference match '
             'alone does not prove harmlessness; complete population/ID/weight/cohort '
             'audits and finite-field, constraint and native-health checks are required '
             'independently. No claim that all raw logs are free of error-severity '
             'messages is made.\n')
    (ROOT / 'reports/TRANSPORT_DIAGNOSTICS.md').write_text(text)
    return result


if __name__ == '__main__':
    audit()
