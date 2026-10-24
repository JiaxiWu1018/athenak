"""Accounting and artifact-gate failure scenarios; no synthetic science artifacts."""
from accounting import summarize
from delivery import checked_visual


def main():
    registry = dict(jobs=[dict(job_id=str(i), kind='test', partition='fixture', nodes=1,
                              requested_seconds=7200) for i in range(1, 7)])
    scheduler = ('1|COMPLETED|0:0|1800|1|start|end\n'
                 '2|FAILED|1:0|60|1|start|end\n'
                 '3|CANCELLED|0:0|0|1|Unknown|end\n'
                 '4|PENDING|0:0|0|1|Unknown|Unknown\n'
                 '5|RUNNING|0:0|5|1|start|Unknown\n')
    result = summarize(registry, scheduler)
    assert result['jobs'][0]['exact_node_hours'] == .5
    assert result['jobs'][1]['exact_node_hours'] == 1 / 60
    assert result['jobs'][2]['exact_node_hours'] == 0
    for row in result['jobs'][3:]:
        assert row['upper_node_hours'] == 2 and row['exact_node_hours'] is None
    assert result['unresolved_job_ids'] == ['4', '5', '6']
    pdfs, movies = {'figure.pdf': 'pdfhash'}, {'movie': 'moviehash'}
    good = dict(actual_visual_acceptance=True, reviewer='fixture', findings=['fixture'],
                pdf_sha256=pdfs, movie_sha256=movies,
                encoded_frames_actually_inspected={'movie': [0, 1]})
    checked_visual(good, pdfs, movies)
    failures = 0
    for field, value in [('actual_visual_acceptance', False),
                         ('pdf_sha256', {'figure.pdf': 'tampered'}),
                         ('movie_sha256', {'movie': 'tampered'}),
                         ('encoded_frames_actually_inspected', {'movie': []})]:
        trial = dict(good, **{field: value})
        try:
            checked_visual(trial, pdfs, movies)
        except AssertionError:
            failures += 1
        else:
            raise AssertionError('Unsafe artifact gate accepted ' + field)
    assert failures == 4
    print('PASS: pending/active/missing scheduler records retain bounds; altered or uninspected artifacts are rejected')


if __name__ == '__main__':
    main()
