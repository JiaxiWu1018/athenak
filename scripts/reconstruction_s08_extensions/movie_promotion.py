"""Promote a completed primary movie only when its data and renderer match."""
import shutil
import sys

import matplotlib
import numpy as np

from common import *
from extension_resources import EXT


def digest(path):
    row = receipt(Path(path))
    return dict(bytes=row['bytes'], sha256=row['sha256'])


def fingerprint(data):
    ident = data['case']['id']
    directory = ROOT / 'reduced' / ident
    tables = {name: digest(directory / name) for name in
              ['summary.json', 'shells.csv', 'history.csv', 'ah.csv', 'cohort_index.csv']}
    shape = ROOT / 'runs' / ident / (ident + '.horizon_shape_0.txt')
    software = {name: digest(Path(__file__).with_name(name)) for name in
                ['products.py', 'movie_helpers.py', 'movie_promotion.py', 'publication_style.py',
                 'run_segment.py', 'common.py']}
    cohorts = [dict(time=row['time'], file=row['file'], **digest(row['file']))
               for row in data['cohort_index']]
    return dict(case=data['case'], tables=tables, cohorts=cohorts,
                sample_profile=digest(ROOT / 'evidence/samples' / (ident + '.json')),
                horizon_shape=digest(shape) if shape.exists() else None,
                software=software, python=sys.version,
                numpy=np.__version__, matplotlib=matplotlib.__version__)


def promote(data):
    if data['case']['mode'] != 'inward0020':
        return None
    ready = EXT / 'primary_movie/evidence/PRIMARY_MOVIE_READY.json'
    if not ready.exists():
        return None
    saved = load(ready)
    if saved['fingerprint'] != fingerprint(data):
        atomic(ROOT / 'evidence/PRIMARY_MOVIE_PROMOTION_NOT_USED.json', dict(
            job=job(), utc=now(), source=str(ready),
            reason='Current movie inputs or renderer differ; render a new individual movie'))
        return None
    product = saved['product']
    assert product['full_decode']

    def copy_verified(row, destination):
        source = Path(row['path'])
        assert digest(source) == dict(bytes=row['bytes'], sha256=row['sha256'])
        if destination.exists():
            assert digest(destination) == digest(source), str(destination)
        else:
            shutil.copy2(source, destination)
        result = receipt(destination)
        assert result['sha256'] == row['sha256'] and result['bytes'] == row['bytes']
        return result

    result = dict(product)
    result.update(copy_verified(product, ROOT / 'movies' / Path(product['path']).name))
    result['decoded_previews'] = [dict(row, **copy_verified(
        row, ROOT / 'plots' / Path(row['path']).name)) for row in product['decoded_previews']]
    result.update(promoted_from=str(ready), source_ready=receipt(ready), source_render_job=saved['job'],
                  promotion_job=job(), source_fingerprint_matched=True)
    atomic(ROOT / 'evidence/PRIMARY_MOVIE_PROMOTION.json', dict(
        utc=now(), job=job(), source_ready=receipt(ready),
        inputs_and_renderer_matched=True, encoded_copy_verified=True,
        product=result, source_decode_reused=True,
        actual_visual_acceptance='Recorded separately against the encoded file and decoded previews'))
    return result


def main():
    from products import assemble, movies
    from publication_style import style
    job()
    data = assemble(['inward0020'])
    style()
    products = movies(data, comparison=False, allow_promotion=False)
    assert len(products) == 1 and products[0]['full_decode']
    atomic(ROOT / 'evidence/PRIMARY_MOVIE_READY.json', dict(
        utc=now(), job=job(), fingerprint=fingerprint(data[0]),
        product=products[0], actual_visual_acceptance=False))
