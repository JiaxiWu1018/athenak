"""Render verified campaign PDFs for visual inspection, inside a Slurm job.

This verifies copied artifact identities and makes previews. It does not certify
scientific conclusions or replace actual visual inspection of the figures.
"""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess

from common import ROOT, atomic, require_job, sha


def png_dimensions(path):
    with open(path, 'rb') as source:
        header = source.read(24)
    assert header[:8] == b'\x89PNG\r\n\x1a\n' and header[12:16] == b'IHDR'
    return struct.unpack('>II', header[16:24])


def main():
    require_job()
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa-receipt', type=Path, required=True)
    args = parser.parse_args()
    assert shutil.which('pdftoppm') and shutil.which('pdfinfo')
    receipt = json.loads(args.qa_receipt.read_text())
    assert sorted(r['endpoint_M'] for r in receipt['reports']) == [100, 300]
    destination = ROOT/'qa_pdf_review'/os.environ['SLURM_JOB_ID']
    destination.mkdir(parents=True, exist_ok=False)
    manifest, inputs, seen = [], [], set()
    index = ['# PDF previews for visual review', '',
             'Rendered from the exact PDFs identified by the complete-ensemble QA receipt. Rendering and integrity checks do not constitute visual or scientific acceptance.', '']
    for report in receipt['reports']:
        assert report['figure_pairs'] == 17
        artifacts = report['artifacts']
        assert len(artifacts) == 34 and len({r['path'] for r in artifacts}) == 34
        for row in artifacts:
            path = (ROOT/row['path']).resolve()
            assert path.is_relative_to(ROOT.resolve())
            assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
            inputs.append(row)
        pdfs = [r for r in artifacts if Path(r['path']).suffix == '.pdf']
        assert len(pdfs) == 17
        for row in pdfs:
            pdf = ROOT/row['path']
            png = pdf.with_suffix('.png')
            assert any(r['path'] == str(png.relative_to(ROOT)) for r in artifacts)
            info = subprocess.check_output(['pdfinfo', str(pdf)], text=True)
            pages = re.search(r'^Pages:\s+(\d+)\s*$', info, re.MULTILINE)
            assert pages and int(pages[1]) == 1
            page_size = re.search(r'^Page size:\s+([\d.]+) x ([\d.]+) pts', info,
                                  re.MULTILINE)
            assert page_size
            page_points = tuple(float(value) for value in page_size.groups())
            assert all(value > 0 for value in page_points)
            prefix = destination/pdf.parent.name/pdf.stem
            assert str(prefix) not in seen
            seen.add(str(prefix))
            prefix.parent.mkdir(exist_ok=True)
            subprocess.run(['pdftoppm', '-png', '-r', '120', '-f', '1', '-l', '1',
                            '-singlefile', str(pdf), str(prefix)], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            preview = prefix.with_suffix('.png')
            width, height = png_dimensions(preview)
            # Validate the raster against its PDF geometry. Figure widths differ;
            # an 8-inch page correctly produces only 960 pixels at 120 dpi.
            expected_pixels = tuple(math.ceil(value*120/72) for value in page_points)
            assert all(abs(actual-expected) <= 1 for actual, expected in
                       zip((width, height), expected_pixels))
            manifest.append({'source_pdf':row, 'preview':str(preview.relative_to(ROOT)),
                             'bytes':preview.stat().st_size, 'sha256':sha(preview),
                             'width':width, 'height':height,
                             'pdf_page_size_points':page_points,
                             'source_png_dimensions':png_dimensions(png), 'pdf_pages':1})
            label = pdf.parent.name+'/'+pdf.stem
            index.append(f'- {label}: [PDF preview]({preview.relative_to(destination)}) / [original PNG](../../{png.relative_to(ROOT)})')
    assert len(manifest) == 34 and len(inputs) == 68
    repo = Path(__file__).resolve().parents[2]
    atomic(destination/'INDEX.md', '\n'.join(index)+'\n')
    atomic(destination/'RENDER_OK.json', {
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'job':os.environ['SLURM_JOB_ID'], 'host':__import__('socket').gethostname(),
        'source_commit':subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'script_sha256':sha(Path(__file__)), 'qa_receipt':str(args.qa_receipt.resolve()),
        'qa_receipt_sha256':sha(args.qa_receipt),
        'tool_version':subprocess.run(['pdftoppm', '-v'], text=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, check=True).stdout,
        'input_artifacts':inputs, 'rendered':manifest,
        'scope':'Exact QA artifact identities, single PDF page, successful rasterization and PNG headers. Actual visual and scientific review still required.'})
    print('PDF_PREVIEWS_READY', destination, flush=True)


if __name__ == '__main__':
    main()
