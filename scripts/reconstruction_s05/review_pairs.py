"""Make paired PNG/PDF contact sheets for actual visual review in Slurm."""
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

from common import ROOT, atomic, require_job, sha


def main():
    require_job()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    rows = receipt['rendered']
    assert len(rows) == 34
    destination = ROOT/'qa_pairs'/os.environ['SLURM_JOB_ID']
    destination.mkdir(parents=True, exist_ok=False)
    output = []
    for start in range(0, len(rows), 2):
        panels = []
        for row in rows[start:start+2]:
            pdf = ROOT/row['source_pdf']['path']
            preview = ROOT/row['preview']
            assert sha(preview) == row['sha256']
            png = pdf.with_suffix('.png')
            source = next(r for r in receipt['input_artifacts'] if r['path']==str(png.relative_to(ROOT)))
            assert sha(png) == source['sha256']
            pair = []
            for path in (png, preview):
                with Image.open(path) as image:
                    image.load()
                    pair.append(ImageOps.contain(image.convert('RGB'), (1200, 1600)))
            height = max(image.height for image in pair)
            panel = Image.new('RGB', (2400, height+40), 'white')
            draw = ImageDraw.Draw(panel)
            label = pdf.parent.name+'/'+pdf.stem
            draw.text((10, 12), label+' | original PNG', fill='black')
            draw.text((1210, 12), label+' | rendered PDF', fill='black')
            for column, image in enumerate(pair):
                panel.paste(image, (column*1200, 40))
            panels.append(panel)
        sheet = Image.new('RGB', (2400, sum(p.height for p in panels)), 'white')
        top = 0
        for panel in panels:
            sheet.paste(panel, (0, top))
            top += panel.height
        path = destination/f'pair_{start//2:02d}.png'
        sheet.save(path)
        output.append({'path':str(path.relative_to(ROOT)), 'sha256':sha(path),
                       'sources':[r['source_pdf']['path'] for r in rows[start:start+2]]})
    atomic(destination/'PAIRS_READY.json', {'job':os.environ['SLURM_JOB_ID'],
           'receipt_sha256':sha(args.receipt), 'script_sha256':sha(Path(__file__)),
           'outputs':output, 'scope':'Contact sheets only; actual inspection still required.'})
    print('PAIRS_READY', destination)


if __name__ == '__main__':
    main()
