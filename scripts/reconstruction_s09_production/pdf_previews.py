"""Rasterize the actual exported PDFs for visual inspection, in Slurm."""
import shutil
import time
from common import *

def main():
    job();assert shutil.which('pdftoppm'),'PDF rasterizer unavailable'
    destination=ROOT/'plots/pdf_preview';destination.mkdir(exist_ok=True)
    rows=[]
    for pdf in sorted((ROOT/'plots').glob('*.pdf')):
        output=destination/(pdf.stem+'.png');marker=destination/(pdf.stem+'.json')
        source=receipt(pdf)
        if marker.exists() and load(marker)['source']['sha256']==source['sha256']:
            previous=load(marker);assert sha(output)==previous['preview']['sha256']
            rows.append(previous);continue
        if output.exists():
            archive=destination/'superseded';archive.mkdir(exist_ok=True)
            stamp=f'{job()}_{time.time_ns()}'
            atomic(archive/(pdf.stem+'_'+stamp+'.json'),receipt(output))
            output.rename(archive/(pdf.stem+'_'+stamp+'.png'))
            if marker.exists():marker.rename(archive/(pdf.stem+'_'+stamp+'_prior.json'))
        cmd(['pdftoppm','-singlefile','-png','-scale-to','1800',pdf,output.with_suffix('')])
        row=dict(utc=now(),job=job(),source=source,preview=receipt(output),actual_visual_acceptance=False)
        atomic(marker,row);rows.append(row)
    assert rows
    atomic(ROOT/'evidence/PDF_PREVIEWS.json',dict(utc=now(),job=job(),files=rows,actual_visual_acceptance=False))

if __name__=='__main__':main()
