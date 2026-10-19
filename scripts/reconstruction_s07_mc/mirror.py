"""Allocated thin delivery mirror: explicit allowed products; never native output trees."""
import subprocess
from cfg import *
def main():
    require_job();assert ROOT==PERSEUS
    for name in ['reports','plots','comparison','reduced','movies']:
        dest=ROOT/name;dest.mkdir(exist_ok=True)
        subprocess.run(['rsync','-a',f'hpcfund.amd.com:{AMD}/{name}/',str(dest)+'/'],check=True)
    target=ROOT/'review';target.mkdir(exist_ok=True)
    subprocess.run(['rsync','-a',f'hpcfund.amd.com:{AMD}/review/figures/',str(target/'figures')+'/'],check=True)
    for group in ['0','1','2','3','scaling']:
        dest=target/'movies'/group;dest.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rsync','-a','--include=/decoded/','--include=/decoded/*.png','--include=/contact_sheet.png','--include=/RENDER_OK.json','--exclude=*',f'hpcfund.amd.com:{AMD}/movie_render/{group}/',str(dest)+'/'],check=True)
    forbidden=[p for folder in ['runs','generation','movie_cache','movie_render']if (p:=ROOT/folder).exists()]
    assert not forbidden,forbidden
    files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p))for name in ['reports','plots','comparison','reduced','movies','review']for p in sorted((ROOT/name).rglob('*'))if p.is_file()]
    atomic(ROOT/'evidence/THIN_MIRROR.json',dict(utc=now(),job=os.environ['SLURM_JOB_ID'],files=files,
        native_data_transferred=False,native_data_host='AMD only',policy='Whitelisted compact reductions, reports, figures, movies and selected actual encoded review frames.'))
    print('THIN_MIRROR_OK',len(files),flush=True)
if __name__=='__main__':main()
