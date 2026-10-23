"""Render actual completed baseline data as a labeled, non-scientific layout test."""
import importlib.util,shutil,sys
from common import *
from extension_resources import EXT,admission

def main():
    job();admission();base=Path(os.environ['ST8X_BASE']);site=EXT/'dependencies/site'
    required=['matplotlib','imageio_ffmpeg','pypdfium2']
    if not all(importlib.util.find_spec(name) for name in required):
        site.mkdir(parents=True,exist_ok=True)
        cmd(['python3','-m','pip','install','--target',site,'numpy==1.26.4','matplotlib==3.9.4','imageio-ffmpeg==0.6.0','pypdfium2==4.30.0'])
        sys.path.insert(0,str(site))
    from reduce import csv_rows
    from products import figures
    from movie_helpers import load_case,panel,encode
    import matplotlib,matplotlib.pyplot as plt,numpy as np,pypdfium2
    for name in ['plots','movies','reports','reduced/yc0500','evidence/samples']:(ROOT/name).mkdir(parents=True,exist_ok=True)
    for p in (base/'reduced/yc0500').iterdir():
        if p.is_file():shutil.copy2(p,ROOT/'reduced/yc0500'/p.name)
    shutil.copy2(base/'evidence/samples/yc0500.json',ROOT/'evidence/samples/yc0500.json')
    case=dict(id='yc0500',yc=.5,unstable=True,mode='render_test',title='Original reference — render test')
    d={name:csv_rows(base/'reduced/yc0500'/(name+'.csv')) for name in ['shells','mass','history','ah']}
    d.update(case=case,summary=load(base/'reduced/yc0500/summary.json'));figures([d])
    data=load_case(case);fig=plt.figure(figsize=(10,6),layout='constrained');g=fig.add_gridspec(2,2,width_ratios=[1.1,1]);axes=[fig.add_subplot(g[:,0]),fig.add_subplot(g[0,1]),fig.add_subplot(g[1,1])]
    p=panel(fig,axes,data);fig.suptitle('Render test only:3 actual reference frames at0,150,300M')
    result=encode('reference_render_test',fig,[(data,p)],np.array([0,150,300.]))
    for path in (ROOT/'plots').glob('*.pdf'):
        doc=pypdfium2.PdfDocument(str(path));doc[0].render(scale=1.5).to_pil().save(path.with_name(path.stem+'_PDF_preview.png'))
    atomic(EXT/'evidence/RENDER_SMOKE_OK.json',dict(job=job(),utc=now(),runtime=dict(numpy=np.__version__,matplotlib=matplotlib.__version__),movie=result,scope='Actual original300M reference only,3 widely spaced test frames. Not an extension result or physical-cadence movie. Final scientific/rendered acceptance stillrequired.'))
if __name__=='__main__':main()
