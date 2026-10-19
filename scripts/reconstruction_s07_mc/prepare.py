"""Full-count seed1 MC generation; prefix, mass, angular and monoenergy QA."""
import subprocess
import numpy as np
from cfg import *
import generate as G

def main():
    require_job()
    assert sha(ROOT/'protected/athena')==EXE_SHA
    import matplotlib,PIL,imageio_ffmpeg,pypdfium2
    atomic(ROOT/'evidence/ENVIRONMENT_OK.json',dict(utc=now(),job=os.environ['SLURM_JOB_ID'],
        numpy=np.__version__,matplotlib=matplotlib.__version__,pillow=PIL.__version__,
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe(),pdf_renderer='pypdfium2',native_data_policy='AMD only'))
    p=dict(np.load(ROOT/'samples/model.npz'));previous=None
    for case in CASES:
        folder=ROOT/'generation'/case['id'];(folder/'samples').mkdir(parents=True,exist_ok=False)
        G.ROOT=folder
        data=G.generate_one(p,'proper','mc',1,n=case['n'])
        generated=folder/'samples/proper_mc_s1.bin'
        receipt=load(folder/'evidence/samples/proper_mc_s1.json')
        if previous is not None:
            _,old=sample(previous)
            for start in range(0,len(old),100000):
                assert np.array_equal(old[start:start+100000,1:7],data[start:min(start+100000,len(old)),1:7])
            assert np.allclose(old[:,7]/data[:len(old),7],2,rtol=1e-14,atol=0)
        path=ROOT/'samples'/(case['id']+'.bin');assert not path.exists()
        generated.rename(path)
        receipt.update(case=case,utc=now(),job=os.environ['SLURM_JOB_ID'],source=source(),
            seed=1,rng='numpy.Generator(PCG64(3001))',nested_previous=str(previous)if previous else None,
            nested_positions_momenta_verified=True,paired=False,recentered=False)
        atomic(ROOT/'evidence/samples'/(case['id']+'.json'),receipt)
        print('SAMPLE_OK',case['id'],receipt['com'],flush=True)
        previous=path;del data
    atomic(ROOT/'evidence/SAMPLES_OK.json',dict(utc=now(),job=os.environ['SLURM_JOB_ID'],source=source(),
        cases=CASES,all_counts_nested=True,all_total_rest_masses_equal=True))
if __name__=='__main__':main()
