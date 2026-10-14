"""Exact decks for the fixed mesh and 24 paired configurations."""
import re
from common import ROOT, AMD, N, CASES, atomic

def make(case,radial,method,seed,n=N,end=300,external=True,user=True):
    text=(ROOT/'inputs/s9_reference.athinput').read_text()
    text=text[text.index('<job>'):]
    text='<comment>\nproblem = Reconstruction Session5 sampling comparison\nnote = Fixed N and mesh; live Z4c; radial measure and sampler explicit below.\n\n'+text
    text=text.replace('-16384','-128').replace('16384','128')
    start=text.index('<refined_region1>'); stop=text.index('<time>',start)
    refined=''
    for lev,half in ((1,64),(2,32)):
        refined+=f'<refined_region{lev}>\nlevel = {lev}\n'
        for d in range(1,4):refined+=f'x{d}min = -{half}\nx{d}max = {half}\n'
        refined+='\n'
    text=text[:start]+refined+text[stop:]
    text=re.sub(r'tlim = .*',f'tlim = {end}',text)
    text=re.sub(r'basename = .*',f'basename = {case}',text)
    text=re.sub(r'cluster_n = .*',f'cluster_n = {n}',text)
    text=re.sub(r'cluster_seed = .*',f'cluster_seed = {seed}',text)
    text=text[:text.index('<output1>')]
    text=text.replace('<time>','<time>\nevaluate_initial_constraints = '+str(user).lower(),1)
    sampling={'mc':'random','stratified':'stratified','cell':'cell','ablation':'stratified'}[method]
    velocity='stratified' if method=='stratified' else 'random'
    if external:
        text+=f'cluster_sampling = {sampling}\ncluster_radial_measure = {radial}\ncluster_velocity_sampling = {velocity}\ncluster_sample_file = {AMD}/samples/{case}.bin\n'
    if user:text+='user_hist = true\n'
    outputs=[('hst',None,2.,{}),('pvtk','prtcl_all',2.,{}),
             ('bin','z4c',2.,{'slice_x3':'0.0','id':'z4c2d'}),
             ('bin','tmunu',2.,{'slice_x3':'0.0','id':'tme2d'}),
             ('bin','con',2.,{'slice_x3':'0.0','id':'con2d'}),
             ('bin','tmunu',100.,{'id':'tme3d'}),
             ('bin','con',100.,{'id':'con3d'}),('rst',None,100.,{})]
    for i,(ft,var,dt,extras) in enumerate(outputs,1):
        text+=f'\n<output{i}>\nfile_type = {ft}\ndt = {dt}\n'
        if var:text+=f'variable = {var}\n'
        if ft=='hst':text+='data_format = %23.15e\n'
        for k,v in extras.items():text+=f'{k} = {v}\n'
    return text

def main():
    for case,rad,method,seed in CASES:atomic(ROOT/'inputs'/(case+'.athinput'),make(case,rad,method,seed))
    # Small default-mode regression; N changes only in this implementation gate.
    atomic(ROOT/'inputs/default_gate.athinput',make('default_gate','legacy','mc',1,n=4096,end=2,external=False,user=False))
    atomic(ROOT/'evidence/cases.json',[dict(case=c,radial=r,method=m,seed=s) for c,r,m,s in CASES])

if __name__=='__main__':main()
