"""Native-slot synchronized x-y movies with fixed initial-radius colours."""
import argparse
import math
import subprocess
import numpy as np
from cfg import *
from compare_cases import NAMES,LABELS

GROUPS={'paired':[0,1],'scaling':[0,2,3],'0':[1],'1':[2],'2':[3]}
RANGE=[0,8.936231120010849];FPS=15

def entry(path):return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))

def datasets():
    sources=[BASE/'particle_movie_20261017']+[ROOT/'movie_cache'/c['id']for c in CASES]
    result=[]
    for cache in sources:
        receipt=load(cache/'DATA_OK.json');path=cache/'trajectories.npz'
        assert sha(path)==receipt['cache']['sha256']
        with np.load(path,allow_pickle=False)as values:d={k:values[k].copy()for k in values.files}
        assert d['positions'].shape==(501,20000,3) and np.isfinite(d['positions']).all()
        assert np.array_equal(np.rint(d['times']/2).astype(int),np.arange(501))
        assert np.all((d['initial_radius']>=0)&(d['initial_radius']<=RANGE[1]))
        d.update(cache_path=path,receipt_path=cache/'DATA_OK.json');result.append(d)
    return result

def render(group,attempt=None):
    require_job();all_data=datasets();indices=GROUPS[group]
    limit=max(16.,2*math.ceil((max(float(np.abs(d['positions'][:,:,:2]).max())for d in all_data)+.5)/2))
    # Every individual and comparison uses the same axis range determined from
    # all four displayed samples; this changes visualization only.
    assert attempt in (None,'layout_v2')
    if attempt:assert len(indices)==1,'Layout repair applies only to individual movies'
    suffix=('_'+attempt) if attempt else ''
    dest=ROOT/('movie_render'+suffix)/group;dest.mkdir(parents=True,exist_ok=False)
    frames=dest/'frames';frames.mkdir()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from PIL import Image
    count=len(indices);fig,axes=plt.subplots(1,count,figsize=(10*count,9),squeeze=False)
    fig.subplots_adjust(left=.065/count,right=(.86 if count==1 else .90),bottom=.13,top=.84,wspace=.16)
    plt.rcParams.update({'font.size':12})
    dots=[];clocks=[];orders=[]
    for ax,index in zip(axes[0],indices):
        d=all_data[index];order=np.random.default_rng(2026101706).permutation(20000);orders.append(order)
        ax.set(xlim=(-limit,limit),ylim=(-limit,limit),aspect='equal',xlabel='$x/M$',ylabel='$y/M$')
        ax.grid(color='#e8ebed',linewidth=.6,zorder=0)
        dots.append(ax.scatter(d['positions'][0,order,0],d['positions'][0,order,1],c=d['initial_radius'][order],
            s=1.6,cmap='viridis',norm=Normalize(*RANGE),linewidths=0,alpha=1,zorder=2))
        clocks.append(ax.set_title('',fontsize=15))
    cb=fig.colorbar(dots[0],cax=fig.add_axes([(.89 if count==1 else .925),.25,.015/count,.5]));cb.set_label('Initial 3D coordinate radius $r_0/M$')
    cb.set_ticks([0,2,4,6,8,RANGE[1]]);cb.set_ticklabels(['0','2','4','6','8','8.94'])
    fig.text(.5,.965,'ST $y_c=0.819$ · proper profile · fixed mesh · stratified seed 1',ha='center',fontsize=17)
    clock=fig.text(.5,.91,'',ha='center',fontsize=16)
    fig.text(.5,.065,'20,000 fixed IDs per panel | x–y projection through all z | fixed origin and common axes',ha='center',fontsize=12)
    fig.text(.5,.032,'Colour remains the initial radius; paired panel displays 10,000 complete antipodal pairs',ha='center',fontsize=12)
    for k in range(501):
        clock.set_text(f'Nominal saved time = {2*k} M | {2*k/P:.3f} surface periods')
        for dot,title,index,order in zip(dots,clocks,indices,orders):
            d=all_data[index];dot.set_offsets(d['positions'][k,order,:2])
            title.set_text(f'{LABELS[index]} | actual t = {d["times"][k]:.3f} M')
        if k==0:
            fig.canvas.draw()
            label_bbox=cb.ax.yaxis.label.get_window_extent()
            assert 0<=label_bbox.x0<label_bbox.x1<=fig.bbox.width,'Colour-bar label cropped'
        fig.savefig(frames/f'frame_{k:04d}.png',dpi=120)
        if k%25==0:print('RENDER',group,k,flush=True)
    plt.close(fig)
    target=ROOT/'movies'/('s6_'+group+'_xy_initial_radius'+suffix+'.mp4');target.parent.mkdir(exist_ok=True)
    assert not target.exists()
    subprocess.run(['ffmpeg','-nostdin','-v','error','-framerate',str(FPS),'-i',str(frames/'frame_%04d.png'),
        '-c:v','libx264','-threads','2','-preset','medium','-crf','18','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(target)],check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=codec_name,width,height,nb_read_frames,r_frame_rate,pix_fmt:format=duration','-of','json',str(target)],text=True))
    stream=probe['streams'][0]
    assert int(stream['nb_read_frames'])==501 and stream['r_frame_rate']=='15/1'
    assert (stream['width'],stream['height'])==(1200*count,1080)
    assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p'
    assert abs(float(probe['format']['duration'])-501/FPS)<.02
    subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','2','-i',str(target),'-f','null','-'],check=True)
    selected=[0,150,300,400,419,500];decoded=dest/'decoded';decoded.mkdir()
    expr='+'.join(f'eq(n\\,{k})'for k in selected)
    subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','2','-i',str(target),'-vf','select='+expr,
        '-vsync','0',str(decoded/'decoded_%02d.png')],check=True)
    images=sorted(decoded.glob('*.png'));assert len(images)==6
    width=1200;panel_height=round(1080*width/(1200*count));sheet=Image.new('RGB',(width,panel_height*6),'white')
    for j,path in enumerate(images):
        with Image.open(path)as im:sheet.paste(im.convert('RGB').resize((width,panel_height),Image.Resampling.LANCZOS),(0,j*panel_height))
    sheet.save(dest/'contact_sheet.png')
    atomic(dest/'RENDER_OK.json',dict(utc=now(),source=source(),job=os.environ['SLURM_JOB_ID'],
        group=group,attempt=attempt,colourbar_label_inside_canvas=True,cases=[NAMES[i]for i in indices],movie=entry(target),full_decode_passed=True,ffprobe=probe,
        display_count_per_panel=20000,frames=501,fps=FPS,color_range_M=RANGE,view_limit_M=limit,
        data=[entry(all_data[i]['cache_path'])for i in indices],review_frames=selected,
        contact_sheet=entry(dest/'contact_sheet.png'),actual_encoded_visual_review_pending=True,
        synchronized_by='Matching nominal 2M output slots; each panel shows its actual native time; no trajectory interpolation',
        clipping=0,draw_order_seed=2026101706))
    print('MOVIE_RENDER_OK',group,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('group',choices=list(GROUPS))
    p.add_argument('--attempt',choices=['layout_v2'])
    args=p.parse_args();render(args.group,args.attempt)
