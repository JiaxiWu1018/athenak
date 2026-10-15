"""Render matched particle movies and initial-position panels from verified caches."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from common import ROOT, atomic, require_job, sha

BLUE = '#246b9e'
INK = '#26313d'
LIMIT = 12
FPS = 10
MOVIE_LABELS = ('proper profile + stratified', 'legacy profile + stratified')
INITIAL_LABELS = ('Monte Carlo', 'Stratified', 'Cell-aware', 'Random-velocity ablation')
STEMS = {'3d':'stratified_proper_vs_legacy_3d_seed1_0-300M',
         '2d':'stratified_proper_vs_legacy_2d_seed1_0-300M'}


def style(ax, three_d=False):
    ax.set_xlim(-LIMIT, LIMIT); ax.set_ylim(-LIMIT, LIMIT)
    ax.set_xlabel('x / M', color=INK); ax.set_ylabel('y / M', color=INK)
    ax.set_xticks([-12,-6,0,6,12]); ax.set_yticks([-12,-6,0,6,12])
    ax.tick_params(labelsize=9, colors=INK)
    if three_d:
        ax.set_zlim(-LIMIT, LIMIT); ax.set_zticks([-12,-6,0,6,12]); ax.set_zlabel('z / M', color=INK)
        ax.set_box_aspect((1,1,1)); ax.set_proj_type('ortho'); ax.view_init(elev=22, azim=-55)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor('white'); axis.pane.set_edgecolor('#dddddd')
            axis._axinfo['grid']['color']='#e4e7ea'
            axis._axinfo['grid']['linewidth']=0.5
    else:
        ax.set_aspect('equal'); ax.grid(False)


def draw(ax, view, cloud, counts, norm):
    if view=='3d':
        ax.scatter(*cloud.T, s=0.8, c=BLUE, alpha=0.20, linewidths=0,
                   depthshade=False, rasterized=True)
    else:
        return ax.imshow(np.ma.masked_equal(counts.T,0), origin='lower',
                         extent=(-LIMIT,LIMIT,-LIMIT,LIMIT), cmap='Blues',
                         norm=norm, interpolation='nearest')


def movie_frame(data, frame, view, path, norm):
    fig = plt.figure(figsize=(12.8,6.8), dpi=120, facecolor='white')
    fig.subplots_adjust(left=0.06,right=0.91 if view=='2d' else 0.98,bottom=0.14,top=0.79,wspace=0.23)
    for column, label in enumerate(MOVIE_LABELS):
        ax = fig.add_subplot(1,2,column+1,projection='3d' if view=='3d' else None)
        style(ax,view=='3d');ax.set_title(label,fontsize=14,pad=14)
        image = draw(ax,view,data['movie_clouds'][column,frame],data['movie_counts'][column,frame],norm)
    fig.text(0.5,0.945,f"ST cluster particles  |  t = {data['times'][0,frame]:.0f} M",ha='center',fontsize=18,color=INK)
    fig.text(0.5,0.892,'Seed 1  |  N = 1,500,000 per run  |  fixed origin and common axes',ha='center',fontsize=11,color=INK)
    if view=='2d':
        color_ax=fig.add_axes((0.94,0.20,0.014,0.53))
        fig.colorbar(image,cax=color_ax,label='Particles per projection pixel')
        note='x–y projection through all z; all particles; fixed logarithmic count scale across both panels and all times'
    else:
        note='20,000 fixed original IDs per panel; unweighted display sample; identical stationary camera'
    fig.text(0.5,0.047,note,ha='center',fontsize=10,color=INK)
    fig.savefig(path,dpi=120);plt.close(fig)


def initial_figure(data, view, destination):
    fig = plt.figure(figsize=(12,11), facecolor='white')
    fig.subplots_adjust(left=0.07,right=0.90 if view=='2d' else 0.96,bottom=0.12,top=0.88,hspace=0.28,wspace=0.25)
    norm=LogNorm(vmin=1,vmax=int(data['initial_counts'].max()))
    for panel,label in enumerate(INITIAL_LABELS):
        ax=fig.add_subplot(2,2,panel+1,projection='3d' if view=='3d' else None)
        style(ax,view=='3d');ax.set_title(label,fontsize=15,pad=13)
        image=draw(ax,view,data['initial_clouds'][panel],data['initial_counts'][panel],norm)
    fig.text(0.5,0.965,'Initial particle positions: proper / corrected radial profile',ha='center',fontsize=17,color=INK)
    fig.text(0.5,0.928,'t = 0 M  |  seed 1  |  N = 1,500,000 per sampler  |  common axes',ha='center',fontsize=11,color=INK)
    if view=='2d':
        fig.colorbar(image,cax=fig.add_axes((0.94,0.24,0.014,0.50)),label='Particles per projection pixel')
        note='x–y projection through all z; all particles; unweighted counts on one shared logarithmic scale.'
    else:
        note='Same 20,000 original IDs per panel; unweighted position samples; identical camera.'
    fig.text(0.5,0.065,note,ha='center',fontsize=10,color=INK)
    fig.text(0.5,0.040,'Stratified and random-velocity ablation have identical positions. Cell-aware points have unequal fixed masses.',ha='center',fontsize=10,color=INK)
    stem=destination/f'initial_positions_four_samplers_proper_seed1_{view}'
    for suffix in ('.png','.pdf'):
        fig.savefig(stem.with_suffix(suffix),dpi=200)
    plt.close(fig)
    subprocess.run(['pdftoppm','-png','-r','110','-singlefile',str(stem.with_suffix('.pdf')),
                    str(destination/f'initial_{view}_pdf_preview')],check=True)


def encode(frames, target, checks):
    subprocess.run(['ffmpeg','-nostdin','-loglevel','error','-framerate',str(FPS),
                    '-i',str(frames/'frame_%04d.png'),'-c:v','libx264','-threads','2',
                    '-preset','medium','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames',
          '-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_read_frames,r_frame_rate,pix_fmt:format=duration',
          '-of','json',str(target)],text=True))
    stream=probe['streams'][0]
    assert int(stream['nb_read_frames'])==151 and stream['r_frame_rate']=='10/1'
    assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p'
    assert stream['width']==1536 and stream['height']==816
    assert abs(float(probe['format']['duration'])-15.1)<0.02
    subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(target),'-f','null','-'],check=True)
    checks.mkdir()
    subprocess.run(['ffmpeg','-nostdin','-loglevel','error','-i',str(target),
                    '-vf',r'select=eq(n\,0)+eq(n\,50)+eq(n\,100)+eq(n\,150)',
                    '-vsync','0',str(checks/'decoded_%02d.png')],check=True)
    assert len(list(checks.glob('decoded_*.png')))==4
    return probe


def main():
    require_job();parser=argparse.ArgumentParser();parser.add_argument('receipt',type=Path)
    parser.add_argument('--preview',action='store_true');args=parser.parse_args()
    for tool in ('ffmpeg','ffprobe','pdftoppm'):
        assert shutil.which(tool),tool
    receipt=json.loads(args.receipt.read_text());cache=ROOT/receipt['cache']['path']
    assert cache.stat().st_size==receipt['cache']['bytes'] and sha(cache)==receipt['cache']['sha256']
    assert receipt['movie_cases']==['proper_stratified_s1','legacy_stratified_s1']
    assert receipt['limit_M']==LIMIT and receipt['frames']==151
    data=dict(np.load(cache,allow_pickle=False))
    assert data['movie_clouds'].shape==(2,151,20000,3)
    assert np.allclose(data['times'],np.arange(0,301,2)[None,:],atol=1e-7,rtol=0)
    assert np.array_equal(data['initial_clouds'][1],data['initial_clouds'][3])
    assert np.array_equal(data['initial_counts'][1],data['initial_counts'][3])
    work=ROOT/'particle_visualization_render'/os.environ['SLURM_JOB_ID'];work.mkdir(parents=True,exist_ok=False)
    artifacts=work/'artifacts';artifacts.mkdir()
    probes={};norm=LogNorm(vmin=1,vmax=int(data['movie_counts'].max()))
    for view in ('3d','2d'):
        initial_figure(data,view,artifacts)
        frames=work/f'frames_{view}';frames.mkdir()
        for frame in ([0,50,100,150] if args.preview else range(151)):
            movie_frame(data,frame,view,frames/f'frame_{frame:04d}.png',norm)
            if frame%25==0:print('RENDER',view,frame,flush=True)
        if not args.preview:
            target=artifacts/(STEMS[view]+'.mp4')
            probes[view]=encode(frames,target,work/f'checks_{view}')
    manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'job':os.environ['SLURM_JOB_ID'],'host':socket.gethostname(),'preview_only':args.preview,
              'source_commit':subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[2]),'rev-parse','HEAD'],text=True).strip(),
              'script_sha256':sha(Path(__file__)),'data_receipt':str(args.receipt),'data_receipt_sha256':sha(args.receipt),
              'frames':151,'fps':FPS,'duration_seconds':15.1,'view_limit_M':LIMIT,
              'camera':{'elevation_degrees':22,'azimuth_degrees':-55,'projection':'orthographic'},
              'projection':'unweighted all-particle x-y count projection, all z',
              'movie_color_norm':{'kind':'logarithmic','minimum':1,'maximum':int(data['movie_counts'].max())},
              'initial_color_norm':{'kind':'logarithmic','minimum':1,'maximum':int(data['initial_counts'].max())},
              'movie_probes':probes,'software':{'numpy':np.__version__,'matplotlib':matplotlib.__version__,
              'ffmpeg':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0]},
              'artifacts':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(artifacts.iterdir())]}
    if not args.preview:
        final=ROOT/'plots/final'
        promoted=[p for p in artifacts.iterdir() if p.suffix in ('.mp4','.pdf','.png') and 'pdf_preview' not in p.name]
        assert len(promoted)==6 and all(not (final/p.name).exists() for p in promoted)
        for p in promoted:
            target=final/p.name;shutil.copy2(p,target)
            assert sha(p)==sha(target)
        manifest['promoted_artifacts']=[{'path':str((final/p.name).relative_to(ROOT)),
                                         'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(promoted)]
    atomic(work/'RENDER_OK.json',manifest)
    print('PARTICLE_VISUAL_RENDER_OK',work,flush=True)


if __name__=='__main__':
    main()
