#!/usr/bin/env python3
"""Fixed sixfold central zoom of the accepted Session010 t0--98 3D movie."""
from __future__ import annotations
import argparse,csv,datetime as dt,hashlib,importlib.util,json,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from PIL import Image,ImageDraw


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--session',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--view-limit',type=float,default=4.0)
    args=ap.parse_args()
    assert os.environ.get('SLURM_JOB_ID'),'Slurm allocation required'
    s=args.session.resolve();out=args.out.resolve();limit=args.view_limit
    assert 3.0<=limit<=24.0
    parent=s/'movies/particles3d_to98_20261019_v1'
    assert out!=parent and not (out/'evidence/COMPLETE.json').exists()
    for name in ('evidence','frames','sequence','qa'): (out/name).mkdir(parents=True,exist_ok=True)
    # Verify the existing release before reading its approved geometry/time mapping.
    sealed=[]
    for line in (parent/'SHA256SUMS.txt').read_text().splitlines():
        expected,name=line.split(maxsplit=1);p=parent/name.lstrip('*')
        assert digest(p)==expected,('Parent release changed',name)
        sealed.append(name)
    accepted=json.loads((parent/'evidence/ACCEPTED.json').read_text())
    assert accepted['status']=='accepted'
    records=json.loads((parent/'FRAMES.json').read_text())
    oldmeta=json.loads((parent/'evidence/COMPLETE.json').read_text())
    assert len(records)==50 and records[0]['time']==0 and records[-1]['time']==98
    oldpath=parent/'reference/cca_render_gi_3d_movie.py'
    spec=importlib.util.spec_from_file_location('s7_zoom_style',oldpath)
    style=importlib.util.module_from_spec(spec);spec.loader.exec_module(style)
    horizon_tables=[np.genfromtxt(s/'reduced_data/vista_20261018'/f'accepted_horizon_{i}.csv',delimiter=',',names=True) for i in range(3)]
    newrecords=[];previous=None;maximum_extent=0.;all_projected_bounds=[]
    for rec in records:
        assert 0<=rec['time']<=98
        path=s/rec['cache'];assert digest(path)==rec['cache_sha256']
        with np.load(path,allow_pickle=False) as z:
            xyz=z['xyz'];tags=z['tag'].astype(np.int64);groups=z['group'];tm=float(z['time'])
        assert tm==rec['time'] and len(tags)==rec['cohort_alive'] and np.isfinite(xyz).all()
        current=set(tags.tolist());assert len(current)==len(tags)
        if previous is not None: assert current<=previous
        previous=current
        assert np.array_equal(groups,np.where(tags<3000000,0,np.where(tags<4000000,1,2)))
        buried=np.zeros(len(xyz),dtype=bool)
        for h in rec['horizons']:
            a=horizon_tables[h['horizon']]
            q=a[(a['cycle']==rec['cycle'])&(a['case']==rec['case'])]
            assert len(q)==1 and float(q[0]['meanradius'])==h['mean_coordinate_radius']
            assert h['center']==[float(q[0][name]) for name in ('center_x','center_y','center_z')]
            assert (h['horizon']<2)==(tm<oldmeta['common_enclosure_time'])
            radius=h['mean_coordinate_radius'];center=np.asarray(h['center'])
            extent=float(np.max(abs(center))+radius);maximum_extent=max(maximum_extent,extent)
            assert extent<limit,('AH clipped by zoom cube',tm,h['horizon'],extent)
            buried|=np.sum((xyz-center)**2,axis=1)<(1.002*radius)**2
        in_view=(abs(xyz)<=limit).all(axis=1);keep=in_view&(~buried)
        fig=plt.figure(figsize=(16,9),dpi=120,facecolor=style.BACKGROUND)
        ax=fig.add_axes([0,0,1,1],projection='3d',facecolor=style.BACKGROUND,computed_zorder=False)
        ax.set_proj_type('persp',focal_length=style.CAMERA['focal_length'])
        ax.view_init(elev=style.CAMERA['elev'],azim=style.CAMERA['azim'],roll=style.CAMERA['roll'])
        ax.set(xlim=(-limit,limit),ylim=(-limit,limit),zlim=(-limit,limit))
        ax.set_box_aspect((1,1,1));ax.set_axis_off()
        for group in (0,1,2):
            q=keep&(groups==group);st=style.GROUP_STYLES[group]
            ax.scatter(*xyz[q].T,s=st['size'],alpha=st['alpha'],color=style.GROUP_COLORS[group],edgecolors='none',linewidths=0,depthshade=False,rasterized=True,zorder=2+group*.01)
        surfaces=[]
        for h in rec['horizons']:
            sphere=style.make_sphere(tuple(h['center']),h['mean_coordinate_radius'])
            ax.plot_surface(*sphere,color='#000000',edgecolor='none',linewidth=0,antialiased=True,shade=False,alpha=1,zorder=20)
            surfaces.append((h,sphere))
        fig.canvas.draw()
        bounds=[]
        for h,sphere in surfaces:
            px,py,_=proj3d.proj_transform(*(a.ravel() for a in sphere),ax.get_proj())
            pixels=ax.transData.transform(np.c_[px,py])
            low=pixels.min(axis=0);high=pixels.max(axis=0)
            assert np.all(low>4) and high[0]<1916 and high[1]<1076,('Projected AH clips',tm,low,high)
            box=dict(horizon=h['horizon'],pixel_bbox=[*low.tolist(),*high.tolist()],diameter_pixels=(high-low).tolist())
            bounds.append(box);all_projected_bounds.append(dict(time=tm,**box))
        frame=out/'frames'/f"state_{rec['index']:04d}.png";assert not frame.exists()
        fig.savefig(frame,facecolor=style.BACKGROUND,edgecolor=style.BACKGROUND,pad_inches=0);plt.close(fig)
        new=dict(rec,parent_frame_sha256=rec['frame_sha256'],frame=str(frame.relative_to(out)),frame_sha256=digest(frame),
                 visible_counts=[int(np.sum(keep&(groups==g))) for g in range(3)],projected_horizons=bounds,view_limit=limit)
        newrecords.append(new)
        print('ZOOM',rec['index'],'t',tm,'AH',[h['horizon'] for h in rec['horizons']],flush=True)
    shutil.copy2(parent/'TIMING.csv',out/'TIMING.csv')
    schedule=list(csv.DictReader((out/'TIMING.csv').open()));assert len(schedule)==393
    lookup={r['index']:r for r in newrecords}
    for k,row in enumerate(schedule):
        rec=lookup[int(row['source_index'])]
        assert rec['time']==float(row['actual_saved_time'])<=float(row['hold_clock_Mref'])<=98
        target=out/'sequence'/f'frame_{k:05d}.png';assert not target.exists()
        target.symlink_to(Path('..')/rec['frame'])
    assert float(schedule[-1]['actual_saved_time'])==98
    save(out/'FRAMES.json',newrecords)
    ffmpeg=shutil.which('ffmpeg');ffprobe=shutil.which('ffprobe');assert ffmpeg and ffprobe
    movie=out/'jeans_session010_particles3d_t0_to98_zoom.mp4'
    command=[ffmpeg,'-n','-v','error','-framerate','12','-i',str(out/'sequence/frame_%05d.png'),'-frames:v','393','-c:v','libx264','-threads','2','-preset','slow','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(movie)]
    subprocess.run(command,check=True)
    with (out/'evidence/full_decode.log').open('w') as f:
        subprocess.run([ffmpeg,'-v','error','-i',str(movie),'-map','0:v:0','-f','null','-'],stdout=f,stderr=subprocess.STDOUT,check=True)
    probe=json.loads(subprocess.check_output([ffprobe,'-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_read_frames,nb_frames,duration','-of','json',str(movie)],text=True))
    stream=probe['streams'][0]
    assert (stream['width'],stream['height'],stream['codec_name'],stream['pix_fmt'],stream['r_frame_rate'],int(stream['nb_read_frames']))==(1920,1080,'h264','yuv420p','12/1',393)
    assert abs(float(stream['duration'])-32.75)<1e-6
    save(out/'evidence/FFPROBE.json',probe)
    selected=(0,10,20,40,58,60,62,66,98)
    contact=Image.new('RGB',(1920,3*385),(7,26,43));draw=ImageDraw.Draw(contact)
    for cell,tm in enumerate(selected):
        rec=next(r for r in newrecords if r['time']==tm)
        kk=[k for k,row in enumerate(schedule) if int(row['source_index'])==rec['index']]
        k=kk[-1] if tm==98 else kk[0]
        decoded=out/'qa'/f'decoded_t{tm:g}.png'
        subprocess.run([ffmpeg,'-n','-v','error','-i',str(movie),'-vf',f'select=eq(n\\,{k})','-frames:v','1',str(decoded)],check=True)
        img=Image.open(decoded).convert('RGB');img.thumbnail((640,360));x=(cell%3)*640;y=(cell//3)*385
        contact.paste(img,(x,y+24));draw.text((x+8,y+4),f"saved t={tm:g}; accepted AH {[h['horizon'] for h in rec['horizons']]}; +/-{limit:g}",fill='white')
    contact.save(out/'qa/representative_frames.jpg',quality=95)
    source=np.asarray(Image.open(out/'frames/state_0049.png').convert('RGB'),dtype=float)
    decoded=np.asarray(Image.open(out/'qa/decoded_t98.png').convert('RGB'),dtype=float)
    mae=float(np.mean(abs(source-decoded)));assert mae<3
    # Check the existing release remained byte-identical after rendering.
    for line in (parent/'SHA256SUMS.txt').read_text().splitlines():
        expected,name=line.split(maxsplit=1);assert digest(parent/name.lstrip('*'))==expected
    receipt=dict(status='machine checks passed; visual review pending',created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                 job=os.environ['SLURM_JOB_ID'],host=os.uname().nodename,parent_release=str(parent),parent_movie_sha256=accepted['sha256'],
                 parent_files_verified_before_after=len(sealed),parent_frames_sha256=digest(parent/'FRAMES.json'),
                 movie=movie.name,sha256=digest(movie),bytes=movie.stat().st_size,
                 field_of_view=[-limit,limit],linear_zoom_factor=24/limit,camera=style.CAMERA,
                 style='Original navy/purple/orange/cyan, marker size/opacity unchanged; hidden axes/text',
                 horizon_layer='Schematic opaque AH glyphs rendered above particle collections for visibility. Measured centers/radii unchanged; layer order is not physical ray-traced occlusion.',
                 source_states=50,encoded_frames=393,actual_time_range=[0,98],fps=12,duration_seconds=32.75,resolution=[1920,1080],
                 same_timing_as_parent=digest(out/'TIMING.csv')==digest(parent/'TIMING.csv'),
                 geometry='Unchanged measured centers/mean radii from parent; independently joined to accepted CSV by exact case/cycle; gaps preserved',
                 strict_common_time=oldmeta['common_enclosure_time'],first_saved_common_time=66,
                 maximum_AH_coordinate_extent=maximum_extent,projected_horizon_bounds=all_projected_bounds,
                 clipping='All displayed AH spheres inside fixed cube and projected image bounds',
                 particles='Same fixed cohort and actual xyz; surviving tags nested, no interpolation/replacement; only central cube displayed',
                 full_decode_passed=True,final_saved_time=98,last_decoded_frame_RGB_MAE=mae,
                 scripts={str(Path(__file__).relative_to(s)):digest(Path(__file__)),str(oldpath.relative_to(s)):digest(oldpath)},
                 native_source='386870f88f60e25e1a8fac76fe04a071b87a01cf',source_head=subprocess.check_output(['git','-C',str(s/'code'),'rev-parse','HEAD'],text=True).strip(),
                 software=dict(python=sys.version,numpy=np.__version__,matplotlib=matplotlib.__version__,ffmpeg=ffmpeg),
                 encode_command=command,native_simulation_modified=False,ah_repair_implemented=False)
    save(out/'evidence/COMPLETE.json',receipt)
    (out/(movie.name+'.sha256')).write_text(receipt['sha256']+'  '+movie.name+'\n')
    print('ZOOM MOVIE MACHINE CHECKS PASSED',movie,flush=True)


if __name__=='__main__':main()
