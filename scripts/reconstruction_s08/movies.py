"""Five fixed-ID movies and a synchronized comparison, encoded/decoded on AMD."""
import argparse,csv,math,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from common import *
from run_segment import cohort
from reduce import csv_rows
from plots import style,array,shell_limits

def equator_basis(phi,lmax=6):
    # Exact real-harmonic convention in utils/spherical_harm.hpp and FastFlow.
    rows=[];fac=math.factorial
    for l in range(lmax+1):
        for m in range(l+1):
            w=sum((-1.)**k*math.cos(math.pi/4)**(2*l+m-2*k)*math.sin(math.pi/4)**(2*k-m)/
              (fac(l+m-k)*fac(l-k)*fac(k)*fac(k-m)) for k in range(m,l+1))
            w*=math.sqrt((2*l+1)/(4*math.pi))*fac(l)*math.sqrt(fac(l+m)*fac(l-m))
            if m==0:rows.append(np.full_like(phi,w))
            else:rows.extend([math.sqrt(2)*w*np.cos(m*phi),math.sqrt(2)*w*np.sin(m*phi)])
    return np.array(rows)

def shapes(path):
    out=[];time=None
    if not path.exists():return out
    for line in path.read_text().splitlines():
        if line.startswith('#'):
            match=re.search(r'Time\s*=\s*([-+0-9.eE]+)',line)
            if match:time=float(match.group(1))
        elif line.strip():
            v=np.fromstring(line,sep=' ');assert len(v)==49 and np.isfinite(v).all()
            assert time is not None;out.append((time,v));time=None
    return out

def load_case(c):
    d=ROOT/'reduced'/c['id'];s=load(d/'summary.json');assert s['complete']
    with (d/'cohort_index.csv').open() as f:idx=list(csv.DictReader(f))
    ts=np.array([float(r['time']) for r in idx]);assert np.all(np.diff(ts)>0)
    _,_,mask,initial=cohort(idx[0]['file']);assert mask.all()
    radius=np.linalg.norm(initial[:,1:4],axis=1);R=load(ROOT/'evidence/samples'/f"{c['id']}.json")['Riso']
    return dict(case=c,summary=s,index=idx,times=ts,radius=radius,R=R,
      shell=csv_rows(d/'shells.csv'),history=csv_rows(d/'history.csv'),
      ah=csv_rows(d/'ah.csv') if c['unstable'] else [],
      shapes=shapes(ROOT/'runs'/c['id']/f"{c['id']}.horizon_shape_0.txt"))

def outline(d,t,phi,basis):
    available=[r for r in d['ah'] if r['time']<=t+1e-10]
    if not available:return [],[]
    latest=available[-1]
    if latest['accepted']!=1 or t-latest['time']>.75:return [],[]
    valid=[s for s in d['shapes'] if abs(s[0]-latest['time'])<.001]
    if not valid:return [],[]
    radii=valid[-1][1]@basis;assert np.isfinite(radii).all() and (radii>0).all()
    return radii*np.cos(phi),radii*np.sin(phi)

def spatial_window(d):
    # One fixed scale for the whole movie, including an expanding trajectory.
    # Use precisely the alive central-slice particles shown by update().
    original=1.15*d['R'];extent=0.;outside=0
    for row in d['index']:
        _,_,mask,v=cohort(row['file'])
        visible=(mask==1)&(np.abs(v[:,3])<.05*d['R'])
        xy=np.abs(v[visible,1:3]);assert np.isfinite(xy).all()
        if xy.size:
            extent=max(extent,float(xy.max()))
            outside=max(outside,int(np.any(xy>original,axis=1).sum()))
    return dict(half_width_M=max(original,1.05*extent),
                maximum_visible_coordinate_M=extent,
                initial_radius_window_half_width_M=original,
                max_particles_outside_initial_radius_window=outside,
                samples=len(d['index']),rule='fixed over all plotted cohort times')


def panel(fig,axes,d):
    if 'spatial_window' not in d:d['spatial_window']=spatial_window(d)
    xy,shell,lap=axes;c=d['case'];R=d['R'];limits=d['spatial_window']['half_width_M']
    xy.set(xlim=(-limits,limits),ylim=(-limits,limits),aspect='equal',xlabel=r'$x/M$',ylabel=r'$y/M$',title=rf'$y_c={c["yc"]:.3f}$')
    ang=np.linspace(0,2*np.pi,257);xy.plot(R*np.cos(ang),R*np.sin(ang),color='.5',ls=':',lw=.7)
    scatter=xy.scatter([],[],s=1.8,c=[],cmap='viridis',norm=Normalize(0,R),alpha=.6,edgecolors='none',rasterized=True)
    ah,=xy.plot([],[],color='.12',lw=1.4)
    text=xy.text(.03,.97,'',transform=xy.transAxes,va='top',fontsize=9)
    ended=xy.text(.5,.5,'',transform=xy.transAxes,ha='center',va='center',fontsize=11)
    for q in range(10,100,10):shell.plot(array(d['shell'],'time'),array(d['shell'],f'r{q}'),color=plt.get_cmap('cividis')((q-10)/80),lw=.8)
    if 'shell_limits' not in d:d['shell_limits']=shell_limits(d['shell'])
    shell.set(yscale='log',ylim=d['shell_limits'],xlim=(0,300),ylabel=r'$r_q/M$')
    lap.plot(array(d['history'],'time'),array(d['history'],'core_lapse'),color='#275d7f',lw=1)
    lap.set(ylim=(0,1),xlim=(0,300),xlabel=r'$t/M$',ylabel=r'$\langle\alpha\rangle_c$')
    lines=[ax.axvline(0,color='#b35c26',lw=1) for ax in [shell,lap]]
    for ax in [shell,lap]:ax.grid(alpha=.15,lw=.5)
    return dict(scatter=scatter,ah=ah,text=text,ended=ended,lines=lines,last=-1,phi=ang,basis=equator_basis(ang))

def update(d,p,t,synchronize=False):
    end=d['summary']['endpoint']
    if t>end+1e-8:
        p['scatter'].set_offsets(np.empty((0,2)));p['ah'].set_data([],[])
        p['ended'].set_text(f'Run ended\nat t/M = {end:.2f}')
        p['text'].set_text('');[line.set_xdata([end,end]) for line in p['lines']];return
    i=max(0,np.searchsorted(d['times'],t+1e-10,side='right')-1)
    if i!=p['last']:
        actual,cy,mask,v=cohort(d['index'][i]['file']);p['last']=i
        # A fixed 20k-ID cohort, shown in a fixed-thickness central slice.
        visible=(mask==1)&(np.abs(v[:,3])<.05*d['R'])
        assert np.all(np.abs(v[visible,1:3])<=d['spatial_window']['half_width_M'])
        p['scatter'].set_offsets(v[visible,1:3]);p['scatter'].set_array(d['radius'][visible])
        x,y=outline(d,actual,p['phi'],p['basis']);p['ah'].set_data(x,y)
        p['text'].set_text(f't/M = {actual:.2f}\n{int(mask.sum()):,} cohort particles alive')
    p['ended'].set_text('');[line.set_xdata([t,t]) for line in p['lines']]

def ffmpeg_exe():
    import shutil
    exe=shutil.which('ffmpeg')
    if not exe:
        import imageio_ffmpeg
        exe=imageio_ffmpeg.get_ffmpeg_exe()
    return exe

def encode(name,fig,states,times,sync=False):
    path=ROOT/'movies'/f'{name}.mp4';exe=ffmpeg_exe();plt.rcParams['animation.ffmpeg_path']=exe
    frames=ROOT/'frames'/f'{name}_{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%S}'
    frames.mkdir(parents=True,exist_ok=False)
    previews={0,len(times)//2,len(times)-1}
    for i,t in enumerate(times):
        for d,_ in states:
            if d['summary']['ah_first'] is not None and abs(t-d['summary']['ah_first'])<.5:previews.add(i)
    # Keep lossless PNG inputs available for direct encoder/decoder comparisons.
    # Retain these frames on AMD; only representative review images are mirrored.
    import shutil
    for i,t in enumerate(times):
        for d,p in states:update(d,p,float(t),sync)
        if sync:fig.suptitle(f'Session 8 • synchronized time t/M = {t:.1f}',fontsize=12)
        frame=frames/f'{i:06d}.png'
        fig.savefig(frame,dpi=120,facecolor='white')
        if i in previews:shutil.copyfile(frame,ROOT/'plots'/f'{name}_review_{i:04d}.png')
    plt.close(fig)
    cmd([exe,'-y','-v','error','-xerror','-framerate','15','-i',frames/'%06d.png',
         '-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',
         '-threads','4',path])
    # Decode every frame; fail on any decoder error, then verify the actual frame count.
    verify=subprocess.run([exe,'-v','error','-xerror','-threads','1','-i',str(path),'-f','null','-'],capture_output=True,text=True)
    assert verify.returncode==0 and not verify.stderr,verify.stderr
    import imageio_ffmpeg
    decoded,duration=imageio_ffmpeg.count_frames_and_secs(str(path))
    assert decoded==len(times) and abs(duration-len(times)/15)<.15,(decoded,duration,len(times))
    # Inspect the encoded artifact itself, not just pre-encoding canvas captures.
    decoded_previews=[]
    for frame in sorted(previews):
        preview=ROOT/'plots'/f'{name}_decoded_{frame:04d}.png'
        cmd([exe,'-v','error','-xerror','-threads','1','-i',path,'-vf',f'select=eq(n\\,{frame})',
             '-frames:v','1','-threads','1',preview])
        assert preview.exists() and preview.stat().st_size>0
        decoded_previews.append(dict(frame=frame,time=float(times[frame]),**receipt(preview)))
    return dict(**receipt(path),frames=decoded,duration=duration,full_decode=True,
                decoded_previews=decoded_previews,frames_directory=str(frames),verification_decoder_threads=1,
                spatial_windows={d['case']['id']:d['spatial_window'] for d,_ in states},
                shell_radius_limits_M={d['case']['id']:d['shell_limits'] for d,_ in states},
                visual_review='pending')

def main():
    job();style();(ROOT/'movies').mkdir(exist_ok=True);(ROOT/'plots').mkdir(exist_ok=True)
    ds=[load_case(c) for c in CASES[::-1]];products=[]
    radius_limits=shell_limits(*(d['shell'] for d in ds))
    for d in ds:d['shell_limits']=radius_limits
    for d in ds:
        fig=plt.figure(figsize=(10,6),layout='constrained');grid=fig.add_gridspec(2,2,width_ratios=[1.1,1])
        axes=[fig.add_subplot(grid[:,0]),fig.add_subplot(grid[0,1]),fig.add_subplot(grid[1,1])]
        p=panel(fig,axes,d)
        fig.suptitle('Fixed particle IDs • color = initial radius • accepted AH at z=0',fontsize=11)
        products.append(encode(d['case']['id'],fig,[(d,p)],d['times']))
    fig=plt.figure(figsize=(16,7.5),layout='constrained');grid=fig.add_gridspec(3,5,height_ratios=[2,1,1]);states=[]
    for j,d in enumerate(ds):
        axes=[fig.add_subplot(grid[k,j]) for k in range(3)];states.append((d,panel(fig,axes,d)))
        if j:
            for ax in axes:ax.set_ylabel('')
    times=np.arange(0,max(d['summary']['endpoint'] for d in ds)+1e-8,.5)
    products.append(encode('s08_comparison',fig,states,times,True))
    atomic(ROOT/'evidence/MOVIES_RENDERED.json',dict(utc=now(),job=job(),products=products))
    (ROOT/'reports/MOVIES.md').write_text('''# Session 8 movies

Five individual movies and one synchronized comparison, 15 fps. Each uses the same 20,000 initial particle IDs (tags divisible by100), with immutable initial-radius colors. Only the slice |z|<0.05 R_iso(0) is shown. The spatial window and slice thickness remain fixed per model. The window encloses the central-slice cohort over all plotted times with a 5% margin and is at least 1.15 times the initial surface radius; it does not zoom during playback. Spatial scales differ between models and axes carry M units. The dotted circle is the initial isotropic surface radius. Black contours are accepted AH equatorial cross-sections reconstructed with the native real spherical-harmonic convention. Rejected searches and stale surfaces are not drawn. Shell curves stop at first removal. All final movie shell panels share a fixed radius range containing every audited shell radius across the five models. Synchronized panels use the latest completed cohort sample and print its actual time; completed runs are explicitly marked and cleared. No frames imply evolution beyond a run endpoint.

All encoded frames were decoded for integrity. Final human-readable visual-review status is recorded separately in the session report.
''')
if __name__=='__main__':main()
