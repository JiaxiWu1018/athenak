#!/usr/bin/env python3
"""Check actual marker visibility, timing and geometry in the zoom release."""
import argparse,csv,datetime as dt,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
from PIL import Image


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--session',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();s=args.session.resolve();out=args.out.resolve()
    assert os.environ.get('SLURM_JOB_ID')
    meta=json.loads((out/'evidence/COMPLETE.json').read_text())
    frames=json.loads((out/'FRAMES.json').read_text())
    assert not (out/'evidence/FINAL_CHECKS.json').exists()
    movie=out/meta['movie'];assert digest(movie)==meta['sha256']
    parent=s/'movies/particles3d_to98_20261019_v1'
    assert digest(out/'TIMING.csv')==digest(parent/'TIMING.csv')
    original=json.loads((parent/'FRAMES.json').read_text())
    schedule=list(csv.DictReader((out/'TIMING.csv').open()));assert len(schedule)==393
    assert float(schedule[0]['actual_saved_time'])==0 and float(schedule[-1]['actual_saved_time'])==98
    # Tiny antialiased sphere facets blend with the underlying scene.
    # A maximum channel of32 is darker than navy (max43); RGB zero is not required.
    checks=[];pairs=0
    for row,old in zip(frames,original):
        assert row['time']==old['time']<=98 and row['horizons']==old['horizons'] and row['missing_surfaces']==old['missing_surfaces']
        assert row['cache_sha256']==old['cache_sha256'] and row['view_limit']==4
        image=np.asarray(Image.open(out/row['frame']).convert('RGB'));assert image.shape==(1080,1920,3)
        assert digest(out/row['frame'])==row['frame_sha256']
        for h in row['projected_horizons']:
            box=h['pixel_bbox'];x=int(round((box[0]+box[2])/2));y=int(round(1080-(box[1]+box[3])/2))
            patch=image[y-1:y+2,x-1:x+2]
            fraction=float(np.mean(np.max(patch,axis=2)<=32));assert fraction>=.8,(row['time'],h['horizon'],fraction)
            checks.append(dict(time=row['time'],horizon=h['horizon'],central_dark_fraction=fraction))
        if len(row['projected_horizons'])==2:
            a,b=[h['pixel_bbox'] for h in row['projected_horizons']]
            assert a[2]<b[0] or b[2]<a[0] or a[3]<b[1] or b[3]<a[1],('Pair overlaps in image',row['time'])
            pairs+=1
        decoded=out/'qa'/f"decoded_t{row['time']:g}.png"
        if decoded.exists():
            image=np.asarray(Image.open(decoded).convert('RGB'))
            for h in row['projected_horizons']:
                a=h['pixel_bbox'];x=int(round((a[0]+a[2])/2));y=int(round(1080-(a[1]+a[3])/2))
                assert np.max(image[y-1:y+2,x-1:x+2].mean(axis=(0,1)))<36,(row['time'],h['horizon'])
    assert len(checks)==56 and pairs==20
    subprocess.run(['git','-C',str(s/'code'),'diff','--exit-code','386870f88f60e25e1a8fac76fe04a071b87a01cf','--','src','CMakeLists.txt','kokkos'],check=True)
    regular=[p for p in out.rglob('*') if p.is_file() and not p.is_symlink()]
    result=dict(passed=True,job=os.environ['SLURM_JOB_ID'],checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                movie_sha256=meta['sha256'],movie_bytes=movie.stat().st_size,checked_frames=len(frames),checked_sphere_markers=len(checks),
                precommon_paired_saved_states=pairs,all_pairs_separated_in_image=True,all_marker_centers_visible_dark=True,
                same_horizon_geometry_and_quality_gaps=True,same_timing_as_original=True,no_later_state=True,final_saved_time=98,
                full_decode='Passed in17392; unchanged MP4 hash checked here',marker_layer='Schematic foreground glyphs, physical centers/radii unchanged',
                marker_checks=checks,native_source_unchanged=True,ah_repair_implemented=False,
                scripts={str(p.relative_to(s)):digest(p) for p in (Path(__file__),Path(__file__).with_name('render_zoom.py'))},
                storage_snapshot=dict(regular_files=len(regular),bytes=sum(p.stat().st_size for p in regular),sequence_links=393,retained=True))
    (out/'evidence/FINAL_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
    seal=[p for p in out.rglob('*') if p.is_file() and not p.is_symlink() and 'matplotlib_cache' not in p.parts and p.suffix!='.log' and p.name!='SHA256SUMS.txt']
    (out/'SHA256SUMS.txt').write_text(''.join(digest(p)+'  '+str(p.relative_to(out))+'\n' for p in sorted(seal)))
    print('ZOOM VISIBILITY/TIMING/GEOMETRY CHECKS PASSED',movie,flush=True)


if __name__=='__main__':main()
