#!/usr/bin/env python3
"""Seal and independently check the accepted movie and its scientific sidecars."""
import argparse,csv,datetime as dt,hashlib,json,os,re,subprocess
from pathlib import Path
import numpy as np
from PIL import Image


def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--session',type=Path,required=True);args=ap.parse_args();s=args.session.resolve()
 assert os.environ.get('SLURM_JOB_ID')
 out=s/'movies/particles3d_to98_20261019_v1';ev=out/'evidence';assert not (ev/'FINAL_CHECKS.json').exists()
 accepted=json.loads((ev/'ACCEPTED.json').read_text());complete=json.loads((ev/'COMPLETE.json').read_text());frames=json.loads((out/'FRAMES.json').read_text())
 movie=out/accepted['movie'];assert accepted['status']=='accepted'
 assert sha(movie)==accepted['sha256']==complete['sha256']==(out/(movie.name+'.sha256')).read_text().split()[0]
 assert movie.stat().st_size==accepted['bytes']==5005907
 rows=list(csv.DictReader((out/'TIMING.csv').open()));assert len(rows)==393
 by_index={r['index']:r for r in frames};assert len(by_index)==50
 for k,row in enumerate(rows):
  source=by_index[int(row['source_index'])]
  assert int(row['frame'])==k and abs(float(row['playback_seconds'])-k/12)<1e-12
  assert float(row['actual_saved_time'])==source['time']<=float(row['hold_clock_Mref'])==k/4<=98
  assert (int(row['cycle']),int(row['case']))==(source['cycle'],source['case'])
 assert float(rows[0]['actual_saved_time'])==0 and float(rows[-1]['actual_saved_time'])==98
 assert sum(float(row['actual_saved_time'])==98 for row in rows)==1
 data=s/'session010_summary_20261019/workspace/runtime_s10/analysis/through_segment002_20261018'
 horizons=[np.genfromtxt(data/f'accepted_horizon_{i}.csv',delimiter=',',names=True) for i in range(3)]
 checked_surfaces=0
 for r in frames:
  assert 0<=r['time']<=98 and int(r['encoded_frames'])==sum(int(x['source_index'])==r['index'] for x in rows)
  for h in r['horizons']:
   hid=h['horizon'];assert (hid<2)==(r['time']<complete['common_enclosure_time'])
   a=horizons[hid];q=a[(a['cycle']==r['cycle'])&(a['case']==r['case'])];assert len(q)==1
   assert float(q[0]['meanradius'])==h['mean_coordinate_radius']
   assert [float(q[0][key]) for key in ('center_x','center_y','center_z')]==h['center']
   assert h['time']<=98+1e-8
   checked_surfaces+=1
  with Image.open(out/r['frame']) as img:assert img.size==(1920,1080);img.verify()
  assert sha(out/r['frame'])==r['frame_sha256']
 summary=s/'session010_summary_20261019'
 sources=json.loads((out/'SOURCE_MANIFEST.json').read_text())
 for record in sources:
  p=summary/record['path'];assert p.stat().st_size==record['bytes'] and sha(p)==record['sha256']
 for record in complete['references']:
  assert sha(out/record['path'])==record['sha256']
 figure=s/'plots/orbit_duration_20261019_v1/orbit_duration_assessment'
 p=subprocess.check_output(['pdfinfo',str(figure.with_suffix('.pdf'))],text=True);assert re.search(r'^Pages:\s+1\s*$',p,re.M)
 with Image.open(figure.with_suffix('.png')) as img:img.verify()
 estimate=json.loads((figure.parent/'ESTIMATES.json').read_text())
 assert abs(estimate['vacuum_baselines'][0]['vacuum_turns_to_6Mbin']-60.41558184249585)<1e-9
 subprocess.run(['git','-C',str(s/'code'),'diff','--exit-code','386870f88f60e25e1a8fac76fe04a071b87a01cf','--','src','CMakeLists.txt','kokkos'],check=True)
 scripts=list((s/'code/scripts/particles/session010/movie_to98_20261019').glob('*.py'))
 scripts += [s/'scripts/perseus_movie_to98_20261019_v1.sbatch',s/'scripts/perseus_orbit_estimate_20261019.sbatch',s/'scripts/perseus_movie_to98_checks_20261019.sbatch']
 regular=[p for p in out.rglob('*') if p.is_file() and not p.is_symlink()]
 result=dict(passed=True,job=os.environ['SLURM_JOB_ID'],checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
 movie_sha256=sha(movie),checked_source_files=len(sources),checked_native_matched_surfaces=checked_surfaces,
 checked_saved_frames=50,checked_encoded_time_rows=393,no_source_state_after98=True,final_t98_included=True,
 horizons_exact_case_cycle_meanradius_centers=True,full_decode='Passed in17088; unchanged MP4 hash verified here',
 visual_acceptance='evidence/ACCEPTED.json',native_source_unchanged=True,ah_repair_implemented=False,simulation_restarted=False,
 scripts={str(p.relative_to(s)):sha(p) for p in scripts},
 storage_snapshot=dict(path=str(out),regular_files=len(regular),bytes=sum(p.stat().st_size for p in regular),
 sequence_links=sum(p.is_symlink() for p in out.rglob('*')),retention='All retained on Perseus; no native raw generated or deletion'),
 scientific_estimates=dict(job=estimate['job'],json_sha256=sha(figure.parent/'ESTIMATES.json'),png_sha256=sha(figure.with_suffix('.png')),pdf_sha256=sha(figure.with_suffix('.pdf'))))
 (ev/'FINAL_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
 # Seal products and metadata, excluding live logs, font caches and this checksum list.
 files=[p for p in out.rglob('*') if p.is_file() and not p.is_symlink()
        and 'matplotlib_cache' not in p.parts and p.suffix!='.log' and p.name!='SHA256SUMS.txt']
 (out/'SHA256SUMS.txt').write_text(''.join(sha(p)+'  '+str(p.relative_to(out))+'\n' for p in sorted(files)))
 print('ACCEPTED MOVIE FINAL CHECKS PASSED',movie,flush=True)
 print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
