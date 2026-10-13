#!/usr/bin/env python3
"""Native mesh-only counts for approved geometry and reference alternatives."""
import json,os,re,subprocess
from pathlib import Path
from checkpoint_io import atomic,digest

def shift_deck(text,half,shift):
 parts=re.split(r'(?m)^(<[^>]+>)\s*$',text);out=[parts[0]];region=0
 for title,body in zip(parts[1::2],parts[2::2]):
  if title=='<mesh>':
   body=re.sub(r'(?m)^(x[123]min\s*=).*$',lambda m:m[1]+' '+str(-half),body)
   body=re.sub(r'(?m)^(x[123]max\s*=).*$',lambda m:m[1]+' '+str(half),body)
  if title=='<mesh_refinement>':body=re.sub(r'(?m)^(num_levels\s*=)\s*(\d+)',lambda m:m[1]+' '+str(int(m[2])+shift),body)
  if title.startswith('<refined_region'):
   level=int(re.search(r'(?m)^level\s*=\s*(\d+)',body)[1])+shift
   if level<=0:continue
   region+=1;title='<refined_region'+str(region)+'>';body=re.sub(r'(?m)^level\s*=\s*\d+','level = '+str(level),body)
  if title=='<z4c_amr>':body=re.sub(r'(?m)^(radius_\d+_reflevel\s*=)\s*(\d+)',lambda m:m[1]+' '+str(max(0,int(m[2])+shift)),body)
  if title=='<z4c>':body=re.sub(r'(?m)^(co_[01]_reflevel\s*=)\s*(\d+)',lambda m:m[1]+' '+str(int(m[2])+shift),body)
  out.extend([title,body])
 return '\n'.join(out)

def main():
 assert os.environ.get('SLURM_JOB_ID')
 s=Path(__file__).resolve().parents[4];campaign=s.parent
 r=s/'evidence'/('mesh_options_'+os.environ['SLURM_JOB_ID']);r.mkdir()
 exe=Path(json.loads((s/'evidence/build_receipt.json').read_text())['patched'])
 prod=(s/'inputs/gi_cluster_s10.athinput').read_text()
 options=[('s9_1024',5384,(campaign/'session_009_orbit_gw_production_20261005/inputs/gi_cluster_s9.athinput').read_text()),('s10_512',4936,prod),('finewave_256',4488,shift_deck(prod,256,-1)),('finewave_128',4040,shift_deck(prod,128,-2)),('actual_s7_256',1352,(campaign/'session_007_part1_vista_handoff_20260908/vista_handoff/inputs/gi_cluster_s7_two_clump_ccw_production.athinput').read_text())]
 rows=[]
 for name,expected,text in options:
  case=r/name;case.mkdir();deck=case/'mesh_only.athinput';deck.write_text(text)
  cmd=[str(exe),'-i',str(deck),'-m','mesh_refinement/max_nmb_per_rank=6000']
  with (case/'mesh.log').open('w') as log:subprocess.run(cmd,cwd=case,stdout=log,stderr=subprocess.STDOUT,check=True)
  result=(case/'mesh.log').read_text();count=int(re.search(r'Total number of MeshBlocks =\s*(\d+)',result)[1]);assert count==expected,(name,count,expected)
  rows.append(dict(option=name,blocks=count,interior_cells=count*32**3,input_sha256=digest(deck),log_sha256=digest(case/'mesh.log')))
 atomic(r/'RESULT.json',dict(passed=True,job=int(os.environ['SLURM_JOB_ID']),executable_sha256=digest(exe),mesh_only=True,rows=rows))
 atomic(s/'evidence/mesh_options_receipt.json',dict(passed=True,root=str(r)));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
