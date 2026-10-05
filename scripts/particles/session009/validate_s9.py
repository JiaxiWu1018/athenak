#!/usr/bin/env python3
"""Full particle ledger and matched Session 007 initial constraint comparison."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import validate_initial_s7 as v

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 sys.path.insert(0,str(a.root/'athenak/vis/python'));import bin_convert
 run=a.root/'runs/gate_reference';logged=v.parse_ledger(run/'run.log')
 written,ledger=v.written_particle_stats(run)
 agreement,detail=v.ledger_agreement(logged,ledger)
 cl=written['components'][1:]
 checks=dict(exact_counts=[x['count'] for x in logged]==[3000000,1000000,1000000],
 tags=written['exact_unique_tag_range'] and written['integer_tags'],finite=written['finite'],
 positive_weights=written['positive_rest_weights'],inside_domain=written['inside_domain'],ledger=agreement,
 signs=logged[1]['P_hat'][1]<0<logged[2]['P_hat'][1],Jz=all(x['J_origin_cov'][2]>0 for x in logged[1:]),
 centers=all(np.max(np.abs(np.asarray(x['position_mean'])-v.CENTERS[x['component']]))<.03 for x in cl),
 widths=all(all(.60<z<.80 for z in x['position_std']) for x in cl),
 thermal=all(all(.019<z<.021 for z in x['rest_frame_thermal_std']) for x in cl))
 con=bin_convert.read_binary(str(v.latest(run,'*.con3d.*.bin')))
 variables=('con_H','con_M','con_Mx','con_My','con_Mz')
 windows=dict(global_region=None,left=(-4.5,-1.5,-1.5,1.5,-1.5,1.5),right=(1.5,4.5,-1.5,1.5,-1.5,1.5),common_central=(-6.,6.,-6.,6.,-6.,6.))
 norms={k:{x:v.volume_stats(con,x,w) for x in variables} for k,w in windows.items()}
 baseline=json.loads((a.root/'evidence/s7_initial_validation.json').read_text())['cases']['boosted']
 ratios={};region_matches={};matches=abs(float(con['time'])-baseline['constraint_time'])<1e-12
 for name,vals in norms.items():
  old=baseline['global' if name=='global_region' else name]
  matched=abs(float(con['time'])-baseline['constraint_time'])<1e-12 and all(abs(vals[x]['volume']-old[x]['volume'])<1e-9*old[x]['volume'] for x in variables)
  region_matches[name]=bool(matched)
  matches=matches and matched
  ratios[name]={x:(vals[x]['rms']/old[x]['rms'] if old[x]['rms'] else None) for x in variables}
 # A mismatch is recorded, never presented as a matched ratio.
 physical_momentum={k:float(np.sqrt(sum(vals[x]['rms']**2 for x in ('con_Mx','con_My','con_Mz')))) for k,vals in norms.items()}
 old_physical={k:float(np.sqrt(sum(baseline['global' if k=='global_region' else k][x]['rms']**2 for x in ('con_Mx','con_My','con_Mz')))) for k in norms}
 result=dict(checks=checks,written=written,logged=logged,ledger_comparison=detail,
 total_P_cov=np.sum([x['P_cov'] for x in logged],axis=0).tolist(),
 total_P_hat=np.sum([x['P_hat'] for x in logged],axis=0).tolist(),
 cancelling_linear_momentum_residual_fraction=float(np.linalg.norm(np.sum([x['P_cov'] for x in logged],axis=0))/sum(np.linalg.norm(x['P_cov']) for x in logged[1:])),
 constraint_time=float(con['time']),constraint_norms=norms,
 constraint_definition='coordinate-volume weighted leaf cells; identical rectangular regions; no field mask. Raw con_M is already squared; its field RMS is not the momentum-vector RMS.',
 physical_momentum_vector_rms=physical_momentum,
 physical_momentum_vector_rms_ratio={k:physical_momentum[k]/old_physical[k] for k in norms} if matches else None,
 session7_comparison_matched=bool(matches),session9_over_session7_rms={k:ratios[k] if region_matches[k] else None for k in ratios},region_matches=region_matches)
 (a.root/'evidence/initial_validation.json').write_text(json.dumps(result,indent=2)+'\n')
 if not all(checks.values()): raise RuntimeError('initialization gate failed: '+str(checks))
 for kind in ('z4c_xy','tmunu_xy','con_xy','z4c_xz','tmunu_xz','con_xz','z4c_yz','tmunu_yz','con_yz','weyl_xy','weyl_xz','weyl_yz','z4c3d','tmunu3d','weyl3d'):
  data=bin_convert.read_binary(str(v.latest(run,'*.'+kind+'.*.bin')))
  if any(not np.isfinite(z).all() for z in data['mb_data'].values()): raise RuntimeError('nonfinite evolved/output '+kind)
 # Full complete startup mesh physical spacing, not copied logical level labels.
 from mesh_audit import audit
 audit(a.root,con)
 print(json.dumps(checks))

if __name__=='__main__':main()
