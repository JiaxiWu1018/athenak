"""Stream the actual removal positions against sampled converged AH bounds."""
import bisect
import csv
import math
import re

from common import *


def audit(source,ah,maximum_step,expected_count):
    job()
    work=source/'runs/yc0500';path=work/'yc0500.prtcl_destroy.csv'
    qualified=[r for r in ah if r['converged']==1 and r['missing']==0
               and math.isfinite(r['hrel']) and r['hrel']<=.05
               and r['rmin']>=4*r['dx'] and r['rmin']>0]
    assert qualified and path.exists()
    times=[r['time'] for r in qualified]
    text=(source/'inputs/yc0500.athinput').read_text()
    center=[float(re.search(r'(?m)^center_'+axis+r'_0\s*=\s*([-+0-9.eE]+)',text)[1]) for axis in 'xyz']
    counts={};early=post=outside=stale=0
    maximum_radius=maximum_ratio=early_maximum_ratio=maximum_early_lead=0.
    with path.open() as f:
        names=next(f).lstrip('# ').strip().split(',')
        assert names[:7]==['cycle','time','tag','reason','x1','x2','x3']
        for row in csv.DictReader(f,fieldnames=names):
            reason=row['reason'];counts[reason]=counts.get(reason,0)+1
            if reason=='exit':continue
            t=float(row['time']);radius=math.sqrt(sum((float(row['x'+str(k+1)])-center[k])**2 for k in range(3)))
            assert math.isfinite(t) and math.isfinite(radius)
            maximum_radius=max(maximum_radius,radius)
            index=bisect.bisect_right(times,t+1e-8)-1
            if index<0:
                early+=1;maximum_early_lead=max(maximum_early_lead,times[0]-t)
                early_maximum_ratio=max(early_maximum_ratio,radius/qualified[0]['rmin'])
            elif t-times[index]>.5+maximum_step+1e-6:
                stale+=1
            else:
                post+=1;ratio=radius/qualified[index]['rmin'];maximum_ratio=max(maximum_ratio,ratio)
                outside+=int(ratio>1+1e-10)
    assert sum(counts.values())==expected_count,(counts,expected_count)
    return dict(source=receipt(path),input=receipt(source/'inputs/yc0500.athinput'),center=center,
                first_qualified_converged_M=times[0],reason_counts=counts,
                events_before_first_converged=early,maximum_early_lead_M=maximum_early_lead,
                early_max_radius_over_first_following_rmin=early_maximum_ratio,
                events_with_recent_preceding_surface=post,events_outside_preceding_inscribed_radius=outside,
                events_without_recent_preceding_surface=stale,maximum_removal_radius_M=maximum_radius,
                maximum_radius_over_preceding_rmin=maximum_ratio,
                scope='Coordinate sanity check against the minimum radius of recent residual-qualified, on-grid, resolved converged surfaces. Persistence acceptance is separate. Early events use the first following surface only and are not assigned a contemporaneous horizon. No angular interpolation or exact worldtube containment claim.')
