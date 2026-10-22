"""Require one dependency lineage and audited preceding scientific endpoints."""
import math,re,sys
from common import *

def dependency_lineage(rows,dependency):
    if not rows:return []
    assert dependency,'Active ST work requires an explicit dependency'
    active={row[0]:row for row in rows};assert len(active)==len(rows)
    lineage=[];current=str(dependency)
    while current in active:
        assert current not in lineage,'Dependency cycle'
        lineage.append(current);link=active[current][4]
        if link in ['(null)','None','']:break
        match=re.fullmatch(r'afterok:(\d+)(?:\([^)]*\))?',link)
        assert match,('Not a single afterok lineage',link)
        current=match[1]
    assert set(active)<=set(lineage),('Parallel/unrelated ST allocation',active,lineage)
    return lineage

def predecessors(case,root=ROOT):
    index=next(i for i,c in enumerate(CASES) if c['id']==case);checked=[]
    for c in CASES[:index]:
        work=root/'runs'/c['id'];reduced=root/'reduced'/c['id']
        state=load(work/'state.json');summary=load(reduced/'summary.json')
        audit=load(reduced/'REDUCTION_OK.json')
        assert state.get('complete') and summary.get('complete'),('Preceding endpoint incomplete',c['id'])
        latest=state['latest'];end=latest['time'];onset=latest['ah_onset']
        target=min(300,onset+20) if c['unstable'] and onset>=0 else 300
        assert math.isfinite(end) and target-1e-8<=end<=300+1e-8,(c['id'],end,target)
        assert abs(summary['endpoint']-end)<1e-8,('Stale reduction',c['id'])
        assert summary['max_closure']<1e-10
        assert all(audit.get(key) for key in ['population_ids','removal_ids','mass_ledger','cohort_vs_full_snapshots'])
        checked.append(dict(case=c['id'],endpoint=end,audit_job=audit['job']))
    return checked

if __name__=='__main__':
    jid=job();case=sys.argv[1];checked=predecessors(case)
    atomic(ROOT/'evidence/jobs'/jid/'PRECEDING_ENDPOINTS_OK.json',dict(utc=now(),job=jid,case=case,predecessors=checked))
