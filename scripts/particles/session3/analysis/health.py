"""Single validity policy for runners, continuation, reductions, plots and movies.

All products must use require_window() and filter time <= last_healthy_time.
Frozen constraints are unavailable, rather than evidence of zero violation.
"""
import csv,json,math,sys
from pathlib import Path

def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith('#')))

def evaluate(run):
    run=Path(run);ledgers=list(run.glob('out/*.plummer_health.csv'))
    if len(ledgers)!=1:
        return dict(valid=False,reason='missing or ambiguous health ledger',last_healthy_time=None)
    rows=read_rows(ledgers[0]);unique={};duplicates={}
    for row in rows:
        try:t=float(row['time'])
        except (ValueError,KeyError):continue
        duplicates.setdefault(t,[]).append(row)
        unique[t]=row
    last=None;reason=None;physical_stop=False
    for t,row in sorted(unique.items()):
        good=True;stopped=False
        # A later duplicate at restart/finalization cannot erase an earlier failure.
        for repeated in duplicates[t]:
            try:
                numbers=[float(repeated[k]) for k in ('N','N_expected','M0','M0_expected',
                         'mass_error','particle_nonfinite','alpha_min','H_core_L2','M_core_L2')]
                finite=math.isfinite(t) and all(math.isfinite(v) for v in numbers)
                good=good and finite and int(repeated['healthy'])==1 and numbers[0]==numbers[1] and numbers[5]==0 and 0<=numbers[4]<=1.e-10
                stopped=stopped or bool(int(repeated['physical_stop'])) or numbers[6]<.2 or int(repeated['constraint_strikes'])>=3
            except (ValueError,KeyError):good=False
        if not good:reason=f'numerical health failure at {t:.17g}';break
        last=t
        if stopped:
            physical_stop=True;reason=f'physical/constraint stop at {t:.17g}';break
    # Fatal field checks can abort before a row is appended. Preserve the last good
    # window while forbidding continuation. Logs are part of the health contract.
    for log in run.glob('segment_*.log'):
        text=log.read_text(errors='replace')
        if 'ISOTROPIC_HEALTH_FAIL' in text:reason='field/particle hard stop in '+log.name
    return dict(valid=last is not None and reason is None,reason=reason,
                physical_stop=physical_stop,last_healthy_time=last,
                ledger=str(ledgers[0]),samples=len(unique))

def require_window(run,allow_stopped=True):
    result=evaluate(run)
    if result['last_healthy_time'] is None:raise RuntimeError(result['reason'])
    if not allow_stopped and not result['valid']:raise RuntimeError(result['reason'])
    return result

if __name__=='__main__':
    result=evaluate(sys.argv[1]);print(json.dumps(result,indent=2))
    sys.exit(0 if result['valid'] else 2)
