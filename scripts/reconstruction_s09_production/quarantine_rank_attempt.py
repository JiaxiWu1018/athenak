"""Preserve the unadmitted478544 attempt and restore its last admitted prefix."""
import fcntl
import time
from common import *
from checkpoint import validate
from recover import output_time


def main():
    job()
    assert ROOT == EXT/'yc0283_u0005'
    marker = ROOT/'evidence/RANK_REPLAY_READY.json'
    if marker.exists():
        raise RuntimeError('Replay preparation is a single, inventoried operation')
    status = subprocess.check_output(['sacct','-X','-n','-P','-j','478544',
                                     '-o','JobIDRaw,State,ExitCode'],text=True)
    assert any('478544|'+s in status for s in ['COMPLETED','FAILED','CANCELLED','TIMEOUT']),status
    mismatch = load(ROOT/'evidence/jobs/478544/RANK_ENV_MISMATCH.json')
    prior = mismatch['state_observed']
    assert prior['latest']['sha256'] == mismatch['previous_verified_checkpoint_sha256']
    assert all(s['job'] != '478544' for s in prior['segments'])
    work = ROOT/'runs/yc0283'
    with (EXT/'evidence/locks/yc0283_u0005.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        checkpoint = validate(prior['latest']['path']);checkpoint.pop('parameters')
        assert checkpoint['sha256'] == prior['latest']['sha256']
        cut, cycle = checkpoint['time'],checkpoint['cycle']
        archive = ROOT/'quarantined'/f'rank_mismatch_478544_{job()}'
        archive.mkdir(parents=True,exist_ok=False)
        selected = []
        for name in ['pvtk','bin','cohort']:
            selected += [p for p in (work/name).glob('*')
                         if p.is_file() and output_time(p)>cut+1e-8]
        selected += [p for p in (work/'rst').glob('*.rst')
                     if validate(p)['time']>cut+1e-8]
        selected += list(work.glob('478544*'))
        selected += [p for p in [work/'REQUEST_STOP',ROOT/'evidence/RESOURCE_STOP.json'] if p.exists()]
        texts = [p for p in work.iterdir() if p.is_file() and p.suffix in ['.hst','.csv']]
        metadata = [work/'state.json',work/'checkpoints.json']
        originals = sorted(set(selected+texts+metadata))
        files = [receipt(p) for p in originals]
        atomic(archive/'BEFORE.json',dict(utc=now(),job=job(),reason=mismatch,
               original_state=load(work/'state.json'),retained_checkpoint=checkpoint,
               files=files,policy='Preserve every original byte; replay all evolution after the previously admitted checkpoint. No native physics change.'))
        for p in originals:
            dest = archive/p.relative_to(ROOT)
            dest.parent.mkdir(parents=True,exist_ok=True)
            p.rename(dest)
        derivatives=[]
        for p in texts:
            original=archive/p.relative_to(ROOT);keep=[]
            for line in original.read_text().splitlines(keepends=True):
                if not line.strip() or line.startswith(('#','time,','cycle,')):
                    keep.append(line);continue
                values=line.strip().split(',') if ',' in line else line.split()
                good=(int(values[0])<cycle and float(values[1])<=cut+1e-8) if p.name.endswith('prtcl_destroy.csv') else float(values[0])<=cut+1e-8
                if good:keep.append(line)
            p.write_text(''.join(keep));derivatives.append(receipt(p))
        old_index=load(archive/(work/'checkpoints.json').relative_to(ROOT))
        atomic(work/'checkpoints.json',[r for r in old_index if r['time']<=cut+1e-8])
        atomic(work/'state.json',prior)
        for row in files:
            original=archive/Path(row['path']).relative_to(ROOT)
            assert sha(original)==row['sha256'],'Preserved bytes differ'
        assert sha(checkpoint['path'])==checkpoint['sha256']
        atomic(marker,dict(utc=now(),job=job(),archive=str(archive),
               retained_checkpoint=checkpoint,original_files=len(files),
               original_bytes_checksum_verified=True,text_derivatives=derivatives,
               original_state_restored=True,actual_eight_rank_zero_step_still_required=True,
               reason='Unadmitted amended-allocation attempt excluded from production. Corrected launcher derives ranks from scheduler metadata; no source/input/sample changes.'))


if __name__=='__main__':main()
