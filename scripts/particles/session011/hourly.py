"""Hourly source-bounded scientific products; never signal an evolving application."""
import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from runtime import atomic,active_job_state,last_csv,require_compute

TOOLING=Path(__file__).resolve().parent


def bounded_lines(source):
    """Read complete lines from the size observed on entry; ignore a partial tail."""
    with source.open('rb') as f:
        limit=os.fstat(f.fileno()).st_size
        while f.tell()<limit:
            line=f.readline(limit-f.tell())
            if not line.endswith(b'\n'):break
            yield line.decode()


def csv_snapshot(source,destination,cutoff):
    rows=csv.reader(bounded_lines(source));header=next(rows)
    names=[n.lstrip('# ').strip() for n in header]
    time_key=next((n for n in ('state_time','time') if n in names),None)
    cycle_key=next((n for n in ('state_cycle','cycle') if n in names),None)
    destination.parent.mkdir(parents=True,exist_ok=True);count=0
    with destination.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(header)
        for row in rows:
            if len(row)!=len(header):raise RuntimeError(f'incomplete CSV row in {source}')
            if time_key and float(row[names.index(time_key)])>cutoff['time']+1.e-10:continue
            if cycle_key and int(row[names.index(cycle_key)])>cutoff['cycle']:continue
            writer.writerow(row);count+=1
    return count


def waves_snapshot(run,target,cutoff):
    times=run/'waveforms/state_times.csv'
    if not times.exists():return
    destination=target/'waveforms';destination.mkdir(exist_ok=True)
    csv_snapshot(times,destination/'state_times.csv',cutoff)
    with (destination/'state_times.csv').open() as f:
        reader=csv.reader(f);header=next(reader);rows=list(reader)
    chunks={}
    for radius in (36,40,44):
        for part in ('real','imag'):
            name=f'rpsi4_{part}_{radius:04d}.txt'
            path=run/'waveforms'/name
            if not path.exists():raise RuntimeError(f'missing wave source {path}')
            chunks[name]=[line for line in bounded_lines(path) if line.strip() and not line.lstrip().startswith('#')]
    count=min([len(rows)]+[len(lines) for lines in chunks.values()])
    with (destination/'state_times.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(header);writer.writerows(rows[:count])
    for name,lines in chunks.items():(destination/name).write_text(''.join(lines[:count]))


def make_view(root,destination):
    require_compute()
    (destination/'runs').mkdir(parents=True);(destination/'control').mkdir();(destination/'evidence').mkdir()
    checkpoint=json.loads((root/'control/latest_checkpoint.json').read_text())
    runs=sorted((root/'runs').glob('segment_*'))
    states=[r for p in runs if (r:=last_csv(p/'s11_amd.gi_status.csv'))]
    if not states:raise RuntimeError('no complete diagnostic state')
    last=max(states,key=lambda r:int(r['cycle']))
    cutoff=dict(time=float(last['time']),cycle=int(last['cycle']))
    if checkpoint['time']>cutoff['time']+1.e-10:raise RuntimeError('native checkpoint ahead of diagnostic snapshot; retry')
    provenance=[];removed={k:0 for k in ('exit','sphere','lapse','horizon')}
    for run in runs:
        state=last_csv(run/'s11_amd.gi_status.csv')
        if not state or float(state['time'])<0:continue
        target=destination/'runs'/run.name;target.mkdir()
        for source in sorted(run.glob('*.csv')):
            csv_snapshot(source,target/source.name,cutoff)
            provenance.append(dict(source=str(source),snapshot=str(target/source.name),
                 snapshot_sha256=hashlib.sha256((target/source.name).read_bytes()).hexdigest(),
                 source_size_observed=source.stat().st_size))
        for name in ('START_EPOCH','provenance.txt','population_audit.json','checkpoint_manifest.json','END_EPOCH','EXIT_CODE'):
            source=run/name
            if source.exists():
                if name in ('population_audit.json','checkpoint_manifest.json'):
                    value=json.loads(source.read_text())
                    times=[v['time'] for v in value] if isinstance(value,list) else [value['time']]
                    if max(times)>cutoff['time']+1.e-10:continue
                (target/name).write_bytes(source.read_bytes())
        atomic(target/'DIAGNOSTIC_SNAPSHOT.json',dict(cutoff,kind='read-only diagnostic snapshot'))
        for source in run.glob('telemetry_*.jsonl'):
            (target/source.name).write_text(''.join(bounded_lines(source)))
        source=run/'storage.jsonl'
        if source.exists():(target/source.name).write_text(''.join(bounded_lines(source)))
        waves_snapshot(run,target,cutoff)
        (target/'trk').mkdir()
        for source in sorted((run/'trk').glob('*.trk2')):
            # Published diagnostic cycle after a cohort write implies MPI output completed.
            with source.open('rb') as f:header=f.read(512).rstrip(b'\0').decode()
            import re
            cycle=int(re.search(r'cycle=(\d+)',header)[1]);at=float(re.search(r'time=([^\s]+)',header)[1])
            count=int(re.search(r'count=(\d+)',header)[1])
            if cycle>cutoff['cycle'] or at>cutoff['time']+1.e-10:continue
            if source.stat().st_size!=512+72*count:
                if cycle==cutoff['cycle']:continue  # newest write may still be in flight
                raise RuntimeError(f'incomplete cohort {source}')
            (target/'trk'/source.name).symlink_to(source.resolve())
        death=target/'s11_amd.prtcl_destroy.csv'
        if death.exists():
            with death.open() as f:
                for row in csv.DictReader(f):
                    if row['reason'] not in removed:raise RuntimeError('unknown removal reason')
                    removed[row['reason']]+=1
    newest=destination/'runs'/max(p.name for p in runs if (destination/'runs'/p.name).exists())
    particles=newest/'s11_amd.gi_particles.csv';alive=0.;components=set()
    with particles.open() as f:
        for row in csv.DictReader(f):
            if int(row['cycle'])==cutoff['cycle'] and row['mask']=='0':
                alive+=float(row['count']);components.add(int(row['component']))
    if components!={0,1,2} or alive!=int(alive) or int(alive)+sum(removed.values())!=5_000_000:
        raise RuntimeError('live full-population diagnostic/removal census fails 5M closure')
    sample=dict(cutoff,particles=int(alive),removed=list(removed.values()))
    atomic(destination/'control/latest_checkpoint.json',checkpoint)
    context=dict(source_root=str(root),created_epoch=time.time(),sample=sample,
        audited_checkpoint=checkpoint,qualification='Live diagnostic census; not a native saved-state partition audit')
    atomic(destination/'control/HOURLY_CONTEXT.json',context)
    atomic(destination/'source_windows.json',provenance)
    return context


def health(root):
    sub=json.loads((root/'control/submission.json').read_text());run=root/'runs'/f'segment_{sub["segment"]:04d}'
    queue=active_job_state(sub['job_id'])
    accounting=subprocess.run(['sacct','-X','-n','-P','-j',sub['job_id'],'-o','JobID,State,ElapsedRaw,AllocNodes,ExitCode'],text=True,capture_output=True)
    issues=[]
    for name in ('PAUSED.json',):
        p=root/'control'/name
        if p.exists():issues.append(dict(source=str(p),details=json.loads(p.read_text())))
    for name in ('STOP_REASON.json','CHECKPOINT_AUDIT_REVIEW.json'):
        p=run/name
        if p.exists():issues.append(dict(source=str(p),details=json.loads(p.read_text())))
    if any(s in accounting.stdout for s in ('FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL')):
        issues.append(dict(source='sacct',details=accounting.stdout.strip()))
    telemetry=[]
    for p in run.glob('telemetry_*.json'):
        row=json.loads(p.read_text());fraction=max(d['used']/d['total'] for d in row['devices'])
        telemetry.append(dict(host=row['host'],age_seconds=time.time()-row['epoch'],max_gpu_fraction=fraction))
        if queue=='RUNNING' and (time.time()-row['epoch']>90 or fraction>=.80):issues.append(dict(source=str(p),details=telemetry[-1]))
    return dict(epoch=time.time(),submission=sub,live_scheduler_state=queue,sacct=accounting.stdout,
                telemetry=telemetry,issues=issues,requires_investigation=bool(issues))


def schedule(root,immediate=False):
    control=root/'hourly_control';control.mkdir(exist_ok=True)
    with (control/'schedule.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        old_path=control/'submission.json'
        if old_path.exists():
            old=json.loads(old_path.read_text())
            if str(old['job_id'])!=os.environ.get('SLURM_JOB_ID') and active_job_state(old['job_id']):return old
        if (root/'control/PRODUCTS_COMPLETE.json').exists():return None
        now=time.time();due=now if immediate else (math.floor(now/3600)+1)*3600
        command=['sbatch','--parsable',f'--export=ALL,SESSION_ROOT={root},HOURLY_TOOLING={TOOLING}',
                 '--output='+str(root/'evidence/hourly_%j.slurm.log')]
        if not immediate:command+=['--begin=now+'+str(max(1,math.ceil(due-now)))+'seconds']
        command+=[str(TOOLING/'amd_hourly.sbatch')]
        intent=control/'submission_intent.json'
        if intent.exists() and json.loads(intent.read_text()).get('status')=='prepared':raise RuntimeError('ambiguous hourly scheduler receipt; review before resubmitting')
        atomic(intent,dict(status='prepared',command=command,due_epoch=due))
        job=subprocess.check_output(command,text=True).strip().splitlines()[-1].split(';')[0]
        if not job.isdigit():raise RuntimeError('unrecognized hourly scheduler receipt')
        result=dict(status='submitted',job_id=job,due_epoch=due,command=command,cadence='hourly on the hour; scheduler start subject to resources')
        atomic(old_path,result);atomic(intent,result);return result


def refresh(root):
    require_compute()
    control=root/'hourly_control';control.mkdir(exist_ok=True)
    # Schedule independently before analysis, so a failed report cannot stop the cadence.
    schedule(root)
    with (control/'refresh.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another hourly refresh is still active; no duplicate renderer started.');return
        stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        out=root/'products'/f'hourly_{stamp}_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True)
        atomic(out/'health.json',health(root))
        try:
            context=make_view(root,out/'source_view');products=out/'review'
            command=[sys.executable,str(TOOLING/'products.py'),'--root',str(out/'source_view'),'--out',str(products),
                     '--partial','--movies','--frame-cache',str(root/'hourly_frame_cache')]
            atomic(out/'command.json',command)
            with (out/'pipeline.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            import products as tools
            metrics=json.loads((products/'metrics.json').read_text())
            assert metrics['state_kind']=='sampled live diagnostics' and metrics['cycle']==context['sample']['cycle']
            assert metrics['particles_alive']+sum(metrics['removed'])==5_000_000
            ffmpeg=__import__('shutil').which('ffmpeg')
            if not ffmpeg:
                import imageio_ffmpeg
                ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
            for movie in ('wide_1080p.mp4','close_1080p.mp4'):
                subprocess.run([ffmpeg,'-nostdin','-v','error','-i',str(products/movie),'-f','null','-'],check=True)
            qualification=('# Hourly partial analysis\n\nLatest diagnostic state t='+str(metrics['endpoint'])+
                '; latest audited native checkpoint t='+str(context['audited_checkpoint']['time'])+
                '. Production continued during analysis. Saved-state angular validation and the t75 final report remain pending.\n')
            (products/'QUALIFICATION.md').write_text(qualification)
            (products/'health.json').write_bytes((out/'health.json').read_bytes())
            names=[p for p in products.iterdir() if p.is_file() and p.name!='manifest.json']
            manifest=[dict(path=p.name,bytes=p.stat().st_size,sha256=tools.sha256(p)) for p in sorted(names)]
            atomic(products/'manifest.json',manifest)
            receipt=dict(status='pass',job_id=os.environ['SLURM_JOB_ID'],created_epoch=time.time(),
                path=str(products),manifest=str(products/'manifest.json'),diagnostic_time=metrics['endpoint'],
                audited_checkpoint_time=context['audited_checkpoint']['time'],movies_full_decode='pass',
                production_interrupted=False,health=json.loads((out/'health.json').read_text()),
                source=str(TOOLING),partial=True)
            atomic(out/'RESULT.json',receipt);atomic(root/'control/HOURLY_LATEST.json',receipt)
            print(json.dumps(receipt,indent=2))
        except Exception as exc:
            error=dict(status='failed',job_id=os.environ['SLURM_JOB_ID'],reason=f'{type(exc).__name__}: {exc}',
                       path=str(out),production_interrupted=False,epoch=time.time())
            atomic(out/'RESULT.json',error);atomic(control/'REVIEW_REQUIRED.json',error)
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--schedule-now',action='store_true');a=p.parse_args()
    if a.schedule_now:print(json.dumps(schedule(a.root,immediate=True)))
    else:refresh(a.root)
