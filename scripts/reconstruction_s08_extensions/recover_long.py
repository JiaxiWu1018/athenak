"""Preserve failed475604 in full; recover only its verified checkpoint prefix.

All inventory, hashing, copying and checkpoint validation run in Slurm. No
interrupted data are deleted. The scientific tail beyond the checkpoint is kept
in its original archive and is excluded from the resumed canonical trajectory.
"""
import csv
import re
import shutil

from common import *
from checkpoint import validate


def output_time(path):
    with path.open('rb') as f:
        if path.suffix == '.vtk':
            f.readline();line=f.readline().decode('ascii')
            return float(re.search(r'time=\s*([-+0-9.eE]+)',line)[1])
        if path.suffix == '.s8c':
            return float(f.readline().split()[1])
        assert path.suffix == '.bin' and f.readline().strip()==b'Athena binary output version=1.1'
        count=int(f.readline().split(b'=')[-1]);header={}
        for _ in range(count-1):
            key,value=f.readline().decode().split('=',1);header[key.strip()]=value.strip()
        return float(header['time'])


def copy_checked(source,destination,expected):
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,destination)
    result=receipt(destination)
    assert result['bytes']==expected['bytes'] and result['sha256']==expected['sha256'],str(source)
    return result


def filtered_text(source,destination,cut,cycle):
    """Preserve native comments/header and keep only the accepted time prefix."""
    kept=[]
    with source.open() as f:
        for line in f:
            if not line.strip() or line.startswith('#') or line.startswith('time,'):
                kept.append(line);continue
            parts=line.strip().split(',') if ',' in line else line.split()
            if source.name.endswith('prtcl_destroy.csv'):
                keep=int(parts[0])<cycle and float(parts[1])<=cut+1e-8
            else:
                keep=float(parts[0])<=cut+1e-8
            if keep:kept.append(line)
    destination.write_text(''.join(kept))


def main():
    job();assert ROOT.name=='long'
    marker=ROOT/'evidence/RECOVERY_475604_OK.json'
    if marker.exists():return load(marker)
    work=ROOT/'runs/yc0500';archive=ROOT/'interrupted/475604'
    assert work.is_dir() and not archive.exists(),'Do not overwrite an interrupted archive'
    original_state=load(work/'state.json')
    assert not original_state['complete'] and abs(original_state['latest']['time']-304.3)<1e-7
    source_log=work/'475604_s001.log';text=source_log.read_text(errors='replace')
    assert 'rocm_ipc_cache.c:144' in text and 'failed to open ipc mem handle' in text
    assert '### FATAL' not in text and not re.search(r'search_fail=[1-9]',text)
    scheduler=subprocess.check_output(['sacct','-X','-j','475604','-n','-P','-o','JobIDRaw,State,ExitCode'],text=True)
    assert '475604|FAILED|' in scheduler,scheduler
    checkpoints=[]
    for path in sorted((work/'rst').glob('*.rst')):
        d=validate(path);d.pop('parameters');checkpoints.append(d)
    latest=max(checkpoints,key=lambda r:r['time']);cut=latest['time'];cycle=latest['cycle']
    assert abs(cut-450)<1e-7 and latest['particle_count']==2000000 and latest['ah_onset']<0
    files=[receipt(p) for p in sorted(work.rglob('*')) if p.is_file()]
    inventory=ROOT/'evidence/INTERRUPTED_475604_MANIFEST.json'
    atomic(inventory,dict(utc=now(),job=job(),original_root=str(work),archive_root=str(archive),
          failed_job='475604',scheduler=scheduler,files=files,
          logical_bytes=sum(r['bytes'] for r in files),verified_checkpoints=checkpoints,
          accepted_cutoff_M=cut,reason='UCX ROCm IPC abort; no accepted AH or native physics fatal'))
    # The exact original byte inventory exists before the single-directory move.
    archive.parent.mkdir(parents=True,exist_ok=True);work.rename(archive);work.mkdir()
    by_relative={str(Path(r['path']).relative_to(work)):r for r in files}
    copied=[];derived=[]
    for name in ['pvtk','cohort','bin']:
        for source in sorted((archive/name).glob('*')):
            if source.is_file() and output_time(source)<=cut+1e-8:
                relative=str(source.relative_to(archive))
                copied.append(copy_checked(source,work/relative,by_relative[relative]))
    restart=Path(latest['path']).name;relative='rst/'+restart
    copied.append(copy_checked(archive/relative,work/relative,by_relative[relative]))
    current=validate(work/relative);current.pop('parameters')
    assert current['sha256']==latest['sha256'] and current['time']==cut
    for source in sorted(archive.iterdir()):
        if not source.is_file():continue
        if source.suffix=='.hst' or source.name.endswith('.mass.csv') or source.name.endswith('.ah_quality_0.csv') or source.name.endswith('.prtcl_destroy.csv'):
            filtered_text(source,work/source.name,cut,cycle)
            derived.append(dict(source=str(source),destination=receipt(work/source.name),rule='native rows at or before verified checkpoint time/cycle'))
        elif source.name=='INITIAL_OK.json' or source.name=='initial_particles.bin' or source.name.startswith('475601'):
            copied.append(copy_checked(source,work/source.name,by_relative[source.name]))
    # Only the valid prefix contributes fallback counts. Retain the original
    # aborted log in full; label this canonical derivative explicitly.
    lines=[];after_cut=False
    for line in text.splitlines(keepends=True):
        match=re.search(r'cycle(?:=|\s+)(\d+)',line)
        if match:
            n=int(match[1]);after_cut=after_cut or n>cycle
            if n<=cycle:lines.append(line)
        elif not after_cut:lines.append(line)
    accepted_log=work/'475604_recovered_prefix.log';accepted_log.write_text(''.join(lines))
    assert 'failed to open ipc mem handle' not in accepted_log.read_text()
    derived.append(dict(source=str(archive/source_log.name),destination=receipt(accepted_log),
                        rule='header plus native cycle-labelled records through verified checkpoint; later failed transport tail remains archived'))
    segments=original_state['segments']
    for segment in segments:
        old=segment.get('checkpoint')
        if old and Path(old['path']).name!=restart:
            old['original_path']=old['path'];old['path']=str(archive/'rst'/Path(old['path']).name)
    launch=load(archive/'475604_s001.launch.json')
    launch.update(checkpoint=current,complete=False,log=str(accepted_log),
                  accepted_partial_prefix=True,source_failed_job='475604',
                  source_failed_native_log=receipt(archive/source_log.name),
                  original_attempt_result=load(archive/'475604_s001.result.json'),
                  accepted_time_start_M=original_state['latest']['time'],accepted_time_end_M=cut,
                  simulation_advanced=cut-original_state['latest']['time'],native_node_hours_per_M=None)
    segments.append(launch)
    atomic(work/'checkpoints.json',[current]);atomic(work/'checkpoint_receipts'/f'{restart}.json',current)
    atomic(work/'state.json',dict(segments=segments,latest=current,complete=False,
          interrupted_attempt_preserved=str(archive),recovery_manifest=receipt(inventory)))
    with (work/'yc0500.mass.csv').open() as f:mass=list(csv.DictReader(f))
    assert abs(float(mass[-1]['time'])-cut)<.11 and int(mass[-1]['n_alive'])==2000000
    result=dict(utc=now(),job=job(),archive=str(archive),manifest=receipt(inventory),
                checkpoint=current,accepted_prefix_endpoint_M=cut,copied=copied,derived=derived,
                data_deleted=False,discarded_tail='Excluded from canonical science, retained byte-for-byte in interrupted archive')
    atomic(marker,result)
    return result


if __name__=='__main__':main()
