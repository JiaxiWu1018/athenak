"""Allocated, one-way, source-hash-verified native diagnostic staging."""
import argparse
import subprocess
from cfg import *
from os_gate import require_release

def main(case):
    require_job();require_release(ROOT)
    name=case['id'];remote=AMD/'runs'/name/'production';dest=ROOT/'runs'/name/'production'
    dest.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-a','--exclude=out/',f'hpcfund.amd.com:{remote}/',str(dest)+'/'],check=True)
    assert (dest/'RUN_OK').exists()
    completed=load(dest/'completion.json');assert completed['endpoint_M']==1000
    manifest=load(dest/'OUTPUT_MANIFEST.json');assert manifest['case']==name
    final=max(completed['checkpoints'],key=lambda r:r['time'])
    last='out/rst/'+Path(final['path']).name
    selected=[row for row in manifest['files'] if row['path'].startswith(('out/pvtk/','out/bin/'))
              or row['path'].endswith('.hst') or row['path']==last]
    assert any(r['path']==last for r in selected)
    assert sum(r['path'].startswith('out/pvtk/')for r in selected)==501
    listpath=ROOT/'evidence/transfers'/(name+'.files.txt');listpath.parent.mkdir(parents=True,exist_ok=True)
    listpath.write_text(''.join(r['path']+'\n'for r in selected))
    subprocess.run(['rsync','-a','--files-from='+str(listpath),f'hpcfund.amd.com:{remote}/',str(dest)+'/'],check=True)
    for row in selected:
        path=dest/row['path'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],path
    # Preserve the original Session6 interior preflight/production replay check.
    # Only the two2M/4M interior frames are needed, not the full short-run raw.
    preflight_remote=AMD/'runs'/name/'preflight'
    preflight_dest=ROOT/'runs'/name/'preflight_validation';preflight_dest.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-a','--exclude=out/',f'hpcfund.amd.com:{preflight_remote}/',str(preflight_dest)+'/'],check=True)
    preflight_manifest=load(preflight_dest/'OUTPUT_MANIFEST.json')
    particle_files=sorted((f for f in preflight_manifest['files'] if f['path'].startswith('out/pvtk/')),key=lambda f:f['path'])
    assert len(particle_files)==11
    interior=[particle_files[1],particle_files[2]]
    interior_list=ROOT/'evidence/transfers'/(name+'.preflight.files.txt')
    interior_list.write_text(''.join(f['path']+'\n'for f in interior))
    subprocess.run(['rsync','-a','--files-from='+str(interior_list),f'hpcfund.amd.com:{preflight_remote}/',str(preflight_dest)+'/'],check=True)
    for f in interior:
        path=preflight_dest/f['path'];assert path.stat().st_size==f['bytes'] and sha(path)==f['sha256']
    for path in ['samples/'+name+'.bin','evidence/samples/'+name+'.json','evidence/SAMPLES_OK.json']:
        target=ROOT/path;target.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rsync','-a',f'hpcfund.amd.com:{AMD}/{path}',str(target)],check=True)
    sample_receipt=load(ROOT/'evidence/samples'/(name+'.json'))
    assert sha(ROOT/'samples'/(name+'.bin'))==sample_receipt['sha256']
    atomic(ROOT/'evidence/transfers'/(name+'.json'),dict(utc=now(),job=os.environ['SLURM_JOB_ID'],
        source=source(),case=name,source_output_manifest_sha256=sha(dest/'OUTPUT_MANIFEST.json'),
        files=selected,logical_bytes=sum(r['bytes']for r in selected),
        all_destination_checksums_verified=True,source_all_outputs_retained=True,
        preflight_interior_files=interior,preflight_manifest_sha256=sha(preflight_dest/'OUTPUT_MANIFEST.json'),
        preflight_interior_logical_bytes=sum(f['bytes']for f in interior),
        source_raw_logical_bytes=manifest['logical_bytes'],retention='No deletion authorized; all native data retained.'))
    print('TRANSFER_OK',name,len(selected),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('index',type=int);main(CASES[p.parse_args().index])
