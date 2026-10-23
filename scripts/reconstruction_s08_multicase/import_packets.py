"""One-way, checksum-verified cluster review import; Slurm only."""
import shlex
from common import *
from extension_resources import admission,storage

AMD_ROOT='/work1/eliasmost/jiaxiwu/st_s08_scan_20261023'
HOST='hpcfund.amd.com'

def remote(args,check=True):
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',HOST,shlex.join(list(map(str,args)))],capture_output=True,text=True,check=check,timeout=90)

def one(key,source):
    probe=remote(['test','-f',source+'/MANIFEST.json'],check=False)
    if probe.returncode==1:return dict(case=key,ready=False)
    assert probe.returncode==0,probe.stderr
    text=remote(['cat',source+'/MANIFEST.json']).stdout;manifest=json.loads(text)
    destination=EXT/'imported'/key
    previous=destination/'IMPORT_OK.json'
    manifest_hash=hashlib.sha256(text.encode()).hexdigest()
    if previous.exists():
        prior=load(previous);assert prior['source_manifest_sha256']==manifest_hash
        for entry in manifest['files']:
            p=destination/entry['relative'];assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
        return prior
    needed=sum(v['bytes'] for v in manifest['files'] if not (destination/v['relative']).exists())
    disk=storage();assert disk['own']+needed+disk['checkpoint_reserve']<disk['limit']
    destination.mkdir(parents=True,exist_ok=True)
    # Source manifests appear only after completed immutable packet export. Do
    # not --delete or sync in the reverse direction, including after retries.
    cmd(['rsync','-a',HOST+':'+source+'/',str(destination)+'/'])
    assert sha(destination/'MANIFEST.json')==manifest_hash
    checked=[]
    for entry in manifest['files']:
        p=destination/entry['relative'];assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],str(p)
        checked.append(dict(relative=entry['relative'],bytes=entry['bytes'],sha256=entry['sha256']))
    result=dict(utc=now(),job=job(),case=key,ready=True,source=HOST+':'+source,destination=str(destination),source_manifest_sha256=manifest_hash,files=checked,logical_bytes=sum(r['bytes'] for r in checked),one_way=True,raw_full_population_moved=False)
    atomic(previous,result)
    return result

def main():
    job();assert MACHINE=='perseus';admission()
    state=load(LOCAL/'evidence/state.json');results=[]
    for key,c in state['cases'].items():
        if c.get('machine')=='amd':results.append(one(key,AMD_ROOT+'/'+key+'/review_packet'))
    results.append(one('legacy500_packet',AMD_ROOT+'/legacy500_packet'))
    atomic(ROOT/'evidence'/('IMPORT_PASS_'+job()+'.json'),dict(utc=now(),job=job(),packets=results,scope='Import only completed source-manifested review bundles; missing endpoints remain pending.'))
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
