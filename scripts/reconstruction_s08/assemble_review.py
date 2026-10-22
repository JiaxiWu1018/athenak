"""Build a checksum-verified review packet; authoritative raw stays at original roots."""
from common import *
import csv,shutil,shlex
P=LOCAL/'perseus/continuations/yc0605_from214_20261022';A=P/'final_review_20261022';E=LOCAL/'evidence/final_source_export_20261022';A.mkdir(exist_ok=True)
for name in ['evidence/source_manifests','inputs','reduced','runs','reports','plots','movies','frames','evidence/samples']:(A/name).mkdir(parents=True,exist_ok=True)
assert not (A/'evidence/ASSEMBLY_READY.json').exists(),'Packet already closed; do not rebuild blindly'
from perseus_resources import storage
budget=storage();assert budget['combined_reserved_bytes']+5*GIB<350*GIB
# Verify true finished PS endpoints before final assembly.
for case in ['yc0605','yc0819','yc0900']:
 state=load(P/'runs'/case/'state.json');s=load(P/'reduced'/case/'summary.json');assert state['complete'] and s['complete'] and s['endpoint']==300 and (P/'reduced'/case/'REDUCTION_OK.json').is_file()
 latest=sorted((P/'case_reviews'/case).glob('endpoint_numeric_*/ENDPOINT_QA.json'))[-1];assert load(latest)['numeric_review']=='passed within stated scope'
remote=AMD/'evidence/final_source_export_20261022'
cmd(['rsync','-a','hpcfund.amd.com:'+str(remote/'READY.json'),str(A/'evidence/source_manifests/AMD_READY.json')])
ready=load(A/'evidence/source_manifests/AMD_READY.json');assert ready['status']=='CLOSED_SOURCE_EXPORT_READY'
manifest=A/'evidence/source_manifests/AMD_RAW_AND_REVIEW_SOURCES.json';cmd(['rsync','-a','hpcfund.amd.com:'+str(remote/'AMD_RAW_AND_REVIEW_SOURCES.json'),manifest]);assert sha(manifest)==ready['manifest']['sha256']
export=load(manifest);selected=export['review_entries'];relpaths=[x['relative'] for x in selected];assert len(set(relpaths))==len(relpaths)
listing=A/'evidence/source_manifests/AMD_REVIEW_FILELIST.txt';listing.write_text(''.join(x+'\n' for x in relpaths))
cmd(['rsync','-a','--files-from='+str(listing),'hpcfund.amd.com:'+str(AMD)+'/',str(A)+'/'])
received=[]
for item in selected:
 dest=A/item['relative'];got=receipt(dest);assert got['bytes']==item['bytes'] and got['sha256']==item['sha256'];received.append(dict(source=item,receiver=got))
atomic(A/'evidence/source_manifests/AMD_RECEIVER_VERIFIED.json',dict(utc=now(),job=job(),files=received,all_match=True,scope='Review-onlyselection; fullrawremainsAMD'))
source_map={}
for case in ['yc0283','yc0500']:
 source_map[case]=dict(host='hpcfund.amd.com',root=str(AMD),raw_root=str(AMD/'runs'/case),full_native_raw_present_in_review_packet=False,raw_manifest=str(manifest),native_state=receipt(A/'runs'/case/'state.json'),audit=receipt(A/'reduced'/case/'REDUCTION_OK.json'),profile=receipt(A/'evidence/samples'/f'{case}.json'))
for case in ['yc0605','yc0819','yc0900']:
 work=P/'runs'/case;dest=A/'runs'/case;dest.mkdir(exist_ok=True)
 needed=set([work/'state.json',work/'checkpoints.json',work/f'{case}.user.hst'])
 shape=work/f'{case}.horizon_shape_0.txt'
 if shape.exists():needed.add(shape)
 for seg in load(work/'state.json')['segments']:
  log=work/(str(seg['job'])+'.log');assert log.exists();needed.add(log)
 with (P/'reduced'/case/'cohort_index.csv').open() as f:
  for row in csv.DictReader(f):needed.add(Path(row['file']))
 bindings=[]
 for src in sorted(needed):
  assert work in src.parents;out=A/src.relative_to(P);out.parent.mkdir(parents=True,exist_ok=True)
  if out.exists():assert sha(out)==sha(src)
  elif src.suffix=='.s8c':os.link(src,out)
  else:shutil.copy2(src,out)
  before=receipt(src);after=receipt(out);assert before['bytes']==after['bytes'] and before['sha256']==after['sha256'];bindings.append(dict(source=before,receiver=after))
 red=A/'reduced'/case;red.mkdir(exist_ok=True)
 for src in (P/'reduced'/case).iterdir():
  if src.is_file():shutil.copy2(src,red/src.name)
 for rel in ['inputs/'+case+'.athinput','evidence/samples/'+case+'.json']:shutil.copy2(P/rel,A/rel)
 marker=P/'evidence'/('NATIVE_SEGMENT_CLOSED.json' if case=='yc0605' else f'NATIVE_SEGMENT_CLOSED_{case}.json');closed=A/'evidence/source_manifests'/f'{case}_NATIVE_CLOSED.json';shutil.copy2(marker,closed)
 source_map[case]=dict(host='Perseus',root=str(P),raw_root=str(work),full_native_raw_present_in_review_packet=False,raw_manifest=str(closed),source_manifest=receipt(marker),native_state=receipt(A/'runs'/case/'state.json'),audit=receipt(A/'reduced'/case/'REDUCTION_OK.json'),profile=receipt(A/'evidence/samples'/f'{case}.json'),receiver_bindings=bindings)
for case,info in source_map.items():
 source=Path(info['root']);index=A/'reduced'/case/'cohort_index.csv';original=index.with_name('cohort_index.source.csv');shutil.copy2(index,original)
 with index.open() as f:rows=list(csv.DictReader(f))
 for row in rows:
  rel=Path(row['file']).relative_to(source);dest=A/rel;assert dest.is_file() and rel.parts[:2]==('runs',case);row['file']=str(dest)
 with index.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 info['cohort_index_original']=receipt(original);info['cohort_index_relocated']=receipt(index);info['endpoint']=load(A/'reduced'/case/'summary.json')['endpoint']
atomic(A/'evidence/SOURCE_MAP.json',dict(utc=now(),job=job(),cases=source_map,raw_scope='Metadata/logs/cohort review packet only. No claim of fullnative raw localcopies.',AMD_original_prefix_retained_with_documented_recovery_rotation=True))
(A/'README.md').write_text('# Session 8 final review packet\n\nAllfive actual300M endpoints are source-backed. This packet contains reduced tables, selected native metadata/logs and fixed-ID moviecohorts. Fullparticle and fieldraw remainat the authoritative machinepaths in evidence/SOURCE_MAP.json. Cohortindices are relocated with originalCSV andsource/receiverSHAproofs retained. No simulations or repeatedfullreductions runhere. Finalvisual/scientificacceptance recordedseparately.\n')
atomic(A/'evidence/ASSEMBLY_READY.json',dict(utc=now(),job=job(),source_map=receipt(A/'evidence/SOURCE_MAP.json'),cases=list(source_map),all_original_endpoints=300,scope='Verifiedreviewpacketready; finalplots/movies/reports/acceptance stillrequired'))
print(json.dumps(load(A/'evidence/ASSEMBLY_READY.json'),indent=2))
