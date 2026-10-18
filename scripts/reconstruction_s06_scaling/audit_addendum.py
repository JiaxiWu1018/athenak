"""Allocated Session6 paired/scaling integrity closeout; original deliveries immutable.

This audits final reviewed artifacts and retained provenance. It does not rerun
closed science, change numerical definitions, or claim scientific stability.
"""
import json
import os
import re
import subprocess
from pathlib import Path
from cfg import ROOT, BASE, AMD, CASES, PHYSICS, KOKKOS, EXE_SHA, P, now, load, sha, atomic, require_job
from os_gate import require_release

DEST = ROOT/'evidence/PAIRED_SCALING_ADDENDUM.json'

def entry(path):
    path=Path(path)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))

def verify(spec,base=ROOT):
    path=Path(spec['path'])
    if not path.is_absolute():path=base/path
    assert path.is_file(),path
    if 'bytes'in spec:assert path.stat().st_size==spec['bytes'],path
    assert sha(path)==spec['sha256'],path
    return path

def command(*args):
    return subprocess.check_output(args,text=True).strip()

def inventory(path):
    files=[p for p in path.rglob('*')if p.is_file()and not p.is_symlink()]
    return dict(path=str(path),files=len(files),logical_bytes=sum(p.stat().st_size for p in files),
                allocated_bytes=int(command('du','-s','-B1',str(path)).split()[0]))

def main():
    require_job();assert not DEST.exists(),'Preserve previous final manifest'
    gate=require_release(ROOT)
    repo=ROOT/'src/athenak';pin=load(ROOT/'evidence/SOURCE_PIN.json')
    head=command('git','-C',str(repo),'rev-parse','HEAD')
    branch=command('git','-C',str(repo),'branch','--show-current')
    remote=command('git','-C',str(repo),'ls-remote','origin','refs/heads/'+branch).split()[0]
    assert head==remote==pin['head']==pin['remote_head_verified']
    assert not command('git','-C',str(repo),'status','--porcelain')
    subprocess.run(['git','-C',str(repo),'diff','--exit-code',PHYSICS,'HEAD','--','src'],check=True)
    assert command('git','-C',str(repo/'kokkos'),'rev-parse','HEAD')==KOKKOS
    assert sha(ROOT/'protected/athena')==EXE_SHA
    for row in pin['scripts']:verify(row,repo)
    paths=set()
    def add(path):
        path=Path(path);assert path.is_file(),path;paths.add(path)
    preservation=load(ROOT/'evidence/before_final_report_addendum/PRESERVATION.json')
    for record in preservation['files']:
        spec=dict(record,path=record['preserved']);verify(spec)
        if record['immutable']:verify(record)
        add(record['preserved'])
    add(ROOT/'evidence/before_final_report_addendum/PRESERVATION.json')
    qa=load(ROOT/'evidence/RESULTS_ARITHMETIC_QA.json')
    assert qa['passed']and sum(qa['window_rows'].values())==612 and len(qa['native_checks'])==9
    # The independent full-N audit already read/recomputed each native state;
    # recheck hashes at final promotion, without repeating science or all271GB.
    for state in qa['native_checks']:verify(state)
    figures=load(ROOT/'evidence/FINAL_FIGURE_VISUAL_REVIEW.json')
    assert figures['accepted']and len(figures['figures'])==8
    for record in figures['figures']+figures['PDF_previews']:add(verify(record))
    review=load(ROOT/'evidence/FINAL_MOVIE_VISUAL_REVIEW.json')
    assert review['accepted']and set(review['groups'])=={'0','1','2','paired','scaling'}
    reviewed_movies={};render_jobs=set();cache_paths=set()
    for group,record in review['groups'].items():
        receipt=verify(record['render_receipt']);add(receipt);render=load(receipt)
        assert render['group']==group and render['full_decode_passed']and render['clipping']==0
        assert render['frames']==501 and render['fps']==15 and render['display_count_per_panel']==20000
        assert render['view_limit_M']==16 and render['color_range_M']==[0,8.936231120010849]
        assert render['review_frames']==[0,150,300,400,419,500]
        stream=render['ffprobe']['streams'][0]
        assert stream['nb_read_frames']=='501'and stream['r_frame_rate']=='15/1'
        assert stream['height']==1080 and stream['width']==1200*len(render['cases'])
        assert record['accepted']and record['actual_encoded_frames_inspected']
        movie=verify(render['movie']);add(movie);add(verify(render['contact_sheet']))
        assert record['movie']['sha256']==render['movie']['sha256']
        decoded=sorted((receipt.parent/'decoded').glob('decoded_*.png'))
        assert len(decoded)==6 and record['decoded_frames_inspected']==[str(p)for p in decoded]
        paths.update(decoded)
        frames=sorted((receipt.parent/'frames').glob('frame_*.png'))
        assert[p.name for p in frames]==[f'frame_{k:04d}.png'for k in range(501)]
        paths.update(frames);reviewed_movies[group]=render['movie'];render_jobs.add(render['job'])
        for data in render['data']:cache_paths.add(verify(data))
    paths.update(cache_paths)
    native=[]
    for c in CASES:
        name=c['id'];accepted=load(ROOT/f'evidence/reductions/ACCEPTED_{name}.json')
        assert accepted['states']==501 and accepted['full_N']==c['n']and accepted['endpoint_M']==1000
        assert accepted['source_native_and_destination_hashes_checked_by_allocated_transfer']
        for evidence in accepted['evidence']:add(verify(evidence))
        transfer_path=ROOT/f'evidence/transfers/{name}.json';transfer=load(transfer_path)
        manifest_path=ROOT/f'evidence/amd/runs/{name}/production/OUTPUT_MANIFEST.json';manifest=load(manifest_path)
        assert sha(manifest_path)==transfer['source_output_manifest_sha256']
        assert transfer['all_destination_checksums_verified']and transfer['source_all_outputs_retained']
        assert manifest['endpoint_M']==1000 and manifest['source']=='e8fa9db615f99eb410aae346239624bb44d3bcd7'
        source={a['path']:a for a in manifest['files']}
        assert len(source)==2039 and len(transfer['files'])==2029
        for f in transfer['files']:
            assert f==source[f['path']]
            path=ROOT/'runs'/name/'production'/f['path']
            assert path.is_file()and path.stat().st_size==f['bytes'],path
        cache=load(ROOT/f'movie_cache/{name}/DATA_OK.json')
        assert cache['frames']==501 and cache['display_count']==20000
        if c['paired']:assert cache['complete_partner_pairs']
        add(verify(cache['initial_sample']));add(verify(cache['cache']))
        launch=load(ROOT/f'evidence/amd/runs/{name}/production/launch.json')
        assert launch['executable_sha256']==EXE_SHA and launch['compiled_source']==PHYSICS and launch['kokkos']==KOKKOS
        assert sha(ROOT/f'inputs/{name}.athinput')==launch['input_sha256']
        add(ROOT/f'inputs/{name}.athinput')
        add(ROOT/f'evidence/reductions/ACCEPTED_{name}.json')
        native.append(dict(case=name,AMD_root=str(AMD/'runs'/name/'production/out'),AMD_files=len(source),
            AMD_logical_bytes=sum(f['bytes']for f in source.values()),
            Perseus_root=str(ROOT/'runs'/name/'production/out'),Perseus_files=len(transfer['files']),
            Perseus_logical_bytes=transfer['logical_bytes'],source_manifest=entry(manifest_path),transfer_receipt=entry(transfer_path),
            native_hash_basis='All selected hashes verified in allocated transfer/reduction; representative9 full-N frames independently recomputed and hash-rechecked here. Final audit checks current sizes of every retained2029nativefile/case; it does not redundantly rehash271GB.'))
    assert sum(n['AMD_logical_bytes']for n in native)==348741231441
    assert sum(n['Perseus_logical_bytes']for n in native)==271556761771
    readout=load(ROOT/'comparison/SCIENCE_READOUT.json')
    for data in readout['inputs']:add(verify(data))
    assert readout['period_M']==P and len(readout['cases'])==4
    science_script=repo/'scripts/reconstruction_s06_scaling/science_readout.py'
    assert sha(science_script)==readout['script']['sha256']
    amd=load(ROOT/'evidence/AMD_ACCOUNTING_FINAL.json')
    assert len(amd['rows'])==7 and all(row[1:3]==['COMPLETED','0:0']for row in amd['rows'])
    assert sum(int(row[3])for row in amd['rows'])==72893
    ps=load(ROOT/'evidence/PERSEUS_ANALYSIS_ACCOUNTING.json')
    assert all(row['terminal_state']=='COMPLETED'and row['exit_code']=='0:0'for row in ps['rows'])
    assert render_jobs.issubset({row['id']for row in ps['rows']})
    # Final delivery audit's own terminal timing is recorded separately after exit.
    storage=[inventory(ROOT/name)for name in ('runs','samples','protected','reference','baseline','preflight_health','reduced','comparison','plots','interim','movie_cache','movie_render','movie_render_layout_v2','movies','src','logs')]
    stat=os.statvfs(ROOT);free=stat.f_bavail*stat.f_frsize
    storage_receipt=dict(utc=now(),job=os.environ['SLURM_JOB_ID'],Perseus_free_bytes=free,paths=storage,native=native,
        AMD_last_allocated_whole_user_bytes=1188105785344,AMD_last_measurement_UTC='2026-10-18T08:53:25Z',
        no_new_AMD_growth=True,no_deletion=True,no_transfer_to_Anta_or_Mac=True,
        retention='All source/raw/checkpoints retained. Ten nonfinal checkpoints per case remain AMD-only. Version1cropped-labelsinglemovies and allframes preserved as superseded evidence; promotedversion2singles separate. No backup claim.')
    atomic(ROOT/'evidence/STORAGE_FINAL.json',storage_receipt)
    rows=['# Follow-up retained storage','',f'Allocated inventory job {os.environ["SLURM_JOB_ID"]}; measured {storage_receipt["utc"]}. All paths remain on Perseus unless explicitly AMD. No deletion, Anta/Mac transfer or backup claim.','',
          '| Perseus path | Files | Logical bytes | Allocated bytes |','|---|---:|---:|---:|']
    rows += [f"| `{v['path']}` | {v['files']} | {v['logical_bytes']} | {v['allocated_bytes']} |"for v in storage]
    rows += ['',f'Perseus available bytes: {free}. The inventory precedes its own evidence/manifest/log closure, so mutable controller logs and final evidence bytes are not a frozen whole-root total.','',
             'AMD native roots and exact source/verified selected-copy inventories: [STORAGE_FINAL.json](../evidence/STORAGE_FINAL.json). All348741231441AMDnativebytes remain retained. SelectedPerseusnativecopies271556761771bytes are independently source-hash-verified. Ten nonfinalcheckpoints/case remainAMD-only. Last allocated AMD whole-user measurement1188105785344bytes at08:53:25UTC18Oct; watchers exited after allproductionsclosed; no furtherAMDgrowth.','',
             'movie_render retains the first exports, including three singles superseded for cropped colour-bar labels. movie_render_layout_v2 retains corrected singles. Allframes,decodedchecks,cachesandbothversionsremain retained. Logical/allocated bytes above count these preserved versions.','']
    atomic(ROOT/'reports/STORAGE.md','\n'.join(rows))
    report=ROOT/'reports/REPORT_paired_scaling.md'
    assert '<!-- FINAL_INTEGRITY -->'not in report.read_text()
    atomic(report,report.read_text()+f'\n\n<!-- FINAL_INTEGRITY -->\nFinal allocated integrity audit{os.environ["SLURM_JOB_ID"]} verified promoted figures/five movies, actual visual acceptances, source/pushed branch, original immutable manifests, transfer/native evidence, report links and retained storage. Completed Perseus science/analysis/render jobs before this integrity allocation total{ps["allocated_seconds"]}seconds = {ps["job_hours"]:.6f}jobh. The final integrity allocation is recorded separately at terminal closeout. [Frozen analysis accounting](../evidence/PERSEUS_ANALYSIS_ACCOUNTING.json), [separate addendum manifest](../evidence/PAIRED_SCALING_ADDENDUM.json), [storage](STORAGE.md).\n')
    documents=[ROOT/'README.md',*[p for p in (ROOT/'reports').glob('*.md')],BASE/'README.md',BASE/'reports/REPORT_session6.md',BASE/'reports/FIGURES.md']
    links=[]
    for doc in documents:
        add(doc)
        for href in re.findall(r'\]\(([^)]+)\)',doc.read_text()):
            if href.startswith(('https:','http:','#','mailto:')):continue
            dest=(doc.parent/href.strip('<>').split('#')[0]).resolve()
            assert dest==DEST or dest.exists(),(doc,href,dest)
            links.append(dict(document=str(doc),href=href,resolved=str(dest)))
    for directory in ('reduced','comparison','plots','inputs','preflight_health','reference','baseline'):
        paths.update(p for p in (ROOT/directory).rglob('*')if p.is_file())
    # Bind immutable receipts and exact job evidence; do not bind mutable live state/controller/accounting.
    fixed=['SOURCE_PIN.json','SOURCE_PIN_FINAL_PROMOTION.json','AMD_ACCOUNTING_FINAL.json','PERSEUS_ANALYSIS_ACCOUNTING.json',
           'OS_PRIORITY_RELEASE.json','PREFLIGHT_REVIEW.json','PREFLIGHT_HEALTH_QA.json','RESULTS_ARITHMETIC_QA.json',
           'FINAL_FIGURE_VISUAL_REVIEW.json','FINAL_MOVIE_VISUAL_REVIEW.json','MOVIE_LAYOUT_REVIEW_V1.json','STORAGE_FINAL.json',
           'POSTPRODUCTION_CHAIN_SUBMITTED.json','MOVIE_LAYOUT_V2_SUBMITTED.json','SCIENCE_READOUT_SUBMITTED.json','FINAL_INTEGRITY_SUBMITTED.json']
    for name in fixed:add(ROOT/'evidence'/name)
    for directory in ('amd','production','reductions','transfers','source_pins','submission_scripts'):
        paths.update(p for p in (ROOT/'evidence'/directory).rglob('*')if p.is_file())
    for row in ps['rows']:
        for p in (ROOT/'evidence/jobs'/row['id']).glob('*'):
            if p.is_file()and p.name!='scontrol_latest.txt':add(p)
        for p in (ROOT/'logs').glob('*.'+row['id']+'.log'):add(p)
    add(ROOT/'protected/athena')
    for p in (repo/'scripts/reconstruction_s06_scaling').glob('*'):
        if p.is_file():add(p)
    files=[entry(p)for p in sorted(paths)]
    print('All final checks passed; freezing artifact hashes',flush=True)
    atomic(DEST,dict(utc=now(),job=os.environ['SLURM_JOB_ID'],status='complete',scope='Session6paired/scalingfollowup only',
        source=head,branch=branch,remote_head_verified=remote,compiled_physics=PHYSICS,kokkos=KOKKOS,executable_sha256=EXE_SHA,
        physics_source_diff_empty=True,OS_priority_gate=gate,native=native,reviewed_movies=reviewed_movies,
        independent_arithmetic_native_QA_passed=True,actual_all_four_PNG_PDF_and_five_encoded_movie_reviews_accepted=True,
        original_science_and_movie_manifests_unchanged=True,original_documents_preserved=True,
        AMD_node_hours=amd['node_hours'],Perseus_analysis_job_hours_excluding_this_integrity_job=ps['job_hours'],
        storage=storage,links=links,files=files,no_new_evolution=True,no_deletion=True,no_paper_suite_launch=True))
    atomic(ROOT/'evidence/FINAL_AUDIT_OK.json',dict(utc=now(),job=os.environ['SLURM_JOB_ID'],passed=True,
        manifest=entry(DEST),files=len(files),links=len(links),science_threshold_imposed=False,
        completion_marker_deferred_to_verified_terminal_exit=True))
    print('PAIRED_SCALING_ADDENDUM_OK',len(files),'files',len(links),'links',flush=True)

if __name__=='__main__':main()
