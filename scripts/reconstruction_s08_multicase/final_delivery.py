"""Validate all five completed products and make a checksum-bound thin mirror.

Run only in a Slurm allocation after FINAL_VISUAL_ACCEPTANCE.json is authored
from actual exported-PDF and encoded-frame inspection. This never launches a
simulation, deletes data, or claims delivery to the Mac.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time

BATCH = Path('/data/jiaxiwu/NRPIC/reconstruction_ST/08_production_20261021/extensions/yc0500_to700_and_inward_scan_20261023')
AMD = Path('/work1/eliasmost/jiaxiwu/st_s08_scan_20261023')
PDFS = ['s08_extension_shells_lapse.pdf', 's08_extension_mass_conservation.pdf', 's08_extension_AH_masses.pdf', 's08_extension_boundary_lapse.pdf']


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for data in iter(lambda: f.read(8*1024**2), b''):
            h.update(data)
    return h.hexdigest()


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.' + os.environ['SLURM_JOB_ID'] + '.tmp')
    assert not tmp.exists(), tmp
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(p)


def receipt(p):
    p = Path(p)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def run(args):
    return subprocess.check_output(list(map(str, args)), text=True, timeout=180)


def endpoint(c, a):
    assert a['numerical_checks_passed'] and a['checkpoint']['initial_count'] == 2_000_000
    ah = a['first_accepted_AH_M']
    expected = min(c['target'], ah + 20) if ah is not None else c['target']
    assert expected-1e-7 <= a['endpoint_M'] <= min(c['target']+1e-7, expected+a['checkpoint']['dt']+1e-7)
    assert a['case']['key'] == c['id']
    return expected


def visual_scope(v, keys, movies, assembly_sha):
    assert v['actual_visual_acceptance'] is True and v['boundary_reviewed'] is True
    assert set(v['case_keys']) == set(keys) and len(v['case_keys']) == 5
    assert set(v['pdf_source_sha256']) == set(PDFS)
    assert set(v['movie_source_sha256']) == set(movies)
    assert set(v['inspected_encoded_frames']) == set(movies)
    assert all(v['inspected_encoded_frames'][name] for name in movies)
    assert v['assembly_sha256'] == assembly_sha


def remote_snapshot(keys):
    # Read small JSON proofs and filesystem metadata on the AMD login host;
    # all raw-content hashing was performed in allocated AMD jobs/steps.
    code = r'''
import datetime,hashlib,json,pathlib,shutil,socket,subprocess,sys
x=pathlib.Path('/work1/eliasmost/jiaxiwu/st_s08_scan_20261023');keys=json.loads(sys.argv[1]);r={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'metadata_host':socket.gethostname(),'manifests':{},'retention':{}}
for key in keys:
 assert key in ['yc0500_long700','yc0605_u0020','yc0819_u0020']
 p=x/key/'evidence/CLOSED_OUTPUT_MANIFEST.json';data=p.read_bytes();assert len(data)<16*1024**2
 r['manifests'][key]={'data':json.loads(data),'source':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 p=x/key/'evidence/ENDPOINT_RETENTION.json'
 if p.exists():r['retention'][key]=json.loads(p.read_text())
paths={'new_batch':x,'whole_AMD_user':pathlib.Path('/work1/eliasmost/jiaxiwu'),**{key:x/key for key in keys}}
r['allocated_bytes']={name:int(subprocess.check_output(['du','-s','-B1',str(p)],text=True,timeout=90).split()[0]) for name,p in paths.items()}
r['filesystem_free_bytes']=shutil.disk_usage(x).free
r['queue']=subprocess.check_output(['squeue','-u','jiaxiwu','-h','-o','%i|%T'],text=True)
print(json.dumps(r))
'''
    return json.loads(run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', 'hpcfund.amd.com', shlex.join(['python3', '-c', code, json.dumps(keys)])]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Allocated compute job required'
    assert re.fullmatch('[0-9a-f]{40}', args.source_commit)
    job = os.environ['SLURM_JOB_ID']
    root = BATCH/'perseus/final'
    cases = load(BATCH/'evidence/cases.json')
    keys = [c['id'] for c in cases]
    assert len(keys) == len(set(keys)) == 5
    names = keys + ['s08_extension_comparison']
    assembly = load(root/'evidence/ASSEMBLY_OK.json')
    v = load(root/'evidence/FINAL_VISUAL_ACCEPTANCE.json')
    visual_scope(v, keys, names, sha(root/'evidence/ASSEMBLY_OK.json'))
    assert assembly['source_files_SHA256_verified'] and {r['case'] for r in assembly['sources']} == set(keys)
    assert load(root/'evidence/REPORTS_READY.json')['source_assembly']['sha256'] == sha(root/'evidence/ASSEMBLY_OK.json')
    packets = {}
    audits = {}
    inputs = root/'inputs'
    inputs.mkdir(exist_ok=True)
    proof = root/'evidence/source_acceptance'
    proof.mkdir(exist_ok=True)
    for c in cases:
        key = c['id']
        binding = next(r for r in assembly['sources'] if r['case'] == key)
        p = Path(binding['packet']['path'])
        assert p.resolve().is_relative_to(BATCH/'perseus') and sha(p) == binding['packet']['sha256']
        packet = load(p)
        a_path = p.parent/'numerical_acceptance.json'
        assert sha(a_path) == binding['numerical_acceptance']['sha256']
        a = load(a_path)
        endpoint(c, a)
        assert a['checkpoint']['sha256'] == packet['checkpoint']['sha256']
        assert packet['case']['key'] == key
        packets[key] = p.parent
        audits[key] = a
        shutil.copy2(a_path, proof/(key+'.json'))
        shutil.copy2(p.parent/'exact.athinput', inputs/(key+'.athinput'))
    legacy = next(r['prior500'] for r in assembly['sources'] if r['case'] == 'yc0500_long700')
    assert legacy and sha(legacy['path']) == legacy['sha256']
    assert load(legacy['path'])['anchor']['sha256'] == '7309c1b16b29d5c3de300dadce3ab37be2ba73796ae92b78ed45681134ff2e64'
    boundary = load(root/'evidence/BOUNDARY_DIAGNOSTICS.json')
    assert abs(boundary['endpoint_M']-audits['yc0500_long700']['endpoint_M']) < 1e-7
    previews = load(root/'evidence/PDF_PREVIEWS.json')
    plot_assets = set()
    for name in PDFS:
        p = root/'plots'/name
        plot_assets.add(p)
        if p.with_suffix('.png').exists():
            plot_assets.add(p.with_suffix('.png'))
        assert sha(p) == v['pdf_source_sha256'][name]
        row = next(r for r in previews['files'] if Path(r['source']['path']).name == name)
        assert row['source']['sha256'] == sha(p)
        assert sha(row['preview']['path']) == row['preview']['sha256']
        assert Path(row['preview']['path']).resolve().is_relative_to(root/'plots')
        plot_assets.add(Path(row['preview']['path']))
    for name in names:
        m = load(root/'evidence'/('MOVIE_'+name+'_OK.json'))
        p = root/'movies'/(name+'.mp4')
        assert m['full_decode'] and sha(p) == m['sha256'] == v['movie_source_sha256'][name]
        assert m['source_assembly']['sha256'] == sha(root/'evidence/ASSEMBLY_OK.json')
        fm = m['input_frames_manifest']
        assert sha(fm['path']) == fm['sha256']
        frame_manifest = load(fm['path'])
        assert frame_manifest['fps'] == 15 and len(frame_manifest['frames']) == m['frames']
        assert set(v['inspected_encoded_frames'][name]).issubset({r['frame'] for r in m['decoded_previews']})
        for preview in m['decoded_previews']:
            p = Path(preview['path'])
            assert p.resolve().is_relative_to(root/'plots') and sha(p) == preview['sha256']
            plot_assets.add(p)
        stream = json.loads(run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=nb_frames,avg_frame_rate,width,height', '-of', 'json', p]))['streams'][0]
        assert stream['avg_frame_rate'] == '15/1' and int(stream['nb_frames']) == m['frames']
    state = load(BATCH/'evidence/state.json')
    ps_raw = run(['squeue', '-u', 'jiaxiwu', '-h', '-o', '%i|%T'])
    ps_live = dict(line.split('|') for line in ps_raw.splitlines())
    registered_ps = {str(r['id']) for r in state['Perseus_jobs']}
    assert not ((registered_ps-{job}) & set(ps_live)), 'Close other registered allocations before delivery'
    remote = remote_snapshot([key for key in keys if state['cases'][key]['machine'] == 'amd'])
    amd_live = {line.split('|')[0] for line in remote['queue'].splitlines()}
    assert not ({str(r['id']) for r in state['jobs']} & amd_live), 'AMD endpoint/retention workers still live'
    inventories = {}
    closed_dir = root/'evidence/closed_output_manifests'
    closed_dir.mkdir(exist_ok=True)
    for key in keys:
        p = closed_dir/(key+'.json')
        if key in remote['manifests']:
            value = remote['manifests'][key]
            # Keep the original byte representation so source SHA256 remains exact.
            source = remote['manifests'][key]['source']
            raw = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', 'hpcfund.amd.com', shlex.join(['cat', source])], timeout=60)
            assert hashlib.sha256(raw).hexdigest() == value['sha256'] and len(raw) == value['bytes']
            p.write_bytes(raw)
        else:
            shutil.copy2(BATCH/'perseus'/key/'evidence/CLOSED_OUTPUT_MANIFEST.json', p)
        m = load(p)
        assert m['stable_files_verified'] and m['case']['key'] == key
        assert m['checkpoint']['sha256'] == audits[key]['checkpoint']['sha256']
        assert m['file_count'] == len(m['files']) and m['logical_bytes'] == sum(r['bytes'] for r in m['files'])
        assert any(r['sha256'] == audits[key]['checkpoint']['sha256'] and r['path'] == audits[key]['checkpoint']['path'] for r in m['files'])
        inventories[key] = receipt(p)
    sys.path.insert(0, str(Path(__file__).parent))
    from submit_continuations import budget, elapsed
    upper, details = budget(state, ps_live)
    own_scheduler = run(['scontrol', 'show', 'job', '-o', job])
    own_limit = elapsed(re.search(r'\bTimeLimit=(\S+)', own_scheduler)[1])
    own_nodes = int(re.search(r'\bNumNodes=(\d+)', own_scheduler)[1])
    if job not in registered_ps:
        own = own_limit*own_nodes/3600
        upper += own
        details.append(dict(job=job, node_hours_upper=own, basis='current closeout declared allocation bound'))
    else:
        own_record = next(r for r in state['Perseus_jobs'] if str(r['id']) == job)
        assert own_limit*own_nodes/3600 <= own_record['hours']*own_record['nodes']+1e-9
    policy = load(BATCH/'evidence/RESOURCE_RESERVATIONS.json')
    assert upper <= policy['node_hour_review']
    account = dict(UTC=now(), allocated_node_hours_upper=upper, per_job=details, registered_native_and_review_jobs_quiescent=True,
        current_closeout_job=job, scope='Terminal scheduler rows are exact. Missing terminal records and the current closeout allocation retain explicit bounds; this is not an exact invoice. Shared-node allocations are conservatively counted as one node each. No analysis step inside an existing allocation is double charged.')
    write(root/'evidence/FINAL_ACCOUNTING.json', account)
    paths = {key:dict(machine=state['cases'][key]['machine'], path=str(AMD/key if state['cases'][key]['machine']=='amd' else BATCH/'perseus'/key), allocated_bytes=remote['allocated_bytes'][key] if key in remote['allocated_bytes'] else int(run(['du','-s','-B1',BATCH/'perseus'/key]).split()[0]), status='Retained', inventory=inventories[key]) for key in keys}
    storage = dict(UTC=now(), raw_case_retention=paths, AMD_metadata=remote, policy=policy, cleanup='No data deletion by closeout. Earlier authorized superseded-checkpoint rotations retain their original inventories. Previous sessions and Jeans are preserved.')
    # The remote snapshot embeds the source inventories; retain their compact
    # verified files once and reference them in the closing storage receipt.
    storage['AMD_metadata'] = {k:v for k,v in remote.items() if k != 'manifests'}
    write(root/'evidence/FINAL_RETENTION.json', storage)
    lines = ['# Accounting and retention', '', f'Conservative allocated node-hour upper total: **{upper:.6f}** against the {policy["node_hour_review"]}-hour review allowance. Exact terminal evidence and missing-terminal bounds are distinguished in FINAL_ACCOUNTING.json; this is not an exact invoice.', '', '| Case | Machine | Retained raw root | Allocated bytes at inventory |', '|---|---|---|---:|']
    for key,r in paths.items():
        lines.append(f'| {key} | {r["machine"]} | `{r["path"]}` | {r["allocated_bytes"]:,} |')
    lines += ['', 'Raw outputs, reduced arrays, logs, source snapshots, restarts and lossless frames stay on the clusters. All five closed-output inventories bind their selected files to the accepted final checkpoints. Shared matched-control files must be deduplicated when summing physical storage. No data deletion occurs during this delivery; previous sessions and Jeans remain preserved.', '', 'The thin review mirror contains reports, exact inputs, final figures/movies and compact validation/provenance records. It is prepared on Perseus for one-way review delivery. **No Mac delivery is claimed.**']
    (root/'reports/ACCOUNTING_AND_RETENTION.md').write_text('\n'.join(lines)+'\n')
    lines = ['# Outer-domain lapse assessment', '', f'The unperturbed trajectory reaches {boundary["endpoint_M"]:.6f} M_ref in the original domain with coordinate half-width {boundary["domain_halfwidth_M"]} M_ref. The diagnostic joins the accepted 0–500 M data to the new continuation with their exact source manifests.', '', 'The heatmap shows coordinate-area weighted equatorial lapse means in 64 logarithmic annuli over 16 <= r < 512 M_ref, relative to their initial values. The outer-band curves equally average the annular means; they are not full-band area-weighted averages.', '', '| Radial band [M_ref] | Initial lapse | Lapse at500M | Endpoint lapse | Max absolute change after500M |', '|---|---:|---:|---:|---:|']
    for r in boundary['outer_bands']:
        lines.append(f'| {r["radius_min_M"]}–{r["radius_max_M"]} | {r["initial_lapse"]:.6g} | {r["lapse_at500M"]:.6g} | {r["endpoint_lapse"]:.6g} | {r["maximum_absolute_change_after500_from500"]:.6g} |')
    lines += ['', 'These curves document outer-domain gauge evolution. No larger-domain control was run, and this diagnostic does not establish causal isolation through the extended interval. It is not an ADM-energy measurement.', '', '[Exported diagnostic figure](../plots/s08_extension_boundary_lapse.pdf); [source-bound numerical record](../evidence/BOUNDARY_DIAGNOSTICS.json).']
    (root/'reports/BOUNDARY_ASSESSMENT.md').write_text('\n'.join(lines)+'\n')
    lines = ['# Session 8: completed five-run extension', '', 'All five requested endpoints passed their source numerical audits and actual exported-PDF/encoded-frame review. The original proper-volume stratified seed-1 samples contain 2,000,000 particles; the three .500 variants share the same realization. Times and AH masses use the original equilibrium mass M_ref=1.', '', '| Case | Endpoint [M_ref] | First accepted AH [M_ref] | Final alive particles | Late qualitative AH mass [M_ref] |', '|---|---:|---:|---:|---:|']
    for c in cases:
        a = audits[c['id']]
        ah = 'none' if a['first_accepted_AH_M'] is None else f'{a["first_accepted_AH_M"]:.3f}'
        w = a['AH_endpoint_window']
        mass = '—' if w is None else f'{w["mirr"]["median"]:.3f}'
        lines.append(f'| {c["id"]} | {a["endpoint_M"]:.3f} | {ah} | {a["final_population"]:,} | {mass} |')
    lines += ['', 'Accepted persistent AHs establish black-hole formation. Survival, expansion or contraction within a finite interval does not establish stable migration. AH masses are qualitative final-window measurements, without a precision or settling claim.', '', 'Shells end strictly before first removal. The particle rest-mass ledger is bookkeeping, separate from AH mass and ADM-energy conservation. Early lapse removals without a preceding qualified AH are explicitly documented in the source audits. The inherited source-time coupling remains unchanged; no new convergence or causal-isolation claim is made.', '', '- [Science and individual outcomes](SCIENCE_REPORT.md)', '- [Shell/lapse fiducial](../plots/s08_extension_shells_lapse.pdf)', '- [Rest-mass ledger](../plots/s08_extension_mass_conservation.pdf)', '- [Accepted AH masses](../plots/s08_extension_AH_masses.pdf)', '- [Outer-domain lapse assessment](../plots/s08_extension_boundary_lapse.pdf)', '- [Synchronized movie](../movies/s08_extension_comparison.mp4)', '- [Accounting and raw retention](ACCOUNTING_AND_RETENTION.md)', '', 'Individual movies:']
    lines += [f'- [{key}](../movies/{key}.mp4)' for key in keys]
    (root/'reports/FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    (root/'README.md').write_text('# Session 8 five-run extension\n\nRead [the final report](reports/FINAL_REPORT.md). Exact inputs, final figures/movies and compact acceptance/provenance records are provided for review. Raw data, reductions, logs, source snapshots, checkpoints and lossless frame sequences remain on the clusters. The one-way thin copy is bound by evidence/REVIEW_PRODUCT_MANIFEST.json. No Mac delivery is claimed.\n')
    write(root/'evidence/FINAL_SOURCE_PROVENANCE.json', dict(UTC=now(), job=job, host=socket.gethostname(), source_commit=args.source_commit,
        source_script=receipt(__file__), assembly=receipt(root/'evidence/ASSEMBLY_OK.json'), actual_visual=receipt(root/'evidence/FINAL_VISUAL_ACCEPTANCE.json'), closed_inventories=inventories,
        exact_inputs={key:receipt(inputs/(key+'.athinput')) for key in keys}, native_backends='Preserved CUDA12.9/Kokkos and ROCm6.4.1/GFX90A only; exact executable/toolchain/launch records remain in source packets and job snapshots.', Mac_delivery=False))
    selected = {root/'README.md', root/'reports/extension_summary.csv', *plot_assets}
    selected.update(root/'movies'/(name+'.mp4') for name in names)
    selected.update(inputs/(key+'.athinput') for key in keys)
    selected.update(root/'reports'/name for name in ['SCIENCE_REPORT.md','FIGURES.md','MOVIES.md','TECHNICAL_REPORT.md','ACCOUNTING_AND_RETENTION.md','BOUNDARY_ASSESSMENT.md','FINAL_REPORT.md'])
    required = ['ASSEMBLY_OK.json','FINAL_VISUAL_ACCEPTANCE.json','PDF_PREVIEWS.json','REPORTS_READY.json','BOUNDARY_DIAGNOSTICS.json','FINAL_ACCOUNTING.json','FINAL_RETENTION.json','FINAL_SOURCE_PROVENANCE.json']
    selected.update(root/'evidence'/name for name in required)
    selected.update(root/'evidence'/('MOVIE_'+name+'_OK.json') for name in names)
    selected.update(proof.glob('*.json'))
    selected.update(closed_dir.glob('*.json'))
    rows = [dict(relative=str(p.relative_to(root)), **receipt(p)) for p in sorted(selected)]
    total = sum(r['bytes'] for r in rows)
    assert total < 1024**3, 'Thin review delivery exceeds1GiB'
    local_before = int(run(['du','-s','-B1',BATCH]).split()[0])
    assert not (root/'evidence/RESOURCE_STOP.json').exists()
    amd_bytes = remote['allocated_bytes']['new_batch']
    whole_amd = remote['allocated_bytes']['whole_AMD_user']
    assert whole_amd < policy['whole_AMD_stop_TiB']*1024**4
    assert whole_amd+max(0,policy['platform_GiB']['amd']*1024**3-amd_bytes) < policy['whole_AMD_projected_limit_TiB']*1024**4
    assert shutil.disk_usage(BATCH).free > 2*1024**4
    assert local_before+total < (policy['platform_GiB']['perseus']-policy['checkpoint_reserve_GiB'])*1024**3
    assert amd_bytes < (policy['platform_GiB']['amd']-policy['checkpoint_reserve_GiB'])*1024**3
    manifest = dict(UTC=now(), job=job, source_root=str(root), source_commit=args.source_commit, files=rows, file_count=len(rows), bytes=total,
        scope='Final reports, exact inputs, figures/movies and compact source/acceptance evidence. Excludes raw/reduced arrays, logs, source trees, checkpoints and lossless movie-frame sequences. This manifest does not include itself or the later mirror-verification receipt.')
    write(root/'evidence/REVIEW_PRODUCT_MANIFEST.json', manifest)
    mirror = BATCH/'review_mirror'
    temporary = BATCH/('review_mirror.'+job+'.partial')
    assert not temporary.exists(), 'Preserve an interrupted attempt and use a new allocation'
    temporary.mkdir()
    for row in rows:
        assert not (root/'evidence/RESOURCE_STOP.json').exists()
        source = Path(row['path'])
        assert sha(source) == row['sha256']
        target = temporary/row['relative']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source,target)
        assert target.stat().st_size == row['bytes'] and sha(target) == row['sha256']
    shutil.copy2(root/'evidence/REVIEW_PRODUCT_MANIFEST.json', temporary/'evidence/REVIEW_PRODUCT_MANIFEST.json')
    assert sha(temporary/'evidence/REVIEW_PRODUCT_MANIFEST.json') == sha(root/'evidence/REVIEW_PRODUCT_MANIFEST.json')
    if mirror.exists():
        old = dict(UTC=now(), previous_manifest=receipt(mirror/'evidence/REVIEW_PRODUCT_MANIFEST.json'))
        archive = BATCH/('review_mirror.previous_'+job+'_'+str(time.time_ns()))
        write(BATCH/'evidence'/('MIRROR_REPLACEMENT_'+job+'.json'), dict(**old, archived_root=str(archive)))
        mirror.rename(archive)
    temporary.rename(mirror)
    final_bytes = int(run(['du','-s','-B1',BATCH]).split()[0])
    assert final_bytes+amd_bytes <= policy['combined_review_GiB']*1024**3
    result = dict(UTC=now(), job=job, host=socket.gethostname(), source_root=str(root), mirror_root=str(mirror), one_way=True, Mac_delivery=False,
        files_verified=len(rows), bytes=total, source_product_manifest=receipt(root/'evidence/REVIEW_PRODUCT_MANIFEST.json'), mirror_product_manifest=receipt(mirror/'evidence/REVIEW_PRODUCT_MANIFEST.json'),
        actual_scientific_and_visual_acceptance_verified=True, Perseus_batch_allocated_bytes=final_bytes, AMD_batch_allocated_bytes=amd_bytes,
        combined_allocated_bytes=final_bytes+amd_bytes, allocated_node_hours_upper=upper, current_job_terminal_accounting_pending=True,
        scope='Checksum-verified one-way thin review copy on Perseus. Native/review jobs are quiescent; this closeout allocation remains bounded in accounting until its terminal scheduler receipt is captured. No Mac delivery or exact billing claim.')
    write(BATCH/'evidence/THIN_REVIEW_VERIFIED.json',result)
    write(mirror/'evidence/THIN_REVIEW_VERIFIED.json',result)
    print(json.dumps({k:result[k] for k in ['UTC','job','files_verified','bytes','combined_allocated_bytes','allocated_node_hours_upper','mirror_root','Mac_delivery']},indent=2))


if __name__ == '__main__':
    main()
