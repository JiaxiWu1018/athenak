"""Fail-closed final audit and thin-product manifest; no automatic visual claims."""
import csv
import shutil
from common import *
from accounting import summarize
from products import verify
from extension_resources import storage, allocated_bytes

FIGURES = ['s09_shells_lapse', 's09_mass_conservation', 's09_AH_masses']


def checked_visual(visual, pdfs, movies):
    assert visual.get('actual_visual_acceptance') is True
    assert visual.get('reviewer') and visual.get('findings')
    assert visual['pdf_sha256'] == pdfs, 'PDFs differ from the actually inspected exports'
    assert visual['movie_sha256'] == movies, 'Movies differ from inspected encoded artifacts'
    for name in movies:
        frames = visual['encoded_frames_actually_inspected'].get(name)
        assert frames and all(isinstance(value, int) and value >= 0 for value in frames)


def scheduler_accounting():
    registry = load(EXT / 'evidence/job_registry.json')
    identifiers = ','.join(row['job_id'] for row in registry['jobs'])
    command = ['sacct', '-X', '-n', '-P', '-j', identifiers, '-o',
               'JobIDRaw,State,ExitCode,ElapsedRaw,AllocNodes,Start,End']
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    account = summarize(registry, result.stdout)
    mirror_reserve = .5
    assert account['allocated_node_hour_upper'] + mirror_reserve <= 40
    queued = subprocess.run(['squeue', '-u', 'jiaxiwu', '-h', '-o', '%i|%T|%D'],
                            capture_output=True, text=True)
    assert queued.returncode == 0, queued.stderr
    known = {row['job_id'] for row in registry['jobs']}
    live = [line for line in queued.stdout.splitlines()
            if line.split('|')[0] in known and line.split('|')[0] != job()]
    assert not live, ('Registered work remains live', live)
    account.update(utc=now(), job=job(), command=command,
                   scheduler_stdout=result.stdout, scheduler_stderr=result.stderr,
                   queue_stdout=queued.stdout, publisher_job=job(),
                   prior_registered_handles_quiescent=True,
                   registry=receipt(EXT / 'evidence/job_registry.json'),
                   Perseus_mirror_verification_reserve_node_hours=mirror_reserve,
                   combined_upper_including_mirror_reserve=account['allocated_node_hour_upper'] + mirror_reserve,
                   publisher_terminal_addendum_required=True)
    atomic(ROOT / 'evidence/FINAL_ACCOUNTING.json', account)
    return account


def main():
    job()
    assert load(ROOT / 'evidence/ASSEMBLY_OK.json')['source_files_SHA256_verified']
    bindings = load(ROOT / 'evidence/REPORT_SOURCE_BINDINGS.json')
    assert [row['case'] for row in bindings['outcomes']] == [c['id'] for c in PLOT_CASES]
    numerical, retention = [], []
    for raw in PLOT_CASES:
        key = raw['id']
        reused = raw.get('reused_session8', False)
        packet = EXT / 'imported' / key if reused else EXT / key / 'review_packet'
        manifest = verify(packet)
        accept = load(packet / 'numerical_acceptance.json')
        assert manifest['case']['yc'] == raw['yc'] and manifest['case']['U'] == -.005
        assert accept['numerical_checks_passed']
        assert manifest['checkpoint']['sha256'] == accept['checkpoint']['sha256']
        numerical.append(dict(case=key, checkpoint=manifest['checkpoint']['sha256'],
                              source_acceptance=receipt(packet / 'numerical_acceptance.json'),
                              source_packet=receipt(packet / 'MANIFEST.json'),
                              original_source_root=manifest['source_root'], reused_session8=reused))
        destination = ROOT / 'evidence/cases' / (key + '.json')
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(packet / 'numerical_acceptance.json', destination)
        if not reused:
            root = EXT / key
            state = load(root / 'runs' / case_config(key)['id'] / 'state.json')
            assert state['complete']
            assert state['latest']['sha256'] == manifest['checkpoint']['sha256']
            closed_path = root / 'evidence/CLOSED_OUTPUT_MANIFEST.json'
            closed = load(closed_path)
            assert closed['stable_files_verified'] and closed['checkpoint']['sha256'] == state['latest']['sha256']
            for record in closed['files']:
                info = Path(record['path']).stat()
                assert info.st_size == record['bytes'] and info.st_mtime_ns == record['mtime_ns'], record['path']
            initial = load(root / 'evidence/INITIAL_CONSTRAINT_ADMISSION.json')
            actual = load(root / 'runs' / case_config(key)['id'] / 'INITIAL_OK.json')
            assert max(initial['constraint_norm_ratios'].values()) <= 1.005
            assert actual['actual_mesh_checked'] and actual['initial_hierarchy_unchanged']
            assert actual['finest_allowed_dx_M'] == raw['finest_dx']
            identity = load(root / 'evidence/PRODUCTION_ADMISSION.json')
            assert identity['zero_steps'] and identity['interior_grid_max_scaled_error'] < 1e-12
            retention.append(dict(case=key, raw_root=str(root), allocated_bytes=allocated_bytes(root),
                                  closed_manifest=receipt(closed_path), file_count=closed['file_count'],
                                  selected_logical_bytes=closed['logical_bytes'],
                                  policy='Retained on AMD. Only inventoried, replacement-verified Session9 recovery rotations; earlier/Jeans untouched.'))
    reference = load(EXT / 'evidence/REFERENCE_PACKET_OK.json')
    assert reference['independent_full_decode_frames'] == 204 and reference['no_new_evolution']
    movie_records, movie_hashes = [], {}
    for key in [c['id'] for c in ALL_CASES] + ['s09_comparison']:
        record = load(ROOT / 'evidence' / ('MOVIE_' + key + '_OK.json'))
        assert record['full_decode'] and record['frames'] > 0
        assert abs(record['duration'] - record['frames'] / 15) < .15
        movie = ROOT / 'movies' / (key + '.mp4')
        assert sha(movie) == record['sha256']
        assert record['decoded_previews']
        movie_records.append(record)
        movie_hashes[key] = record['sha256']
    reference_movie = ROOT / 'movies/yc0500_u0005.mp4'
    assert sha(reference_movie) == reference['reused_movie']['sha256']
    pdf_hashes = {name + '.pdf': sha(ROOT / 'plots' / (name + '.pdf')) for name in FIGURES}
    visual = load(ROOT / 'evidence/VISUAL_ACCEPTANCE.json')
    checked_visual(visual, pdf_hashes, movie_hashes)
    for record in movie_records:
        mode = Path(record['path']).stem
        inspected = set(visual['encoded_frames_actually_inspected'][mode])
        required = {row['frame'] for row in record['decoded_previews']}
        assert required <= inspected, ('Uninspected exported encoded previews', mode, required - inspected)
        assert all(frame < record['frames'] for frame in inspected)
    account = scheduler_accounting()
    disk = storage()
    assert disk['own'] < 400 * GIB and disk['whole'] < 1.7 * TIB
    atomic(ROOT / 'evidence/RAW_RETENTION_SUMMARY.json',
           dict(utc=now(), job=job(), cases=retention, storage=disk,
                reference_original_source_root=reference['original_source_root'],
                reference_original_raw_retained_on='Perseus',
                new_raw_reduced_logs_code_frames_final_products_retained_on='AMD',
                no_closeout_deletion=True))
    account_text = '# Accounting and retention\n\n'
    account_text += (f"Conservative allocated node-hour upper: **{account['allocated_node_hour_upper']:.8f}** "
                     'against 40, including preparation, failures, production and review. '
                     'Terminal scheduler rows are exact; unresolved records retain requested '
                     'allocation bounds. This is not an invoice. The publisher is included '
                     'with its allocation bound until its post-terminal addendum. A separate '
                     '0.5 node-hour reserve covers the allocated Perseus destination verification.\n\n')
    account_text += f"Current session storage: **{disk['own'] / GIB:.6f} GiB** against 400.\n\n"
    account_text += '| Case | Retained AMD raw root | Allocated bytes |\n|---|---|---:|\n'
    for row in retention:
        account_text += f"| {row['case']} | `{row['raw_root']}` | {row['allocated_bytes']} |\n"
    account_text += ('\nOriginal 0.500 reference raw data remain on Perseus; the verified compact '
                     'review packet and reused movie are on AMD. All new raw data, reductions, '
                     'logs, source, checkpoints, lossless frames and final products remain on '
                     'AMD. Superseded session 9 recovery rotations have separate inventories. '
                     'No closeout deletion; earlier sessions and Jeans remain preserved. '
                     'The one-way Perseus mirror contains review artifacts only.\n')
    (ROOT / 'reports/ACCOUNTING_AND_RETENTION.md').write_text(account_text)
    with (ROOT / 'reports/ACCOUNTING.csv').open('w') as stream:
        columns = ['job_id', 'kind', 'partition', 'status', 'upper_node_hours',
                   'exact_node_hours', 'basis']
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in columns} for row in account['jobs']])
    for filename in ['NATIVE_BUILD_PROVENANCE.json', 'REFERENCE_PACKET_OK.json',
                     'INDEPENDENT_CONSTRAINT_REFERENCE_OK.json', 'SOURCE_RECEIPT.json']:
        shutil.copy2(EXT / 'evidence' / filename, ROOT / 'evidence' / filename)
    products = [ROOT / 'README.md']
    products += sorted((ROOT / 'reports').glob('*.md')) + sorted((ROOT / 'reports').glob('*.csv'))
    products += sorted((ROOT / 'inputs').glob('*.athinput'))
    products += [ROOT / 'plots' / (name + '.' + ext) for name in FIGURES for ext in ['pdf', 'png']]
    products += sorted((ROOT / 'movies').glob('*.mp4'))
    products += sorted((ROOT / 'evidence/cases').glob('*.json'))
    products += [ROOT / 'evidence' / name for name in
                 ['FINAL_ACCOUNTING.json', 'RAW_RETENTION_SUMMARY.json', 'VISUAL_ACCEPTANCE.json',
                  'ASSEMBLY_OK.json', 'REPORT_SOURCE_BINDINGS.json', 'NATIVE_BUILD_PROVENANCE.json',
                  'REFERENCE_PACKET_OK.json', 'INDEPENDENT_CONSTRAINT_REFERENCE_OK.json', 'SOURCE_RECEIPT.json']]
    entries = [dict(relative=str(path.relative_to(ROOT)), **receipt(path)) for path in products]
    manifest = dict(utc=now(), job=job(), files=entries, file_count=len(entries),
                    logical_bytes=sum(row['bytes'] for row in entries),
                    scope='Thin review products only; no raw outputs, reduced arrays, source, logs or lossless frames.')
    closing_reserve = 16 << 20
    assert disk['own'] + manifest['logical_bytes'] + closing_reserve < 400 * GIB
    atomic(ROOT / 'evidence/REVIEW_PRODUCT_MANIFEST.json', manifest)
    (ROOT / 'THIN_FILES.txt').write_text('\n'.join(row['relative'] for row in entries) +
                                      '\nevidence/REVIEW_PRODUCT_MANIFEST.json\nevidence/FINAL_ACCEPTANCE.json\n')
    atomic(ROOT / 'evidence/FINAL_ACCEPTANCE.json',
           dict(utc=now(), job=job(), numerical_cases=numerical,
                all_five_numerical_acceptance=True, actual_PDF_and_encoded_frame_inspection=True,
                new_movie_full_decode_count=5, reused_movie_decode_frames=204,
                visual_acceptance=receipt(ROOT / 'evidence/VISUAL_ACCEPTANCE.json'),
                product_manifest=receipt(ROOT / 'evidence/REVIEW_PRODUCT_MANIFEST.json'),
                resource_limits_passed=True, allocated_node_hour_upper=account['allocated_node_hour_upper'],
                storage_GiB=disk['own'] / GIB, previous_sessions_preserved=True,
                thin_copy_reserved_bytes=manifest['logical_bytes'], closing_reserve_bytes=closing_reserve,
                destination_copy_verified=False, publisher_terminal_addendum_required=True,
                complete=False, remaining=['Checksum-verified one-way Perseus copy', 'Publisher post-terminal accounting addendum']))
    print(json.dumps(dict(products=manifest['file_count'], bytes=manifest['logical_bytes'],
                          numerical_and_visual_checks_passed=True, complete=False), indent=2))


if __name__ == '__main__':
    main()
