"""Freeze output provenance and draft the source-backed session report."""
import subprocess
import socket
from session import *

def main():
    require_job()
    load(ROOT / 'evidence/QA_FINAL_OK.json')
    metrics = load(ROOT / 'evidence/DESCRIPTIVE_METRICS.json')
    production = load(ROOT / 'runs/production/completion.json')
    files = []
    for role in ('smoke', 'production'):
        for path in sorted((ROOT / 'runs' / role).rglob('*')):
            if path.is_file():
                files.append({'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    local_raw_bytes = sum(row['bytes'] for row in files if row['path'].startswith('runs/production/out/'))
    amd_root = '/work1/eliasmost/jiaxiwu/st_reconstruction_s06_20261016'
    on_perseus = str(ROOT).startswith('/data/')
    if on_perseus:
        inventory = load(ROOT / 'evidence/AMD_SOURCE_INVENTORY.json')
        raw_bytes = sum(row['bytes'] for row in inventory['files'] if row['path'].startswith('runs/production/out/'))
        user_bytes = int(subprocess.check_output(['ssh', 'hpcfund.amd.com', 'du', '-s', '-B1', '/work1/eliasmost/jiaxiwu'], text=True).split()[0])
        amd_session_bytes = int(subprocess.check_output(['ssh', 'hpcfund.amd.com', 'du', '-s', '-B1', amd_root], text=True).split()[0])
    else:
        raw_bytes = local_raw_bytes
        user_bytes = int(subprocess.check_output(['du', '-s', '-B1', '/work1/eliasmost/jiaxiwu'], text=True).split()[0])
        amd_session_bytes = int(subprocess.check_output(['du', '-s', '-B1', str(ROOT)], text=True).split()[0])
    session_bytes = int(subprocess.check_output(['du', '-s', '-B1', str(ROOT)], text=True).split()[0])
    atomic(ROOT / 'evidence/OUTPUT_MANIFEST.json', {'utc': now(), 'host': socket.gethostname(), 'files': files,
           'local_production_raw_logical_bytes': local_raw_bytes, 'production_raw_logical_bytes': raw_bytes,
           'amd_source_inventory': 'AMD_SOURCE_INVENTORY.json' if on_perseus else None})
    atomic(ROOT / 'evidence/STORAGE_FINAL.json', {'utc': now(), 'root': str(ROOT), 'session_allocated_bytes': session_bytes,
           'amd_root': amd_root, 'amd_session_allocated_bytes': amd_session_bytes,
           'whole_user_allocated_bytes': user_bytes, 'raw_status': 'Retained on AMD; no deletion authorized',
           'warning_TiB': 1.5, 'stop_TiB': 1.7, 'budget_TiB': 1.9})
    reports = ROOT / 'reports'; reports.mkdir(exist_ok=True)
    setup = load(ROOT / 'evidence/SETUP.json')
    lines = ['# ST reconstruction session 6: five surface periods', '',
             '**Status: evolution and automated analysis complete; visual/scientific review pending.**', '',
             '## Setup and execution', '',
             f'Continued proper-profile stratified seed 1 from 300M (cycle 1501) to {metrics["final_time_M"]:g}M. '
             'The 1,500,000 particles, live Z4c, gauge, GR-Boris/CIC, removal settings, fixed mesh and outputs are unchanged.', '',
             'Domain ±128M;128³ root;32³ MeshBlocks;dx=1M within ±64M and0.5M within ±32M. '
             'Protected compiled source5a5db71c;Kokkos6739bc62;ROCm6.4.1;one exclusive AMD mi2104x node/four MPI ranks/four MI210s.', '',
             f'Archived areal radius R/M=9.96420711980792;reference circular surface period at infinity '
             f'P/M=2π(R/M)^1.5={P:.9f}. Five periods=988.129789M;target1000M={END/P:.6f} periods. '
             'This reference period is not an assumed shell oscillation period.', '',
             f'Smoke300→306M job{load(ROOT/"runs/smoke/launch.json")["job"]};'
             f'production300→1000M job{production["job"]};analysis/QA job{os.environ["SLURM_JOB_ID"]} on{socket.gethostname()}. '
             f'Production solver+checkpoint-verification wall time{production["solver_wall_seconds"]:.1f}s. '
             f'Boris forward-Euler fallback events{production["boris_fallbacks"]};distinct cycles{production["boris_fallback_cycles"]}.', '',
             'Exact hashes, commands and hosts are in inputs/,evidence/SETUP.json,runs/*/launch.json and evidence/jobs/. '
             'The tiny saved terminal dt was retained; native timestep recovery was verified by the smoke. '
             'Interior302/304M per-ID smoke/production comparisons and independent shell/window/3D-constraint checks passed.', '',
             '## Descriptive shell response', '',
             'All radii are rest-mass quantiles about the moving COM, normalized by the original t=0 radius. '
             'Raw half-range includes drift; detrending can remove part of a slow oscillation. '
             'Period-window comparisons use P–2P versus4P–5P and exclude incomplete tails. '
             'The full100M/P/2P window tables include all nine shells.', '',
             '|Shell|Early raw excursion %|Late raw excursion %|Late/early raw|Late/early detrended|Mean shift change (percentage points)|Linear drift %/P, P–5P|',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in metrics['shell_comparisons']:
        if row['quantile'] in (10, 50, 90):
            lines.append(f'|r{row["quantile"]}|{100*row["early_raw_fraction"]:.4f}|{100*row["late_raw_fraction"]:.4f}|'
                         f'{row["late_over_early_raw"]:.3f}|{row["late_over_early_detrended"]:.3f}|'
                         f'{100*row["mean_shift_change_fraction"]:+.4f}|{100*row["linear_slope_fraction_per_P"]:+.4f}|')
    lines += ['', 'These are descriptive measures from one realization;time samples,windows and shells are not independent trials. '
              'No preset scientific pass/fail threshold is applied. The user reviews the gate for any later paper rerun.', '',
              '## COM, lapse, density and numerical health', '',
              f'COM magnitude:initial{metrics["com_initial_M"]:.6g}M;final{metrics["com_final_M"]:.6g}M;maximum{metrics["com_max_M"]:.6g}M.', '',
              '|Diagnostic|Early mean, P–2P|Late mean,4P–5P|', '|---|---:|---:|']
    for key, value in metrics['health'].items():
        lines.append(f'|{key}|{value["early"]["mean"]:.8g}|{value["late"]["mean"]:.8g}|')
    lines += ['', 'Core lapse is the existing fixed-origin r<0.75M volume mean. Larger-region E is labelled separately. '
              'Central slice E (new302–1000M slices only) and COM-centred coordinate rest-mass density inside0.5M are distinct proxies. '
              'The old slices were discarded;no historical central-E values are invented. '
              'Core constraints use coordinate volume;global histories retain proper volume. '
              'Completed steps≤1e-6M are masked except genuine t=0;endpoint masking remains explicit in reduced/health.csv.', '',
              '## Figures and interpretation limits', '',
              '![Shells, COM, lapse, density and constraints](../plots/f1_long_baseline_histories.png)', '',
              '[Vector PDF](../plots/f1_long_baseline_histories.pdf)', '',
              '![Window excursions and mean drift](../plots/f2_shell_amplitude_and_drift.png)', '',
              '[Vector PDF](../plots/f2_shell_amplitude_and_drift.pdf)', '',
              'The fixed boundary and gauge can affect a long baseline. No boundary/mesh/particle convergence control is added here. '
              'Coordinate-shell trends alone cannot establish a physical instability;amplitude changes,mean drift,COM,lapse,density and constraints '
              'must be read together. Beating,phase mixing and incomplete cycles can affect window excursions. '
              'The five-case paper rerun was not launched.', '',
              '## Retention and reproduction', '',
              f'Complete raw production output is retained at `{amd_root}/runs/production/out`:**{raw_bytes:,} logical bytes**. '
              f'AMD session:{amd_session_bytes:,} allocated bytes;whole AMD user root:{user_bytes:,} allocated bytes at the recorded measurement. '
              f'Analysis host session `{ROOT}`:{session_bytes:,} allocated bytes,including{local_raw_bytes:,} local production raw logical bytes. '
              'Nothing was deleted. OUTPUT_MANIFEST.json retains exact per-file hashes. '
              'Perseus mirror/visual review/accounting completion will be appended after verification.', '',
              'Run stage_perseus.sbatch in a Slurm allocation,then run.sbatch smoke;inspect SMOKE_OK.json;run.sbatch production;'
              'analyze.sbatch after successful production. Use a fresh root and record new provenance. '
              'Scripts are versioned on experiment/ST-reconstruction-s06-long-stability;protected executable hash is in SETUP.json.', '']
    if on_perseus:
        lines += ['The original AMD analysis466317 was cancelled before allocation after long queue estimates '
                  '(~4h for1h request;~2h after reducing request to30min). '
                  'Allocated Perseus analyze_perseus.sbatch transferred native diagnostics,ran the same reductions/QA/plots '
                  'and retained all AMD originals. It requested1GPU/2CPU/16GB by site policy and used CPU analysis only. '
                  'PERSEUS_ANALYSIS_STAGE.json records copied-file sizes/hashes and exact source-checkpoint verification;'
                  'AMD_SOURCE_INVENTORY.json records the full retained AMD source inventory. '
                  'No physics or analysis-definition change accompanied the analysis-host change.', '']
    (reports / 'REPORT_session6.md').write_text('\n'.join(lines))
    delivery = [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                for directory in ('reduced', 'plots', 'reports') for p in sorted((ROOT / directory).glob('*')) if p.is_file()]
    atomic(ROOT / 'evidence/DELIVERY_MANIFEST.json', {'utc': now(), 'files': delivery})
    atomic(ROOT / 'evidence/ANALYSIS_COMPLETE.json', {'utc': now(), 'job': os.environ['SLURM_JOB_ID'],
           'status': 'Automated analysis complete; human visual/scientific review and checksum mirror still required'})
    print('Final evidence and draft report written', flush=True)

if __name__ == '__main__':
    main()
