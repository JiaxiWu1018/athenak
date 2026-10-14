#!/bin/bash
#SBATCH -J st_s05_report
#SBATCH -N 1
#SBATCH -p mi2104x
#SBATCH -A eliasmost
#SBATCH -n 4
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --exclude=k003-002,k003-009,k003-010
#SBATCH -t 02:00:00
#SBATCH -o /work1/eliasmost/jiaxiwu/st_reconstruction_s05_20261014/logs/%x.%j.log
source /work1/eliasmost/jiaxiwu/st_reconstruction_s05_20261014/scripts/amd_common.sh
python3 "$ROOT/scripts/plot_report.py" --interim
python3 "$ROOT/scripts/plot_report.py"
du -s -B1 "$ROOT" /work1/eliasmost/jiaxiwu > "$ROOT/evidence/FINAL_STORAGE.txt"
sacct -j "$SLURM_JOB_ID" -n -P --format=JobIDRaw,State,ElapsedRaw,AllocNodes > "$ROOT/evidence/report_accounting.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$ROOT/evidence/REPORT_READY"
