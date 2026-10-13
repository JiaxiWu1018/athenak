#!/bin/bash
#SBATCH -J jeans10-kokkos
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH --mail-user=jiaxiwu@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail
SESSION=/data/jiaxiwu/NRPIC/GI_in_cluster/session_010_vista_preparation_20261013
REPO=$SESSION/code/kokkos
OUT=$SESSION/evidence/kokkos_history_${SLURM_JOB_ID}
mkdir "$OUT"
scontrol show job "$SLURM_JOB_ID" > "$OUT/slurm.txt"
git -C "$REPO" rev-parse HEAD HEAD^{tree} > "$OUT/before.txt"
git -C "$REPO" fetch --unshallow --no-tags origin 6739bc623081648af9e752b616d9671527922cbf > "$OUT/fetch.log" 2>&1
git -C "$REPO" rev-parse HEAD HEAD^{tree} > "$OUT/after.txt"
cmp "$OUT/before.txt" "$OUT/after.txt"
test "$(git -C "$REPO" rev-parse --is-shallow-repository)" = false
git -C "$REPO" bundle create "$OUT/kokkos.bundle" HEAD
mkdir "$OUT/verification.git"
git -C "$OUT/verification.git" init --bare
git -C "$OUT/verification.git" bundle verify "$OUT/kokkos.bundle" > "$OUT/verify.log" 2>&1
printf '{"passed":true,"job_id":%s,"commit":"6739bc623081648af9e752b616d9671527922cbf"}\n' "$SLURM_JOB_ID" > "$SESSION/evidence/kokkos_history_receipt.json"
