#!/usr/bin/env bash
#SBATCH --job-name=hyb_case
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --requeue

# One hybrid-campaign case.  Same proven launch pattern as the 2026-08-01
# sampler-causality campaign; only the repository (worktree branch
# sampler-hybrid), executable, and campaign root differ.  Restart-capable:
# if a matching rst exists in the run dir the newest one is used, so a
# requeued or resubmitted job continues instead of restarting from t=0.
set -euo pipefail

: "${CASE:?submit with CASE=<basename>}"
: "${ROOT:=/work1/eliasmost/jiaxiwu/nrpic_hybrid_stability_20260802}"
: "${REPO:=/work1/eliasmost/jiaxiwu/athenak_hybrid_20260802}"
: "${WALL:=11:30:00}"

exe="${REPO}/build_hybrid_gfx90a_rocm6_4_1/src/athena"
input="${ROOT}/inputs/${CASE}.athinput"
run="${ROOT}/runs/${CASE}"
record="${ROOT}/provenance/${CASE}_job${SLURM_JOB_ID}"

mkdir -p "${run}" "${record}"
exec 9>"${run}/.run.lock"
if ! flock -n 9; then
  echo "another job already holds ${run}/.run.lock"; exit 0
fi
if [[ -e "${run}/COMPLETED" ]]; then
  echo "${CASE} already completed"; exit 0
fi

module purge
module load autotools cmake/3.25.2 prun/2.3 rocm/6.4.1 gnu12/12.2.0 openmpi4/4.1.8
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1
export UCX_LOG_LEVEL=error
export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx
# 2026-08-03: all six 512-block runs died simultaneously at ~4h50m in
# uct_rocm_ipc_cache_map_memhandle ("failed to open ipc mem handle",
# rocm_ipc_cache.c:144) inside MPI_Waitall -- the known stale-handle bug of
# UCX's intra-node ROCm IPC transport, first triggered at this mesh size
# (128 blocks/rank; the 344-block campaigns never reached the threshold).
# Disable that transport: intra-node GPU traffic goes host-staged instead.
export UCX_TLS=^rocm_ipc

{
  echo "case=${CASE}"
  echo "slurm_job=${SLURM_JOB_ID}"
  echo "partition=${SLURM_JOB_PARTITION:-unknown}"
  echo "host=$(hostname)"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo=${REPO}"
  echo "branch=$(cd "${REPO}" && git branch --show-current)"
  echo "source_commit=$(cd "${REPO}" && git rev-parse HEAD)"
  echo "source_uncommitted_files=$(cd "${REPO}" && git status --porcelain | wc -l)"
  echo "pgen_sha256=$(sha256sum "${REPO}/src/pgen/nr_pic_homogeneous_cluster.cpp" | cut -d' ' -f1)"
  echo "exe_sha256=$(sha256sum "${exe}" | cut -d' ' -f1)"
  echo "input_sha256=$(sha256sum "${input}" | cut -d' ' -f1)"
  echo "sampler=$(awk -F= '/cluster_sampler/{gsub(/[ \t]/,"",$2); print $2}' "${input}")"
  echo "seed=$(awk -F= '/cluster_seed/{gsub(/[ \t]/,"",$2); print $2}' "${input}")"
  echo "radius_over_mass=$(awk -F= '/cluster_radius_over_mass/{gsub(/[ \t]/,"",$2); print $2}' "${input}")"
  echo "tlim=$(awk -F= '/^tlim/{gsub(/[ \t]/,"",$2); print $2}' "${input}")"
} > "${record}/provenance.txt"
cp "${input}" "${record}/input.athinput"
cat "${record}/provenance.txt"
{ module list; } > "${record}/environment.txt" 2>&1

# Continue from the newest valid restart if one exists (crash/requeue safety).
latest_rst=$(ls -t "${run}"/rst/*.rst 2>/dev/null | head -1 || true)
launch=(prun "${exe}" --kokkos-map-device-id-by=mpi_rank -t "${WALL}")
if [[ -n "${latest_rst}" ]]; then
  echo "restarting_from=${latest_rst}" >> "${record}/provenance.txt"
  launch+=(-r "${latest_rst}")
fi
launch+=(-i "${input}" -d "${run}")
printf '%q ' "${launch[@]}" > "${record}/command.txt"; printf '\n' >> "${record}/command.txt"

cd "${run}"
start=$(date +%s)
set +e
"${launch[@]}" > "${record}/solver.out" 2> "${record}/solver.err"
status=$?
set -e
end=$(date +%s)
{
  echo "exit_status=${status}"
  echo "elapsed_s=$((end-start))"
} >> "${record}/provenance.txt"
grep -A 26 "Homogeneous tangential-orbit cluster initialized" \
  "${record}/solver.out" > "${record}/initialization_banner.txt" 2>/dev/null || true
tail -25 "${record}/solver.out" || true

# Protect integer-period (and half-period) restarts; prune the rolling chain
# down to the newest two so 8P runs do not accumulate ~140 GB of rst files.
period=$(awk -F= '/^# *period/{print $2}' "${input}" 2>/dev/null || true)
if [[ -x "${ROOT}/scripts/protect_restarts.py" ]]; then
  python3 "${ROOT}/scripts/protect_restarts.py" --run "${run}" --case "${CASE}" \
    --root "${ROOT}" >> "${record}/restart_protection.log" 2>&1 || true
fi

# Completion is reaching tlim, not merely exiting 0 (a wall-time stop exits 0
# after writing a final restart; the requeued job then continues).
final_time=$(grep -oE "^time=[0-9.e+-]+" "${record}/solver.out" | tail -1 | cut -d= -f2 || true)
tlim=$(awk -F= '/^tlim/{gsub(/[ \t]/,"",$2); print $2}' "${input}")
done_flag=$(python3 -c "import sys; t=float(sys.argv[1] or 0); L=float(sys.argv[2]); print('yes' if t >= L-1e-9 else 'no')" "${final_time:-0}" "${tlim}")
if [[ ${status} -eq 0 && ${done_flag} == yes ]]; then
  touch "${run}/COMPLETED"
  echo "COMPLETED ${CASE} in $((end-start))s"
elif [[ ${status} -eq 0 ]]; then
  echo "SEGMENT_DONE ${CASE} t=${final_time}/${tlim}; resubmitting"
  sbatch --partition="${SLURM_JOB_PARTITION}" \
    --output="${ROOT}/logs/${CASE}_%j.out" --error="${ROOT}/logs/${CASE}_%j.err" \
    --export="ALL,CASE=${CASE},ROOT=${ROOT},REPO=${REPO},WALL=${WALL}" \
    "${ROOT}/scripts/run_hybrid_case.sbatch" >> "${record}/continuation.txt"
else
  touch "${run}/FAILED"
  echo "FAILED ${CASE} status=${status}"
  tail -60 "${record}/solver.err" || true
fi
exit ${status}
