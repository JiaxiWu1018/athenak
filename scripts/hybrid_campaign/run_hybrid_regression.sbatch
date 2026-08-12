#!/usr/bin/env bash
#SBATCH --job-name=hyb_regress
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

# Default-sampler bitwise regression for the hybrid build.
#
# Runs the HISTORICAL reduced-cost R/M=6.1 input -- which contains no
# cluster_sampler key -- under (a) the preserved 2026-08-01 sampler executable
# (sha 28e66d4a...) and (b) the new hybrid executable, for a few cycles, and
# additionally runs leg (a) twice to measure the GPU atomic-deposition
# nondeterminism floor.  Pass criteria (same as the 2026-08-01 campaign):
# bitwise-identical t=0 particle dump, and old-vs-new late divergence within
# the old-vs-old floor.
set -euo pipefail

ROOT=/work1/eliasmost/jiaxiwu/nrpic_hybrid_stability_20260802
OLD_EXE=/work1/eliasmost/jiaxiwu/athenak_sampler_20260801/build_sampler_gfx90a_rocm6_4_1/src/athena
NEW_EXE=/work1/eliasmost/jiaxiwu/athenak_hybrid_20260802/build_hybrid_gfx90a_rocm6_4_1/src/athena
INPUT=${ROOT}/inputs/_historical_default_q6p1.athinput
OUT=${ROOT}/runs/_default_regression
REC=${ROOT}/provenance/default_regression_job${SLURM_JOB_ID}

module purge
module load autotools cmake/3.25.2 prun/2.3 rocm/6.4.1 gnu12/12.2.0 openmpi4/4.1.8
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1 UCX_LOG_LEVEL=error
export OMPI_MCA_pml=ucx OMPI_MCA_osc=ucx

rm -rf "${OUT}"; mkdir -p "${OUT}/old" "${OUT}/new" "${OUT}/old2" "${REC}"
overrides=(time/tlim=0.25 time/nlim=8 output1/dt=0.05 output2/dt=0.05
           output3/dt=1000000 output4/dt=1000000 output5/dt=1000000
           output6/dt=1000000 output7/dt=1000000)

for leg in old new old2; do
  exe="${OLD_EXE}"; [[ ${leg} == new ]] && exe="${NEW_EXE}"
  echo "=== ${leg}: ${exe}"
  prun "${exe}" --kokkos-map-device-id-by=mpi_rank \
    -i "${INPUT}" -d "${OUT}/${leg}" "${overrides[@]}" \
    > "${REC}/${leg}.out" 2> "${REC}/${leg}.err" || {
      echo "${leg} leg failed"; tail -30 "${REC}/${leg}.err"; exit 1; }
done

{
  echo "old_exe_sha256=$(sha256sum "${OLD_EXE}" | cut -d' ' -f1)"
  echo "new_exe_sha256=$(sha256sum "${NEW_EXE}" | cut -d' ' -f1)"
  echo "input_sha256=$(sha256sum "${INPUT}" | cut -d' ' -f1)"
} > "${REC}/regression_provenance.txt"

# three-leg comparison: old-vs-new against the old-vs-old nondeterminism floor
python3 - "${OUT}" "${REC}/default_regression_verdict.json" <<'PY'
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, "/work1/eliasmost/jiaxiwu/nrpic_hybrid_stability_20260802/scripts")
from compare_regression_numeric import pair_metrics
out = Path(sys.argv[1])
reg = pair_metrics(out/"old", out/"new")
ctl = pair_metrics(out/"old", out/"old2")
t0_ok, late_ok = False, True
rows = []
for r, c in zip(reg, ctl):
    if r["time"] == 0.0:
        t0_ok = bool(r.get("bitwise_identical"))
    else:
        if r["max_rel_dpos"] > max(10.0*c["max_rel_dpos"], 1e-12):
            late_ok = False
    rows.append({"file": r["file"], "time": r["time"],
                 "old_vs_new": r["max_rel_dpos"], "old_vs_old": c["max_rel_dpos"],
                 "t0_bitwise": r.get("bitwise_identical")})
    print(f"{r['file']:55s} t={r['time']:.4f} new={r['max_rel_dpos']:.3e} floor={c['max_rel_dpos']:.3e}")
verdict = {"t0_realization_bitwise_identical": t0_ok,
           "late_divergence_within_nondeterminism_floor": late_ok,
           "verdict": "PASS" if (t0_ok and late_ok) else "FAIL",
           "comparisons": rows}
Path(sys.argv[2]).write_text(json.dumps(verdict, indent=2))
print("VERDICT:", verdict["verdict"])
PY
grep -q '"verdict": "PASS"' "${REC}/default_regression_verdict.json" \
  && echo REGRESSION_PASS || { echo REGRESSION_FAIL; exit 1; }
