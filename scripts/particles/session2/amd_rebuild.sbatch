#!/bin/bash
# Incremental ROCm-6.4.1 rebuild: no rm -rf, so only the changed translation units and
# the link step run.  Use amd_build.sbatch for a clean-slate provenance build.
#SBATCH -J plm2_rebuild
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -o /work1/eliasmost/jiaxiwu/plummer_s02_20260910/logs/plm2_rebuild.%j.log
set -euo pipefail
ROOT=/work1/eliasmost/jiaxiwu/plummer_s02_20260910
repo=$ROOT/src/athenak
build=$ROOT/build_rocm641
hostname; date -Is
module purge
module load gnu12/12.2.0 openmpi4/4.1.8 cmake/3.25.2 prun/2.3 rocm/6.4.1
git -C "$repo" fetch -q origin
git -C "$repo" reset -q --hard origin/project/Plummer-cluster
git -C "$repo" submodule update --init --recursive -q
git -C "$repo" log --oneline -1
git -C "$repo" status --porcelain | head
cmake --build "$build" -j 16
sha256sum "$build/src/athena" | tee "$build/exe.sha256"
sha256sum "$repo/src/pgen/particles/nr_pic_plummer.cpp" | tee "$build/pgen.sha256"
git -C "$repo" rev-parse HEAD | tee "$build/git-head.txt"
date -Is
echo "AMD REBUILD OK"
