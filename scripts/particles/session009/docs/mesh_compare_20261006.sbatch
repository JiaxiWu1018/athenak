#!/bin/bash
#SBATCH -J jeans9-radius-mesh
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -t 00:05:00
#SBATCH -o /data/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/evidence/radius_assessment_20261006/slurm.%j.log
set -euo pipefail
ASSESS_ROOT=/data/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/evidence/radius_assessment_20261006
ASSESS_EXE=/data/jiaxiwu/NRPIC/GI_in_cluster/session_007_part1_vista_handoff_20260908/evidence/perseus_validation/build_cuda_6511/src/athena
export PATH=/home/emost/codes/openmpi/bin:/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/home/emost/codes/openmpi/lib:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export OMPI_MCA_opal_cuda_support=true OMPI_MCA_pml=ucx UCX_TLS=self,sm,cuda_copy,cuda_ipc UCX_MEMTYPE_CACHE=n
{ date -u; hostname; sha256sum "$ASSESS_EXE" "$ASSESS_ROOT/"*.athinput; } > "$ASSESS_ROOT/provenance.txt"
for name in r50_buffer56 r40_floor40 r30_floor30 r40_buffer46 r30_buffer36; do
 mkdir -p "$ASSESS_ROOT/$name"
 (cd "$ASSESS_ROOT/$name"; mpirun -np 1 "$ASSESS_EXE" --kokkos-map-device-id-by=mpi_rank -i "$ASSESS_ROOT/$name.athinput" -m mesh_refinement/max_nmb_per_rank=10000 > mesh.log 2>&1)
done
python3 - <<'ENDPY'
from pathlib import Path
import re,json
root=Path('/data/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005/evidence/radius_assessment_20261006')
rows=[]
for name,radius,extent in [('r50_buffer56',50,56),('r40_floor40',40,40),('r30_floor30',30,30),('r40_buffer46',40,46),('r30_buffer36',30,36)]:
 text=(root/name/'mesh.log').read_text()
 blocks=int(re.search(r'Total number of MeshBlocks = (\d+)',text)[1])
 levels={int(a):int(b) for a,b in re.findall(r'Physical level = (\d+).*?: (\d+) MeshBlocks',text)}
 rows.append(dict(case=name,radius=radius,fine_extent=extent,initial_blocks=blocks,initial_cells=blocks*32**3,levels=levels))
assert rows[0]['initial_blocks']==7456
for row in rows:
 row['saved_blocks']=rows[0]['initial_blocks']-row['initial_blocks']
 row['saved_fraction']=row['saved_blocks']/rows[0]['initial_blocks']
(root/'counts.json').write_text(json.dumps(dict(mesh_only=True,no_particles_initialized=True,production_configuration_unchanged=True,initial_tree_source_identical_to_session9=True,rows=rows),indent=2)+'\n')
print(json.dumps(rows,indent=2))
ENDPY
