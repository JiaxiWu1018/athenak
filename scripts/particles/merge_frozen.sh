#!/bin/bash
# Join the two frozen-metric control segments into one pvtk directory for a single
# reduction pass, so the null is measured on exactly the same footing as the live record.
#
#   runs/pf_frozen0 : indices 00000-00100, t = 0      .. 1192.5 M  (job 408536, fresh)
#   runs/pf_frozen  : indices 00101-00300, t = 1204.5 .. 3577.5 M  (job 408522 chain,
#                     restarted from the 1 P checkpoint, output counter continued)
#
# Verified contiguous: no index appears in both, and 1192.5 -> 1204.5 is one output
# interval at the P/100 cadence. Symlinks, so nothing is copied or moved.
set -euo pipefail
R=/work1/eliasmost/jiaxiwu/plummer_s01_20260906
D=$R/runs/pf_frozen_full/pvtk
rm -rf "$R/runs/pf_frozen_full"
mkdir -p "$D"
n=0
for f in $R/runs/pf_frozen0/out/pvtk/*.part.vtk $R/runs/pf_frozen/out/pvtk/*.part.vtk; do
  b=$(basename "$f")
  [ -e "$D/$b" ] && { echo "FATAL: duplicate index $b"; exit 1; }
  ln -s "$f" "$D/$b"; n=$((n+1))
done
echo "linked $n frames into $D"
ls -1 "$D" | grep -oE '[0-9]{5}' | sort -n | awk 'NR==1{f=$1} {l=$1} END{print "  index range "f" .. "l"  ("NR" frames)"}'
# fail loudly on a gap rather than silently reducing a holey record
ls -1 "$D" | grep -oE '[0-9]{5}' | sort -n | awk '
  NR>1 && $1 != prev+1 {print "  GAP between "prev" and "$1; bad=1} {prev=$1}
  END{ if(bad) exit 1; print "  contiguous, no gaps" }'
