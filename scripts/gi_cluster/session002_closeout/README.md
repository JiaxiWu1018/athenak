# GI Session-002 closeout analysis

These scripts reproduce the final reductions, figures, and combined movie for
the GI-in-cluster Session-002 single-clump campaign.

- analyze_ah.py reconstructs FastFlow shape coefficients and applies the
  independent horizon gates.
- reduce_ah_evolution.py writes complete gate-audit and validated-surface
  tables for g2_L64 and g2_L128.
- reduce_particle_occupancy.py assigns each particle to its active AMR leaf
  cell and performs an independent geometric spot check.
- plot_closeout_figures.py builds the two final PNG/PDF summary figures from
  the reduced tables.
- make_closeout_movie.py renders the fixed-window density/constraint movie
  with AMR boundaries, tracker centre, and validated shape contours. Wide
  views may select a display-appropriate projection level and use
  --batch-size to restart worker pools before Matplotlib memory accumulates.

The scripts are campaign-specific and require the preserved Session-002 file
layout. Full-data reduction and rendering must run through Slurm on a compute
node, not on a login node. The authoritative decks, batch scripts, input
hashes, outputs, and closeout QA remain in the external campaign archive at
/data/jiaxiwu/NRPIC/GI_in_cluster/session_002_amr_ah_validation_20260823.
