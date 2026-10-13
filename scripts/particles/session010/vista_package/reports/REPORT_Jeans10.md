# Jeans/GI-in-cluster Session010

This is a preparation session. The approved Vista experiment restarts the initial-data generator at t=0 in [-512,512]^3, retaining the Session009 R40 fine wave region, five million particles, boost 0.133215 and central resolution ceiling 1/256. The exact root is 256^3, dx=4, with 4,936 initial MeshBlocks. Only the coarse exterior is reduced: this saves 448 blocks (8.32%) from the actual Session009 5,384-block setup. The fine wave region remains the dominant cost. No checkpoint is transferred.

The particle buffer-reuse repair, capacity-safe VTK output and full-precision persisted restart parameters are implemented. Focused AMR/MPI/accounting/restart/controller checks are recorded in REPORT_AGENT.md. The matched memory measurement and remaining growth are in MEMORY_RESULTS.md. Full-mesh Grace/H200 feasibility and timing still require Vista gates.

Boundary-origin arrival estimates for this box are t≈472/light or 334/gauge at R40, and t≈509/light or 360/gauge at the clumps, assuming far-flat coordinate speeds 1 and sqrt(2), negligible shift and a disturbance at t=0. These are estimates, not rigorous bounds; contamination amplitude is unmeasured. SETUP_COMPARISON.md includes the smaller-domain choices and coarse/fine cost separation.

No new orbital, horizon, merger or ringdown scientific result is claimed. Session009 remains stopped at its independently verified final t=9.33125 checkpoint; its old controller metadata is stale and its stops are preserved. Session007's retained output through t=64 is historical context with documented scheduler/evidence limitations.

The handoff is ready only for allocated-GG plan-mode discussion and subsequent native build/full-size gates. Revised resources, outputs, checkpointing and finite scientific/resource termination limits must be agreed before actual GH production scripts are prepared for Jiaxi to submit. Analysis and rendering should run on Vista, with checksum-verified archival to Anta and purge-aware scheduling. No production launch or Vista transfer is performed here.
