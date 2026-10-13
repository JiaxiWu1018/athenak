# Reusable templates, not configured production jobs

These derive from the actual repaired Session007 Vista package on Anta. No Session007 production configuration is copied. Session010 DECISIONS.json is inert.

The Vista agent must discuss resources/outputs/checkpointing/finite termination first, then adapt these templates to the final source/input and measured full4936-block allocation. The old four-node capacity ladder uses a compact fixture; it cannot initialize4936blocks at400/rank. The full check needs a GH allocation with enough ranks (16 is a candidate), not gh-dev's eight-node ceiling. Replace old quick-test canonical/control/orbit assumptions; preserve current.133215 boost and R40 only.

segment_finalize.py now verifies native checkpoint layout/hash and uses its physical time/cycle; tracker time is separate coverage. workflow.py counts clean terminal segments and records actual Slurm allocation accounting. All sbatch calls carry the verified user email. Production submissions require session010_review_complete=true plus the existing authorization gate. No preparation submission is supplied or performed by the Perseus agent.

`vista_build.sbatch` and `vista_segment.sbatch` are reusable after a reviewed site configuration is written. `vista_quicktest.sbatch`, `vista_resource_probe.sbatch`, their summarizers and historical unboosted/control assumptions are reference templates requiring adaptation, not accepted Session010 gates. Their named old fixtures/configuration are deliberately absent. Use only the approved input as the physical starting point and derive validation overrides transparently; do not restore the old fixtures or their boost by accident. The standalone `native_build.sh` and `audit_relocation.sh` use PROVENANCE.json directly.
