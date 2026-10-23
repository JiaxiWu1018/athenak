# Session 11 operating instructions

Read README_FIRST.md, REPORT.md and AGENT_LOGISTICS.md completely before work.
This is a preparation-only handoff, not a production-ready claim. The current
request explicitly forbids physics/science execution in the preparation chat.
Future science execution belongs to a new chat in this folder, with explicit
user direction. No automatic job submission, retries, deletion or transfer.

Source checkout: code/, branch project/GI-in-cluster. Preserve the four upstream
post-Session010 documentation commits. Push only after all required native GPU
validation passes; fast-forward only, never force push or overwrite remote work.
Native compilation is on allocated GG; GPU tests on allocated gh-dev; production
on sixteen GH nodes, one rank/GPU each, at most 36h, fresh t=0, strict t=150 Mref.
Do not change particles, profiles, boost, seeds, gauge, mesh or removal policy.
Keep Session 10 original data/archives unchanged. All new work belongs here.

Every production launch must match the current native/input hashes and genuine
validation/push receipts. Never manufacture a PASS file or treat compilation,
source-contract unit tests or an OS fixture as full-size memory/convergence proof.
Raw outputs and all failures are retained. Record job IDs, paths, source/executable
hashes, output completeness, coverage gaps and actual accounting in REPORT.md.
