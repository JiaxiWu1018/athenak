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

User override (22 October 2026): the user explicitly approved queuing and starting
one manual production attempt before GPU validation and verified push, accepting
an unvalidated run and later manual repair/requeue if needed. The root launcher
uses code/scripts/particles/session011_early/ for this authorized mode; validation
job 1091645 and its original scripts remain unchanged. Source/executable/input/tool
hashes, production physics, all resource/output guards, and failure retention
remain mandatory. No GPU/push PASS receipt may be fabricated. Native GPU
validation is still required before a GitHub push. This supersedes only the older
validation/push-before-production rule; science is launched by the user from login,
not automatically by this preparation chat.

Every production launch must match its submitted native/input/tool hashes and
record its actual validation/push status. Never manufacture a PASS file or treat compilation,
source-contract unit tests or an OS fixture as full-size memory/convergence proof.
Raw outputs and all failures are retained. Record job IDs, paths, source/executable
hashes, output completeness, coverage gaps and actual accounting in REPORT.md.
