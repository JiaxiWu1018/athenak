# User-authorized early Session 11 production

The user explicitly approved starting production before compact GPU validation
and before a verified GitHub push, accepting the possibility of fixing failures
and manually requeuing later. This overrides those two pre-launch requirements
for one production attempt. It does not mark the code scientifically validated,
create validation/push receipts, authorize early push, or change the physics.

From a Vista login node, in the session root:

```sh
bash submit_s11.sh check
bash submit_s11.sh production
```

The production command submits one ordinary, eligible 16-node GH job for at most
36 hours. It has no hold or dependency on validation job 1091645. Pending/running
validation is allowed; another production job or previous production attempt is
blocked. The launcher retains the full Vista banner/response, correctly extracts
the terminal job ID, and prints the job/log/output/receipt paths. Ambiguous
responses still block repeats. No job submits, retries, or cancels another job.

Production uses the exact same native executable, inputs and production runner
as session011. This separate adapter changes its readiness policy and extends
its hash manifest in the production process only. The original validation tools
and their manifest remain unchanged, preserving the already queued validation.
The original strict validation-and-push route remains available in session011.

The committed production_policy.json pins the approved native source, GI binary
and production input hashes. Preflight still requires a clean committed campaign
checkout, account/balance, live queue checks, modules, disk headroom and no
duplicate production. The expected full manifest digest and commit are inherited
by the batch job. Changes between submission/startup or during the run fail the
original guards; use this helper rather than submitting the early batch file
directly. Exporting variables does not fabricate a passing validation receipt.

RUN_MODE.json, the submission receipt and the manifest in COMMAND/FINAL explicitly
identify `user_authorized_unvalidated`, with GPU validation/push unverified. A
successful FINAL is an execution/output-completeness result, not scientific
validation. Existing memory/telemetry/capacity/disk guards, forty-minute reserve,
actual t0 census, final native audits, output coverage, failure retention and
strict t150 stop all remain in the unchanged production runner. Final figures,
movies, reviewed strain and scientific analysis remain post-run work.

The root submit_s11.sh uses this adapter for check/validation/production. Its
validation command retains the original validation batch script and cases.
Native compilation is unnecessary because no native source changed. Native GPU
validation is still required before any GitHub push; the strict push tool is
unchanged. Keep all original data, logs and submission attempts.
