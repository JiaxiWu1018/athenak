# Real-data AH visibility validation and pending deployment

## Real-data visual repair validated — 2026-10-22 00:36 UTC

474110 completed in4s/1node; exact cohort/surface checks passed, but image review
found AHs obscured by Matplotlib collection ordering. Original previews preserved.
Renderer d7eea4cc now draws exact spectral surfaces as explicitly disclosed overlays,
without enlargement. Isolated474151 passed18controller/10product checks, full
synthetic movie decode and real-state exact-ID/rest-weight/pixel bounds checks in
26s/1node. Both wide/close previews visually inspected; all4 render manifest files
and5 QA receipt/log checksums verified on Perseus. Not final75 products.

Guarded renderer-only installer ebabedd7 is queued as474166 afterok474002:474151.
It replaces products.py and its seal only after passing receipts and an initialized
RUNNING production interval under submit.lock. Deferred exit75 changes nothing.
The473992->474002 saved-AH validation/deployment chain remains intact, and remote
checkout stays5fa3100a. Production473967 remains queued from audited8.803125.

Production49.2866667nodeh unchanged; these two checks add0.0083333nodeh, bringing
subsequent debugging/validation to0.2025nodeh, excluding initial builds/preflights.
Queued jobs have no allocation cost. No physical input/engine change or replay.
Next inspect473967 startup,473992/474002/474166 receipts, mirror deployed scripts
only after pass, and continue the full75/AH/report/movie goal.
