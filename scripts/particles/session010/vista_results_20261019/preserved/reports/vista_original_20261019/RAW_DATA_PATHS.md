# Raw data locations and retention

Snapshot: 19 October 2026, Vista. All original files remain in place. This summary contains reduced plotting data and native text diagnostics, but not the bulk native state. No deletion, move, raw transfer or new production submission occurred.

## Original workspace

`/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff`

The full per-file inventory is [RAW_DATA_INVENTORY.jsonl](../inventory/RAW_DATA_INVENTORY.jsonl). It gives exact absolute Vista path, size, modification time, primary-branch membership, copy status, copied relative path, and inherited SHA256 where available. The [directory inventory](../inventory/RAW_DIRECTORIES.json) supplies exact counts/sizes. These are filesystem metadata plus prior seals, not a fresh rehash of roughly two TiB of raw output.

## Analysed primary branch

The current analysis selects 20 ordered closed output folders, 14,883 files at this inventory, 1,712.737 GiB including seal receipts. The historical SEALED receipt counts science files without the seal receipts themselves. Exact branch ordering, input/source hashes and checkpoint are in [INPUTS.json](../workspace/runtime_s10/analysis/through_segment002_20261018/INPUTS.json); [SEALED.json](../workspace/runtime_s10/analysis/through_segment002_20261018/SEALED.json) binds the original closed raw-file receipts. Each `SCIENCE_VERIFIED.json` is copied under its original relative output path.

Native `bin/` files contain 3D/plane metric, matter, constraints and Weyl fields; `vtk/` contains full native particle snapshots; `rst/` contains checkpoints. GW modes and horizon/tracker/removal/constraint text diagnostics are included in this summary. The inventory, not a filename guess, determines which exact file is copied.

### Verified latest checkpoint (not copied)

- Vista path: `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261016_gh_resume_01/segments/segment_002/chunk_012/out/rst/s10_orbit_gw.00020.rst`
- Time: 221.953125000533 Mref; cycle 138,334; bytes 50,996,709,000.
- SHA256 from the independently checked October 18 native audit: `fd11638112e5778b0255b4d6bab9c3652b281151436de224d943e6e8a1c1009c`.
- [Full checkpoint audit](../workspace/runtime_s10/analysis/through_segment002_20261018/RESTART_HEALTH.json). This audit is preserved historical evidence; packaging did not read/hash the fifty-GB checkpoint again.

## All raw output directories

The following table includes primary, validation and historical/debug output directories. All absolute paths below are on **Vista**, not local Perseus paths. “Sealed” means a prior file SHA256 is available, not independently archived elsewhere.

| Absolute Vista output directory | Primary | Files | GiB | Copied text/receipt files | Sealed files |
|---|---:|---:|---:|---:|---:|
| `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261013T200009Z/debug/amr/out` | no | 37 | 5.613 | 17 | 0 |
| `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261013T200009Z/debug/clean_stop/out` | no | 35 | 5.613 | 15 | 0 |
| `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261013T200009Z/debug/reference/out` | no | 58 | 28.852 | 18 | 0 |
| `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261013T200009Z/debug/resumed/out` | no | 35 | 14.428 | 15 | 0 |
| `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261013T200009Z/debug/seed/out` | no | 58 | 28.852 | 18 | 0 |
| `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261014_gh_retry_01/segments/segment_001/chunk_001/out` | yes | 864 | 92.810 | 21 | 863 |
| `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261014_gh_retry_01/segments/segment_001/chunk_002/out` | yes | 604 | 86.654 | 21 | 603 |
| `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261014_gh_retry_01/segments/segment_001/chunk_003/out` | yes | 630 | 90.789 | 21 | 629 |
| `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261014_gh_retry_01/segments/segment_001/chunk_004/out` | yes | 352 | 86.213 | 21 | 351 |
| `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261014_gh_retry_01/segments/segment_001/prefix_resumed/out` | yes | 36 | 86.756 | 16 | 35 |
| `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261014_gh_retry_01/segments/segment_001/prefix_seed/out` | yes | 59 | 173.507 | 19 | 58 |
| `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261014_gh_retry_01/segments/segment_002/chunk_001/out` | yes | 620 | 86.564 | 21 | 619 |
| `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261014_gh_retry_01/segments/segment_002/chunk_002/out` | yes | 626 | 88.471 | 22 | 625 |
| `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261014_gh_retry_01/validation_full/reference/out` | no | 58 | 173.507 | 18 | 0 |
| `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261016_gh_resume_01/segments/segment_002/chunk_001/out` | yes | 580 | 79.742 | 21 | 579 |
| `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261016_gh_resume_01/segments/segment_002/chunk_002/out` | yes | 672 | 74.741 | 22 | 671 |
| `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261016_gh_resume_01/segments/segment_002/chunk_003/out` | yes | 772 | 75.384 | 22 | 771 |
| `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261016_gh_resume_01/segments/segment_002/chunk_004/out` | yes | 869 | 76.017 | 22 | 868 |
| `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261016_gh_resume_01/segments/segment_002/chunk_005/out` | yes | 924 | 76.394 | 22 | 923 |
| `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261016_gh_resume_01/segments/segment_002/chunk_006/out` | yes | 807 | 75.508 | 22 | 806 |
| `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261016_gh_resume_01/segments/segment_002/chunk_007/out` | yes | 843 | 75.697 | 22 | 842 |
| `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261016_gh_resume_01/segments/segment_002/chunk_008/out` | yes | 917 | 76.336 | 22 | 916 |
| `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261016_gh_resume_01/segments/segment_002/chunk_009/out` | yes | 1,127 | 77.597 | 22 | 1126 |
| `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261016_gh_resume_01/segments/segment_002/chunk_010/out` | yes | 1,449 | 79.738 | 22 | 1448 |
| `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261016_gh_resume_01/segments/segment_002/chunk_011/out` | yes | 1,442 | 79.730 | 22 | 1441 |
| `/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/runs/20261016_gh_resume_01/segments/segment_002/chunk_012/out` | yes | 690 | 74.090 | 22 | 689 |

## Independent raw archival remains separate

A verified summary copy on Perseus does not archive the omitted raw files. The older [Anta plan](../workspace/ANALYSIS_AND_ARCHIVAL.md) remains a historical plan, not proof of a transfer. Before any separate bulk transfer, check destination quota/capacity, copy into a new destination, and verify every transferred file against the copied raw SHA256 receipts (or create missing source hashes first). Do not delete the Vista originals. Site scratch-retention rules may still remove old data independently; a path inventory is not a backup.
