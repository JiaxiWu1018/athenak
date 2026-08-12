#!/usr/bin/env python3
"""Protect and prune restart checkpoints for one hybrid-campaign run.

Policy (documented in the campaign README):
  * protect every 0.5P checkpoint through 2P (the early-relaxation phase),
    and every integer-P checkpoint afterwards, plus t=0;
  * keep the newest two rolling restarts for crash recovery;
  * delete the rest.

Restart times are taken from the file index times the <output7> dt of the
run's input file (AthenaK numbers rst dumps sequentially), cross-checked
against the model period recorded in the input manifest.  Protected files are
hard links inside the same filesystem, so protection costs no extra space
until the rolling copy is pruned.
"""
import argparse
import json
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--keep-rolling", type=int, default=2)
    args = ap.parse_args()

    manifest = json.loads((args.root / "inputs" / "input_manifest.json").read_text())
    meta = next((c for c in manifest if c["name"] == args.case), None)
    if meta is None:
        raise SystemExit(f"{args.case}: not in input manifest")
    period = float(meta["period"])
    rst_dt = float(meta.get("rst_dt", period))
    tol = 0.51 * rst_dt

    rst_dir = args.run / "rst"
    if not rst_dir.is_dir():
        print(f"{args.case}: no rst dir")
        return
    protected_dir = args.root / "protected_restarts" / args.case
    protected_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(rst_dir.glob("*.rst"))
    entries = []
    for f in files:
        m = re.search(r"\.(\d+)\.rst$", f.name)
        if not m:
            continue
        t = int(m.group(1)) * rst_dt
        entries.append((t, f))

    def protect_target(t):
        # nearest 0.5P multiple through 2P, integer P beyond, plus t=0
        if t < 2.0 * period + tol:
            grid = 0.5 * period
        else:
            grid = period
        k = round(t / grid)
        target = k * grid
        return abs(t - target) < tol

    protected, kept, pruned = 0, 0, 0
    newest = {f for _, f in sorted(entries)[-args.keep_rolling:]}
    for t, f in entries:
        if protect_target(t):
            dest = protected_dir / f.name
            if not dest.exists():
                try:
                    dest.hardlink_to(f)
                except OSError:
                    import shutil
                    shutil.copy2(f, dest)
            protected += 1
        if f in newest or protect_target(t):
            kept += 1
            continue
        f.unlink()
        pruned += 1
    print(f"{args.case}: {len(entries)} rst; protected {protected}, "
          f"kept rolling {len(newest)}, pruned {pruned}")


if __name__ == "__main__":
    main()
