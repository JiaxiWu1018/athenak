#!/usr/bin/env python3
"""Audit the organized release and reproduce its reduced-data scientific overview."""
import argparse
import ast
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from PIL import Image


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this hashing/replot audit inside Slurm")
    s = args.session.resolve()
    evidence = s / "evidence/vista_reconciliation_20261019"
    result_file = evidence / "FINAL_AUDIT.json"
    if result_file.exists():
        raise RuntimeError("Final audit exists; preserve the release")
    manifest = json.loads((evidence / "PROMOTION_MANIFEST.json").read_text())
    for record in manifest:
        target = s / record["destination"]
        assert target.is_file() and not target.is_symlink(), target
        assert target.stat().st_size == record["bytes"], target
        assert digest(target) == record["sha256"], target
    docs = [s / name for name in ("README.md", "REPORT.md", "REPORT_Jeans10.md",
                                  "REPORT_AGENT.md", "CLOSEOUT.md")]
    docs += [s / p / "README.md" for p in (
        "plots", "movies", "config", "inputs/vista_production", "reduced_data",
        "scripts/vista_original_20261019", "archive/vista_first_production_20261015",
        "reports/vista_original_20261019", "evidence/vista_reconciliation_20261019")]
    links = 0
    for doc in docs:
        assert doc.is_file(), doc
        for link in re.findall(r"\]\(([^)]+)\)", doc.read_text()):
            link = link.strip("<>").split("#", 1)[0]
            if not link or re.match(r"[a-zA-Z]+:", link):
                continue
            target = (doc.parent / link).resolve()
            assert target.exists(), (doc, link)
            links += 1
    code = s / "code/scripts/particles/session010/vista_results_20261019"
    for path in (code / "reconcile.py", code / "final_audit.py"):
        ast.parse(path.read_text(), filename=str(path))
    for name in ("perseus_reconcile_vista_20261019.sbatch", "perseus_audit_vista_20261019.sbatch"):
        subprocess.run(["bash", "-n", str(s / "scripts" / name)], check=True)
    native_paths = ["src", "CMakeLists.txt", "kokkos"]
    subprocess.run(["git", "-C", str(s / "code"), "diff", "--exit-code",
                    "386870f88f60e25e1a8fac76fe04a071b87a01cf", "--"] + native_paths,
                   check=True)
    assert digest(s / "inputs/vista_production/production.athinput") == (
        "aa712954cd7d0594395082670893378530bf8f7efc047ee3f51224a19c389ffe")
    prefix = evidence / "reproduction/00_scientific_summary"
    subprocess.run([sys.executable, str(code / "reconcile.py"), "--session", str(s),
                    "--replot-to", str(prefix)], check=True)
    metrics = json.loads(prefix.with_suffix(".json").read_text())
    assert metrics == json.loads((evidence / "SCIENTIFIC_RECHECK.json").read_text())
    plot_dir = s / "plots/through_segment002_20261018"
    plot_checks = []
    pdfinfo = shutil.which("pdfinfo")
    assert pdfinfo, "Native pdfinfo required for one-page PDF checks"
    for path in sorted(plot_dir.glob("*.png")):
        with Image.open(path) as img:
            size = img.size
            img.verify()
        pdf = path.with_suffix(".pdf")
        p = subprocess.run([pdfinfo, str(pdf)], check=True, text=True, capture_output=True)
        assert re.search(r"^Pages:\s+1\s*$", p.stdout, re.M), pdf
        plot_checks.append(dict(png=str(path.relative_to(s)), width=size[0], height=size[1],
                                png_sha256=digest(path), pdf_sha256=digest(pdf), pdf_pages=1))
    assert len(plot_checks) == 8
    figure_identical = digest(prefix.with_suffix(".png")) == digest(plot_dir / "00_scientific_summary.png")
    assert figure_identical, "Independent overview rendering differed"
    result = dict(passed=True, job=os.environ["SLURM_JOB_ID"], host=os.uname().nodename,
                  checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  promoted_files_rehashed=len(manifest), promoted_bytes=sum(x["bytes"] for x in manifest),
                  canonical_docs_checked=len(docs), canonical_links_checked=links,
                  syntax_checks_passed=True, native_source_unchanged=True,
                  independent_replot_metrics_identical=True, independent_replot_png_identical=True,
                  plots=plot_checks, movie_full_decodes="All six passed in Slurm 17024",
                  original_summary="Already fully verified in 17024; unchanged",
                  production_launched=False, raw_transferred=False, data_deleted=False)
    result_file.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("FINAL AUDIT PASSED", result_file)


if __name__ == "__main__":
    main()
