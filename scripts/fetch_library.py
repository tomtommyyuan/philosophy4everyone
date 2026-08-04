#!/usr/bin/env python3
"""Build `library/` from Project Gutenberg.

The extraction logic lives in `philo.corpus.gutenberg` so that it ships with
the installed package and `philo fetch` works without a checkout. This script
remains the repo-local entry point and writes into ./library.

    python scripts/fetch_library.py            # fetch everything missing
    python scripts/fetch_library.py --force    # re-download and rebuild
    python scripts/fetch_library.py --only kant mill
    python scripts/fetch_library.py --report   # show what was extracted
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from philo.corpus.gutenberg import SOURCES, fetch_all, select  # noqa: E402

LIBRARY = ROOT / "library"
CACHE = ROOT / ".cache" / "gutenberg"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build library/ from Project Gutenberg.")
    parser.add_argument("--only", nargs="*", default=[], help="slug substrings to fetch")
    parser.add_argument("--force", action="store_true", help="re-download and rebuild")
    parser.add_argument("--report", action="store_true", help="show extraction details")
    parser.add_argument("--list", action="store_true", help="list configured works")
    args = parser.parse_args(argv)

    if args.list:
        for s in SOURCES:
            print(f"{s.slug:46s} PG#{s.gid:<6d} {s.philosopher} — {s.work}")
        return 0

    if args.only and not select(args.only):
        print(f"no work matches {args.only}", file=sys.stderr)
        return 2

    report = fetch_all(LIBRARY, CACHE, only=args.only, force=args.force)

    for work in report.works:
        if not work.ok:
            print(f"  ✗ {work.slug}: download failed ({work.error})", file=sys.stderr)
            continue
        flag = "!" if work.warnings else "✓"
        print(f"  {flag} {work.slug:46s} {work.chars:>9,d} chars {work.sections:>4d} sections")
        for warning in work.warnings:
            print(f"      ⚠ {warning}")
        if args.report:
            body = work.path.read_text(encoding="utf-8").split("---\n", 2)[-1]
            body = re.sub(r"^<!--.*?-->\s*", "", body.strip(), flags=re.DOTALL)
            nl = chr(10)
            print(f"      ↦ opens: {textwrap.shorten(body[:400].replace(nl, ' '), 140, placeholder=' …')}")
            print(f"      ↦ ends:  {textwrap.shorten(body[-400:].replace(nl, ' '), 140, placeholder=' …')}")

    print(f"\n  {report.n_ok}/{len(report.works)} works · {report.total_chars:,d} characters → {LIBRARY}")
    if report.failures:
        return 1
    print("  next: philo ingest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
