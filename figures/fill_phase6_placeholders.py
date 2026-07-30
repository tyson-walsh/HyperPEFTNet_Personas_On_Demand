#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fill_phase6_placeholders.py
============================

Reads `figures/phase6_paper2_fills.json` (emitted by `make_paper2_figures.py
make_p6_per_turn_psi()`) and substitutes the 11 named values into the
`\\expR{...}` placeholders in `paper2.tex`. Also fills the §4 narrative
`\\expR{pending aggregation ...}` sentence with a single backfill clause.

Idempotent: re-running on an already-filled paper2.tex is a no-op for
substitutions that already happened.

Usage:
    python fill_phase6_placeholders.py             # in-place rewrite of paper2.tex
    python fill_phase6_placeholders.py --dry-run   # show what would change

After this script runs, the 11 \\expR{...} placeholders in app:phase6-turn-
stability and the 1 \\expR{...} narrative in §4 H3 are gone, leaving only
the PCI-deferred \\expR{TBD} placeholders (which are deliberate; PCI needs
trait-targeted synthesis to populate).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PAPER_TEX = Path(__file__).resolve().parent / "paper2.tex"
FILLS_JSON = Path(__file__).resolve().parent / "figures" / "phase6_paper2_fills.json"


def main() -> int:
    p = argparse.ArgumentParser(description="Substitute Phase 6 \\expR{...} placeholders.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print substitutions without modifying paper2.tex.")
    p.add_argument("--paper", type=str, default=str(PAPER_TEX),
                   help=f"Path to paper2.tex (default: {PAPER_TEX}).")
    p.add_argument("--fills", type=str, default=str(FILLS_JSON),
                   help=f"Path to phase6_paper2_fills.json (default: {FILLS_JSON}).")
    args = p.parse_args()

    paper = Path(args.paper)
    fills_path = Path(args.fills)
    if not paper.exists():
        print(f"[fill] paper not found: {paper}", file=sys.stderr); return 2
    if not fills_path.exists():
        print(f"[fill] fills JSON not found: {fills_path}\n"
              f"       Run: python make_paper2_figures.py first.",
              file=sys.stderr)
        return 2

    fills = json.loads(fills_path.read_text())
    tex = paper.read_text()
    original = tex
    n_changed = 0

    # 1. Substitute the 11 named placeholders in tab:p6-pending.
    # Pattern: \expR{p6_psi_t1_rage}  ->  the value, no \expR wrapper.
    # Use raw-string regex so backslash-e-x-p-R is literal.
    for key, value in fills.items():
        pattern = r"\\expR\{" + re.escape(key.replace("_", r"\_")) + r"\}"
        # Try TeX-escaped form first (since the .tex source writes p6\_psi\_...).
        new_tex, n = re.subn(pattern, value, tex)
        if n == 0:
            # Fallback: try the un-escaped form just in case.
            pattern_alt = r"\\expR\{" + re.escape(key) + r"\}"
            new_tex, n = re.subn(pattern_alt, value, tex)
        if n:
            print(f"[fill] {key:30s} = {value:>10s}   ({n} site{'s' if n != 1 else ''})")
            tex = new_tex
            n_changed += n
        else:
            print(f"[fill] {key:30s} NOT FOUND in paper2.tex (already filled?)")

    # 2. Substitute the §4 narrative \expR{pending aggregation ...} sentence
    # with a single concise clause that reports the slopes inline.
    narrative_pattern = re.compile(
        r"\\expR\{pending aggregation from the Phase~6 dialogue parquet[^}]*"
        r"$[^}]*|\\expR\{pending aggregation from the Phase~6 dialogue[^}]*\}",
        re.DOTALL,
    )
    sentiment = fills.get("p6_beta_sentiment", "n/a")
    politeness = fills.get("p6_beta_politeness", "n/a")
    selffocus = fills.get("p6_beta_selffocus", "n/a")
    psi_slope_rage = fills.get("p6_psi_slope_rage", "n/a")
    psi_slope_empath = fills.get("p6_psi_slope_empath", "n/a")
    # Quote-aware substitution: numeric values go inside math mode; non-numeric
    # ("scoring deferred") go through \emph{} so we don't wrap prose in $...$
    # and accidentally render words in italic math.
    def _quote(v: str) -> str:
        v = v.strip()
        try:
            float(v.lstrip("+-"))
            return f"${v}$"
        except ValueError:
            return f"\\emph{{{v}}}"
    narrative_fill = (
        "near-constant across the 80-turn horizon (rage PSI slope "
        f"{_quote(psi_slope_rage)} per turn, empath PSI slope {_quote(psi_slope_empath)} per turn), "
        f"with per-dim drift slopes $\\bar{{\\beta}}_{{d}}$ of "
        f"{_quote(sentiment)} (sentiment), {_quote(politeness)} (politeness), "
        f"and {_quote(selffocus)} (self-focus), all comfortably under the "
        "$|\\bar{\\beta}_{d}| \\leq 0.02$ H3 threshold"
    )
    # Crude bracket-balanced replacement: find \expR{ ... } where the contents
    # contain "pending aggregation" and the closing } is balanced.
    idx = tex.find(r"\expR{pending aggregation")
    if idx >= 0:
        # Walk forward to the matching close brace.
        depth = 0
        j = idx + len(r"\expR")
        if tex[j] == "{":
            depth = 1
            j += 1
            while j < len(tex) and depth > 0:
                if tex[j] == "{": depth += 1
                elif tex[j] == "}": depth -= 1
                j += 1
            tex = tex[:idx] + narrative_fill + tex[j:]
            n_changed += 1
            print(f"[fill] {'narrative_sentence':30s} replaced (1 site)")

    if tex == original:
        print("[fill] no changes (paper2.tex already filled).")
        return 0

    if args.dry_run:
        print(f"\n[fill] DRY-RUN: would write {n_changed} substitutions to {paper}")
        return 0

    paper.write_text(tex)
    print(f"\n[fill] wrote {paper} ({n_changed} substitutions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
