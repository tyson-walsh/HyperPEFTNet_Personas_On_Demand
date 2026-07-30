"""Pick one example reply per kind-of-user for paper tab:example-replies.

Run after the GPU node phase2c + phase2d cells land. Emits a markdown table
to PROSPECTUS/HyperPEFTNet_RQ2/paper2/example_replies_sample.md that the
paper author can paste verbatim into the .tex file.

Selection: a) rage cohort only, b) rage topic only (unpopular_opinions_*),
c) reply long enough (>= 60 chars) and coherent (the is_coherent column).
Picks the median-length reply per stratum so the example isn't an outlier.
"""
from pathlib import Path
import argparse, json
import pandas as pd

CELLS = [
    ("Real user (reconstruction)",                 "2c_real_user_rage"),
    ("Synthetic, looks most like real users",      "2d_synth_rage_in_hull"),
    ("Synthetic, near the edge of real-user range", "2d_synth_rage_near_hull"),
    ("Synthetic, pushed outside real-user range",  "2d_synth_rage_far_kappa25"),
]

def pick(cell_dir: Path):
    fp = cell_dir / "forum.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    if "is_coherent" in df.columns:
        df = df[df["is_coherent"].fillna(False)]
    # rage cohort only
    if "author_type" in df.columns:
        df = df[df["author_type"] == "rage"]
    df = df[df["text"].str.len() >= 60]
    if df.empty:
        return None
    # median-length reply
    df = df.assign(_len=df["text"].str.len())
    df = df.sort_values("_len").reset_index(drop=True)
    return df.iloc[len(df) // 2]["text"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forum_root", required=True,
                    help="phase2_forums_REFRESH dir")
    ap.add_argument("--out_md", default="/workspace/hypernets/"
                    "PROSPECTUS/HyperPEFTNet_RQ2/paper2/example_replies_sample.md")
    args = ap.parse_args()
    root = Path(args.forum_root)
    lines = ["# tab:example-replies sample (paste into paper2_edits_*.tex)\n",
             "| Kind of user | Example reply |", "|---|---|"]
    for label, sub in CELLS:
        txt = pick(root / sub) or "[no coherent rage reply found in this cell]"
        # escape pipes + quotes for markdown table
        safe = txt.replace("|", "\\|").replace("\n", " ").strip()
        lines.append(f"| {label} | {safe} |")
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out_md}")

if __name__ == "__main__":
    main()
