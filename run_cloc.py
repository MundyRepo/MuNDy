#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def _pie_on_axis(
    ax,
    series,
    title="",
    min_frac=0.03,
    other_label="Other",
):
    """
    Draw a pie chart on the given axis.

    Labels show both percentage and absolute counts, e.g.:
        12.3%
        30,000

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    series : pandas.Series
        Index = labels, values = numeric values to show.
    title : str
        Title for this pie.
    min_frac : float
        Minimum fraction of total to label; slices below this fraction
        (before grouping) are aggregated into 'Other'.
    other_label : str
        Label to use for the aggregated 'other' category.
    """
    series = series[series > 0]

    if series.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return

    total = series.sum()
    frac = series / total

    # Separate "big" slices and aggregate the rest as "Other"
    big = frac[frac >= min_frac]
    small = frac[frac < min_frac]

    if not small.empty:
        big = big.copy()
        big[other_label] = small.sum()

    labels = big.index.tolist()
    values = (big * total).values

    def autopct_generator(total_local, min_frac_local):
        def autopct(pct):
            if pct < min_frac_local * 100:
                return ""
            abs_val = int(round(pct * total_local / 100.0))
            return f"{pct:.1f}%\n{abs_val:,.0f}"
        return autopct

    autopct = autopct_generator(total, min_frac)
    colors = sns.color_palette('muted')

    ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct=autopct,
        startangle=90,
        counterclock=False,
        wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
        textprops={"fontsize": 8},
    )

    ax.set_title(title, fontsize=10)
    ax.axis("equal")  # keep it circular


def plot_dir_pie_for_langs(
    df: pd.DataFrame,
    langs,
    value_col: str = "code",
    min_frac: float = 0.03,
    figsize=(5, 5),
):
    """
    Make ONE pie chart for the aggregate of the requested languages.
    Default use-case: merge 'C++' and 'C/C++ Header' into a single view.

    Parameters
    ----------
    df : DataFrame
        Columns: 'language', 'directory', and `value_col`.
    langs : list[str]
        Languages to include (merged together).
    value_col : str
        Which column to use for sizes (e.g., 'code').
    min_frac : float
        Minimum fraction of total to label; smaller slices aggregated into 'Other'.
    figsize : tuple
        Figure size in inches.
    """
    langs = list(langs)
    sub = df[df["language"].isin(langs)]

    if sub.empty:
        raise ValueError(f"No rows found for languages: {langs}")

    # Group by directory and sum over all chosen languages
    series = (
        sub.groupby("directory")[value_col]
        .sum()
        .sort_values(ascending=False)
    )

    # Strip "mundy/" etc. so we just show the last component (mesh, src, ...)
    series.index = [Path(d).name for d in series.index]

    total = series.sum()

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    lang_label = " & ".join(langs)
    title = f"{lang_label} by directory\n(total {total:,.0f} LOC)"

    _pie_on_axis(
        ax,
        series,
        title=title,
        min_frac=min_frac,
    )

    fig.tight_layout()
    return fig


def parse_langs_arg(s: str):
    # e.g. "C++,C/C++ Header,Python"
    return [part.strip() for part in s.split(",") if part.strip()]

def run_cloc(
    target_dir: Path,
    cloc_path: str,
    exclude_content: str,
) -> Dict[str, Any]:
    """
    Run cloc on target_dir and return parsed JSON.
    """
    cmd = [
        cloc_path,
        "--json",
        f"--exclude-content={exclude_content}",
        str(target_dir),
    ]

    # If cloc is a .pl script, you may need ["perl", cloc_path, ...] instead.
    # Uncomment this if needed:
    # cmd = ["perl", cloc_path, "--json", f"--exclude-content={exclude_content}", str(target_dir)]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[WARN] cloc failed for {target_dir}: {e.stderr.strip()}")
        return {}

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"[WARN] Failed to parse cloc JSON for {target_dir}")
        return {}

    return data


def cloc_json_to_rows(
    cloc_json: Dict[str, Any],
    directory_label: str,
) -> List[Dict[str, Any]]:
    """
    Convert cloc JSON output into a list of row dicts for a DataFrame.
    Each row is (directory, language, files, blank, comment, code).
    """
    rows = []

    for lang, stats in cloc_json.items():
        if lang in ("header", "SUM"):
            continue
        # cloc JSON per-language stats look like:
        # { "nFiles": int, "blank": int, "comment": int, "code": int }
        if not isinstance(stats, dict):
            continue

        rows.append(
            {
                "directory": directory_label,
                "language": lang,
                "n_files": stats.get("nFiles", 0),
                "blank": stats.get("blank", 0),
                "comment": stats.get("comment", 0),
                "code": stats.get("code", 0),
            }
        )

    return rows


def discover_directories(root: Path, include_root: bool = True) -> List[Path]:
    """
    Return a list of directories under root to run cloc on.
    Default: root itself, plus all *immediate* subdirectories (each recursively).
    """
    dirs = []
    if include_root:
        dirs.append(root)

    for entry in sorted(root.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            dirs.append(entry)

    return dirs


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run cloc per subdirectory of a project and collect results into a pandas DataFrame."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./mundy",
        help="Root directory of the project (default: ./mundy)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="C++,C/C++ Header",
        help="Comma-separated list of languages to aggregate "
             '(default: "C++,C/C++ Header")',
    )
    parser.add_argument(
        "--value-col",
        type=str,
        default="code",
        help="Column to use for sizes (default: code)",
    )
    parser.add_argument(
        "--min-frac",
        type=float,
        default=0.03,
        help="Minimum fraction of total to label; smaller slices grouped as 'Other'",
    )
    parser.add_argument(
        "--cloc-path",
        type=str,
        default="/home/bpalmer/Downloads/cloc-1.96.pl",
        help="Path to cloc executable or cloc-*.pl script",
    )
    parser.add_argument(
        "--exclude-content",
        type=str,
        default="Gauss_Legendre_Nodes_and_Weights",
        help="Pattern passed to cloc --exclude-content",
    )
    parser.add_argument(
        "--include-root",
        action="store_true",
        help="Also run cloc on the root directory itself",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="mundy_cloc_breakdown.csv",
        help="CSV file to write the aggregated DataFrame to",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Root directory {root} does not exist or is not a directory")

    dirs = discover_directories(root, include_root=args.include_root)
    all_rows: List[Dict[str, Any]] = []

    print(f"Running cloc on {len(dirs)} directories under {root}...\n")

    for d in dirs:
        rel_label = os.path.relpath(d, root.parent)  # e.g. mundy/src, mundy/tests
        print(f"  -> {rel_label}")
        cloc_json = run_cloc(
            target_dir=d,
            cloc_path=args.cloc_path,
            exclude_content=args.exclude_content,
        )
        rows = cloc_json_to_rows(cloc_json, directory_label=rel_label)
        all_rows.extend(rows)

    if not all_rows:
        print("No cloc data collected. Exiting.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output_csv, index=False)

    langs = parse_langs_arg(args.lang)

    fig = plot_dir_pie_for_langs(
        df,
        langs=langs,
        value_col=args.value_col,
        min_frac=args.min_frac,
        figsize=(4, 4),
    )
    plt.savefig('cloc_breakdown.png')    


if __name__ == "__main__":
    main()
