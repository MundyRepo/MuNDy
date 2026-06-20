#!/usr/bin/env python3
"""
PerfNeighborListScaling.py
──────────────────────────
Thread-scaling driver for MundySearch_PerfTestNeighborList.exe.

Sweeps OMP_NUM_THREADS over powers of 2 (up to the available CPU count),
runs the benchmark in --simple mode at each thread count, then produces
two figures:

  Figure 1  Speedup vs thread count  (one panel per benchmark phase)
  Figure 2  Absolute timing vs N     (one panel per benchmark phase,
                                      coloured by thread count)

Both PNG files are saved next to this script, and plt.show() is called
so the figures appear as interactive windows on a TkAgg-capable display.

Usage
-----
  python3 PerfNeighborListScaling.py [options]

  --exe   PATH          Path to the .exe  (default: look next to this script)
  --threads T [T ...]   Thread counts to test  (default: 1 2 4 8 … ≤ cpu_count)
  --out   PREFIX        Output file prefix  (default: neighbor_list_<backend>)
  --bind  spread|close  OMP_PROC_BIND policy  (default: spread)
  --sort-targets        Pre-sort STK entities by Z-Morton before building
  --sort-neighbors      Sort each target's neighbor row by source ordinal after construction
  --search arborx|stk   Which neighbor-list backend to benchmark  (default: arborx)
"""

import numpy as np
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

_INTERACTIVE = True
try:
    plt.switch_backend("TkAgg")
except Exception:
    _INTERACTIVE = False
    plt.switch_backend("Agg")

# ──────────────────────────────────────────────────────────────────────────────
# Visual style
# ──────────────────────────────────────────────────────────────────────────────

# Visual encoding derived from variant name:
#   marker shape — "o" (circle) = 1d list, "s" (square) = 2d list, "D" (diamond) = other
#   fill style   — solid = Full list,  hollow (mfc='none') = Half list
#   line style   — solid "-" = Full list,  dashed "--" = Half list


def _variant_props(name):
    """Return ax.plot / Line2D kwargs encoding dimension and full/half from the variant name."""
    marker = "o" if "1d" in name else ("s" if "2d" in name else "D")
    half = "Half" in name
    props = dict(
        linestyle="--" if half else "-",
        marker=marker,
        markersize=8,
        linewidth=1.8,
        markeredgewidth=1.5 if half else 0.5,
    )
    if half:
        props["markerfacecolor"] = "none"
    return props


# Short display titles for each section (keyed to the strings emitted by --simple)
_SECTION_LABELS = {
    "Construction":                       "Construction",
    "Iteration overhead (target)":        "Iter overhead\n(target loop)",
    "Iteration overhead (pair)":          "Iter overhead\n(pair loop)",
    "Global reduce (target)":             "Global reduce\n(target loop)",
    "Global reduce (pair)":               "Global reduce\n(pair loop)",
    "Atomic into target (full list)":     "Atomic -> target",
    "Bilateral atomic (half list)":       "Bilateral\natomic",
    "N2 brute force baseline (N<=1000)":  "N² baseline",
}

plt.rcParams.update({
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     ":",
    "font.size":          10,
})


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark execution
# ──────────────────────────────────────────────────────────────────────────────

def locate_exe(hint=None):
    if hint:
        p = Path(hint)
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"Specified exe not found: {hint}")
    here = Path(__file__).resolve().parent
    for name in ("MundySearch_PerfTestNeighborList.exe",
                 "MundySearch_PerfTestNeighborList"):
        p = here / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"Could not find MundySearch_PerfTestNeighborList.exe in {here}\n"
        "Pass --exe to specify the path explicitly.")


def run_bench(exe, nthreads, omp_bind="spread",
              sort_targets=False, sort_neighbors=False, search="arborx"):
    """Run with nthreads OpenMP threads; return stdout text (stderr discarded)."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(nthreads)
    env["OMP_PROC_BIND"] = omp_bind
    env["OMP_PLACES"] = "cores"
    cmd = [exe, "--simple", f"--kokkos-num-threads={nthreads}",
           "--search", search]
    if sort_targets:
        cmd.append("--sort-targets")
    if sort_neighbors:
        cmd.append("--sort-neighbors")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           env=env, timeout=1200)
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] threads={nthreads}", file=sys.stderr)
        return ""
    if r.returncode != 0:
        print(f"  [WARNING] threads={nthreads} exit={r.returncode}: "
              f"{r.stderr[:200]}", file=sys.stderr)
    return r.stdout


# ──────────────────────────────────────────────────────────────────────────────
# Output parser
# ──────────────────────────────────────────────────────────────────────────────

_SECTION_RE = re.compile(r"^\[(.+?)\]\s+\(median .+?\)")
_N_RE = re.compile(r"N=(\d+)")
# Matches either a time value ("12.34 us") or the missing-data sentinel ("---")
_ENTRY_RE = re.compile(r"(\d+\.\d+)\s+(ns|us|ms)|(---)")
_SCALE = {"ns": 1.0, "us": 1e3, "ms": 1e6}


def _to_ns(val_str, unit):
    return float(val_str) * _SCALE[unit]


def parse_output(text):
    """
    Return {section_str: {variant_str: {N_int: ns_per_op_float}}}
    Missing data points ("---", e.g. the N² baseline at large N) are stored
    as float('nan') so they can be skipped cleanly during plotting.
    """
    data = {}
    current = None
    n_vals = []

    for line in text.splitlines():
        stripped = line.strip()

        m = _SECTION_RE.match(stripped)
        if m:
            current = m.group(1)
            data[current] = {}
            n_vals = []
            continue

        if current and "N=" in line and "scaling" in line:
            n_vals = [int(x) for x in _N_RE.findall(line)]
            continue

        if (current and n_vals and ":" in stripped
                and not stripped.startswith("-")):
            colon = stripped.index(":")
            label = stripped[:colon].strip()
            rest = stripped[colon + 1:]
            times = []
            for val, unit, dash in _ENTRY_RE.findall(rest):
                if dash:
                    times.append(float("nan"))
                else:
                    times.append(_to_ns(val, unit))
            if len(times) == len(n_vals):
                data[current][label] = dict(zip(n_vals, times))

    return data


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_grid(n_panels, ncols=3):
    nrows = (n_panels + ncols - 1) // ncols
    return nrows, ncols


def _hide_extras(axes, n_used):
    nrows, ncols = axes.shape
    for k in range(n_used, nrows * ncols):
        axes[k // ncols, k % ncols].set_visible(False)


def _section_label(s):
    return _SECTION_LABELS.get(s, s)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — Speedup vs thread count
# ──────────────────────────────────────────────────────────────────────────────

def plot_scaling(all_data, thread_counts, out_prefix="neighbor_list", search="arborx"):
    """
    For each benchmark section, plot speedup = t(T_ref) / t(T) vs thread count T,
    where T_ref = thread_counts[0] (the smallest supplied thread count).
    One line per (variant, N) pair.  Colours encode N; line styles encode variant.
    """
    ref_threads = thread_counts[0]
    ref = all_data[ref_threads]
    sects = [s for s in ref if ref[s]]

    nrows, ncols = _make_grid(len(sects))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.0 * nrows),
                             sharex="all", sharey="all",
                             squeeze=False)
    fig.suptitle(
        f"Neighbor List ({search}) — Thread Scaling  (speedup relative to {ref_threads} thread"
        f"{'s' if ref_threads != 1 else ''})\n"
        r"unit-density domain  ·  $r_\mathrm{det}=\!\sqrt[3]{16}/4\approx 0.630$  ·  15 avg neighbours",
        fontsize=12, y=1.01,
    )

    threads = np.array(thread_counts, dtype=float)
    # ideal line reaches max_threads / ref_threads
    global_ymax = float(threads[-1] / threads[0])

    # Assign colors by sorted rank of N, not by N value, so the map stays valid
    # when kNValues changes.
    all_N = sorted({N for s in ref.values() for v in s.values() for N in v})
    _n_cmap = plt.cm.viridis
    n_colors = {N: _n_cmap(i / max(len(all_N) - 1, 1))
                for i, N in enumerate(all_N)}

    for ai, section in enumerate(sects):
        ax = axes[ai // ncols, ai % ncols]
        variants = list(ref[section].keys())

        for variant in variants:
            props = _variant_props(variant)
            N_vals = sorted(ref[section][variant])
            for N in N_vals:
                t1 = ref[section][variant].get(N, float("nan"))
                if not (t1 > 0):   # skips nan and zero
                    continue
                xs, ys = [], []
                for nt in thread_counts:
                    tN = all_data.get(nt, {}).get(
                        section, {}).get(variant, {}).get(N)
                    if tN is not None and tN > 0 and not np.isnan(tN):
                        xs.append(nt)
                        ys.append(t1 / tN)
                if xs:
                    global_ymax = max(global_ymax, max(ys))
                    ax.plot(xs, ys, color=n_colors.get(N, "gray"), **props)

        # Ideal scaling reference
        ax.plot(threads, threads / threads[0],
                "k--", linewidth=0.9, alpha=0.35, zorder=0)

        ax.set_title(_section_label(section), fontsize=11, pad=5)

    # ── Shared axis limits and ticks (set once; propagate via sharex/sharey) ──
    axes[0, 0].set_xticks(thread_counts)
    axes[0, 0].set_xticklabels(thread_counts)
    axes[0, 0].set_xlim(threads[0] - 0.5, threads[-1] + 0.5)
    axes[0, 0].set_ylim(0, global_ymax * 1.05)

    # Axis labels only on border panels (inner tick labels hidden by sharex/sharey)
    for col in range(ncols):
        axes[nrows - 1, col].set_xlabel("Thread count")
    for row in range(nrows):
        axes[row, 0].set_ylabel(f"Speedup  (x{ref_threads}-thread)")

    # ── Shared legends ────────────────────────────────────────────────────────
    # all_N and n_colors already computed above
    seen_vs, all_vs = set(), []
    for s in ref.values():
        for v in s:
            if v not in seen_vs:
                seen_vs.add(v)
                all_vs.append(v)

    # Color patches: one per N value
    n_patches = [mpatches.Patch(color=n_colors[N], label=f"N={N:,}")
                 for N in all_N]

    # Line-style/marker handles: one per variant + ideal dashed reference
    v_handles = [mlines.Line2D([], [], color="k", label=v, **_variant_props(v))
                 for v in all_vs]
    v_handles.append(mlines.Line2D([], [], linestyle="--", color="gray",
                                   alpha=0.6, linewidth=0.9, label="ideal"))

    # Reserve bottom margin so legends don't overlap subplot content
    plt.tight_layout(rect=[0, 0.13, 1, 0.97])

    fig.legend(handles=n_patches,
               title="Problem size", title_fontsize=9,
               loc="lower left", bbox_to_anchor=(0.02, 0.01),
               ncol=len(n_patches), fontsize=9, frameon=True)
    fig.legend(handles=v_handles,
               title="Variant / style", title_fontsize=9,
               loc="lower right", bbox_to_anchor=(0.98, 0.01),
               ncol=min(len(v_handles), 4), fontsize=9, frameon=True)

    _hide_extras(axes, len(sects))

    path = f"{out_prefix}_scaling.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — Absolute performance vs N
# ──────────────────────────────────────────────────────────────────────────────

def plot_performance(all_data, thread_counts, out_prefix="neighbor_list", search="arborx"):
    """
    For each section, plot ns/op vs N.  One line per (variant, thread_count) pair.
    Colours encode thread count (plasma); line styles encode variant.
    """
    ref = all_data[thread_counts[0]]
    sects = [s for s in ref if ref[s]]

    nrows, ncols = _make_grid(len(sects))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.0 * nrows),
                             squeeze=False)
    fig.suptitle(
        f"Neighbor List ({search}) — Absolute Performance vs N\n"
        r"unit-density domain  ·  $r_\mathrm{det}=\!\sqrt[3]{16}/4\approx 0.630$  ·  15 avg neighbours",
        fontsize=12, y=1.01,
    )

    tc_cmap = plt.cm.plasma
    tc_colors = {nt: tc_cmap(i / max(len(thread_counts) - 1, 1))
                 for i, nt in enumerate(thread_counts)}

    for ai, section in enumerate(sects):
        ax = axes[ai // ncols, ai % ncols]
        variants = list(ref[section].keys())

        for nt in thread_counts:
            for variant in variants:
                props = _variant_props(variant)
                sect_d = all_data.get(nt, {}).get(section, {}).get(variant, {})
                N_vals = sorted(sect_d)
                times = [sect_d[N] for N in N_vals]
                # Drop NaN entries (e.g. N² baseline skips large N)
                valid = [(N, t) for N, t in zip(N_vals, times)
                         if not np.isnan(t)]
                if not valid:
                    continue
                vN, vt = zip(*valid)
                ax.plot(vN, vt, color=tc_colors[nt], **props)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N  (spheres)")
        ax.set_ylabel("ns / operation")
        ax.set_title(_section_label(section), fontsize=11, pad=5)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # ── Shared legends ────────────────────────────────────────────────────────
    seen_vs, all_vs = set(), []
    for s in ref.values():
        for v in s:
            if v not in seen_vs:
                seen_vs.add(v)
                all_vs.append(v)

    # Solid colored lines: one per thread count
    tc_handles = [mlines.Line2D([], [], color=tc_colors[nt],
                                linewidth=2, label=f"T={nt}")
                  for nt in thread_counts]

    # Shape/fill handles: one per variant (black so color doesn't confuse the encoding)
    v_handles = [mlines.Line2D([], [], color="k", label=v, **_variant_props(v))
                 for v in all_vs]

    # Reserve bottom margin so legends don't overlap subplot content
    plt.tight_layout(rect=[0, 0.13, 1, 0.97])

    fig.legend(handles=tc_handles,
               title="Thread count", title_fontsize=9,
               loc="lower left", bbox_to_anchor=(0.02, 0.01),
               ncol=len(tc_handles), fontsize=9, frameon=True)
    fig.legend(handles=v_handles,
               title="Variant", title_fontsize=9,
               loc="lower right", bbox_to_anchor=(0.98, 0.01),
               ncol=min(len(v_handles), 4), fontsize=9, frameon=True)

    _hide_extras(axes, len(sects))

    path = f"{out_prefix}_performance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Performance table
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_ns(ns):
    """Format a nanosecond value as a fixed-width 10-character string."""
    if ns is None or np.isnan(ns) or ns < 0:
        return "       ---"
    if ns < 1e3:
        return f"{ns:7.1f} ns"
    if ns < 1e6:
        return f"{ns/1e3:7.2f} us"
    return f"{ns/1e6:7.2f} ms"


def print_performance_table(all_data, thread_counts):
    """
    Print a single table showing ns/op at the largest available N for every
    section and variant, with one column per thread count.
    """
    ref = all_data[thread_counts[0]]
    sects = [s for s in ref if ref[s]]

    _FMT_W = 10
    _COL_W = _FMT_W + 2
    _LABEL_W = 36   # "  [Section] variant"

    tc_header = "".join(f"{'T='+str(nt):>{_COL_W}}" for nt in thread_counts)
    divider = "-" * (_LABEL_W + len(tc_header))

    print(f"\n  {'Benchmark (largest N)':^{_LABEL_W}}{tc_header}")
    print(f"  {divider}")

    for section in sects:
        # Largest N that has data in the reference thread count for this section
        all_N = sorted({N for v in ref[section].values() for N in v
                        if not np.isnan(v[N])})
        if not all_N:
            continue
        largest_N = all_N[-1]
        title = f"[{_section_label(section).replace(chr(10), ' ')}]  N={largest_N:,}"

        print(f"\n  {title}")
        variants = list(ref[section].keys())
        for variant in variants:
            cells = "".join(
                f"{_fmt_ns(all_data.get(nt, {}).get(section, {}).get(variant, {}).get(largest_N)):>{_COL_W}}"
                for nt in thread_counts
            )
            print(f"    {variant:<{_LABEL_W - 4}}{cells}")

    print(f"\n  {divider}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def _default_threads():
    n = os.cpu_count() or 1
    counts = [2**k for k in range(6) if 2**k <= n]
    return counts if counts else [1]


def main():
    parser = argparse.ArgumentParser(
        description="Thread-scaling driver for MundySearch_PerfTestNeighborList")
    parser.add_argument("--exe", default=None,
                        help="Path to MundySearch_PerfTestNeighborList.exe")
    parser.add_argument("--threads", nargs="+", type=int, default=None,
                        help="Thread counts to sweep  (default: 1 2 4 … ≤ cpu_count)")
    parser.add_argument("--out", default=None,
                        help="Output file prefix  (default: neighbor_list_<backend>)")
    parser.add_argument("--bind", default="spread",
                        choices=["spread", "close"],
                        help="OMP_PROC_BIND policy  (default: spread)")
    parser.add_argument("--sort-targets", action="store_true",
                        help="Pre-sort STK entities by Z-Morton before building")
    parser.add_argument("--sort-neighbors", action="store_true",
                        help="Sort each target's neighbor row by source ordinal after construction")
    parser.add_argument("--search", default="arborx",
                        choices=["arborx", "stk"],
                        help="Neighbor-list backend to benchmark  (default: arborx)")
    args = parser.parse_args()

    exe = locate_exe(args.exe)
    thread_counts = sorted(
        set(args.threads)) if args.threads else _default_threads()
    out_prefix = args.out if args.out else f"neighbor_list_{args.search}"

    print("=" * 60)
    print("  Neighbor List — Thread Scaling Study")
    print("=" * 60)
    print(f"  Executable : {exe}")
    print(f"  Backend    : {args.search}")
    print(f"  Threads    : {thread_counts}")
    print(f"  OMP_PROC_BIND : {args.bind}  (OMP_PLACES=cores)")
    if args.sort_targets:
        print("  --sort-targets   : entities pre-sorted by Z-Morton via STK sort_entities")
    if args.sort_neighbors:
        print(
            "  --sort-neighbors : neighbor rows sorted by source ordinal after construction")
    print()

    all_data = {}
    for i, nt in enumerate(thread_counts):
        print(f"  [{i + 1}/{len(thread_counts)}] threads={nt} ...",
              end=" ", flush=True)
        stdout = run_bench(exe, nt, omp_bind=args.bind,
                           sort_targets=args.sort_targets,
                           sort_neighbors=args.sort_neighbors,
                           search=args.search)
        all_data[nt] = parse_output(stdout)
        n_sects = sum(1 for s in all_data[nt].values() if s)
        print(f"parsed {n_sects} non-empty sections")

    print()
    print("Generating plots...")
    plot_scaling(all_data, thread_counts,
                 out_prefix=out_prefix, search=args.search)
    plot_performance(all_data, thread_counts,
                     out_prefix=out_prefix, search=args.search)

    print()
    print_performance_table(all_data, thread_counts)

    if _INTERACTIVE:
        plt.show()
    else:
        print("Note: no display available (Agg backend) — PNGs saved, skipping plt.show()")


if __name__ == "__main__":
    main()
