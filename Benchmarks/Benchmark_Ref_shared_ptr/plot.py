#!/usr/bin/env python3
"""
Plot the results of the ilm::Ref vs std::shared_ptr benchmark.

    python3 plot.py <results-dir> [--out plots] [--format png] [--dpi 150] [--show]

Reads   <results-dir>/timings.csv, memory.csv, meta.json   (written by ref_bench)
Writes  one figure per benchmark family into --out (default: <results-dir>/plots)

Requires: pip install matplotlib pandas numpy

timings.csv is tidy/long format, so you can also do your own analysis:

    df = pd.read_csv("results/timings.csv")
    df[df.benchmark == "copy_destroy"].groupby("impl").ns_per_op.median()

Columns: benchmark, impl, param_name, param, threads, repeat, iterations, ns_per_op
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
IMPL_ORDER = ["std_make_shared", "std_shared_new", "ilm_ref"]
IMPL_LABEL = {
    "std_make_shared": "std::make_shared",
    "std_shared_new": "std::shared_ptr(new T)",
    "ilm_ref": "ilm::Ref",
}
IMPL_COLOR = {
    "std_make_shared": "#4C78A8",
    "std_shared_new": "#F58518",
    "ilm_ref": "#54A24B",
}
IMPL_MARKER = {"std_make_shared": "o", "std_shared_new": "s", "ilm_ref": "^"}

BENCH_TITLE = {
    "create_destroy": "Create + destroy one object",
    "create_batch": "Create a batch of 4096 objects, then destroy them (per object)",
    "copy_destroy": "Copy-construct + destroy a handle",
    "copy_assign": "Copy-assign onto a handle that owns another object",
    "move_roundtrip": "Move-construct + move-assign",
    "deref": "operator-> + read",
    "upcast_copy": "Copy with implicit Derived -> Base conversion",
    "weak_create": "Create a weak handle from a strong one",
    "weak_lock": "Lock a weak handle (object alive)",
    "weak_lock_expired": "Lock a weak handle (object already destroyed)",
    "container_copy": "Copy a vector of handles (per element)",
    "container_iterate": "Iterate a shuffled vector of handles and read pointees (per element)",
    "contended_copy": "All threads copy + destroy handles to the same object",
}

PARAM_LABEL = {
    "object_size": "Object size (bytes)",
    "container_size": "Number of elements",
    "threads": "Threads",
}


def label(impl):
    return IMPL_LABEL.get(impl, impl)


def ordered_impls(present):
    known = [i for i in IMPL_ORDER if i in present]
    return known + sorted(i for i in present if i not in IMPL_ORDER)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def summarize(df, value="ns_per_op"):
    """Median plus interquartile range and min/max per (benchmark, impl, param)."""
    g = df.groupby(["benchmark", "impl", "param_name", "param"])[value]
    out = g.agg(
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        lo="min",
        hi="max",
        n="count",
    ).reset_index()
    return out


def footer_text(meta):
    if not meta:
        return ""
    bits = [meta.get("compiler", ""), f"{meta.get('hardware_threads', '?')} hw threads", meta.get("os", "")]
    if not meta.get("ndebug", True):
        bits.append("NOT OPTIMIZED (no NDEBUG)")
    if meta.get("single_thread_fast_path_allowed"):
        bits.append("std single-thread fast path ALLOWED")
    bits.append(f"{meta.get('repeats', '?')} repeats, median + IQR")
    return " | ".join(b for b in bits if b)


def finish(fig, meta, path, show):
    ft = footer_text(meta)
    if ft:
        fig.text(0.01, 0.005, ft, fontsize=7, alpha=0.6, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path)
    print(f"wrote {path}")
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def line_plot(summary, bench, outdir, fmt, meta, show, logx=True, logy=False, value_label="Time per operation (ns)"):
    import matplotlib.pyplot as plt

    s = summary[summary.benchmark == bench]
    if s.empty:
        return
    pname = s.param_name.iloc[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    for impl in ordered_impls(s.impl.unique()):
        d = s[s.impl == impl].sort_values("param")
        ax.plot(d.param, d["median"], marker=IMPL_MARKER.get(impl, "o"), color=IMPL_COLOR.get(impl), label=label(impl))
        ax.fill_between(d.param, d.q25, d.q75, color=IMPL_COLOR.get(impl), alpha=0.18, linewidth=0)

    if logx:
        ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(PARAM_LABEL.get(pname, pname))
    ax.set_ylabel(value_label)
    ax.set_title(BENCH_TITLE.get(bench, bench))
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    if not logy:
        ax.set_ylim(bottom=0)
    finish(fig, meta, outdir / f"{bench}.{fmt}", show)


def scalar_bars(df, outdir, fmt, meta, show):
    """One grouped bar chart for all benchmarks that have no parameter."""
    import matplotlib.pyplot as plt

    d = df[df.param_name == "none"]
    if d.empty:
        return
    s = summarize(d)
    benches = [b for b in BENCH_TITLE if b in set(s.benchmark)]
    impls = ordered_impls(s.impl.unique())
    x = np.arange(len(benches))
    width = 0.8 / max(1, len(impls))

    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(benches)), 5.5))
    for k, impl in enumerate(impls):
        med, err_lo, err_hi = [], [], []
        for b in benches:
            r = s[(s.benchmark == b) & (s.impl == impl)]
            if r.empty:
                med.append(np.nan); err_lo.append(0); err_hi.append(0)
                continue
            r = r.iloc[0]
            med.append(r["median"])
            err_lo.append(r["median"] - r.q25)
            err_hi.append(r.q75 - r["median"])
        ax.bar(x + (k - (len(impls) - 1) / 2) * width, med, width, yerr=[err_lo, err_hi], capsize=2,
               color=IMPL_COLOR.get(impl), label=label(impl))

    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("_", "\n") for b in benches], fontsize=8)
    ax.set_ylabel("Time per operation (ns, log scale)")
    ax.set_yscale("log")
    ax.set_title("Per-operation cost of common handle operations (lower is better)")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(loc="upper right")
    finish(fig, meta, outdir / f"handle_operations.{fmt}", show)


def contended(df, summary_unused, outdir, fmt, meta, show):
    import matplotlib.pyplot as plt

    d = df[df.benchmark == "contended_copy"].copy()
    if d.empty:
        return
    d["throughput_mops"] = d.threads / d.ns_per_op * 1e3  # total copy+destroy ops/s across all threads, in millions
    lat = summarize(d, "ns_per_op")
    thr = summarize(d, "throughput_mops")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, s, ylabel, title in (
        (axes[0], lat, "Latency per copy + destroy (ns)", "Per-thread latency (lower is better)"),
        (axes[1], thr, "Total throughput (million copies/s)", "Aggregate throughput (higher is better)"),
    ):
        for impl in ordered_impls(s.impl.unique()):
            r = s[s.impl == impl].sort_values("param")
            ax.plot(r.param, r["median"], marker=IMPL_MARKER.get(impl, "o"), color=IMPL_COLOR.get(impl), label=label(impl))
            ax.fill_between(r.param, r.q25, r.q75, color=IMPL_COLOR.get(impl), alpha=0.18, linewidth=0)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Threads")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()

    for ax in axes:
        ax.set_xticks(sorted(d.threads.unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    fig.suptitle(BENCH_TITLE["contended_copy"])
    finish(fig, meta, outdir / f"contended_copy.{fmt}", show)


def relative(summary, outdir, fmt, meta, show):
    """Ratio of baseline median / ilm::Ref median for every benchmark point. > 1 means ilm::Ref is faster."""
    import matplotlib.pyplot as plt

    if "ilm_ref" not in set(summary.impl):
        return

    bench_rank = {b: i for i, b in enumerate(BENCH_TITLE)}
    groups = sorted(
        summary.groupby(["benchmark", "param_name", "param"], sort=False),
        key=lambda kv: (bench_rank.get(kv[0][0], 999), kv[0][2]),
    )

    rows = []
    for (bench, pname, param), g in groups:
        m = g.set_index("impl")["median"]
        if "ilm_ref" not in m:
            continue
        name = bench if pname == "none" else f"{bench} [{pname}={int(param)}]"
        for base in ("std_make_shared", "std_shared_new"):
            if base in m and m["ilm_ref"] > 0:
                rows.append({"name": name, "baseline": base, "ratio": m[base] / m["ilm_ref"]})
    r = pd.DataFrame(rows)
    if r.empty:
        return

    names = list(dict.fromkeys(r.name))
    y = np.arange(len(names))
    height = 0.4

    fig, ax = plt.subplots(figsize=(10, max(5, 0.32 * len(names) + 1.5)))
    for k, base in enumerate(("std_make_shared", "std_shared_new")):
        vals = [r[(r.name == n) & (r.baseline == base)].ratio.iloc[0] if not r[(r.name == n) & (r.baseline == base)].empty else np.nan for n in names]
        vals = np.array(vals, dtype=float)
        # Diverge from 1.0: bars to the right = ilm::Ref faster, to the left = slower
        ax.barh(y + (k - 0.5) * height, vals - 1.0, height, left=1.0, color=IMPL_COLOR[base], label=f"vs {label(base)}")

    ax.axvline(1.0, color="black", linewidth=1)
    ax.set_xscale("log")
    lo, hi = np.nanmin(r.ratio), np.nanmax(r.ratio)
    candidates = [0.1, 0.2, 0.25, 0.33, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10, 20]
    ticks = [t for t in candidates if lo / 1.3 <= t <= hi * 1.3] or [1]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Speedup of ilm::Ref = baseline time / ilm::Ref time (log scale).  Left of 1x: ilm::Ref slower.  Right: faster.")
    ax.set_title("ilm::Ref relative to std::shared_ptr")
    ax.grid(True, axis="x", which="both", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
    finish(fig, meta, outdir / f"relative_speedup.{fmt}", show)


def memory(mem, outdir, fmt, meta, show):
    import matplotlib.pyplot as plt

    if mem is None or mem.empty:
        return

    impls = ordered_impls(mem.impl.unique())
    has_usable = mem.bytes_usable.notna().any()
    bytes_col = "bytes_usable" if has_usable else "bytes_requested"
    bytes_name = "allocator-usable bytes" if has_usable else "requested bytes"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # Handle size and allocation count
    first = mem[mem.object_size == mem.object_size.min()].set_index("impl")
    x = np.arange(len(impls))
    axes[0].bar(x, [first.loc[i, "sizeof_ptr"] for i in impls], color=[IMPL_COLOR.get(i) for i in impls])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([label(i) for i in impls], rotation=15, fontsize=8)
    axes[0].set_ylabel("sizeof(handle) in bytes")
    axes[0].set_title("Handle size")
    for xi, i in zip(x, impls):
        axes[0].text(xi, first.loc[i, "sizeof_ptr"], f"{int(first.loc[i, 'sizeof_ptr'])}", ha="center", va="bottom")

    axes[1].bar(x, [first.loc[i, "allocations"] for i in impls], color=[IMPL_COLOR.get(i) for i in impls])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([label(i) for i in impls], rotation=15, fontsize=8)
    axes[1].set_ylabel("Heap allocations per object")
    axes[1].set_title("Allocations to create one object")
    axes[1].set_ylim(0, mem.allocations.max() + 1)
    for xi, i in zip(x, impls):
        axes[1].text(xi, first.loc[i, "allocations"], f"{int(first.loc[i, 'allocations'])}", ha="center", va="bottom")

    # Overhead vs object size
    for impl in impls:
        d = mem[mem.impl == impl].sort_values("object_size")
        axes[2].plot(d.object_size, d[bytes_col] - d.object_size, marker=IMPL_MARKER.get(impl, "o"),
                     color=IMPL_COLOR.get(impl), label=label(impl))
    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel("Object size (bytes)")
    axes[2].set_ylabel(f"Overhead per object ({bytes_name} - object size)")
    axes[2].set_title("Per-object memory overhead (control block + allocator)")
    axes[2].grid(True, which="both", alpha=0.25)
    axes[2].legend()

    finish(fig, meta, outdir / f"memory.{fmt}", show)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: <results_dir>/plots)")
    ap.add_argument("--format", default="png", choices=["png", "svg", "pdf"])
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--show", action="store_true", help="also open interactive windows")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": args.dpi, "axes.spines.top": False, "axes.spines.right": False})

    timings = args.results_dir / "timings.csv"
    if not timings.exists():
        sys.exit(f"{timings} not found. Run ref_bench first.")

    df = pd.read_csv(timings)
    mem_path = args.results_dir / "memory.csv"
    mem = pd.read_csv(mem_path) if mem_path.exists() else None
    meta_path = args.results_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    out = args.out or (args.results_dir / "plots")
    out.mkdir(parents=True, exist_ok=True)

    summary = summarize(df)

    scalar_bars(df, out, args.format, meta, args.show)
    for bench in ("create_destroy", "create_batch"):
        line_plot(summary, bench, out, args.format, meta, args.show, logx=True)
    for bench in ("container_copy", "container_iterate"):
        line_plot(summary, bench, out, args.format, meta, args.show, logx=True)
    contended(df, summary, out, args.format, meta, args.show)
    relative(summary, out, args.format, meta, args.show)
    memory(mem, out, args.format, meta, args.show)


if __name__ == "__main__":
    main()
