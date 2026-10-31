"""Merge the benchmark sweeps into one CSV and draw the report's graphs.

usage: python scripts/analyze_results.py <results-raw dir> <results.csv> <figures dir>

<results-raw dir> holds one folder per Slurm job sweep, as written by the
slurm_bench_*.sh scripts, each with a summary.csv from run_sweep.sh:
  gpu-3090-<mode>-<id>        single RTX 3090 (jrcai06)
  cpu-<id>                    reference CPU solvers (jrcai06's Threadripper)
  multigpu-a100-<mode>-<id>   4x A100 (server02)
where <mode> is fixed50 (ours run 50 iterations) or matched (ours run the
reference's iteration count). Writes every run into <results.csv> with
"system" and "mode" columns, prints median tables in Markdown on stdout, and
saves PNG graphs into <figures dir>.
"""
import csv
import os
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference palette (dataviz skill, light mode). Colour follows the entity: a
# *_nosync build is the same implementation, so it keeps its base colour and
# is told apart by a dotted line and a hollow marker.
BASE_COLOURS = {
    "cub": "#2a78d6", "thrust": "#eb6834", "nccl": "#1baf7a",
    "ref_gpu": "#eda100", "ref_streams": "#e87ba4",
    "ref_seq": "#008300", "ref_omp": "#6250d6",
}
MARKERS = {"cub": "o", "thrust": "s", "nccl": "^", "ref_gpu": "D", "ref_streams": "v",
           "ref_seq": "P", "ref_omp": "X"}
BASE_LABELS = {
    "cub": "ours: CUB", "thrust": "ours: Thrust", "nccl": "ours: NCCL",
    "ref_gpu": "reference: GPU (no streams)", "ref_streams": "reference: GPU + 5 streams",
    "ref_seq": "reference: CPU sequential", "ref_omp": "reference: CPU OpenMP",
}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GPU_SHADES = {1: "#86b6ef", 2: "#3987e5", 4: "#184f95"}  # one-hue ordinal ramp
GPU_MARKERS = {1: "o", 2: "s", 4: "^"}


def base(impl):
    return impl[:-len("_nosync")] if impl.endswith("_nosync") else impl


def label(impl):
    return BASE_LABELS[base(impl)] + (", no step syncs" if impl.endswith("_nosync") else "")


def short(impl):
    return label(impl).split(": ")[-1]


def draw(ax, impl, xs, ys, lbl=None):
    nosync = impl.endswith("_nosync")
    ours = not base(impl).startswith("ref")
    ls = ":" if nosync else ("-" if ours else "--")
    colour = BASE_COLOURS[base(impl)]
    ax.plot(xs, ys, ls, marker=MARKERS[base(impl)], color=colour, linewidth=2, markersize=6,
            markerfacecolor=SURFACE if nosync else colour, markeredgecolor=colour if nosync else SURFACE,
            markeredgewidth=1.4, label=lbl or label(impl))


def load(raw_dir):
    rows = []
    for job in sorted(os.listdir(raw_dir)):
        path = os.path.join(raw_dir, job, "summary.csv")
        if not os.path.isfile(path):
            continue
        system = ("RTX 3090 (jrcai06)" if job.startswith("gpu-3090") else
                  "Threadripper PRO 3975WX (jrcai06)" if job.startswith("cpu") else
                  "A100 (server02)" if job.startswith("multigpu") else job)
        mode = "matched" if "-matched-" in job else "fixed50"
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                r.update(system=system, mode=mode, job=job)
                rows.append(r)
    return rows


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def medians(rows, key, value):
    """{key: median} of a numeric column over ok runs."""
    groups = defaultdict(list)
    for r in rows:
        v = num(r[value])
        if r["status"].startswith("ok") and v is not None:
            groups[key(r)].append(v)
    return {k: statistics.median(v) for k, v in groups.items()}


def spreads(rows, key, value):
    """{key: (max-min)/median} of a numeric column over ok runs."""
    groups = defaultdict(list)
    for r in rows:
        v = num(r[value])
        if r["status"].startswith("ok") and v is not None:
            groups[key(r)].append(v)
    return {k: (max(v) - min(v)) / statistics.median(v) for k, v in groups.items() if len(v) > 1}


def style_axes(ax, title, xlabel, ylabel, xs, logy):
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs], fontsize=8)
    if logy:
        # plain numbers (0.5, 2, 40) rather than 4x10^1 on log axes
        plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(plain)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.grid(False, which="minor")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def direct_labels(fig, ax, ends):
    """ends: [(text, x, y)]. Right-end labels in text ink, pushed apart so they
    never overlap, each with a thin leader back to its line."""
    fig.canvas.draw()
    pts = sorted(([ax.transData.transform((x, y))[1], t, x, y] for t, x, y in ends))
    gap = 11 * fig.dpi / 72
    shifted = []
    for i, (py, t, x, y) in enumerate(pts):
        target = py if i == 0 else max(py, shifted[-1][0] + gap)
        shifted.append((target, py, t, x, y))
    for target, py, t, x, y in shifted:
        ax.annotate(t, (x, y), xytext=(14, (target - py) * 72 / fig.dpi), textcoords="offset points",
                    va="center", fontsize=7.5, color=MUTED,
                    arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.8, shrinkA=0, shrinkB=4))


def panel_chart(panels, path, ylabel, logy, ref_line=None, xlabel="N (million rows)", width=None):
    """panels: [(title, [(impl, [n...], [y...]), ...])], drawn side by side with a shared y-axis."""
    fig, axes = plt.subplots(1, len(panels), figsize=(width or 6.2 * len(panels), 4.8), dpi=150,
                             sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (title, series) in zip(axes[0], panels):
        # Direct labels name the implementation only, unless a panel holds both
        # builds of one implementation (the legend always carries the full name).
        bases = [base(i) for i, _, _ in series]
        both = len(set(bases)) < len(bases)
        ends = []
        for impl, ns, ys in series:
            xs = [n / 1e6 for n in ns]
            draw(ax, impl, xs, ys)
            ends.append((short(impl) if both else short(base(impl)), xs[-1], ys[-1]))
        if ref_line is not None:
            ax.axhline(ref_line, color=MUTED, linewidth=1, linestyle=":")
        allx = sorted({n / 1e6 for _, ns, _ in series for n in ns})
        style_axes(ax, title, xlabel, ylabel if ax is axes[0][0] else "", allx, logy)
        ax.legend(frameon=False, fontsize=7.5, loc="best", labelcolor=INK)
        ax.margins(x=0.3)
        ax._ends = ends
    fig.tight_layout()
    for ax in axes[0]:
        direct_labels(fig, ax, ax._ends)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt_n(n):
    return f"{n / 1e6:g}M"


def series_of(m, impl, ns, f=lambda v, n: v):
    pts = [(n, f(m[(impl, n)], n)) for n in ns if (impl, n) in m]
    return impl, [p[0] for p in pts], [p[1] for p in pts]


def main():
    raw_dir, out_csv, fig_dir = sys.argv[1:4]
    os.makedirs(fig_dir, exist_ok=True)
    rows = load(raw_dir)
    if not rows:
        sys.exit(f"no summary.csv found under {raw_dir}")

    fields = ["system", "mode", "job"] + [k for k in rows[0] if k not in ("system", "mode", "job")]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    failed = [r for r in rows if not r["status"].startswith("ok")]
    print(f"{len(rows)} runs merged into {os.path.basename(out_csv)}; {len(failed)} not ok\n")

    g_fixed = [r for r in rows if r["system"].startswith("RTX") and r["mode"] == "fixed50"]
    g_match = [r for r in rows if r["system"].startswith("RTX") and r["mode"] == "matched"]
    cpu = [r for r in rows if r["system"].startswith("Threadripper")]
    a100 = [r for r in rows if r["system"].startswith("A100")]
    ours = ["cub", "thrust", "nccl"]
    ours_ns = [o + "_nosync" for o in ours]

    # ---------------- single GPU, fixed 50 iterations ----------------
    mpi = medians(g_fixed, lambda r: (r["impl"], int(r["n"])), "ms_per_iter")
    ns = sorted({k[1] for k in mpi})
    if mpi:
        have = lambda i: any(k[0] == i for k in mpi)  # noqa: E731
        cols = [i for i in ours + ours_ns + ["ref_gpu", "ref_streams"] if have(i)]
        print("### RTX 3090, our fixed 50 iterations: median solve-loop time per iteration (ms)\n")
        print(table(["N"] + [label(i) for i in cols],
                    [[fmt_n(n)] + [f"{mpi[(i, n)]:.3f}" if (i, n) in mpi else "-" for i in cols] for n in ns]))
        sp = spreads(g_fixed, lambda r: (r["impl"], int(r["n"])), "ms_per_iter")
        if sp:
            print(f"\nLargest run-to-run spread (max-min)/median across repetitions: {max(sp.values()) * 100:.1f}%\n")

        panel_chart([("Solve-loop time per iteration, one RTX 3090",
                      [series_of(mpi, i, ns) for i in ours + ["ref_gpu", "ref_streams"] if have(i)])],
                    os.path.join(fig_dir, "single_gpu_ms_per_iter.png"),
                    "ms per iteration (median, log scale)", logy=True, width=8.4)

        if have("ref_streams"):
            ref = lambda v, n: mpi[("ref_streams", n)] / v  # noqa: E731
            panels = [("As written (per-step profiling syncs on)",
                       [series_of(mpi, i, ns, ref) for i in ours if have(i)])]
            if any(have(i) for i in ours_ns):
                panels.append(("Built without per-step syncs",
                               [series_of(mpi, i, ns, ref) for i in ours_ns if have(i)]))
            panel_chart(panels, os.path.join(fig_dir, "single_gpu_speedup_vs_ref.png"),
                        "speedup per iteration over reference + 5 streams (x)", logy=False, ref_line=1.0)
            cols2 = [i for i in ours + ours_ns + ["ref_gpu"] if have(i)]
            print("### Per-iteration speedup over the reference's 5-stream solver (x)\n")
            print(table(["N"] + [label(i) for i in cols2],
                        [[fmt_n(n)] + [f"{mpi[('ref_streams', n)] / mpi[(i, n)]:.2f}"
                                       if (i, n) in mpi and ('ref_streams', n) in mpi else "-" for i in cols2]
                         for n in ns]))
            print()

        if any(have(i) for i in ours_ns):
            gain = lambda o: (lambda v, n: (1 - mpi[(o + "_nosync", n)] / v) * 100)  # noqa: E731
            series = [(o + "_nosync",) + series_of(mpi, o, ns, gain(o))[1:] for o in ours
                      if have(o) and have(o + "_nosync")]
            panel_chart([("Time saved per iteration by dropping the per-step syncs", series)],
                        os.path.join(fig_dir, "single_gpu_sync_cost.png"),
                        "time saved per iteration (%)", logy=False, ref_line=0.0, width=8.4)
            print("### Cost of the per-step profiling syncs (RTX 3090, ms per iteration)\n")
            hdr = ["N"]
            for o in ours:
                hdr += [f"{BASE_LABELS[o]}: synced", "no syncs", "saved"]
            print(table(hdr, [[fmt_n(n)] + sum(([f"{mpi[(o, n)]:.3f}", f"{mpi[(o + '_nosync', n)]:.3f}",
                                                 f"{(1 - mpi[(o + '_nosync', n)] / mpi[(o, n)]) * 100:.0f}%"]
                                                if (o, n) in mpi and (o + "_nosync", n) in mpi else ["-"] * 3
                                                for o in ours), []) for n in ns]))
            print()

        if have("ref_gpu") and have("ref_streams"):
            tot = medians(g_fixed, lambda r: (r["impl"], int(r["n"])), "total_ms")
            print("### Reference stream gain on the RTX 3090 (its 5-stream solver vs its single-stream solver)\n")
            print(table(["N", "no streams: loop ms/iter", "streams: loop ms/iter", "gain (loop)",
                         "no streams: own total ms", "streams: own total ms", "gain (own total)"],
                        [[fmt_n(n), f"{mpi[('ref_gpu', n)]:.3f}", f"{mpi[('ref_streams', n)]:.3f}",
                          f"{(1 - mpi[('ref_streams', n)] / mpi[('ref_gpu', n)]) * 100:.1f}%",
                          f"{tot[('ref_gpu', n)]:.1f}", f"{tot[('ref_streams', n)]:.1f}",
                          f"{(1 - tot[('ref_streams', n)] / tot[('ref_gpu', n)]) * 100:.1f}%"]
                         for n in ns if all(k in mpi and k in tot for k in (('ref_gpu', n), ('ref_streams', n)))]))
            print()
            its = medians(g_fixed, lambda r: (r["impl"], int(r["n"])), "iterations")
            print("### Iterations the reference needs (its stopping test); ours: 50\n")
            print(table(["N", "reference: GPU (no streams)", "reference: GPU + 5 streams"],
                        [[fmt_n(n), f"{its.get(('ref_gpu', n), float('nan')):.0f}",
                          f"{its.get(('ref_streams', n), float('nan')):.0f}"] for n in ns]))
            print()

    # ---------------- single GPU, matched iteration counts ----------------
    msolve = medians(g_match, lambda r: (r["impl"], int(r["n"])), "solve_ms")
    mmpi = medians(g_match, lambda r: (r["impl"], int(r["n"])), "ms_per_iter")
    mits = medians(g_match, lambda r: (r["impl"], int(r["n"])), "iterations")
    if msolve:
        mns = sorted({k[1] for k in msolve})
        have = lambda i: any(k[0] == i for k in msolve)  # noqa: E731
        cols = [i for i in ours + ours_ns if have(i)]
        print("### RTX 3090, ours run the reference's iteration count: total solve-loop time (ms)\n")
        print(table(["N", "iterations"] + [label(i) for i in cols] + ["reference: GPU (no streams)",
                                                                      "reference: GPU + 5 streams"],
                    [[fmt_n(n), f"{mits.get(('ref_streams', n), mits.get(('ref_gpu', n), 0)):.0f}"]
                     + [f"{msolve[(i, n)]:.1f}" if (i, n) in msolve else "-" for i in cols]
                     + [f"{msolve[(r_, n)]:.1f}" if (r_, n) in msolve else "-" for r_ in ("ref_gpu", "ref_streams")]
                     for n in mns]))
        print()
        if have("ref_streams"):
            ref = lambda v, n: msolve[("ref_streams", n)] / v  # noqa: E731
            panels = [("As written", [series_of(msolve, i, mns, ref) for i in ours if have(i)])]
            if any(have(i) for i in ours_ns):
                panels.append(("Built without per-step syncs",
                               [series_of(msolve, i, mns, ref) for i in ours_ns if have(i)]))
            panel_chart(panels, os.path.join(fig_dir, "single_gpu_matched_speedup.png"),
                        "total loop-time speedup over reference + 5 streams (x)", logy=False, ref_line=1.0)
        if mpi:
            print("### Per-iteration time does not depend on the iteration count (RTX 3090, ms per iteration)\n")
            cols = [i for i in ours + ours_ns if (i, mns[0]) in mmpi]
            print(table(["N"] + sum(([f"{label(i)}: 50 its", "matched its"] for i in cols), []),
                        [[fmt_n(n)] + sum(([f"{mpi[(i, n)]:.3f}" if (i, n) in mpi else "-",
                                            f"{mmpi[(i, n)]:.3f}" if (i, n) in mmpi else "-"] for i in cols), [])
                         for n in mns]))
            print()

    # ---------------- CPU baselines ----------------
    cmpi = medians(cpu, lambda r: (r["impl"], int(r["n"])), "ms_per_iter")
    if cmpi:
        cns = sorted({k[1] for k in cmpi})
        threads = sorted({r["threads"] for r in cpu if r["impl"] == "ref_omp"})
        reps = max((int(r["rep"]) for r in cpu), default=0)
        print(f"### Reference CPU solvers ({reps} repetitions; OpenMP threads: {', '.join(threads)}): "
              "median ms per iteration\n")
        print(table(["N", "iterations", "sequential", "OpenMP", "OpenMP speedup"],
                    [[fmt_n(n), f"{medians(cpu, lambda r: (r['impl'], int(r['n'])), 'iterations').get(('ref_seq', n), 0):.0f}",
                      f"{cmpi[('ref_seq', n)]:.2f}", f"{cmpi[('ref_omp', n)]:.2f}",
                      f"{cmpi[('ref_seq', n)] / cmpi[('ref_omp', n)]:.2f}x"]
                     for n in cns if ('ref_seq', n) in cmpi and ('ref_omp', n) in cmpi]))
        print()
        if mpi:
            cand = [i for i in ("cub", "cub_nosync", "ref_streams") if any(k[0] == i for k in mpi)]
            over = lambda v, n: cmpi[("ref_seq", n)] / v  # noqa: E731
            series = [series_of(mpi, i, [n for n in ns if ("ref_seq", n) in cmpi], over) for i in cand]
            series.append(series_of(cmpi, "ref_omp", [n for n in cns if ("ref_seq", n) in cmpi], over))
            panel_chart([("Per-iteration speedup over the reference's sequential CPU solver", series)],
                        os.path.join(fig_dir, "speedup_vs_cpu.png"),
                        "speedup per iteration (x, log scale)", logy=True, width=8.4)
            cols = [i for i in ours + ours_ns + ["ref_gpu", "ref_streams"] if any(k[0] == i for k in mpi)]
            print("### Per-iteration speedup over the reference's sequential CPU solver (x)\n")
            print(table(["N"] + [label(i) for i in cols] + ["reference: CPU OpenMP"],
                        [[fmt_n(n)] + [f"{cmpi[('ref_seq', n)] / mpi[(i, n)]:.1f}"
                                       if (i, n) in mpi and ('ref_seq', n) in cmpi else "-" for i in cols]
                         + [f"{cmpi[('ref_seq', n)] / cmpi[('ref_omp', n)]:.1f}"
                            if ('ref_seq', n) in cmpi and ('ref_omp', n) in cmpi else "-"] for n in ns]))
            print()

    # ---------------- multi-GPU (A100) ----------------
    a_fixed = [r for r in a100 if r["mode"] == "fixed50"]
    for variant, title in (("nccl", "as written"), ("nccl_nosync", "without per-step syncs")):
        m = medians([r for r in a_fixed if r["impl"] == variant], lambda r: (int(r["gpus"]), int(r["n"])),
                    "ms_per_iter")
        if not m:
            continue
        gpus = sorted({k[0] for k in m})
        mns = sorted({k[1] for k in m})
        xs = [n / 1e6 for n in mns]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        for g in gpus:
            pts = [(n, m[(g, n)]) for n in mns if (g, n) in m]
            axes[0].plot([p[0] / 1e6 for p in pts], [p[1] for p in pts], "-", marker=GPU_MARKERS.get(g, "o"),
                         color=GPU_SHADES.get(g), linewidth=2, markersize=6, markeredgecolor=SURFACE,
                         label=f"{g} GPU{'s' if g > 1 else ''}")
            if g > 1:
                sp = [(n, m[(1, n)] / m[(g, n)]) for n in mns if (g, n) in m and (1, n) in m]
                axes[1].plot([p[0] / 1e6 for p in sp], [p[1] for p in sp], "-", marker=GPU_MARKERS.get(g, "o"),
                             color=GPU_SHADES.get(g), linewidth=2, markersize=6, markeredgecolor=SURFACE,
                             label=f"{g} GPUs (ideal {g}x, dotted)")
                axes[1].axhline(g, color=GPU_SHADES.get(g), linewidth=1, linestyle=":")
        axes[1].axhline(1, color=MUTED, linewidth=1, linestyle=":")
        style_axes(axes[0], f"NCCL version {title}: time per iteration", "N (million rows)",
                   "ms per iteration (median, log)", xs, True)
        style_axes(axes[1], "Strong-scaling speedup over 1 GPU", "N (million rows)", "speedup (x)", xs, False)
        for ax in axes:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"multi_gpu_scaling_{variant}.png"), facecolor=SURFACE)
        plt.close(fig)

        print(f"### A100 (NVLink), {label(variant)}: median ms per iteration and speedup over 1 GPU\n")
        hdr = ["N"] + [f"{g} GPU ms/iter" for g in gpus] + [f"{g} GPU speedup" for g in gpus if g > 1] \
            + [f"{g} GPU efficiency" for g in gpus if g > 1]
        body = []
        for n in mns:
            r_ = [fmt_n(n)] + [f"{m[(g, n)]:.3f}" if (g, n) in m else "-" for g in gpus]
            r_ += [f"{m[(1, n)] / m[(g, n)]:.2f}x" if (g, n) in m and (1, n) in m else "-" for g in gpus if g > 1]
            r_ += [f"{m[(1, n)] / m[(g, n)] / g * 100:.0f}%" if (g, n) in m and (1, n) in m else "-"
                   for g in gpus if g > 1]
            body.append(r_)
        print(table(hdr, body))
        print()

    if a_fixed:
        ctx = medians([r for r in a_fixed if r["gpus"] in ("1",)], lambda r: (r["impl"], int(r["n"])), "ms_per_iter")
        mns = sorted({k[1] for k in ctx})
        cols = [i for i in ("cub", "cub_nosync", "nccl", "nccl_nosync", "ref_streams") if any(k[0] == i for k in ctx)]
        print("### One A100: median ms per iteration\n")
        print(table(["N"] + [label(i) for i in cols],
                    [[fmt_n(n)] + [f"{ctx[(i, n)]:.3f}" if (i, n) in ctx else "-" for i in cols] for n in mns]))
        print()
    a_match = [r for r in a100 if r["mode"] == "matched"]
    if a_match:
        s = medians(a_match, lambda r: (r["impl"], int(r["gpus"]), int(r["n"])), "solve_ms")
        it = medians(a_match, lambda r: (r["impl"], int(r["n"])), "iterations")
        mns = sorted({k[2] for k in s})
        cols = [(i, g) for i in ("cub", "cub_nosync", "nccl", "nccl_nosync") for g in (1, 2, 4)
                if (i, g, mns[0]) in s]
        print("### A100, ours run the reference's iteration count: total solve-loop time (ms)\n")
        print(table(["N", "iterations"] + [f"{label(i)} ({g} GPU)" for i, g in cols] + ["reference: GPU + 5 streams"],
                    [[fmt_n(n), f"{it.get(('ref_streams', n), 0):.0f}"]
                     + [f"{s[(i, g, n)]:.1f}" if (i, g, n) in s else "-" for i, g in cols]
                     + [f"{s[('ref_streams', 1, n)]:.1f}" if ('ref_streams', 1, n) in s else "-"] for n in mns]))
        print()

    if failed:
        print("### Runs that were not ok\n")
        print(table(["system", "mode", "impl", "N", "gpus", "rep", "status"],
                    [[r["system"], r["mode"], r["impl"], r["n"], r["gpus"], r["rep"], r["status"]] for r in failed]))


if __name__ == "__main__":
    main()
