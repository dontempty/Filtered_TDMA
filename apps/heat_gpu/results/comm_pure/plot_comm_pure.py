#!/usr/bin/env python3
"""
Stacked-bar breakdown of wall time per event from comm_pure/h200 CSVs.

Events shown per bar:
  rhs | solve_x | solve_y | tdma_z_gpu | tdma_z_comm | z_pack | comm | etc

  z_pack = solve_z - tdma_z_gpu - tdma_z_comm  (pack/unpack + sync overhead)

Two bars per group: PaScaL (left) | Filtered (right)
Figures: strong_breakdown.png, weak_breakdown.png, refine_breakdown.png
"""
import os, re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA_DIR = "/scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/results/scaling_1comm/h200"
OUT_DIR  = "/scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/results/comm_pure/image"
os.makedirs(OUT_DIR, exist_ok=True)

# ── parse all CSVs ────────────────────────────────────────────────────────────
RAW_EVENTS = ["rhs", "solve_x", "solve_y", "solve_z",
              "tdma_z_gpu", "tdma_z_comm", "comm", "etc"]

def parse_dir(dirpath):
    records = []
    for fname in sorted(os.listdir(dirpath)):
        if not fname.endswith(".csv") or not fname.startswith("timing_"):
            continue
        m = re.match(
            r"timing_(strong|weak|refine)_(.+)_np(\d+)_(pascal|filtered_v2)_rho(\d+)\.csv",
            fname)
        if not m:
            continue
        study, case, np_str, backend, rho = m.groups()
        np_val = int(np_str)

        grid, cells = "", 0
        timing = defaultdict(list)
        with open(os.path.join(dirpath, fname)) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    gm = re.search(r"grid=(\d+x\d+x\d+)", line)
                    if gm:
                        grid = gm.group(1)
                        dims = [int(x) for x in grid.split("x")]
                        cells = dims[0] * dims[1] * dims[2]
                    continue
                if line.startswith("rank"):
                    continue
                parts = line.split(",")
                if len(parts) < 4:
                    continue
                rank, t_step = int(parts[0]), int(parts[1])
                event, t = parts[2].strip(), float(parts[3])
                if t_step > 1:
                    timing[(rank, event)].append(t)

        ranks = sorted({r for (r, e) in timing.keys()})
        ref_ev = "solve_z"
        n_steps = len(timing[(ranks[0], ref_ev)]) if ranks else 0
        if n_steps == 0:
            continue

        ev = {}
        for event in RAW_EVENTS:
            per_step = [max((timing[(r, event)][i] if timing[(r, event)] else 0.0)
                            for r in ranks)
                        for i in range(n_steps)]
            ev[event] = float(np.mean(per_step)) if per_step else 0.0

        ev["z_pack"] = max(ev["solve_z"] - ev["tdma_z_gpu"] - ev["tdma_z_comm"], 0.0)

        records.append(dict(study=study, case=case, np=np_val,
                            backend=backend, rho=rho,
                            grid=grid, cells=cells, **ev))
    return records

records = parse_dir(DATA_DIR)
print(f"parsed {len(records)} records from {DATA_DIR}")

# ── color / event spec ────────────────────────────────────────────────────────
STACK = [
    ("rhs",         "#a8c8e8", "rhs"),
    ("solve_x",     "#b5d5a0", "solve_x"),
    ("solve_y",     "#7bbf70", "solve_y"),
    ("tdma_z_gpu",  "#3a7fc1", "tdma_z_gpu"),
    ("z_pack",      "#9ecae1", "z_pack"),
    ("tdma_z_comm", "#e34b3a", "tdma_z_comm"),
    ("comm",        "#f4a261", "comm"),
    ("etc",         "#cccccc", "etc"),
]
BCOLORS = {"pascal": "#4477aa", "filtered_v2": "#cc3311"}
BLABELS = {"pascal": "PaScaL", "filtered_v2": "Filtered"}
BAR_W   = 0.35
BNAMES  = ["pascal", "filtered_v2"]

# ── generic stacked-bar draw ──────────────────────────────────────────────────
def get_rec(study, case, np_or_case_key, backend, rho="025"):
    if isinstance(np_or_case_key, int):
        return next((r for r in records
                     if r["study"] == study and r["case"] == case
                     and r["np"] == np_or_case_key and r["backend"] == backend
                     and r["rho"] == rho), None)
    else:
        return next((r for r in records
                     if r["study"] == study and r["case"] == np_or_case_key
                     and r["backend"] == backend and r["rho"] == rho), None)

def draw_group(ax, group_x, backend, study, case, np_or_case_key,
               rho="025", normalize=False):
    rec = get_rec(study, case, np_or_case_key, backend, rho)
    if rec is None:
        return

    vals = [max(rec.get(ev_key, 0.0), 0.0) for ev_key, _, _ in STACK]
    total = sum(vals)
    if normalize and total > 0:
        vals = [v / total for v in vals]

    bi = BNAMES.index(backend)
    x  = group_x + (bi - 0.5) * BAR_W
    bottom = 0.0
    for (ev_key, color, _), val in zip(STACK, vals):
        if val > 0:
            ax.bar(x, val, BAR_W, bottom=bottom, color=color, edgecolor="none")
            bottom += val

# ── figure factory ────────────────────────────────────────────────────────────
def make_figure(study, cases, x_vals, x_labels, xlabel,
                subplot_title_fn, out_fname, figw=5, normalize=False):
    ncases = len(cases)
    fig, axes = plt.subplots(1, ncases, figsize=(figw * ncases, 6))
    if ncases == 1:
        axes = [axes]

    for ax, case in zip(axes, cases):
        for gi, xv in enumerate(x_vals):
            for b in BNAMES:
                draw_group(ax, gi, b, study, case, xv, normalize=normalize)

        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("fraction of total time" if normalize else "wall time / step  [s]")
        if normalize:
            ax.set_ylim(0, 1.0)
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_title(subplot_title_fn(case), fontweight="bold", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    ev_patches = [mpatches.Patch(color=c, label=lbl) for _, c, lbl in STACK]
    b_patches  = [mpatches.Patch(color=BCOLORS[b], label=f"{BLABELS[b]} (left/right bar)")
                  for b in BNAMES]
    fig.legend(handles=ev_patches + b_patches,
               loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    study_label = {"strong": "Strong scaling", "weak": "Weak scaling",
                   "refine": "Refinement"}[study]
    norm_tag = " (normalized)" if normalize else ""
    fig.suptitle(
        f"{study_label} — event breakdown{norm_tag}  (ρ=0.25, H200 GPU)\n"
        "left bar = PaScaL,  right bar = Filtered",
        fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    p = os.path.join(OUT_DIR, out_fname)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)

# ── strong ────────────────────────────────────────────────────────────────────
strong_cases = sorted(
    {r["case"] for r in records if r["study"] == "strong"},
    key=lambda c: int(c.split("x")[0]))
nps = sorted({r["np"] for r in records if r["study"] == "strong"})
make_figure("strong", strong_cases,
            x_vals=nps, x_labels=[f"np={n}" for n in nps], xlabel="# GPUs",
            subplot_title_fn=lambda c: c,
            out_fname="strong_breakdown.png")
make_figure("strong", strong_cases,
            x_vals=nps, x_labels=[f"np={n}" for n in nps], xlabel="# GPUs",
            subplot_title_fn=lambda c: c,
            out_fname="strong_breakdown_norm.png", normalize=True)

# ── weak ──────────────────────────────────────────────────────────────────────
weak_cases = sorted(
    {r["case"] for r in records if r["study"] == "weak"},
    key=lambda c: int(re.sub(r"\D", "", c)))
nps_w = sorted({r["np"] for r in records if r["study"] == "weak"})
make_figure("weak", weak_cases,
            x_vals=nps_w, x_labels=[f"np={n}" for n in nps_w], xlabel="# GPUs",
            subplot_title_fn=lambda c: f"{c}/GPU",
            out_fname="weak_breakdown.png")
make_figure("weak", weak_cases,
            x_vals=nps_w, x_labels=[f"np={n}" for n in nps_w], xlabel="# GPUs",
            subplot_title_fn=lambda c: f"{c}/GPU",
            out_fname="weak_breakdown_norm.png", normalize=True)

# ── refine (one subplot per family, x = case sorted by cells) ────────────────
refine_recs = [r for r in records if r["study"] == "refine"]
fam_key     = lambda c: "x".join(c.split("x")[:2])
fams        = sorted({fam_key(r["case"]) for r in refine_recs},
                     key=lambda f: int(f.split("x")[0]))

fig, axes = plt.subplots(1, len(fams), figsize=(6 * len(fams), 6))
if len(fams) == 1:
    axes = [axes]

def draw_refine_fig(normalize=False):
    fig, axes = plt.subplots(1, len(fams), figsize=(6 * len(fams), 6))
    if len(fams) == 1:
        axes = [axes]

    for ax, fm in zip(axes, fams):
        sub = sorted([r for r in refine_recs if fam_key(r["case"]) == fm
                      and r["backend"] == "pascal" and r["rho"] == "025"],
                     key=lambda r: r["cells"])
        cases_fm = [r["case"] for r in sub]
        nz_labels = [c.split("x")[2] for c in cases_fm]

        for gi, case in enumerate(cases_fm):
            for b in BNAMES:
                draw_group(ax, gi, b, "refine", "", case, normalize=normalize)

        ax.set_xticks(range(len(cases_fm)))
        ax.set_xticklabels(nz_labels, fontsize=8)
        ax.set_xlabel("nz")
        ax.set_ylabel("fraction of total time" if normalize else "wall time / step  [s]")
        if normalize:
            ax.set_ylim(0, 1.0)
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_title(f"{fm}×nz  (np=2)", fontweight="bold", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    ev_patches = [mpatches.Patch(color=c, label=lbl) for _, c, lbl in STACK]
    b_patches  = [mpatches.Patch(color=BCOLORS[b], label=f"{BLABELS[b]} (left/right bar)")
                  for b in BNAMES]
    fig.legend(handles=ev_patches + b_patches,
               loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    norm_tag = " (normalized)" if normalize else ""
    fig.suptitle(
        f"Refinement — event breakdown{norm_tag}  (ρ=0.25, np=2, H200 GPU)\n"
        "left bar = PaScaL,  right bar = Filtered",
        fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    fname = "refine_breakdown_norm.png" if normalize else "refine_breakdown.png"
    p = os.path.join(OUT_DIR, fname)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)

draw_refine_fig(normalize=False)
draw_refine_fig(normalize=True)

print("done ->", OUT_DIR)
