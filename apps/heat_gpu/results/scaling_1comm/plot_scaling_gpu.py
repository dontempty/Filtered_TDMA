#!/usr/bin/env python3
"""
Build metrics.csv from h200 raw timing CSVs, then generate scaling plots.

Dataset: scaling_1comm/h200  (H200 GPU, 1-comm-round Filtered algorithm)
Studies : strong, weak  (no refine in this dataset)
Metric  : solve_z  — the distributed-direction TDMA, the ONLY component where
          PaScaL and Filtered differ; solve_x/y fall back to local Thomas (=PaScaL).

Aggregation:
  per step: max across ranks (wall-clock bottleneck)
  per run : mean over steps 2-N  (skip step 1 as warmup)

Outputs:
  h200/metrics.csv
  h200/gpu_image/strong_{case}_z.png
  h200/gpu_image/weak_{case}_z.png
"""

import os, re, csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ───────────────────────────────────────────────────────────────────
DATA_DIR = "/scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/results/scaling_1comm/h200"
BASE_OUT = "/scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/results/scaling_1comm"
OUT_CSV  = os.path.join(BASE_OUT, "metrics.csv")
OUT_DIR  = "/scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/results/scaling_1comm/gpu_image"
os.makedirs(OUT_DIR, exist_ok=True)

EVENTS      = ["rhs", "solve_x", "solve_y", "solve_z", "comm", "etc"]
COMM_DIR    = "/scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/results/comm_breakdown/h200"
COMM_EVENTS = ["rhs", "solve_x", "solve_y", "solve_z", "comm", "etc",
               "tdma_x_comm", "tdma_y_comm", "tdma_z_comm"]

# ── build comm_breakdown lookup: (study,case,np,backend,rho) → tdma_z_comm ──
def parse_timing_dir(dirpath, event_list):
    """Return list of dicts with per-event mean wall times."""
    result = []
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
        n_steps = len(timing[(ranks[0], "solve_z")]) if ranks else 0
        if n_steps == 0:
            continue
        ev_means = {}
        for event in event_list:
            per_step_max = [
                max((timing[(r, event)][i] if timing[(r, event)] else 0.0)
                    for r in ranks)
                for i in range(n_steps)
            ]
            ev_means[event] = float(np.mean(per_step_max)) if per_step_max else 0.0
        result.append(dict(study=study, case=case, np=np_val,
                           backend=backend, rho=rho,
                           grid=grid, cells=cells, **ev_means))
    return result

comm_rows = parse_timing_dir(COMM_DIR, COMM_EVENTS)
# keyed lookup
comm_lookup = {
    (r["study"], r["case"], r["np"], r["backend"], r["rho"]): r
    for r in comm_rows
}

def get_comm(study, case, np_val, backend, rho, event="tdma_z_comm"):
    key = (study, case, np_val, backend, rho)
    rec = comm_lookup.get(key)
    return rec[event] if rec else None

# ── build metrics.csv ────────────────────────────────────────────────────────
rows = []
for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith(".csv") or not fname.startswith("timing_"):
        continue
    m = re.match(
        r"timing_(strong|weak|refine)_(.+)_np(\d+)_(pascal|filtered_v2)_rho(\d+)\.csv",
        fname)
    if not m:
        continue
    study, case, np_str, backend, rho = m.groups()
    np_val = int(np_str)

    # read raw data
    grid = ""
    cells = 0
    timing = defaultdict(list)          # (rank, event) -> [time per step]
    with open(os.path.join(DATA_DIR, fname)) as f:
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
            if t_step > 1:                  # skip warmup
                timing[(rank, event)].append(t)

    ranks = sorted({r for (r, e) in timing.keys()})
    n_steps = len(timing[(ranks[0], "solve_z")]) if ranks else 0
    if n_steps == 0:
        continue

    # max-across-ranks then mean-across-steps
    event_means = {}
    for event in EVENTS:
        per_step_max = [
            max(timing[(r, event)][i] for r in ranks)
            for i in range(n_steps)
        ]
        event_means[event] = float(np.mean(per_step_max))

    wall = sum(event_means.values())
    rows.append(dict(
        study=study, case=case, np=np_val, backend=backend, rho=rho,
        grid=grid, cells=cells,
        wall_per_step_s=wall,
        rhs=event_means["rhs"],
        solve_x=event_means["solve_x"],
        solve_y=event_means["solve_y"],
        solve_z=event_means["solve_z"],
        comm=event_means["comm"],
    ))

rows.sort(key=lambda x: (x["study"], x["case"], x["np"], x["backend"], x["rho"]))
FIELDS = ["study","case","np","backend","rho","grid","cells",
          "wall_per_step_s","rhs","solve_x","solve_y","solve_z","comm"]
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(f"metrics.csv: {len(rows)} rows → {OUT_CSV}")

# ── plotting helpers ─────────────────────────────────────────────────────────
METRIC = "solve_z"
SERIES = [
    ("pascal",      "025"),
    ("pascal",      "040"),
    ("filtered_v2", "025"),
    ("filtered_v2", "040"),
]

def style(b, rho):
    color = "tab:blue"  if b == "pascal"      else "tab:red"
    ls    = "-"         if rho == "025"        else "--"
    mk    = "o"         if rho == "025"        else "s"
    return color, ls, mk

def label(b, rho):
    bname = "PaScaL"   if b == "pascal"      else "Filtered"
    rname = "0.25"     if rho == "025"        else "0.40"
    return f"{bname}  ρ={rname}"

def sel(study, **kw):
    return [r for r in rows
            if r["study"] == study and all(r[k] == v for k, v in kw.items())]

ZNOTE = "z-direction (distributed) solve only — sole component where PaScaL ≠ Filtered"

# ── strong scaling ───────────────────────────────────────────────────────────
strong_cases = sorted(
    {r["case"] for r in rows if r["study"] == "strong"},
    key=lambda c: int(c.split("x")[0])
)
for case in strong_cases:
    sub  = sel("strong", case=case)
    grid = sub[0]["grid"] if sub else "?"
    nps  = sorted({r["np"] for r in sub})

    fig, (axT, axS) = plt.subplots(1, 2, figsize=(12, 5))

    for b, rho in SERIES:
        s = sorted(
            [r for r in sub if r["backend"] == b and r["rho"] == rho],
            key=lambda r: r["np"])
        if not s:
            continue
        x  = [r["np"]         for r in s]
        y  = [r[METRIC]       for r in s]
        c, ls, mk = style(b, rho)
        axT.plot(x, y, ls, color=c, marker=mk, label=label(b, rho))
        axS.plot(x, [y[0] / v for v in y], ls, color=c, marker=mk, label=label(b, rho))

    # ideal reference from PaScaL ρ=0.25
    base = sorted(
        [r for r in sub if r["backend"] == "pascal" and r["rho"] == "025"],
        key=lambda r: r["np"])
    if base:
        n0, t0 = base[0]["np"], base[0][METRIC]
        xi = np.array(nps, float)
        axT.plot(xi, t0 * n0 / xi, "k:", lw=1, label="ideal (∝ 1/np)")
        axS.plot(xi, xi / n0,       "k--", lw=1, label="ideal linear")

    axT.set_xscale("log", base=2); axT.set_yscale("log")
    axT.set_xticks(nps); axT.set_xticklabels(nps)
    axT.set_xlabel("# GPUs (MPI ranks)"); axT.set_ylabel("solve_z time / step  [s]")
    axT.set_title("z-solve time per step"); axT.grid(True, which="both", alpha=.3)
    axT.legend(fontsize=8)

    axS.set_xticks(nps); axS.set_xticklabels(nps)
    axS.set_xlabel("# GPUs (MPI ranks)")
    axS.set_ylabel(f"speedup  Tz({nps[0]})/Tz(np)")
    axS.set_title("z-solve strong-scaling speedup")
    axS.grid(True, alpha=.3); axS.legend(fontsize=8)

    fig.suptitle(
        f"Strong scaling (z-solve) — {case}  (grid {grid})  H200 GPU\n{ZNOTE}",
        fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(OUT_DIR, f"strong_{case}_z.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)

# ── weak scaling ─────────────────────────────────────────────────────────────
weak_cases = sorted(
    {r["case"] for r in rows if r["study"] == "weak"},
    key=lambda c: int(re.sub(r"\D", "", c))
)
for case in weak_cases:
    sub = sel("weak", case=case)
    nps = sorted({r["np"] for r in sub})

    fig, (axT, axE) = plt.subplots(1, 2, figsize=(12, 5))

    for b, rho in SERIES:
        s = sorted(
            [r for r in sub if r["backend"] == b and r["rho"] == rho],
            key=lambda r: r["np"])
        if not s:
            continue
        x  = [r["np"]   for r in s]
        y  = [r[METRIC] for r in s]
        c, ls, mk = style(b, rho)
        axT.plot(x, y, ls, color=c, marker=mk, label=label(b, rho))
        axE.plot(x, [y[0] / v for v in y], ls, color=c, marker=mk, label=label(b, rho))

    axE.axhline(1.0, color="k", ls="--", lw=1, label="ideal (= 1)")

    for ax in (axT, axE):
        ax.set_xscale("log", base=2)
        ax.set_xticks(nps); ax.set_xticklabels(nps)
        ax.set_xlabel("# GPUs (MPI ranks)")
        ax.grid(True, which="both", alpha=.3)
        ax.legend(fontsize=8)

    axT.set_ylabel("solve_z time / step  [s]")
    axT.set_title("z-solve time per step  (ideal: flat)")
    axE.set_ylabel("weak efficiency  Tz(np₀)/Tz(np)")
    axE.set_ylim(0, 1.2)
    axE.set_title("z-solve weak efficiency")

    fig.suptitle(
        f"Weak scaling (z-solve) — {case}/GPU  H200 GPU\n{ZNOTE}",
        fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(OUT_DIR, f"weak_{case}_z.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)

# ── strong: all cases on one axis (wall time) ────────────────────────────────
CASE_COLORS_S = {c: col for c, col in zip(
    strong_cases, ["tab:blue", "tab:orange", "tab:green"])}

fig, ax = plt.subplots(figsize=(8, 6))
nps_all = sorted({r["np"] for r in rows if r["study"] == "strong"})
comm_legend_added = False
for case in strong_cases:
    col = CASE_COLORS_S[case]
    for b in ("pascal", "filtered_v2"):
        s = sorted(
            [r for r in rows if r["study"]=="strong" and r["case"]==case
             and r["backend"]==b and r["rho"]=="025"],
            key=lambda r: r["np"])
        if not s:
            continue
        x = [r["np"]              for r in s]
        y = [r["wall_per_step_s"] for r in s]
        bname = "PaScaL" if b == "pascal" else "Filtered"
        ax.plot(x, y, "-" if b=="pascal" else "--", color=col,
                marker="o" if b=="pascal" else "s",
                label=f"{case}  {bname}")
        # comm overlay: tdma_z_comm, transparent, no marker
        yc = [get_comm("strong", case, r["np"], b, "025") for r in s]
        if all(v is not None for v in yc):
            lbl = "tdma_z_comm" if not comm_legend_added else "_nolegend_"
            ax.plot(x, yc, "-" if b=="pascal" else "--", color=col,
                    alpha=0.3, lw=1.5, label=lbl)
            comm_legend_added = True
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(nps_all); ax.set_xticklabels(nps_all)
ax.set_xlabel("# GPUs (MPI ranks)"); ax.set_ylabel("wall time / step  [s]")
ax.set_title("Strong scaling — wall time  (ρ=0.25, H200 GPU)\n(transparent = tdma_z_comm)",
             fontweight="bold", fontsize=11)
ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=8)
fig.tight_layout()
p = os.path.join(OUT_DIR, "strong_all_wall.png")
fig.savefig(p, dpi=150); plt.close(fig)
print("wrote", p)

# ── weak: all cases on one axis (wall time) ───────────────────────────────────
CASE_COLORS_W = {c: col for c, col in zip(
    weak_cases, ["tab:blue", "tab:orange", "tab:green"])}

fig, ax = plt.subplots(figsize=(8, 6))
nps_w = sorted({r["np"] for r in rows if r["study"] == "weak"})
comm_legend_added = False
for case in weak_cases:
    col = CASE_COLORS_W[case]
    for b in ("pascal", "filtered_v2"):
        s = sorted(
            [r for r in rows if r["study"]=="weak" and r["case"]==case
             and r["backend"]==b and r["rho"]=="025"],
            key=lambda r: r["np"])
        if not s:
            continue
        x = [r["np"]              for r in s]
        y = [r["wall_per_step_s"] for r in s]
        bname = "PaScaL" if b == "pascal" else "Filtered"
        ax.plot(x, y, "-" if b=="pascal" else "--", color=col,
                marker="o" if b=="pascal" else "s",
                label=f"{case}/GPU  {bname}")
        # comm overlay: tdma_z_comm, transparent, no marker
        yc = [get_comm("weak", case, r["np"], b, "025") for r in s]
        if all(v is not None for v in yc):
            lbl = "tdma_z_comm" if not comm_legend_added else "_nolegend_"
            ax.plot(x, yc, "-" if b=="pascal" else "--", color=col,
                    alpha=0.3, lw=1.5, label=lbl)
            comm_legend_added = True
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(nps_w); ax.set_xticklabels(nps_w)
ax.set_xlabel("# GPUs (MPI ranks)"); ax.set_ylabel("wall time / step  [s]")
ax.set_title("Weak scaling — wall time  (ρ=0.25, H200 GPU)\n(transparent = tdma_z_comm)",
             fontweight="bold", fontsize=11)
ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=8)
fig.tight_layout()
p = os.path.join(OUT_DIR, "weak_all_wall.png")
fig.savefig(p, dpi=150); plt.close(fig)
print("wrote", p)

# ── strong: Filtered / PaScaL speedup ratio per case ────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
for rho in ("025", "040"):
    for case in strong_cases:
        sub = sel("strong", case=case)
        data_p = sorted([r for r in sub if r["backend"]=="pascal"      and r["rho"]==rho], key=lambda r: r["np"])
        data_f = sorted([r for r in sub if r["backend"]=="filtered_v2" and r["rho"]==rho], key=lambda r: r["np"])
        if not data_p or not data_f:
            continue
        x = [r["np"] for r in data_p]
        y = [p[METRIC] / f[METRIC] for p, f in zip(data_p, data_f)]
        ls = "-" if rho == "025" else "--"
        ax.plot(x, y, ls, marker="o",
                label=f"{case}  ρ={'0.25' if rho=='025' else '0.40'}")

ax.axhline(1.0, color="k", ls=":", lw=1, label="equal speed")
ax.set_xticks([2, 4, 8]); ax.set_xticklabels([2, 4, 8])
ax.set_xlabel("# GPUs"); ax.set_ylabel("solve_z speed ratio  PaScaL / Filtered")
ax.set_title(
    "How much faster is Filtered in z-solve? (strong scaling, H200)\n>1 means Filtered is faster",
    fontweight="bold", fontsize=10)
ax.grid(True, alpha=.3); ax.legend(fontsize=8)
fig.tight_layout()
p = os.path.join(OUT_DIR, "strong_zratio.png")
fig.savefig(p, dpi=150); plt.close(fig)
print("wrote", p)

# ── refine: solve_z vs total cells (np=2 fixed) ──────────────────────────────
refine = [r for r in rows if r["study"] == "refine"]
fam    = lambda r: "x".join(r["case"].split("x")[:2])
fams   = sorted({fam(r) for r in refine}, key=lambda f: int(f.split("x")[0]))

# individual per-family plots (kept for reference)
for fm in fams:
    sub = [r for r in refine if fam(r) == fm]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for b, rho in SERIES:
        s = sorted(
            [r for r in sub if r["backend"] == b and r["rho"] == rho],
            key=lambda r: r["cells"])
        if not s:
            continue
        x = [r["cells"]  for r in s]
        y = [r[METRIC]   for r in s]
        c, ls, mk = style(b, rho)
        ax.plot(x, y, ls, color=c, marker=mk, label=label(b, rho))
    base = sorted(
        [r for r in sub if r["backend"] == "pascal" and r["rho"] == "025"],
        key=lambda r: r["cells"])
    if base:
        c0, t0 = base[0]["cells"], base[0][METRIC]
        xi = np.array([r["cells"] for r in base], float)
        ax.plot(xi, t0 * xi / c0, "k:", lw=1, label="O(N) ideal")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("total grid points (nx·ny·nz)")
    ax.set_ylabel("solve_z time / step  [s]")
    ax.set_title(
        f"Refinement (z-solve) — {fm}×nz  (np=2)  H200 GPU\n{ZNOTE}",
        fontweight="bold", fontsize=10)
    ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, f"refine_{fm}_z.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)

# combined: all three families on one axis, distinguished by color (family) + linestyle (backend)
FAM_COLORS = {"64x64": "tab:blue", "128x128": "tab:orange", "256x256": "tab:green"}
BACKEND_LS  = {"pascal": "-", "filtered_v2": "--"}
BACKEND_MK  = {"pascal": "o", "filtered_v2": "s"}

fig, ax = plt.subplots(figsize=(8, 6))
for fm in fams:
    sub = [r for r in refine if fam(r) == fm]
    col = FAM_COLORS[fm]
    for b in ("pascal", "filtered_v2"):
        s = sorted(
            [r for r in sub if r["backend"] == b and r["rho"] == "025"],
            key=lambda r: r["cells"])
        if not s:
            continue
        x = [r["cells"] for r in s]
        y = [r[METRIC]  for r in s]
        bname = "PaScaL" if b == "pascal" else "Filtered"
        ax.plot(x, y, BACKEND_LS[b], color=col, marker=BACKEND_MK[b],
                label=f"{fm}×nz  {bname}")

# O(N) reference anchored to the smallest PaScaL point across all families
all_base = sorted(
    [r for r in refine if r["backend"] == "pascal" and r["rho"] == "025"],
    key=lambda r: r["cells"])
c0, t0 = all_base[0]["cells"], all_base[0][METRIC]
xi = np.array([r["cells"] for r in all_base], float)
ax.plot(xi, t0 * xi / c0, "k:", lw=1, label="O(N) ideal")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("total grid points (nx·ny·nz)")
ax.set_ylabel("solve_z time / step  [s]")
ax.set_title(
    f"Refinement study — z-solve only  (np=2, ρ=0.25, H200 GPU)\n{ZNOTE}",
    fontweight="bold", fontsize=10)
ax.grid(True, which="both", alpha=.3)
ax.legend(fontsize=8)
fig.tight_layout()
p = os.path.join(OUT_DIR, "refine_all_z.png")
fig.savefig(p, dpi=150); plt.close(fig)
print("wrote", p)

# refine: PaScaL/Filtered solve_z ratio vs cells (all families, one figure)
fig, ax = plt.subplots(figsize=(7.5, 5.5))
for rho in ("025", "040"):
    pts = defaultdict(dict)
    for r in refine:
        if r["rho"] == rho:
            pts[r["case"]][r["backend"]] = r[METRIC]
            pts[r["case"]]["cells"]      = r["cells"]
    items = sorted(pts.values(), key=lambda d: d["cells"])
    x = [d["cells"] for d in items if "pascal" in d and "filtered_v2" in d]
    y = [d["pascal"] / d["filtered_v2"] for d in items if "pascal" in d and "filtered_v2" in d]
    ax.plot(x, y, "-o" if rho == "025" else "--s", color="tab:green",
            label=f"ρ={'0.25' if rho=='025' else '0.40'}")
ax.axhline(1.0, color="k", ls=":", lw=1, label="equal speed")
ax.set_xscale("log")
ax.set_xlabel("total grid points (nx·ny·nz)")
ax.set_ylabel("solve_z speed ratio  PaScaL / Filtered")
ax.set_title(
    "How much faster is Filtered in z-solve? (np=2, refine, H200)\n>1 means Filtered is faster",
    fontweight="bold", fontsize=10)
ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=9)
fig.tight_layout()
p = os.path.join(OUT_DIR, "refine_zratio.png")
fig.savefig(p, dpi=150); plt.close(fig)
print("wrote", p)

# ── bar charts: compute vs comm breakdown (strong & weak) ───────────────────
NPS       = [2, 4, 8]
BAR_W     = 0.35
B_NAMES   = ["pascal", "filtered_v2"]
B_COLORS  = {"pascal": "tab:blue", "filtered_v2": "tab:red"}
B_LABELS  = {"pascal": "PaScaL",   "filtered_v2": "Filtered"}

def bar_breakdown(study, cases, out_fname):
    ncases = len(cases)
    fig, axes = plt.subplots(1, ncases, figsize=(5 * ncases, 5.5))
    if ncases == 1:
        axes = [axes]

    for ax, case in zip(axes, cases):
        for bi, b in enumerate(B_NAMES):
            xs, comp_vals, comm_vals = [], [], []
            for ni, np_val in enumerate(NPS):
                rec = next((r for r in rows
                            if r["study"] == study and r["case"] == case
                            and r["np"] == np_val and r["backend"] == b
                            and r["rho"] == "025"), None)
                if rec is None:
                    continue
                wall   = rec["wall_per_step_s"]
                comm_t = get_comm(study, case, np_val, b, "025", "tdma_z_comm") or 0.0
                xs.append(ni + bi * BAR_W - BAR_W / 2)
                comp_vals.append(wall - comm_t)
                comm_vals.append(comm_t)

            col = B_COLORS[b]
            lbl = B_LABELS[b]
            ax.bar(xs, comp_vals, BAR_W, color=col, alpha=0.8,
                   label=f"{lbl} compute")
            ax.bar(xs, comm_vals, BAR_W, bottom=comp_vals, color=col,
                   alpha=0.35, hatch="//", label=f"{lbl} comm")

        ax.set_xticks(range(len(NPS)))
        ax.set_xticklabels([f"np={n}" for n in NPS])
        ax.set_title(case, fontweight="bold", fontsize=10)
        ax.set_ylabel("wall time / step  [s]")
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)

    study_label = "Strong" if study == "strong" else "Weak"
    fig.suptitle(
        f"{study_label} scaling — compute vs comm breakdown  (ρ=0.25, H200 GPU)\n"
        "solid = compute,  hatch = tdma_z_comm",
        fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(OUT_DIR, out_fname)
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)

bar_breakdown("strong", strong_cases, "strong_bar_breakdown.png")
bar_breakdown("weak",   weak_cases,   "weak_bar_breakdown.png")

# ── bar chart: refine compute vs comm — one subplot per family ───────────────
refine_rows = [r for r in rows if r["study"] == "refine"]
fam_key     = lambda r: "x".join(r["case"].split("x")[:2])
refine_fams = sorted({fam_key(r) for r in refine_rows}, key=lambda f: int(f.split("x")[0]))

fig, axes = plt.subplots(1, len(refine_fams), figsize=(6 * len(refine_fams), 5.5))
if len(refine_fams) == 1:
    axes = [axes]

for ax, fm in zip(axes, refine_fams):
    sub = sorted([r for r in refine_rows if fam_key(r) == fm],
                 key=lambda r: r["cells"])
    cases_fm = sorted({r["case"] for r in sub},
                      key=lambda c: next(r["cells"] for r in sub if r["case"] == c))
    n_cases = len(cases_fm)

    for bi, b in enumerate(B_NAMES):
        xs, comp_vals, comm_vals = [], [], []
        for ni, case in enumerate(cases_fm):
            rec = next((r for r in sub if r["case"] == case and r["backend"] == b
                        and r["rho"] == "025"), None)
            if rec is None:
                continue
            wall   = rec["wall_per_step_s"]
            comm_t = get_comm("refine", case, 2, b, "025", "tdma_z_comm") or 0.0
            xs.append(ni + bi * BAR_W - BAR_W / 2)
            comp_vals.append(wall - comm_t)
            comm_vals.append(comm_t)

        col = B_COLORS[b]
        lbl = B_LABELS[b]
        ax.bar(xs, comp_vals, BAR_W, color=col, alpha=0.8,  label=f"{lbl} compute")
        ax.bar(xs, comm_vals, BAR_W, bottom=comp_vals, color=col,
               alpha=0.35, hatch="//", label=f"{lbl} comm")

    # x-tick: nz 값만 표시
    ax.set_xticks(range(n_cases))
    ax.set_xticklabels([c.split("x")[2] for c in cases_fm], fontsize=8)
    ax.set_xlabel("nz")
    ax.set_title(f"{fm}×nz  (np=2)", fontweight="bold", fontsize=10)
    ax.set_ylabel("wall time / step  [s]")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)

fig.suptitle(
    "Refinement — compute vs comm breakdown  (ρ=0.25, np=2, H200 GPU)\n"
    "solid = compute,  hatch = tdma_z_comm",
    fontweight="bold", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
p = os.path.join(OUT_DIR, "refine_bar_breakdown.png")
fig.savefig(p, dpi=150); plt.close(fig)
print("wrote", p)

print("done ->", OUT_DIR)
