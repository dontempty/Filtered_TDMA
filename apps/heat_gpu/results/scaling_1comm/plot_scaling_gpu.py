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
OUT_DIR  = "/scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/results/scaling_1comm(image)/h200"
os.makedirs(OUT_DIR, exist_ok=True)

EVENTS   = ["rhs", "solve_x", "solve_y", "solve_z", "comm", "etc"]

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

print("done ->", OUT_DIR)
