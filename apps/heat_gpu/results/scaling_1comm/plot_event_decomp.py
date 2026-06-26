#!/usr/bin/env python3
"""
Per-event decomposition of the GPU time step (NOT stacked).
One subplot PER EVENT; each shows that event's wall time/step vs the scaling
axis (np for strong/weak, total cells for refine), PaScaL vs Filtered.

Events: rhs | solve_x | solve_y | tdma_z_gpu | z_pack | tdma_z_comm | comm | etc
  z_pack = solve_z - tdma_z_gpu - tdma_z_comm
Source : scaling_1comm/h200 timing CSVs (12-event binary).
Output : scaling_1comm/gpu_image/{strong,weak,refine}_events.png   (rho=0.25)
"""
import os, re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "/scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/results/scaling_1comm/h200"
OUT_DIR  = "/scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/results/scaling_1comm/gpu_image"
os.makedirs(OUT_DIR, exist_ok=True)

RAW = ["rhs", "solve_x", "solve_y", "solve_z", "tdma_z_gpu", "tdma_z_comm", "comm", "etc"]
EVENTS = ["rhs", "solve_x", "solve_y", "tdma_z_gpu", "z_pack", "tdma_z_comm", "comm", "etc"]

def parse_dir(dirpath):
    recs = []
    for fn in sorted(os.listdir(dirpath)):
        m = re.match(r"timing_(strong|weak|refine)_(.+)_np(\d+)_(pascal|filtered_v2)_rho(\d+)\.csv", fn)
        if not m:
            continue
        study, case, nps, be, rho = m.groups()
        timing = defaultdict(list); cells = 0
        for line in open(os.path.join(dirpath, fn)):
            line = line.strip()
            if line.startswith("#"):
                g = re.search(r"grid=(\d+)x(\d+)x(\d+)", line)
                if g: cells = int(g[1]) * int(g[2]) * int(g[3])
                continue
            if line.startswith("rank") or "," not in line:
                continue
            p = line.split(",")
            if len(p) < 4: continue
            r, st, e, v = int(p[0]), int(p[1]), p[2].strip(), float(p[3])
            if st > 1: timing[(r, e)].append(v)
        ranks = sorted({r for (r, e) in timing})
        n = len(timing[(ranks[0], "solve_z")]) if ranks else 0
        if n == 0: continue
        ev = {e: float(np.mean([max(timing[(r, e)][i] if timing[(r, e)] else 0.0 for r in ranks)
                                for i in range(n)])) for e in RAW}
        ev["z_pack"] = max(ev["solve_z"] - ev["tdma_z_gpu"] - ev["tdma_z_comm"], 0.0)
        recs.append(dict(study=study, case=case, np=int(nps), backend=be, rho=rho, cells=cells, **ev))
    return recs

R = parse_dir(DATA_DIR)
print(f"parsed {len(R)} records")

CASE_COLORS = ["tab:blue", "tab:orange", "tab:green"]
def decomp_figure(study, cases, xkey, xlabel, xlog, title, fname, logx=False):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.ravel()
    colors = {c: CASE_COLORS[i % 3] for i, c in enumerate(cases)}
    for ax, ev in zip(axes, EVENTS):
        for c in cases:
            for be in ("pascal", "filtered_v2"):
                s = sorted([r for r in R if r["study"] == study and r["case"] == c
                            and r["backend"] == be and r["rho"] == "025"],
                           key=lambda r: r[xkey])
                if not s: continue
                x = [r[xkey] for r in s]; y = [r[ev] * 1e6 for r in s]   # µs
                ax.plot(x, y, ("-" if be == "pascal" else "--"),
                        color=colors[c], marker=("o" if be == "pascal" else "s"),
                        ms=4, label=f"{c} {'P' if be=='pascal' else 'F'}")
        if xlog: ax.set_xscale("log", base=2)
        if logx: ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(ev, fontweight="bold", fontsize=11)
        ax.set_xlabel(xlabel); ax.set_ylabel("time/step [µs]")
        ax.grid(True, which="both", alpha=.3)
    axes[0].legend(fontsize=7, ncol=2, title="case  (P=PaScaL solid, F=Filtered dashed)")
    fig.suptitle(title, fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(OUT_DIR, fname); fig.savefig(p, dpi=140); plt.close(fig)
    print("wrote", p)

# strong: x = np
strong_cases = sorted({r["case"] for r in R if r["study"]=="strong"}, key=lambda c:int(c.split("x")[0]))
decomp_figure("strong", strong_cases, "np", "# GPUs", True,
              "Strong scaling — per-event decomposition (ρ=0.25, H200)\nsolid=PaScaL, dashed=Filtered",
              "strong_events.png")
# weak: x = np
weak_cases = sorted({r["case"] for r in R if r["study"]=="weak"}, key=lambda c:int(re.sub(r"\D","",c)))
decomp_figure("weak", weak_cases, "np", "# GPUs", True,
              "Weak scaling — per-event decomposition (ρ=0.25, H200)\nsolid=PaScaL, dashed=Filtered",
              "weak_events.png")
# refine: x = cells  (np=2 fixed); group by family
refine_cases = sorted({r["case"] for r in R if r["study"]=="refine"}, key=lambda c:int(c.split("x")[0]))
fams = sorted({"x".join(c.split("x")[:2]) for c in refine_cases}, key=lambda f:int(f.split("x")[0]))
# use family as the "case" dimension; cells as x
def refine_figure():
    fig, axes = plt.subplots(2, 4, figsize=(20, 9)); axes = axes.ravel()
    colors = {f: CASE_COLORS[i % 3] for i, f in enumerate(fams)}
    for ax, ev in zip(axes, EVENTS):
        for f in fams:
            for be in ("pascal", "filtered_v2"):
                s = sorted([r for r in R if r["study"]=="refine"
                            and "x".join(r["case"].split("x")[:2])==f
                            and r["backend"]==be and r["rho"]=="025"], key=lambda r:r["cells"])
                if not s: continue
                x=[r["cells"] for r in s]; y=[r[ev]*1e6 for r in s]
                ax.plot(x, y, ("-" if be=="pascal" else "--"), color=colors[f],
                        marker=("o" if be=="pascal" else "s"), ms=4,
                        label=f"{f} {'P' if be=='pascal' else 'F'}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(ev, fontweight="bold", fontsize=11)
        ax.set_xlabel("total cells"); ax.set_ylabel("time/step [µs]")
        ax.grid(True, which="both", alpha=.3)
    axes[0].legend(fontsize=7, ncol=2, title="family (P/F)")
    fig.suptitle("Refinement (np=2) — per-event decomposition (ρ=0.25, H200)\nsolid=PaScaL, dashed=Filtered",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.96])
    p=os.path.join(OUT_DIR,"refine_events.png"); fig.savefig(p,dpi=140); plt.close(fig); print("wrote",p)
refine_figure()
print("done ->", OUT_DIR)
