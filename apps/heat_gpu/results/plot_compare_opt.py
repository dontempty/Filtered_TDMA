#!/usr/bin/env python3
"""
Before/After optimization comparison plots.
OLD = scaling_gpu/   (pre-optimization)
NEW = scaling_gpu_opt/ (post-optimization)
"""
import os, re, glob, csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OLD_SRC = "/scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/results/scaling_gpu"
NEW_SRC = "/scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/results/scaling_gpu_opt"
OUT     = "/scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/results/image_gpu_opt"
os.makedirs(OUT, exist_ok=True)

EVENTS   = ["rhs", "solve_x", "solve_y", "solve_z", "comm"]
FNAME_RE = re.compile(r"timing_(strong|weak|refine)_(.+)_np(\d+)_(pascal|filtered_v2)_rho(\d+)\.csv$")
GRID_RE  = re.compile(r"grid=(\d+)x(\d+)x(\d+)")

def parse_file(path):
    fn = os.path.basename(path)
    m = FNAME_RE.match(fn)
    if not m:
        return None
    study, case, npv, backend, rho = m.group(1), m.group(2), int(m.group(3)), m.group(4), m.group(5)
    agg = {}
    with open(path) as f:
        header = f.readline()
        gm = GRID_RE.search(header)
        grid = tuple(int(x) for x in gm.groups()) if gm else None
        f.readline()
        for line in f:
            p = line.strip().split(",")
            if len(p) < 4:
                continue
            t = int(p[1]); ev = p[2].strip(); val = float(p[3])
            k = (t, ev)
            if k not in agg or val > agg[k]:
                agg[k] = val
    byev = defaultdict(list)
    for (t, ev), v in agg.items():
        byev[ev].append(v)
    ev_mean = {e: float(np.mean(byev[e])) if byev[e] else 0.0 for e in EVENTS}
    cells = grid[0]*grid[1]*grid[2] if grid else None
    return dict(study=study, case=case, np=npv, backend=backend, rho=rho,
                grid="x".join(map(str, grid)) if grid else "", cells=cells,
                ev_mean=ev_mean)

def load(src):
    return [r for r in (parse_file(p) for p in sorted(glob.glob(os.path.join(src, "*.csv")))) if r]

old = load(OLD_SRC)
new = load(NEW_SRC)
print(f"OLD: {len(old)} records,  NEW: {len(new)} records")

# Build lookup: (study, case, np, backend, rho) -> solve_z
def idx(recs):
    return {(r["study"],r["case"],r["np"],r["backend"],r["rho"]): r for r in recs}

old_idx = idx(old)
new_idx = idx(new)

# ---- speedup summary table --------------------------------------------------
print("\n=== solve_z speedup (NEW/OLD, >1 = faster) ===")
print(f"{'study':8} {'case':22} {'np':>3} {'backend':12} {'rho':>5}  {'old_ms':>9} {'new_ms':>9} {'speedup':>8}")
for k, o in sorted(old_idx.items()):
    n = new_idx.get(k)
    if n is None:
        continue
    ot = o["ev_mean"]["solve_z"] * 1e3
    nt = n["ev_mean"]["solve_z"] * 1e3
    sp = ot / nt if nt > 0 else float("nan")
    study, case, npv, backend, rho = k
    print(f"{study:8} {case:22} {npv:>3} {backend:12} {rho:>5}  {ot:9.3f} {nt:9.3f} {sp:8.3f}x")

# ---- refine: ratio plot before/after per backend ----------------------------
fam = lambda r: "x".join(r["case"].split("x")[:2])

for backend in ("pascal", "filtered_v2"):
    bname = "PaScaL" if backend == "pascal" else "Filtered v2"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, rho in zip(axes, ("025", "040")):
        old_r = sorted([r for r in old if r["study"]=="refine" and r["backend"]==backend and r["rho"]==rho],
                       key=lambda r: r["cells"])
        new_r = sorted([r for r in new if r["study"]=="refine" and r["backend"]==backend and r["rho"]==rho],
                       key=lambda r: r["cells"])

        families = sorted({fam(r) for r in old_r}, key=lambda f: int(f.split("x")[0]))
        FM_MARKERS = {"64x64": "o", "128x128": "s", "256x256": "^"}

        for fm in families:
            mk = FM_MARKERS.get(fm, "o")
            os_ = sorted([r for r in old_r if fam(r)==fm], key=lambda r: r["cells"])
            ns_ = sorted([r for r in new_r if fam(r)==fm], key=lambda r: r["cells"])
            if not os_ or not ns_:
                continue
            x = [r["cells"] for r in os_]
            yo = [r["ev_mean"]["solve_z"]*1e3 for r in os_]
            yn = [r["ev_mean"]["solve_z"]*1e3 for r in ns_]
            ax.plot(x, yo, "b-", marker=mk, lw=1.2, label=f"OLD {fm}" if rho=="025" else "_")
            ax.plot(x, yn, "r-", marker=mk, lw=1.2, label=f"NEW {fm}" if rho=="025" else "_")

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("total grid points"); ax.set_ylabel("solve_z  [ms/step]")
        ax.set_title(f"ρ={'0.25' if rho=='025' else '0.40'}")
        ax.grid(True, which="both", alpha=.3)

    # unified legend
    from matplotlib.lines import Line2D
    handles = ([Line2D([0],[0],color="b",lw=2,label="OLD"),
                Line2D([0],[0],color="r",lw=2,label="NEW (opt)")] +
               [Line2D([0],[0],color="gray",marker=mk,ls="none",label=f"{fm}")
                for fm, mk in FM_MARKERS.items()])
    axes[1].legend(handles=handles, fontsize=8, loc="lower right")
    fig.suptitle(f"{bname}: before vs after optimization (refine, np=2)", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(OUT, f"compare_refine_{backend}.png")
    fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)

# ---- speedup ratio plot (NEW faster by X%) per backend ----------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, backend in zip(axes, ("pascal", "filtered_v2")):
    bname = "PaScaL" if backend == "pascal" else "Filtered v2"
    for rho, mk, ls in [("025","o","-"), ("040","s","--")]:
        old_r = sorted([r for r in old if r["study"]=="refine" and r["backend"]==backend and r["rho"]==rho],
                       key=lambda r: r["cells"])
        new_r = sorted([r for r in new if r["study"]=="refine" and r["backend"]==backend and r["rho"]==rho],
                       key=lambda r: r["cells"])
        old_c = {r["case"]: r for r in old_r}
        new_c = {r["case"]: r for r in new_r}
        common = sorted(set(old_c) & set(new_c), key=lambda c: old_c[c]["cells"])
        x = [old_c[c]["cells"] for c in common]
        y = [old_c[c]["ev_mean"]["solve_z"] / new_c[c]["ev_mean"]["solve_z"] for c in common]
        ax.plot(x, y, f"{ls}", marker=mk, label=f"ρ={'0.25' if rho=='025' else '0.40'}")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("total grid points"); ax.set_ylabel("OLD / NEW (>1 = faster)")
    ax.set_title(f"{bname} speedup from optimization")
    ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=9)
fig.suptitle("solve_z speedup: optimized vs baseline (refine, np=2)", fontweight="bold")
fig.tight_layout()
p = os.path.join(OUT, "compare_speedup_refine.png")
fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)

# ---- strong: before/after for each grid size --------------------------------
for case in sorted({r["case"] for r in old if r["study"]=="strong"}, key=lambda c: int(c.split("x")[0])):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, backend in zip(axes, ("pascal", "filtered_v2")):
        bname = "PaScaL" if backend == "pascal" else "Filtered v2"
        for rho, mk, ls in [("025","o","-"), ("040","s","--")]:
            os_ = sorted([r for r in old if r["study"]=="strong" and r["case"]==case
                          and r["backend"]==backend and r["rho"]==rho], key=lambda r: r["np"])
            ns_ = sorted([r for r in new if r["study"]=="strong" and r["case"]==case
                          and r["backend"]==backend and r["rho"]==rho], key=lambda r: r["np"])
            if not os_ or not ns_:
                continue
            x = [r["np"] for r in os_]
            yo = [r["ev_mean"]["solve_z"]*1e3 for r in os_]
            yn = [r["ev_mean"]["solve_z"]*1e3 for r in ns_]
            ax.plot(x, yo, f"b{ls}", marker=mk, label=f"OLD ρ={'0.25' if rho=='025' else '0.40'}")
            ax.plot(x, yn, f"r{ls}", marker=mk, label=f"NEW ρ={'0.25' if rho=='025' else '0.40'}")
        nps = sorted({r["np"] for r in old if r["study"]=="strong" and r["case"]==case})
        ax.set_xticks(nps); ax.set_xticklabels(nps)
        ax.set_yscale("log")
        ax.set_xlabel("# MPI processes"); ax.set_ylabel("solve_z  [ms/step]")
        ax.set_title(bname); ax.grid(True, alpha=.3); ax.legend(fontsize=8)
    fig.suptitle(f"Strong scaling before/after opt — {case}", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(OUT, f"compare_strong_{case}.png")
    fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)

print(f"\ndone -> {OUT}")
