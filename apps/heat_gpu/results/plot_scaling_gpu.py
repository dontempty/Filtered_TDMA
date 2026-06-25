#!/usr/bin/env python3
"""
GPU scaling plots — z-solve only (PaScaL vs Filtered_v2).
Reads: scaling_gpu/<gpu>/timing_*.csv   (<gpu> = v100|a100|... hardware subfolder)
Writes: image_gpu/<gpu>/

Usage:
    python3 plot_scaling_gpu.py [gpu]     # e.g. v100, a100 (default: v100)
Outputs go to image_gpu/<gpu>/. Pass the bare results root (legacy backup files
in scaling_gpu/ root) with `python3 plot_scaling_gpu.py .`.
"""
import os, re, glob, csv, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GPU  = sys.argv[1] if len(sys.argv) > 1 else "v100"
ROOT = "/scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/results"
# prefer scaling_1comm/<gpu>/ if it exists, fall back to scaling_gpu/<gpu>/
_src1 = os.path.join(ROOT, "scaling_1comm", GPU)
_src0 = os.path.join(ROOT, "scaling_gpu", GPU)
SRC  = _src1 if (GPU != "." and os.path.isdir(_src1)) else (
       os.path.join(ROOT, "scaling_gpu") if GPU == "." else _src0)
OUT  = os.path.join(ROOT, "image_gpu") if GPU == "." else os.path.join(ROOT, "image_gpu", GPU)
os.makedirs(OUT, exist_ok=True)
print(f"[plot] GPU={GPU}  SRC={SRC}  OUT={OUT}")

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
    total = sum(ev_mean.values())
    cells = grid[0]*grid[1]*grid[2] if grid else None
    return dict(study=study, case=case, np=npv, backend=backend, rho=rho,
                grid="x".join(map(str, grid)) if grid else "", cells=cells,
                total=total, ev_mean=ev_mean)

records = [r for r in (parse_file(p) for p in sorted(glob.glob(os.path.join(SRC, "*.csv")))) if r]
print(f"parsed {len(records)} csv files")

# metrics.csv 저장
tbl = os.path.join(OUT, "metrics.csv")
with open(tbl, "w") as f:
    f.write("study,case,np,backend,rho,grid,cells,wall_per_step_s," + ",".join(EVENTS) + "\n")
    for r in sorted(records, key=lambda r: (r["study"], r["case"], r["np"], r["backend"], r["rho"])):
        f.write(f'{r["study"]},{r["case"]},{r["np"]},{r["backend"]},{r["rho"]},'
                f'{r["grid"]},{r["cells"]},{r["total"]:.6e},'
                + ",".join(f'{r["ev_mean"][e]:.6e}' for e in EVENTS) + "\n")

# ---- styling ----------------------------------------------------------------
def style(b, rho):
    return ("tab:blue" if b == "pascal" else "tab:red",
            "-" if rho == "025" else "--",
            "o" if rho == "025" else "s")
def label(b, rho):
    return f'{"PaScaL" if b=="pascal" else "Filtered"}  ρ={"0.25" if rho=="025" else "0.40"}'
SERIES = [("pascal","025"),("pascal","040"),("filtered_v2","025"),("filtered_v2","040")]

ZNOTE = "z-direction (distributed) solve only"

def sel(study, **kw):
    return [r for r in records if r["study"] == study and all(r[k] == v for k, v in kw.items())]

# ---- STRONG -----------------------------------------------------------------
for case in sorted({r["case"] for r in records if r["study"]=="strong"}, key=lambda c: int(c.split("x")[0])):
    sub = sel("strong", case=case)
    grid = sub[0]["grid"]
    nps = sorted({r["np"] for r in sub})
    fig, (axT, axS) = plt.subplots(1, 2, figsize=(12, 5))
    for b, rho in SERIES:
        s = sorted([r for r in sub if r["backend"]==b and r["rho"]==rho], key=lambda r: r["np"])
        if not s: continue
        x = [r["np"] for r in s]; y = [r["ev_mean"]["solve_z"] for r in s]
        c, ls, mk = style(b, rho)
        axT.plot(x, y, ls, color=c, marker=mk, label=label(b, rho))
        axS.plot(x, [y[0]/v for v in y], ls, color=c, marker=mk, label=label(b, rho))
    base = sorted([r for r in sub if r["backend"]=="pascal" and r["rho"]=="025"], key=lambda r: r["np"])
    n0, t0 = base[0]["np"], base[0]["ev_mean"]["solve_z"]
    xi = np.array(nps, float)
    axT.plot(xi, t0*n0/xi, "k:", lw=1, label="ideal (∝1/np)")
    axS.plot(xi, xi/n0, "k--", lw=1, label="ideal linear")
    axT.set_xscale("log", base=2); axT.set_yscale("log")
    axT.set_xticks(nps); axT.set_xticklabels(nps)
    axT.set_xlabel("# MPI processes"); axT.set_ylabel("solve_z time / step  [s]")
    axT.set_title("z-solve time per step"); axT.grid(True, which="both", alpha=.3); axT.legend(fontsize=8)
    axS.set_xticks(nps); axS.set_xticklabels(nps)
    axS.set_xlabel("# MPI processes"); axS.set_ylabel(f"speedup  Tz({n0})/Tz(np)")
    axS.set_title("z-solve strong-scaling speedup"); axS.grid(True, alpha=.3); axS.legend(fontsize=8)
    fig.suptitle(f"Strong scaling (z-solve only) — {case}  (grid {grid})\n{ZNOTE}", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(OUT, f"strong_{case}_z.png"); fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)

# ---- WEAK -------------------------------------------------------------------
for case in sorted({r["case"] for r in records if r["study"]=="weak"}, key=lambda c: int(re.sub(r"\D","",c))):
    sub = sel("weak", case=case)
    nps = sorted({r["np"] for r in sub})
    fig, (axT, axE) = plt.subplots(1, 2, figsize=(12, 5))
    for b, rho in SERIES:
        s = sorted([r for r in sub if r["backend"]==b and r["rho"]==rho], key=lambda r: r["np"])
        if not s: continue
        x = [r["np"] for r in s]; y = [r["ev_mean"]["solve_z"] for r in s]
        c, ls, mk = style(b, rho)
        axT.plot(x, y, ls, color=c, marker=mk, label=label(b, rho))
        axE.plot(x, [y[0]/v for v in y], ls, color=c, marker=mk, label=label(b, rho))
    axE.axhline(1.0, color="k", ls="--", lw=1, label="ideal (=1)")
    for ax in (axT, axE):
        ax.set_xscale("log", base=2); ax.set_xticks(nps); ax.set_xticklabels(nps)
        ax.set_xlabel("# MPI processes"); ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=8)
    axT.set_ylabel("solve_z time / step  [s]"); axT.set_title("z-solve time per step (ideal: flat)")
    axE.set_ylabel("weak efficiency  Tz(np₀)/Tz(np)"); axE.set_ylim(0, 1.2); axE.set_title("z-solve weak efficiency")
    fig.suptitle(f"Weak scaling (z-solve only) — {case}/GPU\n{ZNOTE}", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(OUT, f"weak_{case}_z.png"); fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)

# ---- REFINE -----------------------------------------------------------------
refine = [r for r in records if r["study"]=="refine"]
fam = lambda r: "x".join(r["case"].split("x")[:2])
fams = sorted({fam(r) for r in refine}, key=lambda f: int(f.split("x")[0]))
for fm in fams:
    sub = [r for r in refine if fam(r)==fm]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for b, rho in SERIES:
        s = sorted([r for r in sub if r["backend"]==b and r["rho"]==rho], key=lambda r: r["cells"])
        if not s: continue
        x = [r["cells"] for r in s]; y = [r["ev_mean"]["solve_z"] for r in s]
        c, ls, mk = style(b, rho)
        ax.plot(x, y, ls, color=c, marker=mk, label=label(b, rho))
    base = sorted([r for r in sub if r["backend"]=="pascal" and r["rho"]=="025"], key=lambda r: r["cells"])
    c0, t0 = base[0]["cells"], base[0]["ev_mean"]["solve_z"]
    xi = np.array([r["cells"] for r in base], float)
    ax.plot(xi, t0*xi/c0, "k:", lw=1, label="O(N) ideal")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("total grid points (nx·ny·nz)"); ax.set_ylabel("solve_z time / step  [s]")
    ax.set_title(f"Refinement (z-solve only) — {fm}×nz  (np=2)\n{ZNOTE}", fontweight="bold", fontsize=10)
    ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, f"refine_{fm}_z.png"); fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)

# refine: all families on a single axes
FM_MARKERS = {"64x64": "o", "128x128": "s", "256x256": "^"}
fig, ax = plt.subplots(figsize=(8, 6))
for b, rho in SERIES:
    c, ls, _ = style(b, rho)
    first = True
    for fm in fams:
        sub = [r for r in refine if fam(r)==fm]
        s = sorted([r for r in sub if r["backend"]==b and r["rho"]==rho], key=lambda r: r["cells"])
        if not s: continue
        x = [r["cells"] for r in s]; y = [r["ev_mean"]["solve_z"] for r in s]
        mk = FM_MARKERS.get(fm, "o")
        ax.plot(x, y, ls, color=c, marker=mk,
                label=(label(b, rho) if first else "_nolegend_"))
        first = False
# O(N) reference from smallest pascal rho025 point across all families
all_base = sorted([r for r in refine if r["backend"]=="pascal" and r["rho"]=="025"], key=lambda r: r["cells"])
c0, t0 = all_base[0]["cells"], all_base[0]["ev_mean"]["solve_z"]
xi = np.array([r["cells"] for r in all_base], float)
ax.plot(xi, t0*xi/c0, "k:", lw=1, label="O(N) ideal")
# marker legend for families
for fm, mk in FM_MARKERS.items():
    ax.plot([], [], color="gray", marker=mk, ls="none", label=f"{fm}×nz")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("total grid points (nx·ny·nz)"); ax.set_ylabel("solve_z time / step  [s]")
ax.set_title(f"Refinement (z-solve only, np=2) — all families\n{ZNOTE}", fontweight="bold", fontsize=10)
ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=8)
fig.tight_layout()
p = os.path.join(OUT, "refine_all_z.png"); fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)

# refine: PaScaL/Filtered ratio
fig, ax = plt.subplots(figsize=(7.5, 5.5))
for rho in ("025", "040"):
    pts = defaultdict(dict)
    for r in refine:
        if r["rho"] == rho:
            pts[r["case"]][r["backend"]] = r["ev_mean"]["solve_z"]
            pts[r["case"]]["cells"] = r["cells"]
    items = sorted(pts.values(), key=lambda d: d["cells"])
    x = [d["cells"] for d in items if "pascal" in d and "filtered_v2" in d]
    y = [d["pascal"]/d["filtered_v2"] for d in items if "pascal" in d and "filtered_v2" in d]
    ax.plot(x, y, ("-o" if rho=="025" else "--s"), color="tab:green",
            label=f"ρ={'0.25' if rho=='025' else '0.40'}")
ax.axhline(1.0, color="k", ls=":", lw=1, label="equal")
ax.set_xscale("log")
ax.set_xlabel("total grid points (nx·ny·nz)")
ax.set_ylabel("solve_z speed ratio  PaScaL / Filtered")
ax.set_title("How much faster is Filtered? (np=2, refine)\n>1 means Filtered is faster",
             fontweight="bold", fontsize=10)
ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=9)
fig.tight_layout()
p = os.path.join(OUT, "refine_zratio.png"); fig.savefig(p, dpi=150); plt.close(fig); print("wrote", p)

print(f"done -> {OUT}")
