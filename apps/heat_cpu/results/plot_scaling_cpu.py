#!/usr/bin/env python3
"""
Scaling plots for heat_cpu (PaScaL vs Filtered).

Reads long-format timing CSVs in results/scaling_cpu/:
   timing_<study>_<case>_np<NP>_<backend>_rho<RR>.csv
   header: # grid=AxBxC, np=N (px,py,pz), dt=..., Nt=..., solver_kind=...
   rows  : rank,t_step,event,time_sec   (events: rhs,solve_x,solve_y,solve_z,comm)

Wall time per step = sum_event( mean_t( max_rank(event_time) ) )
  -> critical-path per event (assumes a barrier between events), summed;
     this makes the stacked event breakdown add up to the total.

Studies:
  strong : grid fixed, np = 2,4,8        -> speedup vs np
  weak   : per-proc cube fixed, np 2,4,8 -> weak efficiency vs np
  refine : np=2 fixed, grid refined in z -> time vs problem size (#cells)

PaScaL and Filtered are drawn on the same axes; both rho (0.25, 0.40) shown.
One figure per case (per cross-section family for refine).
"""
import os, re, glob
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "/scratch/x3319a05/Filtered_TDMA/apps/heat_cpu/results/scaling_cpu"
OUT = "/scratch/x3319a05/Filtered_TDMA/apps/heat_cpu/results/cpu_image"
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
    agg = {}  # (t_step, event) -> max over ranks
    with open(path) as f:
        header = f.readline()
        gm = GRID_RE.search(header)
        grid = tuple(int(x) for x in gm.groups()) if gm else None
        f.readline()  # column header
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
    ev_mean = {e: (float(np.mean(byev[e])) if byev[e] else 0.0) for e in EVENTS}
    total = sum(ev_mean.values())
    cells = grid[0]*grid[1]*grid[2] if grid else None
    return dict(study=study, case=case, np=npv, backend=backend, rho=rho,
                grid=grid, cells=cells, total=total, ev_mean=ev_mean)

records = [r for r in (parse_file(p) for p in sorted(glob.glob(os.path.join(SRC, "*.csv")))) if r]
print(f"parsed {len(records)} csv files")

# ---- styling helpers ----------------------------------------------------
def style(backend, rho):
    color = "tab:blue" if backend == "pascal" else "tab:red"
    ls    = "-" if rho == "025" else "--"
    mk    = "o" if rho == "025" else "s"
    return color, ls, mk

def label(backend, rho):
    b = "PaScaL" if backend == "pascal" else "Filtered"
    r = "0.25" if rho == "025" else "0.40"
    return f"{b}  ρ={r}"

SERIES = [("pascal", "025"), ("pascal", "040"),
          ("filtered_v2", "025"), ("filtered_v2", "040")]

def get(recs, **kw):
    out = [r for r in recs if all(r[k] == v for k, v in kw.items())]
    return out

# ========================================================================
#  STRONG  — time vs np  +  speedup vs np   (one figure per grid case)
# ========================================================================
strong_cases = sorted({r["case"] for r in records if r["study"] == "strong"},
                      key=lambda c: int(c.split("x")[0]))
for case in strong_cases:
    sub = get(records, study="strong", case=case)
    grid = sub[0]["grid"]
    fig, (axT, axS) = plt.subplots(1, 2, figsize=(12, 5))
    nps_all = sorted({r["np"] for r in sub})
    for backend, rho in SERIES:
        s = sorted(get(sub, backend=backend, rho=rho), key=lambda r: r["np"])
        if not s:
            continue
        x = [r["np"] for r in s]; y = [r["total"] for r in s]
        c, ls, mk = style(backend, rho)
        axT.plot(x, y, ls, color=c, marker=mk, label=label(backend, rho))
        n0, t0 = x[0], y[0]
        axS.plot(x, [t0 / t for t in y], ls, color=c, marker=mk, label=label(backend, rho))
    # ideal lines (baseline np=min)
    base = sorted(get(sub, backend="pascal", rho="025"), key=lambda r: r["np"])
    n0, t0 = base[0]["np"], base[0]["total"]
    xi = np.array(nps_all, float)
    axT.plot(xi, t0 * n0 / xi, "k:", lw=1, label="ideal (∝1/np)")
    axS.plot(xi, xi / n0, "k--", lw=1, label="ideal linear")
    axT.set_xscale("log", base=2); axT.set_yscale("log")
    axT.set_xticks(nps_all); axT.set_xticklabels(nps_all)
    axT.set_xlabel("# MPI processes"); axT.set_ylabel("wall time / step  [s]")
    axT.set_title("Time per step"); axT.grid(True, which="both", alpha=0.3); axT.legend(fontsize=8)
    axS.set_xticks(nps_all); axS.set_xticklabels(nps_all)
    axS.set_xlabel("# MPI processes"); axS.set_ylabel(f"speedup  T({n0})/T(np)")
    axS.set_title("Strong-scaling speedup"); axS.grid(True, alpha=0.3); axS.legend(fontsize=8)
    fig.suptitle(f"Strong scaling — {case}  (grid {grid[0]}×{grid[1]}×{grid[2]}, z-decomp 1×1×np)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT, f"strong_{case}.png")
    fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)

# ========================================================================
#  WEAK  — time vs np  +  weak efficiency vs np   (one figure per cube)
# ========================================================================
weak_cases = sorted({r["case"] for r in records if r["study"] == "weak"},
                    key=lambda c: int(re.sub(r"\D", "", c)))
for case in weak_cases:
    sub = get(records, study="weak", case=case)
    fig, (axT, axE) = plt.subplots(1, 2, figsize=(12, 5))
    nps_all = sorted({r["np"] for r in sub})
    for backend, rho in SERIES:
        s = sorted(get(sub, backend=backend, rho=rho), key=lambda r: r["np"])
        if not s:
            continue
        x = [r["np"] for r in s]; y = [r["total"] for r in s]
        c, ls, mk = style(backend, rho)
        axT.plot(x, y, ls, color=c, marker=mk, label=label(backend, rho))
        t0 = y[0]
        axE.plot(x, [t0 / t for t in y], ls, color=c, marker=mk, label=label(backend, rho))
    axE.axhline(1.0, color="k", ls="--", lw=1, label="ideal (=1)")
    axT.set_xscale("log", base=2)
    axT.set_xticks(nps_all); axT.set_xticklabels(nps_all)
    axT.set_xlabel("# MPI processes"); axT.set_ylabel("wall time / step  [s]")
    axT.set_title("Time per step (ideal: flat)"); axT.grid(True, which="both", alpha=0.3); axT.legend(fontsize=8)
    axE.set_xscale("log", base=2)
    axE.set_xticks(nps_all); axE.set_xticklabels(nps_all)
    axE.set_ylim(0, 1.15)
    axE.set_xlabel("# MPI processes"); axE.set_ylabel("weak efficiency  T(np₀)/T(np)")
    axE.set_title("Weak-scaling efficiency"); axE.grid(True, alpha=0.3); axE.legend(fontsize=8)
    g0 = sub[0]["grid"]
    fig.suptitle(f"Weak scaling — {case}/proc  (z-decomp; total grid grows with np)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT, f"weak_{case}.png")
    fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)

# ========================================================================
#  REFINE  — time vs #cells  (one figure per cross-section family + combined)
# ========================================================================
refine = [r for r in records if r["study"] == "refine"]
def family(r):  # "64x64x256" -> "64x64"
    a, b, _ = r["case"].split("x")
    return f"{a}x{b}"
fams = sorted({family(r) for r in refine}, key=lambda f: int(f.split("x")[0]))

for fam in fams:
    sub = [r for r in refine if family(r) == fam]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for backend, rho in SERIES:
        s = sorted([r for r in sub if r["backend"] == backend and r["rho"] == rho],
                   key=lambda r: r["cells"])
        if not s:
            continue
        x = [r["cells"] for r in s]; y = [r["total"] for r in s]
        c, ls, mk = style(backend, rho)
        ax.plot(x, y, ls, color=c, marker=mk, label=label(backend, rho))
    base = sorted([r for r in sub if r["backend"] == "pascal" and r["rho"] == "025"],
                  key=lambda r: r["cells"])
    c0, t0 = base[0]["cells"], base[0]["total"]
    xi = np.array([r["cells"] for r in base], float)
    ax.plot(xi, t0 * xi / c0, "k:", lw=1, label="O(N) ideal")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("total grid points  (nx·ny·nz)"); ax.set_ylabel("wall time / step  [s]")
    ax.set_title(f"Refinement scaling — {fam}×nz family  (np=2 fixed, refine z)",
                 fontweight="bold")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(OUT, f"refine_{fam}.png")
    fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)

# combined refine overview
fig, ax = plt.subplots(figsize=(8, 6))
fam_marker = {f: m for f, m in zip(fams, ["o", "s", "^", "D", "v"])}
for backend in ("pascal", "filtered_v2"):
    for fam in fams:
        s = sorted([r for r in refine if r["backend"] == backend and r["rho"] == "025"
                    and family(r) == fam], key=lambda r: r["cells"])
        if not s:
            continue
        x = [r["cells"] for r in s]; y = [r["total"] for r in s]
        c = "tab:blue" if backend == "pascal" else "tab:red"
        b = "PaScaL" if backend == "pascal" else "Filtered"
        ax.plot(x, y, "-", color=c, marker=fam_marker[fam],
                label=f"{b} {fam}×nz")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("total grid points  (nx·ny·nz)"); ax.set_ylabel("wall time / step  [s]")
ax.set_title("Refinement scaling — all families (ρ=0.25, np=2)", fontweight="bold")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
out = os.path.join(OUT, "refine_all.png")
fig.savefig(out, dpi=150); plt.close(fig); print("wrote", out)

# ---- dump metrics table -------------------------------------------------
tbl = os.path.join(OUT, "metrics.csv")
with open(tbl, "w") as f:
    f.write("study,case,np,backend,rho,grid,cells,wall_per_step_s," + ",".join(EVENTS) + "\n")
    for r in sorted(records, key=lambda r: (r["study"], r["case"], r["np"], r["backend"], r["rho"])):
        g = "x".join(map(str, r["grid"])) if r["grid"] else ""
        f.write(f'{r["study"]},{r["case"]},{r["np"]},{r["backend"]},{r["rho"]},{g},'
                f'{r["cells"]},{r["total"]:.6e},'
                + ",".join(f'{r["ev_mean"][e]:.6e}' for e in EVENTS) + "\n")
print("wrote", tbl)
print("done. images in", OUT)
