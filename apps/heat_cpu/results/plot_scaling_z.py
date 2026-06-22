#!/usr/bin/env python3
"""
Scaling plots using ONLY the z-direction (distributed) solve time, solve_z.

Rationale: the domain is decomposed (1,1,np), so solve_x and solve_y run with
nprocs=1 and FilteredTDMA falls back to plain Thomas (== PaScaL). rhs is solver-
independent. The ONLY component where PaScaL and Filtered differ is solve_z
(the distributed-direction TDMA). Isolating solve_z removes the rhs/local
dilution and shows the true method comparison.

Reads cpu_image/metrics.csv (already produced by plot_scaling_cpu.py).
Writes images to cpu_image/z_only/.
"""
import os, csv, re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/scratch/x3319a05/Filtered_TDMA/apps/heat_cpu/results/cpu_image"
OUT  = os.path.join(BASE, "z_only")
os.makedirs(OUT, exist_ok=True)
METRIC = "solve_z"

rows = []
with open(os.path.join(BASE, "metrics.csv")) as f:
    for r in csv.DictReader(f):
        r["np"] = int(r["np"]); r["cells"] = int(r["cells"])
        r["y"] = float(r[METRIC])
        rows.append(r)

def style(b, rho):
    return ("tab:blue" if b == "pascal" else "tab:red",
            "-" if rho == "025" else "--",
            "o" if rho == "025" else "s")
def label(b, rho):
    return f'{"PaScaL" if b=="pascal" else "Filtered"}  ρ={"0.25" if rho=="025" else "0.40"}'
SERIES = [("pascal","025"),("pascal","040"),("filtered_v2","025"),("filtered_v2","040")]
def sel(study, **kw):
    return [r for r in rows if r["study"] == study and all(r[k] == v for k, v in kw.items())]

ZNOTE = "z-direction (distributed) solve only — the sole component where PaScaL ≠ Filtered"

# ---------------- STRONG : solve_z vs np + speedup ----------------
for case in sorted({r["case"] for r in rows if r["study"]=="strong"}, key=lambda c:int(c.split("x")[0])):
    sub = sel("strong", case=case); grid = sub[0]["grid"]
    nps = sorted({r["np"] for r in sub})
    fig,(axT,axS) = plt.subplots(1,2,figsize=(12,5))
    for b,rho in SERIES:
        s = sorted([r for r in sub if r["backend"]==b and r["rho"]==rho], key=lambda r:r["np"])
        if not s: continue
        x=[r["np"] for r in s]; y=[r["y"] for r in s]; c,ls,mk=style(b,rho)
        axT.plot(x,y,ls,color=c,marker=mk,label=label(b,rho))
        axS.plot(x,[y[0]/v for v in y],ls,color=c,marker=mk,label=label(b,rho))
    base=sorted([r for r in sub if r["backend"]=="pascal" and r["rho"]=="025"],key=lambda r:r["np"])
    n0,t0=base[0]["np"],base[0]["y"]; xi=np.array(nps,float)
    axT.plot(xi,t0*n0/xi,"k:",lw=1,label="ideal (∝1/np)")
    axS.plot(xi,xi/n0,"k--",lw=1,label="ideal linear")
    axT.set_xscale("log",base=2); axT.set_yscale("log")
    axT.set_xticks(nps); axT.set_xticklabels(nps)
    axT.set_xlabel("# MPI processes"); axT.set_ylabel("solve_z time / step  [s]")
    axT.set_title("z-solve time per step"); axT.grid(True,which="both",alpha=.3); axT.legend(fontsize=8)
    axS.set_xticks(nps); axS.set_xticklabels(nps)
    axS.set_xlabel("# MPI processes"); axS.set_ylabel(f"speedup  Tz({n0})/Tz(np)")
    axS.set_title("z-solve strong-scaling speedup"); axS.grid(True,alpha=.3); axS.legend(fontsize=8)
    fig.suptitle(f"Strong scaling (z-solve only) — {case}  (grid {grid})\n{ZNOTE}",fontweight="bold",fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.93])
    p=os.path.join(OUT,f"strong_{case}_z.png"); fig.savefig(p,dpi=150); plt.close(fig); print("wrote",p)

# ---------------- WEAK : solve_z vs np + efficiency ----------------
for case in sorted({r["case"] for r in rows if r["study"]=="weak"}, key=lambda c:int(re.sub(r"\D","",c))):
    sub = sel("weak", case=case); nps=sorted({r["np"] for r in sub})
    fig,(axT,axE)=plt.subplots(1,2,figsize=(12,5))
    for b,rho in SERIES:
        s=sorted([r for r in sub if r["backend"]==b and r["rho"]==rho],key=lambda r:r["np"])
        if not s: continue
        x=[r["np"] for r in s]; y=[r["y"] for r in s]; c,ls,mk=style(b,rho)
        axT.plot(x,y,ls,color=c,marker=mk,label=label(b,rho))
        axE.plot(x,[y[0]/v for v in y],ls,color=c,marker=mk,label=label(b,rho))
    axE.axhline(1.0,color="k",ls="--",lw=1,label="ideal (=1)")
    for ax in (axT,axE):
        ax.set_xscale("log",base=2); ax.set_xticks(nps); ax.set_xticklabels(nps)
        ax.set_xlabel("# MPI processes"); ax.grid(True,which="both",alpha=.3); ax.legend(fontsize=8)
    axT.set_ylabel("solve_z time / step  [s]"); axT.set_title("z-solve time per step (ideal: flat)")
    axE.set_ylabel("weak efficiency  Tz(np₀)/Tz(np)"); axE.set_ylim(0,1.2); axE.set_title("z-solve weak efficiency")
    fig.suptitle(f"Weak scaling (z-solve only) — {case}/proc\n{ZNOTE}",fontweight="bold",fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.93])
    p=os.path.join(OUT,f"weak_{case}_z.png"); fig.savefig(p,dpi=150); plt.close(fig); print("wrote",p)

# ---------------- REFINE : solve_z vs cells ----------------
refine=[r for r in rows if r["study"]=="refine"]
fam=lambda r:"x".join(r["case"].split("x")[:2])
fams=sorted({fam(r) for r in refine},key=lambda f:int(f.split("x")[0]))
for fm in fams:
    sub=[r for r in refine if fam(r)==fm]
    fig,ax=plt.subplots(figsize=(7,5.5))
    for b,rho in SERIES:
        s=sorted([r for r in sub if r["backend"]==b and r["rho"]==rho],key=lambda r:r["cells"])
        if not s: continue
        x=[r["cells"] for r in s]; y=[r["y"] for r in s]; c,ls,mk=style(b,rho)
        ax.plot(x,y,ls,color=c,marker=mk,label=label(b,rho))
    base=sorted([r for r in sub if r["backend"]=="pascal" and r["rho"]=="025"],key=lambda r:r["cells"])
    c0,t0=base[0]["cells"],base[0]["y"]; xi=np.array([r["cells"] for r in base],float)
    ax.plot(xi,t0*xi/c0,"k:",lw=1,label="O(N) ideal")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("total grid points (nx·ny·nz)"); ax.set_ylabel("solve_z time / step  [s]")
    ax.set_title(f"Refinement (z-solve only) — {fm}×nz  (np=2)\n{ZNOTE}",fontweight="bold",fontsize=10)
    ax.grid(True,which="both",alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout()
    p=os.path.join(OUT,f"refine_{fm}_z.png"); fig.savefig(p,dpi=150); plt.close(fig); print("wrote",p)

# refine: PaScaL/Filtered solve_z ratio vs cells (one figure, makes the gap explicit)
fig,ax=plt.subplots(figsize=(7.5,5.5))
for rho in ("025","040"):
    pts=defaultdict(dict)
    for r in refine:
        if r["rho"]==rho: pts[r["case"]][r["backend"]]=r["y"]; pts[r["case"]]["cells"]=r["cells"]
    items=sorted(pts.values(),key=lambda d:d["cells"])
    x=[d["cells"] for d in items if "pascal" in d and "filtered_v2" in d]
    y=[d["pascal"]/d["filtered_v2"] for d in items if "pascal" in d and "filtered_v2" in d]
    ax.plot(x,y,("-o" if rho=="025" else "--s"),color="tab:green",
            label=f"ρ={'0.25' if rho=='025' else '0.40'}")
ax.axhline(1.0,color="k",ls=":",lw=1,label="equal")
ax.set_xscale("log"); ax.set_xlabel("total grid points (nx·ny·nz)")
ax.set_ylabel("solve_z speed ratio  PaScaL / Filtered")
ax.set_title("How much faster is Filtered in the z-solve? (np=2, refine)\n>1 means Filtered is faster",
             fontweight="bold",fontsize=10)
ax.grid(True,which="both",alpha=.3); ax.legend(fontsize=9)
fig.tight_layout()
p=os.path.join(OUT,"refine_zratio.png"); fig.savefig(p,dpi=150); plt.close(fig); print("wrote",p)
print("done ->",OUT)
