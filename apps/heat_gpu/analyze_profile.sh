#!/bin/bash
# Post-process profiling results: per-kernel time (nsys), DRAM/SM utilization (ncu).
set -e

PROF_DIR=/scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/profile_results
NSYS=/usr/local/cuda-13.0/bin/nsys

cd "${PROF_DIR}"

echo "######################################################"
echo "##  TASK 2 — Per-kernel time breakdown (nsys)        ##"
echo "######################################################"
for tag in N512_NP2 N512_NP1 N1024_NP2; do
  rep="nsys_${tag}.nsys-rep"
  if [ ! -f "$rep" ]; then echo "[skip] $rep not found"; continue; fi
  echo ""
  echo "=== ${tag} ==="
  "${NSYS}" stats --report cuda_gpu_kern_sum --format csv "$rep" 2>/dev/null \
    | head -25
done

echo ""
echo "######################################################"
echo "##  TASK 1 — GPU saturation (ncu DRAM/SM)            ##"
echo "######################################################"
for tag in N512_NP2 N512_NP1 N1024_NP2; do
  csv="ncu_${tag}.csv"
  if [ ! -f "$csv" ]; then echo "[skip] $csv not found"; continue; fi
  echo ""
  echo "=== ${tag} ==="
  # ncu csv has multi-line headers; print kernel + DRAM throughput + SM active
  python3 - <<PYEOF
import csv, sys
with open("${csv}") as f:
    rows = list(csv.reader(f))
# Find header row (starts with "ID","Process ID",...)
hdr_i = None
for i, r in enumerate(rows):
    if len(r) > 1 and r[0].strip() in ("ID","\"ID\""):
        hdr_i = i; break
if hdr_i is None:
    print("  [no rows]"); sys.exit(0)
hdr = [c.strip().strip('"') for c in rows[hdr_i]]
need = {"Kernel Name":None, "Metric Name":None, "Metric Value":None}
for i, c in enumerate(hdr):
    if c in need: need[c] = i
agg = {}
for r in rows[hdr_i+1:]:
    if len(r) <= max(v for v in need.values() if v is not None): continue
    k = r[need["Kernel Name"]].strip().strip('"')
    m = r[need["Metric Name"]].strip().strip('"')
    v = r[need["Metric Value"]].strip().strip('"')
    if not k or not m: continue
    agg.setdefault(k, {})[m] = v
for k in sorted(agg):
    d = agg[k]
    dram = d.get("dram__throughput.avg.pct_of_peak_sustained_elapsed","?")
    sm = d.get("sm__cycles_active.avg.pct_of_peak_sustained_elapsed","?")
    smt = d.get("sm__throughput.avg.pct_of_peak_sustained_elapsed","?")
    occ = d.get("launch__occupancy_limit_active_warps","?")
    print(f"  {k[:50]:50s}  DRAM={dram:>8s}  SMact={sm:>8s}  SMthru={smt:>8s}  occ={occ:>5s}")
PYEOF
done
