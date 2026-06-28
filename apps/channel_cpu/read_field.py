#!/usr/bin/env python3
"""
read_field.py — read channel Output_field_*.plt files.

As of the binary switch, these files are:
    line 1 : TITLE = "..."                          (ASCII)
    line 2 : VARIABLES = "X" "Y" "Z" "U" ...         (ASCII)
    line 3 : ZONE T="Field", I=.., J=.., K=.., DATAPACKING=POINT   (ASCII)
    <binary float64 data>: 9 values per point, POINT order
            (i fastest, then j, then k across z-ranks)

Returns a dict of (K, J, I) arrays keyed by variable name.

Usage:
    python3 read_field.py instant/re550_p/Output_field_00040000.plt
    # or in code:
    from read_field import load_field
    f = load_field(path); print(f["U"].shape, f["U"].mean())
"""
import re
import sys
import numpy as np

NVARS = 9  # X Y Z U V W P Q Lambda2


def load_field(path):
    with open(path, "rb") as fp:
        title = fp.readline().decode("ascii", "replace")
        varline = fp.readline().decode("ascii", "replace")
        zoneline = fp.readline().decode("ascii", "replace")

        names = re.findall(r'"([^"]+)"', varline)
        if len(names) != NVARS:
            names = ["X", "Y", "Z", "U", "V", "W", "P", "Q", "Lambda2"]

        def geti(key):
            m = re.search(rf'\b{key}\s*=\s*(\d+)', zoneline)
            if not m:
                raise ValueError(f"{key} not found in ZONE line: {zoneline!r}")
            return int(m.group(1))

        I, J, K = geti("I"), geti("J"), geti("K")

        data = np.fromfile(fp, dtype="<f8")

    expected = K * J * I * NVARS
    if data.size != expected:
        raise ValueError(
            f"size mismatch: got {data.size} float64, expected {expected} "
            f"(I={I} J={J} K={K} x {NVARS} vars). Header: {zoneline!r}")

    data = data.reshape(K, J, I, NVARS)            # (k, j, i, var)
    out = {name: data[..., v] for v, name in enumerate(names)}
    out["_dims"] = (I, J, K)
    out["_title"] = title.strip()
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    f = load_field(sys.argv[1])
    I, J, K = f["_dims"]
    print(f"{f['_title']}")
    print(f"dims (I,J,K) = {I},{J},{K}")
    for name in ("U", "V", "W", "P", "Q", "Lambda2"):
        if name in f:
            a = f[name]
            print(f"  {name:8s} min={a.min():+.4e} max={a.max():+.4e} mean={a.mean():+.4e}")


if __name__ == "__main__":
    main()
