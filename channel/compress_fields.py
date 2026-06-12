#!/usr/bin/env python3
"""
compress_fields.py
Usage: python3 compress_fields.py [--delete-original]
Run from channel/ directory.
Finds all *.plt under instant/ and compresses each to *.plt.zst in-place.
"""
import argparse
import glob
import os
import subprocess
import sys


def compress_file(src: str, delete_original: bool) -> None:
    dst = src + ".zst"
    if os.path.exists(dst):
        print(f"  skip (already exists): {dst}")
        return

    size_mb = os.path.getsize(src) / 1024**2
    print(f"  compressing {src}  ({size_mb:.0f} MB) ...", flush=True)

    result = subprocess.run(
        ["zstd", "-T0", "-o", dst, src],
        check=False,
    )

    if result.returncode != 0:
        print(f"  ERROR: zstd failed for {src}", file=sys.stderr)
        if os.path.exists(dst):
            os.remove(dst)
        return

    dst_mb = os.path.getsize(dst) / 1024**2
    ratio = size_mb / dst_mb if dst_mb > 0 else 0
    print(f"  -> {dst}  ({dst_mb:.0f} MB, {ratio:.1f}x)")

    if delete_original:
        os.remove(src)
        print(f"  deleted original: {src}")


def main():
    parser = argparse.ArgumentParser(description="Compress channel field .plt files with zstd")
    parser.add_argument("--delete-original", action="store_true",
                        help="Delete original .plt after successful compression")
    args = parser.parse_args()

    # Must be run from channel/ directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    instant_dir = os.path.join(script_dir, "instant")

    if not os.path.isdir(instant_dir):
        print(f"ERROR: instant/ not found under {script_dir}", file=sys.stderr)
        sys.exit(1)

    plt_files = sorted(glob.glob(os.path.join(instant_dir, "**", "*.plt"), recursive=True))
    # exclude already-compressed sources
    plt_files = [f for f in plt_files if not f.endswith(".plt.zst")]

    if not plt_files:
        print("No .plt files found under instant/")
        return

    print(f"Found {len(plt_files)} .plt file(s) under {instant_dir}")
    for f in plt_files:
        compress_file(f, args.delete_original)

    print("Done.")


if __name__ == "__main__":
    main()
