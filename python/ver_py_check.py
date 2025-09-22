#!/usr/bin/env python3
"""
ver_py_check_fixed.py

Compare Python reference output vs Verilog output, auto-trim leading warmup
samples (if one file is longer), and report mismatches.

Usage:
    python ver_py_check_fixed.py \
        --py fir_61_pyref_file.txt \
        --ver verilog_ref_file.txt  \
        --ideal py_ideal_conv.txt
"""
import numpy as np
from pathlib import Path
import argparse
import sys

def load_int_file(fn):
    p = Path(fn)
    if not p.exists():
        print(f"ERROR: file not found: {fn}")
        sys.exit(2)
    # robust parsing: ignore blank lines and non-numeric lines
    vals = []
    for ln in p.read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith('#'):
            continue
        tok = s.split()[0]
        try:
            # allow float->int conversion if file contains floats
            v = int(float(tok))
        except:
            # skip malformed
            continue
        vals.append(v)
    return np.array(vals, dtype=np.int64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--py',   default='fir_61_pyref_file.txt', help='Python reference output file')
    ap.add_argument('--ver',  default='verilog_ref_file.txt',   help='Verilog DUT output file')
    ap.add_argument('--ideal',   default='py_ideal_conv.txt', help='ideal output file')
    ap.add_argument('--maxshow', type=int, default=16, help='how many initial samples to print')
    args = ap.parse_args()

    py = load_int_file(args.py)
    ver = load_int_file(args.ver)
    ideal = load_int_file(args.ideal)

    Lpy = len(py); Lver = len(ver) ; Lideal = len(ideal)
    print(f"Loaded Python ref '{args.py}' len = {Lpy}")
    print(f"Loaded Verilog ref '{args.ver}' len = {Lver}")
    print(f"Loaded Verilog ref '{args.ideal}' len = {Lideal}")

    if Lpy == 0 or Lver == 0:
        print("ERROR: one of the files is empty (or no numeric rows).")
        sys.exit(2)

    # If lengths differ, trim the longer file at the front by the difference
    if Lpy != Lver:
        if Lpy > Lver:
            diff = Lpy - Lver
            print(f"Length mismatch: Python longer by {diff} samples. Trimming first {diff} samples from Python ref.")
            py = py[diff:]
        else:
            diff = Lver - Lpy
            print(f"Length mismatch: Verilog longer by {diff} samples. Trimming first {diff} samples from Verilog ref.")
            ver = ver[diff:]
    L = min(len(py), len(ver))
    print(f"Comparing {L} samples after trimming (if any).")

    # quick head print
    head_n = min(args.maxshow, L)
    print("\nFirst samples (expected=python | verilog | Ideal):")
    for i in range(head_n):
        print(f"{i:04d}: {py[i]:>12d}  |  {ver[i]:>12d}  |  {ideal[i]:>12d}")

    # find first mismatch
    neq = np.nonzero(py[:L] != ver[:L])[0]
    if neq.size == 0:
        print("\nNo mismatches found. Files match for all compared samples ✅")
        # also print a small checksum to be extra-sure
        print(f"Python sum={int(py.sum())}  Verilog sum={int(ver.sum())}  Ideal sum={int(ideal.sum())}")
        sys.exit(0)
    else:
        first = int(neq[0])
        total_mismatch = len(neq)
        print(f"\nMismatch found at index {first} (0-based within compared block).")
        print(f"  python[{first}] = {int(py[first])}")
        print(f"  verilog[{first}] = {int(ver[first])}")
        print(f"  Ideal[{first}] = {int(ideal[first])}")
        print(f"Total mismatches in compared block: {total_mismatch} / {L}")
        # print up to 8 mismatches to inspect
        to_show = neq[:8]
        print("\nUp to first mismatches (index, python, verilog):")
        for idx in to_show:
            print(f" {int(idx):6d} : {int(py[int(idx)]):12d} | {int(ver[int(idx)]):12d}  |  {int(ideal[int(idx)]):12d}")
        print("\nIf the first mismatch is at index 0, check input alignment and that the TB wrote outputs in the same alignment as Python.")
        sys.exit(1)

if __name__ == '__main__':
    main()
