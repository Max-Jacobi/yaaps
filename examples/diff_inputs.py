#!/bin/env python3
import argparse
from yaaps.input import Input

_default_ignore = ['output', 'surface', 'psi4_extraction', 'hst_windowed']
_sep = '-'*50+'+'+'-'*52+'+'+'-'*51

parser = argparse.ArgumentParser(description="Diff of two input or rst files")
parser.add_argument("input1", type=str,
                    help="first input file")
parser.add_argument("input2", type=str,
                    help="first input file")
parser.add_argument("--float_tol", type=float, default=1e-8,
                    help="Tolerance for floats to be considdered different")
parser.add_argument("--ignore",  nargs="*", default=_default_ignore, type=str,
                    help="List of strings. Ignore groups that start with these strings")

args = parser.parse_args()

inp1 = Input(args.input1)
inp2 = Input(args.input2)

diff = inp1.diff(inp2, float_tol=args.float_tol)

print(f"{'':50s}  {args.input1:>50} | {args.input2:>50}")
for grp in sorted(diff.keys()):
    if any(grp.startswith(ign) for ign in args.ignore):
        continue
    print(grp+_sep[len(grp):])
    for key, (v1, v2) in diff[grp].items():
        v1, v2 = str(v1), str(v2)
        if len(v1) > 50:
            v1 = '...'+v1[-47:]
        if len(v2) > 50:
            v2 = '...'+v2[-47:]

        print(f"{key:50s}| {v1:>50} | {v2:>50}")
