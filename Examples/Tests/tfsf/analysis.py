#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
"""
Verify the TFSF (total-field/scattered-field) source.

A pulsed plane wave is injected through a TFSF surface placed
``tfsf.offset_cells`` inside the domain and analytically cancelled where it
leaves.  The scattered-field frame between the surface and the domain boundary
must stay empty up to the incident-dispersion floor at all times.

The cells adjacent to the surface are excluded from the scattered-field frame:
cell-centered plotfile values there average in field components that belong to
the total-field region.

Usage: analysis.py  (loops over all diags/diag1?????? plotfiles in cwd)
"""

import glob
import os
import sys

import numpy as np
import yt

yt.funcs.mylog.setLevel(50)

E0 = 1.0e9

# scattered-field leakage tolerance (relative to E0): the floor is set by how
# well the analytic incident wave satisfies the discrete dispersion relation,
# which degrades with dimensionality/coarseness of the test grids used here
test_name = os.path.split(os.getcwd())[1]
tolerance = {
    "test_1d_tfsf": 1.0e-3,  # cfl ~ 1: near-dispersionless
    "test_2d_tfsf": 8.0e-3,  # grid-matched carrier; finite-bandwidth floor
    "test_3d_tfsf": 4.0e-2,  # lambda/9.6 grid
}[test_name]

# field component carrying the pulse
component = {"test_1d_tfsf": "Ey", "test_2d_tfsf": "Ey", "test_3d_tfsf": "Ex"}[
    test_name
]

# faces driven in each test, per array dimension: (lo, hi)
faces = {
    "test_1d_tfsf": [(True, True)],
    "test_2d_tfsf": [(True, True), (True, True)],
    "test_3d_tfsf": [(True, True), (True, True), (True, True)],
}[test_name]

offset = 8

worst = 0.0
for pf in sorted(glob.glob("diags/diag1??????")):
    ds = yt.load(pf)
    dims = ds.domain_dimensions
    ndim = ds.dimensionality
    grid = ds.covering_grid(0, left_edge=ds.domain_left_edge, dims=dims)
    F = np.asarray(grid["boxlib", component]).squeeze()

    # scattered-field mask, excluding one extra cell adjacent to the surface
    sf = np.zeros(F.shape, dtype=bool)
    for d in range(ndim):
        lo_face, hi_face = faces[d]
        index = [slice(None)] * ndim
        if lo_face:
            index[d] = slice(0, offset - 1)
            sf[tuple(index)] = True
        if hi_face:
            index[d] = slice(F.shape[d] - offset + 1, F.shape[d])
            sf[tuple(index)] = True

    leak = np.max(np.abs(F[sf])) / E0
    worst = max(worst, leak)
    print(f"{pf}: t = {float(ds.current_time)*1e15:7.1f} fs, SF max |{component}|/E0 = {leak:.3e}")

print(f"\nworst scattered-field leakage: {worst:.3e} (tolerance {tolerance:.1e})")
assert worst < tolerance
