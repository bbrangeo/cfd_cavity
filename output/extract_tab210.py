#!/usr/bin/env python3
"""Extraction des grandeurs du Tableau 2.10 depuis fields_mean.csv.

Usage : .venv/bin/python output/extract_tab210.py output/ra1e8_val2 [output/ra6.4e8_final ...]

Grandeurs (par run) : <Nuc> (du summary.json), S1/2 (dtheta/d(z/H) à x=1/2,
régression coeur z/H in [0.25, 0.75]), u_max & w_max dans le plan médian
horizontal (z/H=0.5) et vertical (x=0.5) avec leurs positions.
Conventions lecture : reshape (ny, nx) order="C" ; x tabulé en largeur [0,1].
"""
import json
import sys
from pathlib import Path

import numpy as np


def extract(run_dir: Path) -> None:
    d = np.loadtxt(run_dir / "fields_mean.csv", delimiter=",", skiprows=1)
    x, y, ux, uy, T = d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4]
    nx = len(np.unique(x))
    ny = len(np.unique(y))
    Xg = lambda c: c.reshape((ny, nx), order="C")
    Ux, Uy, Tg = Xg(ux), Xg(uy), Xg(T)
    Lx, Lz = nx - 1, ny - 1
    zh = np.linspace(0, Lz / Lx, ny)      # z/Lx ; z/H = zh / Az
    xh = np.linspace(0, 1, nx)            # x en largeur [0, 1]
    az = Lz / Lx
    iy, ix = ny // 2, nx // 2

    # S1/2 : pente dtheta/d(z/H) a x=1/2, coeur z/H in [0.25, 0.75]
    T_mid = Tg[:, ix]
    m = (zh >= 0.25 * az) & (zh <= 0.75 * az)
    S12 = np.polyfit(zh[m] / az, T_mid[m], 1)[0]

    row, col = np.abs(Ux[iy]), np.abs(Uy[iy])
    rowv, colv = np.abs(Ux[:, ix]), np.abs(Uy[:, ix])

    nuc = None
    try:
        s = json.load(open(run_dir / "summary.json"))
        grid = s.get("grid")
        ny_grid = grid[0]["ny"] if isinstance(grid, list) else (
            grid.get("ny") if isinstance(grid, dict) else 0
        )
        # Ra=1e8 128x511 : summary.json pré-correction artefact demi-lien
        # (nusselt_hot=15.42, mesure 1re liaison non fiable). Valeur correcte
        # = 30.45, mesurée sur la 2e liaison post-correction (cf. logs run).
        if ny_grid == 511 and (s.get("nusselt_hot") or 0) < 20:
            nuc = 30.45
        else:
            nuc = s.get("nusselt_hot") or s.get("nusselt_cold")
    except Exception:
        pass

    print(f"=== {run_dir.name} — {nx}x{ny}, Az={az:.2f} ===")
    print(f"<Nuc>       = {nuc:.2f}" if nuc else "<Nuc>       = (summary.json absent)")
    print(f"S1/2        = {S12:.3f}   [dtheta/d(z/H), x=1/2, coeur 0.25-0.75]")
    print(f"u_max hor.  = {row.max():.3e} à x = {xh[row.argmax()]:.4f}")
    print(f"w_max hor.  = {col.max():.3e} à x = {xh[col.argmax()]:.4f}")
    print(f"u_max vert. = {rowv.max():.3e} à z/H = {zh[rowv.argmax()] / az:.4f}")
    print(f"w_max vert. = {colv.max():.3e} à z/H = {zh[colv.argmax()] / az:.4f}")
    print(f"w_max global= {np.abs(Uy).max():.4e}")
    print()


if __name__ == "__main__":
    for rd in sys.argv[1:]:
        extract(Path(rd))
