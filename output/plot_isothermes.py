#!/usr/bin/env python3
"""Tracé des isothermes + w̄ depuis fields_mean.csv (cavité DHC Az=4, LBM).

Usage :
    .venv/bin/python output/plot_isothermes.py output/ra1e8_val2 [options]

Options principales :
    --labels / --no-labels   afficher (défaut) ou non les valeurs sur les contours
    --out Fichier.png        sortie (défaut : isothermes.png dans le dossier)
    --wmax WM                échelle couleur w (défaut : max|w| du champ, arrondi)

Conventions lecture : reshape (ny, nx) order="C" — T[iy, ix] ↔ nœud (ix, iy).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="dossier contenant fields_mean.csv")
    ap.add_argument("--labels", dest="labels", action="store_true", default=True,
                    help="afficher les valeurs sur les contours (défaut)")
    ap.add_argument("--no-labels", dest="labels", action="store_false",
                    help="masquer les valeurs sur les contours")
    ap.add_argument("--out", type=Path, default=None,
                    help="fichier PNG de sortie (défaut : <run_dir>/isothermes.png)")
    ap.add_argument("--wmax", type=float, default=None,
                    help="demi-échelle couleur de w (défaut : max|w| du champ)")
    ap.add_argument("--title", type=str, default=None, help="titre de la figure")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    csv = args.run_dir / "fields_mean.csv"
    d = np.loadtxt(csv, delimiter=",", skiprows=1)
    T, uy = d[:, 4], d[:, 3]

    xs = np.unique(d[:, 0])
    ys = np.unique(d[:, 1])
    nx, ny = len(xs), len(ys)
    Tg = T.reshape((ny, nx), order="C")   # T[iy, ix] — JAMAIS order="F"
    Ug = uy.reshape((ny, nx), order="C")
    # Normalisation physique : Lx = nx-1 [u.l. grille], Lz = ny-1 ; Az = Lz/Lx.
    Lx, Lz = nx - 1, ny - 1
    Xn = np.linspace(0, 1, nx)
    Yn = np.linspace(0, Lz / Lx, ny)

    w_scale = args.wmax if args.wmax is not None else round(float(np.abs(Ug).max()), 4)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 5.5))

    # --- Isothermes theta ---
    ax = axes[0]
    levels = np.linspace(-0.5, 0.5, 11)
    cf = ax.contourf(Xn, Yn, Tg, levels=levels, cmap="RdBu_r", extend="both")
    cs = ax.contour(Xn, Yn, Tg, levels=levels, colors="k", linewidths=0.6)
    if args.labels:
        # Une étiquette par niveau, ancrée sur un vrai sommet du contour
        # (portion centrale x∈[0.30,0.70]), cible y répartie uniformément.
        y_targets = np.linspace(0.6, Lz / Lx - 0.6, len(levels))
        for k, lv in enumerate(levels):
            if abs(lv) > 0.35:  # niveaux extrêmes confinés aux BL : pas d'étiquette
                continue
            # path du contour de ce niveau
            pi = list(cs.levels).index(lv)
            v = cs.get_paths()[pi].vertices
            m = (v[:, 0] > 0.30) & (v[:, 0] < 0.70)
            if not m.any():
                continue
            cand = v[m]
            j = np.argmin(np.abs(cand[:, 1] - y_targets[k]))
            pt = cand[j]
            ax.clabel(cs, levels=[lv], inline=True, fmt="%.2f", fontsize=8,
                      colors="k", manual=[(pt[0], pt[1])])
        for txt in cs.labelTexts:
            txt.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))
            txt.set_rotation(0)
    ax.set_xlabel("$x^* = x/H$")
    ax.set_ylabel("$z/H$")
    ax.set_title("Isothermes $\\bar{\\theta}$")
    ax.set_aspect("equal")
    plt.colorbar(cf, ax=ax, ticks=levels)

    # --- Vitesse verticale w ---
    ax = axes[1]
    lvl_w = np.array([-w_scale, -0.67 * w_scale, -0.33 * w_scale,
                      0.33 * w_scale, 0.67 * w_scale, w_scale])
    cf2 = ax.contourf(Xn, Yn, Ug, levels=lvl_w, cmap="RdBu_r", extend="both")
    cs2 = ax.contour(Xn, Yn, Ug, levels=lvl_w, colors="k", linewidths=0.6)
    if args.labels:
        # BL collée aux parois (x<0.05) : clabel snaperait l'étiquette sur le
        # contour et déborderait du cadre → annotations texte en position
        # intérieure fixe, côté chaud (w>0) / froid (w<0), z bien espacés.
        n_w = len(lvl_w)
        z_pos = np.linspace(0.5, Lz / Lx - 0.5, n_w)
        for k, lv in enumerate(lvl_w):
            x_t = 0.13 if lv > 0 else 0.87
            ax.text(x_t, z_pos[k], f"{lv:.3f}", fontsize=8, color="k",
                    ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    ax.set_xlabel("$x^* = x/H$")
    ax.set_title("$\\bar{w}$")
    ax.set_aspect("equal")
    plt.colorbar(cf2, ax=ax, ticks=lvl_w)

    # --- Lignes de courant (fonction de courant psi) ---
    ax = axes[2]
    psi = d[:, 5]
    Pg = psi.reshape((ny, nx), order="C")
    # contours de psi = lignes de courant ; peu de niveaux tracés (les lignes
    # se resserrent physiquement dans la BL pariétale — on limite l'effet)
    n_stream = 5
    lv_stream = np.linspace(Pg.min() * 0.95, Pg.max(), n_stream)
    lv_stream = np.array([lv for lv in lv_stream
                          if abs(lv) > 0.15 * abs(Pg.min())])
    cf3 = ax.contourf(Xn, Yn, Pg, levels=lv_stream, cmap="RdBu_r")
    cs3 = ax.contour(Xn, Yn, Pg, levels=lv_stream, colors="k", linewidths=0.8)
    ax.clabel(cs3, levels=lv_stream[1::2], inline=True, fmt="%.3f", fontsize=7)
    ax.set_xlabel("$x^* = x/H$")
    ax.set_title("Lignes de courant ($\\psi$)")
    ax.set_aspect("equal")
    plt.colorbar(cf3, ax=ax, label="$\\psi$ [u.l.]")

    if args.title is None:
        args.title = (f"DHC Az={Lz / Lx:.2f}, maillage {nx}×{ny}, "
                      f"LBM D2Q9 TRT — champ moyen")
    fig.suptitle(args.title, fontsize=9)
    plt.tight_layout()

    out = args.out if args.out is not None else args.run_dir / "isothermes.png"
    fig.savefig(out, dpi=args.dpi)
    print(f"fig écrite : {out} (labels={'on' if args.labels else 'off'}, w_scale={w_scale})")


if __name__ == "__main__":
    sys.exit(main())
