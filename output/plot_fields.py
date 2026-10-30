# Visualisation type Fig. 2.11/2.12 de la thèse : isothermes + lignes de courant
# Convention de lecture : reshape (ny, nx) order="C" (T[iy, ix] ↔ nœud solveur (ix, iy))
import pyvista as pv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

grid = pv.read("output/ra1e6_full/snapshot.vti")
nx, ny, nz = grid.dimensions
T = grid.point_data["Temperature"].reshape((ny, nx), order="C")
ux = grid.point_data["Ux"].reshape((ny, nx), order="C")
uy = grid.point_data["Uy"].reshape((ny, nx), order="C")

x = np.linspace(0.0, 1.0, nx)          # x normalisé
y = np.linspace(0.0, 1.0, ny)          # z normalisé (hauteur)
X, Y = np.meshgrid(x, y)

# --- Fonction de courant psi : dpsi/dy = ux, dpsi/dx = -uy (intégration) ---
psi = np.cumsum(ux, axis=0) * (y[1] - y[0])
psi -= psi.mean(axis=0, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(10, 10))

# (a) Isothermes — uniformément distribuées entre -0.4 et 0.4 (comme la thèse)
ax = axes[0]
levels_T = np.linspace(-0.4, 0.4, 17)
cf = ax.contourf(X, Y, T, levels=levels_T, cmap="coolwarm", extend="both")
cs = ax.contour(X, Y, T, levels=levels_T, colors="k", linewidths=0.4)
ax.set_aspect(4)  # AR=4 : cavité haute et étroite (hauteur = 4× largeur à l'écran)
ax.set_title("Isothermes (uniformément distribuées -0.4 à 0.4)")
ax.set_xlabel("x/Lx"); ax.set_ylabel("z/Lz")
fig.colorbar(cf, ax=ax, shrink=0.85)

# (b) Lignes de courant (fonction de courant)
ax = axes[1]
lv = np.linspace(psi.min(), psi.max(), 21)
cf = ax.contourf(X, Y, psi, levels=lv, cmap="viridis")
ax.contour(X, Y, psi, levels=lv, colors="w", linewidths=0.5)
ax.set_aspect(4)  # AR=4 : cavité haute et étroite (hauteur = 4× largeur à l'écran)
ax.set_title("Fonction de courant (lignes de courant)")
ax.set_xlabel("x/Lx"); ax.set_ylabel("z/Lz")
fig.colorbar(cf, ax=ax, shrink=0.85, label="ψ")

fig.suptitle("Cavité différentiellement chauffée AR=4 — Ra=1e6, LBM D2Q9 + Smagorinsky (grille 64×256)", fontsize=11)
fig.tight_layout()
fig.savefig("output/ra1e6_full/champs.png", dpi=150)
print("Écrit : output/ra1e6_full/champs.png")

# Sanity check physique
print("T paroi gauche (moy):", round(T[:, 0].mean(), 4), "| droite:", round(T[:, -1].mean(), 4))
print("psi: min", round(psi.min(), 5), "| max", round(psi.max(), 5), "| cellule centrale ψ≈", round(psi[ny//2, nx//2], 5))
