"""Vérification pyvista finale du smoke Ra=1e6 (convention order='C')."""
import pyvista as pv
import numpy as np

g = pv.read("output/smoke_ra1e6/snapshot.vti")
nx, ny, _ = g.dimensions
T = g["Temperature"].reshape(ny, nx, order="C")   # T[iy, ix] = nœud solveur (ix, iy)
U = g["Uy"].reshape(ny, nx, order="C")

ok = True

# 1. Polarité thermique
th, tc = T[:, 0].mean(), T[:, -1].mean()
print(f"Paroi gauche  T = {th:+.6f} (attendu +0.5)")
print(f"Paroi droite  T = {tc:+.6f} (attendu -0.5)")
ok &= abs(th - 0.5) < 1e-6 and abs(tc + 0.5) < 1e-6

# 2. Jet : ascendant côté chaud (gauche), descendant côté froid (droite)
for frac in (0.25, 0.5):
    iy = int(round(frac * (ny - 1)))
    row = U[iy]
    jl, jr = row[1:6].mean(), row[-6:-1].mean()
    print(f"z={frac}: Uy BL gauche (ix1..5) = {jl:+.5f} | Uy BL droite (ix58..62) = {jr:+.5f}")
    ok &= jl > 0 and jr < 0

# 3. w_max(z=0.5) dans la couche limite chaude
row = U[ny // 2]
ix_max = int(np.argmax(np.abs(row[1 : nx // 2]))) + 1
print(f"w_max(z=0.5) BL chaude : {row[ix_max]:+.6f} à ix={ix_max}, x = {ix_max/(nx-1):.4f}·Lx")

# 4. Masse (rho exporté ? sinon check via champs) — rho non exporté en VTI,
#    on vérifie le progress.log (masse/nœud = 1.000000)
log = open("output/smoke_ra1e6/progress.log").read()
masse_ok = "masse = 1.000000" in log
print(f"Masse conservée (progress.log) : {masse_ok}")
ok &= masse_ok

print("\nVERDICT :", "OK" if ok else "ECHEC")
