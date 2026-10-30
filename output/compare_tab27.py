# Comparaison des résultats LBM avec le Tableau 2.7 de la thèse (Ra=1e10)
# Usage : .venv/bin/python output/compare_tab27.py [summary.json]
# Convention de lecture : reshape (ny, nx) order="C" — T[iy,ix] ↔ nœud solveur (ix,iy)
#
# Échelles (relecture thèse §2.3.6, p. internes 54-55 + Trias JFM 2007) :
# - Vitesses en U_CN = (κ/H)·√Ra_H = √(gβΔT·H/Pr) = chute_libre/√Pr
# - x* : Trias tabule en unités de LARGEUR (x ∈ [0, 1/A]) ; Brangeon §2.3.6
#   définit x* = x/H — facteur 4 entre les deux conventions. Les x05/x075
#   ci-dessous sont en unités de largeur (convention Trias/Xin).
# - Nusselt paroi = ∂θ/∂n · H (normalisation hauteur, cf. post.rs)
# - Référence de recalage : Trias, Soria, Oliva & Pérez-Segarra,
#   J. Fluid Mech. 586 (2007), 259-293, table 2 (2D) :
#   Nū = 49.23 / 66.19 / 100.60 et wmax = 0.223 / 0.229 / 0.230
#   aux Ra_H = 6.4e8 / 2e9 / 1e10. Fit : Nu = 0.2529·Ra_H^0.2600.
import json
import sys

# --- Références Tab. 2.7 (Ra_H = 1e10) ---
TAB27 = {
    "SND2D1 (Xin 1995)":        {"Nx": 96,  "Nz": 768, "wmax05": 0.230, "x05": 0.0038, "wmax075": 0.110, "x075": 0.0054, "Nuc": 102.0,  "S12": 1.10, "yplus_c": None, "nusm": None},
    "SND2D2 (Sergent 2000)":    {"Nx": 130, "Nz": 514, "wmax05": 0.226, "x05": 0.0040, "wmax075": 0.120, "x075": 0.0055, "Nuc": 100.5,  "S12": 1.10, "yplus_c": None, "nusm": None},
    "SGE2D (Sergent 2000)":     {"Nx": 66,  "Nz": 130, "wmax05": 0.238, "x05": 0.0036, "wmax075": 0.167, "x075": 0.0052, "Nuc": 122.6,  "S12": 1.17, "yplus_c": None, "nusm": 0.82},
    "C2D1 (Sergent 2000)":      {"Nx": 64,  "Nz": 128, "wmax05": 0.221, "x05": 0.0040, "wmax075": 0.135, "x075": 0.0050, "Nuc": 100.5,  "S12": 1.20, "yplus_c": 1.11, "nusm": 0.87},
    "C2D2 (Sergent 2000)":      {"Nx": 96,  "Nz": 320, "wmax05": 0.223, "x05": 0.0035, "wmax075": 0.130, "x075": 0.0055, "Nuc": 100.2,  "S12": 1.02, "yplus_c": 0.75, "nusm": 0.54},
    "C2D3 (Sergent 2000)":      {"Nx": 128, "Nz": 320, "wmax05": 0.224, "x05": 0.0040, "wmax075": 0.110, "x075": 0.0060, "Nuc": 100.36, "S12": 1.05, "yplus_c": 0.55, "nusm": 0.48},
    "Trias 2007 2D (JFM 586)":  {"Nx": 138, "Nz": 326, "wmax05": 0.230, "x05": 0.0040, "wmax075": None, "x075": None,   "Nuc": 100.60, "S12": None, "yplus_c": None, "nusm": None},
}

path = sys.argv[1] if len(sys.argv) > 1 else "output/ra1e10/summary.json"
try:
    s = json.load(open(path))
except FileNotFoundError:
    print(f"⚠️  {path} introuvable — run Ra=1e10 pas encore faite.")
    print("    Voici la cible Tab. 2.7 à atteindre :\n")
    hdr = f"{'Cas':<22}{'Nx':>4}{'Nz':>5}{'Nuc':>8}{'wmax05':>8}{'x05':>7}{'wmax075':>9}{'x075':>7}{'S^1/2':>7}{'nu_sm/nu':>9}{'y+ c':>6}"
    print(hdr); print("-" * len(hdr))
    def fmt(v, spec):
        return f"{v:{spec}}" if v is not None else "—"
    for name, r in TAB27.items():
        nusm = fmt(r["nusm"], ".2f")
        yp = fmt(r["yplus_c"], ".2f")
        print(f"{name:<22}{r['Nx']:>4}{r['Nz']:>5}{fmt(r['Nuc'],'.2f'):>8}{fmt(r['wmax05'],'.3f'):>8}{fmt(r['x05'],'.4f'):>7}{fmt(r['wmax075'],'.3f'):>9}{fmt(r['x075'],'.4f'):>7}{fmt(r['S12'],'.2f'):>7}{nusm:>9}{yp:>6}")
    sys.exit(0)

# --- Résultats LBM ---
print(f"=== LBM D2Q9 + Smagorinsky — {path} ===")
print(f"Grille : {s['grid']} | Ra = {s['ra']:.1e} | C_s = {s['cs']}")
print(f"Nuc = {s['nusselt_hot']:.4f} | Nuf = {s['nusselt_cold']:.4f}")
print(f"wmax(z=0.5) = {s['wmax_z05']:.4f} en x = {s['x_wmax_z05']:.4f}")
print(f"wmax(z=0.75) = {s['wmax_z075']:.4f} en x = {s['x_wmax_z075']:.4f}")
print(f"y+ (paroi chaude) = {s['y_plus_hot']:.3f}")
print()

# --- Comparaison vs Tab. 2.7 ---
print("=== Comparaison vs Tab. 2.7 (Ra=1e10, cibles) ===")
print(f"{'Quantité':<14}{'LBM':>10}{'Cible (min-max)':>20}{'Verdict':>30}")
print("-" * 74)
# CONVERSION x* (garde-fou) : le solveur normalise x par la LARGEUR du domaine
# (x_norm = ix/(nx-1)), la cavité entière. Les cibles Tab. 2.7 (Sergent, tanh)
# sont aussi en unités de largeur de la cavité entière. Trias JFM 2007 tabule
# x2 ∈ [0, 1/A]=[0, 0.25] en unités de largeur — facteur 4 à appliquer si on
# compare aux x2 de Trias table 2 (x2 = 4·x* ici).
rows = [
    ("Nuc",           s["nusselt_hot"],  100.2, 122.6,  ""),
    ("wmax(z=0.5)",   s["wmax_z05"],     0.221, 0.238,  ""),
    ("x @ wmax05",    s["x_wmax_z05"],   0.0035, 0.0040, "(pose problème en uniforme)"),
    ("wmax(z=0.75)",  s["wmax_z075"],    0.110, 0.167,  ""),
    ("x @ wmax075",   s["x_wmax_z075"],  0.0050, 0.0060, "(pose problème en uniforme)"),
    ("y+ chaud",      s["y_plus_hot"],   0.55,  1.11,   "(uniforme vs tanh)"),
]
for name, val, lo, hi, note in rows:
    ok = lo <= val <= hi
    print(f"{name:<14}{val:>10.4f}{'[' + f'{lo}' + ' – ' + f'{hi}' + ']':>20}{'✅ dans la fourchette' if ok else '⚠️ hors fourchette':^28}{note}")

# Cibles Trias 2007 (2D, extrapolation par Ra si Ra != 1e10)
ra = s["ra"]
fit_C, fit_n = 0.2529, 0.2600
nu_trias_1e10 = 100.60
nu_cible_fit = fit_C * ra**fit_n
print(f"\nCible fit Trias 2D : Nu = {fit_C}·Ra^({fit_n}) -> à Ra={ra:.1e} : Nu_fit = {nu_cible_fit:.2f}"
      f" (LBM = {s['nusselt_hot']:.2f}, écart {abs(s['nusselt_hot']/nu_cible_fit-1)*100:.1f} %)")

if "s_half" in s:
    print(f"\nS^(1/2) = {s['s_half']:.3f} (cible 1.02–1.20)")

# Limitations LBM uniforme à documenter
print("\n--- Limitations connues (LBM grille uniforme) ---")
nx = s["grid"][0]["nx"]
print(f"1. Position du pic wmax : cible x≈0.004 ; premier nœud hors paroi en uniforme : x = {1.0/(nx-1):.4f}")
print("   → le pic BL ne peut pas être localisé à x≈0.004 sans maillage tanh (Eq. 2.104, impossible en LBM)")
print("2. y+ : cible 0.55–1.11 avec clustering tanh ; en uniforme, y+ dépend du calage utarget")
print("3. ν_sm/ν|max : cible 0.48–0.87 (C2D) — à logger dans la progression du solveur")
