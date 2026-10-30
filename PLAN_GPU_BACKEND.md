# Plan de développement — Backend GPU wgpu (`--gpu`) pour cavity LES

> Statut : PLAN (non implémenté). Branche `dev-gpu`. Le code CPU actuel reste la
> référence validée — le backend GPU est une **extension**, jamais un remplacement.

## 0. Objectif et ordre de grandeur du gain

**Mesuré aujourd'hui** (benchmark README, 64×256, 50 000 pas, `--jobs 8`) :
**~32 MLUPS** (32 M nœuds/s). Le run Ra=1e10 (256×1024, 2M pas) ≈ **4.6 h**
théorique ; mesuré ~2.9 h (fixe, 0.19k pas/s) et ~3.2 h (dynamique).

**GPU (D2Q9 double distribution)** : plusieurs centaines de millions à
>1 milliard de nœuds/s courant → espérer **10–30× plus vite**, soit le même run
en **10–30 min**. Estimation à mesurer.

## 1. Choix du framework

| Option | GPU visés | Remarque |
|---|---|---|
| **wgpu (WGSL)** ✅ défaut | Apple (Metal), NVIDIA, AMD, Intel | Portable : M1 Max local comme serveur. Choix retenu |
| cudarc (CUDA) | NVIDIA seulement | Option cluster NVIDIA (ORENI) — plus tard |
| cubecl | multi-backend | Kernels en Rust, plus jeune — à surveiller |

## 2. Le point délicat — précision f32 (LE risque du portage)

Metal ne gère pas le f64 ; f32 obligatoire. À τ⁺ ≈ 0.506, les parties hors-équilibre
sont minuscules devant ρ (f^neq/f ~ 1e-3 près des parois à Ra=1e10) — précision
relative f32 (~7 chiffres) juste mais limite.

**Parade classique (FluidX3D)** : stocker les populations **décalées**
f_i − w_i·ρ₀ (et idem g_i − w_i^t·ρ₀·T̄) au lieu de f_i. On récupère l'essentiel
de la précision sur le non-équilibre.

**Portes de validation précision** (avant tout run Ra=1e10 GPU) :
- Ra=1e6 GPU vs CPU : Nuc écart ≤ 1 %, masse conservée à ~1e-5 (f32 : pas 1e-12),
  stratification/w̄ comparables
- Puis Ra=1e8, puis Ra=1e10 : Nuc, w̄(z=0.5), ν_t_max vs CPU à tolérance

## 3. Architecture proposée

- **CPU = référence** : plus d'identité bitwise CPU↔GPU — comparaison avec
  tolérance (portes §2), pas de diff binaire.
- **Kernel fusionné** collision TRT + streaming pull + bounce-back + conditions
  thermiques (la structure `collide_and_stream` s'y prête : un passage par nœud).
- **Réductions** (Nu, masse, w_max) : sur GPU ou rapatriées tous les N pas
  (éviter la synchro chaque pas).
- **`--dynamic-smag`** : kernel séparé (filtrage box sur les voisins — pattern
  différent du kernel fusionné), même garde-fou τ_floor.
- **Snapshots/restart** : compatibles avec le format actuel (conversion
  f32↔f64 à l'export/import, champs macroscopiques identiques).

## 4. Chemin d'évidence (portes, comme le plan octree)

| # | Claim | Évidence minimale |
|---|---|---|
| G1 | Le kernel GPU reproduit le CPU (Ra=1e6) | Nuc écart ≤ 1 %, champs visuellement identiques, w̄ max |
| G2 | La précision f32+offset suffit | idem à Ra=1e8 (f^neq plus grand) — écarts ≤ 2 % |
| G3 | Le gain est réel | MLUPS mesuré sur 64×256 et 256×1024 : ≥ 10× le CPU (32 MLUPS) |
| G4 | Le dynamique survit au GPU | smoke --dynamic-smag Ra=1e6 : κ_dyn ∈ [0, 0.04], pas de NaN |
| G5 | Ra=1e10 complet | Nuc vs CPU (55.20 fixe / 101.19 256×1024) à tolérance, run < 1 h |

## 5. Ordre de développement

1. **Étape A (G1)** : backend wgpu minimal — kernel fusionné f32+offset, Ra=1e6
   vs CPU. Porte : G1.
2. **Étape B (G2)** : Ra=1e8, tolérances précision. Porte : G2.
3. **Étape C (G3)** : benchmark MLUPS propre (les 2 grilles). Porte : G3.
4. **Étape D (G4)** : port du mode dynamique (kernel séparé). Porte : G4.
5. **Étape E (G5)** : Ra=1e10 complet — le gain est l'objectif final. Porte : G5.

## 6. Risques

| Risque | Symptôme | Parade |
|---|---|---|
| f32 insuffisant à τ→0.5 | Nuc dérivé, bruit | offset f_i − w_iρ₀ (§2) ; si insuffisant : f64 émulé (lent) ou hybrid CPU f64/GPU f32 par comparaison |
| Bounce-back TRT en WGSL | Λ différent du CPU | même formule exacte portée en f32 ; valider sur Poiseuille |
| Déterminisme GPU | résultats non reproductibles | atomiques flottantes interdites ; réductions arborescentes fixes |
| Dynamic-smag filtre GPU | désynchro voisins | kernel séparé avec texture/buffer voisinage, même stencils |
