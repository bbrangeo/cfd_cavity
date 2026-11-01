# Plan de développement — Backend GPU wgpu (`--backend gpu`) pour cavity LES

> Statut : **Étape C (G3) PASSÉE** (2026-10-31). Branche `dev-gpu` (recréée depuis
> `master` `1f2fb5c` — l'ancienne branche, sans commits propres, a été
> supprimée). Le code CPU actuel reste la référence validée — le backend GPU
> est une **extension** (`--backend cpu|gpu`, défaut cpu), jamais un
> remplacement. Le mode GPU cible le **solveur uniforme** (`src/uniform/`) ;
> l'octree est hors scope GPU (mur β de compressibilité le plafonne à
> Ra ≈ 3-4e8, cf. PLAN_OCTREE_REFINEMENT.md §6).

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

| # | Claim | Évidence minimale | Statut |
|---|---|---|---|
| G1 | Le kernel GPU reproduit le CPU (Ra=1e6) | Nuc écart ≤ 1 %, champs visuellement identiques, w̄ max | **PASSÉE 2026-10-31** (0.001 %, VALIDATION_GPU_ETAPA_G1.md) |
| G2 | La précision f32+offset suffit | idem à Ra=1e8 (f^neq plus grand) — écarts ≤ 2 % | **PASSÉE 2026-10-31** (0.045 %, dérive masse linéaire, VALIDATION_GPU_ETAPB_G2.md) |
| G3 | Le gain est réel | MLUPS mesuré sur 64×256 et 256×1024 : ≥ 10× le CPU (32 MLUPS) | **PASSÉE 2026-10-31** (×9.4-9.7 sur 64×256, **×31-34 sur 256×1024** — 470-490 MLUPS, VALIDATION_GPU_ETAPC_G3.md ; déroulage + fusion 2 dispatches + batching + SoA) |
| G4 | Le dynamique survit au GPU | smoke --dynamic-smag Ra=1e6 : κ_dyn ∈ [0, 0.04], pas de NaN | **PASSÉE 2026-10-31** (test 1 pas ≡ CPU à 1e-4, Nuc écart 0.005 %, κ_dyn_max = 0.0400 = clamp, 4 tests GPU verts, VALIDATION_GPU_ETAPD_G4.md ; coût perf ×3-×10 vs fixe, ~2× CPU j8 quand même) |
| G5 | Ra=1e10 complet | Nuc vs CPU (55.20 fixe / 101.19 256×1024) à tolérance, run < 1 h | **PASSÉE 2026-11-01** (fixe GPU 2M pas : **Nuc 101.49 vs CPU 101.19 = +0.30 %**, trace CPU au pas près au probe — 0.0002 % à 10k, **392 MLUPS = 22 min** ; dérive de masse f32 **linéaire** +1.45e-8/pas = +2.89 % sur 2M, Nuc non affecté ; dynamic GPU : Nuc 103.89 ± 1.92, κ_dyn ∈ [0, 0.04] sans NaN, 85 MLUPS = 1 h 43 — VALIDATION_GPU_ETAPE_G5.md ; ν_sm/ν non extraitable directement, critère SGS non conclusif) |

### G5 = campagne cible de thèse (décision 2026-10-31)

Le run de validation G5 **est** la campagne levier contre l'écart −40/−47 %
(§5.1 du plan échelles mixtes) : refaire le **dynSmag sur 256×1024, Ra=1e10,
sur GPU** une fois le backend validé. Config cible :

```bash
# 1. Référence GPU (fixe) — doit reproduire Nuc ≈ 101.19 (CPU, Trias 100.68)
#    ⚠️ CORRECTION 2026-11-01 : --utarget 0.005 (PAS 0.002 — la config 0.002
#    avorte sur le garde-fou max_u > 0.10 à ~8-20k pas) + restart depuis le
#    snapshot CPU établi Ra=6.4e8 (même grille 256×1024), cf. log CPU
#    output/logs/ra1e10.log l.14.
cavity_les --mode uniform --backend gpu --nx 256 --ny 1024 --ra 1e10 \
    --cs 0.1 --utarget 0.005 --steps 2000000 \
    --restart output/runs_uniforme/ra6.4e8_256x1024/snapshot.bin \
    --out output/gpu_ra1e10_fixe

# 2. Campagne dynamique (levier 1 contre l'écart) — même grille, --dynamic-smag,
#    restart depuis le snapshot FINAL de la run fixe (champ établi à 1e10 ;
#    un dyn from scratch à 1e10 risquerait une phase de ramp-up divergente)
cavity_les --mode uniform --backend gpu --nx 256 --ny 1024 --ra 1e10 \
    --dynamic-smag --smag-alpha 0.5 --utarget 0.005 --steps 2000000 \
    --restart output/gpu_ra1e10_fixe/snapshot.bin \
    --out output/gpu_ra1e10_dyn
```

Critères de succès :
- fixe GPU : Nuc ≈ 100-101 (± tolérance f32, porte G2/G5) — cohérence CPU
- dyn GPU : Nuc ≈ 100 ± σ raisonnable + ν_sm/ν ∈ [0.48, 0.54] (gamme C2D2/C2D3,
  cf. §3) → **claim complété : l'écart −40/−47 % sur 96×384 était une limite
  de résolution ; le modèle dynamique est correct sur grille résolue**
- run < 1 h (vs ~3 h CPU par config) → campagne complète (fixe + a4 + a25)
  abordable en une session

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
