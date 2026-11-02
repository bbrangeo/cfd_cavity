# Plan de développement — Solveur 3D D3Q19 GPU + comparaison DNS Trias

> Statut : **V3 PASS (2026-11-02, `VALIDATION_3D_ETAPE_V3.md`)** — après
> **V2 PASSED (2026-11-02, `VALIDATION_3D_ETAPE_V2.md`)** et
> **V1 PASSÉE (2026-11-01, `VALIDATION_3D_ETAPE_V1.md`)**. Objectif
> de thèse noté le 2026-10-31 : **vraie LES-LBM 3D avec GPU, comparaison à
> DNS Trias**. Déclenchement : après la validation du backend GPU D2Q9
> (Étapes B→E, PLAN_GPU_BACKEND.md) — le kernel 3D se porte sur la même
> infrastructure wgpu. V2 FAITE (128×512×256, Nuc 51.06 = +3.7 %, PASS,
> 585 MLUPS RTX 6000). Prochaine étape : V3 (LES Ra=1e9, sensibilité Cs,
> restart 3D à brancher si redémarrage à chaud).

## 0. Pourquoi le 3D (résumé de la décision)

En 2D (D2Q9 actuel), pas de cascade turbulente → pas de contenu SGS
substantiel à modéliser : le Smagorinsky y joue un rôle de **stabilisateur
near-wall** (ν_t/ν ≤ 2.3), pas de modélisation. Une « vraie » LES — grille ne
résolvant qu'une fraction des échelles, SGS portant le reste — est
**structurellement impossible en 2D**. Il faut du 3D (D3Q19 minimum), où des
écoulements turbulents réels avec cascade existent et où le SGS a un contenu
physique à porter.

## 1. Objectif scientifique

- Solveur LBM **D3Q19 + TRT + Guo + Smagorinsky** en 3D, backend GPU wgpu
  (réutilise l'infrastructure Étape A : pipelines collide/stream/macroscopic,
  ping-pong, offset FluidX3D).
- Cavité différentiellement chauffée **3D**, Ra cible 1e8 → 1e9.
  **Cible : réplique Trias Case A (JFM 586, 2007) = cavité AR=4** (largeur 1 ×
  hauteur 4, **x1 périodique L1=2** — pas cubique : le même papier traite
  AR=4 en 2D/3D ; les Ra jusqu'à 1e11 sont dans Trias et al., Part I & II,
  IJHMT 53(4) 2010 — attribution vérifiée 2026-11-02). Cohérent avec
  tout le chapitre 2D (256×1024 = AR4), et la dimension périodique est plus
  simple en LBM (wrap, pas de bounce-back : 4 murs seulement, les 9 vitesses
  à wz=0 se traitent exactement comme le D2Q9 actuel).
- **Comparaison à DNS Trias 3D** (Trias, Soria, Oliva, Pérez-Segarra,
  JFM 586 2007, Table 2 p. 271 — Pr=0.71, latérales isothermes, horizontales
  adiabatiques, clustering tanh(γ) non reproductible en LBM uniforme, comme
  en 2D) : **Nuc 3D = 49.24 (Ra=6.4e8) vs Nuc 2D = 49.23** ; 101.70 vs
  100.60 (1e10). **⚠️ Correction factuelle (cadrage Oracle 2026-11-01)** :
  à AR=4, Nuc 3D n'est pas « plus faible » que 2D — il est légèrement
  supérieur ; ce qui diffère 2D/3D est la **dynamique** (quasi-périodique vs
  chaotique, σ des Nulocal), pas le Nuc moyen.
- Claim visé : le modèle SGS a un **contenu physique réel en 3D** (ν_sm/ν
  rapporté, hiérarchie grille/modèle), contrairement au 2D où il ne
  stabilise que la BL.

## 2. Ce qui change du 2D → 3D (coût de portage)

| Élément | 2D (D2Q9) | 3D (D3Q19) | Effort |
|---|---|---|---|
| Lattice | 9 vitesses | 19 vitesses | kernel WGSL : poids/equilibres/voisins 3D |
| Streaming | pull 2D | pull 3D (19 directions) | réécriture kernel stream |
| Force Guo | odd/even 2D | odd/even 3D | formules §2.3 thèse généralisées |
| CL parois | 4 murs | 6 murs + coins/arêtes | bounce-back 3D |
| Smagorinsky | \|S\| 2D | \|S\| 3D (9 composantes) | formule standard |
| Grille | nx×ny | nx×ny×nz | mémoire : ×nz |
| GPU | Étape A validée | même infrastructure | portage kernel (la structure wgpu reste) |

## 3. Coût mémoire/calcul (⚠️ corrigé 2026-11-01 — le plan initialement
## sous-estimait ×3)

L'architecture 2D réelle (`gpu.rs`) garde **3 jeux de populations** (f_in/
f_out ping-pong + f_col, idem g) = **2×19×3×4 = 456 octets/nœud** (pas 152).
- Ra=6.4e8 3D AR4 (Trias Case A) : grille LBM uniforme **96×384×192** =
  7.1M nœuds ≈ **3.2 Go** — tient.
- Grille affinée 128×512×256 = 16.8M nœuds ≈ **7.6 Go** — tient sur M1 Max
  64 Go.
- Ra=1e10 AR4 (Trias Case C) : ~61 Go f32 — **hors budget** : le plafond
  LES 3D (V3) est Ra=1e9 sur 128×512×256, ou cubic 256³ (7.6 Go).
- CPU j8 à titre de comparaison : ~30× plus lent — GPU obligatoire.

## 4. Étapes proposées (portes V1-V3, GPU-first — décision 2026-10-31)

> **Décision 2026-10-31** : pas de solveur 3D CPU préalable — le solveur
> D3Q19 est développé **directement sur GPU** (kernel-first, même logique que
> l'Étape A 2D). Un solveur CPU 3D j8 serait ~30× plus lent : chaque run de
> développement coûterait des heures là où le GPU coûte des minutes. La
> validation reste possible : les tests unitaires portent sur 1 pas / champs
> initial (comme les 3 tests GPU 2D), et la référence de comparaison n'est
> pas un CPU maison mais **DNS Trias 3D** (littérature).

| # | Étape | Porte | Évidence minimale |
|---|---|---|---|
| 1 | Solveur 3D **GPU** (D3Q19, TRT, Guo, Smagorinsky) + tests unitaires 1-pas | V1 | **✅ PASSÉE 2026-11-01** : Ra=6.4e8 GPU 96×384×192, Nuc 49.98 vs Trias 49.24 (**+1.5 % ≤ 5 %**), 200k pas sans NaN, dérive masse 0.01 % (0.03 %/500k ≪ garde-fou 2 %), 62.8 MLUPS, 9 tests GPU verts (5×3D dont Guo analytique ratio 1.000) — `VALIDATION_3D_ETAPE_V1.md` |
| 2 | Ra=1e8 3D DNS-like (grille résolue) | V2 | Nuc vs Trias 3D (grille qui résout la BL) |
| 3 | Ra=1e9+ 3D LES (SGS actif, sous-résolution volontaire) | V3 | ν_sm/ν rapporté, sensibilité Cs, comparaison DNS Trias |

## 5. Dépendances et ordonnancement

- **Pré-requis** : Étapes B→E du PLAN_GPU_BACKEND.md (backend 2D validé) —
  la structure wgpu (pipelines, ping-pong, offset FluidX3D, portes) est
  réutilisable tels quels, seuls les kernels changent (D2Q9→D3Q19, 2D→3D).
- **Branche** : `dev-3d` depuis `dev-gpu` après G5.
- **Estimation** : ~2-4 semaines (solveur GPU 3D ~1-1.5 sem, validation
  V1-V2 ~1 sem, campagne V3 ~0.5-1 sem). À affiner après V1.

## 6. Risques

1. **Précision f32 à 3D** : même parade FluidX3D (offset). La dérive 2D
   (+1.45e-8/pas, linéaire, Nuc insensible au premier ordre — démontré sur
   2M pas à 1e10) se transposera en 3D : run V1 ~0.5-1M pas → +0.7-1.5 % de
   masse, acceptable. **Décision par garde-fou, pas par anticipation** : si
   pente > 2 %/500k pas sur la run V1, renormalisation Σf périodique à V2
   (aux flush de snapshot, pas dans le batch — elle casse le batching).
2. **Coût mémoire à 1e10 AR4** : ~61 Go — hors budget ; V3 plafonné à
   Ra=1e9 (128×512×256) ou cubic 256³.
3. **Références 3D Trias** : config exacte confirmée (cadrage Oracle
   2026-11-01, JFM 586 2007 Table 1 p. 264 / Table 2 p. 271) : **AR=4, x1
   périodique L1=2**, Pr=0.71,
   latérales isothermes, horizontales adiabatiques, clustering tanh(γ) —
   non reproductible en LBM uniforme (comme en 2D, où le bounce-back
   halfway d'ordre 2 a tout de même donné +0.30 % à 1e10 : argument
   réutilisable).
4. **Mur β en 3D ?** : le forçage Guo porte sur 3 axes — le mur de
   compressibilité 2D pourrait différer ; à documenter dès V1 (ρ_dev vs β).
5. **Pas de CPU de référence maison** : la validation repose sur les **5
   tests 1-pas device** (équilibre exact, streaming uniforme, conduction
   pure, conservation masse 10k, impulsion Guo analytique) + l'ancrage
   Trias Case A (Nuc 49.24 ± 5 %) + le cross-check interne 2D/3D au même
   Ra=6.4e8 (snapshot `ra6.4e8_256x1024` disponible).

## 7. Décision en attente

~~Le scope exact (chapitre thèse complet vs perspective) sera décidé après la
validation 2D complète (G5) et une lecture des références Trias 3D.~~
**Levée 2026-11-01** : G5 close (toutes portes G1-G5 passées), Trias 3D lu
(JFM 586 2007 ; attribution corrigée 2026-11-02 — pas de JFM 686 2011) —
chapitre thèse complet visé, Étape V1 lancée sur branche
`dev-3d` (cadrage Oracle : architecture `src/uniform3d/` copie-adapt,
D3Q19 SoA flat, 5 tests device, Nuc 49.24 ± 5 % à Ra=6.4e8).
