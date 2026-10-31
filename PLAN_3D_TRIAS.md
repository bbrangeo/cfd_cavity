# Plan de développement — Solveur 3D D3Q19 GPU + comparaison DNS Trias

> Statut : PLAN (non implémenté). Objectif de thèse noté le 2026-10-31 :
> **vraie LES-LBM 3D avec GPU, comparaison à DNS Trias**. Déclenchement :
> après la validation du backend GPU D2Q9 (Étapes B→E, PLAN_GPU_BACKEND.md) —
> le kernel 3D se portera sur la même infrastructure wgpu.

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
- Cavité différentiellement chauffée **3D**, Ra cible 1e8 → 1e9 (voire 1e10
  si le GPU le permet).
- **Comparaison à DNS Trias 3D** (références 3D de la littérature — à
  préciser : Trias et al. cavité 3D Ra 1e8-1e11, cf. références 2D actuelles
  Tab. 2.7 ; les valeurs 3D Nuc sont plus faibles que 2D à Ra fixé).
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

## 3. Coût mémoire/calcul (estimation à confirmer)

- Ra=1e8 3D cavité cubique : grille ~128×128×128 = 2.1M nœuds × 19 pops
  (f) + 19 (g) × 4 octets (f32) ≈ **320 Mo** — tient sur GPU (M1 Max 64 Go).
- Ra=1e9 : ~256×256×256 = 16.8M nœuds ≈ **2.5 Go** — tient.
- Ra=1e10 : ~512×512×512 = 134M nœuds ≈ **20 Go** — limite M1 Max, à évaluer
  après G3 (mesure MLUPS réelle).
- CPU j8 à titre de comparaison : ~30× plus lent — GPU obligatoire au-delà
  de 1e8.

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
| 1 | Solveur 3D **GPU** (D3Q19, TRT, Guo, Smagorinsky) + tests unitaires 1-pas | V1 | Ra=1e6-1e8 GPU : Nuc vs référence 3D littérature (± quelques %), stabilité 200k+ pas, tests 1-pas vs valeurs analytiques (conduction, équilibre) |
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

1. **Précision f32 à 3D** : même parade FluidX3D (offset), mais f^neq plus
   grand à Ra élevé — la porte V1 (1e6-1e8) protège.
2. **Coût mémoire à 1e10** : 20 Go f32 — peut nécessiter D3Q15 (moins de
   pops) ou grille asymétrique nz < nx.
3. **Références 3D Trias** : vérifier les configs exactes (aspect ratio 1:1:1
   vs 1:1:2, Pr, CL) avant V1 — les valeurs 2D ne sont pas transposables.
4. **Mur β en 3D ?** : le forçage Guo porte sur 3 axes — le mur de
   compressibilité 2D pourrait différer ; à documenter dès V1 (ρ_dev vs β).
5. **Pas de CPU de référence maison** : la validation repose sur les tests
   1-pas (valeurs analytiques : conduction, équilibre, conservation) + la
   comparaison DNS Trias — les 2 seuls garde-fous indépendants. Renforcer les
   tests unitaires 3D (équilibre exact, masse, conduction pure) pour
   compenser l'absence de run CPU de comparaison.

## 7. Décision en attente

Le scope exact (chapitre thèse complet vs perspective) sera décidé après la
validation 2D complète (G5) et une lecture des références Trias 3D.
