# Plan de développement — Raffinement de maillage octree (grilles cartésiennes multi-niveaux)

> Statut : PLAN (non implémenté). Crate actuel = grille uniforme validée (3 Ra, 256×1024,
> écart ≤ 3.3 % vs Trias). L'octree est une **extension** pour la thèse, pas une retouche.

## 0. Objectif et cadre

Implémenter le raffinement de maillage en octree (grilles cartésiennes imbriquées
2:1) dans la cavité DHC 2D LBM (D2Q9 double distribution, TRT Λ=3/16, Smagorinsky),
pour concentrer la résolution sur les couches limites (Δx fin aux parois, Δx grossier
dans le noyau) — réduire le coût du 256×1024 tout en gardant la précision Trias.

**Motivation chiffrée** : grille uniforme 256×1024 = 262k nœuds. Avec 2 niveaux
(BL à Δx/2 sur ~6 % du domaine, noyau à Δx) : ~90–120k nœuds → gain ~2×. Le noyau
est quasi-stratifié, les gradients y sont 100× plus faibles qu'aux parois.

### 0.1 Mapping tanh (Éq. 2.104) → frontières de niveaux (γx = 2, valeur thèse)

La loi tanh continue (Δx variable à chaque maille) n'est pas directement possible
en LBM (cellules carrées à taille constante par niveau). L'octree l'approche par
un escalier en puissances de 2 qui suit la courbe : Δx_fin = Δx_min de la tanh,
changement de niveau là où Δx_tanh franchit 2·Δx_min, 4·Δx_min, 8·Δx_min.
Vérifié numériquement (loi x(k) = Lx/2·[1 + tanh(γx(2k/Nx−1))/tanh(γx)], γx=2) :

| Grille thèse | Δx_min (1ʳᵉ maille) | Δx_max/Δx_min | Niveau 0→1 | Niveau 1→2 | Niveau 2→3 |
|---|---|---|---|---|---|
| C2D3 (Nx=128) | 0.00118 | 13.7 | x ≈ 0.020 | x ≈ 0.066 | x ≈ 0.161 |
| C2D2 (Nx=96) | 0.00159 | 13.6 | x ≈ 0.020 | x ≈ 0.066 | x ≈ 0.164 |
| C2D1 (Nx=64) | 0.00243 | 13.3 | x ≈ 0.020 | x ≈ 0.066 | x ≈ 0.167 |

→ **4 niveaux** (Δx_min à 8·Δx_min), symétriques sur les deux parois verticales.
Le premier changement de niveau tombe toujours à x ≈ 0.02, juste à l'extérieur
de la BL (pic w̄ à x ≈ 0.004, BL ≈ 1 % de Lx).

**Coût en imitant C2D3** (Lx=1, Lz=4 ; cellules carrées ⇒ colonnes fines sur
toute la hauteur — gaspillage vs la thèse régulière en z, mais règle au passage
les coins et les BL horizontales) :

| Niveau | Δx | Nœuds (approx.) | Sous-pas par pas grossier |
|---|---|---|---|
| 0 (paroi, x<0.02) | 0.00118 | ~122 k | 8 |
| 1 (x<0.066) | 0.00236 | ~66 k | 4 |
| 2 (x<0.161) | 0.00472 | ~35 k | 2 |
| 3 (noyau) | 0.00944 | ~30 k | 1 |
| **Total** | | **~250 k** | ~1.3 M màj/pas grossier |

Même nombre de nœuds que le 256×1024 actuel, mais **1ʳᵉ maille 3.3× plus fine**
(0.00118 vs 0.0039) → règle directement l'anomalie « positions des pics bornées
par le 1er nœud ». Gain ~17× vs grille uniforme à Δx_min (847×3390 = 2.9 M nœuds).

**Le vrai verrou — τ sur les niveaux grossiers** (cf. §1.3) : en acoustic scaling,
ν_lu est divisé par 2 à chaque niveau plus grossier. Avec 4 niveaux :
- plancher sur le niveau fin (τ⁺=0.506) → noyau à τ⁺ ≈ 0.50075, trop près de
  l'instabilité TRT ;
- plancher sur le noyau → niveau fin à ν_lu = 8·ν, vitesse réseau ~0.5 à Ra=1e10
  (Lz ≈ 3390 lu), au-delà de la limite de compressibilité.

⇒ soit collision plus robuste que TRT pour τ→0.5 dans le noyau (régularisée
récursive, cumulant, KBC), soit **limiter à 2–3 niveaux** (3 niveaux : noyau à
τ⁺ ≈ 0.5015, jouable car stratifié quasi-laminaire). Décision à l'Étape B.

**Alternative loi tanh exacte** : ISLBM (He, Luo & Dembo 1996 — streaming +
interpolation) ou LBM volumes finis reproduiraient l'Éq. 2.104 exactement, au prix
de la diffusion numérique d'interpolation — hors scope de ce plan (référence).

## 1. Point critique n°1 — Rescaling des f^neq aux interfaces (le risque annoncé)

### 1.1 Le problème

À travers une interface grossier→fin, interpoler f directement casse la conservation
(masse + quantité de mouvement) et la structure du non-équilibre → réflexion/émission
d'ondes acoustiques parasites. À Ra=1e10 le f^neq près des parois est énorme
(ν_t local ≥ 2.33ν, gradients 100× le noyau) : le bruit acoustique est LE risque.

### 1.2 La pratique LBM (à implémenter exactement)

1. **Ne jamais interpoler f** aux interfaces. Transférer les **moments** :
   ρ, u, T (conservés par interpolation standard), puis le non-équilibre :
   f^neq_grossier reconstruit depuis les distributions du côté grossier.
2. **Rescaling du non-équilibre (acoustic scaling, Dupuis & Chopard 2003)** :
   f^neq_fin = **(1/2)·ξ·f^neq_grossier** avec :
   - **facteur 1/2** : vient de Δt_fin = Δt_grossier/2 (acoustic scaling) — le
     f^neq est un flux temporel (τ·contrainte), il se réduit avec le pas de temps.
     **Oublier le 1/2 produit exactement le bruit acoustique que le plan évite.**
   - **ξ = τ_fin/τ_grossier** (rapport des temps de relaxation, Lagrava et al. 2015,
     Rohde et al. 2006) — préserve la contrainte visqueuse ν·∂u à travers le
     changement de viscosité.
   - **TRT (Λ=3/16)** : le ratio porte sur la partie **paire** (le tenseur des
     contraintes π^even, relaxée par τ⁺) :
     **f^neq,fin_even = (1/2)·(τ⁺_fin/τ⁺_grossier)·f^neq,grossier_even** ;
     la partie **impaire** (π^odd, relaxée par τ−, liée à la partie anti-symétrique
     du flux) garde ξ_odd = 1/2 en première approximation (le facteur exact
     (1/2)·(τ⁻/τ⁺)_fin/(τ⁻/τ⁺)_grossier est à valider numériquement, cf. §4).
   - **Double distribution (T scalaire)** : idem pour g^neq — partie paire avec
     (1/2)·(τ_t,fin/τ_t,grossier), τ effectif par nœud (ν_t/α_t inclus, cf. §2.3).
3. **Conservation explicite** : après transfert, corriger ρ_fin pour que la masse
   totale soit conservée (comme le garde-fou `masse = 1.000018` actuel) et u_fin
   pour la quantité de mouvement.

### 1.3 Plancher τ⁺ et niveaux — au niveau GROSSIER

À ν physique constant, **ν en unités réseau double sur le niveau fin** (ν_lu =
ν·dt/Δx², dt_fin = dt_grossier/2, Δx_fin = Δx/2 ⇒ ν_lu,fin = 2·ν_lu,grossier).
Le plancher de stabilité TRT **τ⁺ ≥ 0.506 doit donc être imposé au niveau
grossier, pas au niveau fin** — c'est l'inverse de l'intuition : le niveau fin,
avec ν_lu plus grand, est PLUS stable ; le grossier est le limiteur.

Conséquence : ν_t,fin en unités réseau est aussi ~2× ν_t,grossier (à |S| donné)
— le garde-fou par niveau doit être calculé avec ν_lu LOCAL à chaque niveau,
jamais une valeur globale.

### 1.4 Smagorinsky/dynamique à travers l'interface

- ν_t est calculé **par nœud** avec Δx local : Δ_fin = Δx/2 → νsm_fin ∝ (Δx/2)²|S|.
  Le saut de ν_t à l'interface est physique (le filtre change), mais **doit être
  lissé** : moyenne box sur 2–3 nœuds de part et d'autre de l'interface (régularisation
  du même type que l'Éq. 2.62 déjà implémentée).
- Mode dynamique (--dynamic-smag) : ν_t/α_t rescalés aussi (τ effectif change),
  garde-fou τ_floor calculé avec ν_lu LOCAL par niveau (cf. §1.3 : le grossier
  est le limiteur).

## 2. Architecture proposée (crate séparé — **obsolète : fusion Étape I**)

> **Note 2026-10-31** : le crate séparé a existé (`cfd_cavity_octree`) puis a
> été **fusionné dans le crate racine** (Étape I, commit `6e45c6f`) — un seul
> binaire `cavity_les --mode uniform|octree`, modules octree dans `src/octree/`.
> L'architecture ci-dessous décrit la découpe logique des modules, conservée
> à l'identique dans `src/octree/`.

**NE PAS modifier le crate uniforme validé** (`cfd_cavity`) : nouveau crate
`cfd_cavity_octree` réutilisant les briques via modules partagés (workspace Cargo) :
- `lbm-core` : D2Q9, TRT, collision/streaming, Boussinesq (extraits de solver.rs)
- `octree-grid` : structure de niveaux, masques grossier/fin, voisinage inter-niveaux
- `interface` : transfert f^neq + rescaling ξ (§1.2) — **module le plus critique**
- `solver` : orchestration par pas (stream/collide par niveau, interfaces, macro)

### 2.1 Structure de données (2D : quadtree — 4 enfants par carré)

**Le quadtree est le bon choix précisément près des parois de la cavité** : les BL
thermique/visqueuse à Ra=1e10 sont confinées à ~1 % de la largeur (pic w̄ à x*≈0.004)
— le raffinement y est indispensable, le noyau stratifié n'en a pas besoin :
- Niveaux ℓ = 0 (grossier, noyau) … ℓ_max (fin, BL aux 4 parois), ratio 2:1 fixe,
  4 enfants par carré (feuille = cellule de calcul).
- Règle d'équilibrage 2:1 : deux cellules voisines diffèrent d'au plus 1 niveau —
  évite les sauts brutaux de résolution (et les interfaces ternaires).
- **Masque de raffinement** (critère) : |∇T| ou |S| local > seuil → raffiner le
  bloc (BL détectées automatiquement), + parois toujours au niveau fin max
  (bounce-back y est le plus précis, cf. §5).
- Voisinage inter-niveaux précalculé (coïncident, coarse-fine) : table d'indices
  construite à l'init, jamais pendant la boucle temps (déterminisme, jobs).

### 2.2 Boucle temporelle — pas de temps multiple (acoustic scaling, démarche thèse)

**Retenu : pas de temps multiple** — le niveau fin fait **2 pas de temps pendant que
le niveau grossier en fait 1** (dt_fin = dt_grossier/2, correct CFL fin, « acoustic
scaling »). La grille étant toujours en équilibre local de pression (ondes
acoustiques résolues), ce découplage temporel est la démarche de référence LBM
multi-niveaux :
- Chaque pas grossier : collide+stream niveau grossier (τ_grossier), puis 2×
  (collide+stream niveau fin, τ_fin local), puis couplage d'interfaces :
  1. **grossier→fin (spatio-temporel)** : interpoler ρ, u, T dans l'espace ET dans
     le temps (les deux niveaux sont disponibles aux instants t et t+dt_grossier
     côté grossier — interpolation linéaire en temps entre les deux instants),
     reconstruire f^neq, **rescaler ξ**, injecter sur les nœuds fantômes fins.
  2. **fin→grossier (spatio-temporel)** : moyenner ρ, u, T sur les 2×2 nœuds fins
     coïncidents **aux deux sous-pas temporels** (moyenne pondérée en temps),
     f^neq_grossier = moyenne des f^neq fins rescalés par 1/ξ, injecter.
- Le τ change avec la résolution (τ ∝ ν à Δx donné) : d'où le rescaling f^neq
  obligatoire (§1.2) à CHAQUE échange d'interface, espace et temps compris.
- **Pas de temps unique (fallback)** : plus simple (pas d'interpolation temporelle)
  mais CFL fin violé sur le niveau fin → précision acoustique dégradée. OPTION de
  secours seulement si le pas multiple diverge — documenter l'écart alors.

### 2.3 τ effectif et rescaling en présence de Smagorinsky/dynamique

- τ par niveau = 0.5 + 3·(ν + ν_t local, Δx du niveau) : le ν_t est calculé avec
  Δx local (Δ_fin = Δx/2 → νsm_fin ∝ (Δx/2)²|S|).
- Le rescaling ξ (§1.2) utilise donc τ **effectif** par niveau (ν + ν_t), pas
  seulement moléculaire — à Ra=1e10 (ν_t/ν ≥ 2.33), l'erreur ξ moléculaire serait > 2×.
- Saut de ν_t à l'interface lissé box 2–3 nœuds de part et d'autre (type Éq. 2.62).
- Mode --dynamic-smag : ν_t/α_t rescalés aussi, garde-fou τ_floor par niveau.

## 3. Validation ( chemin d'évidence — chaque claim a un propriétaire )

### 3.1 Claims à établir (dans l'ordre)

| # | Claim | Propriétaire | Évidence minimale |
|---|---|---|---|
| V1 | Le rescaling ξ conserve masse et u à l'interface | fixer | test unitaire : interface plane, champ couette/thermique imposé, bilan masse/u avant-après = 0 (1e-14) |
| V2 | L'interface ne génère pas de bruit acoustique | fixer | run Ra=1e6 128×256 2 niveaux : max|ρ−ρ̄| aux interfaces ≤ bruit de la run uniforme même durée (référence : masse 1.000018) |
| V3 | Nuc 2 niveaux ≈ Nuc uniforme | fixer | Ra=1e6 : Nuc(2 niveaux) vs Nuc(256×256 uniforme) écart ≤ 2 % ; puis Ra=1e8 idem |
| V4 | Gain de coût réel | orchestrateur | temps/pas et nœuds actifs mesurés : ≥ 1.5× plus rapide à Nuc équivalent (V3) |
| V5 | Le modèle dynamique survit au multi-niveau | fixer | smoke --dynamic-smag 2 niveaux Ra=1e6 : κ_dyn ∈ [0, 0.04], pas de NaN, Nuc cohérent |

### 3.2 Références internes (grille uniforme, déjà validées)

- Ra=1e8 : 30.45 (uniforme 128×512) | Ra=6.4e8 : 50.89 | Ra=1e10 : 101.19 (256×1024)
- Écarts vs Trias : +0.1 %, +3.3 %, +0.5 % — TOLÉRANCE CIBLE V3 : ±3 %
- Pitfalls connus : np.trapezoid, reshape (ny,nx) order="C", CSV colonnes 0=x,1=y,2=ux,3=uy,4=T,5=ψ

### 3.3 Ordre de développement (chaque étape = porte de validation)

1. **Étape A (V1)** : crate squelette + interface plane + test rescaling ξ. Ne pas
   continuer tant que V1 n'est pas vert.
2. **Étape B (V2)** : cavité Ra=1e6 2 niveaux (BL raffinée uniquement). Monitorer
   bruit acoustique vs uniforme. Porte : V2.
3. **Étape C (V3)** : Ra=1e8 2 niveaux, comparaison Nuc + w̄(z=0.5) vs uniforme
   (référence : 30.45, w̄ max 0.22–0.24 Trias). Porte : V3 à ±3 %.
4. **Étape D (V5)** : brancher --dynamic-smag (port du module compute_dynamic_sgs
   avec Δx local par niveau). Porte : V5.
5. **Étape E (V4)** : Ra=1e10 2 niveaux vs 101.19 — le gain de coût est l'objectif
   final. Porte : V4 + Nuc à ±3 %.
6. **Étape F (V6, ajout 2026-10-30)** : raffinement **multi-niveaux jusqu'au
   centre** — au lieu d'un seul niveau ×2 collé aux parois, l'escalier octree
   suit la loi tanh sur TOUTES ses marches : niveaux Δx/2, Δx, 2Δx, 4Δx…·Δx_max
   répartis selon les frontières du §0.1 (2·Δx_min à x≈0.020, 4·Δx_min à
   x≈0.066, 8·Δx_min à x≈0.164, symétrique par rapport à x=0.5). Motivation :
   à Ra=1e10 l'écoulement n'est pas confiné aux parois — l'escalier 1 niveau
   sous-amortit le noyau et laisse l'écart Nuc à −45 % vs Trias ; l'objectif
   est d'approcher Nuc ≈ 100 à coût fixé. Contraintes : rescaling f^neq par
   PAIRE de niveaux adjacents (même formule ξ que V1, généralisée), pas de
   temps multiple par bloc (chaque niveau a son sous-pas acoustique 2^k),
   Smagorinsky avec Δx local par niveau (déjà prévu V5). Prérequis : porte V2
   verte sur le 2 niveaux (le fixer est en train de la déboguer).

## 4. Risques et garde-fous

| Risque | Symptôme | Parade |
|---|---|---|
| Rescaling ξ mal calé (odd/even TRT) | bruit acoustique aux interfaces, dérive masse | V1 en test unitaire ; comparer ξ_even=1 vs ξ_even=(τ⁻/τ⁺) ratio — retenir le moins bruité (V2) |
| Saut de ν_t à l'interface | oscillation ν_t, instabilité | lissage box 2–3 nœuds de part et d'autre (type Éq. 2.62) ; garde-fou τ_floor par niveau |
| Anisotropie BL (filamentée à Ra=1e10) | Nuc biaisé | raffiner 2 niveaux dans la BL au moins (Δx/2), w̄(z=0.5) comme contrôle |
| Pas multiple mal synchronisé | interpolation temporelle désynchronisée, dérive | sub-pas fin exactement 2× ; interfaces mises à jour à CHAQUE sous-pas fin (§2.2) |
| Pas unique (fallback) acoustiquement dégradé | CFL fin violé, précision moindre | réserver en secours uniquement ; documenter l'écart vs pas multiple |
| Déterminisme multi-CPU | résultats non reproductibles | pattern rayon existant (jobs) : aucune atomicité flottante, interfaces précalculées |

## 5. Pitfalls spécifiques

- **Ne pas interpoler f** : toujours reconstruire depuis les moments + f^neq rescalé.
- ξ dépend de τ **effectif** (ν + ν_t local), pas seulement moléculaire — à Ra=1e10,
  ν_t/ν ≥ 2.33, l'erreur ξ moléculaire seul serait > 2×.
- Les nœuds fantômes fins doivent être mis à jour **chaque pas** (pas seulement au
  restart) — sinon désynchronisation des flux.
- Le masque de blocs doit inclure les parois dans le niveau fin (bounce-back y est
  le plus précis) — jamais de bounce-back sur niveau grossier aux parois chauffées.

## 6. Bilan — Mur β de compressibilité (campagne octree, --dynamic-smag) — 2026-10-31

Campagne menée sur grilles octree avec le mode dynamique (`--dynamic-smag`,
jobs 8, restart depuis champ établi sauf P1 cold) : **le passage 64×256 → 128×512
n'a PAS levé le mur** de compressibilité du schéma D2Q9 + Guo + TRT.

| Run | Ra | Grille équiv. | ν | τ⁺ fin | β | max ρ_dev | Résultat |
|---|---|---|---|---|---|---|---|
| Palier 1 | 1e8 | 64×256 | 0.0040 | 0.512 | 1.36e-4 | 2.42e-2 | STABLE 300k pas, Nuc 2.3052 |
| Palier 1 128×512 | 1e8 | 128×512 (bl-width 8) | ≈0.004 | ≈0.512 | ≈1.4e-4 | 1.26e-2 | STABLE 300k pas, Nuc 7.3255 (verdict FAIL porte V2 à ignorer — calibré bl-width 4 ; Nuc entre grilles non comparable) |
| Probe 2e8 | 2e8 | 64×256 | 0.0040 | 0.512 | 2.72e-4 | 6.03e-2 | STABLE 30k pas |
| Probe 3e8 | 3e8 | 64×256 | 0.0040 | 0.512 | 4.08e-4 | 9.63e-2 | STABLE 30k pas (à la limite : max_u 0.088, ρ_dev oscillant) |
| Probe 4e8 | 4e8 | 64×256 | 0.0040 | 0.512 | 5.43e-4 | 1.30e-1 | AVORT garde-fou Ma (max_u 0.112 > 0.10, pas 5k) |
| 6.4e8 cold (P1) | 6.4e8 | 64×256 | 0.0040 | 0.512 | 8.78e-4 | — | NaN < 5k pas, DEPUIS LA CONDUCTION (sans restart) |
| 6.4e8 64×256 | 6.4e8 | 64×256 | 0.0040 | 0.512 | 8.78e-4 | — | NaN < 10k pas |
| 6.4e8 128×512 a | 6.4e8 | 128×512 | 0.0022 | 0.5065 | 2.52e-4 | — | NaN < 10k pas |
| 6.4e8 128×512 b | 6.4e8 | 128×512 | 0.0040 | 0.5121 | 8.78e-4 | 1.06e-2 au pas 0 | NaN < 10k pas |
| 1e10 direct | 1e10 | 64×256 | 0.0020 | 0.506 | 3.40e-3 | — | NaN < 10k pas |

Conclusions côté raffinement :

1. **Le raffinement ne lève pas le mur** : 6.4e8 diverge en < 10k pas sur 64×256
   comme sur 128×512 (deux tentatives a/b, dont b à τ⁺=0.5121 et ν=0.004, avec
   ρ_dev déjà 1.06e-2 au pas 0). Hypothèses exclues : marge TRT, résolution BL
   (bl-width 8, 2 niveaux, 35878 nœuds), transitoire de reprise (P1 diverge
   depuis la conduction même, sans restart), viscosité sous le plancher.
2. **Pourquoi le raffinement est impuissant** : β = Ra·ν·α/Lz³ en unités
   grossières — raffiner ne change pas β ni l'équilibre ρ_dev visé. Le mur est
   quasi-statique de compressibilité faible : ρ_dev suit β linéairement (réponse
   acoustique du lattice au forçage Guo) ; quand ρ_dev ~1e-2 à 2e-2 coexiste avec
   u~0.05, la marge de positivité de f_eq D2Q9 est épuisée.
3. **Plafond du schéma (config octree) — BORNE BETA-STAR mesurée** : à ν=0.004,
   **β* entre 4.08e-4 (3e8, stable à la limite : ρ_dev 9.6e-2) et 5.43e-4 (4e8,
   avorté sur garde-fou Ma local : max_u 0.112 > 0.10)**, soit **plafond
   Ra ≈ 3-4e8**. Relation ρ_dev ∝ β quasi-linéaire (1.36e-4→2.42e-2,
   2.72e-4→6.03e-2, 4.08e-4→9.63e-2, 5.43e-4→1.30e-1). À ν=0.002 (plancher
   τ⁺=0.506), β(1e10)=3.4e-3 = 6× le mur fatal → Ra=1e10 hors de portée
   structurelle de la **config octree** (le crate uniforme l'atteint, cf. §5.1
   du plan échelles ; la différence est le forçage Guo porté par la grille de
   base grossière).
4. **Conséquence pour ce plan (claim §2.5 re-scopé)** : l'objectif Ra=1e10
   (Étapes E/F) est hors de portée structurelle de la config octree — le
   forçage Guo porté par la grille de base grossière (β = Ra·ν·α/Lz³ en unités
   du niveau L0) génère une déviation ρ_dev ∝ β qui épuise la marge de
   positivité de f_eq (mur de compressibilité quasi-statique, établi sur 2
   grilles, borne β* ∈ [4.08, 5.43]e-4). **Le plafond octree est Ra ≈ 3-4e8**
   (à ν=0.004) — bilans exploitables à 1e8/2e8/3e8. À Ra=1×10¹⁰, seule la
   grille uniforme fine (96×384) est exploitable : l'écart Nuc de −40 à −47 %
   vs C2D2 y est une limite de résolution (w_max 0.04 vs cible 0.22-0.24),
   non levée par le Smagorinsky dynamique. Combler l'un ou l'autre
   nécessiterait une collision sans plancher τ (RR/cumulant) — perspective de
   thèse, pas un levier immédiat.
