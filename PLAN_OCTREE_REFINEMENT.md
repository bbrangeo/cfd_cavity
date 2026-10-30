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
2. **Rescaling du non-équilibre** : f^neq_fin = ξ · f^neq_grossier avec :
   - **ξ = τ_fin/τ_grossier** (facteur de rescaling visqueux standard, Lagrava et al. 2015,
     Rohde et al. 2006) — préserve le flux de quantité de mouvement à travers le
     changement de viscosité (τ ∝ ν et le f^neq doit être rescalé par le même rapport
     pour que ν·∂u reste continu à l'interface).
   - **TRT (Λ=3/16)** : ξ porte sur les composantes **odd/even séparément** :
     ξ_odd = τ⁺_fin/τ⁺_grossier, ξ_even = 1 (première approximation ; le facteur
     précis pour les composantes even est ξ_even = (τ⁻/τ⁺)_fin/(τ⁻/τ⁺)_grossier si
     on veut préserver exactement la relaxation — à valider numériquement, cf. §4.3).
   - **Double distribution (T scalaire)** : même rescaling pour g^neq avec
     ξ_thermal = τ_t,fin/τ_t,grossier (ν_t/α_t local inclus : τ effectif par nœud,
     cf. §2.3).
3. **Conservation explicite** : après transfert, corriger ρ_fin pour que la masse
   totale soit conservée (comme le garde-fou `masse = 1.000018` actuel) et u_fin
   pour la quantité de mouvement.

### 1.3 Smagorinsky/dynamique à travers l'interface

- ν_t est calculé **par nœud** avec Δx local : Δ_fin = Δx/2 → νsm_fin ∝ (Δx/2)²|S|.
  Le saut de ν_t à l'interface est physique (le filtre change), mais **doit être
  lissé** : moyenne box sur 2–3 nœuds de part et d'autre de l'interface (régularisation
  du même type que l'Éq. 2.62 déjà implémentée).
- Mode dynamique (--dynamic-smag) : ν_t/α_t rescalés aussi (τ effectif change),
  même garde-fou τ_floor par niveau.

## 2. Architecture proposée (crate séparé)

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
