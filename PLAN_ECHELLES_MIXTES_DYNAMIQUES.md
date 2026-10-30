# Plan de dev — Modèles d'échelles mixtes dynamiques en Rust (§2.3.5 thèse)

> Statut : ÉQUATIONS FIGÉES (extraction pages PDF 80–85 = internes 49–54, décalage +31).
> Justification thèse : « les modèles d'échelles mixtes dynamiques sont particulièrement
> intéressants puisqu'ils permettent une bonne représentation du comportement des
> écoulements anisothermes d'air en cavités fermées. Par la suite, toutes nos SGE
> utiliseront une approche dynamique des modèles d'échelles mixtes (voir §2.3.5) »,
> α = 0,5 (Tab. 2.3). Extraction brute : `_work_pdf52/out2/` (pages 080–087).

## 1. Objectif

Remplacer (optionnellement, derrière un flag CLI) le Smagorinsky à Cs fixe du crate
`cfd_cavity` par un modèle d'échelles mixtes **dynamique** : Cνdyn/Cκdyn calculés par
la procédure dynamique de Lilly (identité de Germano + test-filter, régularisation par
moyenne Éq. 2.62, bornage), le volet « échelles mixtes » interpolant Smagorinsky ×
similarité qsm/Φsm avec α = 0,5 (Tableau 2.3) — équations exactes figées en section 3.

## 2. État de l'existant

### 2.1 Solver actuel (`src/solver.rs`)
- Collision TRT (Λ=3/16) + Smagorinsky : `collide_node` (~l.666), ν_t = (Cs·Δx)²·|S|
  avec |S| issu des moments non équilibrés π (pi_xx/pi_xy/pi_yy), ~l.722
- Cs fixe passé via `CollideCtx.cs` ; ν_t stocké par nœud (`self.nu_t`, l.47)
- Patch τ_t⁻ = τ_t⁺ (~l.730) : BGK pour le scalaire g — à préserver
- Chemin rayon sûr (~l.385) : scalaires copiés avant boucle parallèle, lignes f/g/nu_t
  exclusives par thread — **le calcul de Cs²(x,y) doit respecter ce pattern**
  (test-filter = lecture de u voisin → copier u avant la boucle, comme les scalaires)

### 2.2 Crate `oxiphysics-lbm` 0.1.2 (`turbulence/types.rs`)
- `DynamicSmagorinsky::new(dx, filter_ratio)`, `compute_cs_sq(s_bar, s_hat)` :
  **forme scalaire simplifiée** Cs² ≈ (s_hat − s_bar)/(α·s_hat²), clamp [0, 0.04],
  α = 2(β²−1)Δx², β = filter_ratio (typ. 2) ; test-filter = box average voisins
- ⚠️ **Insuffisante pour la thèse** : pas de contraction tensorielle Germano
  L_ij·M_ij / M_ij·M_ij, pas de volet similarité (Bardina), pas de split
  dynamique/coefficient par nœud sur champ complet. → **Implémentation propre** dans
  le crate projet, la crate ne servant que de référence d'API.

## 3. Équations de référence (§2.3.5 thèse, pages PDF 80–85) — FIGÉES

> Notation thèse : ũ = grandeur au filtre grille (Δ̃), u = grandeur au filtre test (Δ, η=2).
> C'est l'inverse de la convention usuelle — attention dans le code : **Δ̃ = filtre grille**,
> Δ = filtre test (Δ = 2Δ̃).

### 3.1 Viscosité dynamique (Cνdyn, Éq. 2.48–2.62)
- Tenseurs sous-maille (2.48–2.49) : τij = ũiuj − ũiũj ; Tij au filtre test
- Germano / Léonard (2.51) : **Lij = ⟨ũiuj⟩_test − ⟨ũ̃iũ̃j⟩_test** (Léonard au filtre test)
- Modélisation (2.52–2.53) : τdij = −2 Cνdyn Δ̃²|S̃|S̃ij ; Tdij = −2 Cνdyn Δ²|S|Sij
  (même Cνdyn aux deux niveaux)
- Mij (2.55/2.61, Smagorinsky dynamique, η=2) : **Mij = Δ²|S|Sij − Δ̃²|S̃|S̃ij
  = Δ̃²(|S|Sij − (1/4)|S̃|S̃ij)**
- Lilly (2.58) : Cνdyn = (1/2)·(Ldij Mij)/(Mij Mij), en chaque point, à chaque instant
- νsm (2.59) = Cνdyn Δ̃²|S̃|
- **Régularisation retenue par la thèse (2.62)** : moyenne ⟨•⟩ au numérateur ET
  dénominateur → Cνdyn = (1/2)·⟨Ldij Mij⟩/⟨Mij Mij⟩ ; Cνdyn ≥ 0 imposé
  (pas de cascade inverse ; borne négative possible : νsm ≥ −0.1·ν chez Ezzouhri 2007)

### 3.2 Flux thermique dynamique (Cκdyn, Éq. 2.63–2.71)
- Lj (2.65) = ⟨ujT⟩_test − ⟨ũjT̃⟩_test ; hj (2.66) = −Cκdyn Δ̃²|S̃| ∂T̃/∂xj
- Mj (2.69/2.73) = **Δ²|S|∂T/∂xj − Δ̃²|S̃|∂T̃/∂xj = Δ̃²(|S|∂T/∂xj − (1/4)|S̃|∂T̃/∂xj)**
- Lilly (2.70) : Cκdyn = (Lj Mj)/(Mj Mj) — même régularisation ⟨•⟩ et bornage
- κsm (2.71) = Cκdyn Δ̃²|S̃|

### 3.3 ★ Modèle d'échelles mixtes dynamiques (Tableau 2.3, α = 0,5 — vérifié sur capture du tableau)

Le modèle mixte interpole entre Smagorinsky (exposant α) et une forme en qsm/Φsm
(similarité, exposant 1−α), avec **α = 0,5 dans l'étude** :
- **νsm = C^ν_dyn·(Δ̃²|S̃|)^α·(Δ̃·(qsm)^{1/2})^{1−α}** — qsm = énergie cinétique sous-maille
- **κsm = C^κ_dyn·(Δ̃³/ΔT·|T̃|)^α·(Δ̃/ΔT·(Φsm)^{1/2})^{1−α}** — **ΔT = écart de
  température (T_c−T_f), PAS Δ̃·T**
- **Φsm (Éq. 2.74, capture)** : Φsm = (1/2)·φ′i·φ′i (sommation i∈{x,y}), avec
  **φ′i = (1/2)·(ũ_i − ū_i)·(T̃ − T̄)** → Φsm^{1/2} = (1/2)·|T̃−T̄|·√((1/2)·Σ(ũ_i−ū_i)²) ;
  Φ̄sm = test-filter de Φsm
- **qsm (par symétrie avec 2.74)** : qsm = (1/2)·u′_i·u′_i, u′_i = (1/2)(ũ_i−ū_i) →
  qsm^{1/2} = (1/(2√2))·√(Σ(ũ_i−ū_i)²) ; q̄sm = test-filter de qsm
- **|T̃| (Éq. 2.75, capture) — N'EST PAS |∇T̃|** : c'est le gradient de température
  **contracté** avec le tenseur des contraintes : T̃_i = (∂T̃/∂x_j)·S̃_ij, puis
  |T̃| = √(2·T̃_i·T̃_i). En 2D : T̃_x = ∂xT̃·S̃_xx + ∂yT̃·S̃_xy ;
  T̃_y = ∂xT̃·S̃_xy + ∂yT̃·S̃_yy ; |T̃| = √(2(T̃_x²+T̃_y²)). Idem au filtre test (S̄ij, T̄)
- Tenseurs Mij/Mj mixtes : différence des formes interpolées **au filtre test
  (tout est surbarré : Δ̄, S̄, q̄sm, Φ̄sm)** et **au filtre grille (tout est tildé)** :
  Mij = (Δ̄²|S̄|S̄ij)^α(Δ̄(q̄sm)^{1/2}S̄ij)^{1−α} − (Δ̃²|S̃|S̃ij)^α(Δ̃(qsm)^{1/2}S̃ij)^{1−α}
  Mj  = (Δ̄³/ΔT·|T̄|·∂T̄/∂xj)^α(Δ̄/ΔT·(Φ̄sm)^{1/2}·∂T̄/∂xj)^{1−α} − (même forme tildée)
- **C^ν_dyn = (1/2)·⟨L^d_ij M_ij⟩/⟨M_ij M_ij⟩** et **C^κ_dyn = (1/2)·⟨L_j M_j⟩/⟨M_ij M_ij⟩**
  — **le facteur 1/2 est présent pour LES DEUX** au Tableau 2.3 (capture confirmée)
- L_ij = ū_i ū_j − ũ̃_i ũ̃_j (Léonard au filtre test) ; L_j = ū_j T̄ − ũ̃_j T̃ (flux Germano)
- Côté code : α = 0,5 → chaque facteur interpolé devient une **moyenne géométrique**
  νsm = C^ν_dyn·√(Δ̃²|S̃| · Δ̃·qsm^{1/2}) = C^ν_dyn·Δ̃·√(|S̃|·qsm^{1/2})

### 3.4 Points d'attention pour l'implémentation (tranchés par la capture du Tab. 2.3)
- **C^κ_dyn a le facteur 1/2** au Tableau 2.3 (comme C^ν_dyn) : C^κ_dyn = (1/2)⟨L_j M_j⟩/⟨M_ij M_ij⟩ —
  le dénominateur « M_ij M_ij » est une **coquille du tableau** (c'est M_j M_j) ; le facteur
  1/2, lui, est réel et doit être implémenté (cohérent avec la symétrie de la démarche :
  le 1/2 vient du −2 devant les formes τdij/hj modélisées)
- **ΔT ≠ Δ̃·T** : la diffusivité utilise l'écart de température global ΔT = T_c−T_f (θc−θf = 1
  en adimensionnel), pas le produit filtre×température. En adimensionnel θ ∈ [−0.5, +0.5]
  → ΔT = 1 → les facteurs (Δ̃³/ΔT·|T̃|) et (Δ̄³/ΔT·|T̄|) simplifient en Δ̃³|T̃| / Δ̄³|T̄| —
  mais garder ΔT explicite dans le code pour la traçabilité thèse
- **q̄sm au filtre test** : le tableau distingue qsm (grille, tildé) et q̄sm (test, surbarré) —
  q̄sm = test-filter de qsm, pas une réestimation Bardina. Formes similaires pour Φsm/Φ̄sm.
  Estimation : qsm via la similarité (1/2)|ũ−ũ̃|² (cf. 2.74), puis test-filtrer qsm → q̄sm
- Propriétés de symétries (Razafindralandy 2005/2007) : l'approche dynamique et la
  similarité respectent les invariances de Navier-Stokes — argument de fond du choix

### 3.4 Forme adimensionnelle finale (Éq. 2.77–2.80, capture) — branchement scalaire
- **(2.79) quantité de mouvement** : diffusion = Pr·Ra_H^(−1/2)·(**1 + νsm/ν**) →
  code : ν_total = ν + νsm dans le τ⁺ vitesse (déjà le cas via ν_t)
- **(2.80) énergie** : diffusion = Ra_H^(−1/2)·(**1 + κsm/κ**) → code :
  **α_total = α + κsm** (κ_total = α·(1 + κsm/κ)) — le scalaire g doit recevoir
  la diffusivité turbulente κsm, même structure que la vitesse
- (2.79) flottabilité = Pr·θ̃·δiz ; Ra basé sur la HAUTEUR (Ra_H) — cohérent avec
  le code (beta_eff = g·β·ΔT, flottabilité non modifiée par le mode dynamique)

## 4. Architecture proposée

### 4.1 CLI (`src/cli.rs`)
- `--dynamic-smag` : active la procédure dynamique (défaut : off = Cs fixe inchangé,
  rétrocompatibilité bitwise des runs existants)
- `--smag-alpha <f64>` : coefficient α de l'interpolation mixte (défaut thèse : 0.5)
- `--nu-sm-min <f64>` : borne inférieure νsm/ν (défaut thèse Ezzouhri : −0.1 ;
  0.0 si on impose pas de cascade inverse, le plus sûr en LBM)
- `--eta-test <f64>` : ratio filtre test η = Δ/Δ̃ (défaut thèse : 2)

### 4.2 Solver (`src/solver.rs`)
1. **Test-filter** : moyenne box 2Δ̃ sur (ux, uy, T) → champ filtré (ũ̃, T̃ au sens thèse =
   filtre grille ; champ au filtre test = champ brut lissé). Implémentation : calculer
   les quantités filtrées sur les scalaires copiés avant la boucle rayon (pas de
   lecture concurrente des lignes mutables). Parois : filtrage unilatéral (clamp
   indices) — documenter le choix.
2. **Quantités aux deux niveaux** (3.1/3.2/3.3) :
   - Grille : S̃ij depuis π_neq (comme actuellement), |S̃| déjà calculé ;
     qsm^{1/2} = (1/(2√2))·√(Σ(ũ_i−ū_i)²) (symétrie 2.74) ;
     Φsm^{1/2} = (1/2)·|T̃−T̄|·√((1/2)·Σ(ũ_i−ū_i)²) (Éq. 2.74 exacte) ;
     |T̃| = √(2·T̃_i·T̃_i) avec T̃_i = (∂T̃/∂x_j)·S̃_ij (Éq. 2.75 exacte — contracté,
     PAS |∇T|)
   - **ν_t(x,y)** = C^ν_dyn·νsm_mix(grille), borne ν_t ≥ nu_sm_min·ν ;
     **κsm(x,y)** = C^κ_dyn·κsm_mix(grille) → **α_total = α + κsm** (Éq. 2.80,
     cf. 3.4) ; formes mixtes α=0.5 = moyennes géométriques (§3.3)
   - Test : filtrage box 2Δ̃ de (S̃ij→S̄ij, qsm→q̄sm, Φsm→Φ̄sm, ∂T/∂xj→∂T̄/∂xj) —
     **filtrer les quantités construites, pas les champs bruts** (le tableau surbarre
     qsm/Φsm : q̄sm = test-filter de qsm)
   - L_ij/L_j : produits croisés des champs résolus au filtre test
     (L_ij = ⟨u_i u_j⟩_test − ⟨ũ̃_i ũ̃_j⟩_test, L_j = ⟨u_j T⟩_test − ⟨ũ̃_j T̃⟩_test)
3. **C^ν_dyn** = (1/2)·⟨L^d_ijM_ij⟩/⟨M_ijM_ij⟩, **C^κ_dyn** = (1/2)·⟨L_jM_j⟩/⟨M_jM_j⟩ —
   les deux avec facteur 1/2 (capture Tab. 2.3) ; moyenne spatiale locale (box) au
   numérateur/dénominateur (régularisation 2.62 de la thèse), clampée [0, C_max]
   (C_max à caler, cf. clamp crate [0, 0.04])
4. **Collision** : `CollideCtx` — remplacer `cs: f64` par lecture de `nu_t[k]` /
   diffusivité α précalculées (déjà le pattern existant : ν_t stocké par nœud) ;
   le patch τ_t⁻ = τ_t⁺ (BGK scalaire) est préservé tel quel ; si le τ_t scalaire
   ne supporte pas de diffusivité variable, calculer κsm quand même (sorties
   summary) + TODO commenté pour α_total = α + κsm
5. **Coût** : +1 filtrage box (3 champs) + tenseurs par nœud ≈ +30–50 % temps/pas

### 4.3 Sorties
- summary.json : ajouter `cs_dyn_mean`, `cs_dyn_max`, `nu_t_mean`, `alpha_t_mean`
- Progression : logger `nu_t_max` (déjà présent) + `C_nu_dyn_mean`
- CSV nu_t déjà écrit par nœud → étendre à alpha_t si utile pour Tab. 2.10

## 5. Validation

| Test | Référence | Critère |
|---|---|---|
| Ra=1e8, 64×256, dynamic ON vs Cs=0.1 fixe | run `ra1e8_64x256_les` (Nuc 30.58) | Nuc à ±2 %, ν_t/ν même ordre |
| Ra=1e10, 96×384 LES dynamique | run `ra1e10_96x384_les` (en cours, Smagorinsky fixe) + **C2D2/C2D3 thèse** (Nuc 100.2–100.36, ν_sm/ν 0.48–0.54) | Nuc dans la fourchette thèse — **cible principale** (α=0.5, Tab. 2.3 ; comparaison LES↔LES) |
| Ra=1e10, 256×1024 | Nuc 101.19 (ν_t/ν ≈ 13 % au pic) | dégradation minimale vs Cs fixe |
| Ra=1e8, 64×256, Cs=0.2 fixe | Nuc 29.47, ν_t/ν 18.4 % | cohérence curseur (le dynamique doit se rapprocher du fixe optimal) |

- Déterminisme bitwise : même `--jobs 1` → résultats identiques run-to-run
  (test existant `tests.rs` à étendre)
- Critère SGS thèse : ν_sm/ν C2D2/C2D3 = 0.48–0.54 — le dynamique doit produire
  un ratio dans cette gamme sur la grille 96×384 (sinon : moyenne, borne, ou α à revoir)

## 6. Ordre d'implémentation

1. ~~Extraction §2.3.5~~ — **fait** (équations figées §3)
2. Test-filter box + tenseurs Germano + Cνdyn (Smagorinsky dynamique seul, sans
   volet mixte) — valider vs Cs fixe Ra=1e8 : comportement Cs²(x,y) sain (clamp,
   moyenne 2.62, pas de valeurs aberrantes près des parois)
3. Ajout flux thermique dynamique (Cκdyn) — vérifier Cκdyn vs Cνdyn (même ordre,
   cohérence Phys. : les deux ~Cs²)
4. Volet mixte α=0.5 (moyennes géométriques qsm/Φsm) — valider vs C2D2/C2D3
5. Tests unitaires (clamp, parois, déterminisme, α=0 limite Smagorinsky) + docs README

## 7. Risques

- **Coût** : test-filter + tenseurs par nœud ≈ +30–50 % temps/pas — acceptable (2D)
- **Moyenne Germano** : régularisation 2.62 (moyenne au numérateur ET dénominateur)
  obligatoire — sans elle, Cs² bruité → instabilité ; en cavité non homogène,
  la moyenne est locale (box), pas globale : choix cohérent avec la thèse
- **Bornes** : Cνdyn peut diverger (dénominateur MijMij → 0) et prendre des valeurs
  négatives (cascade inverse) — clamp [0, C_max] + option νsm ≥ −0.1ν (Ezzouhri)
- **Notation inversée** : thèse Δ̃ = grille, Δ = test — l'inverse de la convention
  Smagorinsky standard ; traduire soigneusement dans le code (Δ̃ = dx, Δ = 2dx)
- **|T̃| ≠ |∇T̃|** (Éq. 2.75) : c'est le gradient de température contracté avec le
  tenseur de contraintes (T̃_i = ∂T̃/∂x_j·S̃_ij, |T̃| = √(2T̃_iT̃_i)) — une lecture
  naïve |∇T| fausserait κsm d'un facteur de forme
- **Φsm exacte (Éq. 2.74)** : φ′i = (1/2)(ũ_i−ū_i)(T̃−T̄) → Φsm^{1/2} =
  (1/2)|T̃−T̄|·√((1/2)Σ(ũ_i−ū_i)²) — le facteur (1/2)² par composante compte
- **qsm par symétrie** avec 2.74 : qsm^{1/2} = (1/(2√2))·√(Σ(ũ_i−ū_i)²) —
  estimation minimaliste documentée (la thèse n'explicite pas qsm)
- **Coquille Tab. 2.3** : dénominateur C^κ_dyn noté « M_ij M_ij » au lieu de M_j M_j —
  implémenter M_j M_j (cohérent avec la dérivation 2.70)
- **τ_t⁻ scalaire** : garder le patch BGK-g (sinon biais Nuc ×3 réapparaît)
- **Interactions α-mult** : sur les grilles à ν forcé (96×384), le ν_t dynamique se
  superpose à l'amortissement α — watcher ν_t total (déjà 4×ν au pic transitoire
  en Cs fixe sur cette grille)
