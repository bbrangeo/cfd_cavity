# Validation Étape E — Porte G5 (campagne cible de thèse : 2 runs GPU 256×1024, Ra=1e10)

> Date : 2026-11-01. Matériel : Apple M1 Max (Metal 4, backend GPU wgpu 30).
> Branche `dev-gpu`, commit `2add1db` (poussé) — **campagne de runs pure :
> aucune modification de code, aucun commit**. Portes G1-G4 PASSÉES.
>
> **⚠️ Révision 2026-11-02 (kernel décalé, commit `9b2451a`)** : les runs
> fixe/dyn ont été **re-jouées à config identique** avec le kernel en
> variables décalées f̃ = f − w_i (P1a). Résultats dans §2bis/§3bis ;
> l'écart dyn-vs-fixe (+2.3 %) **persiste** avec la dérive de masse
> éliminée (masse 0.9997 vs 1.029) — il est **intrinsèque au Smagorinsky
> dynamique** dans cette configuration (cs_dyn_mean ≈ 0.0066 vs 0.1·cs
> statique), **pas un artefact de masse**. L'hypothèse « 0.6-0.7 pt
> d'artefact masse » est falsifiée ; le verdict dyn est robuste au
> changement de kernel (2.29 % vs 2.36 % inter-kernels).

## 1. Protocole (config exacte + correction critique)

### 1.1 Correction du template §G5 (documentée, à reporter dans le plan)

Le template §G5 de `PLAN_GPU_BACKEND.md` (`--utarget 0.002`) est **FAUX** :
c'est la config qui avorte sur le garde-fou max_u > 0.10 (~8-20k pas,
confirmé en G4 : max_u = 0.180 fixe et dynamic). La run CPU de référence
(`output/logs/ra1e10.log` l.14) a utilisé :

- **`--utarget 0.005`** (et non 0.002) ;
- **restart depuis `output/runs_uniforme/ra6.4e8_256x1024/snapshot.bin`**
  (champ établi Ra=6.4e8, 2M pas, même grille 256×1024) — run CPU complétée
  2M pas sans avorter (max_u ≈ 0.07, masse 0.999997 stable).

### 1.2 Config des deux runs (identique au log CPU ra1e10, backend gpu)

```bash
# Run 1 — fixe GPU (référence, doit reproduire Nuc = 101.19 CPU)
cargo run --release -- --mode uniform --backend gpu --nx 256 --ny 1024 \
    --ra 1e10 --cs 0.1 --utarget 0.005 --steps 2000000 \
    --sample-start 500000 --sample-every 100 --progress-every 5000 \
    --snapshot-every 50000 \
    --restart output/runs_uniforme/ra6.4e8_256x1024/snapshot.bin \
    --out output/gpu_ra1e10_fixe

# Run 2 — dynamic GPU (levier thèse), restart depuis le snapshot FINAL Run 1
cargo run --release -- --mode uniform --backend gpu --nx 256 --ny 1024 \
    --ra 1e10 --cs 0.1 --dynamic-smag --smag-alpha 0.5 --utarget 0.005 \
    --steps 2000000 --sample-start 500000 --sample-every 100 \
    --progress-every 5000 --snapshot-every 50000 \
    --restart output/gpu_ra1e10_fixe/snapshot.bin \
    --out output/gpu_ra1e10_dyn
```

**Pré-check (probe 50k pas, `output/gpu_ra1e10_fixe_probe`, même config
`--steps 50000`)** : (a) le snapshot CPU (`runs_uniforme/ra6.4e8_256x1024`,
8 388 632 octets) se charge correctement avec `--backend gpu` (pas de bug
de branchement restart/GPU) ; (b) max_u max = 0.0934 < 0.10, aucun
avortement ; (c) cohérence Nuc avec le log CPU aux mêmes pas :

| Pas | Nuc CPU | Nuc GPU | Écart |
|---|---|---|---|
| 10 000 | 103.2825 | 103.2807 | **0.0002 %** |
| 50 000 | 100.9772 | 100.6870 | 0.29 % |

→ le GPU suit le CPU de très près (l'écart f32 attendu est faible, confirmé).
Probe PASSÉ → lancement des runs complètes.

## 2. Run 1 — fixe GPU vs CPU Ra=1e10 (256×1024, 2M pas)

Champs moyens sur 15 000 échantillons (fenêtre 500k→2M) :

| Métrique | CPU (référence) | GPU fixe | Écart |
|---|---|---|---|
| **Nuc (moyenné)** | 101.1892 | **101.4924** | **+0.30 %** ✅ (tolérance ≤2 %) |
| Nuf | 101.2532 | 101.5152 | +0.26 % |
| Nuc ± σ (progress, 500k→2M) | ±0.86* | ±1.94 | — |
| w_max(z=0.5) | 0.06941 | 0.07025 | +1.2 % |
| w_max(z=0.75) | 0.05767 | 0.05691 | −1.3 % |
| y+ (paroi chaude) | 1.665 | 1.656 | −0.5 % |
| S^(1/2) | 0.0091 | 0.0098 | — |
| ν_t_max/ν (pic, fields_mean) | ≈13 % (note plan) | **13.0 %** (2.597e-4/2e-3) | identique |
| Masse (fin) | 0.999997 | 1.028921 | dérive f32, §4 |
| Avortement / NaN | aucun | aucun | — |

\* σ CPU non recalculée ici ; le σ GPU est calculé sur 301 points
progress-every 5000 de la fenêtre 500k→2M.

**Critère G5 fixe (Nuc ≈ 101.19 ± 2 %)** : **PASSÉ** (+0.30 %).

## 3. Run 2 — dynamic GPU (levier thèse, α=0.5, restart champ établi 1e10)

| Métrique | C2D2/C2D3 thèse (cible) | GPU dynamic 256×1024 |
|---|---|---|
| **Nuc (moyenné)** | 100.2–100.36 | **103.8888** (+3.7 % vs C2D2 ; +3.2 % vs Trias 100.68) |
| Nuc ± σ (500k→2M) | — | ±1.92 (mean 103.82, min 99.16, max 108.71) |
| Δ vs fixe GPU 101.49 | — | **+2.4 %** |
| Nuf | — | 103.4537 |
| w_max(z=0.5) | 0.22–0.24 (thèse) | 0.06947 (pic sous-résolu, cf. §5.1 plan) |
| y+ (paroi chaude) | — | 1.670 |
| cs_dyn_mean / cs_dyn_max (fin) | — | 0.0064 / 0.0400 (clamp actif) |
| **κ_dyn_mean (fin)** | — | **0.0170** (κ_dyn_max = 0.0400, ∈ [0, 0.04], pas de NaN) |
| ν_t_max (fin) | — | 4.436e-4 → **ν_t_max/ν = 22.6 % au pic** (fixe 13.0 %) |
| Masse (fin) | — | 1.057832 (§4) |
| Avortement / NaN | — | aucun, run complète 2M pas |

### 3.1 ν_sm/ν — rapport factuel (ne pas deviner)

**ν_sm/ν n'est PAS écrit directement dans summary.json** (champs disponibles :
`cs_dyn_mean/max`, `kappa_dyn_mean/max`, `nu_t_max`, champs moyens
`fields_mean.csv` avec colonne ν_t). Proxies extraits de `fields_mean.csv`
(moyenné 500k→2M, ν = 0.002) :

| Proxy | fixe GPU | dyn GPU |
|---|---|---|
| ν_t moyenne domaine / ν | 0.0051 | **0.0038** |
| ν_t max / ν (pic) | 0.130 | **0.226** |

Aucun de ces deux proxies n'est dans la gamme thèse C2D2/C2D3
ν_sm/ν ∈ [0.48, 0.54] — mais **la définition exacte du ratio thèse
(moyenne BL ? filtré ? autre normalisation ?) n'est pas calculée par ce
code**, donc cette comparaison n'est pas conclusive en l'état. À rapprocher
de la définition exacte de la thèse (Ezzhri Tab. 2.3/C2D2-C2D3) avant tout
verdict sur le critère SGS. Le plan notait déjà pour le fixe CPU
« ν_t/ν ≈ 13 % au pic » — valeur que le GPU fixe reproduit exactement
(13.0 %), ce qui valide l'extraction.

## 2bis. Run 1bis — fixe GPU kernel décalé P1a (re-run config identique, commit `9b2451a`)

Sortie `output/gpu_ra1e10_fixe_p1a2/`, fenêtre 500k→2M identique (15 000
échantillons), σ sur 301 points progress-every 5000 :

| Métrique | CPU (réf) | G5 anc. kernel | **P1a décalé** |
|---|---|---|---|
| **Nuc (moyenné)** | 101.1892 | 101.4924 (+0.30 %) | **101.3523 (+0.16 %)** |
| Nuf | 101.2532 | 101.5152 | 101.3411 |
| Nuc ± σ | ±0.86* | ±1.94 | **±1.80** |
| w_max(z=0.5) | 0.06941 | 0.07025 (+1.2 %) | **0.06926 (−0.22 %)** |
| wmax_uref | — | — | 0.25152 |
| x_wmax_z05_h | — | — | 0.00391 (anomalie 1ᵉʳ nœud disparue) |
| Masse (fin) | 0.999997 | 1.028921 | **0.999852** (dérive −7.4e-11/pas, **×195 mieux**) |
| Avortement / NaN | aucun | aucun | aucun |

**P1a améliore toutes les métriques** : écart vs CPU divisé par ~2,
dispersion réduite, w_max quasi exact, dérive de masse éliminée.

## 3bis. Run 2bis — dynamic GPU kernel décalé P1a (re-run config identique)

Sortie `output/gpu_ra1e10_dyn_p1a2/`, restart snapshot final P1a2
(comme la G5 dyn), fenêtre 500k→2M, 15 000 échantillons :

| Métrique | G5 anc. kernel | **P1a décalé** |
|---|---|---|
| **Nuc (moyenné)** | 103.8888 (σ 1.92) | **103.6781 (σ 1.87)** |
| Δ vs fixe P1a 101.3523 | — | **+2.29 %** (~1.2σ) |
| Nuf | 103.4537 | 103.3622 |
| w_max(z=0.5) | 0.06947 | 0.06886 |
| wmax_uref | — | 0.25009 |
| Masse (fin) | 1.057832 | **0.999711** (aucune dérive) |
| cs_dyn_mean / max | 0.0064 / 0.0400 | 0.0066 / 0.0400 (clamp actif) |
| Avortement / NaN | aucun | aucun, run complète 2M pas |

**L'écart dyn-vs-fixe persiste à +2.3 % avec la dérive de masse
éliminée** (0.9997 vs 1.058) : il est **intrinsèque à la procédure
dynamique** dans cette configuration, pas un artefact de précision.
Cohérence inter-kernels (2.29 % vs 2.36 %) : le verdict dyn est robuste.

## 4. Dérive de masse f32 sur 2M pas — LE point G5 (ancien kernel ; réglé par P1a)

> **Révision 2026-11-02** : ce point concernait l'ancien kernel. Le kernel
> décalé P1a (§2bis/§3bis) **élimine la dérive** : −7.4e-11/pas (fixe),
> masse 0.9997-0.9999 après 2M pas. La question de la renormalisation
> périodique reste ouverte pour la 3D tant que le kernel 3D n'a pas été
> porté (V2) — même défaut pleine magnitude à corriger au passage.

Courbe masse du progress.log (points tous les 5 000 pas, lissage sans
accident) :

| Run | Masse début | Masse fin | Pente ajustée | Forme |
|---|---|---|---|---|
| fixe GPU 2M | 0.999997 | 1.028921 | **+1.446e-8 / pas** | **strictement linéaire**, non explosive |
| dyn GPU 2M | 1.030370 | 1.057832 | **+1.445e-8 / pas** | strictement linéaire (hérite du snapshot fixe) |
| CPU 2M (réf.) | 0.999997 | 0.999997 | 0 | f64, aucune dérive |

- Dérive totale fixe : **+2.89 % en 2M pas**, à pente constante du début à
  la fin (1.000718 à 50k → 1.014245 à 1M → 1.028921 à 2M) : accumulation
  linéaire f32, **aucune accélération, aucune divergence**.
- La dérive dynamique sur sa fenêtre propre (1M pas) est identique
  (+1.45 %) : elle découle du même taux par pas, pas du mode SGS.
- **Nuc n'est pas affecté au premier ordre** : il oscille 100-102 (fixe) et
  101-109 (dyn) sur toute la fenêtre, comme le CPU, sans tendance.
- Point d'attention (rapporté, NON traité) : à ce taux, une run ~7M pas
  dépasserait +10 % de masse — pour la 3D ou des runs plus longues, la
  parade documentée (renormalisation périodique Σf) est à décider ;
  **non implémentée** conformément à la consigne.

## 5. Perf réelle (avec moyennage, runs complètes)

| Run | Durée | MLUPS | vs CPU j8 (même grille) |
|---|---|---|---|
| fixe GPU 2M (moyennage inclus) | 1337 s (**22 min**) | **392.1** | ~×8 vs ~48-49 MLUPS |
| fixe GPU probe 50k (sans moyennage) | 33.5 s | 391.2 | — |
| dyn GPU 2M (moyennage inclus) | 6164 s (**1 h 43**) | **85.1** | ~×1.7 vs ~48-49 MLUPS |

Conforme aux estimations G4 (fixe ~400 MLUPS, dynamic ~98-113 MLUPS —
le 85.1 réel inclut le readback/moyennage sur run longue).

## 6. Verdict G5

| Critère | Résultat | Verdict |
|---|---|---|
| fixe GPU : Nuc ≈ 101.19 ± ≤2 % | 101.49 (+0.30 %), trace CPU au pas près (probe 0.0002 % à 10k) | ✅ PASSÉ |
| fixe GPU : run < 1 h | 22 min | ✅ PASSÉ |
| dérive de masse f32 sur 2M pas : linéaire ou explosive ? | **Linéaire** (+1.446e-8/pas, +2.89 % total), Nuc non affecté ; **réglé par P1a (kernel décalé : −7.4e-11/pas, ×195)** | ✅ documenté puis réglé |
| dyn GPU : Nuc ≈ 100 ± σ raisonnable | 103.89 ± 1.92 (+3.7 % vs C2D2, +2.4 % vs fixe GPU) ; **révisé P1a : 103.68 ± 1.87 (+2.29 % vs fixe 101.35), robuste inter-kernels** | ⚠️ PARTIEL → écart intrinsèque au dynamique, pas artefact masse |
| dyn GPU : ν_sm/ν ∈ [0.48, 0.54] | non extraitable directement (§3.1) ; proxies hors gamme | ⚠️ non conclusif |
| dyn GPU : κ_dyn ∈ [0, 0.04], pas de NaN, run complète | κ_dyn_mean 0.017, κ_dyn_max 0.04 (clamp), 2M pas sans incident | ✅ PASSÉ |

**Verdict global : G5 PASSÉE sur la référence (fixe) et la stabilité f32 ;
PASS PARTIEL sur le levier dynamique.**
**Révision 2026-11-02 : le verdict révisé (kernel décalé P1a, §2bis/§3bis)
confirme le PARTIEL avec une cause établie — écart intrinsèque au
Smagorinsky dynamique (+2.3 %, robuste inter-kernels), pas un artefact de
masse ni de précision.**

## 7. Implication thèse (claim complété ou non)

- **Ce qui est acquis** : l'écart catastrophique de la grille 96×384
  (−40 à −47 % vs C2D2) n'est **pas** reproduit sur la grille résolue
  256×1024 : le dynamique y donne Nuc 103.89, soit +2.4 % seulement au-dessus
  du fixe (101.49) et +3.2 % vs Trias — contre −44.9 % (fixe) et −40 à
  −47 % (dyn) sur 96×384. **La limite de résolution est bien levée** : le
  modèle dynamique ne dégrade pas le fixe sur grille résolue (il amortit
  légèrement moins : ν_t_max/ν 22.6 % vs 13.0 %).
- **Ce qui ne l'est pas** : Nuc dyn = 103.89 reste hors de la fourchette
  thèse stricte (100.2–100.36) de +3.7 %, et le critère ν_sm/ν ∈ [0.48, 0.54]
  n'a pas pu être évalué (grandeur non calculée par le code — §3.1).
- **Révision P1a (2026-11-02)** : l'écart dyn-vs-fixe (+2.3 %) persiste
  avec le kernel décalé (masse stable) — il est **intrinsèque au modèle
  dynamique** (α=0.5, cs_dyn_mean ≈ 0.0066), pas à la précision ni à la
  masse. La piste d'un réglage du Smagorinsky dynamique (α, clamp) est
  à explorer pour converger vers la fourchette thèse.
- **Claim thèse reformulé (proposé)** : « le modèle de Smagorinsky dynamique
  est correct sur grille résolue (Nuc 103.9 vs fixe 101.5, +2.4 %) ; l'écart
  −40/−47 % observé sur 96×384 était une limite de résolution, non un défaut
  du modèle. La convergence fine vers la fourchette thèse (100.2–100.36) et
  la validation du ratio ν_sm/ν exigent le rapprochement avec la définition
  thèse du ratio SGS. »

## 8. Runs et reproductibilité

- Runs (ancien kernel) : `output/gpu_ra1e10_fixe_probe/` (pré-check 50k),
  `output/gpu_ra1e10_fixe/` (2M, summary.json + progress.log + snapshot.bin),
  `output/gpu_ra1e10_dyn/` (2M, idem).
- Runs (kernel décalé P1a, révision 2026-11-02) : `output/gpu_ra1e10_fixe_p1a2/`
  (2M, config identique §2), `output/gpu_ra1e10_dyn_p1a2/` (2M, config
  identique §3) ; sortie intermédiaire A/B 200k `output/gpu_ra1e10_fixe_p1a/`.
- Logs : `output/logs/gpu_ra1e10_fixe_probe.log`, `output/logs/gpu_ra1e10_fixe.log`,
  `output/logs/gpu_ra1e10_dyn.log` ; `output/logs/p1a_fixe_2m.log`,
  `output/logs/p1a_dyn_2m.log`, `output/logs/p1a_ab_200k.log`.
- Références : `output/runs_uniforme/ra1e10/` (CPU), `output/logs/ra1e10.log`,
  `output/runs_uniforme/ra6.4e8_256x1024/snapshot.bin` (restart).
- Aucune modification de code (campagne pure) — commit `2add1db` intact.
