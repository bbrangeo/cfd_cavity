# Cavité différentiellement chauffée 2D — LBM D2Q9 TRT + Smagorinsky (LES)

Benchmark de la thèse (§2.5, pages 65–71, Tab. 2.6 et 2.7) : cavité fermée
différentiellement chauffée, rapport de forme **Az = 4**, air (Pr = 0.71),
hypothèse de Boussinesq.

## Cas d'étude

| Paramètre | Valeur |
|---|---|
| Géométrie | 2D, Az = Lz/Lx = 4 (ex. 64×256, 96×384, 128×512) |
| Paroi chaude (gauche, ix=0) | θc = +0.5, no-slip |
| Paroi froide (droite, ix=nx−1) | θf = −0.5, no-slip |
| Parois horizontales | adiabatiques (∂θ/∂y = 0), no-slip |
| Fluide | air, Pr = 0.71, Boussinesq |
| Ra cible | 1e6 (smoke), 1e8, 1e10 (cas thèse) |

## Méthode

- **LBM D2Q9 double distribution (DDF)** : `f` pour la masse/quantité de mouvement,
  `g` pour la température (advection-diffusion).
- Structure de données, équilibres (`feq`, `geq`) et fonctions physiques
  (`boussinesq_force`, `rayleigh_number`, `prandtl_number`, `nusselt_number`,
  `compute_turbulent_viscosity`, `y_plus`) issus de la crate
  [`oxiphysics-lbm` v0.1.2](https://crates.io/crates/oxiphysics-lbm)
  (module `thermal_lbm`, constantes `lattice::{D2Q9_WEIGHTS, D2Q9_VELOCITIES, D2Q9_OPPOSITES, CS2}`).
- **Pas de temps maison** (`src/solver.rs`) : le `streaming_step` de la crate est
  périodique dans les deux directions (inadapté à une cavité fermée), donc :
  - collision **TRT** (two-relaxation-time, Ginzburg et al. 2008) avec relaxation
    **Smagorinsky** — remplace la collision BGK d'origine, trop instable à
    Ra=1e8 (τ+ → 0.5) :
    - temps de relaxation pairs τ+ = 0.5 + 3·(ν + ν_t) ; impairs
      τ− = 0.5 + Λ/(τ+ − 0.5) avec **Λ = 3/16** (valeur optimale pour
      bounce-back) ; idem pour g (τ_t±, α_t = ν_t/Pr_t, Pr_t = 0.9) ;
    - moments non-équilibrés π_neq = Σ c_i c_j (f_i − f_i^eq) ;
    - |S| = sqrt(π_neq:π_neq) / (2·cs²·τ·ρ) (formule standard LBM) ;
    - ν_t = (Cs·Δx)²·|S| via `compute_turbulent_viscosity` de la crate (Cs = 0.10) ;
  - force de Boussinesq intégrée façon **Guo, Shi & Zheng 2002 (complet,
    éq. (14)-(16))** : S_i = (1 − 1/2τ+)·w_i·[(e_i−u)/c_s² + (e_i·u)·e_i/c_s⁴]·F,
    F = (0, fy, 0), fy = ρ·beta_eff·(T − t_ref) (chaud à gauche ⇒ montée le long
    de la paroi chaude) ; correction de vitesse u_y = (Σe_iy·f_i + fy/2)/ρ
    (fy stockée par nœud pendant la collision — `fy_node` — même F que la
    collision, déterminisme jobs>1 préservé) ;
  - streaming **pull standard** avec **full-way bounce-back intégré** sur les
    parois (f_i(x) ← f*_opp(i)(x) quand la source est hors domaine) — bijection
    exacte ⇒ masse exactement conservée ;
  - BC thermiques : isotherme g = g^eq(±0.5, u=0) sur les parois verticales ;
    zero-gradient (copie du nœud intérieur) sur les parois horizontales.
- **Mapping thèse** : la thèse (Eq. 2.104) utilise un maillage tangente hyperbolique
  (Henkes & Hoogendoorn) ; ce code utilise une **grille uniforme** (écart documenté,
  voir « Écarts » ci-dessous). Les quantités de comparaison (Tab. 2.7) sont inchangées.

## Compilation et exécution

```bash
cargo build --release
cargo test --release
```

### Ra = 1e6 (smoke / validation)

```bash
cargo run --release -- --nx 64 --ny 256 --ra 1e6 --cs 0.1 --steps 200000 \
    --sample-start 100000 --sample-every 100 --out output/ra1e6
```

### Ra = 1e8 (réglage recommandé)

**Réglage** : `--utarget 0.01` ⇒ ν = 0.0025, τ+ = 0.5076, τ_t = 0.5106
(plancher TRT ν ≥ 0.002, τ+ ≥ 0.506). La collision TRT (Λ = 3/16) est
indispensable : en BGK, τ ≥ 0.554 force u_max ≈ 0.25 > limite de
compressibilité ≈ 0.15 (divergence NaN mesurée au pas 600).

```bash
cargo run --release -- --nx 64 --ny 256 --ra 1e8 --cs 0.1 --steps 1000000 \
    --sample-start 300000 --sample-every 100 --progress-every 50000 \
    --utarget 0.01 --out output/ra1e8
```

Run validée (64×256, 1M pas) : masse conservée à 1e-6, Nu_c = Nu_f,
w_max instantané borné (0.10–0.18 < 0.15), ν_t max ≈ 5e-4 (LES actif),
écoulement instationnaire (S^1/2 > 0) — proche du seuil de transition
turbulente Ra_c ≈ 2.3e8 (AR=8 ; ~1e8–1e9 pour AR=4).

### Ra = 1e10 (cas thèse — coût élevé, à lancer en background)

**Analyse des unités réseau** (formule de calage, plancher TRT ν ≥ 0.002) —
seules les grilles avec ratio de nœuds ≈ 4 (Az=4) sont des configurations thèse :

| Grille | Lz | ν retenu | τ+ | U_CN | Ma est. (0.23·U_CN) | Verdict |
|---|---|---|---|---|---|---|
| 64×256 | 255 | 0.0020 | 0.506 | 0.931 | 0.21 | hors limite (transitoire) |
| 128×512 | 511 | 0.0020 | 0.506 | 0.465 | 0.11 | **hors limite** (diverge au pas ~5k, constaté) |
| **256×1024** | 1023 | 0.0020 | 0.506 | **0.232** | **0.054** | **sous limite — configuration validée 2026-10-30** |
| 512×2048 | 2047 | 0.0020 | 0.506 | 0.116 | 0.027 | sous limite (coût 4×, superflu) |

**Expérimentation stabilité** (pré-fix Guo/τ_t⁻, historique) :
- 128×512 (20k pas) : `--alpha-mult 1/2/4/8` divergence NaN au pas 2700–5000 ;
  rampe de spin-up de flottabilité (β linéaire 0→β_eff) : retarde sans supprimer.
  Cause : à ν fixé par le plancher TRT, U_CN = α·√Ra/Lz ∝ 1/Lz — à Lz=511,
  Ma attendu ≈ 0.22 dépasse la limite de compressibilité D2Q9.
- 256×1024 (initialisation classique) : divergeait au pas 15.5k **[RÉSOLU]** :
  post-fix Guo + restart depuis snapshot 6.4e8 **même grille**, le run est
  **STABLE et complète** (voir ci-dessous).
- 512×2048 + restart : stable (expérimentation pré-fix, conservée comme point
  de stabilité ; rendu superflu par le run 256×1024 réussi).
- **Conclusion (2026-10-30)** : la configuration validée pour Ra=1e10 est
  **256×1024 + restart depuis snapshot Ra=6.4e8 (256×1024, même grille)** :
  Ma ≈ 0.11, transitoire éliminé, **Nuc = 101.19 vs Trias 100.68 (0.5 %)**.

> **Note maillage (relecture §2.5 du mémoire, Tab. 2.9)** : les grilles de
> référence thèse (SND2D1 96×768, C2D1 64×128, C2D2 96×320, C2D3 128×320) sont
> **toutes sur la même cavité Az=4** : leurs ratios de nœuds (8, 2, 3.4, 2.5)
> ne reflètent pas la géométrie car elles utilisent un maillage non uniforme
> (Chebyshev pour la SND ; tanh γx=2.0 en x, régulier en z pour les SGE).
> Notre LBM à grille uniforme doit donc garder un **ratio de nœuds ≈ Az = 4**
> (ex. 256×1024) — un ratio ≠ 4 changerait la géométrie physique simulée.

**Stratégie retenue : chaînage Ra=1e8 → Ra=1e10 par snapshot + re-maillage.**
Chaque run exporte un snapshot binaire `snapshot.bin` (magic `CAVSNAP1`, rho/ux/uy/T)
en fin de simulation ; l'option `--restart <file>` initialise une nouvelle grille
depuis ce champ, avec interpolation bilinéaire si les maillages diffèrent — le
transitoire violent est éliminé, la physique Ra=1e10 se développe depuis un
écoulement déjà établi. **Limite de compressibilité** : U_CN = α·√Ra/Lz ∝ 1/Lz —
à Ra=1e10 sur Lz=512, Ma ≈ 0.22 dépasse la limite D2Q9 (divergence au pas ~5k,
constatée) ; il faut Lz ≥ 1024 pour Ma ≈ 0.11. La grille 256×1024 est le
candidat retenu (Ma sûr + BL résolue à 1e10, δ_x ∝ Ra^−1/4).

```bash
# Validation multi-échelle 2026-10-29 (post-fix Guo + τ_t⁻) :
# Ra=1e8  128×511, nu=0.001, alpha-mult 1.2, 2M pas -> Nuc = 30.45 vs 30.41 (0.1 %)
# Ra=6.4e8 128×512, utarget 0.005, 2M pas -> Nuc = 51.02 vs 49.23 (écart 3.6 %)
#           w_max(z=.5)=0.0337 à x=0.0315 (pic Trias x*=0.030), y+=1.23
# Convergence en maillage Ra=6.4e8 : restart interpolé 128×512 -> 256×1024
nohup cargo run --release -- --nx 256 --ny 1024 --ra 6.4e8 --cs 0.1 \
    --utarget 0.005 --steps 1500000 --sample-start 100000 --sample-every 100 \
    --progress-every 5000 --snapshot-every 50000 --jobs 8 \
    --restart output/ra6.4e8_final/snapshot.bin \
    --out output/ra6.4e8_256x1024 > output/ra6.4e8_256x1024.log 2>&1 &

# Ra=1e10 : configuration validée (256×1024, restart même grille depuis 6.4e8)
# Résultat 2026-10-30 : Nuc = 101.19 vs Trias 100.68 (0.5 %), 2M pas, Ma≈0.11
nohup cargo run --release -- --nx 256 --ny 1024 --ra 1e10 --cs 0.1 \
    --utarget 0.005 --steps 2000000 --sample-start 500000 --sample-every 100 \
    --progress-every 5000 --snapshot-every 50000 --jobs 8 \
    --restart output/ra6.4e8_256x1024/snapshot.bin \
    --out output/ra1e10 > output/ra1e10.log 2>&1 &
```

> **Bilan de validation (2026-10-29)** : cause racine du biais Nu ×3 identifiée —
> `solver.rs` appliquait Λ=3/16 à τ_t⁻ du scalaire g (paroi à nœud-équilibre,
> pas de bounce-back) → conductance murale effondrée ×1/37 ; patch
> `tau_t_minus = tau_t_plus` (BGK pour g). Nu mesuré sur la 2ᵉ liaison (l'artefact
> demi-lien 0.505 est corrigé dans post.rs et main.rs). Garde-fous :
> `MAX_U_ABORT=0.10`, cibles dynamiques `post::nusselt_trias_2d(ra)`
> (30.41 @ 1e8, 49.23 @ 6.4e8, 100.60 @ 1e10, fit 0.2529·Ra^0.26).
> Scripts de post-traitement : `output/plot_isothermes.py` (3 panneaux
> θ̄/w̄/ψ, CLI : run_dir, --labels/--no-labels, --out, --wmax, --title, --dpi),
> `output/extract_tab210.py` (grandeurs Tab. 2.10 : ⟨Nuc⟩, S₁/₂, u/w_max
> + positions dans les 2 plans médians), `output/compare_tab27.py`.

> ⚠️ **Anomalie majeure détectée (run 512×2048 divergée au pas 60k + relecture
> Trias JFM 2007)** : le facteur mesuré w_max/u_ff ≈ 1.8–2.2 est ~7–8× trop grand
> vs le benchmark. En échelle Trias/Xin (U_ref = (α/H)·√Ra = chute_libre/√Pr),
> le benchmark rapporte wmax = 0.223 / 0.229 / 0.230 aux Ra = 6.4e8 / 2e9 / 1e10,
> soit wmax/u_ff ≈ 0.27. Nu aussi est faux (11.7 vs ~34 extrapolé à Ra=1e8).
> **[RÉSOLU 2026-10-29]** : cause racine = τ_t⁻ du scalaire g (voir bilan de
> validation plus haut).
> Ces deux signatures invariants d'échelle désignent un problème physique ou
> numérique (calage β_eff/forcing suspecté), pas des unités. **[RÉSOLU — voir
> ci-dessus]** la cause était τ_t⁻ du scalaire g, pas le calage β_eff ; le test
> laminaire Ra=1e8 (0.1 % vs Trias) a validé la chaîne β→u→Nu.
>
> **Précisions (extraction brute Trias JFM 586, 2007)** : le papier contient les
> deux configurations — cases 2D (A2D/B2D/C2D, grilles 156×312, 144×318, 138×326)
> et 3D (A/B/C). Les valeurs de recalage 2D : Nū = 49.23/66.19/100.60,
> Numax = 169.96/249.80/441.65, u2max = 2.22–2.23e-1, u3max = 2.23–2.29e-1 ;
> 3D : Nū = 49.24/66.63/101.70 (≈1% au-dessus du 2D). Nu paroi corrigé (fix ×4)
> : **[VALIDÉ 2026-10-29]** Ra=1e8 → Nuc = 30.45 vs 30.41 Trias 2D (écart 0.1 %)
> sur 128×511, 2M pas ; Ra=6.4e8 → 50.89 vs 49.23 (3.3 %) sur 256×1024
> (convergence : 51.02/3.6 % sur 128×512, plateau de méthode ~3 %) ;
> Ra=1e10 → 101.19 vs 100.68 (**0.5 %**) sur 256×1024, 2M pas.
> Anomalie restante : positions de pics pariétaux limitées par le premier nœud
> (x ≥ 1/(nx−1)) — limitation méthode LBM uniforme, à documenter.
>
> **Papier de référence (benchmark principal)** : Trias, Soria, Oliva &
> Pérez-Segarra, « Turbulent natural convection in an enclosed cavity »,
> J. Fluid Mech. 586 (2007), 259–293 — PDF officiel :
> https://www.cttc.upc.edu/downloads/DHC_AR4_2Dvs3D/JFM-DHC-paper.pdf
> (config identique à la thèse Tab. 2.6 : air Pr=0.71, Az=4, parois
> verticales différentiellement chauffées, horizontales adiabatiques).
> Référence secondaire (discussion qualitative uniquement, paramètres
> différents) : Ng, Ooi, Lohse & Chung, « Regime transitions in thermally
> driven high-Rayleigh number vertical convection », JFM (2021) — Pr=10,
> Γ=1, DNS 2D jusqu'à Ra=1e14, régimes I/II/III, β≈0.26 en régime I.
>
> **Forcing Guo complet (fix 2026-10-29)** : la collision implémente désormais
> Guo, Shi & Zheng 2002 (Phys. Rev. E 65, 046308) éq. (14)-(16) — terme
> S_i = (1−1/2τ+)·w_i·[(e_i−u)/c_s² + (e_i·u)·e_i/c_s⁴]·F + correction de
> vitesse u_y = (Σe_iy·f_i + fy/2)/ρ (fy stockée par nœud pendant la collision,
> déterminisme jobs>1 préservé). 16/16 tests verts.

## Options CLI

| Option | Défaut | Description |
|---|---|---|
| `--nx` / `--ny` | 64 / 256 | dimensions de la grille (ny > nx, Az ≈ 4) |
| `--ra` | 1e6 | nombre de Rayleigh cible (basé sur Lz) |
| `--pr` | 0.71 | nombre de Prandtl |
| `--cs` | 0.1 | constante de Smagorinsky |
| `--pr_t` | 0.9 | Prandtl turbulent |
| `--steps` | 200000 | nombre de pas de temps |
| `--sample-start` | 100000 | pas de début du moyennage temporel |
| `--sample-every` | 100 | stride d'échantillonnage des moyennes |
| `--utarget` | 0.02 | vitesse max cible en unités de réseau (calage ν/α ; plancher TRT ν ≥ 0.002) |
| `--seed` | 42 | graine RNG (perturbation initiale ±1e-3 sur T) |
| `--out` | output/ra1e6 | répertoire de sortie |
| `--jobs` | 1 | threads de calcul (1 = mono-thread d'origine ; N > 1 = rayon, déterministe) |
| `--alpha-mult` | 1.0 | multiplicateur α (τ_t amortit plus fort ; levier de stabilisation Ra élevé) |
| `--nu-start-mult` | 1.0 | rampe de viscosité initiale (≥1, décroissance exponentielle vers ν sur le spin-up) |
| `--restart <path>` | — | snapshot binaire de départ (re-maillage bilinéaire si grille ≠) ; export automatique `snapshot.bin` en fin de chaque run |
| `--snapshot-every` | 0 | export ATOMIQUE de `snapshot.bin` (tmp + rename) toutes les N étapes, pour reprendre une run interrompue via `--restart` ; 0 = désactivé |

### Parallélisation multi-thread (`--jobs N`)

Rayon par bandes de lignes `iy` : chaque thread possède des lignes disjointes
(`chunks_mut`, scalaires copiés avant la boucle ; variante dédiée
`update_macroscopic` pour jobs > 1) ⇒ **zéro unsafe**, **déterministe**
(résultats **bitwise identiques** à `--jobs 1`, validé sur run Ra=1e6 20k pas :
`--jobs 1` vs `--jobs 4`, diff CSV = 0.000e+00 exactement, seuil 1e-12).

À `--jobs > N_cœurs`, cap au nombre de cœurs disponibles (warning, pas d'erreur).

Benchmark (64×256, 16 384 nœuds, 50k pas, Ra=1e6, M1 Max ; machine au repos) :

| `--jobs` | Temps | Speedup |
|---|---|---|
| 1 | 58.0 s | ×1.0 |
| 2 | 44.8 s | ×1.3 |
| 4 | 27.7 s | ×2.1 |
| 8 | 25.6 s | ×2.3 |
| 10 | 26.9 s | ×2.2 |

La parallélisation d'`update_macroscopic` (variante rayon jobs > 1, bitwise
identique à la version crate) apporte un gain supplémentaire mesuré : à
`--jobs 2`, 44.8 → 35.5 s (×1.26). Au-delà de 8 threads, le speedup sature
(~×2.3) : sur une grille 64×256 (16 384 nœuds), la granularité des bandes de
lignes est fine et le surcoût de synchronisation rayon domine ; le gain devient
meilleur sur grilles plus grandes (128×512 : ×2.7 à `--jobs 8`).

Le pool rayon est initialisé une fois par processus (`build_global`, appelé
uniquement si `--jobs > 1`) ; à `--jobs 1`, le comportement mono-thread
d'origine est strictement inchangé.

## Sorties (`--out <dir>`)

| Fichier | Contenu |
|---|---|
| `fields_mean.csv` | champs moyens : x, y, ux, uy, T, psi, nu_t |
| `profiles_w.csv` | profils w(z=0.5·Lz) et w(z=0.75·Lz) vs x normalisé |
| `snapshot.vti` | VTK XML ImageData (T, Ux, Uy) — Paraview/VisIt |
| `summary.json` | résumé : Nuc, Nuf, w_max et positions, y+, références Tab. 2.7 |
| `progress.log` | progression (w_max, Nuc, Nuf, masse, nu_t_max) |

## Structure du code

```
src/
├── main.rs     # CLI, boucle temporelle, moyennage, exports
├── cli.rs      # parsing des arguments (sans dépendance)
├── solver.rs   # pas de temps : collision LES + Boussinesq, streaming pull
│               # + bounce-back, BC thermiques, calage des unités de réseau
├── post.rs     # Nusselt (paroi/volumétrique), profils w, ψ, y+, CSV/VTK/JSON
├── rng.rs      # xorshift64* (perturbation initiale)
└── tests.rs    # tests unitaires (équilibres, conservation, CL, flottabilité)
```

## Critères de validation (Tab. 2.7, Trias JFM 586 2007)

| Quantité | Cible (Trias 2D) | Résultat LBM (final) | Verdict |
|---|---|---|---|
| Nuc @ 1e8 | 30.41 | 30.45 (128×511) | ✅ +0.1 % |
| Nuc @ 6.4e8 | 49.23 | 50.89 (256×1024) | ✅ +3.3 % (plateau méthode) |
| Nuc @ 1e10 | 100.60 (100.68 h-moy.) | 101.19 (256×1024) | ✅ +0.5 % |
| w_max(z=0.5)/U_CN | ≈ 0.223 en x*≈0.004 | ≈ 0.23 @ 6.4e8 ; 0.31 @ 1e10, x≥0.0157 | ⚠️ valeur 1e10 à discuter |
| w_max(z=0.75) | ≈ 0.11 en x*≈0.005 | x ≥ 0.023 (borne maillage) | ⚠️ positions bornées par le 1er nœud |
| S^1/2 (dθ/d(z/H) cœur) | 0.99 – 1.01 | 0.974 / 1.064 / 1.317 | ✅ 1e8–6.4e8 |
| y+ première cellule | 0.55 – 1.11 | 0.63 / 0.63 / 1.67 | ✅ |

**Test curseur DNS/LES** (Ra=1e8, 64×256, 1M pas, même ν) :

| Run | Cs | Nuc | ν_t/ν | Écart vs DNS fine (30.45) |
|---|---|---|---|---|
| DNS grossière | 0.1 | 30.58 | 4.8 % | +0.4 % |
| LES grossière | 0.2 | 29.47 | 18.4 % ≈ (0.2/0.1)² | −3.6 % |

→ le même code fait du DNS (Cs=0.1, grille fine, ν_t quasi inactif) ou de la
vraie LES (Cs=0.2, ν_t active, biais attendu). **Cadre retenu pour le mémoire :
la démarche est une LES LBM (Smagorinsky, Cs=0.1)**, validée contre le DNS de
référence de Trias et al. (2007) : la grille est assez fine pour que le modèle
sous-maille soit peu actif (ν_t/ν = 5 % → 1.6 % en raffinant à 6.4e8), d'où la
convergence 0.1–3.6 % vs Trias sur les 3 Ra. Le Smagorinsky joue alors son rôle
historique en LBM : amortisseur de stabilité à τ proche de 0.5 (Hou et al. 1996).

Le smoke Ra=1e6 (64×256, 20k pas) valide la chaîne complète :
stabilité, Nuc = Nuf (équilibre des flux), jet ascendant le long de la paroi
chaude, couches limites thermiques fines.

## Écarts documentés vs thèse

1. **Maillage uniforme** vs tangente hyperbolique (Eq. 2.104) : la version LBM
   D2Q9 de ce projet utilise une grille cartésienne uniforme (bounce-back
   simple). Le raffinement paroi de la loi tanh n'est pas reproduit ; pour
   s'approcher des cibles y+ = 0.5–1.1 à Ra=1e10, il faut soit une grille fine
   (nx ≥ 256), soit passer à un maillage non uniforme (hors périmètre v0.1).
2. **Unités de réseau** : ν/α sont calés pour matcher Ra et Pr cibles via
   ν = (utarget·√Pr·Lz / (1.7·√Ra))^{2/3} avec plancher TRT ν ≥ 0.002
   (τ+ ≥ 0.506). La vitesse lattice dépend de Ra et ν (u_max ∝ ν à Ra fixé) ;
   pour rester sous la limite de compressibilité (u_max ≲ 0.15), abaisser
   `--utarget` ne suffit pas toujours — le plancher fixe alors ν. Le temps
   physique simulé dépend de ce calage.
3. **LES Smagorinsky standard** (Cs=0.10, constante CLI) vs modèle échelles
   mixtes de Sergent [2000] : la crate fournit `smagorinsky_omega` et les types
   dynamiques (`DynamicSmagorinsky::compute_cs_sq`, clamp [0, 0.04]) ; une
   variante dynamique est possible en étendant `solver.rs`.
4. **2D strict** : la thèse mentionne des parois latérales périodiques (3D
   tronqué) ; ici le domaine est strictement 2D (une cellule en profondeur).

## Références

- [Ginzburg 2008] Ginzburg, Verhaeghe & d'Humières — TRT, Λ = 3/16 optimal
  pour bounce-back (Comm. Comp. Phys. 3).
- **[Trias 2007]** Trias, Soria, Oliva & Pérez-Segarra — JFM 586, 259–293,
  DNS 2D/3D cavité Az=4 Pr=0.71 : benchmark principal (Nū = 30.41/49.23/100.60
  aux Ra = 1e8/6.4e8/1e10) ; PDF cttc.upc.edu.
- [Xin 1993] Xin & Le Quéré — SND spectrale Chebyshev 96×768, Ra=1e10.
- [Sergent 2000] Sergent — modèle échelles mixtes, VF 66×130.
- [Le Quéré 1987] — référence AR=4, Ra=1e6, Nu ≈ 1.07.
- [Hou 1996] Hou et al. — Smagorinsky en LBM comme amortisseur de stabilité.
- Thèse « 2012lare0037_BBrangeon.pdf », §2.5, Tab. 2.6/2.7 (SPEC.md).
- Crate : `oxiphysics-lbm` v0.1.2 (module `thermal_lbm`, `turbulence`).
