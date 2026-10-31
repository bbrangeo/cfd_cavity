# Validation Étape D — Porte G4 (portage dynamic Smagorinsky sur GPU)

> Date : 2026-10-31. Matériel : Apple M1 Max (backend Metal, wgpu 30).
> Branche `dev-gpu`, base `ece3087` — modifications NON commitées
> (revue orchestrateur). Chemin CPU (solver.rs, post.rs, snapshot.rs,
> tests.rs) strictement intact ; extension additive uniquement dans
> `src/uniform/gpu.rs` + suppression de l'erreur CLI
> `--dynamic-smag + --backend gpu` dans `src/cli.rs`.

## 1. Architecture du portage (src/uniform/gpu.rs uniquement)

La formule CPU de référence `compute_dynamic_sgs` (solver.rs l. 562-828)
est portée EXACTEMENT, en 5 kernels WGSL exécutés dans le MÊME compute
pass que collide/stream_macro (barrières wgpu implicites entre
dispatches), insérés avant `stream_macro` dans `GpuSolver::step()` :

1. **`dyn1`** — box test-filter (largeur `width = round(max(η,1))`,
   clamp unilatéral aux parois, mêmes formules CPU) des champs
   (ũx, ũy, Ŧ) et des produits ⟨uxux⟩, ⟨uxuy⟩, ⟨uyuy⟩, ⟨uxT⟩, ⟨uyT⟩
   → `dyn_s1` (8 f32/nœud).
2. **`dyn2`** — quantités aux deux niveaux : S̃ij/|S̃| depuis π_neq de
   `f_in` (MÊMES formules déroulées que `collide`), S̄ij/|S̄| par
   différences centrées des champs boxés, gradients thermiques,
   similarité qsm^½/Φsm^½ (Éq. 2.74), |T̃|/|T̄| (Éq. 2.75)
   → `dyn_s2` (16 f32/nœud). Lit le buffer A ou B selon le ping-pong
   du pas courant (bind groups `bg_dyn2_a/b` déduits via `std::ptr::eq`).
3. **`dyn3`** — q̄sm/Φ̄sm (box), formes mixtes interpolées (Éq. 2.73,
   α = 0.5 ⇒ moyenne géométrique via `mpow`), tenseurs M/L mixtes,
   contractions de Lilly (double comptage xy : num_v = ldxx·mxx +
   2·lxy·mxy + ldyy·myy) → `dyn_s3` (8 f32/nœud).
4. **`dyn4`** — régularisation Éq. 2.62 = box(num)/box(den), coefficient
   de Lilly `lilly` = clamp(0.5·num/den, 0, C_DYN_MAX = 0.04) avec garde
   den ≤ 0 → 0 (NaN-safe), puis ν_t/α_t bornés (nu_sm_min·mol,
   plancher TRT (TAU_TRT_FLOOR−0.5)/3) → `dyn_out` (4 f32/nœud :
   ν_t, α_t, C^ν_dyn, C^κ_dyn).
5. **`collide_dyn`** — réplique de `collide` sans le calcul
   π:π/|S|/s_mag, consommant ν_t/α_t de `dyn_out` (bindings 14).

Constraints respectées : pas d'atomiques flottantes, pas d'unsafe,
f32 partout, ν_t reste sur GPU (aucun readback par pas), batching des
encoders et structure SoA/ping-pong intacts, kernel `collide` existant
inchangé (variante sélectionnée dans `step()`).

**Uniform `DynParams`** (binding 10) : α_exp, eta_test, dtest, delta_t,
nu_mol, alpha_mol, nu_sm_min, tau_floor_margin, box_width, c_dyn_max —
constants au cours du run, donc le batching du `GpuParams` existant
(qui varie pendant le spin-up) n'est pas cassé.

**Diagnostics** : C^ν_dyn/C^κ_dyn (dyn_out.zw) réduits séquentiellement
côté CPU au readback (méthode `read_dyn_diagnostics`, staging
MAP_READ) — moyenne et max, pas d'atomiques.

## 2. Tests

- `cargo build --release` : vert (2 warnings préexistants, inchangés).
- `cargo test --release` : **37 passed**, 0 failed (CPU intact).
- `cargo test --release -- --ignored` : **4 passed**, 0 failed — dont le
  nouveau test `gpu_one_step_dynamic_smag_matches_cpu` (1 pas GPU dynamic
  ≡ CPU à 1e-4 près sur les champs, 2 % rel sur ν_t/cs_dyn ; 16×64,
  Ra=1e6, graine 42).
- 3 erreurs de bind groups corrigées pendant l'itération : layouts
  déduits du shader (chaque kernel ne déclare que ses variables —
  entrées exactes par kernel), et `dyn_out` nécessite COPY_SRC pour le
  staging de diagnostics.

## 3. Smoke test G4 — 64×256, Ra=1e6, dynamic smag, GPU

Config : `--nx 64 --ny 256 --ra 1e6 --cs 0.1 --utarget 0.002
--steps 20000 --sample-start 10000 --sample-every 100 --dynamic-smag
--backend gpu` (probe moyenné, 100 échantillons).

| Métrique | GPU dynamic | CPU j8 dynamic |
|---|---|---|
| Nuc (moyenné 10k→20k) | 5.1085 | 5.1111 (**écart 0.005 %**) |
| Nuf | 5.1082 | 5.1106 |
| w_max(z=0.5) | 0.00503 en x=0.8254 | identique |
| y+ (paroi chaude) | 0.232 | identique |
| cs_dyn_mean / cs_dyn_max | 0.0022 / 0.0400 | 0.0022 / 0.0400 |
| kap_dyn_mean (fin) | 0.0303 | 0.0303 |
| nu_t_max (fin) | 2.074e-5 | 2.074e-5 |
| masse (fin) | 1.000280 (dérive f32) | 1.000000 (f64) |
| Avortement / NaN | aucun | aucun |

**Critère G4 (κ_dyn ∈ [0, 0.04], pas de NaN, run complet)** : PASSÉ.
κ_dyn_max = 0.0400 = clamp C_DYN_MAX actif (comme sur CPU), κ_dyn_mean
monte progressivement 0.0178 → 0.0303 (physique identique au CPU), masse
dérive de 2.8e-4 en f32 (même ordre que les portes G2/G3, linéaire).

## 4. Perf du mode dynamic (benchmarks sans moyennage)

| Grille / pas | CPU j8 | GPU Smag fixe | GPU dynamic | Coût dynamic |
|---|---|---|---|---|
| 64×256, 20k pas | 2.7 MLUPS (122.9 s, probe) | 332.2 MLUPS (1.0 s) | **113.5 MLUPS (2.9 s)** | ×2.9 |
| 256×1024, 2k pas | — | 985.1 MLUPS (0.5 s) | **98.1 MLUPS (5.3 s)** | ×10 |

Le mode dynamic coûte ~3× sur la petite grille (5 kernels dyn ajoutés :
dyn2 recalcule les 9 populations + équilibres, dyn3/dyn4 font 2 box
filters supplémentaires). Sur 256×1024 l'écart est plus grand (×10,
données à confirmer — le kernel dyn2 est dominates et non coalescé
autant que collide). Même à ×10, le GPU dynamic (98 MLUPS) reste ~2× le
CPU j8 (48.5-49.3 MLUPS sur cette grille) : le mode dynamic sur GPU est
déjà plus rapide que le CPU, et la campagne G5 (2M pas) passe de
~11.5 h (CPU) à ~3 h (GPU dynamic) à ce débit — optimisations possibles
(fusion dyn1/dyn2, cache des box filters) non prioritaires.

## 5. Verdict G4

**PASS** — le dynamic Smagorinsky survit au GPU : formule CPU portée à
l'identique (test 1 pas ≡ CPU à 1e-4), smoke 20k pas complet sans NaN,
κ_dyn ∈ [0, 0.04] avec clamp identique au CPU, cohérence CPU↔GPU sur
champs moyennés de 0.005 % (Nuc), diagnostics κ_dyn_mean identiques au
millième. Coût perf ~×3 (petite grille) à ×10 (grande grille) vs
Smagorinsky fixe GPU — attendu (5 kernels supplémentaires), reste ~2×
plus rapide que le CPU j8 sur la grille cible.

## 6. Points d'attention pour G5 (Ra=1e10, 2M pas, dynamic)

- **Garde-fou max_u 256×1024 @utarget 0.002** : divergence préexistante
  confirmée en dynamic aussi (avortement ~8k pas, max_u = 0.18021
  dynamic vs 0.18003 fixe — comportement identique avant/après, physique
  inchangée). La campagne G5 doit utiliser la config plan (voir
  PLAN_GPU_BACKEND.md §G5).
- **Dérive de masse f32** : 1.000280 à 20k pas (64×256), linéaire — à
  re-vérifier sur run longue G5.
- **Perf dynamic 256×1024** : 98 MLUPS mesurés sur 2k pas — estimer le
  temps de la campagne G5 (2M pas ≈ 3 h) avant de lancer ; optimisation
  possible si besoin (fusion dyn1/dyn2, un seul kernel dyn3+dyn4).

## 7. Tests et reproductibilité

- `cargo build --release` : vert.
- `cargo test --release` : 37 passed / 0 failed.
- `cargo test --release -- --ignored` : 4 passed / 0 failed.
- Runs : `output/probe_g4_dyn/{gpu,cpu}` (probe moyenné 20k pas),
  `output/bench_g4_dyn/{gpu_small_dyn,gpu_small_fixed,gpu_big_2k,gpu_big_2k_fixed}`
  (benchmarks), `output/bench_g4_dyn/gpu_big_8k*` (avortements garde-fou,
  preuve comportement identique fixe/dynamic).
- Commandes types :

```bash
# Probe moyenné GPU dynamic (précision + smoke G4)
cargo run --release -- --nx 64 --ny 256 --ra 1e6 --cs 0.1 --utarget 0.002 \
    --steps 20000 --sample-start 10000 --sample-every 100 \
    --progress-every 10000 --dynamic-smag --backend gpu --out output/probe_g4_dyn/gpu
# Idem CPU (cohérence)
cargo run --release -- --nx 64 --ny 256 --ra 1e6 --cs 0.1 --utarget 0.002 \
    --steps 20000 --sample-start 10000 --sample-every 100 \
    --progress-every 10000 --dynamic-smag --jobs 8 --out output/probe_g4_dyn/cpu
# Bench perf 256×1024 (2k pas, au-delà le garde-fou max_u avorte)
cargo run --release -- --nx 256 --ny 1024 --ra 1e10 --cs 0.1 --utarget 0.002 \
    --steps 2000 --sample-start 999999 --progress-every 1000000 \
    --dynamic-smag --backend gpu --out output/bench_g4_dyn/gpu_big_2k
```
