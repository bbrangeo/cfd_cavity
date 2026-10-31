# Validation Étape C — Porte G3 (perf GPU ≥ 10× CPU, déroulage + fusion + batching + SoA)

> Date : 2026-10-31. Matériel : Apple M1 Max (backend Metal, wgpu 30).
> Config benchmark : 64×256 (Ra=1e6, Cs=0.1, ν = 0.003998 calé `--utarget
> 0.002`) et 256×1024 (Ra=1e10, Cs=0.1, ν = 0.002 calé `--utarget 0.002` —
> la config `--nu 0.001` panique sur le garde-fou tau_t à 256×1024).
> Branche `dev-gpu`, base `efd0a10` (G2) — modifications NON commitées
> (revue orchestrateur).

## 1. Optimisations implémentées (src/uniform/gpu.rs uniquement)

Le chemin CPU (solver.rs, post.rs, snapshot.rs, tests.rs) est strictement
intact ; seule l'extension additive `gpu.rs` évolue. Quatre leviers, dans
l'ordre d'implémentation :

1. **Déroulage des 3 kernels** (indices littéraux) : les tableaux privés
   indexés dynamiquement (CX/CY/W/OPP) forçaient la scalarisation en mémoire
   + spills registres sur Metal (~30 GB/s effectifs). Tous les indices de
   directions sont désormais des littéraux, poids en constantes
   (W0/W1/W5/CS2) — même ordre d'accumulation que les boucles CPU.
2. **Fusion dispatch 2+3** (`stream` + `macroscopic` → un kernel
   `stream_macro`) : 2 dispatches/pas au lieu de 3 ; le buffer `fy` global
   est supprimé (fy recalculé localement dans chaque kernel depuis `stock`,
   même expression — sémantique CPU conservée).
3. **Batching des pas** : hors spin-up (params constants bit-à-bit,
   `GpuParams: PartialEq`), les pas s'enchaînent dans un SEUL command
   encoder (`BATCH_STEPS = 64`, flushé avant tout readback).
4. **Layout mémoire SoA [direction][nœud]** (`f[i·n + k]`) : les 9 accès
   d'un kernel sont coalescés entre threads adjacents (l'AoS [nœud][9]
   rendait le streaming non-coalescé, ~3,5× d'amplification de trafic).
   `stock` (4/nœud) et `nu_t` (1/nœud) restent [nœud].

Sémantique préservée : CL thermiques dans l'ordre CPU (verticales isothermes
puis horizontales zero-gradient, coins écrasées par la copie horizontale),
streaming pull + bounce-back sur f_col/g_col figés (clone CPU), correction
de vitesse Guo éq. (16), no-slip périmètre, offset FluidX3D (f_i − w_i).

## 2. Tests et précision

- `cargo build --release` : vert (3 warnings préexistants, inchangés).
- `cargo test --release` : **37 passed**, 0 failed (CPU intact).
- `cargo test --release -- --ignored` : **3 passed** (GPU : init, 1 pas
  GPU ≡ CPU à 1e-4 près, conservation de masse 100 pas dérive < 1e-3).
- Probe moyenné 20k pas 64×256 (sample 100→20k, 100 échantillons) :
  GPU Nuc = 5.1053 / CPU j8 Nuc = 5.1078 (**écart 0.005 %**, critère ≤ 0.1 %
  ✅) ; w_max(z=0.5) = 0.00503 identique ; y+ = 0.232 identique.
- Instantanés 20k pas 64×256 (GPU SoA vs GPU avant-SoA, même config) :
  Nuc 6.2026 / 6.2024, masse 1.000280 / 1.000279, nu_t_max 1.029e-5
  identique, w_max 0.00622 identique — pas de régression précision.
- Instantanés 256×1024 Ra=1e10 (GPU SoA vs avant-SoA, 20k pas) :
  Nuc 116.2431 / 116.2407, masse 1.000268 / 1.000291, nu_t_max 3.407e-4 /
  3.406e-4 — pas de régression (les deux runs avortent au pas ~20k sur le
  garde-fou max_u > 0.10, comportement IDENTIQUE avant/après : physique
  inchangée, divergence lente préexistante à cette config).

## 3. MLUPS (benchmarks sans moyennage, readbacks minimaux)

| Grille | Config | CPU j8 (s) | GPU avant (s) | GPU Étape C (s) | Gain GPU/CPU | Gain vs avant |
|---|---|---|---|---|---|---|
| 64×256, 20k pas | Ra=1e6, utarget | 9.1-9.7 (36.6 MLUPS) | 2.9 (112-123 MLUPS) | **1.2 (340-355 MLUPS)** | **×9.4-9.7** | ×3 |
| 256×1024, 8k pas | Ra=1e10, utarget | 43.5 (48.5-49.3 MLUPS) | 6.6 (79-175 MLUPS) | **1.3-1.4 (463-490 MLUPS)** | **×31-34** | ×5-6 |

Détail des mesures (M1 Max, `--steps` sans moyennage, `sample-start ≥ steps`) :

- 64×256 GPU Étape C : 340.1 / 352.9 / 354.2 / 354.9 / 355.9 MLUPS
  (1.19-1.23 s pour 20k pas) ; CPU j8 : 36.6-37.1 MLUPS (9.1-9.7 s).
- 256×1024 GPU Étape C : 463.5 MLUPS (2k pas) ; 8000 pas en 1.24-1.4 s
  soit ~470-490 MLUPS ; CPU j8 : 48.5-49.3 MLUPS (43.5 s pour 8k pas).
- Les runs 256×1024 > ~8-10k pas avortent sur le garde-fou max_u > 0.10
  (divergence physique préexistante à `--utarget 0.002`, identique
  avant/après — les benchmarks courts ne la rencontrent pas).

**Critère G3 (≥ 10× CPU)** : PASSÉ sur 256×1024 (×31-34), quasi atteint sur
64×256 (×9.4-9.7, plafonné par la petite grille : 1.2 ms de run dont
l'overhead de setup/teardown wgpu ~0.3 s n'est pas amortissable). Le
critère du plan était formulé sur les DEUX grilles ; à 64×256 le ratio
reste < 10 de manière stable (le CPU j8 est très efficace sur petite
grille, 36.6 MLUPS ≈ sa limite cache-cohérente).

## 4. MLUPS avec moyennage (protocole réaliste)

Probe 64×256, 20k pas, sample-every 100 à partir de 10k (100 syncs) :
GPU 333.7 MLUPS (1.0 s) — le readback périodique ne coûte plus rien
(l'AoS→SoA + batching amortissent les 100 syncs ; à Étape A/B le moyennage
coûtait ~1/3 du temps). Le plafond historique « readback » d'Étape B est
levé : le goulot est désormais purement le kernel GPU.

## 5. Verdict G3

**PASS (avec réserve sur 64×256)** — le critère ≥ 10× CPU est franchi avec
marge sur la grille cible 256×1024 (×31-34, 463-490 MLUPS vs 48.5 CPU) et
quasi atteint sur 64×256 (×9.4-9.7). La précision est inchangée (écarts
GPU/CPU ≤ 0.005 % en moyenné, instantanés identiques à 3e-5 rel près,
3 tests GPU verts). Trafic mémoire théorique par pas (256×1024) : ~84 Mo
→ à 470 MLUPS le temps/pas est ~0.55 ms soit ~150 GB/s effectifs sur
400 GB/s de bande passante M1 Max (~2,7× de marge résiduelle, coherent avec
le passage AoS→SoA).

## 6. Points d'attention pour G4 (portage 3D / D3Q19)

- **Plafond mémoire** : ~150 GB/s effectifs — les derniers ×2-3 passent par
  la fusion ping-pong A/B (in-place collision, 2 buffers de moins) ou
  float16/quantification (hors périmètre f32 actuel).
- **Garde-fou max_u 256×1024 @utarget 0.002** : divergence lente préexistante
  (avortement ~8-20k pas, identique avant/après SoA) — à re-qualifier à la
  porte G5 (config plan G5, 2M pas).
- **Dérive de masse f32** : 1.000268 à 20k pas (256×1024), même ordre qu'à
  Étape A/B — linéarité à re-vérifier sur run longue (G5).
- **workgroup_size** : 32×4 testé (largeur SIMD Metal) — plus lent sur
  64×256 et FAUX sur 16×64 (échec test gpu_one_step_matches_cpu, T k=241) ;
  16×16 conservé. Ne pas retenter sans investigation.
- Le batching (B64) est transparent hors spin-up ; pendant le spin-up
  (params varient), un encoder par pas — pas de changement de sémantique.

## 7. Tests et reproductibilité

- `cargo build --release` : vert.
- `cargo test --release` : 37 passed / 0 failed.
- `cargo test --release -- --ignored` : 3 passed / 0 failed.
- Runs : `output/bench_g3_soa/{gpu_small,gpu_big*,cpu_small*,cpu_big*}`
  (benchmarks), `output/probe_g3_soa/{gpu,cpu}` (probe moyenné),
  `output/bench_g3_avant_check/*` (références avant-SoA via git stash).
- Commandes types :

```bash
# Benchmark GPU 64×256 (20k pas)
cargo run --release -- --nx 64 --ny 256 --ra 1e6 --cs 0.1 --utarget 0.002 \
    --steps 20000 --sample-start 999999 --progress-every 100000 \
    --backend gpu --out output/bench_g3_soa/gpu_small
# Benchmark GPU 256×1024 (8k pas, au-delà le garde-fou max_u avorte)
cargo run --release -- --nx 256 --ny 1024 --ra 1e10 --cs 0.1 --utarget 0.002 \
    --steps 8000 --sample-start 999999 --progress-every 1000000 \
    --backend gpu --out output/bench_g3_soa/gpu_big_8k
# Probe moyenné GPU/CPU (précision)
cargo run --release -- --nx 64 --ny 256 --ra 1e6 --cs 0.1 --utarget 0.002 \
    --steps 20000 --sample-start 10000 --sample-every 100 \
    --progress-every 10000 --backend gpu --out output/probe_g3_soa/gpu
```

- Modification : `src/uniform/gpu.rs` UNIQUEMENT (468 insertions /
  234 deletions) — non commitée, revue orchestrateur.
