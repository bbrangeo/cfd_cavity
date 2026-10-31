# Validation Étape A — Porte G1 (backend GPU wgpu, solveur uniforme)

> Date : 2026-10-31. Matériel : Apple M1 Max (backend Metal, wgpu 30).
> Config : 64×256, Ra=1e6, Pr=0.71, Cs=0.1, Pr_t=0.9, 200 000 pas,
> sample-start 100 000, sample-every 100 (1 000 échantillons), graine 42.

## 1. Protocole

Deux runs like-for-like (mêmes unités réseau : ν = 0.018557, α = 0.026136,
β_eff = 2.9250e-5, Ra vérifié 1.0000e6) :

```bash
# CPU (référence)
cargo run --release -- --nx 64 --ny 256 --ra 1e6 --cs 0.1 --steps 200000 \
    --sample-start 100000 --sample-every 100 --progress-every 20000 \
    --jobs 8 --out output/cpu_ra1e6_etapa
# GPU (Étape A)
cargo run --release -- --nx 64 --ny 256 --ra 1e6 --cs 0.1 --steps 200000 \
    --sample-start 100000 --sample-every 100 --progress-every 20000 \
    --backend gpu --out output/gpu_ra1e6_etapa
```

## 2. Résultats (champs moyens, fenêtre 100k→200k)

| Quantité | CPU (j8) | GPU (wgpu) | Écart | Critère plan §2 |
|---|---|---|---|---|
| Nuc (2ᵉ liaison, moyen) | 9.8851 | 9.8850 | **0.001 %** | ≤ 1 % ✅ |
| Nuf (2ᵉ liaison, moyen) | 9.8851 | 9.8851 | 0.000 % | comparable ✅ |
| w_max(z=0.5) | 0.02604 en x=0.841 | 0.02604 en x=0.841 | < 1e-5 abs. | comparable ✅ |
| w_max(z=0.75) | 0.02212 en x=0.873 | 0.02212 en x=0.873 | < 1e-5 abs. | comparable ✅ |
| y+ (paroi chaude) | 0.213 | 0.213 | 0 % | comparable ✅ |
| écart vs Trias 2D | 7.7 % | 7.7 % | 0 % | (cohérence CPU/GPU) |

Champs instantanés finaux (pas 200k, écart max GPU−CPU) :
ρ : 2.6e-3 (dérive de masse, cf. §4) · ux : 8.4e-7 · uy : 1.7e-6 · T : 5.8e-5.
Profil w(z=0.5) : écart max 1.6e-6 (w_max ~0.026). Profil T(z) : 5.3e-5 (ΔT=1).

Nusselt instantanés (pas 200k) : 2ᵉ liaison CPU 9.8851 / GPU 9.8850
(0.001 %) ; 1ʳᵉ liaison (sentinelle artefact mur) CPU 5.7302 / GPU 5.7301
(0.002 %).

## 3. Verdict G1

**PASS** — le kernel GPU reproduit le CPU à Ra=1e6 : écart Nuc 0.001 % ≪ 1 %,
champs comparables à la précision f32 près, aucune divergence sur 200k pas.

## 4. Points d'attention (documentés pour G2)

- **Dérive de masse f32** : la masse GPU dérive linéairement de +2.6e-3 sur
  200k pas (~1.3e-8/pas, erreurs d'arrondi f32 de la collision Σf' = Σf) ;
  le CPU conserve la masse exactement (f64). Effet négligeable sur Nuc/w̄ à
  Ra=1e6 (l'offset FluidX3D absorbe la précision du non-équilibre), à
  re-mesurer à Ra=1e8 (porte G2) où f^neq est plus grand.
- **MLUPS** : GPU 90 (200k pas, 36.4 s) vs CPU j8 28.5 (114.8 s) — gain ×3.2
  sur 64×256 (petite grille, over-head dispatch non amorti). Le critère G3
  (≥ 10× CPU) se joue sur les grandes grilles (256×1024).
- **`--dynamic-smag` non porté** : erreur CLI claire (seul Smagorinsky Cs fixe
  est sur le GPU) — Étape D (G4).
- **Readback GPU→CPU** : effectué uniquement les pas où les champs sont
  consommés (diagnostics/moyennage/snapshots), pas à chaque pas.

## 5. Tests

- `cargo build --release` : vert.
- `cargo test --release` : 37 tests existants verts (0 failed), 3 ignorés.
- `cargo test --release -- --ignored` : 3 tests GPU verts —
  `gpu_init_from_solver` (stock initial = champs CPU), `gpu_one_step_matches_cpu`
  (1 pas GPU ≈ 1 pas CPU, tolérance 1e-4), `gpu_mass_conserved_100_steps`.
