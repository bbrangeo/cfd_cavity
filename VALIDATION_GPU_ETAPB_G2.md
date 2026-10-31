# Validation Étape B — Porte G2 (précision f32+offset à Ra=1e8)

> Date : 2026-10-31. Matériel : Apple M1 Max (backend Metal, wgpu 30).
> Config : 128×511 (Az = 4.016), Ra=1e8, Pr=0.71, Cs=0.1, Pr_t=0.9,
> **ν = 0.001 forcée** (`--nu 0.001`, τ+ = 0.503, τ_t = 0.5042),
> `--alpha-mult 1.2`, β_eff = 1.0556e-6 — même config que la run CPU de
> référence validée (README l.190 : 128×511, 2M pas, Nuc = 30.45 vs Trias 30.41).

## 1. Protocole

Config identique au README (l.148-151) adaptée à la référence 128×511 : le
calage `--utarget` est contourné par `--nu 0.001` (le calage utarget donne
ν ≥ 0.0025 à 1e8/128×511 — plancher TRT —, trop amorti pour Nuc ≈ 30 ;
la ν forcée est la config qui reproduit la physique cible). Deux runs
like-for-like (mêmes unités réseau, même graine 42) :

```bash
# CPU (référence, rayon j8)
cargo run --release -- --nx 128 --ny 511 --ra 1e8 --cs 0.1 --nu 0.001 \
    --alpha-mult 1.2 --steps 1000000 --sample-start 300000 --sample-every 100 \
    --progress-every 50000 --jobs 8 --out output/cpu_ra1e8_etapb_1m
# GPU (Étape A)
cargo run --release -- --nx 128 --ny 511 --ra 1e8 --cs 0.1 --nu 0.001 \
    --alpha-mult 1.2 --steps 1000000 --sample-start 300000 --sample-every 100 \
    --progress-every 50000 --backend gpu --out output/gpu_ra1e8_etapb_1m
```

Probe préalable 200k pas (`cpu/gpu_ra1e8_etapb`) : **dérive de masse GPU
linéaire** (pente 1.43e-8/pas, +2.8e-3 à 200k — même ordre qu'Étape A),
pas de divergence → run complet 1M lancé.

## 2. Résultats (champs moyens, fenêtre 300k→1M, 7 000 échantillons)

| Quantité | CPU (j8) | GPU (wgpu) | Écart | Critère plan §G2 |
|---|---|---|---|---|
| Nuc (moyen) | 30.1483 | 30.1618 | **+0.045 %** | ≤ 2 % ✅ |
| Nuf (moyen) | 30.1482 | 30.1629 | +0.049 % | comparable ✅ |
| w_max(z=0.5) | 0.006743 (BL chaude x=0.047) | 0.006741 (x=0.047) | **−0.027 %** | ≤ 2 % ✅ |
| w_max(z=0.75) | 0.005940 (x=0.063) | 0.005933 (x=0.063) | −0.104 % | ≤ 2 % ✅ |
| ν_t_max (champs moyens) | 1.5098e-5 @ (126, 339) | 1.5107e-5 @ (126, 339) | **+0.059 %** | ≤ 2 % ✅ |
| y+ (paroi chaude) | 0.630 | 0.630 | 0 % | comparable ✅ |
| écart vs Trias 2D (30.41) | −0.86 % | −0.82 % | — | cohérence CPU/GPU ✅ |

Note w_max(z=0.5) : le max global du plan médian est quasi dégénéré entre les
 deux couches limites (chaude/froide) — l'argmax global CPU tombe côté froid
 (x=0.953) et GPU côté chaud (x=0.047), mais les valeurs coïncident à 4
 décimales (0.0067) et la position BL chaude est identique (x=0.047).

Instantanés finaux (pas 1M, moy. 10 derniers diagnostics) : Nuc CPU 30.2428 /
GPU 30.2562 (+0.044 %) ; w_max 0.006837 / 0.006839 (+0.029 %) ;
ν_t_max 1.5284e-5 / 1.5305e-5 (+0.137 %).

Écarts champ moyen GPU−CPU sur 128×511 nœuds (fin de run) :
ρ : 0 (non lu — cf. §4) · ux : 3.4e-5 (échelle 4.5e-3) · uy : 4.3e-5
(échelle 6.8e-3) · T : 2.6e-3 (ΔT = 1) · ν_t : 5.9e-8 (échelle 1.5e-5).
Profil w(z=0.5/0.75) : écart max 3.1e-5.

## 3. Dérive de masse GPU (point d'attention hérité d'Étape A)

| Run | Pas | Dérive GPU | Pente | Nature |
|---|---|---|---|---|
| Probe 128×511 @1e8 | 200k | +2.845e-3 | 1.43e-8/pas | linéaire |
| Run 128×511 @1e8 | 1M | **+1.375e-2** | 1.37e-8/pas | **linéaire** (R² ≈ 1) |
| CPU (toutes) | — | 0 exactement | 0 | (f64, bounce-back bijectif) |

L'extrapolation linéaire du probe (1.42e-2 à 1M) est confirmée : la dérive
reste **strictement linéaire** (~1.4e-8/pas, erreurs d'arrondi f32 de la
collision Σf'), **pas d'explosion** à Ra=1e8 alors que f^neq/f est ~100×
plus grand qu'à 1e6 — l'offset FluidX3D absorbe la précision du
non-équilibre comme prévu. La dérive (+1.4 % de masse à 1M pas) reste
absorbée par la dynamique : Nuc/ν_t/w̄ GPU = CPU à < 0.15 %.

## 4. MLUPS (128×511, 1M pas, avec moyennage)

| Backend | Temps | MLUPS | Gain |
|---|---|---|---|
| CPU j8 | 1825 s | 35.8 | ×1.0 |
| GPU wgpu | 688 s | 95.1 | **×2.7** |

Benchmark court sans moyennage (20k pas, readbacks minimaux) :
GPU 108.7 MLUPS (12.0 s) vs CPU j8 49.3 (26.5 s) — ×2.2.
Le gain GPU absolu monte légèrement vs 64×256 (90 → 95-109 MLUPS) mais le
gain relatif (~×2.7) ne dépasse pas le ×3.2 d'Étape A : le CPU j8 monte
aussi sur grande grille (28.5 → 35.8 MLUPS) et le readback périodique
(7 000 syncs au pas ≥ 300k, ~700 s cumulés estimés) plafonne le GPU. Le
critère G3 (≥ 10× CPU) se jouera sur 256×1024 avec readback réduit
(réductions GPU ou readback espacé).

## 5. Verdict G2

**PASS** — la précision f32+offset suffit à Ra=1e8 : tous les écarts
GPU↔CPU ≤ 0.15 % (critère ≤ 2 %), la dérive de masse f32 reste linéaire
(1.4e-8/pas) et absorbée, aucune divergence sur 1M pas, Nuc GPU = 30.16
vs Trias 30.41 (−0.8 %, même écart que le CPU).

## 6. Points d'attention pour G3 (Ra=1e10, 256×1024)

- **Dérive de masse linéaire** : extrapolée à 2M pas (config G5), la dérive
  atteindrait ~+2.7e-2. Restée absorbée à 1e8, elle devra être re-mesurée à
  1e10 (porte G2/G5) ; si elle déborde, la parade FluidX3D est la
  renormalisation périodique Σf (à décider sur evidence, pas avant).
- **Readback readback** : à 128×511, le moyennage (sample-every 100) coûte
  ~1/3 du temps GPU (95 vs 109 MLUPS sans readbacks). À 256×1024, réduire
  la fréquence de sync ou passer les réductions sur GPU (G3).
- **w_max(z=0.5) argmax dégénéré** : comparer la valeur + la position BL
  chaude (l'argmax global bascule de paroi pour des différences f32
  infimes — artefact de comparaison, pas d'écart physique).
- **Config G2 ≠ config planche CLI par défaut** : `--nu 0.001 --alpha-mult
  1.2` est la config physique validée à 1e8/128×511 (README l.190) ;
  `--utarget` seul donne ν ≥ 0.0025 (plancher TRT) et Nuc ≈ 36 à 200k pas.

## 7. Tests

- `cargo build --release` : vert.
- Aucune modification de code (campagne de validation pure) — branche
  `dev-gpu`, commit de base `d714631`.
- Runs : `output/{cpu,gpu}_ra1e8_etapb` (probe 200k),
  `output/{cpu,gpu}_ra1e8_etapb_1m` (1M pas, référence G2),
  `output/bench_etapb/{cpu,gpu}` (chronométrage 20k).
