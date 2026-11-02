# Validation référence Ra=1e8, 64×256 — re-run `--alpha-mult 1` (am1)

> **STATUT : SQUELETTE — run NON lancée** (stop utilisateur 2026-11-01).
> À compléter par la run DNS CPU décrite en §3, ou par le workflow manager.

## 0. Conclusion P0 — CSV = source de vérité ; summary.json ra1e8_val2 PÉRIMÉ

**Établi 2026-11-01 (extraction CSV, pas deviné) :**

- **CSV = source de vérité.** Recalcul sur `fields_mean.csv` de
  `output/runs_uniforme/ra1e8_val2/` (128×511, Lz=510) : Nu paroi 2ᵉ liaison
  = **30.3584** ; le `summary.json` de la même run écrit **15.4212** =
  Nu **1ʳᵉ liaison** (ix=0–1, artefact mur) × Lz — ratio
  15.4212/30.3584 = **0.508** exactement. Le summary.json a été produit par
  l'**ancien binaire** `cavity_les` (log ra1e8_val2 : chemin
  `target/release/cavity_les`, estimateur 1ʳᵉ liaison), pas par le binaire
  actuel (`cfd_les` → 2ᵉ liaison). **Tout summary.json de l'ancien binaire
  est périmé pour Nu** : utiliser les CSV + progress.log.
- **Contrôle croisé du nusselt_volume corrigé** (forme standard
  `Nu_vol = (Lz/ΔT)·⟨u_x·θ/α − ∂θ/∂x⟩`, post.rs patché 2026-11-01) sur le
  même CSV val2 : **Nu_vol = 30.46** vs Nu paroi 2ᵉ liaison 30.36 (+0.3 %) —
  l'identité volumétrique valide le choix de la 2ᵉ liaison. (L'ancienne
  forme `1 + ⟨u_y·θ⟩·Lz/ΔT` écrivait 1.0746 dans ce summary — hors sujet,
  flux vertical sans paroi isotherme.)
- **Cible Trias à viser** : Trias(1e8) = **30.41** = cible **P2′**
  (extrapolation du fit `0.2529·Ra^0.26` — **pas de DNS Trias à 1e8**, les
  ancres publiées sont 6.4e8/2e9/1e10).
- **Cible à Ra_eff du cas α×1.2** : Trias(8.33e7) ≈ **29.0** (même fit) ;
  l'écart réel du run val2 vs cette cible est **≈ +4.7 %** (Nu paroi
  30.36/29.0) à **+5.4 %** (summary 30.5745/29.0 pour la 64×256 am1.2),
  avec un effet Pr_eff 0.59 vs 0.71 attendu ≈ −2.6 %. **Le cas α×1.2 est
  ÉCARTÉ comme validation** (cas physiquement différent) ; la re-run
  `--alpha-mult 1` (§3) reste nécessaire pour un écart licite vs 30.41.

## 1. Contexte et problème

Les runs de référence 64×256 Ra=1e8 (`output/runs_uniforme/ra1e8_64x256_dns/`,
log `output/logs/ra1e8_64x256_dns.log` l.18) ont été lancées avec
`--alpha-mult 1.2`. Or (`src/uniform/solver.rs` l.137-143) :

```rust
let alpha_phys = nu / pr;                     // l.137
let alpha = alpha_phys * alpha_mult;          // l.138 — α amorti (en jeu)
let beta_eff = ra * nu * alpha_phys / lz³;    // l.143 — dérive de α_phys SEUL
```

β_eff et le « Ra vérifié depuis unités réseau » (log l.22) dérivent de α_phys,
mais la **diffusivité réellement en jeu dans l'équation thermique est le α
amorti**. Le cas réellement simulé n'est donc pas (Ra=1e8, Pr=0.71).

## 2. Valeurs de l'ancienne run (alpha-mult 1.2) — extraction faite

### 2.1 Config exacte (log `ra1e8_64x256_dns.log` l.18)

```
cargo run --release -- --nx 64 --ny 256 --ra 1e8 --cs 0.1 --nu 0.001 \
  --alpha-mult 1.2 --steps 1000000 --sample-start 200000 --sample-every 50 \
  --progress-every 10000 --jobs 4 --out output/ra1e8_64x256_dns
```

Artefacts réels : `output/runs_uniforme/ra1e8_64x256_dns/` (summary.json,
progress.log, fields_mean.csv, profiles_w.csv). Durée : 1869.5 s (0.5k pas/s).
⚠️ `--jobs 4` dans l'original (pas 8) ; `--pr` absent → défaut **0.71**
(`src/cli.rs` l.118).

### 2.2 Ra/Pr effectifs — calcul depuis les formules solver.rs (vérifié, pas deviné)

Avec ν=0.001, Pr=0.71, Ra=1e8, alpha_mult=1.2, Lz=255 :

| Quantité | Formule (solver.rs) | Valeur |
|---|---|---|
| α_phys | ν/Pr (l.137) | 1.4085e-3 |
| α en jeu | α_phys·1.2 (l.138) | 1.6901e-3 |
| β_eff | Ra·ν·α_phys/Lz³ (l.143) | 8.4942e-6 ✓ (log l.21) |
| τ | 0.5+3ν (l.149) | 0.5030 ✓ |
| τ_t | 0.5+3α (l.150) | 0.50507 ✓ |
| **Pr_eff** | ν/α_en_jeu | **0.5917** |
| **Ra_eff** | Ra·α_phys/α = Ra/1.2 | **8.333e7** |

Le cas simulé : **Ra ≈ 8.33e7, Pr ≈ 0.592** — pas (1e8, 0.71).

### 2.3 Résultats de l'ancienne run (à comparer à la re-run)

Sources : `summary.json` + fin de `progress.log`/log.

| Grandeur | Valeur am1.2 | Cible (contexte) |
|---|---|---|
| Nuc = Nuf | **30.5745** | Trias 2D à Ra=1e8 : 30.41 (« écart 0.6 % » **non comparable** — cas différent) |
| Trias 2D à Ra_eff=8.33e7 (post.rs l.377, fit 0.2529·Ra^0.26) | — | **29.00** → **écart réel ≈ +5.4 %** |
| w_max(z=0.5) | 0.01419 @ x=0.95238 | thèse (Ra=1e8, Pr=0.71) : 0.22–0.24 @ x≈0.004 — hors sujet Pr/Ra décalés |
| w_max(z=0.75) | 0.01235 @ x=0.95238 | 0.11–0.17 @ x≈0.005 |
| Nuf volume | 1.0818 | — |
| y+ paroi chaude | 1.216 | — |
| masse | 1.000000 sur toute la run | — |
| Nuc dernier pas (999999) | 30.5936 | fenêtre moyennage : 200k→1M, 16000 échantillons |

## 3. Protocole de la re-run (alpha-mult 1) — à exécuter

### 3.1 Garde-fou : le template naïf ABORTE

`--nu 0.001 --alpha-mult 1` → α = ν/Pr = 1.4085e-3 →
τ_t = 0.5+3α = **0.50423 < 0.505** → assertion `solver.rs` l.154
(`tau_t={tau_t} trop proche de 0.5 : ajuste --alpha-mult`).
Nu minimum passant le garde-fou avec am=1 : ν ≥ 0.005·Pr/3 ≈ **0.001183**.

### 3.2 Template corrigé (déviation documentée : ν 0.001 → 0.0012)

```
cargo run --release -- --nx 64 --ny 256 --ra 1e8 --cs 0.1 --nu 0.0012 \
  --alpha-mult 1 --steps 1000000 --sample-start 200000 --sample-every 50 \
  --progress-every 10000 --jobs 8 --out output/runs_uniforme/ra1e8_64x256_dns_am1
```

- ν=0.0012 → τ_t = 0.50507 (OK, identique à l'original), τ = 0.5036.
- **Alternative fidèle au calage utarget** (pas de --nu) : `--utarget 0.02`
  seul → ν_cal ≈ 0.00399, τ_t ≈ 0.5169 — mais change la résolution réseau
  (ν ×4) : comparabilité moindre avec l'original. Préférable si on veut un cas
  (1e8, 0.71) propre plutôt qu'un appariement ν.
- Avec am=1 : Ra_eff = Ra = 1e8 exactement, Pr_eff = Pr = 0.71 — la comparaison
  « Nuc vs Trias 30.41 » redevient licite.
- jobs 8 OK (10 cœurs) ; jobs 4 si CPU saturé par les runs GPU parallèles.
- Durée attendue ~1–2 h (0.5k pas/s × 1M pas, jobs 4) ; un peu moins en jobs 8.
- ⚠️ Surveiller le garde-fou `MAX_U_ABORT = 0.10` (main.rs l.51) : ν plus bas
  qu'à l'habitude de calage, mais τ identique à l'original côté momentum.

### 3.3 Comparaison attendue

- Nuc(am1) vs Nuc(am1.2)=30.5745 : quantifie l'effet (Ra_eff 8.33e7→1e8,
  Pr_eff 0.592→0.71). Attendu : Nuc(am1) **plus élevé** que 30.57 ? Non —
  attention : Trias croît avec Ra, mais Pr_eff plus élevé aussi ; l'issue est
  ouverte, c'est précisément l'objet de la run.
- Nuc(am1) vs Trias(1e8)=30.41 : écart « vrai » à cas identique.

## 4. Statut de l'ancienne run : point de sensibilité (ne pas effacer)

L'ancienne run `ra1e8_64x256_dns` (am1.2) est conservée comme **point de
sensibilité** : « alpha-mult 1.2 → Ra/Pr effectifs décalés (8.33e7, 0.592) →
Nuc 30.5745, écart réel vs Trias ≈ +5.4 % (à confirmer par la re-run am1) ».
Toute mention de « 0.1–0.6 % vs Trias » issue de cette run doit être requalifiée.

## 5. Résultats de la re-run — À REMPLIR

```
Config lancée : <commande exacte>
PID : <pid>
Ra/Pr effectifs : 1e8 / 0.71 (am=1, à vérifier dans le log l.21-22)
Nuc final : — | Nuf : — | w_max(z=0.5) : — | masse : —
Écart vs Trias(1e8)=30.41 : —
Écart vs ancienne run (30.5745) : —
```
