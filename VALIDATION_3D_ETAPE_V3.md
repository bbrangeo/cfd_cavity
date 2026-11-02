# Validation Étape V3 — LES Ra=1e9, 128×512×256, Smagorinsky statique Cs=0.1 — Porte V3 (cavité DHC AR=4)

> Date : 2026-11-02. Matériel : **Quadro RTX 6000 24 Go (Vulkan, tantanka)**.
> Branche `dev-3d`. Suite directe de la Porte V2
> (`VALIDATION_3D_ETAPE_V2.md`, Ra=6.4e8, Nuc 51.06, +3.7 %).

## 1. Cadrage de l'ancre Ra=1e9 — PAS DE DNS 3D TRIAS À CE Ra

**Résultat de la passe librarian (2026-11-02, PDF officiels extraits)** :

- Les DNS 3D de Trias et al. s'arrêtent aux Ra ∈ **{6.4e8, 2e9, 1e10}**
  (JFM 586 (2007), Table 2 p. 271 : Case A/B/C 3D = 49.24 / 66.63 / 101.70) ;
- La base étendue 2010 (IJHMT 53(4), Part I & II) couvre
  6.4e8→1e11 mais **2D uniquement** ;
- Le README officiel des données CTTC
  (`cttc.upc.edu/downloads/DHC_AR4_2Dvs3D/README`) liste exactement les
  configurations 6.4e8, 2e9 et 1e10.

**Aucune valeur Nuc 3D à Ra=1e9 n'existe dans la littérature Trias.**

L'ancre V3 est donc **interpolée log-log** depuis la Table 2 du JFM 2007 :

```
n = ln(66.63/49.24) / ln(2e9/6.4e8) = 0.3024/1.1394 ≈ 0.265
Nuc(Ra=1e9) ≈ 49.24 × (1e9/6.4e8)^0.265 ≈ 55.4 ± 0.5
```

Cette interpolation est documentée comme **incertaine à ±1-2 %** (la
tendance log-log des points Trias 3D n'est pas une loi de puissance pure :
n ≈ 0.265 puis ≈ 0.31 entre 2e9 et 1e10). L'écart V3 doit donc être lu
avec cette barre d'erreur en tête, et non comme un verdict absolu.

## 2. Protocole

### 2.1 Config du run V3 (telle qu'exécutée sur tantanka)

```bash
./target/release/cavity_les --mode uniform3d --nx 128 --ny 512 --nz 256 \
    --ra 1e9 --pr 0.71 --backend gpu --steps 400000 \
    --utarget 0.01 --sample-start 100000 --progress-every 10000 \
    --out output/runs_uniform3d/ra1e9_128x512x256
```

- Même grille que V2 : **128×512×256** (AR = 4.02, 16.78M nœuds),
  z PÉRIODIQUE, démarrage à froid ;
- **Ra = 1e9** (×1.56 vs V2) ⇒ ν = 0.002000 (plancher TRT, τ+ = 0.506),
  α = 2.817e-3, **β_eff = 4.2222e-5** (vs 3.1406e-5 en V2) ;
- **LES : Smagorinsky statique Cs = 0.1**, Pr_t = 0.9 — la sous-résolution
  est volontaire (Δx uniforme 7.87e-3), le ν_sgs compense la BL
  sous-résolue ;
- Fenêtre de moyennage 100k→400k pas, 3000 échantillons.

## 3. Run V3 — trace et bilan (400k pas)

### 3.1 Trace (points représentatifs, progress.log tantanka)

| Pas | w_max | Nuc | Nuf | Masse | nu_t_max |
|---|---|---|---|---|---|
| 0 | 0.00000 | 4.0239 | 4.0239 | 1.000000 | 0 |
| 10 000 (pic transitoire) | 0.08025 | 74.4770 | 74.4768 | 0.999994 | 3.077e-4 |
| 20 000 | 0.08604 | 66.3265 | 66.5082 | 0.999991 | 3.473e-4 |
| 50 000 | 0.05823 | 56.8681 | 56.7184 | 0.999987 | 2.255e-4 |
| 100 000 | 0.05969 | 57.5280 | 56.4901 | 0.999984 | 2.452e-4 |
| 200 000 | 0.04776 | 56.5279 | 56.8608 | 0.999980 | 1.736e-4 |
| 300 000 | 0.04980 | 56.7967 | 56.9716 | 0.999977 | 1.726e-4 |
| 399 999 | 0.05369 | 56.9382 | 57.1133 | 0.999973 | 1.802e-4 |

- Aucun avortement, aucun NaN, 400k pas complets ;
- Nuc converge dès ~50k pas et oscille 56.5-57.5 ensuite ;
- **νt_max/ν ≈ 0.09** (1.7-1.8e-4 sur 0.002) en fenêtre — la sous-grille
  est active près des parois, LE rôle de la Smagorinsky en LES.

### 3.2 Bilan fenêtre 100k→400k (3000 échantillons)

| Métrique | Run V3 3D | Ancre interpolée | Écart |
|---|---|---|---|
| **Nuc (moyenné)** | **56.8460** | 55.44 ± 0.5 | **+2.5 %** ✅ (tolérance ≤ 5 %) |
| Nuf | 56.8199 | — | — |
| Masse finale | 0.999973 | 1 | dérive 0.003 % |
| Pente de masse | ~−7e-10/pas | — | ✅ ≪ garde-fou 2 %/500k |
| MLUPS (moyennage inclus) | 541.4 | — | 3.19 h |

**PORTE V3 (écart Nuc ≤ 5 % + stabilité 400k pas sans NaN) : PASS**
*— avec la réserve documentée : ancre interpolée (±1-2 %), pas un DNS.*

### 3.3 Note sur le premier bilan affiché (FAIL 15.4 % — artefact d'ancre)

Le bilan imprimé par le binaire à la complétion affichait
« ancre Trias 3D (Ra = 1.00e9) = 49.24 | écart = 15.4 % | FAIL » : c'est
un **artefact d'ancre**, l'ancre Case A (Ra=6.4e8) étant codée en dur
pour tous les Ra dans `main.rs` (l.978). **Corrigé (commit `ba0a086`)** :
l'ancre est maintenant fonction du Ra (Case A/B/C + interpolation 1e9).
Avec la bonne ancre interpolée, l'écart réel est **+2.5 % (PASS)** —
aucune re-run nécessaire, le Nuc moyenné est inchangé.

## 4. Convergence inter-Ra (2D, V1, V2, V3)

| Run | Ra | Nuc maison | Ancre Trias | Écart | Remarque |
|---|---|---|---|---|---|
| 2D (256×1024) | 6.4e8 | 50.89 | 49.23 | +3.4 % | Trias AA2D |
| 3D V1 (96×384×192) | 6.4e8 | 49.98 | 49.24 | +1.5 % | Trias Case A |
| 3D V2 (128×512×256) | 6.4e8 | 51.06 | 49.24 | +3.7 % | Trias Case A |
| **3D V3 (128×512×256)** | **1e9** | **56.85** | **55.44 (interpolé)** | **+2.5 %** | **pas de DNS 3D Trias** |

- Les 4 voies encadrent l'ancre à ≤ 4 % ;
- **Croissance physique cohérente** : Nuc ~ Ra^1/3 donnerait
  51.06 × (1e9/6.4e8)^(1/3) ≈ 59.2 pour V3 ; le 56.85 mesuré est
  légèrement sous — cohérent avec un exposant local Trias ≈ 0.265
  (< 1/3), qui donne 57.6 attendu → écart résiduel ~+1.2 % seulement.
  La croissance Ra^0.265 des DNS Trias 3D est bien reproduite par la LES.

## 5. Perf

| | V3 (Ra=1e9) | V2 (Ra=6.4e8) |
|---|---|---|
| MLUPS (400k pas, moyennage inclus) | 541.4 | 585.2 |
| Durée | 3.19 h | 3.19 h |

Le léger recul (−7.5 %) est cohérent avec le νt plus actif (calcul
Smagorinsky légèrement plus coûteux à νt/ν ≈ 0.09 vs 0.07).

## 6. Points ouverts (non bloquants)

- **Sensibilité Cs** : non lancée (Cs = 0.1 seul). Le plan V3 prévoyait
  Cs ∈ {0.08, 0.1, 0.17} — à lancer sur tantanka si requis pour la thèse
  (3 runs × ~3.2 h) ;
- **Ancre interpolée** : si un DNS 3D Ra=1e9 indépendant (autre équipe)
  est identifié, l'ancre peut être remplacée ;
- **y+ / profils** : non extraits (readback non branché côté uniform3d).

## 6bis. Pourquoi pas Ra=2e9 (ancre DNS directe Trias Case B) — cadrage 2026-11-02

Trias Case B (JFM 586 2007, Table 2) fournit un DNS 3D direct à Ra=2e9
(Nuc = 66.63) : une run à ce Ra supprimerait l'interpolation de l'ancre
V3. **Non visé, pour deux contraintes documentées** :

1. **Plancher TRT** : ν réseau ne peut pas descendre sous ~0.002
   (τ+ = 0.506). À grille fixée (128×512×256, u_scale = utarget·lz),
   Ra=2e9 imposerait ν ≈ 0.0012 — sous le plancher ; comme V3 à 1e9,
   il faudrait tricher sur β_eff (≈ 8.4e-5 attendu, ×2.4 la valeur V1),
   ce qui dégrade la fidélité Boussinesq du forçage ;
2. **Mémoire** : la vraie solution (grille plus fine, ν ∝ lz :
   256×1024×512 → ν ≈ 0.0024 > plancher) coûte 134M nœuds × 456
   octets ≈ 61 Go — hors budget RTX 6000 24 Go et M1 Max 64 Go
   (même mur que Ra=1e10, plan §3).

**⚠️ Point non démontré (honnêteté) : le mur β de compressibilité en 3D
uniforme n'est PAS mesuré.** Le mur documenté (campagne octree,
PLAN_ECHELLES_MIXTES_DYNAMIQUES §5.2/§6 : β* ∈ [4.08, 5.43]e-4,
ρ_dev ∝ β linéaire) est porté par le forçage Guo évalué sur la grille de
base grossière — les β_eff 3D uniforme (3.1e-5 à 8.4e-5) sont 5-15×
sous la borne, et l'extrapolation ρ_dev ∝ β donne ρ_dev ~2e-5 à Ra=2e9,
deux ordres de grandeur sous la zone de danger. **Le plancher TRT (pas
le mur β) est la contrainte limitante en 3D uniforme.** Le risque n°4 du
plan (« ρ_dev vs β à documenter dès V1 ») reste ouvert : le mode
uniform3d ne logge pas ρ_dev (contrairement au mode octree,
`max_rho_dev` main.rs l.738) — à ajouter et rapporter avant toute
extension de Ra.

## 7. Verdict V3

| Critère | Résultat | Verdict |
|---|---|---|
| Nuc vs ancre interpolée 55.44 ± 5 % | 56.85 (+2.5 %) | ✅ PASS (réserve ancre ±1-2 %) |
| Stabilité 400k+ pas sans NaN | 400k pas, aucun avortement | ✅ PASS |
| Pente masse | ~0.03 %/500k — rapportée | ✅ PASS |
| ν_sm/ν rapporté | νt_max/ν ≈ 0.09 en fenêtre | ✅ documenté |
| Croissance inter-Ra | cohérente avec exposant Trias 0.265 | ✅ |

**Verdict global : PORTE V3 PASS (ancre interpolée, réserve documentée).**

## 8. Prochaines étapes

- (Optionnel) Sensibilité Cs ∈ {0.08, 0.17} sur tantanka (2 × ~3.2 h) ;
- Intégration thèse §2.5 : valeurs finales P0-P3', tableau de convergence
  2D/V1/V2/V3 (§4), cadrage ancre interpolée V3 ;
- Push `ba0a086` + ce bilan.
