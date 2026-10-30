# Cas d'étude : cavité différentiellement chauffée 2D — Comparaison SND (Xin 1993) vs LES

Référence thèse : « 2012lare0037_BBrangeon.pdf », §2.5, pages 65–71.
Texte extrait : `docs_refs/thesis_p80_110.txt` (sections 2.5.1–2.5.3).

## Configuration physique (thèse, Tab. 2.6 et Fig. 2.10)

- Cavité fermée, rapport de forme **Az = Lz/Lx = 4**
- Fluide : air, **Pr = 0,71**, hypothèse Boussinesq
- **RaH = 1 × 10¹⁰** (cas bien documenté)
- Unités adimensionnelles : u*=u·Lz/ν, θ=(T−Tf)/ΔT

## Conditions aux limites (Tab. 2.6)

| Paroi | Vitesse | Température |
|---|---|---|
| Paroi verticale gauche (chaude) | (0,0,0) | **θc = +0,5** |
| Paroi verticale droite (froide) | (0,0,0) | **θf = −0,5** |
| Plafond (horizontal) | (0,0,0) | adiabatique : ∂θ/∂z = 0 |
| Plancher (horizontal) | (0,0,0) | adiabatique : ∂θ/∂z = 0 |
| Parois latérales (avant/arrière, y) | — | **périodique** (Eq. 2.103) |

En 2D strict : parois horizontales = plafond/plancher adiabatiques, pas de direction y
(ou y périodique tronqué).

Attention : la thèse normalise θc=+0,5 / θf=−0,5 (donc ΔT=1, θ moyen 0).
La littérature Xin & Le Quéré utilise souvent Tc=1, Tf=0 — les deux sont équivalents
à une translation près, mais rester cohérent.

## Maillage (§2.5.3, Eq. 2.104)

Répartition en tangente hyperbolique en x (et z), loi de Henkes & Hoogendoorn [1993] :

```
x(k) = (1/2) * [ 1 + tanh( γ * (2(k-1)/Nx - 1) ) / tanh(γ) ] * Lx
```

- γ = 0 → répartition uniforme
- Colonnes de référence (Tab. 2.7) :
  - SND2D1 (Xin 1993, spectrale Chebyshev) : 96 × 768
  - SND2D2 (Sergent 2000, VF) : 130 × 514
  - C2D1 (LES thèse) : 64 × 128
  - C2D2 : 96 × 320
  - C2D3 : 128 × 320

## Quantités de comparaison (Tab. 2.7)

- w_max à z=0,5 et z=0,75 (+ position x du max)
- yc+, y_adiab+ (première cellule paroi, adimensionnel)
- ν_sm/ν_max, κ_sm/κ_max (contribution sous-maille — LES)
- **Nuc** (Nusselt paroi chaude) : cible SND2D1 ≈ 102, thèse LES ≈ 100,2–100,5
- S^1/2 (écart-type fluctuations vitesse) ≈ 1,10–1,20
- Champs moyens : fonction de courant + isothermes entre −0,4 et 0,4

## Critères de validation

1. Nuc paroi chaude ≈ Nuf paroi froide et régime asymptotique avant moyennage
2. w_max(z=0,5) ≈ 0,22–0,24 en x ≈ 0,004 (couche limite chaude)
3. w_max(z=0,75) ≈ 0,11–0,17 en x ≈ 0,005
4. Stratiﬁcation centrale : θ ≈ linéaire, v ≈ 0
5. Rapport ν_sm/ν max ≈ 1 (sinon LES sous-résolue = SND sous-résolue)

## Plan d'implémentation

### Phase 1 — Géométrie + maillage (oxiphysics)
- Réseau tanh (Eq. 2.104) en x et z, génération grille 2D
- Vérification : y+ paroi ≈ 0,5–1,1 comme Tab. 2.7
- Export du maillage (VTK)

### Phase 2 — Solveur LES
- Équations Navier-Stokes incompressible + Boussinesq, décrites §2.2
- Modèle sous-maille échelles mixtes [Sergent 2000] (ou Smagorinsky dynamique selon API oxiphysics)
- CL : no-slip parois, isotherme ±0,5, adiabatique horizontales, périodique y
- Post-traitement : Nusselt local/intégré, champs moyens temporels

## Références clés

- [Xin 1993] Xin & Le Quéré — SND spectrale Chebyshev 96×768, Ra=1e10
- [Sergent 2000] Sergent — modèle échelles mixtes, VF 66×130
- [Le Quéré 1987], [Trias 2006/2010], [Gorobets 2008] — SND 3D 190×128×462
