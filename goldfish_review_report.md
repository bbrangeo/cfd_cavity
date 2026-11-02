# Goldfish Annotated Review Report

**Target:** `VALIDATION_3D_ETAPE_V3.md` (full file)
**Date:** 2026-11-02
**Source des findings :** revue Elephant de session (diagnostic « LES 3D GPU vs Trias » + cadrage V4)
**Status: resolved**

## Prioritized Findings & AI Suggestions

### F1. Sensibilité Cs absente — courbe Nuc(Cs) jamais mesurée — Severity: High
*   **Affected Scope:** [L122-126](VALIDATION_3D_ETAPE_V3.md#L122-L126) (§6 Points ouverts)
*   **Discrepancy Description:** Le plan V3 prévoyait Cs ∈ {0.08, 0.1, 0.17} ; seul Cs=0.1 a tourné. À BL ~3-4 nœuds (y+ ~10-20), ν_sgs compense l'erreur de sous-résolution : l'écart +2.5 % peut être une compensation fortuite de Cs=0.1, pas une validation de la physique SGS. Sans Nuc(Cs), impossible de distinguer « LES correcte » de « Cs=0.1 masque l'erreur de grille ».
*   **Rationale:** C'est l'objection n°1 d'un reviewer de thèse sur une claim « LES validée » ; le point est listé comme ouvert mais sa portée (ambiguïté sur la claim) n'est pas énoncée.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Le document doit expliciter la conséquence de l'absence : la tolérance ≤ 5 % passée avec une seule valeur de Cs ne valide pas la modélisation SGS, seulement la stabilité + un écart Nuc acceptable pour ce calage.
*   ***Proposed Edit:***
    *   Location: §6, premier point ouvert
    *   Change: Après « (3 runs × ~3.2 h) ; », ajouter : « **Conséquence : avec une seule valeur de Cs, la porte V3 valide un écart Nuc acceptable pour ce calage, pas la modélisation SGS elle-même — la courbe Nuc(Cs) est requise pour appuyer la claim "LES" en thèse.** »

### F2. Ancre V3 interpolée — barre d'incertitude optimiste (±1-2 %) — Severity: Medium
*   **Affected Scope:** [L21-31](VALIDATION_3D_ETAPE_V3.md#L21-L31) (§1 cadrage ancre) + [L176](VALIDATION_3D_ETAPE_V3.md#L176) (verdict)
*   **Discrepancy Description:** L'incertitude annoncée est « ±1-2 % », mais l'exposant local Trias varie (0.265 entre 6.4e8-2e9, puis 0.31 entre 2e9-1e10) : l'interpolation log-log peut sous-estimer l'ancre réelle. L'écart V3 (+2.5 %) et cette barre se cumulent potentiellement.
*   **Rationale:** La non-linéarité documentée du document lui-même (§4 : n ≈ 0.265 puis ≈ 0.31) contredit une barre uniforme ±1-2 %.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Reformuler la barre à ±2 % et noter le sens du biais (l'exposant croissant avec Ra suggère une ancre réelle possiblement plus haute, ce qui rapprocherait l'écart de zéro — favorable, mais à lire).
*   ***Proposed Edit:***
    *   Location: §1, fin du paragraphe « incertaine à ±1-2 % »
    *   Change: Remplacer « incertaine à ±1-2 % » par « incertaine à ±2 % (l'exposant local croît avec Ra : 0.265 puis 0.31 — l'ancre réelle pourrait être légèrement plus haute que l'interpolation, rapprochant l'écart V3 de zéro) ».

### F3. Seul Nuc comparé — aucune statistique du second ordre vs Trias — Severity: Medium
*   **Affected Scope:** [L122-129](VALIDATION_3D_ETAPE_V3.md#L122-L129) (§6) + claim ν_sm/ν §7
*   **Discrepancy Description:** La claim « contenu SGS physique » (ν_sm/ν ≈ 0.09) n'est confrontée à aucune statistique turbulente : Trias fournit profils de vitesse, tensions de Reynolds, Nuc local. Le readback des profils n'est pas branché.
*   **Rationale:** Un reviewer demandera au moins le profil de Nuc local ou u_moyenne contre Trias avant d'accepter la claim SGS.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Le document liste le manque (y+/profils non extraits) mais ne connecte pas ce manque à la claim SGS du verdict.
*   ***Proposed Edit:***
    *   Location: §6, point « y+ / profils »
    *   Change: Remplacer par : « **y+ / profils : non extraits (readback non branché côté uniform3d) — conséquence directe : la claim "contenu SGS physique" (ν_sm/ν ≈ 0.09) repose sur un ratio global, sans confrontation à une statistique turbulente (profil de Nuc local, u_moyenne, tensions de Reynolds Trias). À brancher avant intégration thèse §2.5 si la claim SGS y figure.** »

### F4. §6bis « même mur que Ra=1e10, plan §3 » — faux si V4 tourne sur 128×512×256 — Severity: Medium
*   **Affected Scope:** [L142-145](VALIDATION_3D_ETAPE_V3.md#L142-L145) (§6bis, contrainte mémoire)
*   **Discrepancy Description:** Le §6bis dit que la grille fine 256×1024×512 « ≈ 61 Go — hors budget (même mur que Ra=1e10, plan §3) ». Mais le mur 61 Go du plan §3 visait une grille DNS-résolue à 1e10 ; V4 tourne effectivement sur 128×512×256 (7.6 Go, mémoire OK) — la vraie limite à 1e10 sur cette grille est la résolution BL (~3-4 nœuds pour un Ra 10× V3) et le Ma, pas la mémoire.
*   **Rationale:** Vérifié : la config V4 lancée sur tantanka est `--nx 128 --ny 512 --nz 256 --ra 1e10` (≈ 7.6 Go). La phrase « même mur » est factuellement incorrecte pour cette config.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Le document doit distinguer : (a) mémoire — non limitante sur 128×512×256 ; (b) résolution BL — dégradée d'un facteur √10 en relative vs V3, c'est la vraie limite ; (c) plan §3 61 Go — ne s'applique qu'à une grille DNS-résolue à 1e10, hors scope V4.
*   ***Proposed Edit:***
    *   Location: §6bis, point 2 (Mémoire)
    *   Change: Remplacer « (même mur que Ra=1e10, plan §3) » par « (le mur ~61 Go du plan §3 ne s'applique qu'à une grille DNS-résolue à 1e10 — une run LES à 1e10 sur 128×512×256 reste à 7.6 Go, cf. V4 ; la vraie limite à 1e10 sur cette grille est la résolution BL, pas la mémoire) ».

### F5. Convergence grille seulement à Ra=6.4e8 — non-monotonie V1→V2 non portée au verdict — Severity: Low
*   **Affected Scope:** [L172-182](VALIDATION_3D_ETAPE_V3.md#L172-L182) (§7 verdict) + §4 tableau
*   **Discrepancy Description:** La convergence empirique V1→V2 (robustesse au maillage) n'est établie qu'à Ra=6.4e8 ; à Ra=1e9, une seule grille. La non-monotonie (+1.5 % → +3.7 % en affinant) montre que la tolérance ≤ 5 % n'est pas une preuve de convergence — le verdict §7 ne porte pas cette réserve.
*   **Rationale:** Point de fond déjà documenté dans V2 §2.3, absent du verdict V3.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Une ligne de réserve dans le verdict suffit ; le fond est déjà honnête dans V2.
*   ***Proposed Edit:***
    *   Location: §7, après la ligne « Croissance inter-Ra »
    *   Change: Ajouter la ligne : « | Convergence maillage | établie à 6.4e8 seulement (V1→V2, non-monotone) ; Ra=1e9 : une seule grille | ✅ avec réserve | ».

### F6. Libellé plan V2 « Ra=1e8 DNS-like (grille résolue) » vs run à 6.4e8 non-DNS — Severity: Low
*   **Affected Scope:** [PLAN_3D_TRIAS.md L86](PLAN_3D_TRIAS.md#L86) (§4 tableau étapes)
*   **Discrepancy Description:** Le plan nomme la Porte V2 « Ra=1e8 3D DNS-like (grille résolue) » ; la run réelle est à Ra=6.4e8 et le document V2 admet ~3-4 nœuds/BL (pas DNS). Incohérence de libellé plan/doc, sans impact sur les chiffres.
*   **Rationale:** Risque de confusion pour un lecteur qui croise plan et validation.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Correction d'un libellé, une ligne.
*   ***Proposed Edit:***
    *   Location: PLAN_3D_TRIAS.md §4, ligne étape 2
    *   Change: Remplacer « Ra=1e8 3D DNS-like (grille résolue) » par « Ra=6.4e8 3D grille fine (DNS-like uniforme sous-résolu, ~3-4 nœuds/BL) ».
