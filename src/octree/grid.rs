//! Grille quadtree 2 niveaux (Étape A) : ℓ=0 grossier (nx0 × ny0), ℓ=1 fin
//! (ratio 2:1) couvrant un bloc rectangulaire intérieur. Les interfaces sont
//! les arêtes du bloc (planes, cf. plan §2.1).
//!
//! Table de voisinage inter-niveaux PRÉCALCULÉE à l'init (plan §2.1 : jamais
//! pendant la boucle temps — déterminisme) :
//! - `ghosts` : nœuds fantômes fins, stencil bilinéaire grossier (grossier→fin) ;
//! - `coincident` : nœuds grossiers coïncidents, stencil de restriction 9
//!   nœuds fins ¼-½-¼ ⊗ ¼-½-¼ centré sur le nœud grossier (fin→grossier,
//!   Lagrava et al. 2015 ; l'ancien 2×2 décalé à (cx+0.25, cy+0.25) est
//!   corrigé, biais +¼ cellule confirmé en V2).
//!
//! Convention de position : nœuds grossiers aux coordonnées entières (i, j),
//! espacement 1. Nœuds fins intérieurs à (bx0 + k/2, by0 + m/2), k = 0..2fw
//! (coïncidence exacte avec les nœuds grossiers du bloc aux positions paires).
//! Couches fantômes : 1 nœud fin de chaque côté, aux demi-coordonnées
//! (bx0 − 0.5, bx1 + 0.5, ...) — Stencil bilinéaire à poids 1/2 sur les deux
//! nœuds grossiers encadrants.
//!
//! Étape B (plan §2.1 : « parois toujours au niveau fin max ») : le bloc fin
//! PEUT toucher les bords du domaine (bx0 = 0, bx1 = nx0, by0 = 0, by1 = ny0).
//! Les couches fantômes alors HORS du domaine reçoivent un stencil
//! d'extrapolation linéaire clampé (poids > 1 / < 0, somme = 1) — les champs
//! de paroi étant lisses (T imposé, u = 0), le biais est négligeable.
//! Les nœuds grossiers coïncidents situés SUR le bord physique du domaine
//! sont exclus du transfert fin→grossier : leurs CL (isotherme/adiabatique/
//! no-slip) y restent exactes et ne sont jamais polluées par le moyennage.

/// Entrée de la table grossier→fin : un nœud fantôme fin + son stencil
/// bilinéaire de 4 nœuds grossiers (indice, poids).
pub struct GhostEntry {
    /// Index du nœud fantôme dans la grille fine (k = iy1 * nx1 + ix1).
    pub k_fine: usize,
    /// Stencil bilinéaire : 4 (index grossier, poids), somme des poids = 1.
    pub stencil: [(usize, f64); 4],
}

/// Entrée de la table fin→grossier : un nœud grossier coïncident + son
/// stencil de restriction 9 nœuds fins (séparable ¼-½-¼ ⊗ ¼-½-¼, centré sur
/// le nœud grossier — stencil Lagrava ; l'ancien 2×2 {x, x+0.5}² était
/// décalé de +¼ cellule vers (cx+0.25, cy+0.25), biais confirmé en V2).
pub struct CoarseEntry {
    /// Index du nœud grossier (k = iy0 * nx0 + ix0).
    pub k_coarse: usize,
    /// 9 (index fin, poids), somme des poids = 1, centré sur (cx, cy).
    pub stencil: [(usize, f64); 9],
}

/// Grille quadtree 2 niveaux avec tables de voisinage précalculées.
pub struct OctreeGrid {
    /// Dimensions du niveau grossier.
    pub nx0: usize,
    pub ny0: usize,
    /// Bloc fin en cellules grossières [bx0, bx1) × [by0, by1).
    pub bx0: usize,
    pub by0: usize,
    pub bx1: usize,
    pub by1: usize,
    /// Dimensions de la grille fine AVEC couches fantômes (1 de chaque côté) :
    /// nx1 = 2·(bx1−bx0) + 3.
    pub nx1: usize,
    pub ny1: usize,
    /// Nœuds fantômes fins (grossier→fin), précalculé.
    pub ghosts: Vec<GhostEntry>,
    /// Nœuds grossiers coïncidents (fin→grossier), précalculé.
    pub coincident: Vec<CoarseEntry>,
}

impl OctreeGrid {
    /// Construit la grille et PRÉCALCULE les tables de voisinage (plan §2.1).
    /// Le bloc fin peut toucher les bords du domaine (Étape B : BL de paroi
    /// raffinée, plan §2.1 « parois toujours au niveau fin max ») — les
    /// stencils hors domaine sont alors extrapolés (voir `bilinear_stencil`).
    pub fn new(nx0: usize, ny0: usize, bx0: usize, by0: usize, bx1: usize, by1: usize) -> Self {
        Self::build(nx0, ny0, bx0, by0, bx1, by1, false, false, false, false)
    }

    /// Variante interface imbriquée (V6, interface B dx/2 ⇄ dx/4) : le côté
    /// « grossier » est une grille DÉJÀ raffinée (bande L1) dont certains
    /// bords du bloc sont des parois PHYSIQUES (mur collé + haut/bas). Les
    /// nœuds grossiers de paroi (colonne a=1 du strip chaud, a=nx0−2 du
    /// strip froid, lignes b=1 et b=ny0−2) sont EXCLUS de la table
    /// coïncidente : leurs CL (isotherme/adiabatique/no-slip) restent
    /// exactes, jamais écrasées par le moyennage L2.
    pub fn new_nested(
        nx0: usize,
        ny0: usize,
        bx0: usize,
        by0: usize,
        bx1: usize,
        by1: usize,
        wall_left: bool,
        wall_right: bool,
        wall_bottom: bool,
        wall_top: bool,
    ) -> Self {
        Self::build(nx0, ny0, bx0, by0, bx1, by1, wall_left, wall_right, wall_bottom, wall_top)
    }

    /// Constructeur commun : tables précalculées + exclusions de paroi.
    /// `wall_*` : le bord correspondant du bloc est une paroi physique du
    /// côté grossier (ses nœuds sont exclus du transfert fin→grossier).
    fn build(
        nx0: usize,
        ny0: usize,
        bx0: usize,
        by0: usize,
        bx1: usize,
        by1: usize,
        wall_left: bool,
        wall_right: bool,
        wall_bottom: bool,
        wall_top: bool,
    ) -> Self {
        assert!(bx1 <= nx0 && by1 <= ny0, "bloc fin hors domaine grossier");
        assert!(bx0 < bx1 && by0 < by1, "bloc fin vide");
        let fw = bx1 - bx0;
        let fh = by1 - by0;
        // Intérieur fin : 2fw+1 nœuds par dimension (positions bx0 .. bx1 pas
        // 0.5, coïncidence exacte avec les nœuds grossiers aux pas pairs) ;
        // + 2 couches fantômes.
        let nx1 = 2 * fw + 3;
        let ny1 = 2 * fh + 3;

        // --- Table grossier→fin : tous les nœuds fantômes (bords de la grille
        // fine), positions en demi-coordonnées grossières.
        let mut ghosts = Vec::new();
        for b in 0..ny1 {
            for a in 0..nx1 {
                if a == 0 || b == 0 || a == nx1 - 1 || b == ny1 - 1 {
                    let x_c = bx0 as f64 - 0.5 + 0.5 * a as f64;
                    let y_c = by0 as f64 - 0.5 + 0.5 * b as f64;
                    let stencil = bilinear_stencil(x_c, y_c, nx0, ny0);
                    ghosts.push(GhostEntry {
                        k_fine: b * nx1 + a,
                        stencil,
                    });
                }
            }
        }

        // --- Table fin→grossier : nœuds grossiers coïncidents du bloc,
        // stencil de restriction 9 nœuds fins (séparable ¼-½-¼ ⊗ ¼-½-¼
        // centré sur le nœud grossier, Lagrava et al. 2015) :
        //   x ∈ {cx−0.5, cx, cx+0.5} ⊗ y ∈ {cy−0.5, cy, cy+0.5},
        // poids (¼,½,¼) par dimension — centré EXACTEMENT sur (cx, cy),
        // contrairement à l'ancien 2×2 {x, x+0.5}² centré à (cx+0.25,
        // cy+0.25) (biais +¼ cellule confirmé en V2). Exclusions :
        // 1. nœuds SUR le bord de la grille grossière (Étape B : pour
        //    l'interface A ce sont les parois du domaine ; pour l'interface
        //    B ce sont les couches fantômes de la grille L1) ;
        // 2. nœuds de PAROI physique du côté grossier (interface B, V6 :
        //    mur collé + haut/bas — CL exactes préservées).
        let mut coincident = Vec::new();
        for cy in by0..by1 {
            for cx in bx0..bx1 {
                if cx == 0 || cx + 1 == nx0 || cy == 0 || cy + 1 == ny0 {
                    continue;
                }
                if (wall_left && cx == bx0)
                    || (wall_right && cx + 1 == bx1)
                    || (wall_bottom && cy == by0)
                    || (wall_top && cy + 1 == by1)
                {
                    continue;
                }
                // Centre du stencil : nœud fin coïncident (x = cx, y = cy),
                // indices fins a = 2·(cx−bx0)+1, b = 2·(cy−by0)+1. Le stencil
                // s'étend sur les nœuds a−1..a+1 : pour un bloc collé au mur
                // c'est la couche fantôme d'interface (injectée par le
                // couplage §1.2 avant la restriction) ; sinon un nœud fin
                // intérieur.
                let a = 2 * (cx - bx0) + 1;
                let b = 2 * (cy - by0) + 1;
                let w = [0.25f64, 0.5, 0.25];
                let mut stencil = [(0usize, 0.0f64); 9];
                let mut idx = 0;
                for (jb, &wy) in w.iter().enumerate() {
                    for (ja, &wx) in w.iter().enumerate() {
                        let kf = (b + jb - 1) * nx1 + (a + ja - 1);
                        stencil[idx] = (kf, wx * wy);
                        idx += 1;
                    }
                }
                coincident.push(CoarseEntry {
                    k_coarse: cy * nx0 + cx,
                    stencil,
                });
            }
        }

        OctreeGrid {
            nx0,
            ny0,
            bx0,
            by0,
            bx1,
            by1,
            nx1,
            ny1,
            ghosts,
            coincident,
        }
    }

    pub fn n_ghosts(&self) -> usize {
        self.ghosts.len()
    }
}

/// Stencil bilinéaire de 4 nœuds grossiers encadrant la position (x_c, y_c)
/// en coordonnées grossières. Poids : (1−fx)(1−fy), fx(1−fy), (1−fx)fy, fx·fy.
/// La base (ix, iy) est CLAMPÉE au domaine : pour une position hors domaine
/// (couche fantôme d'un bloc accolé à une paroi, Étape B), fx/fy sortent de
/// [0, 1] et le stencil devient une EXTRAPOLATION linéaire one-sided
/// (poids 1.5/−0.5, somme = 1) — biais négligeable sur les champs de paroi
/// lisses (T imposé, u = 0).
pub fn bilinear_stencil(x_c: f64, y_c: f64, nx0: usize, ny0: usize) -> [(usize, f64); 4] {
    let ix = (x_c.floor() as i64).clamp(0, nx0 as i64 - 2) as usize;
    let iy = (y_c.floor() as i64).clamp(0, ny0 as i64 - 2) as usize;
    let fx = x_c - ix as f64;
    let fy = y_c - iy as f64;
    [
        (iy * nx0 + ix, (1.0 - fx) * (1.0 - fy)),
        (iy * nx0 + ix + 1, fx * (1.0 - fy)),
        ((iy + 1) * nx0 + ix, (1.0 - fx) * fy),
        ((iy + 1) * nx0 + ix + 1, fx * fy),
    ]
}
