//! Orchestrateur octree 2 niveaux (Étape A, plan §2 / §2.2).
//!
//! Boucle temporelle pas multiple (§2.2, acoustic scaling) : le niveau fin
//! fait 2 sous-pas (dt_fin = dt_grossier/2) par pas grossier :
//! 1. collide+stream niveau grossier (τ_grossier) ;
//! 2. 2× (collide+stream niveau fin, τ_fin) ;
//! 3. couplage interfaces : grossier→fin (moments bilinéaires + f^neq
//!    reconstruit + rescaling (1/2)·ξ, §1.2) puis fin→grossier (moyenne 2×2
//!    + 1/ξ).
//!
//! Étape B (plan §3.3, porte V2) : [`CavitySolver`] — cavité DHC 2D Ra=1e6
//! sur l'octree. Géométrie : niveau grossier nx0×ny0 + DEUX blocs fins
//! pleine hauteur collés aux 2 parois verticales (plan §2.1 : « parois
//! toujours au niveau fin max — jamais de bounce-back sur niveau grossier
//! aux parois chauffées »). La boucle `step_cavity` ajoute :
//! - flottabilité Guo par niveau : beta_eff_fin = beta_eff_grossier/2 (§1.3,
//!   u_scale identique sur les deux niveaux, acoustic scaling) ;
//! - CL thermiques copiées du crate uniforme (variante block-aware pour les
//!   blocs fins : la colonne de mur physique fine est en a=1 / a=nx1−2, les
//!   colonnes a=0 / a=nx1−1 étant des couches fantômes hors domaine) ;
//! - correction de vitesse Guo éq. (16) ;
//! - monitoring bruit acoustique max|ρ−ρ̄| global et aux interfaces (plan
//!   §3.1, claim V2).
//!
//! Étape F (V6, plan §3.3) : escalier MULTI-NIVEAUX suivant la loi tanh du
//! bord (§0.1) — chaque bloc L1 porte une bande L2 imbriquée [`NestedBlock`]
//! (dx/4, 1 cellule L0 = 2 cellules L1 de large, pleine hauteur, collée à la
//! paroi ; frontière tanh x ≈ 0.020 → nœud 1). Sous-pas télescopiques : 2
//! sous-pas L1 par pas L0, 2 sous-pas L2 par sous-pas L1 (dt ∝ Δx, acoustic
//! scaling) ; rescaling §1.2 généralisé par paire de niveaux adjacents
//! ([`rescaling_factors_multilevel`]). `--levels 2` restaure la configuration
//! V2 (bandes L2 désactivées).
//!
//! Choix documenté (plan §2.2, « pas de temps unique (fallback) ») : les
//! tests V1 unitaires n'exécutent pas la boucle complète — ils testent les
//! transferts d'interface isolés (pas unique, pas de subcycling) ; la
//! structure 2 sous-pas est néanmoins en place ici pour l'Étape B.

use crate::cavity::{
    apply_thermal_boundary_conditions, apply_thermal_boundary_conditions_block,
    check_finite_levels, collide_cavity, enforce_wall_velocity, enforce_wall_velocity_block,
    max_u, stream_cavity, stream_cavity_block, update_macroscopic_guo, C_SCALE, MAX_U_ABORT,
    UTARGET,
};
use crate::cavity::{NoiseStats, Rng, T_COLD, T_HOT};
use crate::grid::{bilinear_stencil, OctreeGrid};
use crate::interface::{
    coarse_to_fine, fine_to_coarse, rescaling_factors, rescaling_factors_multilevel,
    RescalingFactors,
};
use crate::lbm::{LbmLevel, RelaxTimes};
use oxiphysics_lbm::lattice::D2Q9_VELOCITIES;
use oxiphysics_lbm::thermal_lbm::{feq, geq};

/// Solveur octree 2 niveaux (Étape A, porte V1 — inchangé).
pub struct OctreeSolver {
    pub grid: OctreeGrid,
    pub coarse: LbmLevel,
    pub fine: LbmLevel,
    /// τ des niveaux (constants en Étape A : ν_lu,fin = 2·ν_lu,grossier, §1.3).
    pub tau_coarse: RelaxTimes,
    pub tau_fine: RelaxTimes,
    /// Facteurs de rescaling d'interface (§1.2), précalculés à l'init.
    pub xi: RescalingFactors,
}

impl OctreeSolver {
    /// Construit le solveur : grilles, τ par niveau (ν_lu,fin = 2·ν_lu,grossier,
    /// §1.3), facteurs de rescaling précalculés (variante impaire exacte
    /// désactivable — plan §1.2 : approximation ξ_odd = 1/2 par défaut).
    pub fn new(grid: OctreeGrid, nu_lu_coarse: f64, alpha_lu_coarse: f64, odd_exact: bool) -> Self {
        // §1.3 : à ν physique constant, ν_lu double sur le niveau fin
        // (dt_fin = dt_grossier/2, Δx_fin = Δx/2 ⇒ ν_lu,fin = 2·ν_lu,grossier).
        let tau_coarse = RelaxTimes::from_nu_alpha(nu_lu_coarse, alpha_lu_coarse);
        let tau_fine = RelaxTimes::from_nu_alpha(2.0 * nu_lu_coarse, 2.0 * alpha_lu_coarse);
        let xi = rescaling_factors(&tau_coarse, &tau_fine, odd_exact);
        let coarse = LbmLevel::new(grid.nx0, grid.ny0);
        let fine = LbmLevel::new(grid.nx1, grid.ny1);
        OctreeSolver {
            grid,
            coarse,
            fine,
            tau_coarse,
            tau_fine,
            xi,
        }
    }

    /// Un pas grossier complet (§2.2) : grossier 1 pas, fin 2 sous-pas,
    /// puis couplage des interfaces.
    pub fn step(&mut self) {
        // 1. Niveau grossier : collide + stream.
        crate::lbm::collide(&mut self.coarse, &self.tau_coarse);
        crate::lbm::stream(&mut self.coarse);
        self.coarse.update_macroscopic();

        // 2. Niveau fin : 2 sous-pas (dt_fin = dt_grossier/2).
        for _ in 0..2 {
            crate::lbm::collide(&mut self.fine, &self.tau_fine);
            crate::lbm::stream(&mut self.fine);
            self.fine.update_macroscopic();
        }

        // 3. Couplage interfaces (§1.2) : grossier→fin puis fin→grossier.
        coarse_to_fine(&self.grid, &self.coarse, &mut self.fine, &self.xi);
        fine_to_coarse(&self.grid, &self.fine, &mut self.coarse, &self.xi);
    }
}

// ---------------------------------------------------------------------------
// Étape B : cavité DHC 2D sur octree (porte V2, plan §3.1 / §3.3)
// ---------------------------------------------------------------------------

/// Un bloc fin de cavité : grille quadtree + niveau fin + force Guo stockée.
/// `hot_wall`/`cold_wall` indiquent si le bloc touche les parois isothermes
/// physique (bloc pleine hauteur collé à la paroi : colonne fine a=1 pour la
/// paroi chaude, a=nx1−2 pour la froide — a=0/a=nx1−1 sont des fantômes).
pub struct CavityBlock {
    pub grid: OctreeGrid,
    pub fine: LbmLevel,
    /// Force Guo par nœud (dernière collision) pour la correction éq. (16).
    fy: Vec<f64>,
    pub hot_wall: bool,
    pub cold_wall: bool,
    /// Bande L2 imbriquée (V6, `--levels 3`) : None en config V2
    /// (`--levels 2`).
    pub sub: Option<NestedBlock>,
}

/// Bande L2 imbriquée (V6, escalier tanh §0.1) : interface B dx/2 ⇄ dx/4.
/// Le côté « grossier » de `grid` est la GRILLE L1 du bloc parent (nœuds
/// grossiers = nœuds fins L1, espacement dx_base/2) ; le côté fin est le
/// niveau L2 (dx_base/4). Bande de 1 cellule L0 de large (2 cellules L1),
/// pleine hauteur, collée à la paroi verticale du bloc parent
/// (frontière tanh x ≈ 0.020 → nœud 1 pour nx0 = 64).
pub struct NestedBlock {
    /// Grille d'interface B : côté grossier = grille L1 parent, côté fin = L2.
    /// Exclusions coïncidents (V6) : colonne de bord gauche `wall_left=true`
    /// (chaud : mur physique a=1 ; froid : bord d'interface a=n1−4 — même
    /// régime symétrique, un seul nœud L1 médian restreint), mur bas
    /// `wall_bottom=true` (b=1) ; le mur froid a=n1−2 et le mur haut
    /// b=ny1−2 = bord du bloc sont hors plage, auto-exclus.
    pub grid: OctreeGrid,
    /// Champ L2 (dx/4, avec couches fantômes).
    pub lvl: LbmLevel,
    /// τ du niveau L2 (ν_lu = 4·ν_L0, §1.3).
    pub tau: RelaxTimes,
    /// Facteurs de rescaling L1→L2 (grossier→fin, §1.2 généralisé : facteur
    /// temporel 2^-1 car dt_L2 = dt_L1/2, acoustic scaling).
    pub xi_down: RescalingFactors,
    /// Force Guo stockée par nœud (correction éq. (16)).
    fy: Vec<f64>,
    /// Côté du bloc parent (paroi chaude / froide) : CL thermiques et flags
    /// de streaming du niveau L2.
    pub hot_wall: bool,
    pub cold_wall: bool,
}

/// Solveur cavité DHC 2D sur octree (Étape B, porte V2, plan §3.3) étendu
/// V6 (Étape F) à l'escalier tanh 3 niveaux via [`CavityBlock::sub`].
///
/// Niveaux : L0 = noyau grossier `coarse` (domaine complet, dx_base) ;
/// L1 = `blocks[i].fine` (dx_base/2, bande collée à la paroi) ;
/// L2 = `blocks[i].sub` (dx_base/4, imbriqué dans L1, `--levels 3`).
pub struct CavitySolver {
    /// Dimensions de la grille de base (niveau L0, domaine complet).
    pub nx0: usize,
    pub ny0: usize,
    /// Niveau grossier L0 (domaine complet).
    pub coarse: LbmLevel,
    /// Blocs fins L1 collés aux parois verticales (dx_base/2).
    pub blocks: Vec<CavityBlock>,
    /// τ du niveau grossier L0.
    pub tau_coarse: RelaxTimes,
    /// τ du niveau L1 (ν_lu = 2·ν_L0, §1.3).
    pub tau_fine: RelaxTimes,
    /// Facteurs de rescaling L0→L1 (§1.2, facteur temporel 1/2).
    pub xi: RescalingFactors,
    /// Nombre de niveaux (2 = config V2, 3 = escalier V6).
    pub levels: usize,
    /// ν et α en unités lattice du niveau grossier (calage crate uniforme).
    pub nu_lu: f64,
    pub alpha_lu: f64,
    /// Flottabilité en unités lattice du niveau L0 (§1.3).
    pub beta_eff_coarse: f64,
    /// Flottabilité du niveau L1 (beta_eff_coarse/2, §1.3 acoustic scaling).
    pub beta_eff_fine: f64,
    /// Force Guo stockée par nœud L0 (correction éq. (16)).
    fy_coarse: Vec<f64>,
    /// Threads de calcul (1 = mono-thread, N > 1 = rayon).
    pub jobs: usize,
    /// PRNG perturbation initiale (xorshift64*, copie crate uniforme).
    rng: Rng,
}

impl CavitySolver {
    /// Construit le solveur cavité et initialise le champ (T linéaire en x +
    /// perturbation ±1e-3, copie crate uniforme solver.rs l.159-176).
    ///
    /// Calage ν (copie du crate uniforme, solver.rs l.110-143) :
    /// ν = (utarget·√Pr·Lz/(c·√Ra))^(2/3), c = 1.7, plancher ν ≥ 0.002
    /// (τ⁺ ≥ 0.506, tenable en TRT Λ = 3/16) ; α = ν/Pr ;
    /// beta_eff_coarse = Ra·ν·α/Lz³ avec Lz = ny0 − 1 (unités grossières).
    ///
    /// `levels` : 2 = configuration V2 (bandes L2 désactivées) ;
    /// 3 = escalier tanh V6 (bande L2 dx/4 imbriquée, plan §3.3 Étape F).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nx0: usize,
        ny0: usize,
        bl_width: usize,
        ra: f64,
        pr: f64,
        utarget: f64,
        seed: u64,
        jobs: usize,
        odd_exact: bool,
        levels: usize,
    ) -> Self {
        assert!(bl_width >= 2, "largeur de BL fine >= 2 cellules grossieres requise");
        assert!(2 * bl_width < nx0, "les 2 blocs fins doivent laisser un noyau grossier");
        assert!(
            levels == 2 || levels == 3,
            "levels doit valoir 2 (config V2) ou 3 (escalier tanh V6)"
        );

        // --- Calage ν/α (crate uniforme, Lz en unités grossières) ---
        let _ = UTARGET; // documenté : utarget est le paramètre de calage
        let lz = (ny0 - 1) as f64;
        let nu = ((utarget * pr.sqrt() * lz / (C_SCALE * ra.sqrt())).powf(2.0 / 3.0)).max(0.002);
        let alpha = nu / pr;
        let beta_eff_coarse = ra * nu * alpha / lz.powi(3);

        // --- τ par niveau (§1.3 : ν_lu,n = 2^(n−1)·ν_L0, Δx_n = dx_base/2^(n−1))
        // L1 = 2ν, L2 = 4ν. Garde-fou τ⁺ (plan §1.3 : plancher tenable TRT).
        let tau_coarse = RelaxTimes::from_nu_alpha(nu, alpha);
        let tau_fine = RelaxTimes::from_nu_alpha(2.0 * nu, 2.0 * alpha);
        let tau_sub = RelaxTimes::from_nu_alpha(4.0 * nu, 4.0 * alpha);
        let tau_plus_min = [tau_coarse.tau_plus, tau_fine.tau_plus, tau_sub.tau_plus];
        if let Some(tp) = tau_plus_min.iter().copied().find(|&t| t <= 0.506) {
            eprintln!(
                "WARNING : tau+ = {tp:.5} <= 0.506 au niveau le plus fin (plancher §1.3, risque d'instabilite TRT)"
            );
        }

        // --- Facteurs de rescaling §1.2 généralisés (V6, par paire adjacente)
        // : facteur temporel 1/substeps (acoustic scaling, dt ∝ Δx) —
        // interface A : 2 sous-pas L1 par pas L0 ; interface B : 2 sous-pas
        // L2 par sous-pas L1 (4 par pas L0). ---
        let xi = rescaling_factors_multilevel(&tau_coarse, &tau_fine, 2, odd_exact);
        let xi_sub = rescaling_factors_multilevel(&tau_fine, &tau_sub, 2, odd_exact);

        // --- Grilles : grossier domaine complet + 2 blocs fins pleine
        // hauteur collés aux parois verticales (plan §2.1) ---
        // Bloc chaud [0, bl_width) : nœuds fins intérieurs x ∈ [0, bl_width−0.5],
        // mur x=0 = colonne fine a=1 exactement, fantômes à −0.5/+0.5.
        // Bloc froid [nx0−1−bl_width, nx0−1) : nœuds fins intérieurs
        // x ∈ [nx0−1−bl_width, nx0−1], mur x=nx0−1 = colonne fine a=nx1−2
        // exactement. (PAS [nx0−bl_width, nx0) : le dernier nœud fin serait
        // à x=nx0, HORS du domaine.)
        // Verticalement : bloc [0, ny0−1] — nœuds fins intérieurs y ∈
        // [0, ny0−1], murs y=0 (b=1) et y=ny0−1 (b=ny1−2) EXACTS, fantômes
        // b=0/b=ny1−1 à y=−0.5/ny0−0.5 hors domaine (extrapolés, jamais
        // utilisés : bounce-back). (PAS [0, ny0) : les nœuds fins seraient
        // à y=ny0−0.5 et ny0, HORS du domaine — la cavité fine serait une
        // cellule plus haute que la cavité grossière et le no-slip/
        // adiabatique serait imposé au-dessus du mur physique.)
        let coarse = LbmLevel::new(nx0, ny0);
        let mut blocks = Vec::with_capacity(2);
        for &(bx0, bx1, hot, cold) in &[
            (0usize, bl_width, true, false),            // paroi chaude (x=0)
            (nx0 - 1 - bl_width, nx0 - 1, false, true), // paroi froide (x=nx0-1)
        ] {
            let grid = OctreeGrid::new(nx0, ny0, bx0, 0, bx1, ny0 - 1);
            let fine = LbmLevel::new(grid.nx1, grid.ny1);
            let n1 = grid.nx1 * grid.ny1;

            // --- Bande L2 imbriquée (V6, `--levels 3`) ---
            // Côté grossier = GRILLE L1 du bloc (nœuds grossiers = nœuds fins
            // L1, espacement dx_base/2). Bande de 1 cellule L0 = 2 cellules L1
            // de large, pleine hauteur, collée à la paroi verticale du bloc
            // parent (frontière tanh x ≈ 0.020, §0.1) :
            // - chaud : cellules [1, 3) — bord gauche a=1 = mur physique L1
            //   (exclu des coïncidents via wall_left), bord droit a=3 = bord
            //   du bloc (auto-exclu car hors [bx0,bx1) de la grille L1) ;
            // - froid : cellules [n1−4, n1−2) — bord droit a=n1−2 = mur
            //   physique L1 (hors plage, auto-exclu), bord gauche a=n1−4 =
            //   bord du bloc (exclu via wall_left, régime symétrique du
            //   chaud : un seul nœud L1 médian restreint) ;
            // - vertical : cellules [1, ny1−2) — b=1 = mur bas physique L1
            //   (exclu via wall_bottom), b=ny1−2 = mur haut (bord du bloc,
            //   auto-exclu).
            let sub = if levels >= 3 {
                let (sbx0, sbx1) = if hot {
                    (1usize, 3usize)
                } else {
                    (grid.nx1 - 4, grid.nx1 - 2)
                };
                // wall_left pour les DEUX strips : côté mur (chaud) ET côté
                // interface (froid) excluent exactement 1 colonne de nœuds L1
                // (mur physique ou bord d'interface) de la table coïncidente.
                let sub_grid = OctreeGrid::new_nested(
                    grid.nx1, grid.ny1, sbx0, 1, sbx1, grid.ny1 - 2,
                    true, false, true, false,
                );
                let (sn1, sny1) = (sub_grid.nx1, sub_grid.ny1);
                let sub_lvl = LbmLevel::new(sn1, sny1);
                Some(NestedBlock {
                    grid: sub_grid,
                    lvl: sub_lvl,
                    tau: tau_sub,
                    xi_down: xi_sub,
                    fy: vec![0.0; sn1 * sny1],
                    hot_wall: hot,
                    cold_wall: cold,
                })
            } else {
                None
            };

            blocks.push(CavityBlock {
                grid,
                fine,
                fy: vec![0.0; n1],
                hot_wall: hot,
                cold_wall: cold,
                sub,
            });
        }

        let mut solver = CavitySolver {
            nx0,
            ny0,
            coarse,
            blocks,
            tau_coarse,
            tau_fine,
            xi,
            levels,
            nu_lu: nu,
            alpha_lu: alpha,
            beta_eff_coarse,
            beta_eff_fine: 0.5 * beta_eff_coarse,
            fy_coarse: vec![0.0; nx0 * ny0],
            jobs: jobs.max(1),
            rng: Rng::new(seed),
        };
        solver.init_cavity();
        solver
    }

    /// Nombre total de nœuds (grossier + blocs fins + bandes L2, avec
    /// fantômes).
    pub fn n_nodes_total(&self) -> usize {
        self.nx0 * self.ny0
            + self
                .blocks
                .iter()
                .map(|b| {
                    b.grid.nx1 * b.grid.ny1
                        + b.sub.as_ref().map_or(0, |s| s.grid.nx1 * s.grid.ny1)
                })
                .sum::<usize>()
    }

    /// Initialise la cavité — copie de l'initialisation du crate uniforme
    /// (solver.rs l.159-176) : T linéaire en x (T_HOT gauche → T_COLD
    /// droite), ρ = 1, u = 0, perturbation ±1e-3 sur T (PRNG seed 42 par
    /// défaut). Distributions = équilibres exacts.
    /// Les blocs fins sont initialisés par interpolation bilinéaire du champ
    /// grossier (T linéaire exact aux positions fines + perturbation
    /// interpolée) — cohérence inter-niveaux au pas 0.
    fn init_cavity(&mut self) {
        let nx0 = self.nx0;
        let ny0 = self.ny0;
        for iy in 0..ny0 {
            for ix in 0..nx0 {
                let k = iy * nx0 + ix;
                let xf = ix as f64 / (nx0 - 1) as f64;
                let t_lin = T_HOT + (T_COLD - T_HOT) * xf;
                let pert = 1.0e-3 * (2.0 * self.rng.next_f64() - 1.0);
                let t = t_lin + pert;
                self.coarse.rho[k] = 1.0;
                self.coarse.ux[k] = 0.0;
                self.coarse.uy[k] = 0.0;
                self.coarse.temp[k] = t;
                for i in 0..9 {
                    self.coarse.f[k][i] = feq(i, 1.0, 0.0, 0.0);
                    self.coarse.g[k][i] = geq(i, t, 0.0, 0.0);
                }
            }
        }
        // Blocs fins : bilinéaire du champ grossier (le stencil clampé gère
        // les couches fantômes hors domaine, écrasées au 1ᵉʳ couplage).
        // Bandes L2 : bilinéaire du champ grossier aux positions physiques
        // exactes — un nœud L2 d'index (a2, b2) dans la grille L2 est à la
        // position L1 (sbx0 − 0.5 + 0.5·a2, sby0 − 0.5 + 0.5·b2), elle-même
        // à la position physique grossière (bx0 − 0.5 + 0.5·a1, ...).
        for block in &mut self.blocks {
            let nx1 = block.grid.nx1;
            for b in 0..block.grid.ny1 {
                for a in 0..nx1 {
                    let k = b * nx1 + a;
                    let x_c = block.grid.bx0 as f64 - 0.5 + 0.5 * a as f64;
                    let y_c = block.grid.by0 as f64 - 0.5 + 0.5 * b as f64;
                    let stencil = bilinear_stencil(x_c, y_c, nx0, ny0);
                    let mut t = 0.0;
                    for &(k_c, w) in &stencil {
                        t += w * self.coarse.temp[k_c];
                    }
                    block.fine.rho[k] = 1.0;
                    block.fine.ux[k] = 0.0;
                    block.fine.uy[k] = 0.0;
                    block.fine.temp[k] = t;
                    for i in 0..9 {
                        block.fine.f[k][i] = feq(i, 1.0, 0.0, 0.0);
                        block.fine.g[k][i] = geq(i, t, 0.0, 0.0);
                    }
                }
            }
            if let Some(sub) = &mut block.sub {
                let nx2 = sub.grid.nx1;
                for b2 in 0..sub.grid.ny1 {
                    for a2 in 0..nx2 {
                        let k = b2 * nx2 + a2;
                        let x_c = block.grid.bx0 as f64 - 0.5
                            + 0.5 * (sub.grid.bx0 as f64 - 0.5 + 0.5 * a2 as f64);
                        let y_c = block.grid.by0 as f64 - 0.5
                            + 0.5 * (sub.grid.by0 as f64 - 0.5 + 0.5 * b2 as f64);
                        let stencil = bilinear_stencil(x_c, y_c, nx0, ny0);
                        let mut t = 0.0;
                        for &(k_c, w) in &stencil {
                            t += w * self.coarse.temp[k_c];
                        }
                        sub.lvl.rho[k] = 1.0;
                        sub.lvl.ux[k] = 0.0;
                        sub.lvl.uy[k] = 0.0;
                        sub.lvl.temp[k] = t;
                        for i in 0..9 {
                            sub.lvl.f[k][i] = feq(i, 1.0, 0.0, 0.0);
                            sub.lvl.g[k][i] = geq(i, t, 0.0, 0.0);
                        }
                    }
                }
            }
        }
    }

    /// Un pas grossier complet de cavité (Étape B, plan §2.2 + §3.3) :
    /// 1. grossier : collide (force Guo, domaine complet) → stream → CL
    ///    thermiques → update_macroscopic_guo → no-slip ;
    /// 2. couplage grossier→fin (§1.2) : injection FRAÎCHE des fantômes
    ///    d'interface AVANT les sous-pas fins (Lagrava et al. 2015) ;
    /// 3. par bloc fin, 2 sous-pas (dt_fin = dt_grossier/2) : collide
    ///    interior_only → stream block-aware avec bounce-back EXACT sur les
    ///    parois physiques (haut/bas + mur collé) → CL thermiques block-aware
    ///    → update_macroscopic_guo → no-slip block-aware ;
    /// 4. couplage fin→grossier (§1.2) : moyenne 2×2 des coïncidents ;
    /// 5. moments des nœuds grossiers coïncidents recalculés depuis les
    ///    distributions injectées (la collision du pas suivant lit ρ, u, T).
    pub fn step_cavity(&mut self) {
        // --- 1. Niveau L0 (force Guo, CL crate uniforme) ---
        let beta_c = self.beta_eff_coarse;
        let jobs = self.jobs;
        collide_cavity(
            &mut self.coarse,
            &self.tau_coarse,
            beta_c,
            &mut self.fy_coarse,
            false,
            jobs,
        );
        stream_cavity(&mut self.coarse, jobs);
        apply_thermal_boundary_conditions(&mut self.coarse);
        update_macroscopic_guo(&mut self.coarse, &self.fy_coarse, jobs);
        enforce_wall_velocity(&mut self.coarse);

        // --- 2. Couplage A L0→L1 AVANT les sous-pas fins (§1.2, Lagrava
        // et al. 2015 : injection FRAÎCHE de l'état post-couplage grossier,
        // pas de retard d'un pas complet) ---
        let xi = self.xi;
        let coarse = &self.coarse;
        for block in &mut self.blocks {
            coarse_to_fine(&block.grid, coarse, &mut block.fine, &xi);
        }

        // --- 3. Blocs : télescopage L1 (2 sous-pas, dt_L1 = dt_L0/2) + bande
        // L2 imbriquée (V6, `--levels 3`) : L1 sp1 → couplage B frais →
        // 2 sous-pas L2 (dt_L2 = dt_L1/2) → restriction L2→L1 → L1 sp2 →
        // restriction L1→L0 ---
        let tau_fine = self.tau_fine;
        let beta_f = self.beta_eff_fine;
        let beta_sub = 0.25 * self.beta_eff_coarse; // §1.3 : /2 par niveau
        for bi in 0..self.blocks.len() {
            let block = &mut self.blocks[bi];
            let (wl, wr) = (block.hot_wall, block.cold_wall);

            // --- L1 sous-pas 1 : collide interior_only → stream block-aware
            // avec bounce-back EXACT sur les parois physiques (haut/bas +
            // mur collé) → CL thermiques block-aware → update → no-slip ---
            collide_cavity(&mut block.fine, &tau_fine, beta_f, &mut block.fy, true, jobs);
            stream_cavity_block(&mut block.fine, jobs, wl, wr, true, true);
            apply_thermal_boundary_conditions_block(&mut block.fine, wl, wr);
            update_macroscopic_guo(&mut block.fine, &block.fy, jobs);
            enforce_wall_velocity_block(&mut block.fine, wl, wr);

            // --- Bande L2 imbriquée (V6) ---
            if let Some(sub) = &mut block.sub {
                // Couplage B frais L1→L2 : injection des fantômes L2 depuis
                // l'état L1 post-sous-pas 1 (Lagrava, injection fraîche).
                let xi_down = sub.xi_down;
                coarse_to_fine(&sub.grid, &block.fine, &mut sub.lvl, &xi_down);
                let tau_sub = sub.tau;
                for _ in 0..2 {
                    collide_cavity(&mut sub.lvl, &tau_sub, beta_sub, &mut sub.fy, true, jobs);
                    stream_cavity_block(&mut sub.lvl, jobs, wl, wr, true, true);
                    apply_thermal_boundary_conditions_block(&mut sub.lvl, wl, wr);
                    update_macroscopic_guo(&mut sub.lvl, &sub.fy, jobs);
                    enforce_wall_velocity_block(&mut sub.lvl, wl, wr);
                }
                // Restriction L2→L1 (stencil 9 centré, 1/ξ_B §1.2) puis
                // moments des nœuds L1 coïncidents recalculés (la collision
                // du sous-pas 2 lit ρ, u, T).
                fine_to_coarse(&sub.grid, &sub.lvl, &mut block.fine, &xi_down);
                let idxs: Vec<usize> =
                    sub.grid.coincident.iter().map(|c| c.k_coarse).collect();
                refresh_moments_nodes(&mut block.fine, &idxs);
            }

            // --- L1 sous-pas 2 ---
            collide_cavity(&mut block.fine, &tau_fine, beta_f, &mut block.fy, true, jobs);
            stream_cavity_block(&mut block.fine, jobs, wl, wr, true, true);
            apply_thermal_boundary_conditions_block(&mut block.fine, wl, wr);
            update_macroscopic_guo(&mut block.fine, &block.fy, jobs);
            enforce_wall_velocity_block(&mut block.fine, wl, wr);

            // --- Restriction L1→L0 (stencil 9 centré, 1/ξ_A §1.2) ---
            fine_to_coarse(&block.grid, &block.fine, &mut self.coarse, &xi);
        }

        // --- 4. Moments coïncidents L0 recalculés (post-injection) ---
        let idxs: Vec<usize> = self
            .blocks
            .iter()
            .flat_map(|b| b.grid.coincident.iter().map(|c| c.k_coarse))
            .collect();
        let coarse = &mut self.coarse;
        refresh_moments_nodes(coarse, &idxs);
    }

    /// Monitoring bruit acoustique d'un pas (LA porte V2, plan §3.1) :
    /// max|ρ−ρ̄| global (tous les nœuds des niveaux, ρ̄ = moyenne
    /// grossière) et aux interfaces (nœuds fantômes fins injectés + nœuds
    /// grossiers coïncidents). Ajout strictement par nœud ⇒ déterministe
    /// quel que soit --jobs.
    pub fn noise_stats(&self) -> NoiseStats {
        let n0 = self.nx0 * self.ny0;
        let rho_bar: f64 = self.coarse.rho.iter().sum::<f64>() / n0 as f64;

        // Bruit global : tous les nœuds des niveaux.
        let mut max_dev = 0.0f64;
        for &v in &self.coarse.rho {
            max_dev = max_dev.max((v - rho_bar).abs());
        }
        for block in &self.blocks {
            for &v in &block.fine.rho {
                max_dev = max_dev.max((v - rho_bar).abs());
            }
            if let Some(sub) = &block.sub {
                for &v in &sub.lvl.rho {
                    max_dev = max_dev.max((v - rho_bar).abs());
                }
            }
        }

        // Bruit aux interfaces : fantômes fins (injectés §1.2) + coïncidents.
        let mut max_dev_iface = 0.0f64;
        for block in &self.blocks {
            for gh in &block.grid.ghosts {
                let v = block.fine.rho[gh.k_fine];
                max_dev_iface = max_dev_iface.max((v - rho_bar).abs());
            }
            for c in &block.grid.coincident {
                let v = self.coarse.rho[c.k_coarse];
                max_dev_iface = max_dev_iface.max((v - rho_bar).abs());
            }
            if let Some(sub) = &block.sub {
                for gh in &sub.grid.ghosts {
                    let v = sub.lvl.rho[gh.k_fine];
                    max_dev_iface = max_dev_iface.max((v - rho_bar).abs());
                }
                for c in &sub.grid.coincident {
                    let v = block.fine.rho[c.k_coarse];
                    max_dev_iface = max_dev_iface.max((v - rho_bar).abs());
                }
            }
        }

        NoiseStats {
            max_rho_dev: max_dev,
            max_rho_dev_ifaces: max_dev_iface,
            masse: rho_bar,
        }
    }

    /// Vitesse max tous niveaux (diagnostic de stabilité).
    pub fn max_u_all(&self) -> f64 {
        let mut m = max_u(&self.coarse);
        for block in &self.blocks {
            m = m.max(max_u(&block.fine));
            if let Some(sub) = &block.sub {
                m = m.max(max_u(&sub.lvl));
            }
        }
        m
    }

    /// Garde-fous (identiques crate uniforme) : max_u vs MAX_U_ABORT
    /// (Ma_local = max_u·√3) et champs finis. Retourne un message d'erreur,
    /// ou None.
    pub fn check_guards(&self) -> Option<String> {
        let u_max = self.max_u_all();
        if u_max > MAX_U_ABORT {
            let ma_local = u_max * 3.0f64.sqrt();
            return Some(format!(
                "AVORT : max_u = {u_max:.5} dépasse le garde-fou {MAX_U_ABORT:.2} (Ma_local = max_u·√3 ≈ {ma_local:.3})"
            ));
        }
        let mut lvls = vec![&self.coarse];
        for block in &self.blocks {
            lvls.push(&block.fine);
            if let Some(sub) = &block.sub {
                lvls.push(&sub.lvl);
            }
        }
        if !check_finite_levels(&lvls) {
            return Some("DIVERGENCE détectée (valeurs non finies).".to_string());
        }
        None
    }
}

/// Recalcule ρ, u, T d'un sous-ensemble de nœuds depuis les distributions
/// (post-injection d'un transfert fin→grossier — la collision du pas suivant
/// lit ρ, u, T des moments, pas les distributions).
fn refresh_moments_nodes(lvl: &mut LbmLevel, nodes: &[usize]) {
    for &k in nodes {
        let mut rho = 0.0;
        let mut mx = 0.0;
        let mut my = 0.0;
        let mut t = 0.0;
        for i in 0..9 {
            let fi = lvl.f[k][i];
            rho += fi;
            mx += D2Q9_VELOCITIES[i][0] as f64 * fi;
            my += D2Q9_VELOCITIES[i][1] as f64 * fi;
            t += lvl.g[k][i];
        }
        lvl.rho[k] = rho;
        lvl.temp[k] = t;
        if rho > f64::EPSILON {
            lvl.ux[k] = mx / rho;
            lvl.uy[k] = my / rho;
        } else {
            lvl.ux[k] = 0.0;
            lvl.uy[k] = 0.0;
        }
    }
}
