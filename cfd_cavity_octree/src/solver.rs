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
use crate::interface::{coarse_to_fine, fine_to_coarse, rescaling_factors, RescalingFactors};
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
/// physique (blo pleine hauteur collé à la paroi : colonne fine a=1 pour la
/// paroi chaude, a=nx1−2 pour la froide — a=0/a=nx1−1 sont des fantômes).
pub struct CavityBlock {
    pub grid: OctreeGrid,
    pub fine: LbmLevel,
    /// Force Guo par nœud (dernière collision) pour la correction éq. (16).
    fy: Vec<f64>,
    pub hot_wall: bool,
    pub cold_wall: bool,
}

/// Solveur cavité DHC 2D sur octree 2 niveaux (Étape B, porte V2).
pub struct CavitySolver {
    /// Dimensions du niveau grossier.
    pub nx0: usize,
    pub ny0: usize,
    /// Niveau grossier (domaine complet).
    pub coarse: LbmLevel,
    /// Blocs fins (2 : paroi chaude et paroi froide, pleine hauteur).
    pub blocks: Vec<CavityBlock>,
    /// τ des niveaux (ν_lu,fin = 2·ν_lu,grossier, §1.3).
    pub tau_coarse: RelaxTimes,
    pub tau_fine: RelaxTimes,
    /// Facteurs de rescaling d'interface (§1.2) — identiques pour les deux
    /// blocs (les τ sont les mêmes).
    pub xi: RescalingFactors,
    /// ν et α en unités lattice du niveau grossier (calage crate uniforme).
    pub nu_lu: f64,
    pub alpha_lu: f64,
    /// Flottabilité en unités lattice par niveau (§1.3) :
    /// beta_eff_niveau = Ra·ν_lu·α_lu/Lz_lu³. Le niveau fin a Lz_lu,fin =
    /// 2·Lz_grossier (même hauteur physique, Δx/2) et ν_lu,fin = 2·ν_lu,
    /// donc beta_eff_fin = Ra·(2ν)(2α)/(2Lz)³ = beta_eff_grossier/2 —
    /// u_scale = √(beta_eff·Lz_lu) identique sur les deux niveaux.
    pub beta_eff_coarse: f64,
    pub beta_eff_fine: f64,
    /// Force Guo stockée par nœud côté grossier (correction éq. (16)).
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
    ) -> Self {
        assert!(bl_width >= 2, "largeur de BL fine >= 2 cellules grossieres requise");
        assert!(2 * bl_width < nx0, "les 2 blocs fins doivent laisser un noyau grossier");

        // --- Calage ν/α (crate uniforme, Lz en unités grossières) ---
        let _ = UTARGET; // documenté : utarget est le paramètre de calage
        let lz = (ny0 - 1) as f64;
        let nu = ((utarget * pr.sqrt() * lz / (C_SCALE * ra.sqrt())).powf(2.0 / 3.0)).max(0.002);
        assert!(0.5 + 3.0 * nu > 0.505, "tau+ = {} trop proche de 0.5", 0.5 + 3.0 * nu);
        let alpha = nu / pr;
        let beta_eff_coarse = ra * nu * alpha / lz.powi(3);

        // --- τ par niveau (§1.3 : ν_lu,fin = 2·ν_lu,grossier) ---
        let tau_coarse = RelaxTimes::from_nu_alpha(nu, alpha);
        let tau_fine = RelaxTimes::from_nu_alpha(2.0 * nu, 2.0 * alpha);
        let xi = rescaling_factors(&tau_coarse, &tau_fine, odd_exact);

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
            (0usize, bl_width, true, false),                      // paroi chaude (x=0)
            (nx0 - 1 - bl_width, nx0 - 1, false, true),           // paroi froide (x=nx0-1)
        ] {
            let grid = OctreeGrid::new(nx0, ny0, bx0, 0, bx1, ny0 - 1);
            let fine = LbmLevel::new(grid.nx1, grid.ny1);
            let n1 = grid.nx1 * grid.ny1;
            blocks.push(CavityBlock {
                grid,
                fine,
                fy: vec![0.0; n1],
                hot_wall: hot,
                cold_wall: cold,
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

    /// Nombre total de nœuds (grossier + blocs fins, avec fantômes).
    pub fn n_nodes_total(&self) -> usize {
        self.nx0 * self.ny0
            + self.blocks.iter().map(|b| b.grid.nx1 * b.grid.ny1).sum::<usize>()
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
        // --- 1. Niveau grossier (force Guo, CL crate uniforme) ---
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

        // --- 2. Couplage grossier→fin AVANT les sous-pas fins (§1.2, Lagrava
        // et al. 2015 : injection FRAÎCHE de l'état post-couplage grossier,
        // pas de retard d'un pas complet) — fantômes côté interface
        // préparés pour les 2 sous-pas ---
        let xi = self.xi;
        let coarse = &self.coarse;
        for block in &mut self.blocks {
            coarse_to_fine(&block.grid, coarse, &mut block.fine, &xi);
        }

        // --- 3. Blocs fins : 2 sous-pas chacun (dt_fin = dt_grossier/2) ---
        let tau_fine = self.tau_fine;
        let beta_f = self.beta_eff_fine;
        for block in &mut self.blocks {
            let (wl, wr) = (block.hot_wall, block.cold_wall);
            for _ in 0..2 {
                collide_cavity(
                    &mut block.fine,
                    &tau_fine,
                    beta_f,
                    &mut block.fy,
                    true,
                    jobs,
                );
                // Bounce-back EXACT sur les parois physiques du bloc (haut/
                // bas toujours, gauche/droite selon le mur collé) — sinon la
                // paroi fine tire ses distributions entrantes d'une
                // extrapolation grossière (no-slip corrompu).
                stream_cavity_block(&mut block.fine, jobs, wl, wr, true, true);
                apply_thermal_boundary_conditions_block(
                    &mut block.fine,
                    block.hot_wall,
                    block.cold_wall,
                );
                update_macroscopic_guo(&mut block.fine, &block.fy, jobs);
                enforce_wall_velocity_block(&mut block.fine, block.hot_wall, block.cold_wall);
            }
        }

        // --- 4. Couplage fin→grossier (§1.2), par bloc ---
        for block in &mut self.blocks {
            fine_to_coarse(&block.grid, &block.fine, &mut self.coarse, &xi);
        }

        // --- 5. Moments coïncidents grossiers recalculés (post-injection) ---
        let idxs: Vec<usize> = self
            .blocks
            .iter()
            .flat_map(|b| b.grid.coincident.iter().map(|c| c.k_coarse))
            .collect();
        let coarse = &mut self.coarse;
        for k in idxs {
            let mut rho = 0.0;
            let mut mx = 0.0;
            let mut my = 0.0;
            let mut t = 0.0;
            for i in 0..9 {
                let fi = coarse.f[k][i];
                rho += fi;
                mx += D2Q9_VELOCITIES[i][0] as f64 * fi;
                my += D2Q9_VELOCITIES[i][1] as f64 * fi;
                t += coarse.g[k][i];
            }
            coarse.rho[k] = rho;
            coarse.temp[k] = t;
            if rho > f64::EPSILON {
                coarse.ux[k] = mx / rho;
                coarse.uy[k] = my / rho;
            } else {
                coarse.ux[k] = 0.0;
                coarse.uy[k] = 0.0;
            }
        }
    }

    /// Monitoring bruit acoustique d'un pas (LA porte V2, plan §3.1) :
    /// max|ρ−ρ̄| global (tous les nœuds des deux niveaux, ρ̄ = moyenne
    /// grossière) et aux interfaces (nœuds fantômes fins injectés + nœuds
    /// grossiers coïncidents). Ajout strictement par nœud ⇒ déterministe
    /// quel que soit --jobs.
    pub fn noise_stats(&self) -> NoiseStats {
        let n0 = self.nx0 * self.ny0;
        let rho_bar: f64 = self.coarse.rho.iter().sum::<f64>() / n0 as f64;

        // Bruit global : tous les nœuds des deux niveaux.
        let mut max_dev = 0.0f64;
        for &v in &self.coarse.rho {
            max_dev = max_dev.max((v - rho_bar).abs());
        }
        for block in &self.blocks {
            for &v in &block.fine.rho {
                max_dev = max_dev.max((v - rho_bar).abs());
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
        }
        if !check_finite_levels(&lvls) {
            return Some("DIVERGENCE détectée (valeurs non finies).".to_string());
        }
        None
    }
}
