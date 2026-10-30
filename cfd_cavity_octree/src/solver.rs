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
//! Choix documenté (plan §2.2, « pas de temps unique (fallback) ») : les
//! tests V1 unitaires n'exécutent pas la boucle complète — ils testent les
//! transferts d'interface isolés (pas unique, pas de subcycling) ; la
//! structure 2 sous-pas est néanmoins en place ici pour l'Étape B.

use crate::grid::OctreeGrid;
use crate::interface::{coarse_to_fine, fine_to_coarse, rescaling_factors, RescalingFactors};
use crate::lbm::{collide, stream, LbmLevel, RelaxTimes};

/// Solveur octree 2 niveaux.
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
    pub fn new(
        grid: OctreeGrid,
        nu_lu_coarse: f64,
        alpha_lu_coarse: f64,
        odd_exact: bool,
    ) -> Self {
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
        collide(&mut self.coarse, &self.tau_coarse);
        stream(&mut self.coarse);
        self.coarse.update_macroscopic();

        // 2. Niveau fin : 2 sous-pas (dt_fin = dt_grossier/2).
        for _ in 0..2 {
            collide(&mut self.fine, &self.tau_fine);
            stream(&mut self.fine);
            self.fine.update_macroscopic();
        }

        // 3. Couplage interfaces (§1.2) : grossier→fin puis fin→grossier.
        coarse_to_fine(&self.grid, &self.coarse, &mut self.fine, &self.xi);
        fine_to_coarse(&self.grid, &self.fine, &mut self.coarse, &self.xi);
    }
}
