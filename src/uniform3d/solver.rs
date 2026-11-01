//! Solveur 3D — calage unités + champs initiaux (Étape V1, plan 3D Trias §3).
//!
//! Cavité différentiellement chauffée AR ≈ 4 : parois verticales x isothermes
//! (chaude ix=0 / froide ix=nx−1), parois horizontales y adiabatiques,
//! profondeur z PÉRIODIQUE (L1 = 2, Trias Case A, JFM 2011 — wrap sans
//! bounce-back).
//!
//! Ce module ne contient AUCUN pas de temps CPU 3D : la validation passe par
//! les tests device 1-pas de `gpu.rs` (équilibre, streaming, conduction,
//! conservation, impulsion de Guo) et par l'ancre littéraire Trias
//! Nuc = 49.24 ± 5 % à Ra = 6.4e8. Le calage unités réplique exactement
//! `uniform/solver.rs` (c_scale = 1.7, plancher ν = 0.002, lz = hauteur).

use crate::uniform3d::rng::Rng;

/// Température paroi chaude (+0.5).
pub const T_HOT: f64 = 0.5;
/// Température paroi froide (−0.5).
pub const T_COLD: f64 = -0.5;
/// ΔT = T_hot − T_cold.
pub const DELTA_T: f64 = 1.0;

/// Solveur 3D : calage unités + champs macroscopiques initiaux.
///
/// Les populations f/g ne sont pas stockées côté CPU : le backend GPU
/// (`GpuSolver3D::from_solver`) les reconstruit depuis les équilibres
/// (feq_d3q19 / geq inline) à l'initialisation.
#[derive(Debug, Clone)]
pub struct CavitySolver3D {
    /// Nœuds en x (largeur, parois isothermes).
    pub nx: usize,
    /// Nœuds en y (hauteur, AR = (ny−1)/(nx−1) ≈ 4, parois adiabatiques).
    pub ny: usize,
    /// Nœuds en z (profondeur, PÉRIODIQUE — wrap sans bounce-back).
    pub nz: usize,
    /// Densité (layout k = ix + nx·(iy + ny·iz)).
    pub rho: Vec<f64>,
    /// Vitesse x.
    pub ux: Vec<f64>,
    /// Vitesse y (direction de la flottabilité).
    pub uy: Vec<f64>,
    /// Vitesse z (périodique).
    pub uz: Vec<f64>,
    /// Température (θ ∈ [−0.5, +0.5]).
    pub temperature: Vec<f64>,
    /// Viscosité turbulente par nœud (Smagorinsky statique, diagnostic).
    pub nu_t: Vec<f64>,
    /// Viscosité moléculaire calée (unités réseau).
    pub nu: f64,
    /// Diffusivité thermique α = ν/Pr (alpha_mult = 1.0 en V1).
    pub alpha: f64,
    /// Diffusivité physique ν/Pr (entre dans beta_eff, indépendante du levier
    /// de stabilisation).
    pub alpha_phys: f64,
    /// Force de flottabilité effective beta_eff = Ra·ν·α_phys/lz³.
    pub beta_eff: f64,
    /// Constante de Smagorinsky Cs.
    pub cs: f64,
    /// Prandtl turbulent.
    pub pr_t: f64,
    /// Température de référence Boussinesq (= 0).
    pub t_ref: f64,
    /// Hauteur (adimensionnalisation Nusselt : Nu = ∂θ/∂n·lz).
    pub lz: f64,
    /// Largeur.
    pub lx: f64,
    /// Durée de la rampe de spin-up (pas) : clamp(ny/10, 200, 2000).
    pub spin_up_steps: usize,
    /// Multiplicateur de la viscosité initiale (rampe exponentielle).
    pub nu_start_mult: f64,
}

impl CavitySolver3D {
    /// Crée le solveur et initialise : T linéaire en x + perturbation ±1e-3,
    /// u = 0, ρ = 1. Calage unités identique à `uniform/solver.rs` (lz = ny−1).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        ra: f64,
        pr: f64,
        cs: f64,
        pr_t: f64,
        utarget: f64,
        seed: u64,
        nu_start_mult: f64,
    ) -> Self {
        // --- unités de réseau calées pour matcher Ra et Pr (réplique 2D) ---
        // ν = (utarget·√Pr·lz / (c·√Ra))^(2/3), c ≈ 1.7 ; plancher ν ≥ 0.002
        // (τ+ ≥ 0.506, tenable avec TRT Λ = 3/16).
        let lz = (ny - 1) as f64;
        let lx = (nx - 1) as f64;

        let c_scale = 1.7;
        let nu_cal = (utarget * pr.sqrt() * lz / (c_scale * ra.sqrt())).powf(2.0 / 3.0);
        let nu = nu_cal.max(0.002);
        let alpha_phys = nu / pr;
        // Pas d'override --nu ni --alpha-mult en V1 3D (validé côté CLI).
        let alpha = alpha_phys;
        // beta_eff dérivé de la valeur PHYSIQUE de α (ν/Pr) — même convention
        // que le 2D : la flottabilité est une donnée physique.
        let beta_eff = ra * nu * alpha_phys / lz.powi(3);

        let tau = 0.5 + 3.0 * nu;
        let tau_t = 0.5 + 3.0 * alpha;
        assert!(
            tau > 0.505,
            "tau={tau} trop proche de 0.5 : augmente --utarget ou réduis la grille (plancher TRT : τ+ ≥ 0.506, soit ν ≥ 0.002)"
        );
        assert!(
            tau_t > 0.505,
            "tau_t={tau_t} trop proche de 0.5 : ajuste --utarget (Pr trop grand pour cette vitesse cible)"
        );

        let n = nx * ny * nz;
        let mut temperature = vec![0.0f64; n];
        let mut rng = Rng::new(seed);

        // --- initialisation : T linéaire en x + perturbation ±1e-3 ---
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let k = ix + nx * (iy + ny * iz);
                    let xf = ix as f64 / (nx - 1) as f64;
                    let t_lin = T_HOT + (T_COLD - T_HOT) * xf;
                    let pert = 1.0e-3 * (2.0 * rng.next_f64() - 1.0);
                    temperature[k] = t_lin + pert;
                }
            }
        }

        Self {
            nx,
            ny,
            nz,
            rho: vec![1.0; n],
            ux: vec![0.0; n],
            uy: vec![0.0; n],
            uz: vec![0.0; n],
            temperature,
            nu_t: vec![0.0; n],
            nu,
            alpha,
            alpha_phys,
            beta_eff,
            cs,
            pr_t,
            t_ref: 0.0,
            lz,
            lx,
            // Rampe de spin-up : 10 % des zones verticales, bornée [200, 2000]
            // (réplique 2D).
            spin_up_steps: ((ny / 10).max(200)).min(2000),
            nu_start_mult: nu_start_mult.max(1.0),
        }
    }

    /// Index plat (ix + nx·(iy + ny·iz)).
    #[inline]
    pub fn idx(&self, ix: usize, iy: usize, iz: usize) -> usize {
        ix + self.nx * (iy + self.ny * iz)
    }

    /// Force beta_eff (test conduction pure : set_beta_eff(0.0)).
    pub fn set_beta_eff(&mut self, v: f64) {
        self.beta_eff = v;
    }

    /// Nusselt paroi chaude : gradient 2ᵉ liaison (ix=1→2), moyenné sur iy
    /// ET iz (convention 2D étendue à la profondeur périodique), × lz/ΔT.
    pub fn nusselt_hot(&self) -> f64 {
        let mut sum = 0.0;
        for iz in 0..self.nz {
            for iy in 0..self.ny {
                let t1 = self.temperature[self.idx(1, iy, iz)];
                let t2 = self.temperature[self.idx(2, iy, iz)];
                sum += (t1 - t2).abs();
            }
        }
        let avg_grad = sum / (self.ny * self.nz) as f64;
        avg_grad * self.lz / DELTA_T
    }

    /// Nusselt paroi froide : gradient 2ᵉ liaison (ix=nx−3→nx−2), moyenné
    /// sur iy et iz, × lz/ΔT.
    pub fn nusselt_cold(&self) -> f64 {
        let mut sum = 0.0;
        for iz in 0..self.nz {
            for iy in 0..self.ny {
                let t0 = self.temperature[self.idx(self.nx - 3, iy, iz)];
                let t1 = self.temperature[self.idx(self.nx - 2, iy, iz)];
                sum += (t1 - t0).abs();
            }
        }
        let avg_grad = sum / (self.ny * self.nz) as f64;
        avg_grad * self.lz / DELTA_T
    }

    /// Vitesse max du champ (diagnostic global |u|∞).
    pub fn max_u(&self) -> f64 {
        self.ux
            .iter()
            .zip(self.uy.iter())
            .zip(self.uz.iter())
            .fold(0.0f64, |m, ((&x, &y), &z)| m.max(x.abs().max(y.abs().max(z.abs()))))
    }

    /// Masse totale Σρ (diagnostic de conservation).
    pub fn total_mass(&self) -> f64 {
        self.rho.iter().sum()
    }

    /// Vérifie que tous les champs sont finis (ρ > 0).
    pub fn all_finite(&self) -> bool {
        self.temperature.iter().all(|v| v.is_finite())
            && self.ux.iter().all(|v| v.is_finite())
            && self.uy.iter().all(|v| v.is_finite())
            && self.uz.iter().all(|v| v.is_finite())
            && self.rho.iter().all(|v| v.is_finite() && *v > 0.0)
    }
}
