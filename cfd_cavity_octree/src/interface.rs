//! Module interface (LE point critique, plan §1.2) — transfert grossier⇄fin.
//!
//! Règle absolue (§1.2, Lagrava et al. 2015, Rohde et al. 2006, Dupuis &
//! Chopard 2003) : **ne JAMAIS interpoler f directement**. On transfère :
//! 1. les moments ρ, u, T (interpolation bilinéaire, conservés) ;
//! 2. le non-équilibre f^neq reconstruit depuis les distributions du côté
//!    émetteur (fneq = f − feq(ρ, u, T)), interpolé bilinéairement, puis
//!    **rescalé** :
//!    f^neq_fin = (1/2)·ξ·f^neq_grossier, avec décomposition TRT odd/even :
//!    - partie **paire** (stress, relaxée par τ⁺) :
//!      f^neq,fin_even = (1/2)·(τ⁺_fin/τ⁺_grossier)·f^neq,grossier_even ;
//!    - partie **impaire** (momentum, relaxée par τ⁻) : ξ_odd = 1/2 en
//!      première approximation (variante exacte (1/2)·(τ⁻_fin/τ⁻_grossier)
//!      disponible via `odd_exact`, à valider en V2, cf. plan §4) ;
//!    - facteur 1/2 : Δt_fin = Δt_grossier/2 (acoustic scaling) — le f^neq
//!      est un flux temporel (τ·contrainte), il se réduit avec le pas de temps.
//! 3. le scalaire g (double distribution) : g^neq_fin =
//!    (1/2)·(τ_t,fin/τ_t,grossier)·g^neq_grossier (BGK : τ_t⁻ = τ_t⁺, un seul
//!    facteur — la décomposition odd/even serait redondante).
//!
//! fin→grossier (§1.2) : moyenne des moments sur les 2×2 nœuds fins
//! coïncidents, f^neq_grossier = moyenne des f^neq fins rescalée par 1/ξ
//! (inversion exacte des facteurs : ×2·τ⁺_gross/τ⁺_fin sur la partie paire,
//! ×2 sur l'impaire).
//!
//! Propriété de conservation (testée en V1) : la décomposition odd/even
//! préserve Σ_i fneq_i = 0 et Σ_i e_i·fneq_i = 0 ⇒ l'injection
//! f_fin = feq(moments interpolés) + fneq rescalé reproduit EXACTEMENT les
//! moments transférés (masse, momentum, température) à ε_machine près.

use crate::grid::OctreeGrid;
use crate::lbm::{LbmLevel, RelaxTimes};
use oxiphysics_lbm::lattice::{D2Q9_OPPOSITES, D2Q9_VELOCITIES};
use oxiphysics_lbm::thermal_lbm::{feq, geq};

/// Facteurs de rescaling du non-équilibre (grossier→fin), plan §1.2.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RescalingFactors {
    /// Partie paire (stress) : (1/2)·τ⁺_fin/τ⁺_grossier.
    pub xi_even: f64,
    /// Partie impaire (momentum) : 1/2 (approx.) ou (1/2)·τ⁻_fin/τ⁻_grossier.
    pub xi_odd: f64,
    /// Scalaire thermique g (BGK) : (1/2)·τ_t,fin/τ_t,grossier.
    pub xi_thermal: f64,
}

/// Calcule les facteurs de rescaling depuis les τ des deux niveaux.
/// `odd_exact = true` active la variante exacte de la partie impaire
/// ((1/2)·(τ⁻_fin/τ⁻_grossier), à valider en V2 — plan §1.2/§4).
pub fn rescaling_factors(tau_coarse: &RelaxTimes, tau_fine: &RelaxTimes, odd_exact: bool) -> RescalingFactors {
    let xi_even = 0.5 * tau_fine.tau_plus / tau_coarse.tau_plus;
    let xi_odd = if odd_exact {
        0.5 * tau_fine.tau_minus / tau_coarse.tau_minus
    } else {
        0.5
    };
    let xi_thermal = 0.5 * tau_fine.tau_t / tau_coarse.tau_t;
    RescalingFactors {
        xi_even,
        xi_odd,
        xi_thermal,
    }
}

/// Décompose un non-équilibre D2Q9 en parties paire (stress) et impaire
/// (momentum) : even_i = (fneq_i + fneq_opp)/2, odd_i = (fneq_i − fneq_opp)/2.
/// (even, odd) retournés.
pub fn split_odd_even(fneq: &[f64; 9]) -> ([f64; 9], [f64; 9]) {
    let mut even = [0.0f64; 9];
    let mut odd = [0.0f64; 9];
    for i in 0..9 {
        let opp = D2Q9_OPPOSITES[i];
        even[i] = 0.5 * (fneq[i] + fneq[opp]);
        odd[i] = 0.5 * (fneq[i] - fneq[opp]);
    }
    (even, odd)
}

/// Rescaling grossier→fin d'un non-équilibre (plan §1.2) :
/// fneq_fin_i = ξ_even·even_i + ξ_odd·odd_i.
/// Préserve Σ fneq = 0 et Σ e·fneq = 0 (les deux parties sont à somme et
/// momentum nuls) ⇒ moments transférés conservés exactement.
pub fn rescale_fneq(fneq: &[f64; 9], xi: &RescalingFactors) -> [f64; 9] {
    let (even, odd) = split_odd_even(fneq);
    let mut out = [0.0f64; 9];
    for i in 0..9 {
        out[i] = xi.xi_even * even[i] + xi.xi_odd * odd[i];
    }
    out
}

/// Rescaling inverse (fin→grossier, plan §1.2 : « × 1/ξ ») :
/// fneq_gross_i = even_i/ξ_even + odd_i/ξ_odd (inversion exacte : le facteur
/// temporel 1/2 devient 2, le ratio τ⁺ s'inverse).
pub fn rescale_fneq_inverse(fneq: &[f64; 9], xi: &RescalingFactors) -> [f64; 9] {
    let (even, odd) = split_odd_even(fneq);
    let mut out = [0.0f64; 9];
    for i in 0..9 {
        out[i] = even[i] / xi.xi_even + odd[i] / xi.xi_odd;
    }
    out
}

/// Transfert grossier→fin : injection des nœuds fantômes fins (§1.2).
/// Pour chaque fantôme : interpolation bilinéaire de ρ, u, T et du f^neq
/// reconstruit côté grossier, rescaling (1/2)·ξ, puis
/// f_fin = feq(ρ, u, T interpolés) + fneq rescalé (idem g avec ξ_thermal).
pub fn coarse_to_fine(grid: &OctreeGrid, coarse: &LbmLevel, fine: &mut LbmLevel, xi: &RescalingFactors) {
    for ghost in &grid.ghosts {
        let k_f = ghost.k_fine;
        // --- Moments interpolés (bilinear, conservés) ---
        let mut rho = 0.0;
        let mut ux = 0.0;
        let mut uy = 0.0;
        let mut temp = 0.0;
        for &(k_c, w) in &ghost.stencil {
            rho += w * coarse.rho[k_c];
            ux += w * coarse.ux[k_c];
            uy += w * coarse.uy[k_c];
            temp += w * coarse.temp[k_c];
        }
        // --- Non-équilibre reconstruit côté grossier, interpolé, rescalé ---
        let mut fneq_int = [0.0f64; 9];
        let mut gneq_int = [0.0f64; 9];
        for i in 0..9 {
            for &(k_c, w) in &ghost.stencil {
                fneq_int[i] += w * (coarse.f[k_c][i] - feq(i, coarse.rho[k_c], coarse.ux[k_c], coarse.uy[k_c]));
                gneq_int[i] += w * (coarse.g[k_c][i] - geq(i, coarse.temp[k_c], coarse.ux[k_c], coarse.uy[k_c]));
            }
        }
        let fneq_fin = rescale_fneq(&fneq_int, xi);
        // Scalaire BGK : un seul facteur (τ_t⁻ = τ_t⁺, pas de split odd/even).
        // --- Injection : équilibre des moments transférés + fneq rescalé ---
        for i in 0..9 {
            fine.f[k_f][i] = feq(i, rho, ux, uy) + fneq_fin[i];
            fine.g[k_f][i] = geq(i, temp, ux, uy) + xi.xi_thermal * gneq_int[i];
        }
    }
    // Moments des fantômes recalculés depuis les distributions injectées.
    refresh_moments(fine, &grid.ghosts.iter().map(|gh| gh.k_fine).collect::<Vec<_>>());
}

/// Transfert fin→grossier : moyenne 2×2 + 1/ξ (§1.2), injection sur les
/// nœuds grossiers coïncidents du bloc.
pub fn fine_to_coarse(grid: &OctreeGrid, fine: &LbmLevel, coarse: &mut LbmLevel, xi: &RescalingFactors) {
    for cell in &grid.coincident {
        let k_c = cell.k_coarse;
        // --- Moyenne des moments sur les 2×2 nœuds fins coïncidents ---
        let mut rho = 0.0;
        let mut ux = 0.0;
        let mut uy = 0.0;
        let mut temp = 0.0;
        for &k_f in &cell.fine_idx {
            rho += fine.rho[k_f];
            ux += fine.ux[k_f];
            uy += fine.uy[k_f];
            temp += fine.temp[k_f];
        }
        rho *= 0.25;
        ux *= 0.25;
        uy *= 0.25;
        temp *= 0.25;
        // --- Non-équilibre fin moyenné puis rescalé par 1/ξ ---
        let mut fneq_avg = [0.0f64; 9];
        let mut gneq_avg = [0.0f64; 9];
        for i in 0..9 {
            for &k_f in &cell.fine_idx {
                fneq_avg[i] += fine.f[k_f][i] - feq(i, fine.rho[k_f], fine.ux[k_f], fine.uy[k_f]);
                gneq_avg[i] += fine.g[k_f][i] - geq(i, fine.temp[k_f], fine.ux[k_f], fine.uy[k_f]);
            }
            fneq_avg[i] *= 0.25;
            gneq_avg[i] *= 0.25;
        }
        let fneq_gross = rescale_fneq_inverse(&fneq_avg, xi);
        // --- Injection côté grossier ---
        for i in 0..9 {
            coarse.f[k_c][i] = feq(i, rho, ux, uy) + fneq_gross[i];
            coarse.g[k_c][i] = geq(i, temp, ux, uy) + gneq_avg[i] / xi.xi_thermal;
        }
    }
}

/// Recalcule ρ, u, T d'un sous-ensemble de nœuds depuis les distributions.
fn refresh_moments(lvl: &mut LbmLevel, nodes: &[usize]) {
    for &k in nodes {
        let mut rho = 0.0;
        let mut mx = 0.0;
        let mut my = 0.0;
        let mut t = 0.0;
        for i in 0..9 {
            let fi = lvl.f[k][i];
            rho += fi;
            mx += fi * D2Q9_VELOCITIES[i][0] as f64;
            my += fi * D2Q9_VELOCITIES[i][1] as f64;
            t += lvl.g[k][i];
        }
        lvl.rho[k] = rho;
        lvl.ux[k] = mx / rho;
        lvl.uy[k] = my / rho;
        lvl.temp[k] = t;
    }
}


