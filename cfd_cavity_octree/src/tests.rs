//! Tests V1 (porte de l'Étape A, plan §3.1) — module interface.
//!
//! 1. `v1_interface_conserves_mass_and_momentum` : transfert grossier→fin sur
//!    champ imposé (couette uy linéaire en x, gradient T linéaire en y,
//!    ρ = 1 + ε·sin(2πy/Ny), non-équilibre non nul) — sur les nœuds fantômes
//!    injectés : masse/temperature/momentum à 1e-12 (moments interpolés +
//!    ξ_odd·momentum du non-équilibre ; la partie paire ne porte PAS de
//!    momentum).
//! 2. `v1_fine_to_coarse_conserves_and_inverse` : transfert fin→grossier —
//!    masse/temperature/momentum des blocs coïncidents à 1e-12 (moyenne 2×2 +
//!    1/ξ), et inversion algébrique exacte rescale⁻¹∘rescale = id.
//! 3. `v1_xi_matches_relaxation_time_ratio` : ξ == (1/2)·τ_fin/τ_grossier exact
//!    sur les composantes paires (τ⁺) et thermiques, ξ_odd = 1/2 (impaires).
//!
//! Note de périmètre (choix documenté, plan §2.2) : `coarse_to_fine` n'écrit
//! que les nœuds fantômes (les nœuds fins intérieurs évoluent par
//! collide+stream) et `fine_to_coarse` ne lit que les 2×2 nœuds fins
//! coïncidents — les tests vérifient donc exactement ces sous-ensembles.

use crate::grid::OctreeGrid;
use crate::interface::{rescale_fneq, rescale_fneq_inverse, rescaling_factors, split_odd_even};
use crate::lbm::{RelaxTimes, LAMBDA_TRT};
use crate::solver::OctreeSolver;
use oxiphysics_lbm::lattice::{D2Q9_VELOCITIES, D2Q9_WEIGHTS};
use oxiphysics_lbm::thermal_lbm::{feq, geq};

/// Géométrie de test : domaine grossier 16×16, bloc fin [4,12)×[4,12).
const NX0: usize = 16;
const NY0: usize = 16;
const BX0: usize = 4;
const BY0: usize = 4;
const BX1: usize = 12;
const BY1: usize = 12;

/// ν et α en unités lattice du niveau grossier (τ⁺ = 0.53, τ_t = 0.56).
const NU_COARSE: f64 = 0.01;
const ALPHA_COARSE: f64 = 0.02;

/// Stencil bilinéaire aux coordonnées grossières (x, y) — réplique indépendante
/// de `grid::bilinear_stencil` pour vérifier les tables précalculées.
fn bilinear_at(x: f64, y: f64) -> [(usize, f64); 4] {
    let ix = x.floor() as usize;
    let iy = y.floor() as usize;
    let fx = x - ix as f64;
    let fy = y - iy as f64;
    [
        (iy * NX0 + ix, (1.0 - fx) * (1.0 - fy)),
        (iy * NX0 + ix + 1, fx * (1.0 - fy)),
        ((iy + 1) * NX0 + ix, (1.0 - fx) * fy),
        ((iy + 1) * NX0 + ix + 1, fx * fy),
    ]
}

/// Initialise un solveur avec le champ imposé V1 :
/// - couette : uy = U0·(x − Nx/2)/(Nx/2) (linéaire en x), ux = 0 ;
/// - gradient thermique : T = T0 + cT·(y − Ny/2)/(Ny/2) (linéaire en y) ;
/// - ρ = 1 + ε·sin(2π·y/Ny) ;
/// - non-équilibre imposé : fneq = εf·w_i·e_iy (impair, porte le momentum),
///   gneq = εg·w_i·e_ix (pair, somme nulle).
/// Les moments stockés sont recalculés depuis les distributions (cohérence
/// exacte ρ = Σf, ρu = Σef).
fn init_solver_v1() -> OctreeSolver {
    let grid = OctreeGrid::new(NX0, NY0, BX0, BY0, BX1, BY1);
    let mut s = OctreeSolver::new(grid, NU_COARSE, ALPHA_COARSE, false);
    let eps = 1e-3;
    let eps_f = 1e-4;
    let eps_g = 1e-5;
    let u0 = 0.01;
    let ct = 0.01;
    for iy in 0..NY0 {
        for ix in 0..NX0 {
            let k = iy * NX0 + ix;
            let x = ix as f64;
            let y = iy as f64;
            let uy = u0 * (x - NX0 as f64 / 2.0) / (NX0 as f64 / 2.0);
            let temp = 0.5 + ct * (y - NY0 as f64 / 2.0) / (NY0 as f64 / 2.0);
            s.coarse.rho[k] = 1.0 + eps * (std::f64::consts::TAU * y / NY0 as f64).sin();
            s.coarse.ux[k] = 0.0;
            s.coarse.uy[k] = uy;
            s.coarse.temp[k] = temp;
            for i in 0..9 {
                let ey = D2Q9_VELOCITIES[i][1] as f64;
                let ex = D2Q9_VELOCITIES[i][0] as f64;
                s.coarse.f[k][i] = feq(i, s.coarse.rho[k], 0.0, uy) + eps_f * D2Q9_WEIGHTS[i] * ey;
                s.coarse.g[k][i] = geq(i, temp, 0.0, uy) + eps_g * D2Q9_WEIGHTS[i] * ex;
            }
        }
    }
    s.coarse.update_macroscopic();
    // Niveau fin : distributions nulles (seuls les fantômes sont injectés par
    // le transfert — les nœuds intérieurs sont du ressort de collide+stream).
    s
}

/// Moments attendus sur un nœud fin à la position grossière (x, y) :
/// interpolation bilinéaire de ρ, u, T (choix standard Dupuis & Chopard :
/// on interpole la VITESSE u, pas le momentum ρu — §1.2 « transférer les
/// moments ρ, u, T ») + contribution ξ_odd·momentum du f^neq interpolé
/// (la partie paire ξ_even a un momentum exactement nul par antisymétrie
/// e_opp = −e_i). Retourne (rho, mx, my, T).
fn expected_moments(s: &OctreeSolver, x: f64, y: f64) -> (f64, f64, f64, f64) {
    let stencil = bilinear_at(x, y);
    let mut rho = 0.0;
    let mut ux = 0.0;
    let mut uy = 0.0;
    let mut t = 0.0;
    let mut fneq_mx = 0.0;
    let mut fneq_my = 0.0;
    for &(k_c, w) in &stencil {
        rho += w * s.coarse.rho[k_c];
        ux += w * s.coarse.ux[k_c];
        uy += w * s.coarse.uy[k_c];
        t += w * s.coarse.temp[k_c];
        for i in 0..9 {
            let fneq = s.coarse.f[k_c][i]
                - feq(i, s.coarse.rho[k_c], s.coarse.ux[k_c], s.coarse.uy[k_c]);
            fneq_mx += w * D2Q9_VELOCITIES[i][0] as f64 * fneq;
            fneq_my += w * D2Q9_VELOCITIES[i][1] as f64 * fneq;
        }
    }
    // Momentum total attendu : ρ_interp·u_interp + ξ_odd·m(fneq_interp).
    // (m(fneq_k) est algébriquement nul par nœud — uy_k est défini comme
    // m(f)/ρ_k — donc ce terme est ~0 numérique ; le vérifier explicitement
    // teste que le rescaling ne pollue pas le momentum transféré.)
    (rho, rho * ux + s.xi.xi_odd * fneq_mx, rho * uy + s.xi.xi_odd * fneq_my, t)
}

/// Test V1 (LA porte, plan §3.1) : transfert grossier→fin sur les nœuds
/// fantômes — masse (ρ), température (T) et momentum (ρu) à 1e-12 :
/// ρ/T = interpolation bilinéaire exacte des moments grossiers (le non-équilibre
/// ne pollue NI la masse NI la température : Σ fneq = Σ gneq = 0 préservé par
/// la décomposition odd/even, §1.2) ; ρu = moments interpolés +
/// ξ_odd·(momentum du f^neq).
#[test]
fn v1_interface_conserves_mass_and_momentum() {
    let mut s = init_solver_v1();
    crate::interface::coarse_to_fine(&s.grid, &s.coarse, &mut s.fine, &s.xi);

    let nx1 = s.fine.nx;
    let mut err_rho = 0.0f64;
    let mut err_t = 0.0f64;
    let mut err_mx = 0.0f64;
    let mut err_my = 0.0f64;
    let (mut sum_rho, mut sum_t, mut sum_mx, mut sum_my) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut exp_rho, mut exp_t, mut exp_mx, mut exp_my) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);

    for gh in &s.grid.ghosts {
        let k = gh.k_fine;
        // Moments réels du fantôme, calculés depuis les distributions injectées
        // (sans passer par update_macroscopic : les nœuds intérieurs non
        // injectés ont ρ = 0 et ne font pas partie du périmètre du transfert).
        let mut rho = 0.0;
        let mut mx = 0.0;
        let mut my = 0.0;
        let mut t = 0.0;
        for i in 0..9 {
            rho += s.fine.f[k][i];
            mx += D2Q9_VELOCITIES[i][0] as f64 * s.fine.f[k][i];
            my += D2Q9_VELOCITIES[i][1] as f64 * s.fine.f[k][i];
            t += s.fine.g[k][i];
        }
        let a = k % nx1;
        let b = k / nx1;
        let x = BX0 as f64 - 0.5 + 0.5 * a as f64;
        let y = BY0 as f64 - 0.5 + 0.5 * b as f64;
        let (e_rho, e_mx, e_my, e_t) = expected_moments(&s, x, y);
        err_rho = err_rho.max((rho - e_rho).abs());
        err_t = err_t.max((t - e_t).abs());
        err_mx = err_mx.max((mx - e_mx).abs());
        err_my = err_my.max((my - e_my).abs());
        sum_rho += rho;
        sum_t += t;
        sum_mx += mx;
        sum_my += my;
        exp_rho += e_rho;
        exp_t += e_t;
        exp_mx += e_mx;
        exp_my += e_my;
    }

    println!("V1 conservation (grossier→fin, {} fantômes) :", s.grid.ghosts.len());
    println!("  max |Δρ|   par fantôme = {err_rho:.3e}");
    println!("  max |ΔT|   par fantôme = {err_t:.3e}");
    println!("  max |Δρux| par fantôme = {err_mx:.3e}");
    println!("  max |Δρuy| par fantôme = {err_my:.3e}");
    println!("  Σρ  = {sum_rho:.15e}  attendu = {exp_rho:.15e}");
    println!("  ΣT  = {sum_t:.15e}  attendu = {exp_t:.15e}");
    println!("  Σρux = {sum_mx:.15e} attendu = {exp_mx:.15e}");
    println!("  Σρuy = {sum_my:.15e} attendu = {exp_my:.15e}");

    assert!(err_rho < 1e-12, "masse par fantôme : {err_rho}");
    assert!(err_t < 1e-12, "température par fantôme : {err_t}");
    assert!(err_mx < 1e-12, "momentum x par fantôme : {err_mx}");
    assert!(err_my < 1e-12, "momentum y par fantôme : {err_my}");
    assert!((sum_rho - exp_rho).abs() < 1e-12, "Σρ : {}", (sum_rho - exp_rho).abs());
    assert!((sum_t - exp_t).abs() < 1e-12, "ΣT : {}", (sum_t - exp_t).abs());
    assert!((sum_mx - exp_mx).abs() < 1e-12, "Σρux : {}", (sum_mx - exp_mx).abs());
    assert!((sum_my - exp_my).abs() < 1e-12, "Σρuy : {}", (sum_my - exp_my).abs());
}

/// Test V1 symétrique (fin→grossier, §1.2 : « moyenne 2×2 + 1/ξ ») :
/// 1. conservation par bloc coïncident — ρ_c = moyenne 2×2 de ρ_f, T_c idem,
///    j_c = moyenne 2×2 de j_f + (1/ξ_odd)·momentum du f^neq moyenné ;
/// 2. inversion algébrique exacte : rescale⁻¹(rescale(fneq)) == fneq à 1e-15
///    (les facteurs 1/ξ s'annulent bit à bit sur les parties paire/impaire).
#[test]
fn v1_fine_to_coarse_conserves_and_inverse() {
    let grid = OctreeGrid::new(NX0, NY0, BX0, BY0, BX1, BY1);
    let mut s = OctreeSolver::new(grid, NU_COARSE, ALPHA_COARSE, false);

    // Champ fin imposé (nœuds intérieurs, seuls lus par fine_to_coarse) :
    // couette + gradient T + ρ = 1 + ε·sin(2πy/ny1) + f^neq pair et impair.
    let eps = 1e-3;
    let eps_e = 2e-4;
    let eps_o = 5e-4;
    let u0 = 0.02;
    for b in 1..s.fine.ny - 1 {
        for a in 1..s.fine.nx - 1 {
            let k = b * s.fine.nx + a;
            let x = BX0 as f64 - 0.5 + 0.5 * a as f64;
            let y = BY0 as f64 - 0.5 + 0.5 * b as f64;
            let uy = u0 * (x - NX0 as f64 / 2.0) / (NX0 as f64 / 2.0);
            let temp = 0.5 + 0.01 * (y - NY0 as f64 / 2.0) / (NY0 as f64 / 2.0);
            s.fine.rho[k] = 1.0 + eps * (std::f64::consts::TAU * y / s.fine.ny as f64).sin();
            s.fine.ux[k] = 0.0;
            s.fine.uy[k] = uy;
            s.fine.temp[k] = temp;
            for i in 0..9 {
                let ex = D2Q9_VELOCITIES[i][0] as f64;
                let ey = D2Q9_VELOCITIES[i][1] as f64;
                let fneq = eps_e * D2Q9_WEIGHTS[i] * ex * ey + eps_o * D2Q9_WEIGHTS[i] * ex;
                s.fine.f[k][i] = feq(i, s.fine.rho[k], 0.0, uy) + fneq;
                s.fine.g[k][i] = geq(i, temp, 0.0, uy) + 1e-5 * D2Q9_WEIGHTS[i] * ey;
            }
        }
    }

    crate::interface::fine_to_coarse(&s.grid, &s.fine, &mut s.coarse, &s.xi);

    let mut err_rho = 0.0f64;
    let mut err_t = 0.0f64;
    let mut err_mx = 0.0f64;
    let mut err_my = 0.0f64;
    for cell in &s.grid.coincident {
        let k_c = cell.k_coarse;
        // Moyenne 2×2 attendue des moments fins + contribution 1/ξ_odd du
        // f^neq (la partie paire rescalée par 1/ξ_even a un momentum nul).
        let (mut e_rho, mut e_mx, mut e_my, mut e_t) = (0.0, 0.0, 0.0, 0.0);
        let mut fneq_mx = 0.0;
        let mut fneq_my = 0.0;
        for &k_f in &cell.fine_idx {
            e_rho += s.fine.rho[k_f];
            e_mx += s.fine.rho[k_f] * s.fine.ux[k_f];
            e_my += s.fine.rho[k_f] * s.fine.uy[k_f];
            e_t += s.fine.temp[k_f];
            for i in 0..9 {
                let fneq = s.fine.f[k_f][i]
                    - feq(i, s.fine.rho[k_f], s.fine.ux[k_f], s.fine.uy[k_f]);
                fneq_mx += D2Q9_VELOCITIES[i][0] as f64 * fneq;
                fneq_my += D2Q9_VELOCITIES[i][1] as f64 * fneq;
            }
        }
        e_rho *= 0.25;
        e_mx = 0.25 * e_mx + fneq_mx / (4.0 * s.xi.xi_odd);
        e_my = 0.25 * e_my + fneq_my / (4.0 * s.xi.xi_odd);
        e_t *= 0.25;

        // Moments réels du nœud grossier injecté (depuis les distributions).
        let mut rho = 0.0;
        let mut mx = 0.0;
        let mut my = 0.0;
        let mut t = 0.0;
        for i in 0..9 {
            rho += s.coarse.f[k_c][i];
            mx += D2Q9_VELOCITIES[i][0] as f64 * s.coarse.f[k_c][i];
            my += D2Q9_VELOCITIES[i][1] as f64 * s.coarse.f[k_c][i];
            t += s.coarse.g[k_c][i];
        }
        err_rho = err_rho.max((rho - e_rho).abs());
        err_t = err_t.max((t - e_t).abs());
        err_mx = err_mx.max((mx - e_mx).abs());
        err_my = err_my.max((my - e_my).abs());
    }
    println!(
        "V1 fin→grossier ({} blocs coïncidents) : max |Δρ|={err_rho:.3e}, |ΔT|={err_t:.3e}, |Δρux|={err_mx:.3e}, |Δρuy|={err_my:.3e}",
        s.grid.coincident.len()
    );
    assert!(err_rho < 1e-12, "masse par bloc : {err_rho}");
    assert!(err_t < 1e-12, "température par bloc : {err_t}");
    assert!(err_mx < 1e-12, "momentum x par bloc : {err_mx}");
    assert!(err_my < 1e-12, "momentum y par bloc : {err_my}");

    // Inversion algébrique : rescale⁻¹(rescale(fneq)) == fneq (1/ξ exact).
    let mut fneq = [0.0f64; 9];
    for i in 0..9 {
        let ex = D2Q9_VELOCITIES[i][0] as f64;
        let ey = D2Q9_VELOCITIES[i][1] as f64;
        fneq[i] = eps_e * D2Q9_WEIGHTS[i] * ex * ey + eps_o * D2Q9_WEIGHTS[i] * ex;
    }
    let up = rescale_fneq(&fneq, &s.xi);
    let down = rescale_fneq_inverse(&up, &s.xi);
    let err_inv: f64 = fneq.iter().zip(down.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("V1 inversion 1/ξ : max |Δfneq| = {err_inv:.3e}");
    assert!(err_inv < 1e-15, "inversion 1/ξ non exacte : {err_inv}");
}

/// Test V1 ξ : les facteurs valent exactement (1/2)·τ_fin/τ_grossier sur la
/// partie paire (τ⁺, stress) et thermique (τ_t), ξ_odd = 1/2 sur l'impaire
/// (approximation V1 ; variante exacte (1/2)·τ⁻_fin/τ⁻_grossier via flag) ;
/// le rescaling d'un f^neq purement pair (resp. pur impair) le multiplie par
/// ξ_even (resp. ξ_odd) composante par composante ; Σ fneq = 0 préservé.
#[test]
fn v1_xi_matches_relaxation_time_ratio() {
    let tau_c = RelaxTimes::from_nu_alpha(NU_COARSE, ALPHA_COARSE);
    let tau_f = RelaxTimes::from_nu_alpha(2.0 * NU_COARSE, 2.0 * ALPHA_COARSE);

    // §1.3 : ν_lu double sur le niveau fin ⇒ τ⁺_fin − 0.5 = 2·(τ⁺_gross − 0.5).
    assert!((tau_f.tau_plus - 0.5 - 2.0 * (tau_c.tau_plus - 0.5)).abs() < 1e-15);

    // ξ = (1/2)·τ_fin/τ_grossier : exact sur τ⁺ (parties paires) et τ_t.
    let xi = rescaling_factors(&tau_c, &tau_f, false);
    assert_eq!(xi.xi_even, 0.5 * tau_f.tau_plus / tau_c.tau_plus, "ξ_even");
    assert_eq!(xi.xi_thermal, 0.5 * tau_f.tau_t / tau_c.tau_t, "ξ_thermal");
    assert_eq!(xi.xi_odd, 0.5, "ξ_odd (approximation V1)");
    // Variante exacte impaire (V2) : (1/2)·τ⁻_fin/τ⁻_grossier.
    let xi_exact = rescaling_factors(&tau_c, &tau_f, true);
    assert_eq!(xi_exact.xi_odd, 0.5 * tau_f.tau_minus / tau_c.tau_minus);
    // Λ = 3/16 conservé côté f (bounce-back).
    assert_eq!(tau_c.tau_minus, 0.5 + LAMBDA_TRT / (tau_c.tau_plus - 0.5));

    // Structure odd/even : f^neq purement pair ⇒ rescalé par ξ_even exact ;
    // purement impair ⇒ ξ_odd exact.
    let mut fneq_even = [0.0f64; 9];
    let mut fneq_odd = [0.0f64; 9];
    for i in 0..9 {
        let ex = D2Q9_VELOCITIES[i][0] as f64;
        let ey = D2Q9_VELOCITIES[i][1] as f64;
        fneq_even[i] = 3e-3 * D2Q9_WEIGHTS[i] * ex * ey; // pair
        fneq_odd[i] = 5e-3 * D2Q9_WEIGHTS[i] * ex; // impair
    }
    let out_even = rescale_fneq(&fneq_even, &xi);
    let out_odd = rescale_fneq(&fneq_odd, &xi);
    let err_e: f64 = out_even
        .iter()
        .zip(fneq_even.iter())
        .map(|(o, i)| (o - xi.xi_even * i).abs())
        .fold(0.0, f64::max);
    let err_o: f64 = out_odd
        .iter()
        .zip(fneq_odd.iter())
        .map(|(o, i)| (o - xi.xi_odd * i).abs())
        .fold(0.0, f64::max);
    println!(
        "V1 ξ : ξ_even = {:.15}, ξ_odd = {:.3}, ξ_thermal = {:.15}",
        xi.xi_even, xi.xi_odd, xi.xi_thermal
    );
    println!("V1 ξ : max |Δ| pair = {err_e:.3e}, impair = {err_o:.3e}");
    assert_eq!(err_e, 0.0, "rescaling pair non exact");
    assert_eq!(err_o, 0.0, "rescaling impair non exact");

    // split_odd_even : recombinaison even + odd == fneq exactement.
    let (ev, od) = split_odd_even(&fneq_even);
    let err_split: f64 = ev.iter().zip(od.iter()).zip(fneq_even.iter()).map(|((e, o), i)| (e + o - i).abs()).fold(0.0, f64::max);
    assert_eq!(err_split, 0.0, "split_odd_even non exact");

    // Σ fneq = 0 préservé par le rescaling (masse) : entrée à somme nulle
    // (les deux champs fneq_even/fneq_odd vérifient Σ w·e = 0) ⇒ sortie idem.
    let mut mixed = [0.0f64; 9];
    for i in 0..9 {
        mixed[i] = fneq_even[i] + fneq_odd[i];
    }
    let out = rescale_fneq(&mixed, &xi);
    let sum_out: f64 = out.iter().sum();
    println!("V1 ξ : |Σ fneq rescalé| = {sum_out:.3e}");
    assert!(sum_out.abs() < 1e-15, "Σ fneq rescalé ≠ 0 : {sum_out}");
}
