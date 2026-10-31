//! Cavité DHC 2D sur grille octree (Étape B, plan §3.3 étape 2 — porte V2).
//!
//! Physique copiée à l'IDENTIQUE du crate uniforme validé
//! `cfd_cavity/src/solver.rs` (NE PAS modifier, plan §2) :
//! - force de Boussinesq façon Guo, Shi & Zheng 2002 (éq. (14)-(16)) avec
//!   exactitude PAR MODE TRT : facteur (1 − 1/2τ⁻) sur la partie impaire
//!   (momentum) et (1 − 1/2τ⁺) sur la partie paire (stress) — formules
//!   exactes de `collide_node` (solver.rs l.1159-1179) ;
//! - correction de vitesse u_y += F·Δt/2ρ (Guo éq. (16)) dans la mise à jour
//!   macroscopique, avec la MÊME F que la collision (stockée par nœud dans
//!   `fy`, solver.rs l.295-366) ;
//! - conditions thermiques : parois verticales isothermes (θ = ±0.5),
//!   horizontales adiabatiques (zero-gradient), no-slip (u = 0 aux parois) —
//!   `apply_thermal_boundary_conditions` / `enforce_wall_velocity` copiées ;
//! - Nusselt parois chaude/froide : 2ᵉ liaison (formule exacte du crate
//!   uniforme `post.rs`).
//!
//! Différence avec le crate uniforme (§1.3 du plan) : ν et α en unités
//! lattice sont LOCAUX à chaque niveau (ν_lu,fin = 2·ν_lu,grossier en
//! acoustic scaling) ; la flottabilité en unités lattice vaut
//! beta_eff_niveau = Ra·ν_lu·α_lu/Lz_lu³, soit beta_eff_fin =
//! beta_eff_grossier/2 (le facteur dt²/Δx est halvé : dt_fin = dt_gross/2,
//! Δx_fin = Δx_gross/2) — la vitesse de flottabilité u_scale = √(beta_eff·Lz)
//! est alors IDENTIQUE sur les deux niveaux, comme l'exige l'acoustic scaling.
//!
//! Pas de Smagorinsky en Étape B (port --dynamic-smag : Étape D, plan §3.3).

use crate::lbm::{LbmLevel, RelaxTimes};
use oxiphysics_lbm::lattice::{D2Q9_OPPOSITES, D2Q9_VELOCITIES, D2Q9_WEIGHTS, CS2};
use oxiphysics_lbm::thermal_lbm::{boussinesq_force, feq, geq};
use rayon::prelude::*;

/// Température paroi chaude (gauche) — normalisation thèse θc = +0.5.
pub const T_HOT: f64 = 0.5;
/// Température paroi froide (droite) — normalisation thèse θf = −0.5.
pub const T_COLD: f64 = -0.5;
/// Écart de température ΔT = T_hot − T_cold = 1 (crate uniforme).
pub const DELTA_T: f64 = T_HOT - T_COLD;

/// Garde-fou de stabilité (identique crate uniforme) : abort si max_u dépasse
/// cette valeur (Ma_local = max_u·√3 ≈ 0.17 à la limite, marge ×3).
pub const MAX_U_ABORT: f64 = 0.10;

/// Vitesse max cible du calage ν (identique crate uniforme, --utarget 0.02).
pub const UTARGET: f64 = 0.02;
/// Constante de calage c ≈ 1.7 (crate uniforme : ν ≈ 0.0186 à Ra=1e6 validé).
pub const C_SCALE: f64 = 1.7;

// ---------------------------------------------------------------------------
// Collision TRT + Boussinesq Guo (formules EXACTES du crate uniforme)
// ---------------------------------------------------------------------------

/// Collision TRT f + BGK g avec force de Boussinesq Guo sur UN nœud — copie
/// EXACTE de `collide_node` du crate uniforme (solver.rs l.1054-1185). Les τ
/// effectifs (τ⁺, τ⁻, τ_t) viennent de `RelaxTimes` du niveau (τ⁺ = 0.5 +
/// 3·ν_lu LOCAL, §1.3) en mode OFF ; la force est stockée par nœud
/// (`fy_out`) pour la correction de vitesse de `update_macroscopic_guo`
/// (la F de la correction DOIT être la même que celle de la collision).
///
/// `nu_t_pre`/`alpha_t_pre` (mode échelles mixtes dynamiques, Étape D) :
/// si Some, ν_t/α_t sont pris tels quels (précalculés par
/// `smag::compute_dynamic_sgs_level`, §2.3.5 thèse) et les τ effectifs sont
/// recalculés PAR NŒUD — miroir du crate uniforme (solver.rs l.1118-1149) :
///   τ⁺ = 0.5 + 3·(ν_lu + ν_t), τ⁻ = 0.5 + Λ/(τ⁺−0.5),
///   τ_t = 0.5 + 3·(α_lu + α_t) (g reste BGK : τ_t⁻ = τ_t⁺) ;
/// les `RelaxTimes` fixés au niveau ne servent alors que de fallback. Si
/// None (mode OFF, défaut), les τ du niveau sont utilisés — chemin
/// historique BITWISE IDENTIQUE (flag OFF jamais touché).
#[allow(clippy::too_many_arguments)]
pub fn collide_node_cavity(
    f_node: &mut [f64; 9],
    g_node: &mut [f64; 9],
    fy_out: &mut f64,
    rho: f64,
    ux: f64,
    uy: f64,
    temp: f64,
    tau: &RelaxTimes,
    beta_eff: f64,
    nu_t_pre: Option<f64>,
    alpha_t_pre: Option<f64>,
) {
    // Force de Boussinesq (crate uniforme l.1078-1085) : fy = ρ·beta_eff·(T − t_ref),
    // t_ref = 0 (θ normalisé ±0.5), gravité +1 (« positive upward »).
    let fy = boussinesq_force(rho, beta_eff, 1.0, temp, 0.0);
    *fy_out = fy;

    // Équilibres des 9 directions (une seule fois par nœud).
    let mut f_eq = [0.0f64; 9];
    let mut g_eq = [0.0f64; 9];
    for i in 0..9 {
        f_eq[i] = feq(i, rho, ux, uy);
        g_eq[i] = geq(i, temp, ux, uy);
    }

    // Temps de relaxation TRT (Λ = 3/16 côté f ; BGK côté g).
    // Mode dynamique (Some) : τ effectifs par nœud — miroir du crate uniforme
    // l.1125-1144. Les RelaxTimes du niveau (fallback) ne servent que si
    // nu_t/alpha_t sont absents (mode OFF).
    let (tau_plus, tau_minus, tau_t) = match (nu_t_pre, alpha_t_pre) {
        (Some(nu_t), Some(alpha_t)) => {
            let tp = 0.5 + 3.0 * (tau.tau_plus - 0.5) / 3.0
                + 3.0 * nu_t; // = 0.5 + 3·(ν_lu + ν_t) (ν_lu = (tau.tau_plus−0.5)/3)
            let tm = 0.5 + crate::lbm::LAMBDA_TRT / (tp - 0.5);
            let tt = 0.5 + 3.0 * (tau.tau_t - 0.5) / 3.0
                + 3.0 * alpha_t; // = 0.5 + 3·(α_lu + α_t)
            (tp, tm, tt)
        }
        // Mode OFF : chemin actuel BITWISE IDENTIQUE.
        _ => (tau.tau_plus, tau.tau_minus, tau.tau_t),
    };
    // Coefficients TRT : a = (ω+ + ω−)/2, b = (ω+ − ω−)/2 (crate uniforme l.1146-1149).
    let a_f = 0.5 * (1.0 / tau_plus + 1.0 / tau_minus);
    let b_f = 0.5 * (1.0 / tau_plus - 1.0 / tau_minus);
    // Scalaire thermique BGK (τ_t⁻ = τ_t⁺) : a_g = ω_t, b_g = 0.
    let a_g = 1.0 / tau_t;
    let b_g = 0.0;

    // Déviations pré-collision (Σ f' = Σ f + Σ S_i, et Σ S_i = 0 : le moment
    // d'ordre 3 de D2Q9 est nul ⇒ masse conservée, testée en V2).
    let mut d_f = [0.0f64; 9];
    let mut d_g = [0.0f64; 9];
    for i in 0..9 {
        d_f[i] = f_node[i] - f_eq[i];
        d_g[i] = g_node[i] - g_eq[i];
    }

    // Collision f TRT avec force (Guo, Shi & Zheng 2002, éq. (14)-(16)) —
    // formules EXACTES du crate uniforme (l.1159-1179) :
    //   S_i = w_i·[(e_i−u)/c_s² + (e_i·u)·e_i/c_s⁴]·F, F = (0, fy, 0).
    // Exactitude PAR MODE TRT : (1 − 1/(2τ⁻)) sur l'impaire (momentum),
    // (1 − 1/(2τ⁺)) sur la paire (stress).
    for i in 0..9 {
        let cx = D2Q9_VELOCITIES[i][0] as f64;
        let cy = D2Q9_VELOCITIES[i][1] as f64;
        let opp = D2Q9_OPPOSITES[i];
        let ei_u = cx * ux + cy * uy;
        let s_i = D2Q9_WEIGHTS[i] * ((cy - uy) / CS2 + ei_u * cy / (CS2 * CS2)) * fy;
        let s_minus = D2Q9_WEIGHTS[i] * (cy / CS2) * fy; // impair : porte le momentum
        let s_plus = s_i - s_minus;
        f_node[i] += -a_f * d_f[i] - b_f * d_f[opp]
            + (1.0 - 0.5 / tau_minus) * s_minus // exactitude momentum
            + (1.0 - 0.5 / tau_plus) * s_plus; // exactitude stress
    }
    // Collision g BGK.
    for i in 0..9 {
        let opp = D2Q9_OPPOSITES[i];
        g_node[i] += -a_g * d_g[i] - b_g * d_g[opp];
    }
}

/// Collision TRT + Boussinesq sur un niveau entier, fenêtre de nœuds
/// [ix0..ix1) × [iy0..iy1). Les grilles fines passent `interior_only = true`
/// : les nœuds fantômes sont EXCLUS de la collision — les distributions
/// injectées par le transfert d'interface (§1.2) sont utilisées telles quelles
/// par le streaming, sans relaxation supplémentaire (pratique multi-bloc
/// standard, Lagrava et al. 2015 ; une double relaxation des fantômes
/// polluerait le non-équilibre transféré).
///
/// `nu_t_pre`/`alpha_t_pre` (mode échelles mixtes dynamiques, Étape D) :
/// Option<&[f64]> par nœud — None = mode OFF (chemin inchangé, valeurs
/// factices jamais lées) ; Some = indexation par nœud k (précalculées par
/// `smag::compute_dynamic_sgs_level`), mêmes patterns rayon/mono.
/// `jobs > 1` : bandes de lignes rayon, opérations strictement par nœud
/// (aucune réduction flottante) ⇒ déterministe, bitwise identique à jobs = 1
/// (même pattern que le crate uniforme, solver.rs l.462-511).
pub fn collide_cavity(
    lvl: &mut LbmLevel,
    tau: &RelaxTimes,
    beta_eff: f64,
    fy_out: &mut [f64],
    interior_only: bool,
    jobs: usize,
    nu_t_pre: Option<&[f64]>,
    alpha_t_pre: Option<&[f64]>,
) {
    let nx = lvl.nx;
    let (ix0, ix1, iy0, iy1) = if interior_only {
        (1, nx - 1, 1, lvl.ny - 1)
    } else {
        (0, nx, 0, lvl.ny)
    };
    if jobs > 1 {
        // Scalaires copiés ligne par ligne AVANT la boucle parallèle (la
        // collision ne modifie pas rho/ux/uy/T) — pattern rayon sûr du crate
        // uniforme, zéro unsafe, régions d'écriture disjointes.
        let scalars: Vec<(f64, f64, f64, f64)> = (0..lvl.rho.len())
            .map(|k| (lvl.rho[k], lvl.ux[k], lvl.uy[k], lvl.temp[k]))
            .collect();
        let f_rows: Vec<&mut [[f64; 9]]> = lvl.f.chunks_mut(nx).collect();
        let g_rows: Vec<&mut [[f64; 9]]> = lvl.g.chunks_mut(nx).collect();
        let fy_rows: Vec<&mut [f64]> = fy_out.chunks_mut(nx).collect();
        f_rows
            .into_par_iter()
            .zip(g_rows)
            .zip(fy_rows)
            .enumerate()
            .for_each(|(iy, ((f_row, g_row), fy_row))| {
                if iy < iy0 || iy >= iy1 {
                    return;
                }
                for ix in ix0..ix1 {
                    let k = iy * nx + ix;
                    let (rho, ux, uy, temp) = scalars[k];
                    collide_node_cavity(
                        &mut f_row[ix],
                        &mut g_row[ix],
                        &mut fy_row[ix],
                        rho,
                        ux,
                        uy,
                        temp,
                        tau,
                        beta_eff,
                        nu_t_pre.map(|v| v[k]),
                        alpha_t_pre.map(|v| v[k]),
                    );
                }
            });
        return;
    }
    // Chemin mono-thread.
    for iy in iy0..iy1 {
        for ix in ix0..ix1 {
            let k = iy * nx + ix;
            let rho = lvl.rho[k];
            let ux = lvl.ux[k];
            let uy = lvl.uy[k];
            let temp = lvl.temp[k];
            collide_node_cavity(
                &mut lvl.f[k],
                &mut lvl.g[k],
                &mut fy_out[k],
                rho,
                ux,
                uy,
                temp,
                tau,
                beta_eff,
                nu_t_pre.map(|v| v[k]),
                alpha_t_pre.map(|v| v[k]),
            );
        }
    }
}

/// Mise à jour macroscopique avec correction de vitesse de Guo et al. 2002,
/// éq. (16) — copie EXACTE du crate uniforme (solver.rs l.295-366) :
/// u = Σ e_i·f_i/ρ + F·Δt/2ρ (Δt = 1), F = (0, fy, 0) lue depuis `fy`
/// (stockée par la DERNIÈRE collision, évaluée aux champs du temps t).
/// Ajout strictement par nœud ⇒ déterministe quel que soit --jobs.
pub fn update_macroscopic_guo(lvl: &mut LbmLevel, fy: &[f64], jobs: usize) {
    let nx = lvl.nx;
    if jobs > 1 {
        let rho_rows: Vec<&mut [f64]> = lvl.rho.chunks_mut(nx).collect();
        let ux_rows: Vec<&mut [f64]> = lvl.ux.chunks_mut(nx).collect();
        let uy_rows: Vec<&mut [f64]> = lvl.uy.chunks_mut(nx).collect();
        let t_rows: Vec<&mut [f64]> = lvl.temp.chunks_mut(nx).collect();
        let f_rows: Vec<&[[f64; 9]]> = lvl.f.chunks(nx).collect();
        let g_rows: Vec<&[[f64; 9]]> = lvl.g.chunks(nx).collect();
        let fy_rows: Vec<&[f64]> = fy.chunks(nx).collect();
        rho_rows
            .into_par_iter()
            .zip(ux_rows)
            .zip(uy_rows)
            .zip(t_rows)
            .zip(f_rows)
            .zip(g_rows)
            .zip(fy_rows)
            .for_each(|((((((r_row, u_row), v_row), t_row), f_row), g_row), fy_row)| {
                for n in 0..nx {
                    let mut r = 0.0;
                    let mut un = 0.0;
                    let mut vn = 0.0;
                    let mut tn = 0.0;
                    for (j, (&fj, &gj)) in f_row[n].iter().zip(g_row[n].iter()).enumerate() {
                        r += fj;
                        un += D2Q9_VELOCITIES[j][0] as f64 * fj;
                        vn += D2Q9_VELOCITIES[j][1] as f64 * fj;
                        tn += gj;
                    }
                    // Correction de Guo éq. (16) : même ordre d'opérations que
                    // le chemin crate (vn/r puis ajout de fy/(2ρ) après coup).
                    let f = fy_row[n];
                    if r > f64::EPSILON {
                        un /= r;
                        vn /= r;
                        vn += 0.5 * f / r;
                        u_row[n] = un;
                        v_row[n] = vn;
                    } else {
                        u_row[n] = 0.0;
                        v_row[n] = 0.0;
                    }
                    r_row[n] = r;
                    t_row[n] = tn;
                }
            });
        return;
    }
    // Chemin mono-thread (identique crate uniforme l.358-365).
    for k in 0..nx * lvl.ny {
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
            // Correction Guo éq. (16) : u_y ← u_y + fy/(2ρ).
            lvl.uy[k] = my / rho + 0.5 * fy[k] / rho;
        } else {
            lvl.ux[k] = 0.0;
            lvl.uy[k] = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming + conditions aux limites (copie crate uniforme)
// ---------------------------------------------------------------------------

/// Streaming pull avec bounce-back, variante --jobs > 1 (bandes de lignes
/// rayon, régions d'écriture disjointes, lecture seule des instantanés
/// clonés — même pattern que le crate uniforme l.844-886). jobs = 1 :
/// délègue au chemin mono-thread V1 `lbm::stream` (strictement identique).
pub fn stream_cavity(lvl: &mut LbmLevel, jobs: usize) {
    if jobs > 1 {
        let nx = lvl.nx;
        let ny = lvl.ny;
        let f_old = lvl.f.clone();
        let g_old = lvl.g.clone();
        let f_rows: Vec<&mut [[f64; 9]]> = lvl.f.chunks_mut(nx).collect();
        let g_rows: Vec<&mut [[f64; 9]]> = lvl.g.chunks_mut(nx).collect();
        f_rows
            .into_par_iter()
            .zip(g_rows)
            .enumerate()
            .for_each(|(iy, (f_row, g_row))| {
                for ix in 0..nx {
                    let k = iy * nx + ix;
                    for i in 0..9 {
                        let cx = D2Q9_VELOCITIES[i][0] as i64;
                        let cy = D2Q9_VELOCITIES[i][1] as i64;
                        let sx = ix as i64 - cx;
                        let sy = iy as i64 - cy;
                        if sx >= 0 && sx < nx as i64 && sy >= 0 && sy < ny as i64 {
                            let src = (sy as usize) * nx + sx as usize;
                            f_row[ix][i] = f_old[src][i];
                            g_row[ix][i] = g_old[src][i];
                        } else {
                            // Bounce-back : source hors domaine (paroi).
                            f_row[ix][i] = f_old[k][D2Q9_OPPOSITES[i]];
                            // g : valeur d'avant streaming, CL thermiques ensuite.
                            g_row[ix][i] = g_old[k][i];
                        }
                    }
                }
            });
        return;
    }
    crate::lbm::stream(lvl);
}

/// Streaming pull block-aware pour un bloc fin de cavité (Étape B).
///
/// - Les couches fantômes (bord de la grille fine) ne sont PAS streamées :
///   elles portent l'injection §1.2 du couplage (côté interface) et sont
///   recalculées à chaque couplage grossier→fin — les écraser par un
///   streaming les polluerait avec les valeurs intérieures.
/// - Nœud intérieur : pull normal. Si la source tombe dans une couche
///   fantôme CÔTÉ PAROI physique (hors domaine, `wall_*`), bounce-back
///   mid-link EXACT comme le crate uniforme (f ← f*_post-collision(opp),
///   solver.rs l.838-900) — le no-slip des parois fines est alors identique
///   à celui du niveau grossier. (Sans cela, la paroi fine tirerait ses
///   distributions entrantes d'une extrapolation grossière hors domaine :
///   momentum parasite ≈ −u_intérieur/2, no-slip corrompu.)
///   Si la source tombe dans une couche fantôme côté INTERFACE (dans le
///   domaine physique), pull de l'injection §1.2 (couplage).
/// - g : idem (bounce-back garde la valeur pré-streaming, CL thermiques
///   appliquées ensuite par `apply_thermal_boundary_conditions_block`).
///
/// `jobs > 1` : bandes de lignes rayon (régions d'écriture disjointes,
/// instantanés clonés — même pattern que `stream_cavity`).
pub fn stream_cavity_block(
    lvl: &mut LbmLevel,
    jobs: usize,
    wall_left: bool,
    wall_right: bool,
    wall_bottom: bool,
    wall_top: bool,
) {
    let nx = lvl.nx;
    let ny = lvl.ny;
    let f_old = lvl.f.clone();
    let g_old = lvl.g.clone();

    /// Stream une ligne intérieure (helpers communs mono/rayon). Les sources
    /// d'un nœud intérieur (a ∈ [1, nx−2], b ∈ [1, ny−2], |c| ≤ 1) tombent
    /// toujours dans la grille fine [0, nx−1] × [0, ny−1].
    #[inline]
    fn stream_inner_row(
        f_row: &mut [[f64; 9]],
        g_row: &mut [[f64; 9]],
        f_old: &[[f64; 9]],
        g_old: &[[f64; 9]],
        iy: usize,
        nx: usize,
        ny: usize,
        wall_left: bool,
        wall_right: bool,
        wall_bottom: bool,
        wall_top: bool,
    ) {
        for ix in 1..nx - 1 {
            let k = iy * nx + ix;
            for i in 0..9 {
                let cx = D2Q9_VELOCITIES[i][0] as i64;
                let cy = D2Q9_VELOCITIES[i][1] as i64;
                let sx = ix as i64 - cx;
                let sy = iy as i64 - cy;
                let wall_src = (sx == 0 && wall_left)
                    || (sx == nx as i64 - 1 && wall_right)
                    || (sy == 0 && wall_bottom)
                    || (sy == ny as i64 - 1 && wall_top);
                if wall_src {
                    // Bounce-back mid-link (crate uniforme) : réflexion de la
                    // distribution sortante post-collision ; g garde sa valeur
                    // pré-streaming (CL thermiques ensuite).
                    f_row[ix][i] = f_old[k][D2Q9_OPPOSITES[i]];
                    g_row[ix][i] = g_old[k][i];
                } else {
                    // Pull normal (intérieur ou injection §1.2 côté interface).
                    let src = (sy as usize) * nx + sx as usize;
                    f_row[ix][i] = f_old[src][i];
                    g_row[ix][i] = g_old[src][i];
                }
            }
        }
    }

    if jobs > 1 {
        let f_rows: Vec<&mut [[f64; 9]]> = lvl.f.chunks_mut(nx).collect();
        let g_rows: Vec<&mut [[f64; 9]]> = lvl.g.chunks_mut(nx).collect();
        f_rows
            .into_par_iter()
            .zip(g_rows)
            .enumerate()
            .for_each(|(iy, (f_row, g_row))| {
                if iy == 0 || iy + 1 == ny {
                    return; // couches fantômes : non streamées
                }
                stream_inner_row(
                    f_row,
                    g_row,
                    &f_old,
                    &g_old,
                    iy,
                    nx,
                    ny,
                    wall_left,
                    wall_right,
                    wall_bottom,
                    wall_top,
                );
            });
        return;
    }
    // Chemin mono-thread.
    for iy in 1..ny - 1 {
        let f_row = &mut lvl.f[iy * nx..(iy + 1) * nx];
        let g_row = &mut lvl.g[iy * nx..(iy + 1) * nx];
        stream_inner_row(
            f_row,
            g_row,
            &f_old,
            &g_old,
            iy,
            nx,
            ny,
            wall_left,
            wall_right,
            wall_bottom,
            wall_top,
        );
    }
}

/// Conditions aux limites thermiques (après streaming) — copie EXACTE du
/// crate uniforme (solver.rs l.951-988) : parois verticales isothermes
/// (g paroi = geq(T_wall, u=0)), horizontales adiabatiques (zero-gradient,
/// copie du nœud intérieur adjacent).
pub fn apply_thermal_boundary_conditions(lvl: &mut LbmLevel) {
    let nx = lvl.nx;
    let ny = lvl.ny;
    // Parois verticales : chaude à gauche, froide à droite (isotherme).
    for iy in 0..ny {
        {
            let k = iy * nx;
            for i in 0..9 {
                lvl.g[k][i] = geq(i, T_HOT, 0.0, 0.0);
            }
        }
        {
            let k = iy * nx + (nx - 1);
            for i in 0..9 {
                lvl.g[k][i] = geq(i, T_COLD, 0.0, 0.0);
            }
        }
    }
    // Parois horizontales : adiabatiques (zero-gradient ⇒ ∂T/∂y = 0).
    for ix in 0..nx {
        {
            let k = ix;
            let k_in = nx + ix;
            let row_bottom = lvl.g[k_in];
            lvl.g[k].copy_from_slice(&row_bottom);
        }
        {
            let k = (ny - 1) * nx + ix;
            let k_in = (ny - 2) * nx + ix;
            let row_top = lvl.g[k_in];
            lvl.g[k].copy_from_slice(&row_top);
        }
    }
}

/// Variante block-aware (Étape B, blocs fins collés aux parois verticales) :
/// les colonnes fines a = 0 et a = nx1−1 sont des couches fantômes HORS du
/// domaine physique — la paroi isotherme fine est en a = 1 (chaude) /
/// a = nx1−2 (froide) ; les parois horizontales (adiabatiques) sont aux
/// lignes b = 1 et b = ny1−2. Les couches fantômes (a=0, a=nx1−1, b=0,
/// b=ny1−1) reçoivent le zero-gradient depuis l'intérieur — elles sont de
/// toute façon écrasées par le couplage §1.2 à chaque pas.
pub fn apply_thermal_boundary_conditions_block(
    lvl: &mut LbmLevel,
    hot_wall: bool,
    cold_wall: bool,
) {
    let nx = lvl.nx;
    let ny = lvl.ny;
    // Parois verticales isothermes : colonne fine intérieure adjacente au mur.
    let cols: Vec<(usize, f64)> = {
        let mut c = Vec::new();
        if hot_wall {
            c.push((1, T_HOT));
        }
        if cold_wall {
            c.push((nx - 2, T_COLD));
        }
        c
    };
    for iy in 0..ny {
        for &(ix, tw) in &cols {
            let k = iy * nx + ix;
            for i in 0..9 {
                lvl.g[k][i] = geq(i, tw, 0.0, 0.0);
            }
        }
    }
    // Parois horizontales adiabatiques : zero-gradient (lignes b=1, b=ny1−2).
    for ix in 1..nx - 1 {
        {
            let k = nx + ix;
            let k_in = 2 * nx + ix;
            let row = lvl.g[k_in];
            lvl.g[k].copy_from_slice(&row);
        }
        {
            let k = (ny - 2) * nx + ix;
            let k_in = (ny - 3) * nx + ix;
            let row = lvl.g[k_in];
            lvl.g[k].copy_from_slice(&row);
        }
    }
    // Couches fantômes : zero-gradient depuis la couche intérieure adjacente
    // (écrasées au couplage §1.2, valeurs lissées pour le monitoring).
    for iy in 0..ny {
        for &ix in &[0usize, nx - 1] {
            let ix_in = if ix == 0 { 1 } else { nx - 2 };
            let k = iy * nx + ix;
            let k_in = iy * nx + ix_in;
            let row = lvl.g[k_in];
            lvl.g[k].copy_from_slice(&row);
        }
    }
    for ix in 0..nx {
        let k = ix;
        let k_in = nx + ix;
        let row = lvl.g[k_in];
        lvl.g[k].copy_from_slice(&row);
        let k = (ny - 1) * nx + ix;
        let k_in = (ny - 2) * nx + ix;
        let row = lvl.g[k_in];
        lvl.g[k].copy_from_slice(&row);
    }
}

/// Force u = 0 aux nœuds de paroi (no-slip) — copie du crate uniforme
/// (solver.rs l.992-1009).
pub fn enforce_wall_velocity(lvl: &mut LbmLevel) {
    let nx = lvl.nx;
    let ny = lvl.ny;
    for iy in 0..ny {
        for &ix in &[0usize, nx - 1] {
            let k = iy * nx + ix;
            lvl.ux[k] = 0.0;
            lvl.uy[k] = 0.0;
        }
    }
    for &iy in &[0usize, ny - 1] {
        for ix in 0..nx {
            let k = iy * nx + ix;
            lvl.ux[k] = 0.0;
            lvl.uy[k] = 0.0;
        }
    }
}

/// Variante block-aware (Étape B) : no-slip sur la paroi FINE physique
/// (colonne a=1 / a=nx1−2 et lignes b=1 / b=ny1−2) — les couches fantômes
/// sont écrasées par le couplage §1.2, on force quand même u = 0 pour la
/// cohérence du monitoring.
pub fn enforce_wall_velocity_block(lvl: &mut LbmLevel, hot_wall: bool, cold_wall: bool) {
    let nx = lvl.nx;
    let ny = lvl.ny;
    let mut cols: Vec<usize> = Vec::new();
    if hot_wall {
        cols.push(1);
    }
    if cold_wall {
        cols.push(nx - 2);
    }
    for iy in 0..ny {
        for &ix in &cols {
            let k = iy * nx + ix;
            lvl.ux[k] = 0.0;
            lvl.uy[k] = 0.0;
        }
    }
    for &iy in &[1usize, ny - 2] {
        for ix in 0..nx {
            let k = iy * nx + ix;
            lvl.ux[k] = 0.0;
            lvl.uy[k] = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Post-traitement : Nusselt (formules EXACTES du crate uniforme, post.rs)
// ---------------------------------------------------------------------------

/// Nusselt paroi chaude : moyenne sur iy de |dθ/dx|·L/ΔT, gradient entre les
/// nœuds ix=1 et ix=2 (DEUXIÈME liaison) — estimateur CALÉ SUR LA RUN DE
/// RÉFÉRENCE `output/runs_octree/ra1e6_full` : son Nuc(0) = 0.9995 ≈ 1 (état conductif)
/// prouve l'échelle Lx = nx−1 (largeur), l'échelle Lz donnerait 4.05 et un
/// Nuc stationnaire 1.3660 < 4.05 serait impossible (Nu paroi ≥ Nu conductif).
/// Le crate uniforme actuel (post.rs) est passé à Lz APRÈS la run de
/// référence — pour la porte V2 (écart ≤ 5 % vs 1.3660), c'est l'estimateur
/// de la référence qu'il faut répliquer. Sur l'octree, les nœuds grossiers
/// ix=1..2 portent les valeurs injectées du niveau fin (moyenne 2×2 centrée
/// au même x) ⇒ même estimateur du gradient physique, champ sous-jacent fin.
pub fn nusselt_hot(coarse: &LbmLevel) -> f64 {
    let nx = coarse.nx;
    let ny = coarse.ny;
    let mut sum = 0.0;
    for iy in 0..ny {
        let t1 = coarse.temp[iy * nx + 1];
        let t2 = coarse.temp[iy * nx + 2];
        sum += (t1 - t2).abs(); // la paroi chaude est la plus chaude
    }
    let avg_grad = sum / ny as f64;
    avg_grad * (nx - 1) as f64 / DELTA_T
}

/// Nusselt paroi chaude mesuré sur la PREMIÈRE liaison (ix=0–1) — diagnostic
/// uniquement (artefact mur, cf. crate uniforme `nusselt_hot_wall_link`) :
/// permet la comparaison like-for-like avec le Nuc=1.3660 reporté par la run
/// uniforme de référence (qui utilise cet estimateur).
pub fn nusselt_hot_link0(coarse: &LbmLevel) -> f64 {
    let nx = coarse.nx;
    let ny = coarse.ny;
    let mut sum = 0.0;
    for iy in 0..ny {
        let t0 = coarse.temp[iy * nx];
        let t1 = coarse.temp[iy * nx + 1];
        sum += (t1 - t0).abs();
    }
    let avg_grad = sum / ny as f64;
    avg_grad * (nx - 1) as f64 / DELTA_T
}

/// Nusselt paroi froide (gradient entre ix=nx-3 et ix=nx-2, 2ᵉ liaison) —
/// même échelle Lx que `nusselt_hot` (estimateur de la run de référence).
pub fn nusselt_cold(coarse: &LbmLevel) -> f64 {
    let nx = coarse.nx;
    let ny = coarse.ny;
    let mut sum = 0.0;
    for iy in 0..ny {
        let t0 = coarse.temp[iy * nx + (nx - 3)];
        let t1 = coarse.temp[iy * nx + (nx - 2)];
        sum += (t1 - t0).abs();
    }
    let avg_grad = sum / ny as f64;
    avg_grad * (nx - 1) as f64 / DELTA_T
}

/// Nusselt paroi chaude mesuré DIRECTEMENT sur le champ fin (diagnostic V2) :
/// gradient entre la colonne fine de mur (a=1, x=0) et la colonne fine
/// suivante (a=2, x=0.5), échelle physique `lx_coarse` = nx0−1 passée
/// explicitement (le bloc fin ne couvre que la BL, pas le domaine entier).
/// Vérification conductive : T chute de ΔT sur lx_coarse cellules
/// grossières ⇒ |T(a=1)−T(a=2)| = ΔT·0.5/lx_coarse (liaison 0.5 cellule
/// fine), ×2 ⇒ ΔT/lx_coarse, ×lx_coarse/ΔT ⇒ Nu = 1 exact.
/// Compare le gradient PHYSIQUE fin au gradient lu sur le champ grossier
/// injecté : si Nu_fin ≈ Nu_grossier, la sur-prédiction est un vrai champ
/// (solution) ; si Nu_fin ≪ Nu_grossier, c'est le transfert 2×2 qui corrompt
/// l'estimateur grossier.
pub fn nusselt_hot_fine(fine: &LbmLevel, lx_coarse: f64) -> f64 {
    let nx = fine.nx;
    let ny = fine.ny;
    let mut sum = 0.0;
    for b in 1..ny - 1 {
        // Mur fin a=1 (x=0), colonne suivante a=2 (x=0.5) : liaison 0.5.
        let t1 = fine.temp[b * nx + 1];
        let t2 = fine.temp[b * nx + 2];
        sum += (t1 - t2).abs() * 2.0; // ×2 : la liaison vaut 0.5 cellule fine
    }
    let avg_grad = sum / (ny - 2) as f64;
    avg_grad * lx_coarse / DELTA_T
}

/// Vitesse max du champ (diagnostic de stabilité, crate uniforme `post::max_uy`).
pub fn max_u(lvl: &LbmLevel) -> f64 {
    lvl.ux
        .iter()
        .zip(lvl.uy.iter())
        .fold(0.0f64, |m, (&x, &v)| m.max(x.abs().max(v.abs())))
}

/// Tous les champs finis (garde-fou divergence, crate uniforme `post::check_finite`).
pub fn check_finite_levels(lvls: &[&LbmLevel]) -> bool {
    for lvl in lvls {
        for v in &lvl.rho {
            if !v.is_finite() {
                return false;
            }
        }
        for v in &lvl.ux {
            if !v.is_finite() {
                return false;
            }
        }
        for v in &lvl.uy {
            if !v.is_finite() {
                return false;
            }
        }
        for v in &lvl.temp {
            if !v.is_finite() {
                return false;
            }
        }
        for row in &lvl.f {
            for v in row {
                if !v.is_finite() {
                    return false;
                }
            }
        }
        for row in &lvl.g {
            for v in row {
                if !v.is_finite() {
                    return false;
                }
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Monitoring bruit acoustique (LA porte V2, plan §3.1)
// ---------------------------------------------------------------------------

/// Statistiques de bruit acoustique d'un pas (porte V2, plan §3.1 : « l'interface
/// ne génère pas de bruit acoustique — max|ρ−ρ̄| aux interfaces ≤ bruit de la
/// run uniforme même durée »).
#[derive(Clone, Copy, Debug)]
pub struct NoiseStats {
    /// max|ρ−ρ̄| global (tous les nœuds des deux niveaux), ρ̄ = moyenne grossière.
    pub max_rho_dev: f64,
    /// max|ρ−ρ̄| aux interfaces : nœuds fantômes fins (coïncidents exclus,
    /// injectés par le transfert §1.2) + nœuds grossiers coïncidents.
    pub max_rho_dev_ifaces: f64,
    /// Masse moyenne Σρ/N sur le niveau grossier (métrique du crate uniforme).
    pub masse: f64,
}

// ---------------------------------------------------------------------------
// PRNG xorshift64* — copie EXACTE du crate uniforme (src/rng.rs), perturbation
// initiale de température déterministe (seed 42 comme les runs de référence).
// ---------------------------------------------------------------------------

/// PRNG xorshift64* déterministe (aucune dépendance).
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Crée le générateur avec une graine non nulle.
    pub fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    /// Prochain entier 64 bits.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Prochain flottant dans [0, 1).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}
