//! Briques LBM D2Q9 double distribution (f vitesse + g scalaire T) — copie
//! adaptée du crate uniforme `cavity_les/src/solver.rs` (formules IDENTIQUES :
//! collision TRT Λ = 3/16, force Guo odd/even, streaming pull bounce-back).
//!
//! Étape A : pas de Smagorinsky, ν et α constants par niveau ⇒ τ⁺/τ⁻ constants
//! par niveau (τ⁺ = 0.5 + 3·ν_lu, τ⁻ = 0.5 + Λ/(τ⁺−0.5) côté f ; côté g BGK
//! τ_t⁻ = τ_t⁺ = 0.5 + 3·α_lu). Le rescaling d'interface (§1.2) utilise
//! exactement ces τ par niveau.

use oxiphysics_lbm::lattice::{D2Q9_OPPOSITES, D2Q9_VELOCITIES};
use oxiphysics_lbm::thermal_lbm::{feq, geq};

/// Paramètre TRT optimal pour bounce-back (identique crate uniforme).
pub const LAMBDA_TRT: f64 = 3.0 / 16.0;

/// Grille LBM d'un niveau : distributions + champs macroscopiques.
/// Convention d'indexation identique au crate uniforme : k = iy * nx + ix.
pub struct LbmLevel {
    pub nx: usize,
    pub ny: usize,
    /// Distribution vitesse f[i][k] (post-collision / post-streaming).
    pub f: Vec<[f64; 9]>,
    /// Distribution scalaire g[i][k].
    pub g: Vec<[f64; 9]>,
    /// ρ = Σf.
    pub rho: Vec<f64>,
    /// u = (Σef)/ρ (+ correction force Guo côté solver, Éq. (16)).
    pub ux: Vec<f64>,
    pub uy: Vec<f64>,
    /// T = Σg.
    pub temp: Vec<f64>,
}

impl LbmLevel {
    pub fn new(nx: usize, ny: usize) -> Self {
        let n = nx * ny;
        LbmLevel {
            nx,
            ny,
            f: vec![[0.0; 9]; n],
            g: vec![[0.0; 9]; n],
            rho: vec![0.0; n],
            ux: vec![0.0; n],
            uy: vec![0.0; n],
            temp: vec![0.0; n],
        }
    }

    #[inline]
    pub fn idx(&self, ix: usize, iy: usize) -> usize {
        iy * self.nx + ix
    }

    /// Recalcule ρ, u, T depuis les distributions (sans correction de force —
    /// la correction Guo u += F/2ρ est appliquée côté solver si besoin).
    pub fn update_macroscopic(&mut self) {
        for k in 0..self.nx * self.ny {
            let mut rho = 0.0;
            let mut mx = 0.0;
            let mut my = 0.0;
            let mut t = 0.0;
            for i in 0..9 {
                let fi = self.f[k][i];
                rho += fi;
                mx += fi * D2Q9_VELOCITIES[i][0] as f64;
                my += fi * D2Q9_VELOCITIES[i][1] as f64;
                t += self.g[k][i];
            }
            self.rho[k] = rho;
            self.ux[k] = mx / rho;
            self.uy[k] = my / rho;
            self.temp[k] = t;
        }
    }
}

/// Temps de relaxation effectifs TRT d'un niveau (ν_lu constant en Étape A).
/// Identique au crate uniforme : τ⁺ = 0.5 + 3·ν, τ⁻ = 0.5 + Λ/(τ⁺−0.5).
#[derive(Clone, Copy, Debug)]
pub struct RelaxTimes {
    /// τ⁺ : partie paire (stress).
    pub tau_plus: f64,
    /// τ⁻ : partie impaire (momentum).
    pub tau_minus: f64,
    /// τ scalaire thermique (BGK : τ_t⁻ = τ_t⁺).
    pub tau_t: f64,
}

impl RelaxTimes {
    pub fn from_nu_alpha(nu_lu: f64, alpha_lu: f64) -> Self {
        let tau_plus = 0.5 + 3.0 * nu_lu;
        let tau_minus = 0.5 + LAMBDA_TRT / (tau_plus - 0.5);
        RelaxTimes {
            tau_plus,
            tau_minus,
            tau_t: 0.5 + 3.0 * alpha_lu,
        }
    }
}

/// Collision TRT f + BGK g, un nœud — copie EXACTE de `collide_node` du crate
/// uniforme (solver.rs l.1054-1185) en mode ν/α constants (nu_t = 0) et sans
/// force (la force Guo est hors scope V1 : couette/gradient imposés).
/// Coefficients a = (ω⁺+ω⁻)/2, b = (ω⁺−ω⁻)/2 ; force Guo omise (fy = 0).
pub fn collide_node(f_node: &mut [f64; 9], g_node: &mut [f64; 9], rho: f64, ux: f64, uy: f64, temp: f64, tau: &RelaxTimes) {
    // Équilibres des 9 directions (une seule fois par nœud).
    let mut f_eq = [0.0f64; 9];
    let mut g_eq = [0.0f64; 9];
    for i in 0..9 {
        f_eq[i] = feq(i, rho, ux, uy);
        g_eq[i] = geq(i, temp, ux, uy);
    }

    // Temps de relaxation TRT (Λ = 3/16 côté f ; BGK côté g).
    let tau_plus = tau.tau_plus;
    let tau_minus = tau.tau_minus;
    let tau_t = tau.tau_t;

    // Coefficients TRT : a = (ω+ + ω−)/2, b = (ω+ − ω−)/2 (crate uniforme l.1146-1145).
    let a_f = 0.5 * (1.0 / tau_plus + 1.0 / tau_minus);
    let b_f = 0.5 * (1.0 / tau_plus - 1.0 / tau_minus);
    // Scalaire thermique BGK (τ_t⁻ = τ_t⁺) : a_g = ω_t, b_g = 0.
    let a_g = 1.0 / tau_t;
    let b_g = 0.0;

    // Déviations pré-collision (Σ f' = Σ f : masse conservée).
    let mut d_f = [0.0f64; 9];
    let mut d_g = [0.0f64; 9];
    for i in 0..9 {
        d_f[i] = f_node[i] - f_eq[i];
        d_g[i] = g_node[i] - g_eq[i];
    }

    // Collision f TRT (force Guo omise en V1 : pas de flottabilité dans les
    // tests porte ; les facteurs d'exactitude (1−1/2τ±) ne servent qu'avec F).
    for i in 0..9 {
        let opp = D2Q9_OPPOSITES[i];
        f_node[i] += -a_f * d_f[i] - b_f * d_f[opp];
    }
    // Collision g BGK.
    for i in 0..9 {
        let opp = D2Q9_OPPOSITES[i];
        g_node[i] += -a_g * d_g[i] - b_g * d_g[opp];
    }
}

/// Streaming pull standard avec bounce-back intégré, un niveau entier — copie
/// du chemin mono-thread du crate uniforme (solver.rs l.838-900) :
/// f[i](x) ← f[i](x − c_i) si la source est dans le domaine, sinon
/// f[i](x) ← f*(x, opp(i)) post-collision (réflexion de la distribution
/// sortante). Masse exactement conservée, no-slip au milieu du lien.
pub fn stream(lvl: &mut LbmLevel) {
    let nx = lvl.nx;
    let ny = lvl.ny;
    let f_old = lvl.f.clone();
    let g_old = lvl.g.clone();

    for iy in 0..ny {
        for ix in 0..nx {
            let k = iy * nx + ix;
            for i in 0..9 {
                let cx = D2Q9_VELOCITIES[i][0] as i64;
                let cy = D2Q9_VELOCITIES[i][1] as i64;
                let sx = ix as i64 - cx;
                let sy = iy as i64 - cy;
                if sx >= 0 && sx < nx as i64 && sy >= 0 && sy < ny as i64 {
                    let src = (sy as usize) * nx + sx as usize;
                    lvl.f[k][i] = f_old[src][i];
                    lvl.g[k][i] = g_old[src][i];
                } else {
                    // Bounce-back : source hors domaine (paroi).
                    lvl.f[k][i] = f_old[k][D2Q9_OPPOSITES[i]];
                    // g : valeur d'avant streaming (BC thermiques ensuite).
                    lvl.g[k][i] = g_old[k][i];
                }
            }
        }
    }
}

/// Collision TRT/BGK sur un niveau entier (chemin séquentiel, V1).
pub fn collide(lvl: &mut LbmLevel, tau: &RelaxTimes) {
    for k in 0..lvl.nx * lvl.ny {
        collide_node(
            &mut lvl.f[k],
            &mut lvl.g[k],
            lvl.rho[k],
            lvl.ux[k],
            lvl.uy[k],
            lvl.temp[k],
            tau,
        );
    }
}
