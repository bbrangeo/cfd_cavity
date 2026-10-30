//! Post-traitement : Nusselt, profils w, fonction de courant ψ, moyennes
//! temporelles, exports CSV / VTK / JSON.

use std::fs;
use std::io::Write;
use std::path::Path;

use crate::solver::{CavitySolver, DELTA_T};

/// Accumulateur de moyennes temporelles (champs et diagnostics).
pub struct TimeAverager {
    /// Somme des champs (indexés k = iy*nx + ix).
    pub sum_ux: Vec<f64>,
    pub sum_uy: Vec<f64>,
    pub sum_t: Vec<f64>,
    pub sum_nu_t: Vec<f64>,
    /// Somme des carrés (pour S^1/2, écart-type vitesse).
    pub sum_uu: Vec<f64>,
    /// Nombre d'échantillons.
    pub n: u64,
}

impl TimeAverager {
    /// Crée un accumulateur zéro.
    pub fn new(n: usize) -> Self {
        Self {
            sum_ux: vec![0.0; n],
            sum_uy: vec![0.0; n],
            sum_t: vec![0.0; n],
            sum_nu_t: vec![0.0; n],
            sum_uu: vec![0.0; n],
            n: 0,
        }
    }

    /// Ajoute un échantillon instantané.
    pub fn accumulate(&mut self, s: &CavitySolver) {
        let n_nodes = s.nx() * s.ny();
        for k in 0..n_nodes {
            let ux = s.lbm.ux[k];
            let uy = s.lbm.uy[k];
            self.sum_ux[k] += ux;
            self.sum_uy[k] += uy;
            self.sum_t[k] += s.lbm.temperature[k];
            self.sum_nu_t[k] += s.nu_t[k];
            self.sum_uu[k] += ux * ux + uy * uy;
        }
        self.n += 1;
    }

    /// Champs moyens.
    pub fn means(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let inv = 1.0 / self.n as f64;
        let mean = |s: &Vec<f64>| s.iter().map(|v| v * inv).collect();
        (mean(&self.sum_ux), mean(&self.sum_uy), mean(&self.sum_t), mean(&self.sum_nu_t), mean(&self.sum_uu))
    }
}

/// Nusselt paroi chaude : moyenne sur iy de |dθ/dx|·Lz/ΔT (gradient entre
/// les nœuds ix=1 et ix=2 — DEUXIÈME liaison). Échelle de longueur = hauteur
/// Lz (adimensionnalisation thèse §2.3.6 : x* = x/H, Nu = ∂θ/∂n·H ; l'ancien
/// code utilisait Lx → biais ×Az).
///
/// Pourquoi la 2ᵉ liaison : même sain, la 1ʳᵉ liaison (ix=0–1) sous-estime le
/// gradient (artefact mur — nœud de paroi à équilibre imposé) ; la 2ᵉ liaison
/// est représentative du gradient intérieur. La 1ʳᵉ liaison reste accessible
/// via `nusselt_hot_wall_link` comme sentinelle de diagnostic mural.
pub fn nusselt_hot(s: &CavitySolver) -> f64 {
    let ny = s.ny();
    let mut sum = 0.0;
    for iy in 0..ny {
        let t1 = s.lbm.temperature[s.idx(1, iy)];
        let t2 = s.lbm.temperature[s.idx(2, iy)];
        sum += (t1 - t2).abs(); // la paroi chaude est la plus chaude
    }
    let avg_grad = sum / ny as f64;
    avg_grad * s.lz / DELTA_T
}

/// Sentinelle : Nusselt paroi chaude mesuré sur la PREMIÈRE liaison
/// (ix=0–1). Usage diagnostique uniquement — sous-estime le gradient même
/// sur un solveur sain (artefact mur). Après le patch TRT-thermique
/// (τ_t⁻ = τ_t⁺), ce ratio doit revenir proche du gradient intérieur ;
/// un ratio ≫ 1 signalait le biais Nuc ×3 (Λ=3/16 injustifié sur g).
#[allow(dead_code)]
pub fn nusselt_hot_wall_link(s: &CavitySolver) -> f64 {
    let ny = s.ny();
    let mut sum = 0.0;
    for iy in 0..ny {
        let t0 = s.lbm.temperature[s.idx(0, iy)];
        let t1 = s.lbm.temperature[s.idx(1, iy)];
        sum += (t0 - t1).abs();
    }
    let avg_grad = sum / ny as f64;
    avg_grad * s.lz / DELTA_T
}

/// Nusselt paroi froide (gradient entre ix=nx-3 et ix=nx-2 — DEUXIÈME
/// liaison, cf. nusselt_hot). Échelle de longueur = hauteur Lz.
pub fn nusselt_cold(s: &CavitySolver) -> f64 {
    let nx = s.nx();
    let ny = s.ny();
    let mut sum = 0.0;
    for iy in 0..ny {
        let t0 = s.lbm.temperature[s.idx(nx - 3, iy)];
        let t1 = s.lbm.temperature[s.idx(nx - 2, iy)];
        sum += (t1 - t0).abs();
    }
    let avg_grad = sum / ny as f64;
    avg_grad * s.lz / DELTA_T
}

/// Nusselt volumétrique (par intération, référence croisée) :
/// Nu = ⟨u_y·θ·Lz⟩_x + Lz/(ΔT)·⟨∂T/∂x⟩... version standard :
/// Nu_vol = 1 + ⟨u_y·θ⟩·Lz/ΔT... implémente la forme par flux convectif.
pub fn nusselt_volume(s: &CavitySolver) -> f64 {
    let nx = s.nx();
    let ny = s.ny();
    // Nu = (Lz/ΔT) · (1/Lx) · ∫ (⟨u_y θ⟩ − α ∂⟨θ⟩/∂y) dy ... version simplifiée
    // moyenne en x du flux vertical total :
    let mut sum = 0.0;
    for iy in 1..ny - 1 {
        for ix in 1..nx - 1 {
            let k = s.idx(ix, iy);
            let theta = s.lbm.temperature[k]; // déjà θ (±0.5)
            sum += s.lbm.uy[k] * theta;
        }
    }
    let conv = sum / (((nx - 2) * (ny - 2)) as f64);
    // Nu_conv = conv · Lz / ΔT ; Nu_total = Nu_conv + 1 (conduction de référence)
    1.0 + conv * s.lz / DELTA_T
}

/// Recherche du max de |w| (uy) sur une ligne horizontale iy donnée.
/// Retourne (w_max, x_normalisé 0..1).
#[allow(dead_code)]
pub fn max_uy_on_row(s: &CavitySolver, iy: usize) -> (f64, f64) {
    let nx = s.nx();
    let mut best = 0.0f64;
    let mut best_ix = 1usize;
    for ix in 1..nx - 1 {
        let uy = s.lbm.uy[s.idx(ix, iy)].abs();
        if uy > best {
            best = uy;
            best_ix = ix;
        }
    }
    let x_norm = best_ix as f64 / (nx - 1) as f64;
    (best, x_norm)
}

/// Vitesse verticale max du champ (diagnostic global).
pub fn max_uy(s: &CavitySolver) -> f64 {
    s.lbm.uy.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

/// Fonction de courant ψ par intégration de u_x : ψ(iy) = ψ(iy-1) − ∫u_x dx
/// (ψ constant aux parois horizontales ; Δψ = débit de la cellule convective).
/// Implémentation : pour chaque colonne x, intégrer u_x en y depuis le bas,
/// puis recentrer (ψ=0 sur la paroi basse). Retourne le champ ψ.
#[allow(dead_code)]
pub fn stream_function(s: &CavitySolver) -> Vec<f64> {
    let nx = s.nx();
    let ny = s.ny();
    let mut psi = vec![0.0f64; nx * ny];
    for ix in 0..nx {
        let mut acc = 0.0;
        psi[ix] = 0.0; // paroi basse : ψ = 0
        for iy in 1..ny {
            // intégration de u_x entre iy-1 et iy (trapèzes)
            let u0 = s.lbm.ux[s.idx(ix, iy - 1)];
            let u1 = s.lbm.ux[s.idx(ix, iy)];
            acc += 0.5 * (u0 + u1);
            psi[iy * nx + ix] = acc;
        }
    }
    psi
}

/// Contrainte de cisaillement paroi chaude (au nœud intérieur iy) :
/// τ_w = ν_eff · |∂u_y/∂x| (la vitesse principale le long de la paroi chaude est u_y).
pub fn wall_shear_hot(s: &CavitySolver, iy: usize) -> f64 {
    let _nx = s.nx();
    let nu_eff = s.nu + s.nu_t[s.idx(1, iy)];
    let du = (s.lbm.uy[s.idx(1, iy)] - s.lbm.uy[s.idx(0, iy)]).abs();
    nu_eff * du // ∂u/∂x sur Δx=1
}

/// y+ paroi chaude moyenné : u_τ = sqrt(τ_w/ρ), y+ = y1·u_τ/ν (y1 = 0.5, au demi-nœud).
pub fn y_plus_hot(s: &CavitySolver) -> f64 {
    let ny = s.ny();
    let mut sum = 0.0;
    for iy in 1..ny - 1 {
        let tau_w = wall_shear_hot(s, iy);
        let u_tau = (tau_w.max(0.0)).sqrt();
        sum += oxiphysics_lbm::turbulence::y_plus(0.5, u_tau, s.nu);
    }
    sum / (ny - 2) as f64
}

/// Export CSV des champs moyens : x, y, ux, uy, T, psi, nu_t.
pub fn write_fields_csv(
    path: &Path,
    s: &CavitySolver,
    ux: &[f64],
    uy: &[f64],
    t: &[f64],
    psi: &[f64],
    nu_t: &[f64],
) -> std::io::Result<()> {
    let nx = s.nx();
    let ny = s.ny();
    let mut f = fs::File::create(path)?;
    writeln!(f, "x,y,ux,uy,T,psi,nu_t")?;
    for iy in 0..ny {
        for ix in 0..nx {
            let k = iy * nx + ix;
            writeln!(
                f,
                "{:.6},{:.6},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e}",
                ix as f64,
                iy as f64,
                ux[k],
                uy[k],
                t[k],
                psi[k],
                nu_t[k]
            )?;
        }
    }
    Ok(())
}

/// Export CSV des profils w(z=0.5·Lz) et w(z=0.75·Lz).
pub fn write_profiles_csv(path: &Path, s: &CavitySolver, uy_mid: &[f64], uy_075: &[f64]) -> std::io::Result<()> {
    let nx = s.nx();
    let mut f = fs::File::create(path)?;
    writeln!(f, "x_norm,w_z05,w_z075")?;
    for ix in 0..nx {
        writeln!(
            f,
            "{:.6},{:.8e},{:.8e}",
            ix as f64 / (nx - 1) as f64,
            uy_mid[ix],
            uy_075[ix]
        )?;
    }
    Ok(())
}

/// Export snapshot VTK XML (StructuredPoints) : T, ux, uy.
pub fn write_vtk(path: &Path, s: &CavitySolver) -> std::io::Result<()> {
    let nx = s.nx();
    let ny = s.ny();
    let mut f = fs::File::create(path)?;
    writeln!(f, r#"<?xml version="1.0"?>"#)?;
    writeln!(
        f,
        r#"<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">"#
    )?;
    writeln!(
        f,
        r#"  <ImageData WholeExtent="0 {} 0 {} 0 0" Origin="0 0 0" Spacing="1 1 1">"#,
        nx - 1,
        ny - 1
    )?;
    writeln!(
        f,
        r#"    <Piece Extent="0 {} 0 {} 0 0">"#,
        nx - 1,
        ny - 1
    )?;
    // Convention VTK ImageData : les données de points sont ordonnées avec
    // x qui varie le PLUS VITE, puis y, puis z (index plat = ix + nx·iy).
    // Nos champs sont k = iy*nx + ix : la boucle (iy externe, ix interne)
    // ci-dessous écrit donc directement l'ordre VTK, sans réordonnancement.
    //
    // Lecture pyvista (dims = (nx, ny, 1)) :
    //   T = g["Temperature"].reshape((ny, nx), order="C")
    //   → T[iy, ix] correspond exactement au nœud solveur (ix, iy) ;
    //   T[:, 0]  = paroi GAUCHE ix=0 (chaude, +0.5) ;
    //   T[:, -1] = paroi DROITE ix=nx-1 (froide, −0.5).
    // NE PAS utiliser order="F" (ou reshape (nx, ny)) : la vue est transposée,
    // ce qui inverse apparent la polarité thermique et met Uy=0 (nœuds de
    // paroi) sur les lignes horizontales — cf. smoke Ra=1e6, erreur de
    // vérification, pas de physique.
    let mut scalars_t = String::with_capacity(nx * ny * 14);
    let mut scalars_ux = String::with_capacity(nx * ny * 14);
    let mut scalars_uy = String::with_capacity(nx * ny * 14);
    for iy in 0..ny {
        for ix in 0..nx {
            let k = iy * nx + ix;
            scalars_t.push_str(&format!("{:.6e} ", s.lbm.temperature[k]));
            scalars_ux.push_str(&format!("{:.6e} ", s.lbm.ux[k]));
            scalars_uy.push_str(&format!("{:.6e} ", s.lbm.uy[k]));
        }
        scalars_t.push('\n');
        scalars_ux.push('\n');
        scalars_uy.push('\n');
    }
    writeln!(f, r#"      <PointData Scalars="Temperature">"#)?;
    writeln!(f, r#"        <DataArray type="Float64" Name="Temperature" format="ascii">"#)?;
    writeln!(f, "{scalars_t}        </DataArray>")?;
    writeln!(f, r#"        <DataArray type="Float64" Name="Ux" format="ascii">"#)?;
    writeln!(f, "{scalars_ux}        </DataArray>")?;
    writeln!(f, r#"        <DataArray type="Float64" Name="Uy" format="ascii">"#)?;
    writeln!(f, "{scalars_uy}        </DataArray>")?;
    writeln!(f, r#"      </PointData>"#)?;
    writeln!(f, r#"      <CellData>"#)?;
    writeln!(f, r#"      </CellData>"#)?;
    writeln!(f, r#"    </Piece>"#)?;
    writeln!(f, r#"  </ImageData>"#)?;
    writeln!(f, r#"</VTKFile>"#)?;
    Ok(())
}

/// Résumé JSON final (à la main, sérialisation simple).
#[allow(clippy::too_many_arguments)]
pub fn write_summary_json(
    path: &Path,
    nuc: f64,
    nuf: f64,
    nu_vol: f64,
    wmax_mid: f64,
    x_mid: f64,
    wmax_075: f64,
    x_075: f64,
    yplus: f64,
    cs: f64,
    ra: f64,
    nx: usize,
    ny: usize,
    steps: usize,
    n_samples: u64,
) -> std::io::Result<()> {
    let mut f = fs::File::create(path)?;
    writeln!(
        f,
        r#"{{
  "case": "differentially_heated_cavity_2d",
  "ra": {ra:e},
  "cs": {cs},
  "grid": [{{"nx": {nx}, "ny": {ny}}}],
  "aspect_ratio": {:.4},
  "steps": {steps},
  "time_samples": {n_samples},
  "nusselt_hot": {nuc:.4},
  "nusselt_cold": {nuf:.4},
  "nusselt_volume": {nu_vol:.4},
  "wmax_z05": {wmax_mid:.5},
  "x_wmax_z05": {x_mid:.5},
  "wmax_z075": {wmax_075:.5},
  "x_wmax_z075": {x_075:.5},
  "y_plus_hot": {yplus:.3},
  "references_thesis_tab27": {{
    "nuc_snd": 102.0,
    "nuc_les_these": [100.2, 100.5],
    "wmax_z05": [0.22, 0.24],
    "x_wmax_z05": 0.004,
    "wmax_z075": [0.11, 0.17],
    "x_wmax_z075": 0.005,
    "s_half": [1.10, 1.20]
  }}
}}"#,
        (ny - 1) as f64 / (nx - 1) as f64,
    )?;
    Ok(())
}

/// Cible Nusselt Trias et al. 2007 (corrélation 2D, cavité différentiellement
/// chauffée, paroi verticale chaude) — remplace les cibles fixes "thèse
/// ≈ 100.2–100.5, SND ≈ 102" valables uniquement à Ra=1e10.
///
/// Points d'ancrage exacts : Nū = 49.23 à Ra=6.4e8, 66.19 à Ra=2e9,
/// 100.60 à Ra=1e10. Entre les ancres : interpolation log-log ;
/// hors bornes : fit `Nu = 0.2529·Ra^0.26` (ex. ~30.4 à Ra=1e8).
pub fn nusselt_trias_2d(ra: f64) -> f64 {
    // (Ra, Nū) — ordre croissant
    const ANCHORS: [(f64, f64); 3] = [(6.4e8, 49.23), (2e9, 66.19), (1e10, 100.60)];

    if ra <= ANCHORS[0].0 || ra >= ANCHORS[2].0 {
        return 0.2529 * ra.powf(0.26);
    }
    // Interpolation log-log sur le segment [ra_i, ra_{i+1}] contenant ra.
    for w in ANCHORS.windows(2) {
        let (ra_i, nu_i) = w[0];
        let (ra_j, nu_j) = w[1];
        if ra >= ra_i && ra <= ra_j {
            let t = (ra.ln() - ra_i.ln()) / (ra_j.ln() - ra_i.ln());
            let ln_nu = nu_i.ln() + t * (nu_j.ln() - nu_i.ln());
            return ln_nu.exp();
        }
    }
    unreachable!("ra dans [anchor0, anchor2] mais aucun segment trouvé")
}

/// Écrit un fichier texte simple (log).
pub fn write_log(path: &Path, lines: &[String]) -> std::io::Result<()> {    let mut f = fs::File::create(path)?;
    for l in lines {
        writeln!(f, "{l}")?;
    }
    Ok(())
}

/// Vérifie que toutes les valeurs des champs sont finies.
pub fn check_finite(s: &CavitySolver) -> bool {
    s.lbm.temperature.iter().all(|v| v.is_finite())
        && s.lbm.ux.iter().all(|v| v.is_finite())
        && s.lbm.uy.iter().all(|v| v.is_finite())
        && s.lbm.rho.iter().all(|v| v.is_finite() && *v > 0.0)
}
