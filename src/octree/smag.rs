//! Modèle sous-maille d'échelles mixtes DYNAMIQUES multi-niveaux (Étape D,
//! porte V5 — plan PLAN_OCTREE_REFINEMENT.md §3.3 étape 4, §1.4).
//!
//! Port EXACT du crate uniforme validé `cfd_cavity/src/solver.rs`
//! (NE PAS modifier, plan §2 — c'est la référence absolue, pas de
//! reformulation) :
//! - `compute_dynamic_sgs` (solver.rs l.562-828) : filtre test box η = 2,
//!   tenseurs Germano M/L, régularisation de Lilly Éq. 2.62, formes mixtes
//!   Tab. 2.3 (α = 0.5 ⇒ moyenne géométrique), garde-fou plancher
//!   τ_trt_floor = 0.505 → ν_t ≥ max(nu_sm_min·ν, (0.505−0.5)/3 − ν) ;
//! - helpers `box_filter`, `ddx`, `ddy`, `lilly_coefficient`
//!   (solver.rs l.1204-1261) + const `C_DYN_MAX` = 0.04 (l.25) et
//!   `TAU_TRT_FLOOR` = 0.505 (l.31).
//!
//! Différence multi-niveaux (§1.4) : tout est calculé en unités lattice
//! LOCALES du niveau (Δx = 1) — c'est cohérent car τ_effectif local =
//! 0.5 + 3·(ν_lu + ν_t_lu) et chaque niveau a son ν_lu propre (×2 par
//! niveau, §1.3 : ν_lu,n = 2^(n−1)·ν_L0). [`compute_dynamic_sgs_level`]
//! remplace la méthode du crate uniforme : MÊMES formules (passe 1 filtre
//! box, passe 2 S̃ depuis π_neq + S̄ différences centrées, similarité
//! Éq. 2.74/2.75, passe 3 Germano/Lilly, passe 4 bornes), état lu depuis un
//! [`LbmLevel`].

use oxiphysics_lbm::lattice::{D2Q9_VELOCITIES, CS2};
use oxiphysics_lbm::thermal_lbm::feq;

use crate::octree::cavity::DELTA_T;
use crate::octree::lbm::LbmLevel;

/// Borne supérieure du coefficient dynamique C^ν_dyn / C^κ_dyn (clamp [0, 0.04],
/// comme la crate de référence oxiphysics-lbm : Cs²_max = 0.2²) — copie crate
/// uniforme (solver.rs l.25).
pub const C_DYN_MAX: f64 = 0.04;

/// Plancher TRT sur les temps de relaxation effectifs (τ⁺, τ_t⁺) — copie crate
/// uniforme (solver.rs l.31). En mode dynamique, les bornes sur ν_t/α_t
/// garantissent τ = 0.5 + 3·(ν + ν_t) ≥ 0.505 et τ_t = 0.5 + 3·(α + α_t) ≥
/// 0.505, quel que soit nu_sm_min.
pub const TAU_TRT_FLOOR: f64 = 0.505;

/// Facteur interpolé mixte (Tab. 2.3) : a^α · b^(1−α). α = 0.5 ⇒ moyenne
/// géométrique (les deux volets Smagorinsky × similarité). Copie crate
/// uniforme (solver.rs l.1195).
pub fn mixed_factor(a: f64, b: f64, alpha: f64) -> f64 {
    a.powf(alpha) * b.powf(1.0 - alpha)
}

/// Filtre box de largeur `width` nœuds (width = η·Δ̃/Δx, typ. 2 pour η = 2),
/// centré au plus près : width pair ⇒ fenêtre {ix, ix+1} (demi-nœud de
/// décalage, incontournable sur grille colocalisée), width impair ⇒ fenêtre
/// symétrique. Parois : clamp unilatéral (la fenêtre rétrécit vers
/// l'intérieur du domaine, aucune valeur de bord dupliquée).
/// Copie EXACTE du crate uniforme (solver.rs l.1204-1224).
pub fn box_filter(field: &[f64], nx: usize, ny: usize, width: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; nx * ny];
    let half_left = (width - 1) / 2; // nœuds avant ix
    for iy in 0..ny {
        let y0 = iy.saturating_sub(half_left);
        let y1 = (iy + width - half_left).min(ny); // exclusif
        for ix in 0..nx {
            let x0 = ix.saturating_sub(half_left);
            let x1 = (ix + width - half_left).min(nx);
            let mut s = 0.0;
            for yy in y0..y1 {
                for xx in x0..x1 {
                    s += field[yy * nx + xx];
                }
            }
            let cnt = ((y1 - y0) * (x1 - x0)) as f64;
            out[iy * nx + ix] = s / cnt;
        }
    }
    out
}

/// Dérivée centrale ∂/∂x d'un champ (pas Δx = 1) ; parois : clamp unilatéral
/// (différence unilatérale dégradée par duplication du nœud de paroi).
/// Copie EXACTE du crate uniforme (solver.rs l.1229-1234).
#[inline]
pub fn ddx(field: &[f64], nx: usize, ny: usize, ix: usize, iy: usize) -> f64 {
    let _ = ny;
    let xm = if ix == 0 { 0 } else { ix - 1 };
    let xp = if ix + 1 >= nx { nx - 1 } else { ix + 1 };
    0.5 * (field[iy * nx + xp] - field[iy * nx + xm])
}

/// Dérivée centrale ∂/∂y d'un champ (pas Δy = 1) ; parois : clamp unilatéral.
/// Copie EXACTE du crate uniforme (solver.rs l.1238-1242).
#[inline]
pub fn ddy(field: &[f64], nx: usize, ny: usize, ix: usize, iy: usize) -> f64 {
    let ym = if iy == 0 { 0 } else { iy - 1 };
    let yp = if iy + 1 >= ny { ny - 1 } else { iy + 1 };
    0.5 * (field[yp * nx + ix] - field[ym * nx + ix])
}

/// Coefficient dynamique de Lilly régularisé (Éq. 2.58/2.70 + régularisation
/// Éq. 2.62 côté appelant) : C = (1/2)·num/den.
///
/// Garde-fous (danger dénominateur, plan §7) : den ≈ 0 ou non fini ⇒ C = 0
/// (pas de division par zéro, pas de NaN) ; clamp [0, C_DYN_MAX] = [0, 0.04]
/// (comme la crate de référence, C ≥ 0 : pas de cascade inverse).
/// Copie EXACTE du crate uniforme (solver.rs l.1251-1261).
#[inline]
pub fn lilly_coefficient(num: f64, den: f64) -> f64 {
    if !den.is_finite() || !num.is_finite() || den <= 0.0 {
        return 0.0;
    }
    let c = 0.5 * num / den;
    if c.is_finite() {
        c.clamp(0.0, C_DYN_MAX)
    } else {
        0.0
    }
}

/// Modèle d'échelles mixtes dynamiques sur UN niveau de l'octree — port EXACT
/// de `compute_dynamic_sgs` du crate uniforme (solver.rs l.562-828), état lu
/// depuis [`LbmLevel`].
///
/// Arguments :
/// - `tau_mol` : τ moléculaire EFFECTIF du niveau = 0.5 + 3·ν_lu (le crate
///   uniforme utilise τ = 0.5 + 3·ν_eff du calage ; ici ν_lu est local au
///   niveau, §1.3, et le plancher ν ≥ 0.002 est déjà appliqué par le calage
///   de `CavitySolver::new` sur L0 — les niveaux fins ont ν_lu ×2/×4, le
///   plancher ne peut pas lier) ;
/// - `smag_alpha` : exposant α de l'interpolation mixte (0.5 = moyenne
///   géométrique) ;
/// - `eta_test` : ratio filtre test η = Δ/Δ̃ (≥ 1) ;
/// - `nu_lu`/`alpha_lu` : ν/α moléculaires en unités lattice LOCALES du
///   niveau (ν_L0, 2·ν_L0, 4·ν_L0 — §1.3) ;
/// - `nu_sm_min` : borne νsm ≥ nu_sm_min·ν (défaut thèse : −0.1).
///
/// Retour : (nu_t, alpha_t, cnu_mean, cnu_max, ck_mean, ck_max) — ν_t/α_t
/// par nœud (unités lattice du niveau, prêts pour `collide_node_cavity` via
/// nu_t_pre/alpha_t_pre), coefficients C^ν/C^κ moyen/max (diagnostics).
///
/// Tout est séquentiel (aucune réduction flottante parallèle) ⇒ résultats
/// déterministes quel que soit --jobs. Les lectures se font sur des
/// instantanés clonés AVANT la collision (f n'est pas encore modifié —
/// |S̃| est reconstruit depuis π_neq avec EXACTEMENT la formule de
/// `collide_node_cavity`), les écritures sur des Vec locaux transmis ensuite
/// à collide via nu_t_pre/alpha_t_pre.
#[allow(clippy::too_many_arguments)]
pub fn compute_dynamic_sgs_level(
    lvl: &LbmLevel,
    tau_mol: f64,
    smag_alpha: f64,
    eta_test: f64,
    nu_lu: f64,
    alpha_lu: f64,
    nu_sm_min: f64,
) -> (Vec<f64>, Vec<f64>, f64, f64, f64, f64) {
    let nx = lvl.nx;
    let ny = lvl.ny;
    let n = nx * ny;

    // τ moléculaire effectif du niveau (voir doc : 0.5 + 3·ν_lu local).
    let tau = tau_mol;

    let alpha_exp = smag_alpha; // exposant α (0.5 = moyenne géométrique)
    let eta = eta_test.max(1.0);
    let width = (eta.round() as usize).max(1); // largeur du filtre box (nœuds)
    let dx: f64 = 1.0; // Δ̃ = Δx (unités lattice LOCALES du niveau)
    let dtest = eta * dx; // Δ = η·Δ̃ (filtre test)
    // Écart de température GLOBAL ΔT = θc − θf = 1 (PAS Δ̃·T — Tab. 2.3).
    let delta_t = DELTA_T;
    let nu_mol = nu_lu;
    let alpha_mol = alpha_lu;

    // --- Instantanés immuables (la collision n'a pas encore modifié f) ---
    let rho = lvl.rho.clone();
    let ux = lvl.ux.clone();
    let uy = lvl.uy.clone();
    let temp = lvl.temp.clone();
    let f = lvl.f.clone();

    // --- Passe 1 : filtre test box (η·Δ̃) des champs et des produits ---
    // ũ_test = box-filter du champ résolu. Parois : clamp unilatéral
    // (la fenêtre rétrécit à l'intérieur du domaine, cf. box_filter).
    let ubx = box_filter(&ux, nx, ny, width);
    let uby = box_filter(&uy, nx, ny, width);
    let tb = box_filter(&temp, nx, ny, width);
    // ⟨u_i u_j⟩_test et ⟨u_j T⟩_test : box des produits des champs RÉSOLUS
    // (Léonard au filtre test, Éq. 2.51 / Tab. 2.3).
    let pxx: Vec<f64> = (0..n).map(|k| ux[k] * ux[k]).collect();
    let pxy: Vec<f64> = (0..n).map(|k| ux[k] * uy[k]).collect();
    let pyy: Vec<f64> = (0..n).map(|k| uy[k] * uy[k]).collect();
    let ptx: Vec<f64> = (0..n).map(|k| ux[k] * temp[k]).collect();
    let pty: Vec<f64> = (0..n).map(|k| uy[k] * temp[k]).collect();
    let lbuxx = box_filter(&pxx, nx, ny, width);
    let lbuxy = box_filter(&pxy, nx, ny, width);
    let lbuyy = box_filter(&pyy, nx, ny, width);
    let lbutx = box_filter(&ptx, nx, ny, width);
    let lbuty = box_filter(&pty, nx, ny, width);

    // --- Passe 2 : quantités aux deux niveaux (S, qsm, Φsm, |T|) ---
    // Grille : S̃ij et |S̃| depuis π_neq (MÊMES formules que collide_node) ;
    // test : S̄ij par différences centrées des champs box-filtrés.
    let mut shxx = vec![0.0f64; n];
    let mut shxy = vec![0.0f64; n];
    let mut shyy = vec![0.0f64; n];
    let mut smag_hat = vec![0.0f64; n]; // |S̃| = sqrt(π:π)/(2·cs²·τ·ρ)
    let mut sbxx = vec![0.0f64; n];
    let mut sbxy = vec![0.0f64; n];
    let mut sbyy = vec![0.0f64; n];
    let mut smag_bar = vec![0.0f64; n]; // |S̄| (même convention de contraction)
    let mut gtx = vec![0.0f64; n]; // ∂T̃/∂x (champ résolu, notation thèse)
    let mut gty = vec![0.0f64; n]; // ∂T̃/∂y
    let mut gbx = vec![0.0f64; n]; // ∂T̄/∂x (champ test-filtré)
    let mut gby = vec![0.0f64; n]; // ∂T̄/∂y
    let mut qsm_h = vec![0.0f64; n]; // (qsm)^{1/2} = (1/2)|ũ − ũ_test| (Éq. 2.74)
    let mut phi_h = vec![0.0f64; n]; // (Φsm)^{1/2} = (1/2)|ũ − ũ_test|·|T − T_test|
    let mut tmag_hat = vec![0.0f64; n]; // |T̃| = √(2 T̃i T̃i), T̃i = ∂T̃/∂xj S̃ij (Éq. 2.75)
    let mut tmag_bar = vec![0.0f64; n]; // |T̄| idem au filtre test (S̄, ∂T̄)

    for iy in 0..ny {
        for ix in 0..nx {
            let k = iy * nx + ix;
            let rho_k = rho[k];
            let ux_k = ux[k];
            let uy_k = uy[k];
            let t_k = temp[k];

            // --- S̃ij, |S̃| depuis π_neq : EXACTEMENT les formules de
            // collide_node (|S| = sqrt(π:π)/(2·cs²·τ·ρ), π:π = Πxx²+2Πxy²+Πyy²)
            // et S̃ij = −πij/(2·cs²·τ·ρ) (cohérent : |S̃| = sqrt(S̃:S̃)).
            let mut pi_xx = 0.0;
            let mut pi_xy = 0.0;
            let mut pi_yy = 0.0;
            for i in 0..9 {
                let cx = D2Q9_VELOCITIES[i][0] as f64;
                let cy = D2Q9_VELOCITIES[i][1] as f64;
                let fneq = f[k][i] - feq(i, rho_k, ux_k, uy_k);
                pi_xx += fneq * cx * cx;
                pi_xy += fneq * cx * cy;
                pi_yy += fneq * cy * cy;
            }
            let rho_m = rho_k.max(1e-12);
            let denom = 2.0 * CS2 * tau * rho_m;
            let pi2 = pi_xx * pi_xx + 2.0 * pi_xy * pi_xy + pi_yy * pi_yy;
            shxx[k] = -pi_xx / denom;
            shxy[k] = -pi_xy / denom;
            shyy[k] = -pi_yy / denom;
            smag_hat[k] = pi2.sqrt() / denom; // identique collide_node

            // --- S̄ij, |S̄| par différences centrées des champs filtrés
            // (même convention de contraction que |S̃|).
            let sxx = ddx(&ubx, nx, ny, ix, iy);
            let syy = ddy(&uby, nx, ny, ix, iy);
            let sxy = 0.5 * (ddx(&uby, nx, ny, ix, iy) + ddy(&ubx, nx, ny, ix, iy));
            sbxx[k] = sxx;
            sbxy[k] = sxy;
            sbyy[k] = syy;
            smag_bar[k] = (sxx * sxx + 2.0 * sxy * sxy + syy * syy).sqrt();

            // --- Gradients thermiques (différences centrées, parois
            // clampées unilatéralement) : ∂T̃/∂xj (résolu) et ∂T̄/∂xj (filtré).
            gtx[k] = ddx(&temp, nx, ny, ix, iy);
            gty[k] = ddy(&temp, nx, ny, ix, iy);
            gbx[k] = ddx(&tb, nx, ny, ix, iy);
            gby[k] = ddy(&tb, nx, ny, ix, iy);

            // --- Similarité (Éq. 2.74, formes exactes) :
            // u′_i = (1/2)(ũ_i − ū_i) ⇒ qsm = (1/2)u′_i u′_i, soit
            //   qsm^{1/2} = (1/(2√2))·√(Σ_i(ũ_i − ū_i)²)
            // (choix par symétrie avec 2.74 : même différence de filtres
            // pour la vitesse que pour la température).
            // φ′_i = (1/2)(ũ_i − ū_i)(T̃ − T̄) ⇒ Φsm = (1/2)φ′_i φ′_i, soit
            //   Φsm^{1/2} = (1/2)·|T̃ − T̄|·√((1/2)·Σ_i(ũ_i − ū_i)²)
            let dux = ux_k - ubx[k];
            let duy = uy_k - uby[k];
            let du2 = dux * dux + duy * duy;
            qsm_h[k] = (0.5 * du2).sqrt() / 2.0; // = (1/(2√2))·√(du²+dv²)
            let dt_local = t_k - tb[k]; // T̃ − T̄ (peut être signé)
            // Φsm^{1/2} = (1/2)|T̃−T̄|·√(u′_i u′_i) = (1/2)|ΔT_filtre|·√((1/2)Σ(Δu_i)²)
            phi_h[k] = 0.5 * dt_local.abs() * (0.5 * du2).sqrt();

            // --- |T̃| (Éq. 2.75) : T̃i = ∂T̃/∂xj S̃ij, |T̃| = √(2 T̃i T̃i) ;
            // idem au filtre test avec S̄ et ∂T̄/∂xj.
            let th_x = gtx[k] * shxx[k] + gty[k] * shxy[k];
            let th_y = gtx[k] * shxy[k] + gty[k] * shyy[k];
            tmag_hat[k] = (2.0 * (th_x * th_x + th_y * th_y)).sqrt();
            let tb_x = gbx[k] * sbxx[k] + gby[k] * sbxy[k];
            let tb_y = gbx[k] * sbxy[k] + gby[k] * sbyy[k];
            tmag_bar[k] = (2.0 * (tb_x * tb_x + tb_y * tb_y)).sqrt();
        }
    }

    // q̄sm et Φ̄sm au filtre test : box-filter des quantités construites
    // (le tableau surbarre qsm/Φsm — pas une réestimation Bardina).
    let qbar_h = box_filter(&qsm_h, nx, ny, width);
    let phibar_h = box_filter(&phi_h, nx, ny, width);

    // --- Passe 3 : formes mixtes interpolées, tenseurs Germano M/L,
    // numérateurs/dénominateurs de Lilly (Éq. 2.58/2.70 + Tab. 2.3).
    let mut num_v = vec![0.0f64; n]; // L^d_ij M_ij
    let mut den_v = vec![0.0f64; n]; // M_ij M_ij
    let mut num_k = vec![0.0f64; n]; // L_j M_j
    let mut den_k = vec![0.0f64; n]; // M_j M_j (coquille « M_ij M_ij » du tableau corrigée)
    let mut nu_mix_grid = vec![0.0f64; n]; // (Δ̃²|S̃|)^α(Δ̃ qsm^{1/2})^{1−α}
    let mut kap_mix_grid = vec![0.0f64; n]; // (Δ̃³/ΔT·|T̃|)^α(Δ̃/ΔT·Φsm^{1/2})^{1−α}

    for iy in 0..ny {
        for ix in 0..nx {
            let k = iy * nx + ix;

            // Formes mixtes interpolées (Tab. 2.3) — α = 0.5 ⇒ moyenne
            // géométrique des volets Smagorinsky et similarité, aux DEUX
            // niveaux (grille tildé / test surbarré).
            let nu_mix_t = (dtest * dtest * smag_bar[k]).powf(alpha_exp)
                * (dtest * qbar_h[k]).powf(1.0 - alpha_exp);
            let nu_mix_g = (dx * dx * smag_hat[k]).powf(alpha_exp)
                * (dx * qsm_h[k]).powf(1.0 - alpha_exp);
            let kap_mix_t = (dtest * dtest * dtest / delta_t * tmag_bar[k]).powf(alpha_exp)
                * (dtest / delta_t * phibar_h[k]).powf(1.0 - alpha_exp);
            let kap_mix_g = (dx * dx * dx / delta_t * tmag_hat[k]).powf(alpha_exp)
                * (dx / delta_t * phi_h[k]).powf(1.0 - alpha_exp);
            nu_mix_grid[k] = nu_mix_g;
            kap_mix_grid[k] = kap_mix_g;

            // Tenseur M_ij mixte (Tab. 2.3) : différence des formes
            // interpolées test − grille, facteur Sij sorti des puissances
            // (les préfacteurs scalaires sont ≥ 0) :
            //   M_ij = νsm_mix(test)·S̄ij − νsm_mix(grille)·S̃ij.
            let mxx = nu_mix_t * sbxx[k] - nu_mix_g * shxx[k];
            let mxy = nu_mix_t * sbxy[k] - nu_mix_g * shxy[k];
            let myy = nu_mix_t * sbyy[k] - nu_mix_g * shyy[k];
            // Vecteur M_j mixte (Tab. 2.3, coquille dénominateur corrigée) :
            //   M_j = κsm_mix(test)·∂T̄/∂xj − κsm_mix(grille)·∂T̃/∂xj.
            let mx = kap_mix_t * gbx[k] - kap_mix_g * gtx[k];
            let my = kap_mix_t * gby[k] - kap_mix_g * gty[k];

            // Léonard au filtre test (Éq. 2.51 / Tab. 2.3) :
            // L_ij = ⟨u_i u_j⟩_test − ⟨ũ_i ũ_j⟩_test ; L_j = ⟨u_j T⟩_test − ⟨ũ_j T̃⟩_test
            // (champ résolu au filre test moins produit des champs filtrés).
            let lxx = lbuxx[k] - ubx[k] * ubx[k];
            let lxy = lbuxy[k] - ubx[k] * uby[k];
            let lyy = lbuyy[k] - uby[k] * uby[k];
            // L^d_ij = L_ij − (1/3)L_kk δij (formule 3D du tableau gardée,
            // δij avec i,j ∈ {x,y} — pas de composante zz en 2D).
            let lkk3 = (lxx + lyy) / 3.0;
            let ldxx = lxx - lkk3;
            let ldyy = lyy - lkk3; // L^d_xy = L_xy (δxy = 0)
            let ljx = lbutx[k] - ubx[k] * tb[k];
            let ljy = lbuty[k] - uby[k] * tb[k];

            // Contractions de Lilly (double comptage xy, symétrique) :
            num_v[k] = ldxx * mxx + 2.0 * lxy * mxy + ldyy * myy;
            den_v[k] = mxx * mxx + 2.0 * mxy * mxy + myy * myy;
            num_k[k] = ljx * mx + ljy * my;
            den_k[k] = mx * mx + my * my;
        }
    }

    // Régularisation Éq. 2.62 : moyenne box (locale) au numérateur ET
    // dénominateur — sans elle, C² bruité → instabilité (cf. plan §7).
    let num_v_b = box_filter(&num_v, nx, ny, width);
    let den_v_b = box_filter(&den_v, nx, ny, width);
    let num_k_b = box_filter(&num_k, nx, ny, width);
    let den_k_b = box_filter(&den_k, nx, ny, width);

    // --- Passe 4 : coefficients dynamiques + ν_t/α_t bornés ---
    let mut nu_t_pre = vec![0.0f64; n];
    let mut alpha_t_pre = vec![0.0f64; n];
    let mut cnu_sum = 0.0f64;
    let mut cnu_max = 0.0f64;
    let mut ck_sum = 0.0f64;
    let mut ck_max = 0.0f64;
    for k in 0..n {
        // Lilly régularisé (Éq. 2.58/2.70 + 2.62) : C = (1/2)·num/den,
        // garde-fou dénominateur ≈ 0 → C = 0, clamp [0, 0.04].
        let cnu = lilly_coefficient(num_v_b[k], den_v_b[k]);
        let ck = lilly_coefficient(num_k_b[k], den_k_b[k]);
        cnu_sum += cnu;
        if cnu > cnu_max {
            cnu_max = cnu;
        }
        ck_sum += ck;
        if ck > ck_max {
            ck_max = ck;
        }
        // ν_t(x,y) = C^ν_dyn · νsm_mix(grille) ; α_t(x,y) = C^κ_dyn · κsm_mix(grille).
        // Garde-fou TRT (Éq. 2.79/2.80) : τ⁺ = 0.5 + 3·ν_total et
        // τ_t⁺ = 0.5 + 3·α_total doivent rester ≥ τ_FLOOR = 0.505 (marge
        // de stabilité de la construction TRT, cf. assertion du solveur).
        // La borne nu_sm_min seule (défaut −0.1) peut amener
        // α_total = α + α_t sous (τ_FLOOR − 0.5)/3 quand α est petit
        // (ex. Ra=1e10, α_mult 2.5 : divergence au pas 15000) ⇒ on borne
        // ν_t/α_t par max(nu_sm_min·(mol), (τ_FLOOR − 0.5)/3 − mol).
        // OBSERVATION Ra=1e10 (ν = 0.0005) : le plancher τ_floor domine —
        // ν_t ≥ (0.505−0.5)/3 − ν = 2.33·ν en pratique dans tout nœud où
        // le modèle dynamique produit moins ; κsm ne module que vers le
        // haut. Le garde-fou transforme de facto le modèle en ν_t plancher
        // uniforme 2.33·ν + module κsm au-dessus.
        let tau_floor_margin = (TAU_TRT_FLOOR - 0.5) / 3.0;
        let nt = cnu * nu_mix_grid[k];
        let nu_t_min = (nu_sm_min * nu_mol).max(tau_floor_margin - nu_mol);
        nu_t_pre[k] = if nt < nu_t_min { nu_t_min } else { nt };
        let at = ck * kap_mix_grid[k];
        let alpha_t_min = (nu_sm_min * alpha_mol).max(tau_floor_margin - alpha_mol);
        alpha_t_pre[k] = if at < alpha_t_min { alpha_t_min } else { at };
    }

    // Diagnostics (somme/max séquentiels ⇒ déterministes quel que soit
    // --jobs). C^κ_dyn diagnostiqué aussi : κsm = C^κ_dyn·κsm_mix est
    // déjà branché dans la diffusion du scalaire g (Éq. 2.80,
    // κ_total = alpha + κsm, cf. collide_node).
    (
        nu_t_pre,
        alpha_t_pre,
        cnu_sum / n as f64,
        cnu_max,
        ck_sum / n as f64,
        ck_max,
    )
}
