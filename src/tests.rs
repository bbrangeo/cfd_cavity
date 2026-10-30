//! Tests unitaires : équilibres, conservation, CL, sens de la flottabilité.

use oxiphysics_lbm::lattice::{D2Q9_VELOCITIES, D2Q9_WEIGHTS};
use oxiphysics_lbm::thermal_lbm::{feq, geq};

use crate::snapshot;
use crate::solver::{CavitySolver, DELTA_T, T_COLD, T_HOT};

const EPS: f64 = 1e-10;

fn small_solver() -> CavitySolver {
    CavitySolver::new(16, 40, 1.0e5, 0.71, 0.1, 0.9, 0.02, 42, 1, 1.0, 1.0, None)
}

// ── Snapshot (export/import binaire + interpolation) ─────────────────────

#[test]
fn snapshot_atomic_write_preserves_previous_on_failure_and_roundtrips() {
    let nx = 10usize;
    let ny = 14usize;
    let n = nx * ny;
    let rho: Vec<f64> = (0..n).map(|k| 1.0 + 1e-4 * k as f64).collect();
    let ux: Vec<f64> = (0..n).map(|k| 0.001 * k as f64).collect();
    let uy: Vec<f64> = (0..n).map(|k| -0.002 * k as f64).collect();
    let t: Vec<f64> = (0..n).map(|k| 0.5 - k as f64 / n as f64).collect();

    let dir = std::env::temp_dir().join(format!("cav_snap_atomic_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("snapshot.bin");

    // 1er export atomique : roundtrip exact via le chemin tmp + rename.
    snapshot::write_snapshot_atomic(&path, nx, ny, &rho, &ux, &uy, &t).unwrap();
    let (rnx, rny, rrho, rux, ruy, rt) = snapshot::read_snapshot(&path).unwrap();
    assert_eq!((rnx, rny), (nx, ny));
    assert_eq!(rrho, rho);
    assert_eq!(rux, ux);
    assert_eq!(ruy, uy);
    assert_eq!(rt, t);

    // 2e export atomique écrasant : le contenu doit être remplacé sans
    // laisser de fichier temporaire résiduel dans le répertoire.
    let rho2: Vec<f64> = vec![2.0; n];
    snapshot::write_snapshot_atomic(&path, nx, ny, &rho2, &ux, &uy, &t).unwrap();
    let (_, _, rrho2, _, _, _) = snapshot::read_snapshot(&path).unwrap();
    assert_eq!(rrho2, rho2);
    let leftovers: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".tmp"))
        .collect();
    assert!(leftovers.is_empty(), "temporaires résiduels : {leftovers:?}");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn snapshot_roundtrip() {
    let nx = 12usize;
    let ny = 20usize;
    let n = nx * ny;
    let rho: Vec<f64> = (0..n).map(|k| 1.0 + 0.001 * k as f64).collect();
    let ux: Vec<f64> = (0..n).map(|k| 0.01 * k as f64).collect();
    let uy: Vec<f64> = (0..n).map(|k| -0.02 * k as f64).collect();
    let t: Vec<f64> = (0..n).map(|k| 0.5 - k as f64 / n as f64).collect();

    let dir = std::env::temp_dir().join(format!("cav_snap_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("snapshot.bin");
    snapshot::write_snapshot(&path, nx, ny, &rho, &ux, &uy, &t).unwrap();
    let (rnx, rny, rrho, rux, ruy, rt) = snapshot::read_snapshot(&path).unwrap();
    assert_eq!((rnx, rny), (nx, ny));
    assert_eq!(rrho, rho);
    assert_eq!(rux, ux);
    assert_eq!(ruy, uy);
    assert_eq!(rt, t);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn snapshot_interpolate_same_grid_identity() {
    let src = vec![1.0f64; 5 * 9];
    let dst = snapshot::interpolate_field(&src, 5, 9, 5, 9);
    assert_eq!(dst, src);
}

#[test]
fn snapshot_interpolate_grid_change_preserves_mean() {
    // Champ T linéaire en x : l'interpolation 8×32 → 16×64 doit préserver
    // T(0)=+0.5, T(1)=−0.5 sur les bords (mêmes coordonnées normalisées).
    let (nx0, ny0) = (8usize, 32usize);
    let mut t0 = Vec::with_capacity(nx0 * ny0);
    for iy in 0..ny0 {
        for ix in 0..nx0 {
            let xf = ix as f64 / (nx0 - 1) as f64;
            t0.push(T_HOT + (T_COLD - T_HOT) * xf);
        }
    }
    let t1 = snapshot::interpolate_field(&t0, nx0, ny0, 16, 64);
    assert_eq!(t1.len(), 16 * 64);
    // Interpolation exacte d'un champ linéaire : T(ix/15) partout.
    for iy in 0..64 {
        assert!((t1[iy * 16] - T_HOT).abs() < EPS);
        assert!((t1[iy * 16 + 15] - T_COLD).abs() < EPS);
        for ix in 0..16 {
            let xf = ix as f64 / 15.0;
            let t_exact = T_HOT + (T_COLD - T_HOT) * xf;
            assert!((t1[iy * 16 + ix] - t_exact).abs() < EPS);
        }
    }
}

// ── Équilibres ───────────────────────────────────────────────────────────

#[test]
fn feq_sum_equals_rho() {
    let rho = 1.3;
    let sum: f64 = (0..9).map(|i| feq(i, rho, 0.07, -0.03)).sum();
    assert!((sum - rho).abs() < 1e-12);
}

#[test]
fn geq_sum_equals_temperature() {
    let t = 0.37;
    let sum: f64 = (0..9).map(|i| geq(i, t, 0.05, 0.02)).sum();
    assert!((sum - t).abs() < 1e-12);
}

#[test]
fn weights_sum_to_one() {
    let s: f64 = D2Q9_WEIGHTS.iter().sum();
    assert!((s - 1.0).abs() < 1e-12);
}

// ── Initialisation ───────────────────────────────────────────────────────

#[test]
fn init_temperature_linear_and_bounded() {
    let s = small_solver();
    let nx = s.nx();
    // Paroi gauche = +0.5, paroi droite = −0.5
    assert!((s.lbm.temperature[s.idx(0, 10)] - T_HOT).abs() < 2e-3);
    assert!((s.lbm.temperature[s.idx(nx - 1, 10)] - T_COLD).abs() < 2e-3);
    // Vitesse nulle partout à l'init
    for k in 0..nx * s.ny() {
        assert!(s.lbm.ux[k].abs() < EPS);
        assert!(s.lbm.uy[k].abs() < EPS);
    }
}

// ── Conservation de masse sur un pas ─────────────────────────────────────

#[test]
fn mass_conserved_per_phase() {
    let mut s = small_solver();
    let m0 = s.total_f_mass();
    s.collide(100);
    let m1 = s.total_f_mass();
    s.stream();
    let m2 = s.total_f_mass();
    s.apply_bounce_back();
    let m3 = s.total_f_mass();
    s.apply_thermal_boundary_conditions();
    let m4 = s.total_f_mass();
    assert!(
        (m1 - m0).abs() < 1e-8 && (m2 - m0).abs() < 1e-8 && (m3 - m0).abs() < 1e-8 && (m4 - m0).abs() < 1e-8,
        "m0={m0:.12} collide={m1:.12} stream={m2:.12} bb={m3:.12} thermal={m4:.12}"
    );
}

// ── Forcing Guo (consistance) ────────────────────────────────────────────

#[test]
fn guo_forcing_preserves_mass_and_upward_sign() {
    // Cohérence du forcing complet de Guo et al. 2002 :
    // 1) masse conservée après un pas COMPLET (collision Guo + streaming +
    //    correction de vitesse F/2ρ) — le terme de force Guo est
    //    masse-conservant (Σ_i S_i = 0) et la correction ne touche que u_y ;
    // 2) signe préservé : fluide chaud (θ > 0) ⇒ fy > 0 ⇒ correction
    //    +fy/(2ρ) pousse u_y vers le haut, jamais vers le bas.
    let mut s = small_solver();
    let m0 = s.total_f_mass();
    // 210 pas : on sort de la fenêtre de spin-up (200 pas sur cette grille)
    // pour que fy_node = ρ·β_eff·θ avec le β_eff complet.
    for _ in 0..210 {
        s.step();
    }
    let m1 = s.total_f_mass();
    assert!(
        (m1 - m0).abs() / m0.abs() < 1e-9,
        "masse non conservée avec forcing Guo actif : m0={m0:.12} m1={m1:.12}"
    );
    // Aux nœuds intérieurs, u_y doit être exactement
    // (Σ e_iy·f_i + fy_node/2)/ρ — cohérence entre le forcing Guo de la
    // collision et la correction de vitesse de update_macroscopic
    // (éq. (16), Δt = 1). On utilise fy_node stockée par la collision
    // (évaluée aux champs du temps t, conformément à Guo) et NON la
    // température post-mise à jour (advectée depuis le pas). On teste tous
    // les nœuds intérieurs (les parois sont écrasées à u = 0 par
    // enforce_wall_velocity).
    let nx = s.nx();
    let ny = s.ny();
    for iy in 1..ny - 1 {
        for ix in 1..nx - 1 {
            let k = s.idx(ix, iy);
            let mut mom_y = 0.0;
            for j in 0..9 {
                mom_y += D2Q9_VELOCITIES[j][1] as f64 * s.lbm.f[k][j];
            }
            // Même ordre d'opérations que le code (deux divisions
            // successives) pour une vérification bitwise-exacte.
            let mut uy_expected = mom_y / s.lbm.rho[k];
            uy_expected += 0.5 * s.fy_node[k] / s.lbm.rho[k];
            assert!(
                (s.lbm.uy[k] - uy_expected).abs() < 1e-12,
                "u_y incohérent avec Guo éq.(16) : lu {}, attendu {} (nœud {ix},{iy})",
                s.lbm.uy[k],
                uy_expected
            );
        }
    }
}

// ── Forcing Guo (exactitude momentum : canal de Poiseuille TRT) ──────────
//
// Discriminateur analytique du facteur du lattice effect en TRT. La
// correction de vitesse u = Σef/ρ + F/2ρ (éq. (16), update_macroscopic)
// impose j_neq = −F/2 ; le bilan de momentum par pas est
// Δu = (F/ρ)·(1/(2τ⁻) + guo). L'exactitude d'Euler de Guo-BGK (bilan
// exactement F/ρ pour tout τ) exige guo = 1 − 1/(2τ⁻) sur la partie
// IMPAIRE de S_i (porte le momentum, relaxe à ω⁻) ; la partie PAIRE
// (stress, ω⁺) prend 1 − 1/(2τ⁺). Avec le facteur τ⁺ sur le S_i COMPLET
// (ancien code), le facteur effectif serait 1/(2τ⁻) + 1 − 1/(2τ⁺) ≈ 0.13
// ici : u_max mesuré ~0.05× la cible, impossible à rater.
//
// Canal D2Q9 TRT Λ=3/16 : périodique en x, bounce-back mi-lien en y
// (H = NY exact avec Λ = 3/16), force horizontale uniforme F = fx·ρ par
// nœud (masse-conservante, indépendante de τ). Cible analytique : régime
// transitoire exact du premier mode u_max·(1 − e^{−N·ν·π²/H²}) — le
// forçage étant exact, la cible est indépendante du schéma de collision ;
// à N·ν·π²/H² ≈ 3.36, le premier mode estime la solution à ~0.1% près
// (modes supérieurs en e^{−9·3.36}), d'où la tolérance ±10%.
//
// Référence : Guo, Shi & Zheng, Phys. Rev. E 65, 046308 (2002), éq. (14)-(16).
#[test]
fn guo_forcing_exact_momentum_poiseuille() {
    use oxiphysics_lbm::lattice::{D2Q9_OPPOSITES, CS2};

    // Canal : périodique en x, murs bounce-back à y = −0.5 et NY − 0.5
    // (mi-lien) ⇒ H = NY.
    let (nx, ny) = (30usize, 21usize);
    let n = nx * ny;
    let h = ny as f64;
    // Viscosité fixée (τ⁺ = 0.53) ; force volumique fx (u_max ≈ 5.5e-3,
    // régime linéaire, erreur de compressibilité négligeable).
    let nu = 0.01;
    let fx = 1.0e-6;
    let tau_plus = 0.5 + 3.0 * nu;
    let tau_minus = 0.5 + (3.0 / 16.0) / (tau_plus - 0.5);
    let u_max_steady = fx * h * h / (8.0 * nu); // Poiseuille permanent
    // Coefficients TRT (identiques au solver : a = (ω+ + ω−)/2, b = (ω+ − ω−)/2).
    let a_f = 0.5 * (1.0 / tau_plus + 1.0 / tau_minus);
    let b_f = 0.5 * (1.0 / tau_plus - 1.0 / tau_minus);

    // 15 000 pas : N·ν·π²/H² ≈ 3.36 ⇒ fraction de convergence ≈ 0.965 du
    // régime permanent ; runtime ≪ 1 s en release (630 nœuds).
    let nsteps = 15_000usize;
    let conv = 1.0 - (-(nsteps as f64) * nu * std::f64::consts::PI.powi(2) / (h * h)).exp();
    let u_max_expected = u_max_steady * conv;

    // f à repos (ρ = 1, u = 0).
    let mut f = vec![[0.0f64; 9]; n];
    for node in &mut f {
        for (i, fi) in node.iter_mut().enumerate() {
            *fi = D2Q9_WEIGHTS[i];
        }
    }

    for _ in 0..nsteps {
        // Macroscopique depuis f courant (Guo canonique : ρ, u recalculés).
        let mut rho = [0.0f64; 30 * 21];
        let mut ux = [0.0f64; 30 * 21];
        let mut uy = [0.0f64; 30 * 21];
        for (k, node) in f.iter().enumerate() {
            let mut r = 0.0;
            let mut mx = 0.0;
            let mut my = 0.0;
            for (j, &fj) in node.iter().enumerate() {
                r += fj;
                mx += D2Q9_VELOCITIES[j][0] as f64 * fj;
                my += D2Q9_VELOCITIES[j][1] as f64 * fj;
            }
            rho[k] = r;
            ux[k] = mx / r;
            uy[k] = my / r;
        }

        // Collision f TRT avec forcing Guo — structure EXACTEMENT celle de
        // collide_node (impair → τ⁻, pair → τ⁺), F = (fx·ρ, 0) par nœud.
        let mut f_post = f.clone();
        for (k, node) in f.iter().enumerate() {
            let f_node = fx * rho[k]; // force par nœud
            let u_y = uy[k];
            // Correction de vitesse de Guo (éq. (16)) : u ← u + F/2ρ
            // (seulement la composante x, F_y = 0).
            let u_x = ux[k] + 0.5 * f_node / rho[k];
            // Équilibres aux champs corrigés (d_f pré-collision).
            let mut d_f = [0.0f64; 9];
            for i in 0..9 {
                let cx = D2Q9_VELOCITIES[i][0] as f64;
                let cy = D2Q9_VELOCITIES[i][1] as f64;
                let eu = cx * u_x + cy * u_y;
                let u2 = u_x * u_x + u_y * u_y;
                let feq_i = D2Q9_WEIGHTS[i]
                    * rho[k]
                    * (1.0 + eu / CS2 + eu * eu / (2.0 * CS2 * CS2) - u2 / (2.0 * CS2));
                d_f[i] = node[i] - feq_i;
            }
            for i in 0..9 {
                let cx = D2Q9_VELOCITIES[i][0] as f64;
                let cy = D2Q9_VELOCITIES[i][1] as f64;
                let opp = D2Q9_OPPOSITES[i];
                let ei_u = cx * u_x + cy * u_y;
                let s_i =
                    D2Q9_WEIGHTS[i] * ((cx - u_x) / CS2 + ei_u * cx / (CS2 * CS2)) * f_node;
                let s_minus = D2Q9_WEIGHTS[i] * (cx / CS2) * f_node; // impair : momentum
                let s_plus = s_i - s_minus;
                f_post[k][i] += -a_f * d_f[i] - b_f * d_f[opp]
                    + (1.0 - 0.5 / tau_minus) * s_minus // exactitude momentum
                    + (1.0 - 0.5 / tau_plus) * s_plus; // exactitude stress
            }
        }

        // Streaming pull : périodique en x (wrap), bounce-back intégré en y
        // (source hors domaine ⇒ distribution post-collision opposée, même
        // structure que stream() du solver — no-slip au milieu du lien).
        let mut f_new = f_post.clone();
        for iy in 0..ny {
            for ix in 0..nx {
                let k = iy * nx + ix;
                for i in 0..9 {
                    let cx = D2Q9_VELOCITIES[i][0] as i64;
                    let cy = D2Q9_VELOCITIES[i][1] as i64;
                    let sx = (ix as i64 - cx).rem_euclid(nx as i64) as usize;
                    let sy = iy as i64 - cy;
                    if sy < 0 || sy >= ny as i64 {
                        f_new[k][i] = f_post[k][D2Q9_OPPOSITES[i]];
                    } else {
                        f_new[k][i] = f_post[sy as usize * nx + sx][i];
                    }
                }
            }
        }
        f = f_new;
    }

    // Vitesse physique finale u = Σef/ρ + F/2ρ (éq. (16), F/2ρ = fx/2).
    let mut u_max = 0.0f64;
    for node in &f {
        let mut r = 0.0;
        let mut mx = 0.0;
        for (j, &fj) in node.iter().enumerate() {
            r += fj;
            mx += D2Q9_VELOCITIES[j][0] as f64 * fj;
        }
        u_max = u_max.max(mx / r + 0.5 * fx);
    }

    assert!(
        (u_max - u_max_expected).abs() / u_max_steady < 0.10,
        "u_max Poiseuille : mesuré {u_max:.6e}, attendu {u_max_expected:.6e} \
         (permanent {u_max_steady:.6e}, conv {conv:.4}) — ratio mesuré/attendu \
         {:.4} (~0.05 signifierait le facteur τ⁺ faux sur tout S_i)",
        u_max / u_max_expected
    );
}

// ── Sens de la flottabilité ──────────────────────────────────────────────

#[test]
fn hot_fluid_pushed_upward() {
    // Après plusieurs pas, le fluide chaud près de la paroi gauche doit
    // s'élever : u_y > 0 dans le tiers supérieur gauche.
    let mut s = CavitySolver::new(16, 40, 1.0e5, 0.71, 0.1, 0.9, 0.02, 42, 1, 1.0, 1.0, None);
    for _ in 0..200 {
        s.step();
    }
    // Moyenne de u_y sur la colonne ix=1..4, moitié haute intérieure
    let nx = s.nx();
    let ny = s.ny();
    let mut sum = 0.0;
    let mut cnt = 0;
    for iy in (ny / 2)..(ny - 2) {
        for ix in 1..4 {
            sum += s.lbm.uy[s.idx(ix, iy)];
            cnt += 1;
        }
    }
    let mean_uy = sum / cnt as f64;
    assert!(mean_uy > 0.0, "le fluide chaud doit monter (u_y > 0), lu {mean_uy}");
}

// ── Conditions aux limites thermiques ────────────────────────────────────

#[test]
fn isothermal_walls_fixed() {
    let mut s = small_solver();
    for _ in 0..50 {
        s.step();
    }
    let nx = s.nx();
    let ny = s.ny();
    for iy in 0..ny {
        assert!(
            (s.lbm.temperature[s.idx(0, iy)] - T_HOT).abs() < 1e-9,
            "paroi chaude : T = {} ≠ {T_HOT}",
            s.lbm.temperature[s.idx(0, iy)]
        );
        assert!(
            (s.lbm.temperature[s.idx(nx - 1, iy)] - T_COLD).abs() < 1e-9,
            "paroi froide : T = {} ≠ {T_COLD}",
            s.lbm.temperature[s.idx(nx - 1, iy)]
        );
    }
}

#[test]
fn adiabatic_walls_zero_gradient() {
    let mut s = small_solver();
    for _ in 0..50 {
        s.step();
    }
    let nx = s.nx();
    let ny = s.ny();
    // ∂T/∂y = 0 en haut et en bas (zero-gradient) : T[iy=0] == T[iy=1]
    for ix in 0..nx {
        let t_bottom0 = s.lbm.temperature[s.idx(ix, 0)];
        let t_bottom1 = s.lbm.temperature[s.idx(ix, 1)];
        let t_top0 = s.lbm.temperature[s.idx(ix, ny - 1)];
        let t_top1 = s.lbm.temperature[s.idx(ix, ny - 2)];
        assert!(
            (t_bottom0 - t_bottom1).abs() < 1e-9,
            "adiabatique bas : ΔT = {}",
            t_bottom0 - t_bottom1
        );
        assert!(
            (t_top0 - t_top1).abs() < 1e-9,
            "adiabatique haut : ΔT = {}",
            t_top0 - t_top1
        );
    }
}

// ── No-slip ──────────────────────────────────────────────────────────────

#[test]
fn no_slip_walls() {
    let mut s = small_solver();
    for _ in 0..50 {
        s.step();
    }
    let nx = s.nx();
    let ny = s.ny();
    for iy in 0..ny {
        for &ix in &[0usize, nx - 1] {
            let k = s.idx(ix, iy);
            assert!(s.lbm.ux[k].abs() < 1e-12 && s.lbm.uy[k].abs() < 1e-12);
        }
    }
    for &iy in &[0usize, ny - 1] {
        for ix in 0..nx {
            let k = s.idx(ix, iy);
            assert!(s.lbm.ux[k].abs() < 1e-12 && s.lbm.uy[k].abs() < 1e-12);
        }
    }
}

// ── Diagnostic Nusselt ───────────────────────────────────────────────────

#[test]
fn nusselt_conduction_limit_near_one() {
    // Profil purement conductif (T linéaire imposé) ⇒ Nu ≈ 1.
    // Nu = |dθ/dx|·Lz/ΔT avec θ normalisé par ΔT : en x* = x/Lx,
    // dθ/dx* = ΔT·Lz/Lx = Az ⇒ Nu conductif = Az... NON : le gradient
    // physique par unité de HAUTEUR est dθ/dz = ΔT/Lx (T varie en x),
    // donc Nu = (dθ/dz)·Lz/ΔT = Lz/Lx = Az. Le profil conductif dans une
    // cavité Az≠1 a Numécaniquement Nu=Az en normalisation hauteur.
    // La convention Trias/Xin (x* par la LARGEUR, Nu = ∂θ/∂x*·A) donne
    // Nu=1 : on normalise donc le gradient paroi par Lx/ΔT (cohérent
    // benchmark) et le test revient à sa valeur d'origine.
    let mut s = small_solver();
    let nx = s.nx();
    let ny = s.ny();
    // Réinitialise T linéaire exactement (sans perturbation)
    for iy in 0..ny {
        for ix in 0..nx {
            let xf = ix as f64 / (nx - 1) as f64;
            let t = T_HOT + (T_COLD - T_HOT) * xf;
            let k = s.idx(ix, iy);
            s.lbm.temperature[k] = t;
        }
    }
    let nuc = crate::post::nusselt_hot(&s);
    let az = (s.ny() - 1) as f64 / (s.nx() - 1) as f64;
    assert!((nuc - az).abs() < 0.05, "Nu conductif = Az en normalisation hauteur, lu {nuc}");
}

// ── Delta T et constantes ────────────────────────────────────────────────

#[test]
fn delta_t_is_one() {
    assert!((DELTA_T - 1.0).abs() < EPS);
}

// ── Échelles mixtes dynamiques (§2.3.5 thèse, --dynamic-smag) ────────────

#[test]
fn lilly_coefficient_clamps_on_zero_denominator_and_bounds() {
    use crate::solver::lilly_coefficient;
    // Dénominateur nul / négatif / non fini ⇒ garde-fou C = 0 (pas de NaN).
    assert_eq!(lilly_coefficient(1.0, 0.0), 0.0);
    assert_eq!(lilly_coefficient(1.0, -1e-12), 0.0);
    assert_eq!(lilly_coefficient(f64::NAN, 1.0), 0.0);
    // Facteur 1/2 de Lilly (Tab. 2.3) : C = 0.5·num/den.
    assert!((lilly_coefficient(0.02, 1.0) - 0.01).abs() < EPS);
    // Clamp [0, 0.04] (C ≥ 0 : pas de cascade inverse ; borne sup crate).
    assert_eq!(lilly_coefficient(100.0, 1.0), 0.04);
    assert_eq!(lilly_coefficient(-100.0, 1.0), 0.0);
}

#[test]
fn mixed_factor_alpha_half_is_geometric_mean() {
    use crate::solver::mixed_factor;
    // α = 0.5 ⇒ moyenne géométrique √(a·b) (Tab. 2.3).
    let (a, b) = (4.0, 9.0);
    let m = mixed_factor(a, b, 0.5);
    assert!((m - (a * b).sqrt()).abs() < 1e-12, "m = {m}");
    // Limites : α = 0 → b (similarité pure), α = 1 → a (Smagorinsky pur).
    assert!((mixed_factor(a, b, 0.0) - b).abs() < 1e-12);
    assert!((mixed_factor(a, b, 1.0) - a).abs() < 1e-12);
}

#[test]
fn dynamic_smag_off_matches_fixed_cs_nu_t() {
    // Flag OFF (défaut) : le chemin de collision ne doit PAS être altéré —
    // ν_t de la run dynamique désactivée = ν_t de la run Cs fixe classique,
    // et les buffers nu_t_pre/alpha_t_pre restent None.
    let mut s = small_solver();
    s.set_dynamic_sgs(false, 0.5, -0.1, 2.0);
    let mut s_ref = small_solver();
    for _ in 0..30 {
        s.step();
        s_ref.step();
    }
    assert!(s.nu_t_pre.is_none() && s.alpha_t_pre.is_none());
    for k in 0..s.nu_t.len() {
        assert_eq!(s.nu_t[k], s_ref.nu_t[k], "ν_t altéré en mode OFF (nœud {k})");
    }
}

#[test]
fn dynamic_smag_on_runs_sane_and_reports_coefficients() {
    // Flag ON : un pas dynamique s'exécute sans NaN, remplit nu_t_pre et
    // reporte des coefficients C^ν_dyn bornés [0, 0.04].
    let mut s = small_solver();
    s.set_dynamic_sgs(true, 0.5, -0.1, 2.0);
    for _ in 0..5 {
        s.step();
    }
    let n = s.nx() * s.ny();
    assert!(s.nu_t_pre.is_some() && s.alpha_t_pre.is_some());
    assert!(s.cs_dyn_mean.is_finite() && s.cs_dyn_max.is_finite());
    assert!((0.0..=0.04).contains(&s.cs_dyn_max), "cs_dyn_max = {}", s.cs_dyn_max);
    assert!(s.cs_dyn_mean <= s.cs_dyn_max + EPS);
    for k in 0..n {
        assert!(s.nu_t[k].is_finite(), "ν_t non fini (nœud {k})");
    }
}

// ── Conductance murale thermique (discriminateur TRT-thermique) ──────────
//
// Avant patch : le temps de relaxation impair du scalaire thermique était
// calculé avec Λ = 3/16 (constante magic du bounce-back de QUANTITÉ DE
// MOUVEMENT), alors que la paroi thermique est un mur à nœud-équilibre
// (g[wall] = geq(T_wall, 0), pas de rebond — cf. apply_thermal_boundary_
// conditions). À α petit, τ_t⁻ explose (37.3 à α = 0.00169) : les
// populations thermiques impaires sont gelées, la conductance de liaison
// murale s'effondre (×1/37) et le saut de température à la première
// liaison explose (0.187 contre ~0.005/cellule à l'intérieur) → biais
// Nuc ×3. Après patch (τ_t⁻ = τ_t⁺, TRT réduit à BGK pour g) : pas de
// saut mural, profil linéaire.
//
// Champ de diffusion pur (Ra = 1 → flottabilité négligeable, pas de flux),
// α = 0.00169 (τ_t⁺ = 0.50507), grille 21×21, murs T = ±0.5 (BC existantes).
// Après 20 000 pas (le profil initial linéaire est déjà la solution
// stationnaire de la diffusion), la 1ʳᵉ liaison murale présente l'artefact
// classique du BC à nœud-équilibre (mur effectif au demi-lien) : saut
// ≈ MOITIÉ du saut intérieur (ratio ≈ 0.5). Avant patch : ratio ≈ 37
// (saut mural 0.187 vs 0.005/cellule, τ_t⁻ = 37.3 gelé) — discriminant
// net entre les deux régimes, assertion à ±10 % de 0.5.
#[test]
fn thermal_wall_link_conductance_first_link_matches_interior() {
    let mut s = CavitySolver::new(
        21, 21, 1.0, 1.0, 0.0, 1.0, 0.02, 42, 1, 1.0, 1.0, Some(0.00169),
    );
    for _ in 0..20_000 {
        s.step();
    }
    let nx = s.nx();
    let ny = s.ny();
    // Moyenne sur les lignes intérieures (évite les coins) des sauts
    // première liaison et liaison intérieure, côté chaud et côté froid.
    let mut hot1 = 0.0;
    let mut hot2 = 0.0;
    let mut cold1 = 0.0;
    let mut cold2 = 0.0;
    let mut cnt = 0usize;
    for iy in 1..ny - 1 {
        let t0 = s.lbm.temperature[s.idx(0, iy)];
        let t1 = s.lbm.temperature[s.idx(1, iy)];
        let t2 = s.lbm.temperature[s.idx(2, iy)];
        hot1 += (t0 - t1).abs();
        hot2 += (t1 - t2).abs();
        let tn1 = s.lbm.temperature[s.idx(nx - 1, iy)];
        let tn2 = s.lbm.temperature[s.idx(nx - 2, iy)];
        let tn3 = s.lbm.temperature[s.idx(nx - 3, iy)];
        cold1 += (tn1 - tn2).abs();
        cold2 += (tn2 - tn3).abs();
        cnt += 1;
    }
    let hot1 = hot1 / cnt as f64;
    let hot2 = hot2 / cnt as f64;
    let cold1 = cold1 / cnt as f64;
    let cold2 = cold2 / cnt as f64;
    let r_hot = hot1 / hot2;
    let r_cold = cold1 / cold2;
    assert!(
        (r_hot - 0.5).abs() < 0.05 && (r_cold - 0.5).abs() < 0.05,
        "conductance murale : ratio 1ʳᵉ liaison/intérieur chaud {r_hot:.3}, froid {r_cold:.3} \
         (attendu ≈0.5, artefact demi-lien du BC à nœud-équilibre ; ≈37 signifierait \
         Λ=3/16 injustifié sur g / τ_t⁻ explosé)"
    );
}
