//! Cavité différentiellement chauffée 2D — benchmark thèse (§2.5, Tab. 2.6/2.7).
//!
//! Solveur LBM D2Q9 double distribution avec Smagorinsky LES, construit sur la
//! crate `oxiphysics-lbm` (structure ThermalD2Q9 + équilibres + fonctions physiques),
//! pas de temps / CL / post-traitement maison.

mod cli;
mod post;
mod rng;
mod snapshot;
mod solver;
#[cfg(test)]
mod tests;

use std::path::Path;
use std::time::Instant;

use post::TimeAverager;
use solver::CavitySolver;

/// Garde-fou de stabilité : abort si max_u dépasse cette valeur.
/// Ma_local = max_u·√3 ≈ 0.17 à la limite — marge ×3 sous les divergences
/// observées (max_u ≈ 0.34, Ma_local ≈ 0.58 → NaN).
const MAX_U_ABORT: f64 = 0.10;

fn main() {
    let args = match cli::parse_args() {
        Ok(a) => a,
        Err(e) if e == "help" => {
            eprint!("{}", cli::usage());
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("Erreur : {e}\n\n{}", cli::usage());
            std::process::exit(1);
        }
    };

    println!("=== Cavité différentiellement chauffée 2D (LBM D2Q9 + Smagorinsky) ===");
    println!(
        "Grille : {}×{} (Az = {:.2}) | Ra = {:.3e} | Pr = {} | Cs = {} | Pr_t = {} | jobs = {}",
        args.nx,
        args.ny,
        (args.ny - 1) as f64 / (args.nx - 1) as f64,
        args.ra,
        args.pr,
        args.cs,
        args.pr_t,
        args.jobs
    );

    // Pool rayon configuré AVANT toute boucle parallèle (--jobs N).
    if args.jobs > 1 {
        // Cap au nombre de cœurs disponibles avec warning (pas d'erreur) :
        // au-delà, rayon survend le CPU sans gain (bench : saturation ~×2.3).
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        if args.jobs > cores {
            eprintln!(
                "Warning : --jobs {} > {} cœurs disponibles ; cap à {}.",
                args.jobs, cores, cores
            );
            CavitySolver::set_jobs(cores);
        } else {
            CavitySolver::set_jobs(args.jobs);
        }
    }

    let mut sim = CavitySolver::new(
        args.nx,
        args.ny,
        args.ra,
        args.pr,
        args.cs,
        args.pr_t,
        args.utarget,
        args.seed,
        args.jobs,
        args.alpha_mult,
        args.nu_start_mult,
        args.nu,
    );

    // --- Échelles mixtes dynamiques (§2.3.5 thèse, --dynamic-smag) ---
    // Désactivées par défaut : le chemin de collision reste bitwise identique.
    sim.set_dynamic_sgs(args.dynamic_smag, args.smag_alpha, args.nu_sm_min, args.eta_test);
    if args.dynamic_smag {
        println!(
            "LES dynamique : échelles mixtes α = {} (Tab. 2.3) | η_test = {} | νsm_min = {}·ν",
            args.smag_alpha, args.eta_test, args.nu_sm_min
        );
    }

    // --- Option --restart : initialisation depuis un snapshot binaire ---
    if let Some(ref snap_path) = args.restart {
        let (snx, sny, rho_s, ux_s, uy_s, t_s) =
            snapshot::read_snapshot(Path::new(snap_path)).unwrap_or_else(|e| {
                eprintln!("Erreur de lecture du snapshot {snap_path} : {e}");
                std::process::exit(6);
            });
        println!(
            "Restart depuis {snap_path} : snapshot {}×{} → grille {}×{}{}",
            snx,
            sny,
            args.nx,
            args.ny,
            if (snx, sny) == (args.nx, args.ny) {
                ""
            } else {
                " (interpolation bilinéaire)"
            }
        );
        let rho_i = snapshot::interpolate_field(&rho_s, snx, sny, args.nx, args.ny);
        let ux_i = snapshot::interpolate_field(&ux_s, snx, sny, args.nx, args.ny);
        let uy_i = snapshot::interpolate_field(&uy_s, snx, sny, args.nx, args.ny);
        let t_i = snapshot::interpolate_field(&t_s, snx, sny, args.nx, args.ny);
        sim.init_from_fields(&rho_i, &ux_i, &uy_i, &t_i);
    }

    println!(
        "Unités réseau : nu = {:.6} | alpha = {:.6} | tau = {:.4} | tau_t = {:.4} | beta_eff = {:.4e}",
        sim.nu, sim.alpha, sim.lbm.tau, sim.lbm.tau_t, sim.beta_eff
    );
    println!(
        "Ra vérifié depuis unités réseau : {:.4e} (cible {:.4e})",
        oxiphysics_lbm::thermal_lbm::rayleigh_number(
            1.0,
            sim.beta_eff,
            solver::DELTA_T,
            sim.lz,
            sim.nu,
            sim.alpha_phys
        ),
        args.ra
    );

    let n_nodes = sim.nx() * sim.ny();
    let mut averager = TimeAverager::new(n_nodes);
    let mut log_lines: Vec<String> = Vec::new();
    let start = Instant::now();

    // --- Export périodique (--snapshot-every N, 0 = désactivé) ---
    let snap_out_path = if args.snapshot_every > 0 {
        let out = Path::new(&args.out);
        if let Err(e) = std::fs::create_dir_all(out) {
            eprintln!("Impossible de créer {} : {e}", out.display());
            std::process::exit(4);
        }
        Some(out.join("snapshot.bin"))
    } else {
        None
    };

    for step in 0..args.steps {
        sim.step();

        // Diagnostic de stabilité toutes les progress-every itérations
        if step % args.progress_every == 0 || step + 1 == args.steps {
            let wmax = post::max_uy(&sim);
            let nuc = post::nusselt_hot(&sim);
            let nuf = post::nusselt_cold(&sim);
            let nu_t_max = sim
                .nu_t
                .iter()
                .fold(0.0f64, |m, &v| m.max(v.abs()));
            // LES dynamique : coefficients C^ν_dyn moy/max reportés dans la
            // ligne de progression (champs additionnels en fin de ligne —
            // ne casse pas le format existant en mode OFF).
            let dyn_suffix = if args.dynamic_smag {
                format!(
                    " | cs_dyn_mean = {:.4} | cs_dyn_max = {:.4} | kap_dyn_mean = {:.4}",
                    sim.cs_dyn_mean, sim.cs_dyn_max, sim.kappa_dyn_mean
                )
            } else {
                String::new()
            };
            let msg = format!(
                "step {:>7} | w_max = {:.5} | max_u = {:.5} | Nuc = {:.4} | Nuf = {:.4} | masse = {:.6} | nu_t_max = {:.3e}{}",
                step,
                wmax,
                sim.lbm
                    .ux
                    .iter()
                    .zip(sim.lbm.uy.iter())
                    .fold(0.0f64, |m, (&x, &v)| m.max(x.abs().max(v.abs()))),
                nuc,
                nuf,
                sim.total_mass() / n_nodes as f64,
                nu_t_max,
                dyn_suffix
            );
            println!("{msg}");
            log_lines.push(msg);

            // Garde-fou max_u : sortie gracieuse AVANT divergence NaN (heures
            // CPU gaspillées sinon). Le message est poussé dans log_lines
            // AVANT l'abort… mais process::exit ne passe pas par write_log :
            // on écrit donc le log de progression directement ici.
            let max_u = sim
                .lbm
                .ux
                .iter()
                .zip(sim.lbm.uy.iter())
                .fold(0.0f64, |m, (&x, &v)| m.max(x.abs().max(v.abs())));
            if max_u > MAX_U_ABORT {
                let ma_local = max_u * 3.0f64.sqrt();
                let abort_msg = format!(
                    "AVORT : max_u = {max_u:.5} dépasse le garde-fou {:.2} (Ma_local = max_u·√3 ≈ {ma_local:.3} > marge de sécurité)",
                    MAX_U_ABORT
                );
                eprintln!("{abort_msg}");
                log_lines.push(abort_msg.clone());
                // Log de progression gracieux : même chemin que la fin du main.
                let out = Path::new(&args.out);
                let _ = std::fs::create_dir_all(out);
                let _ = post::write_log(&out.join("progress.log"), &log_lines);
                std::process::exit(2);
            }

            if !post::check_finite(&sim) {
                // DIAGNOSTIC TEMPORAIRE (divergence Ra=1e10 dynamique) :
                // localise le premier nœud non fini et l'état des distributions.
                let nx = sim.nx();
                let ny = sim.ny();
                let first_nonfinite = |v: &[f64]| v.iter().position(|x| !x.is_finite());
                let pos_t = first_nonfinite(&sim.lbm.temperature);
                let pos_ux = first_nonfinite(&sim.lbm.ux);
                let pos_uy = first_nonfinite(&sim.lbm.uy);
                let pos_rho = first_nonfinite(&sim.lbm.rho);
                let f_max = sim
                    .lbm
                    .f
                    .iter()
                    .flatten()
                    .fold(0.0f64, |m, &v| m.max(v.abs()));
                let f_nan = sim.lbm.f.iter().flatten().any(|v| !v.is_finite());
                let g_max = sim
                    .lbm
                    .g
                    .iter()
                    .flatten()
                    .fold(0.0f64, |m, &v| m.max(v.abs()));
                let g_nan = sim.lbm.g.iter().flatten().any(|v| !v.is_finite());
                let (field, pos) = [
                    ("temperature", pos_t),
                    ("ux", pos_ux),
                    ("uy", pos_uy),
                    ("rho", pos_rho),
                ]
                .into_iter()
                .find(|(_, p)| p.is_some())
                .unwrap_or(("?", None));
                if let Some(k) = pos {
                    let x = k % nx;
                    let y = k / nx;
                    eprintln!(
                        "DIAG: premier champ non fini = {field} au nœud k={k} (x={x}, y={y}/{ny}) ; \
                         f: max|f|={f_max:.6e} NaN={f_nan} ; g: max|g|={g_max:.6e} NaN={g_nan} ; \
                         bords: y_proche_paroi={}",
                        y.min(ny - 1 - y)
                    );
                }
                eprintln!("DIVERGENCE détectée au pas {step} (valeurs non finies).");
                std::process::exit(2);
            }
        }

        // Moyennage temporel après sample_start
        if step >= args.sample_start && (step - args.sample_start) % args.sample_every == 0 {
            averager.accumulate(&sim);
        }

        // Export périodique du snapshot (écriture atomique tmp + rename)
        if args.snapshot_every > 0 && (step + 1) % args.snapshot_every == 0 {
            let path = snap_out_path.as_deref().expect("snapshot path");
            let res = snapshot::write_snapshot_atomic(
                path,
                sim.nx(),
                sim.ny(),
                &sim.lbm.rho,
                &sim.lbm.ux,
                &sim.lbm.uy,
                &sim.lbm.temperature,
            );
            if let Err(e) = res {
                eprintln!("Erreur d'export du snapshot au pas {step} : {e}");
                std::process::exit(5);
            }
            println!("snapshot exporté au pas {step}");
        }
    }

    let elapsed = start.elapsed();
    println!(
        "Simulation terminée en {:.1} s ({:.1}k pas/s)",
        elapsed.as_secs_f64(),
        args.steps as f64 / elapsed.as_secs_f64() / 1000.0
    );

    if averager.n == 0 {
        eprintln!("Aucun échantillon moyenné : sample_start ≥ steps ?");
        std::process::exit(3);
    }
    println!("Moyennage sur {} échantillons.", averager.n);

    // --- Post-traitement sur les champs MOYENS ---
    let (ux_m, uy_m, t_m, nu_t_m, uu_m) = averager.means();
    let psi_m = stream_function_mean(&sim, &ux_m);

    // Moyenne en temps des Nusselt : recalcule sur les champs moyens
    let nuc = nusselt_from_mean(&sim, &t_m, true);
    let nuf = nusselt_from_mean(&sim, &t_m, false);

    // Profils w moyens sur les lignes z=0.5 et z=0.75
    let iy_mid = sim.ny() / 2;
    let iy_075 = 3 * sim.ny() / 4;
    let uy_mid: Vec<f64> = (0..sim.nx()).map(|ix| uy_m[sim.idx(ix, iy_mid)]).collect();
    let uy_075: Vec<f64> = (0..sim.nx()).map(|ix| uy_m[sim.idx(ix, iy_075)]).collect();
    let (wmax_mid, x_mid) = max_of(&uy_mid, sim.nx());
    let (wmax_075, x_075) = max_of(&uy_075, sim.nx());
    // BL de la paroi chaude (moitié gauche) : cible thèse x ≈ 0.004
    let (wmax_mid_hot, x_mid_hot) = max_of_range(&uy_mid, 1, sim.nx() / 2, sim.nx());
    let (wmax_075_hot, x_075_hot) = max_of_range(&uy_075, 1, sim.nx() / 2, sim.nx());

    // y+ sur champ instantané final
    let yplus = post::y_plus_hot(&sim);

    // S^1/2 : écart-type vitesse moyen (fluctuations)
    let n_nodes = sim.nx() * sim.ny();
    let mut s2_sum = 0.0;
    for k in 0..n_nodes {
        // u'_rms² = ⟨u²⟩ − ⟨u⟩²
        s2_sum += (uu_m[k] - ux_m[k] * ux_m[k] - uy_m[k] * uy_m[k]).max(0.0);
    }
    let s_half = (s2_sum / n_nodes as f64).sqrt();

    // --- Exports ---
    let out = Path::new(&args.out);
    if let Err(e) = std::fs::create_dir_all(out) {
        eprintln!("Impossible de créer {} : {e}", out.display());
        std::process::exit(4);
    }

    let write_res = (
        post::write_fields_csv(&out.join("fields_mean.csv"), &sim, &ux_m, &uy_m, &t_m, &psi_m, &nu_t_m),
        post::write_profiles_csv(&out.join("profiles_w.csv"), &sim, &uy_mid, &uy_075),
        post::write_vtk(&out.join("snapshot.vti"), &sim),
        snapshot::write_snapshot(
            &out.join("snapshot.bin"),
            sim.nx(),
            sim.ny(),
            &sim.lbm.rho,
            &sim.lbm.ux,
            &sim.lbm.uy,
            &sim.lbm.temperature,
        ),
        post::write_summary_json(
            &out.join("summary.json"),
            nuc,
            nuf,
            post::nusselt_volume(&sim),
            wmax_mid,
            x_mid,
            wmax_075,
            x_075,
            yplus,
            args.cs,
            args.ra,
            args.nx,
            args.ny,
            args.steps,
            averager.n,
        ),
        post::write_log(&out.join("progress.log"), &log_lines),
    );
    for r in [write_res.0, write_res.1, write_res.2, write_res.3, write_res.4, write_res.5] {
        if let Err(e) = r {
            eprintln!("Erreur d'écriture : {e}");
            std::process::exit(5);
        }
    }

    // --- LES dynamique : quantités C^ν_dyn/C^κ_dyn (§2.3.5, --dynamic-smag) ---
    // Écrites dans un fichier compagnon (post.rs inchangé — le summary.json
    // existant n'est pas altéré, champs existants préservés).
    if args.dynamic_smag {
        let dyn_json = format!(
            r#"{{
  "dynamic_smag": true,
  "smag_alpha": {},
  "eta_test": {},
  "nu_sm_min": {},
  "cs_dyn_mean": {:.6},
  "cs_dyn_max": {:.6},
  "kappa_dyn_mean": {:.6},
  "kappa_dyn_max": {:.6},
  "nu_t_mean": {:.6e},
  "nu_t_max": {:.6e}
}}"#,
            args.smag_alpha,
            args.eta_test,
            args.nu_sm_min,
            sim.cs_dyn_mean,
            sim.cs_dyn_max,
            sim.kappa_dyn_mean,
            sim.kappa_dyn_max,
            nu_t_m.iter().sum::<f64>() / nu_t_m.len() as f64,
            nu_t_m.iter().fold(0.0f64, |m, &v| m.max(v.abs())),
        );
        std::fs::write(out.join("dynamic_sgs.json"), dyn_json)
            .unwrap_or_else(|e| eprintln!("Warning : dynamic_sgs.json non écrit : {e}"));
        println!(
            "LES dynamique : cs_dyn_mean = {:.4} | cs_dyn_max = {:.4} (clamp [0, 0.04])",
            sim.cs_dyn_mean, sim.cs_dyn_max
        );
    }

    println!("--- Résultats (champs moyens) ---");
    let nu_target = post::nusselt_trias_2d(args.ra);
    let ecart_pct = ((nuc - nu_target) / nu_target).abs() * 100.0;
    println!(
        "Nuc = {nuc:.4} | Nuf = {nuf:.4} | cible Trias 2D (Ra = {:.2e}) = {nu_target:.2} | écart = {ecart_pct:.1}%",
        args.ra
    );
    println!(
        "w_max(z=0.5) = {wmax_mid:.5} en x = {x_mid:.5} (global) ; BL chaude : {wmax_mid_hot:.5} en x = {x_mid_hot:.5} (cible ≈ 0.23 en x ≈ 0.004)"
    );
    println!(
        "w_max(z=0.75) = {wmax_075:.5} en x = {x_075:.5} (global) ; BL chaude : {wmax_075_hot:.5} en x = {x_075_hot:.5} (cible ≈ 0.11 en x ≈ 0.005)"
    );
    println!("y+ (paroi chaude) = {yplus:.3} | S^(1/2) = {s_half:.4}");
    println!("Artefacts écrits dans {}", out.display());
}

/// Fonction de courant calculée sur le champ moyen u_x.
fn stream_function_mean(sim: &CavitySolver, ux: &[f64]) -> Vec<f64> {
    let nx = sim.nx();
    let ny = sim.ny();
    let mut psi = vec![0.0f64; nx * ny];
    for ix in 0..nx {
        let mut acc = 0.0;
        for iy in 1..ny {
            let u0 = ux[sim.idx(ix, iy - 1)];
            let u1 = ux[sim.idx(ix, iy)];
            acc += 0.5 * (u0 + u1);
            psi[iy * nx + ix] = acc;
        }
    }
    psi
}

/// Nusselt depuis un champ de température moyen (chaque nœud).
fn nusselt_from_mean(sim: &CavitySolver, t: &[f64], hot: bool) -> f64 {
    let nx = sim.nx();
    let ny = sim.ny();
    let mut sum = 0.0;
    for iy in 0..ny {
        let grad = if hot {
            t[sim.idx(1, iy)] - t[sim.idx(2, iy)]
        } else {
            t[sim.idx(nx - 3, iy)] - t[sim.idx(nx - 2, iy)]
        };
        sum += grad.abs();
    }
    (sum / ny as f64) * sim.lz / solver::DELTA_T
}

/// Max et position normalisée d'un profil 1D.
fn max_of(profile: &[f64], nx: usize) -> (f64, f64) {
    let mut best = 0.0f64;
    let mut best_i = 1usize;
    for (i, v) in profile.iter().enumerate().skip(1).take(nx.saturating_sub(2)) {
        if v.abs() > best {
            best = v.abs();
            best_i = i;
        }
    }
    (best, best_i as f64 / (nx - 1) as f64)
}

/// Max et position normalisée sur une plage [i_start, i_end) d'un profil 1D.
fn max_of_range(profile: &[f64], i_start: usize, i_end: usize, nx: usize) -> (f64, f64) {
    let mut best = 0.0f64;
    let mut best_i = i_start;
    for (i, v) in profile
        .iter()
        .enumerate()
        .skip(i_start)
        .take(i_end.saturating_sub(i_start))
    {
        if v.abs() > best {
            best = v.abs();
            best_i = i;
        }
    }
    (best, best_i as f64 / (nx - 1) as f64)
}
