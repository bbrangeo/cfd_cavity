//! Cavité différentiellement chauffée 2D — benchmark thèse (§2.5, Tab. 2.6/2.7).
//!
//! Solveur LBM D2Q9 double distribution avec Smagorinsky LES, construit sur la
//! crate `oxiphysics-lbm` (structure ThermalD2Q9 + équilibres + fonctions physiques),
//! pas de temps / CL / post-traitement maison.

mod cli;
mod octree;
mod uniform;

use cli::{Args, Backend, Mode};
use std::path::Path;
use std::time::Instant;

use uniform::gpu::GpuSolver;
use uniform::post::{self, TimeAverager};
use uniform::snapshot;
use uniform::solver::{self, CavitySolver};

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

    // Dispatch unifié des deux solveurs (ancien binaire cfd_cavity_octree
    // fusionné derrière --mode octree).
    match args.mode {
        Mode::Uniform => run_uniform(args),
        Mode::Octree => run_octree(args),
    }
}

/// Solveur uniforme de référence (ancien point d'entrée unique du crate,
/// logique inchangée).
fn run_uniform(args: Args) {
    /// Garde-fou de stabilité : abort si max_u dépasse cette valeur.
    /// Ma_local = max_u·√3 ≈ 0.17 à la limite — marge ×3 sous les divergences
    /// observées (max_u ≈ 0.34, Ma_local ≈ 0.58 → NaN).
    const MAX_U_ABORT: f64 = 0.10;

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
    if args.backend == Backend::Gpu {
        println!("Backend : GPU wgpu (Étape A, porte G1 — pas de temps uniforme en f32)");
    }

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

    // --- Backend GPU (Étape A, porte G1) : init après le --restart éventuel
    // (les populations partent des buffers du solveur CPU). Erreur claire si
    // aucun adaptateur disponible — pas de fallback silencieux vers le CPU.
    let mut gpu: Option<GpuSolver> = if args.backend == Backend::Gpu {
        Some(GpuSolver::from_solver(&sim).unwrap_or_else(|e| {
            eprintln!("Erreur : {e}");
            std::process::exit(7);
        }))
    } else {
        None
    };
    if let Some(ref g) = gpu {
        println!("Backend GPU initialisé : {}", g.adapter_name());
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
        // Backend GPU : 3 dispatches (collide → stream → macroscopic) puis
        // réinjection du macroscopique/ν_t dans les champs du solveur CPU —
        // le pas CPU (`sim.step()`) n'est PAS exécuté, seuls les champs sont
        // mis à jour (post.rs, TimeAverager et snapshots inchangés).
        //
        // Le readback GPU→CPU (latence non négligeable) n'est effectué que
        // les pas où les champs sont consommés : diagnostics (progress-every),
        // moyennage (sample-every), export snapshot (snapshot-every).
        let gpu = gpu.as_mut();
        if let Some(g) = gpu {
            g.step(&sim);
            let need_sync = step % args.progress_every == 0
                || step + 1 == args.steps
                || (step >= args.sample_start
                    && (step - args.sample_start) % args.sample_every == 0)
                || (args.snapshot_every > 0 && (step + 1) % args.snapshot_every == 0);
            if need_sync {
                if let Err(e) = g.sync_into(&mut sim) {
                    eprintln!("Erreur de lecture GPU au pas {step} : {e}");
                    std::process::exit(7);
                }
            }
        } else {
            sim.step();
        }

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
        "Simulation terminée en {:.1} s ({:.1}k pas/s) | MLUPS = {:.1}",
        elapsed.as_secs_f64(),
        args.steps as f64 / elapsed.as_secs_f64() / 1000.0,
        args.steps as f64 * (args.nx * args.ny) as f64 / elapsed.as_secs_f64() / 1.0e6
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

/// Solveur octree multi-niveaux (ancien crate `cfd_cavity_octree`, logique
/// inchangée — mêmes flags, mêmes défauts, mêmes sorties que l'ancien binaire).
fn run_octree(args: Args) {
    use octree::cavity::{nusselt_cold, nusselt_hot, nusselt_hot_fine, nusselt_hot_link0, UTARGET};
    use octree::solver::CavitySolver as OctreeSolver;

    println!("=== Cavité DHC 2D octree (Étape B, porte V2) ===");
    println!(
        "Grossier : {}×{} | BL fine : {} cellules grossières ×2 parois (équiv. uniforme {}×{}) | Ra = {:.3e} | Pr = {} | jobs = {} | ξ_odd = {}",
        args.nx0,
        args.ny0,
        args.bl_width,
        2 * args.bl_width,
        args.ny0,
        args.ra,
        args.pr,
        args.jobs,
        if args.odd_exact { "exact (τ⁻)" } else { "1/2 (approx.)" }
    );
    // Étape H : calage ν (--utarget / --nu override).
    if let Some(n) = args.nu {
        println!("Étape H : nu forcé = {n} (bypass calage utarget = {})", args.utarget);
    } else if args.utarget != UTARGET {
        println!("Étape H : calage ν via utarget = {}", args.utarget);
    }
    if args.nu_start_mult > 1.0 {
        // Étape H : rampe exponentielle (τ_ramp ≈ 1000 pas, terminée à
        // ~5000 pas) + filet P3 : rampe beta linéaire sur spin_up_steps =
        // max(2000, ny0/10) pas.
        let spin_up = (args.ny0 / 10).max(2000);
        println!(
            "Étape H : rampe de viscosité exponentielle nu-start-mult = {} (tau_ramp = 1000 pas, terminée ~5000 pas) + rampe beta linéaire sur {} pas (spin-up)",
            args.nu_start_mult, spin_up
        );
        // Contrainte 3 : la rampe ν (≈ 5000 pas) doit être terminée bien
        // avant la fenêtre de moyennage.
        if args.sample_start < 5000 {
            eprintln!(
                "WARNING : rampe de viscosité terminée ~5000 pas mais sample-start = {} : augmentez --sample-start (> 5000) pour ne pas moyenner pendant la rampe",
                args.sample_start
            );
        }
    }

    // Pool rayon configuré avant toute boucle parallèle (--jobs N), même
    // pattern que le solveur uniforme : cap au nombre de cœurs avec warning.
    if args.jobs > 1 {
        if let Err(e) = rayon::ThreadPoolBuilder::new()
            .num_threads(args.jobs)
            .build_global()
        {
            // Déjà initialisé (tests ou double appel) : ignoré silencieusement.
            eprintln!("Note : pool rayon déjà configuré ({e})");
        }
    }

    let mut sim = OctreeSolver::new(
        args.nx0,
        args.ny0,
        args.bl_width,
        args.ra,
        args.pr,
        args.utarget,
        args.seed,
        args.jobs,
        args.odd_exact,
        args.levels,
        args.nu_start_mult,
        args.nu,
    );
    // Étape D : modèle d'échelles mixtes dynamiques (porte V5) — setter du
    // solveur, même sémantique que le solveur uniforme.
    sim.set_dynamic_sgs(args.dynamic_smag, args.smag_alpha, args.nu_sm_min, args.eta_test);
    if args.dynamic_smag {
        println!(
            "LES dynamique (Étape D) : alpha_mixte = {} | nu_sm_min = {} | eta_test = {}",
            args.smag_alpha, args.nu_sm_min, args.eta_test
        );
    }

    // --- Étape G : option --restart (initialisation depuis un snapshot
    // binaire multi-niveaux, ancien main.rs l.375-390) ---
    if let Some(ref snap_path) = args.restart {
        let snap = octree::snapshot::read_snapshot(Path::new(snap_path))
            .unwrap_or_else(|e| {
                eprintln!("Erreur de lecture du snapshot {snap_path} : {e}");
                std::process::exit(6);
            });
        println!(
            "Restart depuis {snap_path} : snapshot levels={} ({} grilles), pas {} — injection des champs du solveur octree",
            snap.levels,
            snap.grids.len(),
            snap.step
        );
        if let Err(e) = sim.set_from_snapshot(&snap) {
            eprintln!("Erreur de restart : {e}");
            std::process::exit(6);
        }
    }

    println!(
        "Unités réseau : nu = {:.6} | alpha = {:.6} | tau+ = {:.4}/{:.4}{} | beta_eff = {:.4e}/{:.4e}{} | {} niveaux | {} nœuds (grossier {} + fins {})",
        sim.nu_lu,
        sim.alpha_lu,
        sim.tau_coarse.tau_plus,
        sim.tau_fine.tau_plus,
        if args.levels == 3 {
            format!("/{:.4}", sim.blocks[0].sub.as_ref().unwrap().tau.tau_plus)
        } else {
            String::new()
        },
        sim.beta_eff_coarse,
        sim.beta_eff_fine,
        if args.levels == 3 {
            format!("/{:.4e}", 0.25 * sim.beta_eff_coarse)
        } else {
            String::new()
        },
        args.levels,
        sim.n_nodes_total(),
        args.nx0 * args.ny0,
        sim.n_nodes_total() - args.nx0 * args.ny0
    );
    println!(
        "Ra vérifié depuis unités réseau (grossier) : {:.4e} (cible {:.4e})",
        sim.beta_eff_coarse * ((sim.ny0 - 1) as f64).powi(3) / (sim.nu_lu * sim.alpha_lu),
        args.ra
    );

    let out = Path::new(&args.out);
    if let Err(e) = std::fs::create_dir_all(out) {
        eprintln!("Impossible de créer {} : {e}", out.display());
        std::process::exit(4);
    }

    let mut log_lines: Vec<String> = Vec::new();
    // Moyennage Nuc : somme + n d'échantillons (fenêtre sample_start→fin).
    let mut nuc_sum = 0.0f64;
    let mut nuf_sum = 0.0f64;
    let mut nuc_n = 0usize;
    // Estimateur like-for-like (1ère liaison ix=0–1, artefact mur — même
    // estimateur que la run uniforme de référence Nuc=1.3660).
    let mut nucl0_sum = 0.0f64;
    // Diagnostic Nu_fin : moyenné sur le bloc fin de paroi chaude (même
    // fenêtre), pour distinguer vrai champ vs corruption d'estimateur 2×2.
    let mut nufine_sum = 0.0f64;
    let hot_block_idx = sim.blocks.iter().position(|b| b.hot_wall).expect("bloc paroi chaude");
    let lx_coarse = (args.nx0 - 1) as f64;
    let mut noise_max = 0.0f64;
    let mut noise_iface_max = 0.0f64;
    let start = Instant::now();

    // --- Étape G : export périodique (--snapshot-every N, 0 = désactivé) ---
    let snap_out_path = if args.snapshot_every > 0 {
        Some(out.join("snapshot.bin"))
    } else {
        None
    };

    for step in 0..args.steps {
        sim.step_cavity();

        // Moyennage Nuc après sample_start (fenêtre sample-start→fin).
        // Diagnostic Nu_fin : moyenné sur le bloc fin de paroi chaude (même
        // fenêtre), pour distinguer vrai champ vs corruption d'estimateur.
        if step >= args.sample_start && (step - args.sample_start) % args.sample_every == 0 {
            nuc_sum += nusselt_hot(&sim.coarse);
            nuf_sum += nusselt_cold(&sim.coarse);
            nucl0_sum += nusselt_hot_link0(&sim.coarse);
            nufine_sum += nusselt_hot_fine(&sim.blocks[hot_block_idx].fine, lx_coarse);
            nuc_n += 1;
        }

        // Diagnostic + garde-fous toutes les progress-every itérations.
        if step % args.progress_every == 0 || step + 1 == args.steps {
            let noise = sim.noise_stats();
            noise_max = noise_max.max(noise.max_rho_dev);
            noise_iface_max = noise_iface_max.max(noise.max_rho_dev_ifaces);
            let nuc = nusselt_hot(&sim.coarse);
            let nuf = nusselt_cold(&sim.coarse);
            let nucl0 = nusselt_hot_link0(&sim.coarse);
            let nufine = nusselt_hot_fine(&sim.blocks[hot_block_idx].fine, lx_coarse);
            // Étape D : diagnostics LES dynamique en suffixe de ligne —
            // ne casse pas le format existant en mode OFF.
            let dyn_suffix = if args.dynamic_smag {
                format!(
                    " | cs_dyn = {:.4} | kap_dyn = {:.4} | nu_t_max = {:.3e}",
                    sim.cs_dyn_mean, sim.kappa_dyn_mean, sim.nu_t_max
                )
            } else {
                String::new()
            };
            let msg = format!(
                "step {:>7} | max_rho_dev = {:.3e} | max_rho_dev_ifaces = {:.3e} | masse = {:.6} | Nuc = {:.4} | Nuc_link0 = {:.4} | Nuf = {:.4} | Nu_fin = {:.4} | max_u = {:.5}{}",
                step, noise.max_rho_dev, noise.max_rho_dev_ifaces, noise.masse, nuc, nucl0, nuf, nufine, sim.max_u_all(),
                dyn_suffix
            );
            println!("{msg}");
            log_lines.push(msg);

            // Garde-fous identiques au solveur uniforme (max_u, finitude) :
            // sortie gracieuse avec log AVANT divergence NaN.
            if let Some(abort_msg) = sim.check_guards() {
                eprintln!("{abort_msg}");
                log_lines.push(abort_msg);
                let _ = std::fs::write(out.join("progress.log"), log_lines.join("\n") + "\n");
                std::process::exit(2);
            }
        }

        // --- Étape G : export périodique du snapshot (écriture atomique
        // tmp + rename) ---
        if let Some(path) = &snap_out_path {
            if (step + 1) % args.snapshot_every == 0 {
                let snap = sim.snapshot(step as u64);
                if let Err(e) = octree::snapshot::write_snapshot_atomic(path, &snap) {
                    eprintln!("Erreur d'export du snapshot au pas {step} : {e}");
                    std::process::exit(5);
                }
                println!("snapshot exporté au pas {step}");
            }
        }
    }

    // --- Étape G : export final du snapshot (si activé) ---
    if let Some(path) = &snap_out_path {
        let snap = sim.snapshot(args.steps as u64);
        if let Err(e) = octree::snapshot::write_snapshot_atomic(path, &snap) {
            eprintln!("Erreur d'export du snapshot final : {e}");
            std::process::exit(5);
        }
        println!("snapshot final exporté au pas {}", args.steps - 1);
    }

    let elapsed = start.elapsed();
    println!(
        "Simulation terminée en {:.1} s ({:.1}k pas/s)",
        elapsed.as_secs_f64(),
        args.steps as f64 / elapsed.as_secs_f64() / 1000.0
    );

    if nuc_n == 0 {
        eprintln!("Aucun échantillon moyenné : sample_start ≥ steps ?");
        std::process::exit(3);
    }
    let nuc_mean = nuc_sum / nuc_n as f64;
    let nuf_mean = nuf_sum / nuc_n as f64;
    let nucl0_mean = nucl0_sum / nuc_n as f64;
    let nufine_mean = nufine_sum / nuc_n as f64;

    // --- Références et verdict porte V2 ---
    // Like-for-like (2ᵉ liaison, estimateur Nuc standard) : baseline
    // uniforme Ra=1e6 = 2.3585 (output/runs_octree/ra1e6_full, grad 2ᵉ liaison).
    // 1ʳᵉ liaison (estimateur nusselt_hot_link0, artefact mur) : 1.3660.
    let nuc_ref = 2.3585;
    let nucl0_ref = 1.3660;
    let ecart_pct = ((nuc_mean - nuc_ref) / nuc_ref).abs() * 100.0;
    let portee_v2 = ecart_pct <= 5.0;

    let summary = format!(
        "--- Bilan (moyenné sur {nuc_n} échantillons, fenêtre {}→{}) ---
Nuc (2ᵉ liaison) = {nuc_mean:.4} | écart like-for-like vs {nuc_ref} = {ecart_pct:.1} %
Nuc_link0 (1ʳᵉ liaison) = {nucl0_mean:.4} | Nuf = {nuf_mean:.4} | Nu_fin (bloc chaud) = {nufine_mean:.4}
Références uniforme Ra=1e6 : 2ᵉ liaison = {nuc_ref} | 1ʳᵉ liaison = {nucl0_ref}
Bruit acoustique (max sur la run) : max|ρ−ρ̄| = {noise_max:.3e} | interfaces = {noise_iface_max:.3e}
PORTE V2 (écart Nuc ≤ 5 %) : {}",
        args.sample_start,
        args.steps,
        if portee_v2 { "PASS" } else { "FAIL" }
    );
    println!("{summary}");
    log_lines.push(summary);

    let _ = std::fs::write(out.join("progress.log"), log_lines.join("\n") + "\n");
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
