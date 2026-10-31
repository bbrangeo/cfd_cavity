//! Cavité DHC 2D Ra=1e6 sur grille octree (Étape B, porte V2 — plan §3.1/§3.3).
//!
//! CLI minimale en std::env (aucune dépendance, style `cfd_cavity/src/cli.rs`).
//! Monitoring : bruit acoustique max|ρ−ρ̄| global et aux interfaces (claim
//! V2 : « ≤ bruit de la run uniforme même durée »), Nuc/Nuf, masse, garde-fous
//! identiques au crate uniforme.

use std::path::Path;
use std::time::Instant;

use cfd_cavity_octree::cavity::{nusselt_cold, nusselt_hot, nusselt_hot_fine, nusselt_hot_link0};
use cfd_cavity_octree::solver::CavitySolver;

/// Paramètres parsés (std::env, clé=valeur).
struct Args {
    /// Taille du domaine grossier (équiv. grille uniforme 2·bl_width × ny0).
    nx0: usize,
    ny0: usize,
    /// Largeur de BL fine (cellules grossières) par paroi verticale.
    bl_width: usize,
    ra: f64,
    pr: f64,
    steps: usize,
    sample_start: usize,
    sample_every: usize,
    progress_every: usize,
    out: String,
    jobs: usize,
    seed: u64,
    /// Variante rescaling ξ_odd exacte (1/2·τ⁻_fin/τ⁻_grossier, plan §1.2).
    odd_exact: bool,
    /// Nombre de niveaux : 2 = config V2 (L0+L1), 3 = escalier tanh V6
    /// (bande L2 dx/4 imbriquée, Étape F).
    levels: usize,
    /// Étape D : flag --dynamic-smag (booléen sans valeur, même sémantique
    /// que src/cli.rs du crate uniforme).
    dynamic_smag: bool,
    /// Étape D : exposant α de l'interpolation mixte (défaut 0.5).
    smag_alpha: f64,
    /// Étape D : borne νsm ≥ nu_sm_min·ν (défaut -0.1).
    nu_sm_min: f64,
    /// Étape D : ratio filtre test η = Δ/Δ̃ (défaut 2.0).
    eta_test: f64,
}

impl Default for Args {
    fn default() -> Self {
        // Grille V2 (plan §3.1) : équivalente 128×512 → grossier 64×256,
        // BL fine ×2 (8 cellules grossières = 16 fines) près des 2 parois.
        Self {
            nx0: 64,
            ny0: 256,
            bl_width: 8,
            ra: 1.0e6,
            pr: 0.71,
            steps: 200_000,
            sample_start: 100_000,
            sample_every: 100,
            progress_every: 1_000,
            out: String::from("output/runs_octree/ra1e6_octree"),
            jobs: 1,
            seed: 42,
            odd_exact: false,
            levels: 2,
            dynamic_smag: false,
            smag_alpha: 0.5,
            nu_sm_min: -0.1,
            eta_test: 2.0,
        }
    }
}

const USAGE: &str = "\
Usage: cfd_cavity_octree [OPTIONS]

Options (clé=valeur) :
  --nx0 <usize>           largeur grossière              [défaut: 64]
  --ny0 <usize>           hauteur grossière              [défaut: 256]
  --bl-width <usize>      largeur BL fine (cellules grossières) [défaut: 8]
  --ra <f64>              Rayleigh cible                 [défaut: 1e6]
  --pr <f64>              Prandtl                        [défaut: 0.71]
  --steps <usize>         pas de temps                   [défaut: 200000]
  --sample-start <usize>  début du moyennage Nuc         [défaut: 100000]
  --sample-every <usize>  stride de moyennage            [défaut: 100]
  --progress-every <usize> cadence d'affichage           [défaut: 1000]
  --out <path>            répertoire de sortie           [défaut: output/runs_octree/ra1e6_octree]
  --jobs <usize>          threads de calcul (1=mono)     [défaut: 1]
  --seed <u64>            graine RNG                     [défaut: 42]
  --odd-exact             variante ξ_odd exacte ((1/2)·τ⁻_fin/τ⁻_grossier,
                          plan §1.2/§4) ; défaut : ξ_odd = 1/2
  --levels <usize>        niveaux : 2 = config V2, 3 = escalier tanh V6
                          (bande L2 dx/4, Étape F)  [défaut: 2]
  --dynamic-smag          modèle d'échelles mixtes dynamiques (Étape D,
                          porte V5) ; défaut : OFF (booléen sans valeur)
  --smag-alpha <f64>      exposant α de l'interpolation mixte [défaut: 0.5]
  --nu-sm-min <f64>       borne νsm ≥ nu_sm_min·ν [défaut: -0.1]
  --eta-test <f64>        ratio filtre test η = Δ/Δ̃ [défaut: 2.0]
";

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let arg = &argv[i];
        let mut split = arg.splitn(2, '=');
        let key = split.next().unwrap_or_default().trim_start_matches('-').to_string();
        let inline_val = split.next().map(str::to_string);

        // Flag booléen SANS valeur (traité avant l'extraction de valeur).
        if key == "odd-exact" || key == "odd_exact" {
            match inline_val {
                Some(v) if v == "true" || v == "1" => args.odd_exact = true,
                Some(v) if v == "false" || v == "0" => args.odd_exact = false,
                Some(v) => return Err(format!("--odd-exact : booléen attendu, lu '{v}'")),
                None => args.odd_exact = true,
            }
            i += 1;
            continue;
        }
        // Étape D : flag booléen SANS valeur --dynamic-smag (même sémantique
        // que src/cli.rs du crate uniforme : présence = activé).
        if key == "dynamic-smag" || key == "dynamic_smag" {
            match inline_val {
                Some(v) if v == "true" || v == "1" => args.dynamic_smag = true,
                Some(v) if v == "false" || v == "0" => args.dynamic_smag = false,
                Some(v) => return Err(format!("--dynamic-smag : booléen attendu, lu '{v}'")),
                None => args.dynamic_smag = true, // présence = activé
            }
            i += 1;
            continue;
        }

        let value = match inline_val {
            Some(v) => v,
            None => {
                i += 1;
                argv.get(i).cloned().ok_or_else(|| format!("valeur manquante pour --{key}"))?
            }
        };
        match key.as_str() {
            "nx0" => args.nx0 = value.parse().map_err(|_| format!("--nx0 : entier attendu, lu '{value}'"))?,
            "ny0" => args.ny0 = value.parse().map_err(|_| format!("--ny0 : entier attendu, lu '{value}'"))?,
            "bl-width" | "bl_width" => {
                args.bl_width = value.parse().map_err(|_| format!("--bl-width : entier attendu, lu '{value}'"))?
            }
            "levels" => args.levels = value.parse().map_err(|_| format!("--levels : entier attendu, lu '{value}'"))?,
            "smag-alpha" | "smag_alpha" => {
                args.smag_alpha = value
                    .parse()
                    .map_err(|_| format!("--smag-alpha : flottant attendu, lu '{value}'"))?
            }
            "nu-sm-min" | "nu_sm_min" => {
                args.nu_sm_min = value
                    .parse()
                    .map_err(|_| format!("--nu-sm-min : flottant attendu, lu '{value}'"))?
            }
            "eta-test" | "eta_test" => {
                args.eta_test = value
                    .parse()
                    .map_err(|_| format!("--eta-test : flottant attendu, lu '{value}'"))?
            }
            "ra" => args.ra = value.parse().map_err(|_| format!("--ra : flottant attendu, lu '{value}'"))?,
            "pr" => args.pr = value.parse().map_err(|_| format!("--pr : flottant attendu, lu '{value}'"))?,
            "steps" => args.steps = value.parse().map_err(|_| format!("--steps : entier attendu, lu '{value}'"))?,
            "sample-start" | "sample_start" => {
                args.sample_start = value.parse().map_err(|_| format!("--sample-start : entier attendu, lu '{value}'"))?
            }
            "sample-every" | "sample_every" => {
                args.sample_every = value.parse().map_err(|_| format!("--sample-every : entier attendu, lu '{value}'"))?
            }
            "progress-every" | "progress_every" => {
                args.progress_every = value.parse().map_err(|_| format!("--progress-every : entier attendu, lu '{value}'"))?
            }
            "out" => args.out = value,
            "jobs" | "j" => args.jobs = value.parse().map_err(|_| format!("--jobs : entier attendu, lu '{value}'"))?,
            "seed" => args.seed = value.parse().map_err(|_| format!("--seed : entier attendu, lu '{value}'"))?,
            "help" | "h" => return Err(String::from("help")),
            other => return Err(format!("option inconnue '--{other}'")),
        }
        i += 1;
    }
    if args.nx0 < 16 || args.ny0 < 16 {
        return Err("nx0 et ny0 doivent être ≥ 16".into());
    }
    if args.ny0 <= args.nx0 {
        return Err("cavité haute : ny0 doit être > nx0 (Az ≈ 4)".into());
    }
    if !(args.bl_width >= 2 && 2 * args.bl_width < args.nx0) {
        return Err("bl-width doit être ≥ 2 et 2·bl-width < nx0 (noyau grossier requis)".into());
    }
    if args.ra <= 0.0 || args.pr <= 0.0 {
        return Err("ra et pr doivent être > 0".into());
    }
    // Étape D : validations copiées du crate uniforme (src/cli.rs).
    if !(args.smag_alpha >= 0.0 && args.smag_alpha <= 1.0) || !args.smag_alpha.is_finite() {
        return Err("smag-alpha doit être ∈ [0, 1] (0 = Smagorinsky pur, 1 = similarité pure)".into());
    }
    if !(args.eta_test >= 1.0) || !args.eta_test.is_finite() {
        return Err("eta-test doit être ≥ 1 (ratio filtre test/grille)".into());
    }
    if !(args.levels == 2 || args.levels == 3) {
        return Err("levels doit valoir 2 (config V2) ou 3 (escalier tanh V6)".into());
    }
    if args.levels == 3 && args.bl_width != 4 {
        eprintln!("Note : escalier V6 conçu pour bl-width = 4 cellules grossières (frontières tanh §0.1) ; lu {}", args.bl_width);
    }
    if args.sample_every == 0 {
        return Err("sample-every doit être ≥ 1".into());
    }
    if args.jobs == 0 {
        return Err("jobs doit être ≥ 1".into());
    }
    Ok(args)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) if e == "help" => {
            eprint!("{USAGE}");
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("Erreur : {e}\n\n{USAGE}");
            std::process::exit(1);
        }
    };

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

    // Pool rayon configuré avant toute boucle parallèle (--jobs N), même
    // pattern que le crate uniforme : cap au nombre de cœurs avec warning.
    if args.jobs > 1 {
        if let Err(e) = rayon::ThreadPoolBuilder::new()
            .num_threads(args.jobs)
            .build_global()
        {
            // Déjà initialisé (tests ou double appel) : ignoré silencieusement.
            eprintln!("Note : pool rayon déjà configuré ({e})");
        }
    }

    let mut sim = CavitySolver::new(
        args.nx0,
        args.ny0,
        args.bl_width,
        args.ra,
        args.pr,
        cfd_cavity_octree::cavity::UTARGET,
        args.seed,
        args.jobs,
        args.odd_exact,
        args.levels,
    );
    // Étape D : modèle d'échelles mixtes dynamiques (porte V5) — setter du
    // solveur, même sémantique que le crate uniforme.
    sim.set_dynamic_sgs(args.dynamic_smag, args.smag_alpha, args.nu_sm_min, args.eta_test);
    if args.dynamic_smag {
        println!(
            "LES dynamique (Étape D) : alpha_mixte = {} | nu_sm_min = {} | eta_test = {}",
            args.smag_alpha, args.nu_sm_min, args.eta_test
        );
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

            // Garde-fous identiques au crate uniforme (max_u, finitude) :
            // sortie gracieuse avec log AVANT divergence NaN.
            if let Some(abort_msg) = sim.check_guards() {
                eprintln!("{abort_msg}");
                log_lines.push(abort_msg);
                let _ = std::fs::write(out.join("progress.log"), log_lines.join("\n") + "\n");
                std::process::exit(2);
            }
        }
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
