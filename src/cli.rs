//! Parsing des arguments CLI (à la main, aucune dépendance).
//!
//! CLI unifiée des deux solveurs : le flag `--mode uniform|octree` (défaut :
//! uniform) sélectionne le binaire logique. Les champs spécifiques au mode
//! octree (`--nx0`, `--ny0`, `--bl-width`, `--levels`, `--odd-exact`) sont
//! ignorés en mode uniform et vice-versa (`--nx`, `--ny`, ...).

/// Mode de solveur sélectionné par `--mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Solveur uniforme de référence (défaut).
    Uniform,
    /// Solveur octree multi-niveaux (ancien crate `cfd_cavity_octree`).
    Octree,
}

/// Backend de calcul sélectionné par `--backend` (mode uniform uniquement).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// CPU rayon (défaut, référence bitwise).
    Cpu,
    /// GPU wgpu (Étape A, porte G1 — solveur uniforme en f32).
    Gpu,
}

/// Paramètres de la simulation, parsés depuis `std::env::args`.
#[derive(Debug, Clone)]
pub struct Args {
    /// Mode de solveur (--mode uniform|octree, défaut : uniform).
    pub mode: Mode,
    /// Backend de calcul (--backend cpu|gpu, défaut : cpu, uniform uniquement).
    pub backend: Backend,
    // --- Champs uniformes (ignorés en mode octree) ---
    /// Nœuds en x (largeur de la cavité).
    pub nx: usize,
    /// Nœuds en y (hauteur, Az = (ny-1)/(nx-1) ≈ 4).
    pub ny: usize,
    /// Nombre de Rayleigh cible (basé sur la hauteur Lz).
    pub ra: f64,
    /// Nombre de Prandtl (air : 0.71).
    pub pr: f64,
    /// Constante de Smagorinsky Cs.
    pub cs: f64,
    /// Prandtl turbulent.
    pub pr_t: f64,
    /// Nombre total de pas de temps.
    pub steps: usize,
    /// Pas de début de moyennage temporel.
    pub sample_start: usize,
    /// Stride d'échantillonnage des moyennes.
    pub sample_every: usize,
    /// Répertoire de sortie.
    pub out: String,
    /// Vitesse max cible en unités de réseau (calage ν/α, voir solver::derive_lattice_units).
    pub utarget: f64,
    /// Graine du générateur aléatoire (perturbation initiale).
    pub seed: u64,
    /// Période d'affichage de la progression.
    pub progress_every: usize,
    /// Nombre de threads de calcul (--jobs). 1 = mono-thread, N>1 = rayon.
    pub jobs: usize,
    /// Multiplicateur de la diffusivité thermique α (τ_t amortit plus fort,
    /// levier de stabilisation Ra élevé ; 1.0 = α physique ν/Pr).
    pub alpha_mult: f64,
    /// Multiplicateur de la viscosité initiale (rampe exponentielle vers ν
    /// cible sur la durée de spin-up ; 1.0 = pas de rampe).
    pub nu_start_mult: f64,
    /// Snapshot binaire de départ (option --restart). Si None : initialisation
    /// classique T linéaire + perturbation.
    pub restart: Option<String>,
    /// Viscosité forcée (option --nu). Bypass du calage utarget ; permet de
    /// descendre sous le plancher ν=0.002 (τ+ proche de 0.5, stabilité TRT
    /// à valider) pour accéder à des grilles fines type SND à Ra élevé.
    pub nu: Option<f64>,
    /// Période d'export périodique du snapshot (--snapshot-every N).
    /// Toutes les N étapes, snapshot.bin est réécrit de manière atomique
    /// (tmp + rename) pour permettre la reprise d'une run interrompue.
    /// 0 = désactivé (défaut).
    pub snapshot_every: usize,
    /// Active le modèle d'échelles mixtes dynamiques (--dynamic-smag,
    /// §2.3.5 thèse). Défaut : off = Smagorinsky Cs fixe, rétrocompatibilité
    /// bitwise des runs existants.
    pub dynamic_smag: bool,
    /// Exposant α de l'interpolation mixte (défaut thèse Tab. 2.3 : 0.5 =
    /// moyenne géométrique Smagorinsky × similarité).
    pub smag_alpha: f64,
    /// Borne inférieure νsm ≥ nu_sm_min·ν (défaut thèse Ezzouhri : −0.1).
    pub nu_sm_min: f64,
    /// Ratio filtre test η = Δ/Δ̃ (défaut thèse : 2).
    pub eta_test: f64,
    // --- Champs octree (ignorés en mode uniform) ---
    /// Taille du domaine grossier (équiv. grille uniforme 2·bl_width × ny0).
    pub nx0: usize,
    /// Hauteur du domaine grossier.
    pub ny0: usize,
    /// Largeur de BL fine (cellules grossières) par paroi verticale.
    pub bl_width: usize,
    /// Variante rescaling ξ_odd exacte (1/2·τ⁻_fin/τ⁻_grossier, plan §1.2).
    pub odd_exact: bool,
    /// Nombre de niveaux : 2 = config V2 (L0+L1), 3 = escalier tanh V6
    /// (bande L2 dx/4 imbriquée, Étape F).
    pub levels: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mode: Mode::Uniform,
            backend: Backend::Cpu,
            nx: 64,
            ny: 256,
            ra: 1.0e6,
            pr: 0.71,
            cs: 0.10,
            pr_t: 0.9,
            steps: 200_000,
            sample_start: 100_000,
            sample_every: 100,
            out: String::from("output/runs_uniforme/ra1e6"),
            utarget: 0.02,
            seed: 42,
            progress_every: 1000,
            jobs: 1,
            alpha_mult: 1.0,
            nu_start_mult: 1.0,
            restart: None,
            nu: None,
            snapshot_every: 0,
            dynamic_smag: false,
            smag_alpha: 0.5,
            nu_sm_min: -0.1,
            eta_test: 2.0,
            // Champs octree (grille V2, plan §3.1 : équivalente 128×512 →
            // grossier 64×256, BL fine ×2 = 8 cellules grossières par paroi).
            nx0: 64,
            ny0: 256,
            bl_width: 8,
            odd_exact: false,
            levels: 2,
        }
    }
}

const USAGE: &str = "\
Usage: cavity_les [OPTIONS]

Mode de solveur :
  --mode <uniform|octree> solveur uniforme (défaut) ou octree multi-niveaux

Backend de calcul (mode --mode uniform uniquement) :
  --backend <cpu|gpu>     CPU rayon (défaut, référence bitwise) ou GPU wgpu
                          (Étape A, porte G1 — pas de temps uniforme en f32,
                          critère : écart Nuc ≤ 1 % vs CPU)

Options uniformes (mode --mode uniform) :
  --nx <usize>            largeur de la grille          [défaut: 64]
  --ny <usize>            hauteur de la grille          [défaut: 256]
  --ra <f64>              Rayleigh cible                [défaut: 1e6]
  --pr <f64>              Prandtl                       [défaut: 0.71]
  --cs <f64>              constante de Smagorinsky      [défaut: 0.1]
  --pr_t <f64>            Prandtl turbulent             [défaut: 0.9]
  --steps <usize>         pas de temps                  [défaut: 200000]
  --sample-start <usize>  début du moyennage            [défaut: 100000]
  --sample-every <usize>  stride de moyennage           [défaut: 100]
  --out <path>            répertoire de sortie          [défaut: output/runs_uniforme/ra1e6]
  --utarget <f64>         u_max cible (unités réseau)   [défaut: 0.02]
  --seed <u64>            graine RNG                    [défaut: 42]
  --progress-every <usize> cadence d'affichage          [défaut: 1000]
  --jobs <usize>          threads de calcul (1=mono)    [défaut: 1]
  --alpha-mult <f64>      multiplicateur α (stabilisation) [défaut: 1.0]
  --nu-start-mult <f64>   rampe viscosité initiale/ν     [défaut: 1.0]
  --restart <path>        snapshot binaire de départ (initialise u/T depuis
                          le champ ; grille quelconque, interpolation si ≠)
  --nu <f64>              viscosité forcée (bypass calage utarget ; descendre
                          sous le plancher 0.002 pour grilles fines type SND)
  --snapshot-every <usize> export atomique de snapshot.bin toutes les N étapes
                          [défaut: 0 = désactivé] ; reprise via --restart
  --dynamic-smag          active les échelles mixtes dynamiques (§2.3.5 thèse,
                          procédure de Lilly + test-filter, défaut : off = Cs fixe)
  --smag-alpha <f64>      exposant α de l'interpolation mixte [défaut: 0.5]
                          (0.5 = moyenne géométrique, Tab. 2.3)
  --nu-sm-min <f64>       borne νsm ≥ nu_sm_min·ν [défaut: -0.1]
  --eta-test <f64>        ratio filtre test η = Δ/Δ̃ [défaut: 2.0]

Options octree (mode --mode octree, ancien crate cfd_cavity_octree) :
  --nx0 <usize>           largeur grossière              [défaut: 64]
  --ny0 <usize>           hauteur grossière              [défaut: 256]
  --bl-width <usize>      largeur BL fine (cellules grossières) [défaut: 8]
  --odd-exact             variante ξ_odd exacte ((1/2)·τ⁻_fin/τ⁻_grossier,
                          plan §1.2/§4) ; défaut : ξ_odd = 1/2
  --levels <usize>        niveaux : 2 = config V2, 3 = escalier tanh V6
                          (bande L2 dx/4, Étape F)  [défaut: 2]

Options communes aux deux modes :
  --ra/--pr/--steps/--sample-start/--sample-every/--progress-every/--out/
  --jobs/--seed/--utarget/--nu-start-mult/--nu/--restart/--snapshot-every/
  --dynamic-smag/--smag-alpha/--nu-sm-min/--eta-test (mêmes défauts)
";

/// Texte d'aide complet.
pub fn usage() -> &'static str {
    USAGE
}

/// Parse les arguments de la ligne de commande.
/// Retourne une erreur descriptive (affichée par `main` avec l'usage).
pub fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();

    let mut i = 0;
    while i < argv.len() {
        let arg = &argv[i];
        let mut split = arg.splitn(2, '=');
        let key = split.next().unwrap_or_default().trim_start_matches('-').to_string();
        let inline_val = split.next().map(str::to_string);

        // Flag booléen SANS valeur : traité AVANT l'extraction de la valeur,
        // sinon le token suivant (--out, etc.) serait consommé comme valeur.
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
        // Flag booléen SANS valeur --odd-exact (mode octree, même sémantique).
        if key == "odd-exact" || key == "odd_exact" {
            match inline_val {
                Some(v) if v == "true" || v == "1" => args.odd_exact = true,
                Some(v) if v == "false" || v == "0" => args.odd_exact = false,
                Some(v) => return Err(format!("--odd-exact : booléen attendu, lu '{v}'")),
                None => args.odd_exact = true, // présence = activé
            }
            i += 1;
            continue;
        }

        // valeur soit inline (--key=val), soit au token suivant (--key val)
        let value = match inline_val {
            Some(v) => v,
            None => {
                i += 1;
                argv.get(i).cloned().ok_or_else(|| format!("valeur manquante pour --{key}"))?
            }
        };

        match key.as_str() {
            "mode" => match value.to_lowercase().as_str() {
                "uniform" => args.mode = Mode::Uniform,
                "octree" => args.mode = Mode::Octree,
                other => {
                    return Err(format!(
                        "--mode : 'uniform' ou 'octree' attendu, lu '{other}'"
                    ))
                }
            },
            "backend" => match value.to_lowercase().as_str() {
                "cpu" => args.backend = Backend::Cpu,
                "gpu" => args.backend = Backend::Gpu,
                other => {
                    return Err(format!(
                        "--backend : 'cpu' ou 'gpu' attendu, lu '{other}'"
                    ))
                }
            },
            "nx" => args.nx = value.parse().map_err(|_| format!("--nx : entier attendu, lu '{value}'"))?,
            "ny" => args.ny = value.parse().map_err(|_| format!("--ny : entier attendu, lu '{value}'"))?,
            "ra" => args.ra = value.parse().map_err(|_| format!("--ra : flottant attendu, lu '{value}'"))?,
            "pr" => args.pr = value.parse().map_err(|_| format!("--pr : flottant attendu, lu '{value}'"))?,
            "cs" => args.cs = value.parse().map_err(|_| format!("--cs : flottant attendu, lu '{value}'"))?,
            "pr_t" | "prt" => {
                args.pr_t = value.parse().map_err(|_| format!("--pr_t : flottant attendu, lu '{value}'"))?
            }
            "steps" => args.steps = value.parse().map_err(|_| format!("--steps : entier attendu, lu '{value}'"))?,
            "sample-start" | "sample_start" => {
                args.sample_start = value
                    .parse()
                    .map_err(|_| format!("--sample-start : entier attendu, lu '{value}'"))?
            }
            "sample-every" | "sample_every" => {
                args.sample_every = value
                    .parse()
                    .map_err(|_| format!("--sample-every : entier attendu, lu '{value}'"))?
            }
            "out" => args.out = value,
            "utarget" => {
                args.utarget = value.parse().map_err(|_| format!("--utarget : flottant attendu, lu '{value}'"))?
            }
            "seed" => args.seed = value.parse().map_err(|_| format!("--seed : entier attendu, lu '{value}'"))?,
            "progress-every" | "progress_every" => {
                args.progress_every = value
                    .parse()
                    .map_err(|_| format!("--progress-every : entier attendu, lu '{value}'"))?
            }
            "jobs" | "j" => {
                args.jobs = value
                    .parse()
                    .map_err(|_| format!("--jobs : entier attendu, lu '{value}'"))?
            }
            "alpha-mult" | "alpha_mult" => {
                args.alpha_mult = value
                    .parse()
                    .map_err(|_| format!("--alpha-mult : flottant attendu, lu '{value}'"))?
            }
            "nu-start-mult" | "nu_start_mult" => {
                args.nu_start_mult = value
                    .parse()
                    .map_err(|_| format!("--nu-start-mult : flottant attendu, lu '{value}'"))?
            }
            "restart" => args.restart = Some(value),
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
            "snapshot-every" | "snapshot_every" => {
                args.snapshot_every = value
                    .parse()
                    .map_err(|_| format!("--snapshot-every : entier attendu, lu '{value}'"))?
            }
            "nu" => {
                let v: f64 = value
                    .parse()
                    .map_err(|_| format!("--nu : flottant attendu, lu '{value}'"))?;
                if !(v > 0.0) || !v.is_finite() {
                    return Err("--nu doit être > 0 et fini".into());
                }
                if v >= 0.002 {
                    return Err(format!("--nu {v} ≥ 0.002 : inutile, le calage utarget suffit ; --nu est réservé au sous-plancher"));
                }
                args.nu = Some(v)
            }
            // --- Champs octree (mode --mode octree) ---
            "nx0" => args.nx0 = value.parse().map_err(|_| format!("--nx0 : entier attendu, lu '{value}'"))?,
            "ny0" => args.ny0 = value.parse().map_err(|_| format!("--ny0 : entier attendu, lu '{value}'"))?,
            "bl-width" | "bl_width" => {
                args.bl_width = value.parse().map_err(|_| format!("--bl-width : entier attendu, lu '{value}'"))?
            }
            "levels" => args.levels = value.parse().map_err(|_| format!("--levels : entier attendu, lu '{value}'"))?,
            "help" | "h" => return Err(String::from("help")),
            other => return Err(format!("option inconnue '--{other}'")),
        }
        i += 1;
    }

    if args.nx < 8 || args.ny < 8 {
        return Err("nx et ny doivent être ≥ 8".into());
    }
    // Backend GPU : le noyau GPU porte Smagorinsky Cs fixe ET le dynamic
    // Smagorinsky (Étape D, porte G4 : kernels dyn1-dyn4 + collide_dyn,
    // mêmes formules que `compute_dynamic_sgs`).
    if args.backend == Backend::Gpu && args.mode == Mode::Octree {
        return Err(
            "--backend gpu : non supporté en mode octree (Étape A = solveur uniforme uniquement) ; utilisez --backend cpu".into(),
        );
    }
    if args.ny <= args.nx {
        return Err("cavité haute : ny doit être > nx (Az ≈ 4)".into());
    }
    if args.ra <= 0.0 || args.pr <= 0.0 {
        return Err("ra et pr doivent être > 0".into());
    }
    if args.sample_every == 0 {
        return Err("sample-every doit être ≥ 1".into());
    }
    if args.jobs == 0 {
        return Err("jobs doit être ≥ 1".into());
    }
    if !(args.alpha_mult > 0.0) || !args.alpha_mult.is_finite() {
        return Err("alpha-mult doit être > 0 et fini".into());
    }
    if !(args.nu_start_mult >= 1.0) || !args.nu_start_mult.is_finite() {
        return Err("nu-start-mult doit être ≥ 1 et fini".into());
    }
    if !(args.smag_alpha >= 0.0 && args.smag_alpha <= 1.0) || !args.smag_alpha.is_finite() {
        return Err("smag-alpha doit être ∈ [0, 1] (0 = Smagorinsky pur, 1 = similarité pure)".into());
    }
    if !(args.eta_test >= 1.0) || !args.eta_test.is_finite() {
        return Err("eta-test doit être ≥ 1 (ratio filtre test/grille)".into());
    }
    // Validations spécifiques au mode octree (copiées de l'ancien crate
    // cfd_cavity_octree) — ignorées en mode uniform.
    if args.mode == Mode::Octree {
        if args.nx0 < 16 || args.ny0 < 16 {
            return Err("nx0 et ny0 doivent être ≥ 16".into());
        }
        if args.ny0 <= args.nx0 {
            return Err("cavité haute : ny0 doit être > nx0 (Az ≈ 4)".into());
        }
        if !(args.bl_width >= 2 && 2 * args.bl_width < args.nx0) {
            return Err("bl-width doit être ≥ 2 et 2·bl-width < nx0 (noyau grossier requis)".into());
        }
        if !(args.levels == 2 || args.levels == 3) {
            return Err("levels doit valoir 2 (config V2) ou 3 (escalier tanh V6)".into());
        }
        if args.levels == 3 && args.bl_width != 4 {
            eprintln!(
                "Note : escalier V6 conçu pour bl-width = 4 cellules grossières (frontières tanh §0.1) ; lu {}",
                args.bl_width
            );
        }
    }
    Ok(args)
}
