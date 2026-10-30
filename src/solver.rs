//! Solveur LBM D2Q9 double distribution (f, g) pour la cavité différentiellement
//! chauffée, avec modèle sous-maille de Smagorinsky.
//!
//! Structure de données, équilibres et fonctions physiques issus de la crate
//! `oxiphysics-lbm` (`thermal_lbm::{ThermalD2Q9, feq, geq, boussinesq_force, ...}`).
//! Le pas de temps (collision LES + streaming + conditions aux limites) est écrit
//! ici : le `streaming_step` de la crate est périodique dans les deux directions,
//! donc inadapté à une cavité fermée.

use oxiphysics_lbm::lattice::{D2Q9_OPPOSITES, D2Q9_VELOCITIES, D2Q9_WEIGHTS, CS2};
use oxiphysics_lbm::thermal_lbm::{feq, geq, ThermalD2Q9};
use rayon::prelude::*;

use crate::rng::Rng;

/// Température paroi chaude (gauche) — normalisation thèse θc = +0.5.
pub const T_HOT: f64 = 0.5;
/// Température paroi froide (droite) — normalisation thèse θf = −0.5.
pub const T_COLD: f64 = -0.5;
/// Écart de température ΔT = T_hot − T_cold = 1.
pub const DELTA_T: f64 = T_HOT - T_COLD;

/// Borne supérieure du coefficient dynamique C^ν_dyn / C^κ_dyn (clamp [0, 0.04],
/// comme la crate de référence oxiphysics-lbm : Cs²_max = 0.2²).
pub const C_DYN_MAX: f64 = 0.04;

/// Plancher TRT sur les temps de relaxation effectifs (τ⁺, τ_t⁺) : même marge
/// que l'assertion de construction (solver : `tau > 0.505`). En mode dynamique,
/// les bornes sur ν_t/α_t garantissent τ = 0.5 + 3·(ν + ν_t) ≥ 0.505 et
/// τ_t = 0.5 + 3·(α + α_t) ≥ 0.505, quel que soit nu_sm_min.
pub const TAU_TRT_FLOOR: f64 = 0.505;

/// Solveur de cavité : enveloppe la structure `ThermalD2Q9` de la crate et
/// implémente le pas de temps LES maison.
pub struct CavitySolver {
    /// Structure LBM de la crate (champs publics manipulés directement).
    pub lbm: ThermalD2Q9,
    /// Hauteur physique Lz = ny − 1 (en unités de réseau).
    pub lz: f64,
    /// Largeur physique Lx = nx − 1.
    pub lx: f64,
    /// Force de flottabilité effective beta_eff = g·β·ΔT (Ra·ν·α/Lz³).
    pub beta_eff: f64,
    /// Viscosité moléculaire ν.
    pub nu: f64,
    /// Diffusivité thermique α.
    pub alpha: f64,
    /// Diffusivité thermique PHYSIQUE α_phys = ν/Pr (avant amortissement par
    /// `alpha_mult`). La flottabilité et le Ra vérifié dérivent de cette
    /// valeur, pas du α amorti.
    pub alpha_phys: f64,
    /// Constante de Smagorinsky.
    pub cs: f64,
    /// Prandtl turbulent Pr_t.
    pub pr_t: f64,
    /// Viscosité turbulente ν_t (dernier pas, pour post-traitement).
    pub nu_t: Vec<f64>,
    /// Force de flottabilité locale fy (dernière collision, par nœud) —
    /// réutilisée par `update_macroscopic` pour la correction de vitesse de
    /// Guo et al. 2002, éq. (16) : la F de la correction DOIT être la même
    /// que celle de la collision (évaluée aux champs du temps t), d'où le
    /// stockage plutôt qu'un recalcul après mise à jour.
    pub(crate) fy_node: Vec<f64>,
    /// Nombre de threads de calcul (≥ 1). 1 = chemin mono-thread d'origine.
    pub jobs: usize,
    /// Compteur de pas de temps (pour la rampe de spin-up).
    pub time_step: usize,
    /// Durée de la rampe de spin-up de la flottabilité (pas de temps).
    pub spin_up_steps: usize,
    /// Multiplicateur de viscosité initial (rampe exponentielle 0→spin_up_steps).
    pub nu_start_mult: f64,
    /// Pommeur (PRNG xorshift64*), utilisé pour la perturbation initiale.
    #[allow(dead_code)]
    rng: Rng,

    // --- Modèle d'échelles mixtes dynamiques (§2.3.5 thèse, --dynamic-smag) ---
    /// Active la procédure dynamique (défaut : off = Smagorinsky Cs fixe,
    /// chemin de code bitwise identique aux runs existants).
    pub dynamic_smag: bool,
    /// Exposant α de l'interpolation mixte (défaut thèse Tab. 2.3 : 0.5 =
    /// moyenne géométrique Smagorinsky × similarité).
    pub smag_alpha: f64,
    /// Borne inférieure νsm ≥ nu_sm_min·ν (défaut thèse Ezzouhri : −0.1).
    pub nu_sm_min: f64,
    /// Ratio filtre test η = Δ/Δ̃ (défaut thèse : 2).
    pub eta_test: f64,
    /// ν_t précalculé par nœud (mode dynamique uniquement) — transmis à
    /// `collide_node` qui l'utilise tel quel à la place de (Cs·Δx)²·|S|.
    pub(crate) nu_t_pre: Option<Vec<f64>>,
    /// α_t précalculé par nœud (mode dynamique uniquement) — diffusivité
    /// turbulente du scalaire g (branche τ_t⁺ = 0.5 + 3(α + α_t), le patch
    /// τ_t⁻ = τ_t⁺ est inchangé).
    pub(crate) alpha_t_pre: Option<Vec<f64>>,
    /// Diagnostic : moyenne spatiale de C^ν_dyn (dernier pas dynamique).
    pub cs_dyn_mean: f64,
    /// Diagnostic : max spatial de C^ν_dyn (dernier pas dynamique).
    pub cs_dyn_max: f64,
    /// Diagnostic : moyenne spatiale de C^κ_dyn (dernier pas dynamique) —
    /// κsm = C^κ_dyn·κsm_mix est branché dans la diffusion du scalaire g
    /// (Éq. 2.80 : κ_total = alpha + κsm).
    pub kappa_dyn_mean: f64,
    /// Diagnostic : max spatial de C^κ_dyn (dernier pas dynamique).
    pub kappa_dyn_max: f64,
}

impl CavitySolver {
    /// Crée le solveur et initialise : T linéaire en x, u = 0, perturbation ±1e-3 sur T.
    pub fn new(nx: usize, ny: usize, ra: f64, pr: f64, cs: f64, pr_t: f64, utarget: f64, seed: u64, jobs: usize, alpha_mult: f64, nu_start_mult: f64, nu_override: Option<f64>) -> Self {

        // --- unités de réseau calées pour matcher Ra et Pr ---
        // Lz = ny - 1. L'échelle de vitesse de flottabilité est
        //   u_scale = sqrt(beta_eff · Lz) avec beta_eff = Ra·ν·α/Lz³,
        //   Pr = ν/α  ⇒  u_scale = sqrt(Ra)·ν/(sqrt(Pr)·Lz)   (linéaire en ν),
        // et u_max ≈ c·u_scale avec c ≈ 0.3 en cavité différentiellement
        // chauffée (mesuré : u_max/u_scale ≈ 0.31 sur la run Ra=1e6 validée).
        // Le calage empirique retenu (calibré pour reproduire ν ≈ 0.0186 sur
        // la run Ra=1e6 validée avec utarget=0.02) est :
        //   ν = (utarget·sqrt(Pr)·Lz / (c·sqrt(Ra)))^{2/3},  c ≈ 1.7
        // Plancher : ν ≥ 0.002 pour τ+ ≥ 0.506 — tenable uniquement avec la
        // collision TRT (Λ = 3/16) ; la collision BGK exigerait ν ≥ 0.018
        // (τ ≥ 0.554), ce qui rend Ra=1e8 instable (u_max ≈ 0.25 > limite de
        // compressibilité ≈ 0.15).
        let lz = (ny - 1) as f64;
        let lx = (nx - 1) as f64;

        let c_scale = 1.7;
        let nu_cal = (utarget * pr.sqrt() * lz / (c_scale * ra.sqrt())).powf(2.0 / 3.0);
        let nu = match nu_override {
            Some(n) => {
                assert!(n > 0.0 && n.is_finite(), "--nu doit être > 0 et fini");
                // ν forcé : τ+ = 0.5 + 3ν doit rester > 0.5 strictement.
                assert!(0.5 + 3.0 * n > 0.5005, "--nu {n} trop bas : tau+={:.5} proche de 0.5", 0.5 + 3.0 * n);
                n
            }
            None => nu_cal.max(0.002), // τ+ ≥ 0.506 (stabilité TRT ; BGK exigerait 0.018)
        };
        let alpha_phys = nu / pr;
        let alpha = alpha_phys * alpha_mult;
        // beta_eff dérivé de la valeur PHYSIQUE de α (ν/Pr), PAS du α amorti :
        // la flottabilité est une donnée physique, indépendante du levier de
        // stabilisation thermique (sinon alpha-mult augmente aussi u_ff et
        // s'autodétruit).
        let beta_eff = ra * nu * alpha_phys / lz.powi(3);

        // tau = 0.5 + 3*nu ; contrainte LBM : tau pas trop proche de 0.5.
        // Assertion dure (τ+ ≥ 0.506) uniquement sur le chemin calage utarget
        // ; avec --nu forcé, l'utilisateur a explicitement choisi de descendre
        // sous le plancher TRT (seul τ+ > 0.5 strict est exigé).
        let tau = 0.5 + 3.0 * nu;
        let tau_t = 0.5 + 3.0 * alpha;
        if nu_override.is_none() {
            assert!(tau > 0.505, "tau={tau} trop proche de 0.5 : augmente --utarget ou réduis la grille (plancher TRT : tau+ ≥ 0.506, soit ν ≥ 0.002)");
        }
        assert!(tau_t > 0.505, "tau_t={tau_t} trop proche de 0.5 : ajuste --alpha-mult (Pr trop grand pour cette vitesse cible)");

        let mut lbm = ThermalD2Q9::new(nx, ny, tau, tau_t, 1.0, 0.0, -1.0);
        let mut rng = Rng::new(seed);

        // --- initialisation : T linéaire en x + perturbation ±1e-3 ---
        for iy in 0..ny {
            for ix in 0..nx {
                let k = iy * nx + ix;
                let xf = ix as f64 / (nx - 1) as f64;
                let t_lin = T_HOT + (T_COLD - T_HOT) * xf;
                let pert = 1.0e-3 * (2.0 * rng.next_f64() - 1.0);
                let t = t_lin + pert;
                lbm.temperature[k] = t;
                lbm.rho[k] = 1.0;
                lbm.ux[k] = 0.0;
                lbm.uy[k] = 0.0;
                for i in 0..9 {
                    lbm.f[k][i] = feq(i, 1.0, 0.0, 0.0);
                    lbm.g[k][i] = geq(i, t, 0.0, 0.0);
                }
            }
        }

        Self {
            lbm,
            lz,
            lx,
            beta_eff,
            nu,
            alpha,
            alpha_phys,
            cs,
            pr_t,
            nu_t: vec![0.0; nx * ny],
            fy_node: vec![0.0; nx * ny],
            jobs: jobs.max(1),
            time_step: 0,
            // Rampe de spin-up : 10 % du nombre de zones verticales, bornée
            // [200, 2000] pas — suffisant pour amortir l'overshoot initial.
            spin_up_steps: ((ny / 10).max(200)).min(2000),
            nu_start_mult: nu_start_mult.max(1.0),
            rng,
            // Échelles mixtes dynamiques : désactivées par défaut (bitwise
            // identique aux runs existants) ; activées via set_dynamic_sgs.
            dynamic_smag: false,
            smag_alpha: 0.5,
            nu_sm_min: -0.1,
            eta_test: 2.0,
            nu_t_pre: None,
            alpha_t_pre: None,
            cs_dyn_mean: 0.0,
            cs_dyn_max: 0.0,
            kappa_dyn_mean: 0.0,
            kappa_dyn_max: 0.0,
        }
    }

    /// Active/configure le modèle d'échelles mixtes dynamiques (§2.3.5 thèse).
    /// Appelé par `main` depuis les options CLI (--dynamic-smag et associées) ;
    /// jamais appelé en mode OFF ⇒ chemin de collision bitwise inchangé.
    pub fn set_dynamic_sgs(&mut self, dynamic_smag: bool, smag_alpha: f64, nu_sm_min: f64, eta_test: f64) {
        self.dynamic_smag = dynamic_smag;
        self.smag_alpha = smag_alpha;
        self.nu_sm_min = nu_sm_min;
        self.eta_test = eta_test;
    }

    /// Index plat (iy*nx + ix), délègue à la crate.
    #[inline]
    pub fn idx(&self, ix: usize, iy: usize) -> usize {
        self.lbm.idx(ix, iy)
    }

    /// Réinitialise les champs depuis un snapshot (option --restart).
    /// Les tranches doivent être à la taille de la grille du solveur
    /// (interpolées au préalable si la grille source diffère). Reconstruit
    /// f/g depuis les équilibres et remet le compteur de pas à zéro.
    pub fn init_from_fields(&mut self, rho: &[f64], ux: &[f64], uy: &[f64], t: &[f64]) {
        let nx = self.lbm.nx;
        let ny = self.lbm.ny;
        let n = nx * ny;
        assert!(
            rho.len() == n && ux.len() == n && uy.len() == n && t.len() == n,
            "init_from_fields : champs de taille {n} attendue (rho={}, ux={}, uy={}, t={})",
            rho.len(), ux.len(), uy.len(), t.len()
        );
        for iy in 0..ny {
            for ix in 0..nx {
                let k = iy * nx + ix;
                let r = rho[k];
                let un = ux[k];
                let vn = uy[k];
                let tn = t[k];
                self.lbm.rho[k] = r;
                self.lbm.ux[k] = un;
                self.lbm.uy[k] = vn;
                self.lbm.temperature[k] = tn;
                for i in 0..9 {
                    self.lbm.f[k][i] = feq(i, r, un, vn);
                    self.lbm.g[k][i] = geq(i, tn, un, vn);
                }
            }
        }
        // No-slip strict sur les parois (l'interpolation peut y laisser un
        // petit résidu de vitesse).
        self.enforce_wall_velocity();
        self.time_step = 0;
    }

    /// Nombre de nœuds en x.
    #[inline]
    pub fn nx(&self) -> usize {
        self.lbm.nx
    }

    /// Nombre de nœuds en y.
    #[inline]
    pub fn ny(&self) -> usize {
        self.lbm.ny
    }

    /// Un pas de temps complet : collision LES + force de Boussinesq,
    /// streaming pull avec bounce-back intégré, conditions aux limites
    /// thermiques, mise à jour macroscopique.
    pub fn step(&mut self) {
        // Mode échelles mixtes dynamiques (§2.3.5 thèse) : ν_t/α_t calculés
        // par nœud AVANT la collision, puis réutilisés par le chemin de
        // collision existant via nu_t_pre/alpha_t_pre. En mode OFF (défaut)
        // ce calcul n'est jamais exécuté : chemin bitwise identique.
        if self.dynamic_smag {
            self.compute_dynamic_sgs();
        }
        self.collide(self.time_step);
        self.stream();
        self.apply_thermal_boundary_conditions();
        self.update_macroscopic();
        self.enforce_wall_velocity();
        self.time_step += 1;
    }

    /// Mise à jour macroscopique (ρ, u, T depuis f, g) avec correction de
    /// vitesse de Guo et al. 2002, éq. (16) : u = Σ e_i·f_i/ρ + F·Δt/2ρ
    /// (Δt = 1, F = (0, fy, 0) lue depuis fy_node).
    pub(crate) fn update_macroscopic(&mut self) {
        // Correction de vitesse de Guo et al. 2002, éq. (16) :
        //   u = Σ e_i·f_i/ρ + F·Δt/2ρ   (Δt = 1)
        // F = (0, fy, 0) est lue depuis fy_node (stockée par la DERNIÈRE
        // collision, évaluée aux champs du temps t) — la F de la correction
        // doit être la même que celle de la collision, d'où le stockage.
        // F_x = 0 ⇒ u_x inchangé ; u_y reçoit fy/(2ρ) par nœud. L'ajout est
        // strictement par nœud — aucune réduction inter-threads — donc le
        // chemin jobs > 1 reste bitwise déterministe et identique à jobs = 1.
        if self.jobs > 1 {
            let nx = self.lbm.nx;
            let ThermalD2Q9 { f, g, rho, ux, uy, temperature, .. } = &mut self.lbm;
            rho.par_chunks_mut(nx)
                .zip(ux.par_chunks_mut(nx))
                .zip(uy.par_chunks_mut(nx))
                .zip(temperature.par_chunks_mut(nx))
                .zip(f.par_chunks(nx))
                .zip(g.par_chunks(nx))
                .zip(self.fy_node.par_chunks(nx))
                .for_each(
                    |((((((r_row, u_row), v_row), t_row), f_row), g_row), fy_row)| {
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
                            // Correction de Guo et al. 2002, éq. (16) :
                            // u = (Σ e_i·f_i + F·Δt/2)/ρ, Δt = 1, F = (0, fy, 0)
                            // lue depuis fy_node (même F que la collision) —
                            // l'ajout est par nœud, déterminisme préservé.
                            let fy = fy_row[n];
                            if r > f64::EPSILON {
                                un /= r;
                                // Même ordre d'opérations que le chemin crate :
                                // vn/r puis ajout de fy/(2ρ) (après coup).
                                vn /= r;
                                vn += 0.5 * fy / r;
                            }
                            r_row[n] = r;
                            u_row[n] = un;
                            v_row[n] = vn;
                            t_row[n] = tn;
                        }
                    },
                );
            return;
        }
        // Chemin crate : `ThermalD2Q9::update_macroscopic` ne connaît pas la
        // force, on applique donc la correction de Guo et al. 2002, éq. (16),
        // APRÈS coup : u_y ← u_y + fy/(2ρ) par nœud, avec fy lu depuis
        // fy_node (stockée par la dernière collision — même F que la
        // collision, pas de recalcul). L'ajout est strictement par nœud (pas
        // de réduction inter-threads), donc le déterminisme jobs > 1 est
        // préservé.
        self.lbm.update_macroscopic();
        let rho = &self.lbm.rho;
        let fy = &self.fy_node;
        for (uy, (&r, &fyk)) in self.lbm.uy.iter_mut().zip(rho.iter().zip(fy.iter())) {
            if r > f64::EPSILON {
                *uy += 0.5 * fyk / r;
            }
        }
    }

    /// Configure le nombre de threads rayon (option --jobs N).
    /// N = 1 : pool mono-thread ; N > 1 : pool de N threads de calcul.
    /// Déterminisme garanti : les opérations par nœud sont indépendantes
    /// (aucune dépendance en écriture entre voisins), donc l'ordre
    /// d'exécution n'affecte pas les résultats — bitwise identique à --jobs 1.
    pub fn set_jobs(jobs: usize) {
        rayon::ThreadPoolBuilder::new()
            .num_threads(jobs.max(1))
            .build_global()
            .unwrap_or_else(|e| {
                // Le pool global ne peut être initialisé qu'une fois par
                // processus (cas des tests) : on ignore silencieusement,
                // le pool par défaut reste actif.
                let _ = e;
            });
    }

    /// Rampe anti-overshoot (viscosité + diffusivité) — expressions bitwise
    /// identiques à l'ancien code inline de `collide`, factorisées pour être
    /// réutilisées par le calcul dynamique des échelles mixtes (|S̃| issu de
    /// π_neq dépend du τ moléculaire effectif, qui doit être le même que
    /// celui de la collision du même pas).
    fn ramp_nu_alpha(&self, step: usize) -> (f64, f64) {
        // Rampe de viscosité (anti-overshoot) : ν effectif élevé au départ
        // (×ν_start_mult), décroissant exponentiellement vers ν moléculaire
        // cible sur les premiers pas.
        let nu_eff = if step < self.spin_up_steps {
            let x = step as f64 / self.spin_up_steps as f64; // 0 → 1
            let nu_start = self.nu * self.nu_start_mult;
            nu_start * (self.nu / nu_start).powf(x)
        } else {
            self.nu
        };
        // Viscosités thermiques avec la même rampe (α_t = ν_t/Pr_t suit).
        let alpha_eff = if step < self.spin_up_steps {
            let x = step as f64 / self.spin_up_steps as f64;
            let alpha_start = self.alpha * self.nu_start_mult;
            alpha_start * (self.alpha / alpha_start).powf(x)
        } else {
            self.alpha
        };
        (nu_eff, alpha_eff)
    }

    // -----------------------------------------------------------------------
    // Collision : TRT (two-relaxation-time) + Smagorinsky + Boussinesq
    // (forcing façon Guo, Shi & Zheng 2002, éq. (14)-(16), avec correction
    // de vitesse F·Δt/2ρ dans update_macroscopic). TRT relaxe séparément les
    // parties paires/impaires
    // de f par rapport à l'équilibre, avec Λ = (τ+ − 1/2)(τ− − 1/2) = 3/16
    // (valeur optimale pour bounce-back : Ginzburg et al. 2008). Beaucoup plus
    // robuste que BGK quand τ+ → 0.5, sans changer la physique visée.
    // -----------------------------------------------------------------------
    pub(crate) fn collide(&mut self, step: usize) {
        let nx = self.lbm.nx;
        let t_ref = self.lbm.t_ref;
        let cs = self.cs;
        let pr_t = self.pr_t;
        // Paramètre TRT optimal pour bounce-back (Λ = 3/16).
        let lambda_trt: f64 = 3.0 / 16.0;

        // Rampe de spin-up de la flottabilité : beta effectif croît linéairement
        // de 0 → beta_eff sur les premiers pas (anti-overshoot du transitoire ;
        // pratique standard pour Ra élevé — le steady state est identique, seul
        // le chemin transitoire est amorti).
        let beta_eff = if step < self.spin_up_steps {
            self.beta_eff * (step as f64 / self.spin_up_steps as f64)
        } else {
            self.beta_eff
        };

        // Rampe de viscosité/diffusivité (factorisée dans ramp_nu_alpha,
        // expressions bitwise identiques à l'ancien code inline).
        let (nu_eff, alpha_eff) = self.ramp_nu_alpha(step);
        let tau = 0.5 + 3.0 * nu_eff.max(0.002 / 3.0);

        // Contexte immuable partagé (Send + Sync) pour la variante parallèle.
        let ctx = CollideCtx {
            tau,
            beta_eff,
            t_ref,
            cs,
            pr_t,
            nu: nu_eff,
            alpha: alpha_eff,
            lambda_trt,
        };

        // ν_t/α_t précalculés par nœud (mode échelles mixtes dynamiques
        // uniquement ; None en mode Smagorinsky Cs fixe ⇒ comportement
        // historique strictement inchangé dans collide_node).
        let nup_rows: Option<Vec<&[f64]>> = self.nu_t_pre.as_ref().map(|v| v.chunks(nx).collect());
        let atp_rows: Option<Vec<&[f64]>> = self.alpha_t_pre.as_ref().map(|v| v.chunks(nx).collect());

        if self.jobs > 1 {
            // --- Variante parallèle (rayon, bandes de lignes iy) ---
            // Chaque nœud lit/écrit UNIQUEMENT ses propres données
            // (k = iy*nx + ix, aucune dépendance voisine), donc le
            // partitionnement est sans race et l'ordre d'exécution
            // n'affecte pas les résultats ⇒ déterministe quel que soit --jobs.
            //
            // Pattern rayon sûr (zéro unsafe) : les scalaires lus
            // (rho, ux, uy, T) sont copiés ligne par ligne AVANT la boucle
            // parallèle (la collision ne modifie pas rho/ux/uy/T — vérifié :
            // update_macroscopic est appelé après), puis chaque thread
            // possède ses lignes f/g/nu_t exclusives.
            let scalars: Vec<(f64, f64, f64, f64)> = (0..self.lbm.rho.len())
                .map(|k| (self.lbm.rho[k], self.lbm.ux[k], self.lbm.uy[k], self.lbm.temperature[k]))
                .collect();

            let f_rows: Vec<&mut [[f64; 9]]> = self.lbm.f.chunks_mut(nx).collect();
            let g_rows: Vec<&mut [[f64; 9]]> = self.lbm.g.chunks_mut(nx).collect();
            let nt_rows: Vec<&mut [f64]> = self.nu_t.chunks_mut(nx).collect();
            let fy_rows: Vec<&mut [f64]> = self.fy_node.chunks_mut(nx).collect();

            f_rows
                .into_par_iter()
                .zip(g_rows)
                .zip(nt_rows)
                .zip(fy_rows)
                .enumerate()
                .for_each(|(iy, (((f_row, g_row), nt_row), fy_row))| {
                    for (ix, f_node) in f_row.iter_mut().enumerate() {
                        let k = iy * nx + ix;
                        let (rho, ux, uy, temp) = scalars[k];
                        // ν_t/α_t précalculés du nœud (None en mode OFF).
                        let nup = nup_rows.as_ref().map(|rows| rows[iy][ix]);
                        let atp = atp_rows.as_ref().map(|rows| rows[iy][ix]);
                        collide_node(
                            f_node,
                            &mut g_row[ix],
                            &mut nt_row[ix],
                            &mut fy_row[ix],
                            rho,
                            ux,
                            uy,
                            temp,
                            nup,
                            atp,
                            &ctx,
                        );
                    }
                });
            return;
        }

        // --- Chemin mono-thread d'origine (strictement inchangé hors
        //     transmission des ν_t/α_t précalculés, None en mode OFF) ---
        for iy in 0..self.lbm.ny {
            for ix in 0..nx {
                let k = iy * nx + ix;
                let rho = self.lbm.rho[k];
                let ux = self.lbm.ux[k];
                let uy = self.lbm.uy[k];
                let temp = self.lbm.temperature[k];
                let nup = nup_rows.as_ref().map(|rows| rows[iy][ix]);
                let atp = atp_rows.as_ref().map(|rows| rows[iy][ix]);
                collide_node(
                    &mut self.lbm.f[k],
                    &mut self.lbm.g[k],
                    &mut self.nu_t[k],
                    &mut self.fy_node[k],
                    rho,
                    ux,
                    uy,
                    temp,
                    nup,
                    atp,
                    &ctx,
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Échelles mixtes dynamiques (§2.3.5 thèse, Tab. 2.3) — mode --dynamic-smag.
    //
    // Calcule ν_t(x,y) et α_t(x,y) par nœud AVANT la collision, via la
    // procédure dynamique de Lilly (identité de Germano + test-filter box,
    // régularisation Éq. 2.62, bornage), le volet « mixte » interpolant
    // Smagorinsky (exposant α) × similarité qsm/Φsm (exposant 1−α) ;
    // α = 0.5 ⇒ chaque facteur interpolé est une moyenne géométrique.
    //
    // NOTATION THÈSE (inversée vs convention usuelle — attention) :
    //   Δ̃ = filtre grille = Δx ;  Δ = filtre test = η·Δx (η = eta_test).
    //   ũ (thèse) = champ résolu (filtre grille) ; u (thèse) = champ test-filtré.
    //
    // Tout est séquentiel (aucune réduction flottante parallèle) ⇒ résultats
    // bitwise identiques quel que soit --jobs. Les lectures se font sur des
    // instantanés clonés AVANT la collision (f n'est pas encore modifié —
    // |S̃| est reconstruit depuis π_neq avec EXACTEMENT la formule de
    // collide_node), les écritures sur des Vec locaux transmis ensuite à
    // collide via nu_t_pre/alpha_t_pre.
    // -----------------------------------------------------------------------
    fn compute_dynamic_sgs(&mut self) {
        let nx = self.lbm.nx;
        let ny = self.lbm.ny;
        let n = nx * ny;
        let step = self.time_step;

        // τ moléculaire effectif — MÊME rampe/expression que `collide` du
        // même pas (|S̃| issu de π_neq en dépend, cf. collide_node).
        let (nu_eff, _alpha_eff) = self.ramp_nu_alpha(step);
        let tau = 0.5 + 3.0 * nu_eff.max(0.002 / 3.0);

        let alpha_exp = self.smag_alpha; // exposant α (0.5 = moyenne géométrique)
        let eta = self.eta_test.max(1.0);
        let width = (eta.round() as usize).max(1); // largeur du filtre box (nœuds)
        let dx: f64 = 1.0;      // Δ̃ = Δx (unités de réseau)
        let dtest = eta * dx;   // Δ = η·Δ̃ (filtre test)
        // Écart de température GLOBAL ΔT = θc − θf = 1 (PAS Δ̃·T — Tab. 2.3).
        let delta_t = DELTA_T;
        let nu_mol = self.nu;
        let alpha_mol = self.alpha;
        let nu_sm_min = self.nu_sm_min;

        // --- Instantanés immuables (la collision n'a pas encore modifié f) ---
        let rho = self.lbm.rho.clone();
        let ux = self.lbm.ux.clone();
        let uy = self.lbm.uy.clone();
        let temp = self.lbm.temperature.clone();
        let f = self.lbm.f.clone();

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
        let mut qsm_h = vec![0.0f64; n];   // (qsm)^{1/2} = (1/2)|ũ − ũ_test| (Éq. 2.74)
        let mut phi_h = vec![0.0f64; n];   // (Φsm)^{1/2} = (1/2)|ũ − ũ_test|·|T − T_test|
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
        self.cs_dyn_mean = cnu_sum / n as f64;
        self.cs_dyn_max = cnu_max;
        self.kappa_dyn_mean = ck_sum / n as f64;
        self.kappa_dyn_max = ck_max;
        self.nu_t_pre = Some(nu_t_pre);
        self.alpha_t_pre = Some(alpha_t_pre);
    }

    // -----------------------------------------------------------------------
    // Streaming pull standard (pas périodique) avec bounce-back intégré :
    // f[i](x) ← f[i](x − c_i) si la source est dans le domaine, sinon
    // f[i](x) ← f*(x, opp(i)) post-collision (réflexion de la distribution
    // sortante). Chaque valeur sortante est réutilisée exactement une fois
    // (bijection entre {k : k−c_i hors domaine} et {j : j+c_opp(i) hors domaine})
    // ⇒ masse exactement conservée, no-slip au milieu du lien.
    // -----------------------------------------------------------------------
    pub(crate) fn stream(&mut self) {
        let nx = self.lbm.nx;
        let ny = self.lbm.ny;
        let f_old = self.lbm.f.clone();
        let g_old = self.lbm.g.clone();

        if self.jobs > 1 {
            // --- Variante parallèle (rayon, bandes de lignes iy) ---
            // Chaque nœud (ix, iy) lit UNIQUEMENT f_old/g_old (instantanés
            // clonés avant mutation, lecture seule entre threads) et écrit
            // sa propre ligne iy : régions d'écriture disjointes ⇒ aucune
            // race, opérations flottantes par nœud identiques ⇒ déterministe.
            //
            // Pattern rayon sûr (zéro unsafe) : les chunks mutables de f et g
            // sont collectés séquentiellement (emprunts disjoints, autorisés),
            // puis zippés ligne à ligne en parallèle — chaque thread possède
            // sa ligne.
            let f_rows: Vec<&mut [[f64; 9]]> = self.lbm.f.chunks_mut(nx).collect();
            let g_rows: Vec<&mut [[f64; 9]]> = self.lbm.g.chunks_mut(nx).collect();

            f_rows
                .into_par_iter()
                .zip(g_rows)
                .enumerate()
                .for_each(|(iy, (f_row, g_row))| {
                    for (ix, fnode) in f_row.iter_mut().enumerate() {
                        let k = iy * nx + ix;
                        for i in 0..9 {
                            let cx = D2Q9_VELOCITIES[i][0] as i64;
                            let cy = D2Q9_VELOCITIES[i][1] as i64;
                            let sx = ix as i64 - cx;
                            let sy = iy as i64 - cy;
                            if sx >= 0 && sx < nx as i64 && sy >= 0 && sy < ny as i64 {
                                let src = (sy as usize) * nx + sx as usize;
                                fnode[i] = f_old[src][i];
                                g_row[ix][i] = g_old[src][i];
                            } else {
                                // Bounce-back : la source est hors domaine
                                // (paroi), on réfléchit la distribution
                                // sortante opposée.
                                fnode[i] = f_old[k][D2Q9_OPPOSITES[i]];
                                // g : valeur d'avant streaming provisoire,
                                // remplacée par les BC thermiques ensuite.
                                g_row[ix][i] = g_old[k][i];
                            }
                        }
                    }
                });
            return;
        }

        // --- Chemin mono-thread d'origine (strictement inchangé) ---
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
                        self.lbm.f[k][i] = f_old[src][i];
                        self.lbm.g[k][i] = g_old[src][i];
                    } else {
                        // Bounce-back : la source est hors domaine (paroi),
                        // on réfléchit la distribution sortante opposée.
                        self.lbm.f[k][i] = f_old[k][D2Q9_OPPOSITES[i]];
                        // g : valeur d'avant streaming provisoire, remplacée par
                        // les BC thermiques (isotherme / zero-gradient) ensuite.
                        self.lbm.g[k][i] = g_old[k][i];
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Full-way bounce-back autonome (utilisé par les tests de conservation) :
    // sur chaque nœud de paroi, on échange f_i ↔ f_opp(i) — bijection exacte.
    // -----------------------------------------------------------------------
    #[allow(dead_code)]
    pub(crate) fn apply_bounce_back(&mut self) {
        let nx = self.lbm.nx;
        let ny = self.lbm.ny;
        for iy in 0..ny {
            for &ix in &[0usize, nx - 1] {
                self.swap_opposite(iy * nx + ix);
            }
        }
        for &iy in &[0usize, ny - 1] {
            for ix in 0..nx {
                self.swap_opposite(iy * nx + ix);
            }
        }
    }

    /// Échange f_i ↔ f_opp(i) sur un nœud (full-way bounce-back, bijection exacte).
    #[inline]
    #[allow(dead_code)]
    fn swap_opposite(&mut self, k: usize) {
        let fnode = &mut self.lbm.f[k];
        for i in 1..5 {
            let j = D2Q9_OPPOSITES[i];
            let tmp = fnode[i];
            fnode[i] = fnode[j];
            fnode[j] = tmp;
        }
    }

    // -----------------------------------------------------------------------
    // Conditions aux limites thermiques (après streaming)
    // -----------------------------------------------------------------------
    pub(crate) fn apply_thermal_boundary_conditions(&mut self) {
        let nx = self.lbm.nx;
        let ny = self.lbm.ny;

        // --- Parois verticales (chaude à gauche, froide à droite) ---
        // g paroi = geq(T_wall, u=0) (isotherme).
        for iy in 0..ny {
            {
                let k = iy * nx;
                for i in 0..9 {
                    self.lbm.g[k][i] = geq(i, T_HOT, 0.0, 0.0);
                }
            }
            {
                let k = iy * nx + (nx - 1);
                for i in 0..9 {
                    self.lbm.g[k][i] = geq(i, T_COLD, 0.0, 0.0);
                }
            }
        }

        // --- Parois horizontales (haut/bas) : adiabatiques ---
        // g paroi = copie du nœud intérieur adjacent (zero-gradient ⇒ ∂T/∂y = 0).
        for ix in 0..nx {
            {
                let k = ix;
                let k_in = nx + ix;
                let row_bottom = self.lbm.g[k_in];
                self.lbm.g[k].copy_from_slice(&row_bottom);
            }
            {
                let k = (ny - 1) * nx + ix;
                let k_in = (ny - 2) * nx + ix;
                let row_top = self.lbm.g[k_in];
                self.lbm.g[k].copy_from_slice(&row_top);
            }
        }
    }

    /// Force u = 0 aux nœuds de paroi (après bounce-back, les vitesses
    /// macroscopiques des parois doivent être nulles — no-slip).
    fn enforce_wall_velocity(&mut self) {
        let nx = self.lbm.nx;
        let ny = self.lbm.ny;
        for iy in 0..ny {
            for &ix in &[0usize, nx - 1] {
                let k = iy * nx + ix;
                self.lbm.ux[k] = 0.0;
                self.lbm.uy[k] = 0.0;
            }
        }
        for &iy in &[0usize, ny - 1] {
            for ix in 0..nx {
                let k = iy * nx + ix;
                self.lbm.ux[k] = 0.0;
                self.lbm.uy[k] = 0.0;
            }
        }
    }

    /// Masse totale Σρ (diagnostic de conservation).
    pub fn total_mass(&self) -> f64 {
        self.lbm.rho.iter().sum()
    }

    /// Somme directe des distributions f (indépendante de rho, pour les tests).
    #[allow(dead_code)]
    pub fn total_f_mass(&self) -> f64 {
        self.lbm.f.iter().flat_map(|row| row.iter()).sum()
    }
}

// ---------------------------------------------------------------------------
// Parallélisation multi-thread (option --jobs N)
// Principe : collision et streaming pull sont des opérations par nœud
// (k = iy*nx + ix) SANS dépendance en écriture entre voisins. Le
// partitionnement par bandes de lignes iy garantit des régions d'écriture
// disjointes ⇒ aucune race, aucun verrou, et les opérations flottantes par
// nœud sont identiques quel que soit le nombre de threads ⇒ résultats
// bitwise-identiques à --jobs 1 (déterminisme).
// ---------------------------------------------------------------------------

/// Contexte immuable de collision (partagé entre threads, Send + Sync).
struct CollideCtx {
    tau: f64,
    beta_eff: f64,
    t_ref: f64,
    cs: f64,
    pr_t: f64,
    nu: f64,
    alpha: f64,
    lambda_trt: f64,
}

/// Collision TRT + Smagorinsky + Boussinesq sur UN nœud (équation exacte
/// de l'ancienne boucle séquentielle, bitwise identique).
///
/// `nu_t_pre`/`alpha_t_pre` (mode échelles mixtes dynamiques uniquement) :
/// si Some, ν_t/α_t sont pris tels quels (précalculés par
/// `compute_dynamic_sgs`, §2.3.5 thèse) ; si None (mode OFF, défaut),
/// ν_t = (Cs·Δx)²·|S| et α_t = ν_t/Pr_t — chemin historique bitwise identique.
#[inline]
#[allow(clippy::too_many_arguments)]
fn collide_node(
    f_node: &mut [f64; 9],
    g_node: &mut [f64; 9],
    nu_t_out: &mut f64,
    fy_out: &mut f64,
    rho: f64,
    ux: f64,
    uy: f64,
    temp: f64,
    nu_t_pre: Option<f64>,
    alpha_t_pre: Option<f64>,
    ctx: &CollideCtx,
) {
    let CollideCtx {
        tau,
        beta_eff,
        t_ref,
        cs,
        pr_t,
        nu,
        alpha,
        lambda_trt,
    } = *ctx;

    // Force de Boussinesq : fy = rho · beta_eff · (T − t_ref), dirigée vers +y
    // quand T > t_ref (chaud monte, gravité orientée −y). Convention de la
    // crate : "positive upward" (gravity = +1), beta = beta_eff = g·β·ΔT.
    // Stockée par nœud : la correction de vitesse de update_macroscopic
    // (Guo éq. (16)) doit réutiliser EXACTEMENT cette F, évaluée aux champs
    // du temps t — pas de recalcul après mise à jour.
    let fy = oxiphysics_lbm::thermal_lbm::boussinesq_force(rho, beta_eff, 1.0, temp, t_ref);
    *fy_out = fy;

    // Équilibres des 9 directions (une seule fois par nœud).
    let mut f_eq = [0.0f64; 9];
    let mut g_eq = [0.0f64; 9];
    for i in 0..9 {
        f_eq[i] = feq(i, rho, ux, uy);
        g_eq[i] = geq(i, temp, ux, uy);
    }

    // Moments non-équilibrés π_neq (avant relaxation) pour |S|.
    let mut pi_xx = 0.0;
    let mut pi_xy = 0.0;
    let mut pi_yy = 0.0;
    for i in 0..9 {
        let cx = D2Q9_VELOCITIES[i][0] as f64;
        let cy = D2Q9_VELOCITIES[i][1] as f64;
        let fneq = f_node[i] - f_eq[i];
        pi_xx += fneq * cx * cx;
        pi_xy += fneq * cx * cy;
        pi_yy += fneq * cy * cy;
    }
    // |S| = sqrt(π_neq:π_neq) / (2·cs²·τ·ρ) ; π:π = Πxx² + 2Πxy² + Πyy².
    let pi2 = pi_xx * pi_xx + 2.0 * pi_xy * pi_xy + pi_yy * pi_yy;
    let s_mag = pi2.sqrt() / (2.0 * CS2 * tau * rho.max(1e-12));

    // Viscosité turbulente : en mode dynamique, ν_t est précalculé par nœud
    // (échelles mixtes §2.3.5) et transmis tel quel ; sinon ν_t = (Cs·Δx)²·|S|
    // (Smagorinsky classique, Δx = 1) — chemin OFF bitwise identique.
    let nu_t = match nu_t_pre {
        Some(v) => v,
        None => oxiphysics_lbm::turbulence::compute_turbulent_viscosity(s_mag, cs),
    };
    *nu_t_out = nu_t;

    // Temps de relaxation effectifs TRT (τ+ pairs, τ− impairs, Λ = 3/16).
    let tau_plus = 0.5 + 3.0 * (nu + nu_t);
    let tau_minus = 0.5 + lambda_trt / (tau_plus - 0.5);
    // Diffusivité turbulente du scalaire g (Éq. 2.80) : le coefficient de
    // diffusion de l'énergie filtrée est Ra^(-1/2)·(1 + κsm/κ) ⇒ dans le
    // code, κ_total = α·(1 + κsm/κ) = α + κsm, soit τ_t⁺ = 0.5 + 3·κ_total
    // avec α_t = κsm = C^κ_dyn·κsm_mix(grille) précalculé (mode dynamique) ;
    // sinon ν_t/Pr_t (chemin historique). Le patch τ_t⁻ = τ_t⁺ ci-dessous
    // (BGK scalaire) reste EXACTEMENT tel quel dans les deux modes.
    // (Éq. 2.79 symétrique côté vitesse : ν_total = ν + νsm via ν_t, déjà le
    // cas ; flottabilité Pr·θ̃·δiz avec Ra basé sur la HAUTEUR — inchangée,
    // beta_eff = g·β·ΔT en unités réseau est cohérent.)
    let alpha_t = match alpha_t_pre {
        Some(v) => v, // κsm (Éq. 2.80) : κ_total = alpha + κsm
        None => nu_t / pr_t,
    };
    let tau_t_plus = 0.5 + 3.0 * (alpha + alpha_t);
    // Scalaire thermique : paroi à nœud-équilibre (pas de bounce-back) → Λ=3/16
    // injustifié → TRT réduit à BGK pour g (τ_t⁻ = τ_t⁺). Côté f (momentum),
    // bounce-back réel : Λ=3/16 conservé (Guo-forcing Poiseuille validé).
    let tau_t_minus = tau_t_plus;
    // Coefficients TRT : a = (ω+ + ω−)/2, b = (ω+ − ω−)/2.
    let a_f = 0.5 * (1.0 / tau_plus + 1.0 / tau_minus);
    let b_f = 0.5 * (1.0 / tau_plus - 1.0 / tau_minus);
    let a_g = 0.5 * (1.0 / tau_t_plus + 1.0 / tau_t_minus);
    let b_g = 0.5 * (1.0 / tau_t_plus - 1.0 / tau_t_minus);

    // Déviations pré-collision (pour que Σ f' = Σ f, masse conservée).
    let mut d_f = [0.0f64; 9];
    let mut d_g = [0.0f64; 9];
    for i in 0..9 {
        d_f[i] = f_node[i] - f_eq[i];
        d_g[i] = g_node[i] - g_eq[i];
    }

    // Collision f TRT avec force (Guo, Shi & Zheng 2002, éq. (14)-(16)) :
    //   S_i = w_i·[(e_i−u)/c_s² + (e_i·u)·e_i/c_s⁴]·F, F = (0, fy, 0).
    // En TRT, l'exactitude du lattice effect s'applique PAR MODE de relaxation :
    //   - partie impaire (porte le momentum, relaxe à ω⁻ = 1/τ⁻) : facteur (1 − 1/(2τ⁻))
    //   - partie paire (porte le stress, relaxe à ω⁺ = 1/τ⁺) : facteur (1 − 1/(2τ⁺))
    // La correction de vitesse u = Σef/ρ + F/2ρ (update_macroscopic) impose
    // j_neq = −F/2 ; le bilan par pas Δu = (F/ρ)·(1/(2τ⁻) + guo) exige
    // guo = 1 − 1/(2τ⁻) sur la partie impaire pour l'exactitude d'Euler de Guo-BGK.
    // Sur D2Q9 : e_i·u = cx·ux + cy·uy, c_s² = 1/3.
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
    // Collision g TRT.
    for i in 0..9 {
        let opp = D2Q9_OPPOSITES[i];
        g_node[i] += -a_g * d_g[i] - b_g * d_g[opp];
    }
}

// ---------------------------------------------------------------------------
// Helpers des échelles mixtes dynamiques (§2.3.5 thèse)
// ---------------------------------------------------------------------------

/// Facteur interpolé mixte (Tab. 2.3) : a^α · b^(1−α). α = 0.5 ⇒ moyenne
/// géométrique (les deux volets Smagorinsky × similarité). Version test :
/// identique aux `powf` inline de `compute_dynamic_sgs`.
#[cfg(test)]
pub(crate) fn mixed_factor(a: f64, b: f64, alpha: f64) -> f64 {
    a.powf(alpha) * b.powf(1.0 - alpha)
}

/// Filtre box de largeur `width` nœuds (width = η·Δ̃/Δx, typ. 2 pour η = 2),
/// centré au plus près : width pair ⇒ fenêtre {ix, ix+1} (demi-nœud de
/// décalage, incontournable sur grille colocalisée), width impair ⇒ fenêtre
/// symétrique. Parois : clamp unilatéral (la fenêtre rétrécit vers
/// l'intérieur du domaine, aucune valeur de bord dupliquée).
fn box_filter(field: &[f64], nx: usize, ny: usize, width: usize) -> Vec<f64> {
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
#[inline]
fn ddx(field: &[f64], nx: usize, ny: usize, ix: usize, iy: usize) -> f64 {
    let _ = ny;
    let xm = if ix == 0 { 0 } else { ix - 1 };
    let xp = if ix + 1 >= nx { nx - 1 } else { ix + 1 };
    0.5 * (field[iy * nx + xp] - field[iy * nx + xm])
}

/// Dérivée centrale ∂/∂y d'un champ (pas Δy = 1) ; parois : clamp unilatéral.
#[inline]
fn ddy(field: &[f64], nx: usize, ny: usize, ix: usize, iy: usize) -> f64 {
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
#[inline]
pub(crate) fn lilly_coefficient(num: f64, den: f64) -> f64 {
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
