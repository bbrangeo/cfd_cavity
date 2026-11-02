//! Backend GPU wgpu (Étapes A-C, portes G1-G3) — pas de temps uniforme en f32.
//!
//! Architecture 2 dispatches ordonnés (un seul compute pass, deux
//! `dispatch_workgroups` — chaque dispatch est un usage-scope wgpu, les
//! barrières storage sont insérées entre dispatches, aucune race) :
//!
//!   1. `collide`      : collision TRT (Λ = 3/16) + force de Guo (odd/even) +
//!                       Smagorinsky par nœud (ν_t, τ+, τ_t+ dépendent de ν_t)
//!                       → écrit f_col/g_col + nu_t.
//!   2. `stream_macro` : streaming pull + bounce-back (lit f_col/g_col FIGÉS
//!                       du dispatch 1 — c'est la même sémantique que le CPU,
//!                       qui clone f/g avant le streaming) → écrit f_out/g_out,
//!                       PUIS CL thermiques (verticales isothermes puis
//!                       horizontales zero-gradient, les coins écrasées par la
//!                       copie horizontale — ordre CPU ; la valeur du nœud
//!                       intérieur est recalculée depuis g_col figé, aucune
//!                       course) PUIS macroscopique (ρ, u, T) avec correction
//!                       de vitesse Guo éq. (16) — fy recalculé localement
//!                       depuis le stock du temps t (même F que la collision)
//!                       — no-slip périmètre → écrit stock.
//!
//! Étape C (porte G3) : les boucles 9 directions sont déroulées (indices
//! littéraux — plus de tableaux privés indexés dynamiquement qui forçaient la
//! scalarisation/spills sur Metal) et le pas s'exécute en 2 dispatches.
//! Le layout mémoire est SoA [direction][nœud] (f_in[i·n + k]) : les accès
//! des 9 directions sont coalescés entre threads adjacents (l'AoS [nœud][9]
//! rendait le streaming non coalescé, ~3,5× d'amplification de trafic).
//! De plus, hors spin-up (params constants bit-à-bit), les pas s'enchaînent
//! dans un SEUL command encoder (batch de 64 pas, flushé avant tout readback)
//! pour amortir l'overhead CPU de soumission.
//!
//! Étape D (porte G4) : portage du dynamic Smagorinsky (échelles mixtes
//! §2.3.5 thèse, `compute_dynamic_sgs` du CPU) en 3 dispatches additionnels
//! exécutés dans le MÊME compute pass, avant la collision :
//!
//!   0a. `dyn1` : qsm^{1/2}, Φsm^{1/2} par nœud (Éq. 2.74) → dyn_s1.
//!   0b. `dyn2` : S̃/|S̃| (π_neq), S̄/|S̄| (box stock), gradients, formes
//!       mixtes interpolées (Éq. 2.73), tenseurs Germano M/L, contractions
//!       de Lilly → dyn_s2/dyn_s3.
//!   0c. `dyn3` : régularisation Éq. 2.62 (box num/den), coefficient de
//!       Lilly (clamp [0, 0.04] = C_DYN_MAX), ν_t/α_t bornés (plancher TRT
//!       + nu_sm_min) → dyn_out[k] = (ν_t, α_t, C^ν, C^κ).
//!   1.  `collide_dyn` : réplique exacte de `collide` qui CONSOMME ν_t/α_t
//!       de dyn_out (aucun readback — ν_t reste sur GPU sauf diagnostics).
//!
//! Les barrières wgpu entre dispatches d'un même pass garantissent dyn1→dyn2
//!→dyn3→collide_dyn. Le batching d'encoders est préservé : les paramètres
//! dynamiques (α, η, ν_mol, α_mol, nu_sm_min, ΔT, marge plancher, largeur box)
//! sont constants au cours du run. Les diagnostics C^ν_dyn/C^κ_dyn (dyn_out.zw)
//! sont réduits CÔTÉ CPU au readback (pas d'atomiques flottantes, réduction
//! séquentielle déterministe).
//!
//! Précision (parade FluidX3D, plan §2) : les populations f sont stockées
//! DÉCALÉES — f_i − w_i·ρ₀ (ρ₀ = 1) — pour récupérer la précision relative
//! f32 sur le non-équilibre ; reconstruction f_i = w_i + décalage dans le
//! shader. g est stocké tel quel (T̄ = 0 ⇒ décalage g_i − w_i^t·T̄ = g_i).
//!
//! Code CPU inchangé : ce module est une extension additive — le solveur de
//! référence (`solver.rs`) reste la vérité terrain. Le macroscopique est
//! réinjecté dans les champs du solveur CPU par `sync_into` (sans toucher au
//! pas de temps CPU), ce qui permet de réutiliser `post.rs` inchangé.

use bytemuck::Pod;
use bytemuck::Zeroable;
use wgpu::util::DeviceExt;

use crate::uniform::solver::CavitySolver;
use crate::uniform::solver::C_DYN_MAX;
use crate::uniform::solver::TAU_TRT_FLOOR;
use crate::uniform::solver::{T_COLD, T_HOT};
use crate::uniform::solver::DELTA_T;

/// Paramètres uniformes du pas de temps (passés au shader via uniform buffer).
/// Convertis en f32 à l'interface CPU (WGSL n'a pas de f64 portable).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq)]
struct GpuParams {
    /// τ moléculaire pour |S| : 0.5 + 3·max(ν_eff, 0.002/3) (rampe incluse).
    tau_mol: f32,
    /// ν effectif (rampe exponentielle, NON planché) — entre dans τ+.
    nu_eff: f32,
    /// α effectif (rampe exponentielle) — entre dans τ_t+.
    alpha_eff: f32,
    /// Force de flottabilité effective beta_eff (rampe spin-up linéaire).
    beta_eff: f32,
    /// Température de référence Boussinesq t_ref (= 0 ici).
    t_ref: f32,
    /// c_s² = 1/3 (littéral réseau).
    cs2: f32,
    /// Λ TRT = 3/16.
    lambda_trt: f32,
    /// Constante de Smagorinsky Cs (Δx = 1 ⇒ longueur de mélange = Cs).
    cs_smag: f32,
    /// Prandtl turbulent Pr_t (α_t = ν_t/Pr_t).
    pr_t: f32,
    /// Température paroi chaude (+0.5).
    t_hot: f32,
    /// Température paroi froide (−0.5).
    t_cold: f32,
    /// Nœuds en x (indices plats et CL).
    nx: u32,
    /// Nœuds en y.
    ny: u32,
    /// Padding (taille multiple de 16 octets).
    _pad: u32,
}

/// Paramètres du modèle dynamique (Étape D, porte G4) — constants au cours
/// du run (pas de rampe : ν_mol/α_mol sont les valeurs MOLÉCULAIRES cibles,
/// même sémantique que `compute_dynamic_sgs` qui lit self.nu/self.alpha).
/// Buffers séparés du GpuParams (qui varie pendant le spin-up) pour ne pas
/// casser le batching d'encoders (bit-à-bit constant).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq)]
struct GpuDynParams {
    /// Exposant α de l'interpolation mixte (0.5 = moyenne géométrique).
    alpha_exp: f32,
    /// Ratio filtre test η = Δ/Δ̃ (largeur box = round(max(η,1))).
    eta_test: f32,
    /// dtest = η·Δx (Δx = 1).
    dtest: f32,
    /// ΔT global = T_hot − T_cold = 1 (PAS Δ̃·T — Tab. 2.3).
    delta_t: f32,
    /// ν moléculaire cible (nu_sm_min·ν).
    nu_mol: f32,
    /// α moléculaire cible (nu_sm_min·α).
    alpha_mol: f32,
    /// Borne inférieure νsm ≥ nu_sm_min·ν.
    nu_sm_min: f32,
    /// Marge plancher TRT (TAU_TRT_FLOOR − 0.5)/3.
    tau_floor_margin: f32,
    /// Largeur du filtre box en nœuds (round(max(η,1))), ≥ 1.
    box_width: u32,
    /// Borne C_DYN_MAX = 0.04 (clamp du coefficient de Lilly).
    c_dyn_max: f32,
    /// Padding (taille 64 octets, alignement uniform WGSL).
    _pad: [u32; 6],
}

/// Solveur GPU : enveloppe wgpu du pas de temps uniforme, buffers f32.
pub struct GpuSolver {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipe_collide: wgpu::ComputePipeline,
    pipe_stream_macro: wgpu::ComputePipeline,
    /// Étape D (G4) : kernels du dynamic Smagorinsky (None = chemin OFF).
    pipe_dyn1: Option<wgpu::ComputePipeline>,
    pipe_dyn2: Option<wgpu::ComputePipeline>,
    pipe_dyn3: Option<wgpu::ComputePipeline>,
    pipe_dyn4: Option<wgpu::ComputePipeline>,
    pipe_collide_dyn: Option<wgpu::ComputePipeline>,
    params_buf: wgpu::Buffer,
    /// Uniform du modèle dynamique (constants au cours du run).
    /// Uniforms du mode dynamique (DynParams, binding 10) — constants au
    /// cours du run, lus uniquement via les bind groups GPU.
    #[allow(dead_code)] // lu par les bind groups (GPU), pas par le code Rust
    dyn_params_buf: Option<wgpu::Buffer>,

    /// Populations f DÉCALÉES f_i − w_i·ρ₀ (f32), layout SoA [direction][nœud]
    /// (f[i·n + k]) — buffer A.
    /// Populations f DÉCALÉES f_i − w_i·ρ₀ (f32), layout SoA [direction][nœud]
    /// (f[i·n + k]) — buffer A. Les champs struct sont lus via les bind
    /// groups pré-créés (pas d'accès direct Rust hors from_solver/tests).
    #[allow(dead_code)] // lu par les bind groups (GPU), pas par le code Rust
    f_a: wgpu::Buffer,
    /// Populations g (f32), layout SoA [direction][nœud] — buffer A.
    #[allow(dead_code)]
    g_a: wgpu::Buffer,
    /// Idem buffer B, layout SoA [direction][nœud] (ping-pong : le pas pair
    /// lit A/écrit B, inverse ensuite).
    #[allow(dead_code)]
    f_b: wgpu::Buffer,
    #[allow(dead_code)]
    g_b: wgpu::Buffer,
    /// Scratch post-collision (réécrit à chaque pas par le dispatch collide).
    #[allow(dead_code)]
    f_col: wgpu::Buffer,
    #[allow(dead_code)]
    g_col: wgpu::Buffer,
    /// Viscosité turbulente par nœud (diagnostic + TimeAverager).
    nu_t_buf: wgpu::Buffer,
    /// Étape D (G4) : buffers de travail du dynamic Smagorinsky.
    /// dyn_s1 : (qsm^½, Φsm^½) par nœud (2 f32). dyn_s2 : formes mixtes
    /// interpolées (ν_mix_t, ν_mix_g, κ_mix_t, κ_mix_g — 4 f32).
    /// dyn_s3 : quantités de Lilly (num_v, den_v, num_k, den_k — 4 f32).
    /// dyn_out : (ν_t, α_t, C^ν_dyn, C^κ_dyn) par nœud (4 f32).
    #[allow(dead_code)]
    dyn_s1: Option<wgpu::Buffer>,
    #[allow(dead_code)]
    dyn_s2: Option<wgpu::Buffer>,
    #[allow(dead_code)]
    dyn_s3: Option<wgpu::Buffer>,
    dyn_out: Option<wgpu::Buffer>,
    /// Champs macroscopiques stock (ρ, ux, uy, T) — temps t+1 après dispatch 2.
    stock_buf: wgpu::Buffer,
    /// Staging MAP_READ (stock 4n + nu_t n f32) pour les lectures CPU.
    read_buf: wgpu::Buffer,
    /// Staging MAP_READ des diagnostics dynamiques (dyn_out 4n f32).
    dyn_read_buf: Option<wgpu::Buffer>,

    /// Bind groups pré-créés (buffers fixes) — variantes ping-pong A/B.
    bg_collide_a: wgpu::BindGroup,
    bg_collide_b: wgpu::BindGroup,
    bg_stream_macro_a: wgpu::BindGroup,
    bg_stream_macro_b: wgpu::BindGroup,
    /// Étape D (G4) : bind groups des kernels dynamiques + collision
    /// consommant ν_t/α_t de dyn_out (variantes ping-pong A/B).
    bg_dyn1: Option<wgpu::BindGroup>,
    bg_dyn2_a: Option<wgpu::BindGroup>,
    bg_dyn2_b: Option<wgpu::BindGroup>,
    bg_dyn3: Option<wgpu::BindGroup>,
    bg_dyn4: Option<wgpu::BindGroup>,
    bg_collide_dyn_a: Option<wgpu::BindGroup>,
    bg_collide_dyn_b: Option<wgpu::BindGroup>,

    nx: usize,
    ny: usize,
    /// Pas de temps courant (rampe spin-up, même sémantique que le CPU).
    time_step: usize,
    /// true = le pas suivant lit A et écrit B (ping-pong).
    even: bool,
    /// Encoder batché : hors spin-up (params constants bit-à-bit), les pas
    /// s'enchaînent dans un SEUL command encoder (Étape C, amortit l'overhead
    /// CPU de soumission) — flushé avant tout readback et tous BATCH_STEPS.
    batch_enc: Option<wgpu::CommandEncoder>,
    /// Nombre de pas en attente dans l'encoder batché.
    batch_len: usize,
    /// Params du dernier pas du batch (re-vérifiés à chaque pas : le batching
    /// n'est valable que si params reste identique bit-à-bit).
    batch_params: GpuParams,
    /// true = le chemin dynamic Smagorinsky est actif (kernels dyn1/dyn2/dyn3
    /// + collide_dyn au lieu de collide).
    dynamic: bool,
    /// Info adaptateur (affichée au démarrage).
    adapter_info: String,
}

/// Taille du batch de pas (encoder unique) : amortit l'overhead CPU de
/// création/soumission d'encoder — flushé avant tout readback.
const BATCH_STEPS: usize = 64;

// Bindings des kernels dynamiques (Étape D) : le module WGSL déclare les
// bindings 10-15 (uniform DynParams + 4 buffers de travail) en plus des
// bindings 0-9 du module existant. Les deux déclarations vivent dans le
// même source WGSL : chaque entry point ne lie que ses bindings.
const WGSL: &str = r#"
// Paramètres uniformes (uniform buffer, f32).
struct Params {
    tau_mol: f32,     // 0.5 + 3·ν_eff — pour |S| (Smagorinsky)
    nu_eff: f32,      // ν rampe (non planché) — τ+ = 0.5 + 3(ν_eff + ν_t)
    alpha_eff: f32,   // α rampe — τ_t+ = 0.5 + 3(α_eff + α_t)
    beta_eff: f32,    // flottabilité effective (rampe spin-up incluse)
    t_ref: f32,       // température de référence Boussinesq (= 0)
    cs2: f32,         // c_s² = 1/3
    lambda_trt: f32,  // Λ = 3/16
    cs_smag: f32,     // constante de Smagorinsky (Δx = 1)
    pr_t: f32,        // Prandtl turbulent
    t_hot: f32,       // paroi chaude (+0.5)
    t_cold: f32,      // paroi froide (−0.5)
    nx: u32,
    ny: u32,
    pad: u32,
};

// Layout mémoire SoA [direction][nœud] (Étape C, porte G3) : f[i·n + k] avec
// n = nx·ny. Les 9 accès d'un kernel sont coalescés entre threads adjacents
// (l'AoS [nœud][9] rendait le streaming non coalescé, ~3,5× d'amplification
// de trafic — plafond ~175 MLUPS sur 256×1024). stock reste [nœud][4] (les 4
// composantes d'un nœud sont lues ensemble).

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> f_in: array<f32>;
@group(0) @binding(2) var<storage, read> g_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> f_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> g_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> stock: array<f32>;
@group(0) @binding(6) var<storage, read_write> f_col: array<f32>;
@group(0) @binding(7) var<storage, read_write> g_col: array<f32>;
@group(0) @binding(9) var<storage, read_write> nu_t: array<f32>;

// D2Q9 : ordre rest, E, N, W, S, NE, NW, SW, SE (crate oxiphysics-lbm).
// Étape C (porte G3) : boucles 9 directions DÉROULÉES — plus aucun tableau
// privé indexé dynamiquement (CX/CY/W/OPP supprimés). L'indexation dynamique
// de tableaux privés force la scalarisation en mémoire + spills registres sur
// Metal. Tous les indices sont des littéraux ; les poids et vecteurs sont des
// constantes (même ordre d'accumulation que les boucles).
const W0: f32 = 4.0 / 9.0;
const W1: f32 = 1.0 / 9.0; // E, N, W, S
const W5: f32 = 1.0 / 36.0; // NE, NW, SW, SE
const CS2: f32 = 1.0 / 3.0;

// Nombre total de nœuds (stride SoA des directions).
fn n_tot() -> u32 {
    return P.nx * P.ny;
}

// Équilibre hydrodynamique en VARIABLE DÉCALÉE (P1a) : feq_i − w_i =
// w_i·[(ρ−1) + ρ·A], A = eu/cs² + eu²/(2cs⁴) − u²/(2cs²). Identité exacte
// (feq_i = w_i·ρ·[1+A] = w_i + w_i·(ρ−1) + w_i·ρ·A, Σw_i = 1) évaluée
// UNIQUEMENT sur les petits nombres : ρ−1 = Σ f̃_i ~ 1e-6 et ρ·A ~ 1e-4,
// jamais le ~0.44 de w_i·ρ. rho_m1 = Σ f̃_i (somme des décalages, sans le 1),
// rho = 1 + ρ−1 (pleine magnitude, uniquement en facteur de ρ·A).
fn feq_off_c(w: f32, cx: f32, cy: f32, rho_m1: f32, rho: f32, ux: f32, uy: f32) -> f32 {
    let eu = cx * ux + cy * uy;
    let u2 = ux * ux + uy * uy;
    let a_term = eu / CS2 + eu * eu / (2.0 * CS2 * CS2) - u2 / (2.0 * CS2);
    return w * (rho_m1 + rho * a_term);
}

// Équilibre thermique geq (crate : w·T·[1 + eu/cs²]).
fn geq_c(w: f32, cx: f32, cy: f32, t: f32, ux: f32, uy: f32) -> f32 {
    let eu = cx * ux + cy * uy;
    return w * t * (1.0 + eu / CS2);
}

// Équilibre thermique de paroi isotherme (u = 0) : g_i = w_i·T — les poids
// seuls suffisent (eu = 0). Indices littéraux, stride SoA (a = 9·n + k).
fn wall_iso(t: f32, a: u32) {
    g_out[a + 0u * P.nx * P.ny] = W0 * t;
    g_out[a + 1u * P.nx * P.ny] = W1 * t;
    g_out[a + 2u * P.nx * P.ny] = W1 * t;
    g_out[a + 3u * P.nx * P.ny] = W1 * t;
    g_out[a + 4u * P.nx * P.ny] = W1 * t;
    g_out[a + 5u * P.nx * P.ny] = W5 * t;
    g_out[a + 6u * P.nx * P.ny] = W5 * t;
    g_out[a + 7u * P.nx * P.ny] = W5 * t;
    g_out[a + 8u * P.nx * P.ny] = W5 * t;
}

// Collision TRT d'une direction avec force de Guo (Guo, Shi & Zheng 2002),
// F = (0, fy, 0) : partie impaire (momentum, facteur 1 − 1/(2τ−)) et partie
// paire (stress, facteur 1 − 1/(2τ+)). P1a (variables décalées) : fin et la
// valeur de retour sont DÉCALÉS (f̃ = f − w_i, f̃' = f' − w_i) — l'offset
// w_i se simplifie EXACTEMENT dans la formule CPU : f' = (w + f̃) − a·d −
// b·d_opp + S = w + [f̃ − a·d − b·d_opp + S] puisque d_i = f̃_i − (feq_i −
// w_i) contient déjà le même w_i de part et d'autre. AUCUN (w + fin)
// réinjecté : le gros nombre (~0.44) ne traverse plus la collision. Réplique
// exacte, direction par direction, de la boucle de collision de
// `collide_node` (solver.rs) :
//   S_i = w_i·[(e_i−u)/cs² + (e_i·u)·e_i/cs⁴]·F
//   f̃_new = f̃ − a_f·d_i − b_f·d_opp + (1−1/(2τ−))·S⁻ + (1−1/(2τ+))·S⁺
// (le paramètre w reste pour la signature/les termes S_i = w_i·[…]·fy, qui
// ne dépendent pas de f.)
fn trt_f(w: f32, cx: f32, cy: f32, fin: f32, d: f32, d_opp: f32, ux: f32, uy: f32,
         fy_node: f32, a_f: f32, b_f: f32, tau_minus: f32, tau_plus: f32) -> f32 {
    let ei_u = cx * ux + cy * uy;
    let s_i = w * ((cy - uy) / CS2 + ei_u * cy / (CS2 * CS2)) * fy_node;
    let s_minus = w * (cy / CS2) * fy_node;
    let s_plus = s_i - s_minus;
    return fin - a_f * d - b_f * d_opp
        + (1.0 - 0.5 / tau_minus) * s_minus
        + (1.0 - 0.5 / tau_plus) * s_plus;
}

// ---------------------------------------------------------------------------
// Dispatch 1 — collision TRT + Smagorinsky + Boussinesq (par nœud, sans
// dépendance voisine). Réplique `collide_node` (solver.rs) : ν_t = (Cs·Δx)²·|S|
// par nœud, τ+ = 0.5 + 3(ν_eff + ν_t), τ_t+ = 0.5 + 3(α_eff + ν_t/Pr_t),
// force de Guo odd/even avec fy évaluée aux champs du temps t.
// ---------------------------------------------------------------------------
@compute @workgroup_size(16, 16)
fn collide(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    let iy = gid.y;
    if (ix >= P.nx || iy >= P.ny) { return; }
    let k = iy * P.nx + ix;
    let n = n_tot();

    // Champs macroscopiques du temps t (stock, écrits par le dispatch 2 du
    // pas précédent — y compris la correction de vitesse Guo et le no-slip).
    // P1a : le ρ est reconstruit des DÉCALAGES f̃ (plus lu depuis stock —
    // cohérence des moments : feq, |S| et fy utilisent le même ρ décalé).
    let ux = stock[4u * k + 1u];
    let uy = stock[4u * k + 2u];
    let temp = stock[4u * k + 3u];

    // P1a (variables décalées) : ρ−1 = Σ f̃_i, calculé comme la SOMME DES
    // DÉCALAGES (jamais les populations recollées W + f̃) — la physique est
    // dans l'écart f_i − w_i (1e-6..1e-3), la somme des 0.44 annulerait le
    // signal dans f32 (biais d'arrondi systématique, dérive de masse G5 §4 :
    // +1.446e-8/pas).
    let f0s = f_in[0u * n + k];
    let rho_m1 = f0s + f_in[1u * n + k] + f_in[2u * n + k] + f_in[3u * n + k]
        + f_in[4u * n + k] + f_in[5u * n + k] + f_in[6u * n + k] + f_in[7u * n + k]
        + f_in[8u * n + k];
    let rho = 1.0 + rho_m1;

    // Force de Boussinesq (P1b, ρ₀ dans la flottabilité) : fy =
    // β_eff·(T − t_ref)·(1 + rho_m1), évaluée aux champs du temps t —
    // utilisée par la collision de CE pas. Identique en arithmétique exacte
    // à l'ancien ρ·β_eff·(T−t_ref) (le terme dominant ρ₀ = 1 absorbé, seul
    // le petit ρ−1 = Σ f̃ multiplie — plus de grosse magnitude ρ dans la
    // force). Le dispatch 2 (correction de vitesse Guo éq. (16)) recalcule
    // localement depuis f_out avec la MÊME expression (même F — sémantique
    // CPU) ; le buffer fy global est supprimé (Étape C). Formule STRICTEMENT
    // identique au CPU (solver.rs : beta_eff·(T−t_ref)·(1 + (rho−1))).
    let fy_node = P.beta_eff * (temp - P.t_ref) * (1.0 + rho_m1);

    // Populations d'entrée en DÉCALÉ (f̃_i = f_i − w_i, la physique) —
    // stride SoA : direction i à f_in[i·n + k]. AUCUN recollage W + f̃ :
    // le gros nombre (≈0.44) serait recollé puis re-soustrait à chaque
    // opération, annulant le gain de précision de l'offset.
    let f1s = f_in[1u * n + k];
    let f2s = f_in[2u * n + k];
    let f3s = f_in[3u * n + k];
    let f4s = f_in[4u * n + k];
    let f5s = f_in[5u * n + k];
    let f6s = f_in[6u * n + k];
    let f7s = f_in[7u * n + k];
    let f8s = f_in[8u * n + k];
    let g0 = g_in[0u * n + k];
    let g1 = g_in[1u * n + k];
    let g2 = g_in[2u * n + k];
    let g3 = g_in[3u * n + k];
    let g4 = g_in[4u * n + k];
    let g5 = g_in[5u * n + k];
    let g6 = g_in[6u * n + k];
    let g7 = g_in[7u * n + k];
    let g8 = g_in[8u * n + k];

    // Équilibres en DÉCALÉ (P1a : feq_i − w_i, mêmes expressions que la
    // boucle au terme w_i·ρ·A près — identité exacte). rho_m1 = Σ f̃, rho
    // pleine magnitude en facteur de ρ·A uniquement.
    let u2 = ux * ux + uy * uy;
    let c1 = 1.0 - u2 / (2.0 * CS2);
    let f_eq0 = feq_off_c(W0, 0.0, 0.0, rho_m1, rho, ux, uy);
    let f_eq1 = feq_off_c(W1, 1.0, 0.0, rho_m1, rho, ux, uy);
    let f_eq2 = feq_off_c(W1, 0.0, 1.0, rho_m1, rho, ux, uy);
    let f_eq3 = feq_off_c(W1, -1.0, 0.0, rho_m1, rho, ux, uy);
    let f_eq4 = feq_off_c(W1, 0.0, -1.0, rho_m1, rho, ux, uy);
    let f_eq5 = feq_off_c(W5, 1.0, 1.0, rho_m1, rho, ux, uy);
    let f_eq6 = feq_off_c(W5, -1.0, 1.0, rho_m1, rho, ux, uy);
    let f_eq7 = feq_off_c(W5, -1.0, -1.0, rho_m1, rho, ux, uy);
    let f_eq8 = feq_off_c(W5, 1.0, -1.0, rho_m1, rho, ux, uy);
    let g_eq0 = geq_c(W0, 0.0, 0.0, temp, ux, uy);
    let g_eq1 = geq_c(W1, 1.0, 0.0, temp, ux, uy);
    let g_eq2 = geq_c(W1, 0.0, 1.0, temp, ux, uy);
    let g_eq3 = geq_c(W1, -1.0, 0.0, temp, ux, uy);
    let g_eq4 = geq_c(W1, 0.0, -1.0, temp, ux, uy);
    let g_eq5 = geq_c(W5, 1.0, 1.0, temp, ux, uy);
    let g_eq6 = geq_c(W5, -1.0, 1.0, temp, ux, uy);
    let g_eq7 = geq_c(W5, -1.0, -1.0, temp, ux, uy);
    let g_eq8 = geq_c(W5, 1.0, -1.0, temp, ux, uy);

    // Moments non-équilibrés π_neq (avant relaxation) pour |S| — P1a :
    // d_i = f̃_i − (feq_i − w_i), calculés UNIQUEMENT sur les petits nombres
    // (hors équilibre, mêmes contributions et ordre que la boucle : i = 0..9 ;
    // les directions cardinales ont cx²=1 ou cy²=1, les 4 diagonales
    // cx²=cy²=1 ; cx·cy est nul sur les cardinales, ±1 sur les diagonales).
    let d0 = f0s - f_eq0;
    let d1 = f1s - f_eq1;
    let d2 = f2s - f_eq2;
    let d3 = f3s - f_eq3;
    let d4 = f4s - f_eq4;
    let d5 = f5s - f_eq5;
    let d6 = f6s - f_eq6;
    let d7 = f7s - f_eq7;
    let d8 = f8s - f_eq8;
    let pi_xx = d1 + d3 + d5 + d6 + d7 + d8;
    let pi_xy = d5 - d6 + d7 - d8;
    let pi_yy = d2 + d4 + d5 + d6 + d7 + d8;
    // |S| = sqrt(π:π) / (2·cs²·τ_mol·ρ) ; π:π = Πxx² + 2Πxy² + Πyy²
    // (même convention que le CPU : |S|_canonique/√2).
    let pi2 = pi_xx * pi_xx + 2.0 * pi_xy * pi_xy + pi_yy * pi_yy;
    let s_mag = sqrt(pi2) / (2.0 * P.cs2 * P.tau_mol * max(rho, 1e-12));
    // Viscosité turbulente : ν_t = (Cs·Δx)²·|S| (Δx = 1), par nœud.
    let nu_t_node = P.cs_smag * P.cs_smag * s_mag;
    nu_t[k] = nu_t_node;

    // Temps de relaxation effectifs (par nœud — dépendent de ν_t) :
    // τ+ = 0.5 + 3(ν_eff + ν_t), τ− = 0.5 + Λ/(τ+ − 0.5) ;
    // scalaire : α_t = ν_t/Pr_t, τ_t+ = 0.5 + 3(α_eff + α_t), τ_t− = τ_t+
    // (BGK scalaire, patch CPU inchangé).
    let tau_plus = 0.5 + 3.0 * (P.nu_eff + nu_t_node);
    let tau_minus = 0.5 + P.lambda_trt / (tau_plus - 0.5);
    let alpha_t = nu_t_node / P.pr_t;
    let tau_t_plus = 0.5 + 3.0 * (P.alpha_eff + alpha_t);
    let a_f = 0.5 * (1.0 / tau_plus + 1.0 / tau_minus);
    let b_f = 0.5 * (1.0 / tau_plus - 1.0 / tau_minus);
    let a_g = 1.0 / tau_t_plus; // τ_t− = τ_t+ ⇒ b_g = 0

    // Collision TRT avec force (Guo) — paires opposées explicites
    // (opp(0)=0, opp(1)=3, opp(2)=4, opp(5)=7, opp(6)=8). P1a : trt_f
    // travaille en DÉCALÉ (fin = f̃, retourne f̃') — l'offset w_i se
    // simplifie EXACTEMENT dans l'identité f' = w + f̃' : la partie
    // (w + fin) − ... − w s'annule, plus aucun gros nombre (~0.44) dans
    // la collision. Σ f̃' = Σ f̃ : masse conservée, sémantique TRT
    // identique. Deviations d_i/d_opp en petits nombres (identiques à la
    // boucle au terme w_i près, qui se simplifie exactement).
    f_col[0u * n + k] = trt_f(W0, 0.0, 0.0, f_in[0u * n + k], d0, d0, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[3u * n + k] = trt_f(W1, -1.0, 0.0, f_in[3u * n + k], d3, d1, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[1u * n + k] = trt_f(W1, 1.0, 0.0, f_in[1u * n + k], d1, d3, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[4u * n + k] = trt_f(W1, 0.0, -1.0, f_in[4u * n + k], d4, d2, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[2u * n + k] = trt_f(W1, 0.0, 1.0, f_in[2u * n + k], d2, d4, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[7u * n + k] = trt_f(W5, -1.0, -1.0, f_in[7u * n + k], d7, d5, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[5u * n + k] = trt_f(W5, 1.0, 1.0, f_in[5u * n + k], d5, d7, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[8u * n + k] = trt_f(W5, 1.0, -1.0, f_in[8u * n + k], d8, d6, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[6u * n + k] = trt_f(W5, -1.0, 1.0, f_in[6u * n + k], d6, d8, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus);

    // Collision scalaire BGK (τ_t− = τ_t+ ⇒ b_g = 0).
    g_col[0u * n + k] = g0 - a_g * (g0 - g_eq0);
    g_col[1u * n + k] = g1 - a_g * (g1 - g_eq1);
    g_col[2u * n + k] = g2 - a_g * (g2 - g_eq2);
    g_col[3u * n + k] = g3 - a_g * (g3 - g_eq3);
    g_col[4u * n + k] = g4 - a_g * (g4 - g_eq4);
    g_col[5u * n + k] = g5 - a_g * (g5 - g_eq5);
    g_col[6u * n + k] = g6 - a_g * (g6 - g_eq6);
    g_col[7u * n + k] = g7 - a_g * (g7 - g_eq7);
    g_col[8u * n + k] = g8 - a_g * (g8 - g_eq8);
}

// ---------------------------------------------------------------------------
// Dispatch 2 — streaming pull + bounce-back PUIS CL thermiques PUIS
// macroscopique (fusion Étape C : 2 dispatches/pas au lieu de 3). Lit
// f_col/g_col FIGÉS du dispatch 1 (même sémantique que le CPU, qui clone f/g
// avant streaming) : f[i](x) ← f_col(x − c_i), sinon bounce-back
// f_col(x, opp(i)). Les CL thermiques recalcule la valeur du nœud intérieur
// DEPUIS g_col (figé, post-collision — la CL verticale du voisin est
// geq(T_wall) recalculable localement) : aucune lecture inter-thread.
// ---------------------------------------------------------------------------
@compute @workgroup_size(16, 16)
fn stream_macro(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    let iy = gid.y;
    if (ix >= P.nx || iy >= P.ny) { return; }
    let k = iy * P.nx + ix;
    let n = n_tot();

    // Streaming pull + bounce-back, direction par direction (indices
    // littéraux, offsets ±1/±nx explicites — mêmes CL que la boucle) :
    // source (ix − cx, iy − cy) dans le domaine → pull, sinon bounce-back
    // de la distribution sortante opposée (g : valeur provisoire, remplacée
    // par les CL thermiques ci-dessous).
    // i=0 rest : (0,0), opp=0 — toujours dans le domaine.
    f_out[0u * n + k] = f_col[0u * n + k];
    g_out[0u * n + k] = g_col[0u * n + k];
    // i=1 E : source (ix−1, iy), opp=3 (W).
    if (ix >= 1u) {
        let src = k - 1u;
        f_out[1u * n + k] = f_col[1u * n + src];
        g_out[1u * n + k] = g_col[1u * n + src];
    } else {
        f_out[1u * n + k] = f_col[3u * n + k];
        g_out[1u * n + k] = g_col[1u * n + k];
    }
    // i=2 N : source (ix, iy−1), opp=4 (S).
    if (iy >= 1u) {
        let src = k - P.nx;
        f_out[2u * n + k] = f_col[2u * n + src];
        g_out[2u * n + k] = g_col[2u * n + src];
    } else {
        f_out[2u * n + k] = f_col[4u * n + k];
        g_out[2u * n + k] = g_col[2u * n + k];
    }
    // i=3 W : source (ix+1, iy), opp=1 (E).
    if (ix + 1u < P.nx) {
        let src = k + 1u;
        f_out[3u * n + k] = f_col[3u * n + src];
        g_out[3u * n + k] = g_col[3u * n + src];
    } else {
        f_out[3u * n + k] = f_col[1u * n + k];
        g_out[3u * n + k] = g_col[3u * n + k];
    }
    // i=4 S : source (ix, iy+1), opp=2 (N).
    if (iy + 1u < P.ny) {
        let src = k + P.nx;
        f_out[4u * n + k] = f_col[4u * n + src];
        g_out[4u * n + k] = g_col[4u * n + src];
    } else {
        f_out[4u * n + k] = f_col[2u * n + k];
        g_out[4u * n + k] = g_col[4u * n + k];
    }
    // i=5 NE : source (ix−1, iy−1), opp=7 (SW).
    if (ix >= 1u && iy >= 1u) {
        let src = k - P.nx - 1u;
        f_out[5u * n + k] = f_col[5u * n + src];
        g_out[5u * n + k] = g_col[5u * n + src];
    } else {
        f_out[5u * n + k] = f_col[7u * n + k];
        g_out[5u * n + k] = g_col[5u * n + k];
    }
    // i=6 NW : source (ix+1, iy−1), opp=8 (SE).
    if (ix + 1u < P.nx && iy >= 1u) {
        let src = k - P.nx + 1u;
        f_out[6u * n + k] = f_col[6u * n + src];
        g_out[6u * n + k] = g_col[6u * n + src];
    } else {
        f_out[6u * n + k] = f_col[8u * n + k];
        g_out[6u * n + k] = g_col[6u * n + k];
    }
    // i=7 SW : source (ix+1, iy+1), opp=5 (NE).
    if (ix + 1u < P.nx && iy + 1u < P.ny) {
        let src = k + P.nx + 1u;
        f_out[7u * n + k] = f_col[7u * n + src];
        g_out[7u * n + k] = g_col[7u * n + src];
    } else {
        f_out[7u * n + k] = f_col[5u * n + k];
        g_out[7u * n + k] = g_col[7u * n + k];
    }
    // i=8 SE : source (ix−1, iy+1), opp=6 (NW).
    if (ix >= 1u && iy + 1u < P.ny) {
        let src = k + P.nx - 1u;
        f_out[8u * n + k] = f_col[8u * n + src];
        g_out[8u * n + k] = g_col[8u * n + src];
    } else {
        f_out[8u * n + k] = f_col[6u * n + k];
        g_out[8u * n + k] = g_col[8u * n + k];
    }

    // --- CL thermiques. Le CPU applique les BC thermiques AVANT
    // update_macroscopic : le T de paroi est calculé sur le g post-CL.
    // Ordre CPU : verticales d'abord, horizontales ensuite (les coins sont
    // écrasées par la copie horizontale). La valeur du nœud intérieur
    // (post-stream) est recalculée DEPUIS g_col FIGÉ (post-collision) —
    // aucune lecture g_out inter-thread, aucune course (la valeur pull/BB
    // du voisin est identique à la sienne : g_col est le champ post-collision
    // du même pas, avant tout streaming).

    // --- Parois verticales (chaude à gauche, froide à droite) : isothermes,
    // g paroi = geq(T_wall, u = 0) pour les 9 directions.
    if (ix == 0u) {
        wall_iso(P.t_hot, k);
    } else if (ix == P.nx - 1u) {
        wall_iso(P.t_cold, k);
    }
    // --- Parois horizontales : adiabatiques, g paroi = copie du nœud
    // intérieur adjacent (post-stream, recalculé depuis g_col). Si ce nœud
    // est lui-même un coin (paroi verticale), sa valeur post-CL est
    // geq(T_wall) — recalculée localement (identité CPU).
    if (iy == 0u || iy == P.ny - 1u) {
        let iyin = select(iy + 1u, iy - 1u, iy == P.ny - 1u);
        let kin = iyin * P.nx + ix;
        // i=0 rest : post-stream = g_col du voisin (direction 0).
        g_out[0u * n + k] = g_col[0u * n + kin];
        // i=1 E : source du voisin (ix−1, iyin).
        g_out[1u * n + k] = select(
            g_col[1u * n + (kin - 1u)],
            g_col[1u * n + kin],
            ix < 1u,
        );
        // i=2 N : source du voisin (ix, iyin−1) — si le voisin est à
        // iyin−1 hors domaine, bounce-back provisoire (même identité CPU).
        g_out[2u * n + k] = select(
            g_col[2u * n + (kin - P.nx)],
            g_col[2u * n + kin],
            iyin < 1u,
        );
        // i=3 W : source du voisin (ix+1, iyin).
        g_out[3u * n + k] = select(
            g_col[3u * n + (kin + 1u)],
            g_col[3u * n + kin],
            ix + 1u >= P.nx,
        );
        // i=4 S : source du voisin (ix, iyin+1).
        g_out[4u * n + k] = select(
            g_col[4u * n + (kin + P.nx)],
            g_col[4u * n + kin],
            iyin + 1u >= P.ny,
        );
        // i=5 NE : source du voisin (ix−1, iyin−1).
        g_out[5u * n + k] = select(
            g_col[5u * n + (kin - P.nx - 1u)],
            g_col[5u * n + kin],
            ix < 1u || iyin < 1u,
        );
        // i=6 NW : source du voisin (ix+1, iyin−1).
        g_out[6u * n + k] = select(
            g_col[6u * n + (kin - P.nx + 1u)],
            g_col[6u * n + kin],
            ix + 1u >= P.nx || iyin < 1u,
        );
        // i=7 SW : source du voisin (ix+1, iyin+1).
        g_out[7u * n + k] = select(
            g_col[7u * n + (kin + P.nx + 1u)],
            g_col[7u * n + kin],
            ix + 1u >= P.nx || iyin + 1u >= P.ny,
        );
        // i=8 SE : source du voisin (ix−1, iyin+1).
        g_out[8u * n + k] = select(
            g_col[8u * n + (kin + P.nx - 1u)],
            g_col[8u * n + kin],
            ix < 1u || iyin + 1u >= P.ny,
        );
        // Coin (ix ∈ {0, nx−1}) : le voisin intérieur (ix, iy±1) est lui-même
        // sur la paroi verticale — le CPU copie sa valeur POST-CL verticale
        // (= geq(T_wall), appliquée AVANT la copie horizontale), pas son
        // post-stream. Écrasement identique au kernel validé 3 dispatches.
        if (ix == 0u) {
            wall_iso(P.t_hot, k);
        } else if (ix == P.nx - 1u) {
            wall_iso(P.t_cold, k);
        }
    }

    // --- Macroscopique (ρ, u, T) depuis f_out / g_out (post-stream + CL) —
    // P1a (variables décalées) : ρ−1 = Σ f̃_i (somme des décalages, SANS
    // recoller les W + f̃ — la somme des 0.44 annulerait le signal f32) ;
    // mx/my = Σ c_i·f_i = Σ c_i·f̃_i EXACTEMENT (Σ c_i·w_i = 0, symétrie
    // D2Q9 — les poids opposés sont égaux), donc moments en décalé
    // identiques au CPU au bit près, sans aucun gros nombre. Déroulé, même
    // ordre d'accumulation que la boucle (j = 0..9).
    var rho_m1 = 0.0;
    var mx = 0.0;
    var my = 0.0;
    var temp = 0.0;
    let f0s = f_out[0u * n + k];
    rho_m1 = rho_m1 + f0s;
    temp = temp + g_out[0u * n + k];
    let f1s = f_out[1u * n + k];
    let f2s = f_out[2u * n + k];
    let f3s = f_out[3u * n + k];
    let f4s = f_out[4u * n + k];
    let f5s = f_out[5u * n + k];
    let f6s = f_out[6u * n + k];
    let f7s = f_out[7u * n + k];
    let f8s = f_out[8u * n + k];
    rho_m1 = rho_m1 + f1s + f2s + f3s + f4s + f5s + f6s + f7s + f8s;
    mx = mx + f1s - f3s + f5s - f6s - f7s + f8s;
    my = my + f2s - f4s + f5s + f6s - f7s - f8s;
    temp = temp + g_out[1u * n + k] + g_out[2u * n + k] + g_out[3u * n + k]
        + g_out[4u * n + k] + g_out[5u * n + k] + g_out[6u * n + k]
        + g_out[7u * n + k] + g_out[8u * n + k];
    let rho = 1.0 + rho_m1;
    // Correction de vitesse Guo et al. 2002, éq. (16) : u_y += fy/(2ρ) —
    // fy recalculé localement avec l'expression P1b (ρ₀ dans la flottabilité,
    // identique en arithmétique exacte à l'ancien ρ·β_eff·(T−t_ref)) depuis
    // f_out (même expression que la collision de CE pas — même F, sémantique
    // CPU `update_macroscopic` ; le buffer fy global est supprimé). Formule
    // STRICTEMENT identique au CPU. ρ pleine magnitude uniquement au
    // dénominateur (division, inchangé).
    var ux = 0.0;
    var uy = 0.0;
    let fy_node = P.beta_eff * (temp - P.t_ref) * (1.0 + rho_m1);
    if (rho > 1e-12) {
        ux = mx / rho;
        uy = my / rho + 0.5 * fy_node / rho;
    }
    stock[4u * k + 0u] = rho;
    stock[4u * k + 1u] = ux;
    stock[4u * k + 2u] = uy;
    stock[4u * k + 3u] = temp;

    // --- No-slip : u = 0 sur le périmètre (enforce_wall_velocity du CPU).
    if (ix == 0u || ix == P.nx - 1u || iy == 0u || iy == P.ny - 1u) {
        stock[4u * k + 1u] = 0.0;
        stock[4u * k + 2u] = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Étape D (porte G4) — dynamic Smagorinsky (échelles mixtes §2.3.5 thèse).
// Réplique exacte, en f32, de `compute_dynamic_sgs` (solver.rs) : 4 kernels
// ordonnés dans le MÊME compute pass, avant la collision (les barrières
// storage wgpu entre dispatches d'un même pass garantissent l'ordre) :
//   dyn1 : box test-filter (η·Δ̃) des champs stock et des produits (Léonard).
//   dyn2 : S̃ij/|S̃| depuis π_neq (f_in), S̄ij/|S̄| (box dyn_s1), gradients,
//          qsm^½, Φsm^½, |T̃|, |T̄| (Éq. 2.74/2.75) → dyn_s2.
//   dyn3 : box(qsm^½/Φsm^½), formes mixtes interpolées (Éq. 2.73), tenseurs
//          Germano M/L, contractions de Lilly (Éq. 2.58/2.70) → dyn_s3.
//   dyn4 : régularisation Éq. 2.62 (box num/den), coefficient de Lilly
//          (clamp [0, 0.04]), ν_t/α_t bornés (nu_sm_min + plancher TRT)
//          → dyn_out[k] = (ν_t, α_t, C^ν_dyn, C^κ_dyn).
//   collide_dyn : réplique de `collide` consommant ν_t/α_t de dyn_out.
// NOTATION THÈSE (inversée vs convention usuelle) : Δ̃ = filtre grille = Δx ;
//   Δ = filtre test = η·Δx ; ũ (thèse) = champ résolu, u (thèse) = test-filtré.
// ---------------------------------------------------------------------------

// Paramètres du modèle dynamique (constants au cours du run — voir Rust).
struct DynParams {
    alpha_exp: f32,       // exposant α de l'interpolation mixte
    eta_test: f32,        // η = Δ/Δ̃
    dtest: f32,           // Δ = η·Δx (Δx = 1)
    delta_t: f32,         // ΔT global = 1 (PAS Δ̃·T — Tab. 2.3)
    nu_mol: f32,          // ν moléculaire cible (nu_sm_min·ν)
    alpha_mol: f32,       // α moléculaire cible
    nu_sm_min: f32,       // borne νsm ≥ nu_sm_min·ν
    tau_floor_margin: f32, // (TAU_TRT_FLOOR − 0.5)/3
    box_width: u32,       // largeur du filtre box (nœuds, ≥ 1)
    c_dyn_max: f32,       // clamp C ∈ [0, 0.04]
    pad0: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
    pad4: u32,
    pad5: u32,
};
@group(0) @binding(10) var<uniform> D: DynParams;
@group(0) @binding(11) var<storage, read_write> dyn_s1: array<f32>;
@group(0) @binding(12) var<storage, read_write> dyn_s2: array<f32>;
@group(0) @binding(13) var<storage, read_write> dyn_s3: array<f32>;
@group(0) @binding(14) var<storage, read_write> dyn_out: array<f32>;

// dyn_s1 (stride 8/nœud) : [ũx, ũy, Ŧ, ⟨uxux⟩, ⟨uxuy⟩, ⟨uyuy⟩, ⟨uxT⟩, ⟨uyT⟩]
// dyn_s2 (stride 16) : [S̃xx, S̃xy, S̃yy, |S̃|, S̄xx, S̄xy, S̄yy, |S̄|,
//                       ∂T̃x, ∂T̃y, ∂T̄x, ∂T̄y, qsm^½, Φsm^½, |T̃|, |T̄|]
// dyn_s3 (stride 8) : [num_v, den_v, num_k, den_k, νsm_mix(grille), κsm_mix(grille), −, −]
// dyn_out (stride 4) : [ν_t, α_t, C^ν_dyn, C^κ_dyn]

// Fenêtre box [lo, hi) le long d'un axe (clamp unilatéral aux parois, comme
// box_filter du CPU : half_left = (width−1)/2, fenêtre [i−half_left, i+w−half_left)).
fn box_lo(i: i32) -> i32 {
    let half_left = i32((D.box_width - 1u) / 2u);
    return max(i - half_left, 0);
}
fn box_hi(i: i32, n: i32) -> i32 {
    let half_left = i32((D.box_width - 1u) / 2u);
    let w = i32(D.box_width);
    return min(i + w - half_left, n);
}

// Box d'une composante c du stock (1=ux, 2=uy, 3=T).
fn box_stock(ix: u32, iy: u32, c: u32) -> f32 {
    let x0 = box_lo(i32(ix));
    let x1 = box_hi(i32(ix), i32(P.nx));
    let y0 = box_lo(i32(iy));
    let y1 = box_hi(i32(iy), i32(P.ny));
    var s = 0.0;
    var yy = y0;
    loop {
        if (yy >= y1) { break; }
        var xx = x0;
        loop {
            if (xx >= x1) { break; }
            s = s + stock[4u * (u32(yy) * P.nx + u32(xx)) + c];
            xx = xx + 1;
        }
        yy = yy + 1;
    }
    return s / f32((y1 - y0) * (x1 - x0));
}

// Box du produit de deux composantes du stock (Léonard au filtre test).
fn box_stock_prod(ix: u32, iy: u32, c1: u32, c2: u32) -> f32 {
    let x0 = box_lo(i32(ix));
    let x1 = box_hi(i32(ix), i32(P.nx));
    let y0 = box_lo(i32(iy));
    let y1 = box_hi(i32(iy), i32(P.ny));
    var s = 0.0;
    var yy = y0;
    loop {
        if (yy >= y1) { break; }
        var xx = x0;
        loop {
            if (xx >= x1) { break; }
            let kk = u32(yy) * P.nx + u32(xx);
            s = s + stock[4u * kk + c1] * stock[4u * kk + c2];
            xx = xx + 1;
        }
        yy = yy + 1;
    }
    return s / f32((y1 - y0) * (x1 - x0));
}

// Box d'un composant c de dyn_s1 (stride 8).
fn box_s1(ix: u32, iy: u32, c: u32) -> f32 {
    let x0 = box_lo(i32(ix));
    let x1 = box_hi(i32(ix), i32(P.nx));
    let y0 = box_lo(i32(iy));
    let y1 = box_hi(i32(iy), i32(P.ny));
    var s = 0.0;
    var yy = y0;
    loop {
        if (yy >= y1) { break; }
        var xx = x0;
        loop {
            if (xx >= x1) { break; }
            s = s + dyn_s1[8u * (u32(yy) * P.nx + u32(xx)) + c];
            xx = xx + 1;
        }
        yy = yy + 1;
    }
    return s / f32((y1 - y0) * (x1 - x0));
}

// Box d'un composant c de dyn_s2 (stride 16).
fn box_s2(ix: u32, iy: u32, c: u32) -> f32 {
    let x0 = box_lo(i32(ix));
    let x1 = box_hi(i32(ix), i32(P.nx));
    let y0 = box_lo(i32(iy));
    let y1 = box_hi(i32(iy), i32(P.ny));
    var s = 0.0;
    var yy = y0;
    loop {
        if (yy >= y1) { break; }
        var xx = x0;
        loop {
            if (xx >= x1) { break; }
            s = s + dyn_s2[16u * (u32(yy) * P.nx + u32(xx)) + c];
            xx = xx + 1;
        }
        yy = yy + 1;
    }
    return s / f32((y1 - y0) * (x1 - x0));
}

// Box d'un composant c de dyn_s3 (stride 8).
fn box_s3(ix: u32, iy: u32, c: u32) -> f32 {
    let x0 = box_lo(i32(ix));
    let x1 = box_hi(i32(ix), i32(P.nx));
    let y0 = box_lo(i32(iy));
    let y1 = box_hi(i32(iy), i32(P.ny));
    var s = 0.0;
    var yy = y0;
    loop {
        if (yy >= y1) { break; }
        var xx = x0;
        loop {
            if (xx >= x1) { break; }
            s = s + dyn_s3[8u * (u32(yy) * P.nx + u32(xx)) + c];
            xx = xx + 1;
        }
        yy = yy + 1;
    }
    return s / f32((y1 - y0) * (x1 - x0));
}

// Dérivées centrales, parois clampées unilatéralement (ddx/ddy du CPU).
fn ddx_s1(ix: u32, iy: u32, c: u32) -> f32 {
    let xm = select(ix - 1u, 0u, ix == 0u);
    let xp = select(ix + 1u, P.nx - 1u, ix + 1u >= P.nx);
    return 0.5 * (dyn_s1[8u * (iy * P.nx + xp) + c] - dyn_s1[8u * (iy * P.nx + xm) + c]);
}
fn ddy_s1(ix: u32, iy: u32, c: u32) -> f32 {
    let ym = select(iy - 1u, 0u, iy == 0u);
    let yp = select(iy + 1u, P.ny - 1u, iy + 1u >= P.ny);
    return 0.5 * (dyn_s1[8u * (yp * P.nx + ix) + c] - dyn_s1[8u * (ym * P.nx + ix) + c]);
}
fn ddx_stock(ix: u32, iy: u32, c: u32) -> f32 {
    let xm = select(ix - 1u, 0u, ix == 0u);
    let xp = select(ix + 1u, P.nx - 1u, ix + 1u >= P.nx);
    return 0.5 * (stock[4u * (iy * P.nx + xp) + c] - stock[4u * (iy * P.nx + xm) + c]);
}
fn ddy_stock(ix: u32, iy: u32, c: u32) -> f32 {
    let ym = select(iy - 1u, 0u, iy == 0u);
    let yp = select(iy + 1u, P.ny - 1u, iy + 1u >= P.ny);
    return 0.5 * (stock[4u * (yp * P.nx + ix) + c] - stock[4u * (ym * P.nx + ix) + c]);
}

// Puissance mixte : 0^0 → 1 (même sémantique que powf du CPU, cas α = 1).
fn mpow(x: f32, e: f32) -> f32 {
    return select(1.0, pow(x, e), e != 0.0);
}

// Coefficient de Lilly régularisé (lilly_coefficient du CPU) : C = (1/2)·num/den,
// garde-fou den ≤ 0 / non fini → C = 0, clamp [0, C_DYN_MAX].
fn lilly(num: f32, den: f32) -> f32 {
    if (!(den > 0.0)) { return 0.0; }
    let c = 0.5 * num / den;
    let ok = (c == c) && (abs(c) < 3.4028235e38);
    return select(0.0, clamp(c, 0.0, D.c_dyn_max), ok);
}

// --- dyn1 : box test-filter des champs stock et des produits (passe 1 CPU).
@compute @workgroup_size(16, 16)
fn dyn1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    let iy = gid.y;
    if (ix >= P.nx || iy >= P.ny) { return; }
    let k = iy * P.nx + ix;
    dyn_s1[8u * k + 0u] = box_stock(ix, iy, 1u);           // ũx (champ résolu boxé)
    dyn_s1[8u * k + 1u] = box_stock(ix, iy, 2u);           // ũy
    dyn_s1[8u * k + 2u] = box_stock(ix, iy, 3u);           // Ŧ (test)
    dyn_s1[8u * k + 3u] = box_stock_prod(ix, iy, 1u, 1u);  // ⟨ux ux⟩_test
    dyn_s1[8u * k + 4u] = box_stock_prod(ix, iy, 1u, 2u);  // ⟨ux uy⟩_test
    dyn_s1[8u * k + 5u] = box_stock_prod(ix, iy, 2u, 2u);  // ⟨uy uy⟩_test
    dyn_s1[8u * k + 6u] = box_stock_prod(ix, iy, 1u, 3u);  // ⟨ux T⟩_test
    dyn_s1[8u * k + 7u] = box_stock_prod(ix, iy, 2u, 3u);  // ⟨uy T⟩_test
}

// --- dyn2 : quantités aux deux niveaux (passe 2 CPU). S̃ij/|S̃| depuis
// π_neq de f_in (MÊMES formules que collide), S̄ij/|S̄| par différences
// centrées des champs boxés, gradients thermiques, similarité (Éq. 2.74),
// |T̃|/|T̄| (Éq. 2.75).
@compute @workgroup_size(16, 16)
fn dyn2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    let iy = gid.y;
    if (ix >= P.nx || iy >= P.ny) { return; }
    let k = iy * P.nx + ix;
    let n = n_tot();

    let ux = stock[4u * k + 1u];
    let uy = stock[4u * k + 2u];
    let temp = stock[4u * k + 3u];

    // Populations + équilibres en DÉCALÉ (P1a, identiques à collide :
    // ρ−1 = Σ f̃, feq_i − w_i = w_i·[(ρ−1) + ρ·A], d_i = f̃ − (feq_i − w_i) —
    // aucun recollage W + f̃). P1a : le ρ est reconstruit des DÉCALAGES
    // (plus lu depuis stock — cohérence des moments avec collide).
    let f0s = f_in[0u * n + k];
    let rho_m1 = f0s + f_in[1u * n + k] + f_in[2u * n + k] + f_in[3u * n + k]
        + f_in[4u * n + k] + f_in[5u * n + k] + f_in[6u * n + k] + f_in[7u * n + k]
        + f_in[8u * n + k];
    let rho = 1.0 + rho_m1;
    let f1s = f_in[1u * n + k];
    let f2s = f_in[2u * n + k];
    let f3s = f_in[3u * n + k];
    let f4s = f_in[4u * n + k];
    let f5s = f_in[5u * n + k];
    let f6s = f_in[6u * n + k];
    let f7s = f_in[7u * n + k];
    let f8s = f_in[8u * n + k];
    let u2 = ux * ux + uy * uy;
    let c1 = 1.0 - u2 / (2.0 * CS2);
    let f_eq0 = feq_off_c(W0, 0.0, 0.0, rho_m1, rho, ux, uy);
    let f_eq1 = feq_off_c(W1, 1.0, 0.0, rho_m1, rho, ux, uy);
    let f_eq2 = feq_off_c(W1, 0.0, 1.0, rho_m1, rho, ux, uy);
    let f_eq3 = feq_off_c(W1, -1.0, 0.0, rho_m1, rho, ux, uy);
    let f_eq4 = feq_off_c(W1, 0.0, -1.0, rho_m1, rho, ux, uy);
    let f_eq5 = feq_off_c(W5, 1.0, 1.0, rho_m1, rho, ux, uy);
    let f_eq6 = feq_off_c(W5, -1.0, 1.0, rho_m1, rho, ux, uy);
    let f_eq7 = feq_off_c(W5, -1.0, -1.0, rho_m1, rho, ux, uy);
    let f_eq8 = feq_off_c(W5, 1.0, -1.0, rho_m1, rho, ux, uy);
    let d0 = f0s - f_eq0;
    let d1 = f1s - f_eq1;
    let d2 = f2s - f_eq2;
    let d3 = f3s - f_eq3;
    let d4 = f4s - f_eq4;
    let d5 = f5s - f_eq5;
    let d6 = f6s - f_eq6;
    let d7 = f7s - f_eq7;
    let d8 = f8s - f_eq8;
    let pi_xx = d1 + d3 + d5 + d6 + d7 + d8;
    let pi_xy = d5 - d6 + d7 - d8;
    let pi_yy = d2 + d4 + d5 + d6 + d7 + d8;
    let pi2 = pi_xx * pi_xx + 2.0 * pi_xy * pi_xy + pi_yy * pi_yy;
    let denom = 2.0 * P.cs2 * P.tau_mol * max(rho, 1e-12);
    let shxx = -pi_xx / denom;
    let shxy = -pi_xy / denom;
    let shyy = -pi_yy / denom;
    let smag_hat = sqrt(pi2) / denom; // |S̃| identique collide

    // S̄ij, |S̄| par différences centrées des champs boxés (dyn_s1).
    let sxx = ddx_s1(ix, iy, 0u);
    let syy = ddy_s1(ix, iy, 1u);
    let sxy = 0.5 * (ddx_s1(ix, iy, 1u) + ddy_s1(ix, iy, 0u));
    let sbxx = sxx;
    let sbxy = sxy;
    let sbyy = syy;
    let smag_bar = sqrt(sxx * sxx + 2.0 * sxy * sxy + syy * syy);

    // Gradients thermiques : ∂T̃/∂xj (stock brut), ∂T̄/∂xj (champ boxé).
    let gtx = ddx_stock(ix, iy, 3u);
    let gty = ddy_stock(ix, iy, 3u);
    let gbx = ddx_s1(ix, iy, 2u);
    let gby = ddy_s1(ix, iy, 2u);

    // Similarité (Éq. 2.74) : qsm^½ = (1/(2√2))·√(Σ(ũ−ū)²) ;
    // Φsm^½ = (1/2)|T̃−T̄|·√((1/2)Σ(ũ−ū)²).
    let ubx_v = dyn_s1[8u * k + 0u];
    let uby_v = dyn_s1[8u * k + 1u];
    let tb_v = dyn_s1[8u * k + 2u];
    let dux = ux - ubx_v;
    let duy = uy - uby_v;
    let du2 = dux * dux + duy * duy;
    let qsm_h = sqrt(0.5 * du2) / 2.0;
    let dt_local = temp - tb_v;
    let phi_h = 0.5 * abs(dt_local) * sqrt(0.5 * du2);

    // |T̃|/|T̄| (Éq. 2.75) : Ti = ∂T/∂xj Sij, |T| = √(2 Ti Ti).
    let th_x = gtx * shxx + gty * shxy;
    let th_y = gtx * shxy + gty * shyy;
    let tmag_hat = sqrt(2.0 * (th_x * th_x + th_y * th_y));
    let tb_x = gbx * sbxx + gby * sbxy;
    let tb_y = gbx * sbxy + gby * sbyy;
    let tmag_bar = sqrt(2.0 * (tb_x * tb_x + tb_y * tb_y));

    dyn_s2[16u * k + 0u] = shxx;
    dyn_s2[16u * k + 1u] = shxy;
    dyn_s2[16u * k + 2u] = shyy;
    dyn_s2[16u * k + 3u] = smag_hat;
    dyn_s2[16u * k + 4u] = sbxx;
    dyn_s2[16u * k + 5u] = sbxy;
    dyn_s2[16u * k + 6u] = sbyy;
    dyn_s2[16u * k + 7u] = smag_bar;
    dyn_s2[16u * k + 8u] = gtx;
    dyn_s2[16u * k + 9u] = gty;
    dyn_s2[16u * k + 10u] = gbx;
    dyn_s2[16u * k + 11u] = gby;
    dyn_s2[16u * k + 12u] = qsm_h;
    dyn_s2[16u * k + 13u] = phi_h;
    dyn_s2[16u * k + 14u] = tmag_hat;
    dyn_s2[16u * k + 15u] = tmag_bar;
}

// --- dyn3 : q̄sm/Φ̄sm (box), formes mixtes interpolées (Éq. 2.73), tenseurs
// M/L mixtes, contractions de Lilly (passe 3 CPU).
@compute @workgroup_size(16, 16)
fn dyn3(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    let iy = gid.y;
    if (ix >= P.nx || iy >= P.ny) { return; }
    let k = iy * P.nx + ix;

    let shxx = dyn_s2[16u * k + 0u];
    let shxy = dyn_s2[16u * k + 1u];
    let shyy = dyn_s2[16u * k + 2u];
    let smag_hat = dyn_s2[16u * k + 3u];
    let sbxx = dyn_s2[16u * k + 4u];
    let sbxy = dyn_s2[16u * k + 5u];
    let sbyy = dyn_s2[16u * k + 6u];
    let smag_bar = dyn_s2[16u * k + 7u];
    let gtx = dyn_s2[16u * k + 8u];
    let gty = dyn_s2[16u * k + 9u];
    let gbx = dyn_s2[16u * k + 10u];
    let gby = dyn_s2[16u * k + 11u];
    let qsm_h = dyn_s2[16u * k + 12u];
    let phi_h = dyn_s2[16u * k + 13u];
    let tmag_hat = dyn_s2[16u * k + 14u];
    let tmag_bar = dyn_s2[16u * k + 15u];

    let ubx = dyn_s1[8u * k + 0u];
    let uby = dyn_s1[8u * k + 1u];
    let tb = dyn_s1[8u * k + 2u];
    let lbuxx = dyn_s1[8u * k + 3u];
    let lbuxy = dyn_s1[8u * k + 4u];
    let lbuyy = dyn_s1[8u * k + 5u];
    let lbutx = dyn_s1[8u * k + 6u];
    let lbuty = dyn_s1[8u * k + 7u];

    // q̄sm et Φ̄sm au filtre test : box des quantités construites (passe 2).
    let qbar_h = box_s2(ix, iy, 12u);
    let phibar_h = box_s2(ix, iy, 13u);

    // Formes mixtes interpolées (Tab. 2.3) — α = 0.5 ⇒ moyenne géométrique.
    let a = D.alpha_exp;
    let b = 1.0 - a;
    let dx = 1.0; // Δx
    let nu_mix_t = mpow(D.dtest * D.dtest * smag_bar, a) * mpow(D.dtest * qbar_h, b);
    let nu_mix_g = mpow(dx * dx * smag_hat, a) * mpow(dx * qsm_h, b);
    let kap_mix_t = mpow(D.dtest * D.dtest * D.dtest / D.delta_t * tmag_bar, a)
        * mpow(D.dtest / D.delta_t * phibar_h, b);
    let kap_mix_g = mpow(dx * dx * dx / D.delta_t * tmag_hat, a)
        * mpow(dx / D.delta_t * phi_h, b);

    // Tenseur M_ij mixte : νsm_mix(test)·S̄ij − νsm_mix(grille)·S̃ij.
    let mxx = nu_mix_t * sbxx - nu_mix_g * shxx;
    let mxy = nu_mix_t * sbxy - nu_mix_g * shxy;
    let myy = nu_mix_t * sbyy - nu_mix_g * shyy;
    // Vecteur M_j mixte : κsm_mix(test)·∂T̄/∂xj − κsm_mix(grille)·∂T̃/∂xj.
    let mx = kap_mix_t * gbx - kap_mix_g * gtx;
    let my = kap_mix_t * gby - kap_mix_g * gty;

    // Léonard au filtre test : L_ij = ⟨uiuj⟩_test − ũi ũj (Éq. 2.51).
    let lxx = lbuxx - ubx * ubx;
    let lxy = lbuxy - ubx * uby;
    let lyy = lbuyy - uby * uby;
    // L^d_ij = L_ij − (1/3)L_kk δij (formule 3D gardée, pas de zz en 2D).
    let lkk3 = (lxx + lyy) / 3.0;
    let ldxx = lxx - lkk3;
    let ldyy = lyy - lkk3;
    let ljx = lbutx - ubx * tb;
    let ljy = lbuty - uby * tb;

    // Contractions de Lilly (double comptage xy, symétrique).
    let num_v = ldxx * mxx + 2.0 * lxy * mxy + ldyy * myy;
    let den_v = mxx * mxx + 2.0 * mxy * mxy + myy * myy;
    let num_k = ljx * mx + ljy * my;
    let den_k = mx * mx + my * my;

    dyn_s3[8u * k + 0u] = num_v;
    dyn_s3[8u * k + 1u] = den_v;
    dyn_s3[8u * k + 2u] = num_k;
    dyn_s3[8u * k + 3u] = den_k;
    dyn_s3[8u * k + 4u] = nu_mix_g;
    dyn_s3[8u * k + 5u] = kap_mix_g;
}

// --- dyn4 : régularisation Éq. 2.62 (box num/den), coefficient de Lilly
// (clamp [0, 0.04]), ν_t/α_t bornés (nu_sm_min + plancher TRT) → dyn_out.
@compute @workgroup_size(16, 16)
fn dyn4(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    let iy = gid.y;
    if (ix >= P.nx || iy >= P.ny) { return; }
    let k = iy * P.nx + ix;

    let num_v_b = box_s3(ix, iy, 0u);
    let den_v_b = box_s3(ix, iy, 1u);
    let num_k_b = box_s3(ix, iy, 2u);
    let den_k_b = box_s3(ix, iy, 3u);
    let cnu = lilly(num_v_b, den_v_b);
    let ck = lilly(num_k_b, den_k_b);

    // ν_t = C^ν_dyn·νsm_mix(grille) ; α_t = C^κ_dyn·κsm_mix(grille).
    // Garde-fou TRT : ν_t/α_t ≥ max(nu_sm_min·mol, marge_plancher − mol).
    let nu_mix_g = dyn_s3[8u * k + 4u];
    let kap_mix_g = dyn_s3[8u * k + 5u];
    let nt = cnu * nu_mix_g;
    let nu_t_min = max(D.nu_sm_min * D.nu_mol, D.tau_floor_margin - D.nu_mol);
    let nu_t = max(nt, nu_t_min);
    let at = ck * kap_mix_g;
    let alpha_t_min = max(D.nu_sm_min * D.alpha_mol, D.tau_floor_margin - D.alpha_mol);
    let alpha_t = max(at, alpha_t_min);

    dyn_out[4u * k + 0u] = nu_t;
    dyn_out[4u * k + 1u] = alpha_t;
    dyn_out[4u * k + 2u] = cnu;
    dyn_out[4u * k + 3u] = ck;
}

// --- collide_dyn : réplique exacte de `collide` consommant ν_t/α_t de
// dyn_out (kernel dyn4 du même pass, barrière wgpu) au lieu de
// ν_t = (Cs·Δx)²·|S| et α_t = ν_t/Pr_t. Le π:π n'est plus nécessaire.
@compute @workgroup_size(16, 16)
fn collide_dyn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    let iy = gid.y;
    if (ix >= P.nx || iy >= P.ny) { return; }
    let k = iy * P.nx + ix;
    let n = n_tot();

    let ux = stock[4u * k + 1u];
    let uy = stock[4u * k + 2u];
    let temp = stock[4u * k + 3u];

    // P1b (ρ₀ dans la flottabilité) : fy = β_eff·(T − t_ref)·(1 + rho_m1)
    // — identique en arithmétique exacte à l'ancien ρ_stock·β_eff·(T−t_ref)
    // (le terme dominant ρ₀ = 1 absorbé, seul le petit ρ−1 = Σ f̃ multiplie
    // — plus de grosse magnitude ρ dans la force). rho_m1 est la somme des
    // DÉCALAGES f̃ (calculée ci-dessous, aucun recollage W + f̃ — le gros
    // nombre ≈0.44 annulerait le signal f32). Formule STRICTEMENT identique
    // au CPU (solver.rs : beta_eff·(T−t_ref)·(1 + (rho−1))) et au kernel
    // `collide` (P.beta_eff·(temp − P.t_ref)·(1 + rho_m1)).
    let f0s = f_in[0u * n + k];
    let rho_m1 = f0s + f_in[1u * n + k] + f_in[2u * n + k] + f_in[3u * n + k]
        + f_in[4u * n + k] + f_in[5u * n + k] + f_in[6u * n + k] + f_in[7u * n + k]
        + f_in[8u * n + k];
    let rho = 1.0 + rho_m1;
    let fy_node = P.beta_eff * (temp - P.t_ref) * (1.0 + rho_m1);
    let f1s = f_in[1u * n + k];
    let f2s = f_in[2u * n + k];
    let f3s = f_in[3u * n + k];
    let f4s = f_in[4u * n + k];
    let f5s = f_in[5u * n + k];
    let f6s = f_in[6u * n + k];
    let f7s = f_in[7u * n + k];
    let f8s = f_in[8u * n + k];
    let g0 = g_in[0u * n + k];
    let g1 = g_in[1u * n + k];
    let g2 = g_in[2u * n + k];
    let g3 = g_in[3u * n + k];
    let g4 = g_in[4u * n + k];
    let g5 = g_in[5u * n + k];
    let g6 = g_in[6u * n + k];
    let g7 = g_in[7u * n + k];
    let g8 = g_in[8u * n + k];

    let u2 = ux * ux + uy * uy;
    let c1 = 1.0 - u2 / (2.0 * CS2);
    let f_eq0 = feq_off_c(W0, 0.0, 0.0, rho_m1, rho, ux, uy);
    let f_eq1 = feq_off_c(W1, 1.0, 0.0, rho_m1, rho, ux, uy);
    let f_eq2 = feq_off_c(W1, 0.0, 1.0, rho_m1, rho, ux, uy);
    let f_eq3 = feq_off_c(W1, -1.0, 0.0, rho_m1, rho, ux, uy);
    let f_eq4 = feq_off_c(W1, 0.0, -1.0, rho_m1, rho, ux, uy);
    let f_eq5 = feq_off_c(W5, 1.0, 1.0, rho_m1, rho, ux, uy);
    let f_eq6 = feq_off_c(W5, -1.0, 1.0, rho_m1, rho, ux, uy);
    let f_eq7 = feq_off_c(W5, -1.0, -1.0, rho_m1, rho, ux, uy);
    let f_eq8 = feq_off_c(W5, 1.0, -1.0, rho_m1, rho, ux, uy);
    let g_eq0 = geq_c(W0, 0.0, 0.0, temp, ux, uy);
    let g_eq1 = geq_c(W1, 1.0, 0.0, temp, ux, uy);
    let g_eq2 = geq_c(W1, 0.0, 1.0, temp, ux, uy);
    let g_eq3 = geq_c(W1, -1.0, 0.0, temp, ux, uy);
    let g_eq4 = geq_c(W1, 0.0, -1.0, temp, ux, uy);
    let g_eq5 = geq_c(W5, 1.0, 1.0, temp, ux, uy);
    let g_eq6 = geq_c(W5, -1.0, 1.0, temp, ux, uy);
    let g_eq7 = geq_c(W5, -1.0, -1.0, temp, ux, uy);
    let g_eq8 = geq_c(W5, 1.0, -1.0, temp, ux, uy);

    let d0 = f0s - f_eq0;
    let d1 = f1s - f_eq1;
    let d2 = f2s - f_eq2;
    let d3 = f3s - f_eq3;
    let d4 = f4s - f_eq4;
    let d5 = f5s - f_eq5;
    let d6 = f6s - f_eq6;
    let d7 = f7s - f_eq7;
    let d8 = f8s - f_eq8;

    // Étape D : ν_t/α_t dynamiques de dyn_out (passe 3 du même compute pass).
    let nu_t_node = dyn_out[4u * k + 0u];
    let alpha_t = dyn_out[4u * k + 1u];
    nu_t[k] = nu_t_node;

    let tau_plus = 0.5 + 3.0 * (P.nu_eff + nu_t_node);
    let tau_minus = 0.5 + P.lambda_trt / (tau_plus - 0.5);
    let tau_t_plus = 0.5 + 3.0 * (P.alpha_eff + alpha_t);
    let a_f = 0.5 * (1.0 / tau_plus + 1.0 / tau_minus);
    let b_f = 0.5 * (1.0 / tau_plus - 1.0 / tau_minus);
    let a_g = 1.0 / tau_t_plus; // τ_t− = τ_t+ ⇒ b_g = 0

    // Collision TRT en DÉCALÉ (P1a, identique à collide : trt_f retourne
    // f̃', l'offset w_i se simplifie exactement — plus aucun − W_i final).
    f_col[0u * n + k] = trt_f(W0, 0.0, 0.0, f_in[0u * n + k], d0, d0, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[3u * n + k] = trt_f(W1, -1.0, 0.0, f_in[3u * n + k], d3, d1, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[1u * n + k] = trt_f(W1, 1.0, 0.0, f_in[1u * n + k], d1, d3, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[4u * n + k] = trt_f(W1, 0.0, -1.0, f_in[4u * n + k], d4, d2, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[2u * n + k] = trt_f(W1, 0.0, 1.0, f_in[2u * n + k], d2, d4, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[7u * n + k] = trt_f(W5, -1.0, -1.0, f_in[7u * n + k], d7, d5, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[5u * n + k] = trt_f(W5, 1.0, 1.0, f_in[5u * n + k], d5, d7, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[8u * n + k] = trt_f(W5, 1.0, -1.0, f_in[8u * n + k], d8, d6, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[6u * n + k] = trt_f(W5, -1.0, 1.0, f_in[6u * n + k], d6, d8, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus);

    g_col[0u * n + k] = g0 - a_g * (g0 - g_eq0);
    g_col[1u * n + k] = g1 - a_g * (g1 - g_eq1);
    g_col[2u * n + k] = g2 - a_g * (g2 - g_eq2);
    g_col[3u * n + k] = g3 - a_g * (g3 - g_eq3);
    g_col[4u * n + k] = g4 - a_g * (g4 - g_eq4);
    g_col[5u * n + k] = g5 - a_g * (g5 - g_eq5);
    g_col[6u * n + k] = g6 - a_g * (g6 - g_eq6);
    g_col[7u * n + k] = g7 - a_g * (g7 - g_eq7);
    g_col[8u * n + k] = g8 - a_g * (g8 - g_eq8);
}
"#;

impl GpuSolver {
    /// Initialise le backend GPU depuis le solveur CPU (champs identiques :
    /// nu, alpha, beta_eff, cs, pr_t — mêmes formules de `CavitySolver::new`).
    /// Les populations partent des buffers du CPU (converties f64 → f32 avec
    /// décalage f_i − w_i), le macroscopique part des champs du CPU.
    /// La rampe de spin-up est recalculée côté CPU à chaque pas (uniform
    /// buffer réécrit), mêmes expressions que `collide` (solver.rs).
    pub fn from_solver(sim: &CavitySolver) -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| {
            format!("wgpu : aucun adaptateur GPU disponible ({e}) — backend GPU indisponible, utilisez --backend cpu")
        })?;
        let info = adapter.get_info();
        let adapter_info = format!("{} ({:?})", info.name, info.backend);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("cavity-les-gpu"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| format!("wgpu : échec de création du device ({e})"))?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cavity_kernel_soa_2dispatch"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let mk_pipe = |label: &'static str, entry: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None, // layout déduit du shader
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipe_collide = mk_pipe("cavity_collide", "collide");
        let pipe_stream_macro = mk_pipe("cavity_stream_macro", "stream_macro");

        // Étape D (porte G4) : kernels du dynamic Smagorinsky (sim.dynamic_smag).
        let dynamic = sim.dynamic_smag;
        let (pipe_dyn1, pipe_dyn2, pipe_dyn3, pipe_dyn4, pipe_collide_dyn) = if dynamic {
            (
                Some(mk_pipe("cavity_dyn1", "dyn1")),
                Some(mk_pipe("cavity_dyn2", "dyn2")),
                Some(mk_pipe("cavity_dyn3", "dyn3")),
                Some(mk_pipe("cavity_dyn4", "dyn4")),
                Some(mk_pipe("cavity_collide_dyn", "collide_dyn")),
            )
        } else {
            (None, None, None, None, None)
        };

        let nx = sim.nx();
        let ny = sim.ny();
        let n = nx * ny;

        // Buffers f/g : populations DÉCALÉES (f_i − w_i·ρ₀, g_i tel quel car
        // T̄ = 0). Conversion f64 → f32 à l'interface (parade FluidX3D).
        // Layout SoA [direction][nœud] : f[i·n + k] (Étape C, porte G3).
        let mut f_off = vec![0.0f32; 9 * n];
        let mut g_off = vec![0.0f32; 9 * n];
        for k in 0..n {
            for i in 0..9 {
                f_off[i * n + k] = (sim.lbm.f[k][i] - Self::w_weight(i)) as f32;
                g_off[i * n + k] = sim.lbm.g[k][i] as f32;
            }
        }
        let mk_pop = |label: &'static str, f: bool| {
            let data: Vec<f32> = if f {
                f_off.clone()
            } else {
                g_off.clone()
            };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let f_a = mk_pop("f_a", true);
        let g_a = mk_pop("g_a", false);
        let f_b = mk_pop("f_b", true);
        let g_b = mk_pop("g_b", false);
        let f_col = Self::empty_storage(&device, 9 * n, "f_col");
        let g_col = Self::empty_storage(&device, 9 * n, "g_col");

        // ν_t : copie de l'état CPU (sanité des lectures avant le 1er pas).
        let nu_t_init: Vec<f32> = sim.nu_t.iter().map(|&v| v as f32).collect();
        let nu_t_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nu_t"),
            contents: bytemuck::cast_slice(&nu_t_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // Étape D (G4) : buffers de travail du dynamic Smagorinsky + uniform
        // DynParams + staging de lecture des diagnostics (C^ν, C^κ).
        let (dyn_params_buf, dyn_s1, dyn_s2, dyn_s3, dyn_out, dyn_read_buf) = if dynamic {
            // Mêmes quantités que `compute_dynamic_sgs` du CPU : ν/α MOLÉCULAIRES
            // cibles (pas les valeurs rampées — rampe côté collide seulement),
            // largeur box = round(max(η,1)), dtest = η·Δx, marge plancher TRT.
            let eta = sim.eta_test.max(1.0);
            let dparams = GpuDynParams {
                alpha_exp: sim.smag_alpha as f32,
                eta_test: eta as f32,
                dtest: eta as f32,
                delta_t: DELTA_T as f32,
                nu_mol: sim.nu as f32,
                alpha_mol: sim.alpha as f32,
                nu_sm_min: sim.nu_sm_min as f32,
                tau_floor_margin: ((TAU_TRT_FLOOR - 0.5) / 3.0) as f32,
                box_width: ((eta.round() as usize).max(1)) as u32,
                c_dyn_max: C_DYN_MAX as f32,
                _pad: [0; 6],
            };
            let dpb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dyn_params"),
                contents: bytemuck::bytes_of(&dparams),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let mk_scratch = |label: &str, n_f32: usize| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: (n_f32 * std::mem::size_of::<f32>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            };
            let s1 = mk_scratch("dyn_s1", 8 * n);
            let s2 = mk_scratch("dyn_s2", 16 * n);
            let s3 = mk_scratch("dyn_s3", 8 * n);
            let dout = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dyn_out"),
                size: (4 * n * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let drb = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dyn_read_staging"),
                size: (4 * n * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (Some(dpb), Some(s1), Some(s2), Some(s3), Some(dout), Some(drb))
        } else {
            (None, None, None, None, None, None)
        };

        // Stock macroscopique : champs du solveur CPU (temps t initial,
        // correction de vitesse Guo et no-slip déjà appliqués par le CPU).
        let mut stock_init = vec![0.0f32; 4 * n];
        for k in 0..n {
            stock_init[4 * k] = sim.lbm.rho[k] as f32;
            stock_init[4 * k + 1] = sim.lbm.ux[k] as f32;
            stock_init[4 * k + 2] = sim.lbm.uy[k] as f32;
            stock_init[4 * k + 3] = sim.lbm.temperature[k] as f32;
        }
        let stock_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stock"),
            contents: bytemuck::cast_slice(&stock_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        // Staging de lecture : stock (4n) + nu_t (n) f32.
        let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read_staging"),
            size: (5 * n * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = Self::params_from(sim, 0);
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Bind groups pré-créés (les buffers ne changent jamais, seul le
        // CONTENU de params_buf est réécrit) — variantes ping-pong A/B.
        let uniform = params_buf.as_entire_binding();
        let mk_bg = |label: &'static str, layout: &wgpu::BindGroupLayout, entries: &[wgpu::BindGroupEntry]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries,
            })
        };
        // Helper avec lifetimes explicites (la closure équivalente ne
        // compile pas : les lifetimes de BindGroupEntry sont liées à l'entrée).
        fn ent<'a>(binding: u32, resource: wgpu::BindingResource<'a>) -> wgpu::BindGroupEntry<'a> {
            wgpu::BindGroupEntry { binding, resource }
        }
        let (fa, ga, fb, gb) = (
            f_a.as_entire_binding(),
            g_a.as_entire_binding(),
            f_b.as_entire_binding(),
            g_b.as_entire_binding(),
        );
        let fcol = f_col.as_entire_binding();
        let gcol = g_col.as_entire_binding();
        let ntb = nu_t_buf.as_entire_binding();
        let stockb = stock_buf.as_entire_binding();

        let bg_collide_a = mk_bg(
            "bg_collide_a",
            &pipe_collide.get_bind_group_layout(0),
            &[
                ent(0, uniform.clone()),
                ent(1, fa.clone()),
                ent(2, ga.clone()),
                ent(5, stockb.clone()),
                ent(6, fcol.clone()),
                ent(7, gcol.clone()),
                ent(9, ntb.clone()),
            ],
        );
        let bg_collide_b = mk_bg(
            "bg_collide_b",
            &pipe_collide.get_bind_group_layout(0),
            &[
                ent(0, uniform.clone()),
                ent(1, fb.clone()),
                ent(2, gb.clone()),
                ent(5, stockb.clone()),
                ent(6, fcol.clone()),
                ent(7, gcol.clone()),
                ent(9, ntb.clone()),
            ],
        );
        let bg_stream_macro_a = mk_bg(
            "bg_stream_macro_a",
            &pipe_stream_macro.get_bind_group_layout(0),
            &[
                ent(0, uniform.clone()),
                ent(3, fb.clone()),
                ent(4, gb.clone()),
                ent(5, stockb.clone()),
                ent(6, fcol.clone()),
                ent(7, gcol.clone()),
            ],
        );
        let bg_stream_macro_b = mk_bg(
            "bg_stream_macro_b",
            &pipe_stream_macro.get_bind_group_layout(0),
            &[
                ent(0, uniform.clone()),
                ent(3, fa.clone()),
                ent(4, ga.clone()),
                ent(5, stockb.clone()),
                ent(6, fcol.clone()),
                ent(7, gcol.clone()),
            ],
        );

        // Étape D (G4) : bind groups des kernels dynamiques + collision
        // dynamique (variantes ping-pong A/B — collide_dyn lit f_in A/B).
        let (bg_dyn1, bg_dyn2_a, bg_dyn2_b, bg_dyn3, bg_dyn4, bg_collide_dyn_a, bg_collide_dyn_b) =
            if dynamic {
                let dpb = dyn_params_buf.as_ref().expect("dyn_params_buf");
                let s1b = dyn_s1.as_ref().expect("dyn_s1");
                let s2b = dyn_s2.as_ref().expect("dyn_s2");
                let s3b = dyn_s3.as_ref().expect("dyn_s3");
                let dob = dyn_out.as_ref().expect("dyn_out");
                let p1 = pipe_dyn1.as_ref().expect("pipe_dyn1");
                let p2 = pipe_dyn2.as_ref().expect("pipe_dyn2");
                let p3 = pipe_dyn3.as_ref().expect("pipe_dyn3");
                let p4 = pipe_dyn4.as_ref().expect("pipe_dyn4");
                let pcd = pipe_collide_dyn.as_ref().expect("pipe_collide_dyn");
                // dyn1 lit stock, écrit dyn_s1 (uniforms + 2 buffers).
                let bg1 = mk_bg(
                    "bg_dyn1",
                    &p1.get_bind_group_layout(0),
                    &[
                        ent(0, uniform.clone()),
                        ent(5, stockb.clone()),
                        ent(10, dpb.as_entire_binding()),
                        ent(11, s1b.as_entire_binding()),
                    ],
                );
                // dyn2 lit stock + f_in (A ou B) + dyn_s1, écrit dyn_s2.
                let bg2a = mk_bg(
                    "bg_dyn2_a",
                    &p2.get_bind_group_layout(0),
                    &[
                        ent(0, uniform.clone()),
                        ent(1, f_a.as_entire_binding()),
                        ent(5, stockb.clone()),
                        ent(11, s1b.as_entire_binding()),
                        ent(12, s2b.as_entire_binding()),
                    ],
                );
                let bg2b = mk_bg(
                    "bg_dyn2_b",
                    &p2.get_bind_group_layout(0),
                    &[
                        ent(0, uniform.clone()),
                        ent(1, f_b.as_entire_binding()),
                        ent(5, stockb.clone()),
                        ent(11, s1b.as_entire_binding()),
                        ent(12, s2b.as_entire_binding()),
                    ],
                );
                let bg3 = mk_bg(
                    "bg_dyn3",
                    &p3.get_bind_group_layout(0),
                    &[
                        ent(0, uniform.clone()),
                        ent(10, dpb.as_entire_binding()),
                        ent(11, s1b.as_entire_binding()),
                        ent(12, s2b.as_entire_binding()),
                        ent(13, s3b.as_entire_binding()),
                    ],
                );
                let bg4 = mk_bg(
                    "bg_dyn4",
                    &p4.get_bind_group_layout(0),
                    &[
                        ent(0, uniform.clone()),
                        ent(10, dpb.as_entire_binding()),
                        ent(13, s3b.as_entire_binding()),
                        ent(14, dob.as_entire_binding()),
                    ],
                );
                // collide_dyn : bind group du collide existant + dyn_out (15).
                let bcd_a = mk_bg(
                    "bg_collide_dyn_a",
                    &pcd.get_bind_group_layout(0),
                    &[
                        ent(0, uniform.clone()),
                        ent(1, fa.clone()),
                        ent(2, ga.clone()),
                        ent(5, stockb.clone()),
                        ent(6, fcol.clone()),
                        ent(7, gcol.clone()),
                        ent(9, ntb.clone()),
                        ent(14, dob.as_entire_binding()),
                    ],
                );
                let bcd_b = mk_bg(
                    "bg_collide_dyn_b",
                    &pcd.get_bind_group_layout(0),
                    &[
                        ent(0, uniform.clone()),
                        ent(1, fb.clone()),
                        ent(2, gb.clone()),
                        ent(5, stockb.clone()),
                        ent(6, fcol.clone()),
                        ent(7, gcol.clone()),
                        ent(9, ntb.clone()),
                        ent(14, dob.as_entire_binding()),
                    ],
                );
                (Some(bg1), Some(bg2a), Some(bg2b), Some(bg3), Some(bg4), Some(bcd_a), Some(bcd_b))
            } else {
                (None, None, None, None, None, None, None)
            };

        Ok(Self {
            device,
            queue,
            pipe_collide,
            pipe_stream_macro,
            pipe_dyn1,
            pipe_dyn2,
            pipe_dyn3,
            pipe_dyn4,
            pipe_collide_dyn,
            params_buf,
            dyn_params_buf,
            f_a,
            g_a,
            f_b,
            g_b,
            f_col,
            g_col,
            nu_t_buf,
            dyn_s1,
            dyn_s2,
            dyn_s3,
            dyn_out,
            stock_buf,
            read_buf,
            dyn_read_buf,
            bg_collide_a,
            bg_collide_b,
            bg_stream_macro_a,
            bg_stream_macro_b,
            bg_dyn1,
            bg_dyn2_a,
            bg_dyn2_b,
            bg_dyn3,
            bg_dyn4,
            bg_collide_dyn_a,
            bg_collide_dyn_b,
            nx,
            ny,
            time_step: 0,
            even: true,
            batch_enc: None,
            batch_len: 0,
            batch_params: params,
            dynamic,
            adapter_info,
        })
    }

    /// Nom de l'adaptateur GPU (informatif, affiché au démarrage).
    pub fn adapter_name(&self) -> String {
        self.adapter_info.clone()
    }

    fn empty_storage(device: &wgpu::Device, n_f32: usize, label: &str) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (n_f32 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Poids w_i côté CPU (pour le décalage f_i − w_i·ρ₀).
    fn w_weight(i: usize) -> f64 {
        const W: [f64; 9] = [
            4.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
        ];
        W[i]
    }

    /// Paramètres du pas (rampes spin-up incluses, mêmes expressions que
    /// `collide`/`ramp_nu_alpha` du CPU — bitwise côté f64, converties f32).
    fn params_from(sim: &CavitySolver, step: usize) -> GpuParams {
        // Rampe spin-up beta (linéaire) — mêmes expressions que solver.rs.
        let beta_eff = if step < sim.spin_up_steps {
            sim.beta_eff * (step as f64 / sim.spin_up_steps as f64)
        } else {
            sim.beta_eff
        };
        // Rampe viscosité/diffusivité (exponentielle) — ramp_nu_alpha.
        let (nu_eff, alpha_eff) = if step < sim.spin_up_steps {
            let x = step as f64 / sim.spin_up_steps as f64;
            let nu_start = sim.nu * sim.nu_start_mult;
            let alpha_start = sim.alpha * sim.nu_start_mult;
            (
                nu_start * (sim.nu / nu_start).powf(x),
                alpha_start * (sim.alpha / alpha_start).powf(x),
            )
        } else {
            (sim.nu, sim.alpha)
        };
        // τ moléculaire pour |S| (plancher ν, même marge que le CPU).
        let tau_mol = 0.5 + 3.0 * nu_eff.max(0.002 / 3.0);

        GpuParams {
            tau_mol: tau_mol as f32,
            nu_eff: nu_eff as f32,
            alpha_eff: alpha_eff as f32,
            beta_eff: beta_eff as f32,
            t_ref: sim.lbm.t_ref as f32,
            cs2: (1.0f64 / 3.0) as f32,
            lambda_trt: (3.0f64 / 16.0) as f32,
            cs_smag: sim.cs as f32,
            pr_t: sim.pr_t as f32,
            t_hot: T_HOT as f32,
            t_cold: T_COLD as f32,
            nx: sim.nx() as u32,
            ny: sim.ny() as u32,
            _pad: 0,
        }
    }

    /// Un pas de temps GPU : 2 dispatches ordonnés (collide → stream_macro)
    /// dans un seul compute pass — chaque dispatch est un usage-scope wgpu,
    /// les barrières storage sont garanties entre dispatches, aucune race.
    /// Le macroscopique est réécrit dans `stock` (lu au sampling via
    /// `sync_into`/`read_fields`).
    ///
    /// Étape C (batching) : hors spin-up (params constants bit-à-bit), les
    /// pas s'enchaînent dans un SEUL command encoder, flushé tous les
    /// BATCH_STEPS pas ou avant tout readback — amortit l'overhead CPU de
    /// création/soumission d'encoder. Pendant le spin-up (params varient à
    /// chaque pas), un encoder par pas (sémantique identique à l'Étape A).
    pub fn step(&mut self, sim: &CavitySolver) {
        let params = Self::params_from(sim, self.time_step);
        let batchable = self.batch_len > 0 && params == self.batch_params;

        // Flush du batch en cours si le pas courant n'est pas batchable
        // (changement de params : spin-up) ou batch plein.
        if self.batch_len > 0 && !batchable {
            self.flush();
        }

        if self.batch_len == 0 {
            // Nouvel encoder : écrit les params du batch (constants sur
            // toute sa durée — vérifié bit-à-bit à chaque pas).
            self.queue
                .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
            self.batch_enc = Some(self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("step_batch") },
            ));
            self.batch_params = params;
        }

        let encoder = self
            .batch_enc
            .as_mut()
            .expect("encoder batché présent après flush/création");

        // Ping-pong : pas pair lit A écrit B, pas impair inverse.
        let (bg_collide, bg_stream_macro, bg_collide_dyn) = if self.even {
            (
                &self.bg_collide_a,
                &self.bg_stream_macro_a,
                &self.bg_collide_dyn_a,
            )
        } else {
            (
                &self.bg_collide_b,
                &self.bg_stream_macro_b,
                &self.bg_collide_dyn_b,
            )
        };
        self.even = !self.even;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("step_pass"),
                timestamp_writes: None,
            });
            let wg_x = (self.nx as u32).div_ceil(16);
            let wg_y = (self.ny as u32).div_ceil(16);
            // Étape D (G4) : le dynamic Smagorinsky précède la collision —
            // dyn1 → dyn2 (f_in A/B du pas courant) → dyn3 → dyn4 →
            // collide_dyn, tous dans le MÊME compute pass (les barrières
            // wgpu entre dispatches garantissent l'ordre ; ν_t/α_t restent
            // sur GPU, aucun readback). La variante dyn2 suit le ping-pong
            // de collide_dyn (même buffer f_in lu).
            if self.dynamic {
                let bg_collide_dyn = bg_collide_dyn.as_ref().expect("bg_collide_dyn");
                let bg_dyn2 = if std::ptr::eq(
                    bg_collide_dyn,
                    self.bg_collide_dyn_a.as_ref().expect("bg_collide_dyn_a"),
                ) {
                    self.bg_dyn2_a.as_ref().expect("bg_dyn2_a")
                } else {
                    self.bg_dyn2_b.as_ref().expect("bg_dyn2_b")
                };
                let bg_dyn1 = self.bg_dyn1.as_ref().expect("bg_dyn1");
                let bg_dyn3 = self.bg_dyn3.as_ref().expect("bg_dyn3");
                let bg_dyn4 = self.bg_dyn4.as_ref().expect("bg_dyn4");
                let p_dyn1 = self.pipe_dyn1.as_ref().expect("pipe_dyn1");
                let p_dyn2 = self.pipe_dyn2.as_ref().expect("pipe_dyn2");
                let p_dyn3 = self.pipe_dyn3.as_ref().expect("pipe_dyn3");
                let p_dyn4 = self.pipe_dyn4.as_ref().expect("pipe_dyn4");
                let p_collide_dyn = self.pipe_collide_dyn.as_ref().expect("pipe_collide_dyn");
                pass.set_pipeline(p_dyn1);
                pass.set_bind_group(0, bg_dyn1, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
                pass.set_pipeline(p_dyn2);
                pass.set_bind_group(0, bg_dyn2, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
                pass.set_pipeline(p_dyn3);
                pass.set_bind_group(0, bg_dyn3, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
                pass.set_pipeline(p_dyn4);
                pass.set_bind_group(0, bg_dyn4, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
                pass.set_pipeline(p_collide_dyn);
                pass.set_bind_group(0, bg_collide_dyn, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            } else {
                pass.set_pipeline(&self.pipe_collide);
                pass.set_bind_group(0, bg_collide, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            pass.set_pipeline(&self.pipe_stream_macro);
            pass.set_bind_group(0, bg_stream_macro, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        self.time_step += 1;
        self.batch_len += 1;
        if self.batch_len >= BATCH_STEPS {
            self.flush();
        }
    }

    /// Soumet l'encoder batché en attente (appelé avant tout readback et
    /// tous les BATCH_STEPS pas).
    fn flush(&mut self) {
        if let Some(enc) = self.batch_enc.take() {
            self.queue.submit(Some(enc.finish()));
        }
        self.batch_len = 0;
    }

    /// Lecture brute du staging (stock 4n + nu_t n f32) — synchro GPU→CPU.
    /// Précondition : l'appelant a flushé l'encoder batché (sync_into /
    /// read_fields / total_mass appellent `flush()` avant `read_raw`).
    fn read_raw(&self) -> Result<Vec<f32>, String> {
        let n = self.nx * self.ny;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("read") });
        encoder.copy_buffer_to_buffer(
            &self.stock_buf,
            0,
            &self.read_buf,
            0,
            (4 * n * std::mem::size_of::<f32>()) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.nu_t_buf,
            0,
            &self.read_buf,
            (4 * n * std::mem::size_of::<f32>()) as u64,
            (n * std::mem::size_of::<f32>()) as u64,
        );
        self.queue.submit(Some(encoder.finish()));
        let slice = self.read_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // Poll bloquant : attend la complétion du map_async (wgpu-core/Metal).
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| format!("wgpu : timeout/erreur du poll GPU ({e})"))?;
        rx.recv()
            .map_err(|e| format!("wgpu : canal map_async interrompu ({e})"))?
            .map_err(|e| format!("wgpu : échec du mapping du buffer de lecture ({e})"))?;
        let mapped = slice
            .get_mapped_range()
            .map_err(|e| format!("wgpu : échec du get_mapped_range ({e})"))?;
        let data: Vec<f32> = bytemuck::cast_slice(&mapped[..]).to_vec();
        drop(mapped);
        self.read_buf.unmap();
        Ok(data)
    }

    /// Réinjecte le macroscopique GPU (ρ, ux, uy, T) et ν_t dans les champs
    /// du solveur CPU — SANS toucher au pas de temps CPU. Permet de réutiliser
    /// `post.rs`, `TimeAverager` et les exports de snapshots inchangés.
    /// Étape D (G4) : en mode dynamic, les coefficients C^ν_dyn/C^κ_dyn de
    /// dyn_out (zw) sont réduits SÉQUENTIELLEMENT côté CPU (moyenne/max —
    /// mêmes grandeurs que les diagnostics `compute_dynamic_sgs`, pas
    /// d'atomiques flottantes) et écrits dans sim.cs_dyn_mean/cs_dyn_max/
    /// kappa_dyn_mean/kappa_dyn_max.
    pub fn sync_into(&mut self, sim: &mut CavitySolver) -> Result<(), String> {
        self.flush(); // le batch en attente doit être soumis avant le readback
        let n = self.nx * self.ny;
        let data = self.read_raw()?;
        for k in 0..n {
            sim.lbm.rho[k] = data[4 * k] as f64;
            sim.lbm.ux[k] = data[4 * k + 1] as f64;
            sim.lbm.uy[k] = data[4 * k + 2] as f64;
            sim.lbm.temperature[k] = data[4 * k + 3] as f64;
            sim.nu_t[k] = data[4 * n + k] as f64;
        }
        // Diagnostics du dynamic Smagorinsky (le pas GPU le plus récent).
        if self.dynamic {
            let (cnu_mean, cnu_max, ck_mean, ck_max) = self.read_dyn_diagnostics()?;
            sim.cs_dyn_mean = cnu_mean;
            sim.cs_dyn_max = cnu_max;
            sim.kappa_dyn_mean = ck_mean;
            sim.kappa_dyn_max = ck_max;
        }
        Ok(())
    }

    /// Lit dyn_out (ν_t, α_t, C^ν_dyn, C^κ_dyn par nœud) et réduit les
    /// coefficients dynamiques côté CPU (réduction séquentielle f64 —
    /// déterministe, pas d'atomiques flottantes côté GPU).
    fn read_dyn_diagnostics(&self) -> Result<(f64, f64, f64, f64), String> {
        let n = self.nx * self.ny;
        let dout = self.dyn_out.as_ref().expect("dyn_out");
        let drb = self.dyn_read_buf.as_ref().expect("dyn_read_buf");
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dyn_read") });
        encoder.copy_buffer_to_buffer(
            dout,
            0,
            drb,
            0,
            (4 * n * std::mem::size_of::<f32>()) as u64,
        );
        self.queue.submit(Some(encoder.finish()));
        let slice = drb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| format!("wgpu : timeout/erreur du poll GPU ({e})"))?;
        rx.recv()
            .map_err(|e| format!("wgpu : canal map_async interrompu ({e})"))?
            .map_err(|e| format!("wgpu : échec du mapping dyn_read_buf ({e})"))?;
        let mapped = slice
            .get_mapped_range()
            .map_err(|e| format!("wgpu : échec du get_mapped_range ({e})"))?;
        let data: Vec<f32> = bytemuck::cast_slice(&mapped[..]).to_vec();
        drop(mapped);
        drb.unmap();
        // Réduction séquentielle en f64 (conversions f32 → f64) — mêmes
        // grandeurs que la passe 4 de `compute_dynamic_sgs` (moyenne/max).
        let mut cnu_sum = 0.0f64;
        let mut cnu_max = 0.0f64;
        let mut ck_sum = 0.0f64;
        let mut ck_max = 0.0f64;
        for k in 0..n {
            let cnu = data[4 * k + 2] as f64;
            let ck = data[4 * k + 3] as f64;
            cnu_sum += cnu;
            if cnu > cnu_max {
                cnu_max = cnu;
            }
            ck_sum += ck;
            if ck > ck_max {
                ck_max = ck;
            }
        }
        Ok((cnu_sum / n as f64, cnu_max, ck_sum / n as f64, ck_max))
    }

    /// Lit le buffer stock (ρ, ux, uy, T) — synchro GPU→CPU. Retourne les
    /// champs en f64 (conversions f32→f64 à l'interface), layout k = iy*nx+ix.
    #[allow(dead_code)] // API publique conservée (sync_into est le chemin actif)
    pub fn read_fields(&mut self) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), String> {
        self.flush();
        let n = self.nx * self.ny;
        let data = self.read_raw()?;
        let mut rho = vec![0.0f64; n];
        let mut ux = vec![0.0f64; n];
        let mut uy = vec![0.0f64; n];
        let mut t = vec![0.0f64; n];
        for k in 0..n {
            rho[k] = data[4 * k] as f64;
            ux[k] = data[4 * k + 1] as f64;
            uy[k] = data[4 * k + 2] as f64;
            t[k] = data[4 * k + 3] as f64;
        }
        Ok((rho, ux, uy, t))
    }

    /// Masse totale Σρ (lit le stock — diagnostic de conservation).
    #[allow(dead_code)] // API publique conservée (diagnostic de conservation futur)
    pub fn total_mass(&mut self) -> Result<f64, String> {
        let (rho, _, _, _) = self.read_fields()?;
        Ok(rho.iter().sum())
    }

    /// Nombre de pas exécutés.
    #[allow(dead_code)] // API publique conservée (utilisée par les diagnostics futurs)
    pub fn time_step(&self) -> usize {
        self.time_step
    }

    /// Dimensions de la grille.
    #[allow(dead_code)] // API publique conservée (utilisée par les diagnostics futurs)
    pub fn dims(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }
}

// ---------------------------------------------------------------------------
// Tests GPU (ignorés par défaut : nécessitent un adaptateur GPU, Metal ici).
// Run : cargo test --release -- --ignored
// ---------------------------------------------------------------------------
#[cfg(test)]
mod gpu_tests {
    use super::*;

    /// Petit solveur déterministe (graine 42, mêmes unités réseau que la run
    /// de référence : ν calé → plancher 0.002).
    fn small_solver() -> CavitySolver {
        CavitySolver::new(16, 64, 1.0e6, 0.71, 0.10, 0.9, 0.02, 42, 1, 1.0, 1.0, None)
    }

    #[test]
    #[ignore = "nécessite un adaptateur GPU (Metal) — cargo test --release -- --ignored"]
    fn gpu_init_from_solver() {
        let sim = small_solver();
        let mut gpu = GpuSolver::from_solver(&sim).expect("init GPU");
        assert_eq!(gpu.dims(), (16, 64));
        assert_eq!(gpu.time_step(), 0);
        // Le stock initial doit correspondre aux champs du solveur CPU.
        let (rho, ux, uy, t) = gpu.read_fields().expect("read_fields");
        let n = 16 * 64;
        for k in 0..n {
            assert!((rho[k] - sim.lbm.rho[k]).abs() < 1e-6);
            assert!((ux[k] - sim.lbm.ux[k]).abs() < 1e-6);
            assert!((uy[k] - sim.lbm.uy[k]).abs() < 1e-6);
            assert!((t[k] - sim.lbm.temperature[k]).abs() < 1e-6);
        }
    }

    #[test]
    #[ignore = "nécessite un adaptateur GPU (Metal) — cargo test --release -- --ignored"]
    fn gpu_one_step_matches_cpu() {
        // Deux solveurs identiques (même graine) : 1 pas CPU vs 1 pas GPU.
        let mut sim_cpu = small_solver();
        let mut sim_gpu = small_solver();
        let mut gpu = GpuSolver::from_solver(&sim_gpu).expect("init GPU");

        sim_cpu.step();
        gpu.step(&sim_gpu);
        gpu.sync_into(&mut sim_gpu).expect("sync");

        let n = 16 * 64;
        for k in 0..n {
            assert!(
                (sim_gpu.lbm.rho[k] - sim_cpu.lbm.rho[k]).abs() < 1e-4,
                "rho k={k} : gpu={} cpu={}",
                sim_gpu.lbm.rho[k],
                sim_cpu.lbm.rho[k]
            );
            assert!(
                (sim_gpu.lbm.ux[k] - sim_cpu.lbm.ux[k]).abs() < 1e-4,
                "ux k={k} : gpu={} cpu={}",
                sim_gpu.lbm.ux[k],
                sim_cpu.lbm.ux[k]
            );
            assert!(
                (sim_gpu.lbm.uy[k] - sim_cpu.lbm.uy[k]).abs() < 1e-4,
                "uy k={k} : gpu={} cpu={}",
                sim_gpu.lbm.uy[k],
                sim_cpu.lbm.uy[k]
            );
            assert!(
                (sim_gpu.lbm.temperature[k] - sim_cpu.lbm.temperature[k]).abs() < 1e-4,
                "T k={k} : gpu={} cpu={}",
                sim_gpu.lbm.temperature[k],
                sim_cpu.lbm.temperature[k]
            );
            assert!(
                (sim_gpu.nu_t[k] - sim_cpu.nu_t[k]).abs() < 1e-6,
                "nu_t k={k} : gpu={} cpu={}",
                sim_gpu.nu_t[k],
                sim_cpu.nu_t[k]
            );
        }
    }

    #[test]
    #[ignore = "nécessite un adaptateur GPU (Metal) — cargo test --release -- --ignored"]
    fn gpu_mass_conserved_100_steps() {
        let mut sim = small_solver();
        let mut gpu = GpuSolver::from_solver(&sim).expect("init GPU");
        let n = 16 * 64;
        let m0 = gpu.total_mass().expect("masse initiale");
        for _ in 0..100 {
            gpu.step(&sim);
        }
        let m1 = gpu.total_mass().expect("masse finale");
        // Dérive f32 attendue ≪ 1e-3 (le CPU dérive ~1e-5 sur 100 pas).
        let derive = (m1 - m0).abs() / n as f64;
        assert!(
            derive < 1e-3,
            "dérive de masse excessive : m0={m0} m1={m1} dérive/nœud={derive:.3e}"
        );
    }

    // -----------------------------------------------------------------------
    // Étape D (porte G4) — dynamic Smagorinsky sur GPU.
    // -----------------------------------------------------------------------

    /// Solveur dynamic Smagorinsky (α = 0.5, η = 2, défauts CLI).
    fn small_solver_dyn() -> CavitySolver {
        let mut sim = CavitySolver::new(16, 64, 1.0e6, 0.71, 0.10, 0.9, 0.02, 42, 1, 1.0, 1.0, None);
        sim.set_dynamic_sgs(true, 0.5, -0.1, 2.0);
        sim
    }

    #[test]
    #[ignore = "nécessite un adaptateur GPU (Metal) — cargo test --release -- --ignored"]
    fn gpu_one_step_dynamic_smag_matches_cpu() {
        // 1 pas CPU vs 1 pas GPU en mode échelles mixtes dynamiques :
        // le kernel dyn1..dyn4 doit reproduire `compute_dynamic_sgs` +
        // `collide_node(nu_t_pre)` à la tolérance f32 près.
        let mut sim_cpu = small_solver_dyn();
        let mut sim_gpu = small_solver_dyn();
        let mut gpu = GpuSolver::from_solver(&sim_gpu).expect("init GPU");

        sim_cpu.step();
        gpu.step(&sim_gpu);
        gpu.sync_into(&mut sim_gpu).expect("sync");

        let n = 16 * 64;
        for k in 0..n {
            assert!(
                (sim_gpu.lbm.rho[k] - sim_cpu.lbm.rho[k]).abs() < 1e-4,
                "rho k={k} : gpu={} cpu={}",
                sim_gpu.lbm.rho[k],
                sim_cpu.lbm.rho[k]
            );
            assert!(
                (sim_gpu.lbm.uy[k] - sim_cpu.lbm.uy[k]).abs() < 1e-4,
                "uy k={k} : gpu={} cpu={}",
                sim_gpu.lbm.uy[k],
                sim_cpu.lbm.uy[k]
            );
            assert!(
                (sim_gpu.lbm.temperature[k] - sim_cpu.lbm.temperature[k]).abs() < 1e-4,
                "T k={k} : gpu={} cpu={}",
                sim_gpu.lbm.temperature[k],
                sim_cpu.lbm.temperature[k]
            );
            // ν_t dynamique : tolérance relative 2 % (f32, box/powf).
            let nt_gpu = sim_gpu.nu_t[k];
            let nt_cpu = sim_cpu.nu_t[k];
            assert!(
                (nt_gpu - nt_cpu).abs() <= 0.02 * nt_cpu.abs().max(1e-12) + 1e-9,
                "nu_t k={k} : gpu={nt_gpu:.6e} cpu={nt_cpu:.6e}"
            );
        }
        // Diagnostics dynamiques cohérents (même procédure de Lilly).
        assert!(
            (sim_gpu.cs_dyn_mean - sim_cpu.cs_dyn_mean).abs() < 0.02 * sim_cpu.cs_dyn_mean.abs().max(1e-12) + 1e-9,
            "cs_dyn_mean : gpu={} cpu={}",
            sim_gpu.cs_dyn_mean,
            sim_cpu.cs_dyn_mean
        );
        assert!(sim_gpu.cs_dyn_max >= 0.0 && sim_gpu.cs_dyn_max <= 0.04 + 1e-6);
        assert!(sim_gpu.kappa_dyn_max >= 0.0 && sim_gpu.kappa_dyn_max <= 0.04 + 1e-6);
    }
}
