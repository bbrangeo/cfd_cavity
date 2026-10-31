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
use crate::uniform::solver::{T_COLD, T_HOT};

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

/// Solveur GPU : enveloppe wgpu du pas de temps uniforme, buffers f32.
pub struct GpuSolver {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipe_collide: wgpu::ComputePipeline,
    pipe_stream_macro: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,

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
    /// Champs macroscopiques stock (ρ, ux, uy, T) — temps t+1 après dispatch 2.
    stock_buf: wgpu::Buffer,
    /// Staging MAP_READ (stock 4n + nu_t n f32) pour les lectures CPU.
    read_buf: wgpu::Buffer,

    /// Bind groups pré-créés (buffers fixes) — variantes ping-pong A/B.
    bg_collide_a: wgpu::BindGroup,
    bg_collide_b: wgpu::BindGroup,
    bg_stream_macro_a: wgpu::BindGroup,
    bg_stream_macro_b: wgpu::BindGroup,

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
    /// Info adaptateur (affichée au démarrage).
    adapter_info: String,
}

/// Taille du batch de pas (encoder unique) : amortit l'overhead CPU de
/// création/soumission d'encoder — flushé avant tout readback.
const BATCH_STEPS: usize = 64;
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

// Équilibre hydrodynamique feq (crate : w·ρ·[1 + eu/cs² + eu²/(2cs⁴) − u²/(2cs²)]).
fn feq_c(w: f32, cx: f32, cy: f32, rho: f32, ux: f32, uy: f32) -> f32 {
    let eu = cx * ux + cy * uy;
    let u2 = ux * ux + uy * uy;
    return w * rho * (1.0 + eu / CS2 + eu * eu / (2.0 * CS2 * CS2) - u2 / (2.0 * CS2));
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
// paire (stress, facteur 1 − 1/(2τ+)). Réplique exacte, direction par
// direction, de la boucle de collision de `collide_node` (solver.rs) :
//   S_i = w_i·[(e_i−u)/cs² + (e_i·u)·e_i/cs⁴]·F
//   f_new = (w_i + f_in) − a_f·d_i − b_f·d_opp + (1−1/(2τ−))·S⁻ + (1−1/(2τ+))·S⁺
fn trt_f(w: f32, cx: f32, cy: f32, fin: f32, d: f32, d_opp: f32, ux: f32, uy: f32,
         fy_node: f32, a_f: f32, b_f: f32, tau_minus: f32, tau_plus: f32) -> f32 {
    let ei_u = cx * ux + cy * uy;
    let s_i = w * ((cy - uy) / CS2 + ei_u * cy / (CS2 * CS2)) * fy_node;
    let s_minus = w * (cy / CS2) * fy_node;
    let s_plus = s_i - s_minus;
    return (w + fin) - a_f * d - b_f * d_opp
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
    let rho = stock[4u * k + 0u];
    let ux = stock[4u * k + 1u];
    let uy = stock[4u * k + 2u];
    let temp = stock[4u * k + 3u];

    // Force de Boussinesq : fy = ρ·β_eff·(T − t_ref), évaluée aux champs du
    // temps t — utilisée par la collision de CE pas. Le dispatch 2
    // (correction de vitesse Guo éq. (16)) la recalcule localement depuis
    // stock (même expression, même F — sémantique CPU) ; le buffer fy global
    // est supprimé (Étape C).
    let fy_node = rho * P.beta_eff * (temp - P.t_ref);

    // Populations d'entrée (reconstruction depuis le stockage décalé :
    // f_i = w_i + décalage) — stride SoA : direction i à f_in[i·n + k].
    let f0 = W0 + f_in[0u * n + k];
    let f1 = W1 + f_in[1u * n + k];
    let f2 = W1 + f_in[2u * n + k];
    let f3 = W1 + f_in[3u * n + k];
    let f4 = W1 + f_in[4u * n + k];
    let f5 = W5 + f_in[5u * n + k];
    let f6 = W5 + f_in[6u * n + k];
    let f7 = W5 + f_in[7u * n + k];
    let f8 = W5 + f_in[8u * n + k];
    let g0 = g_in[0u * n + k];
    let g1 = g_in[1u * n + k];
    let g2 = g_in[2u * n + k];
    let g3 = g_in[3u * n + k];
    let g4 = g_in[4u * n + k];
    let g5 = g_in[5u * n + k];
    let g6 = g_in[6u * n + k];
    let g7 = g_in[7u * n + k];
    let g8 = g_in[8u * n + k];

    // Équilibres (direction par direction — mêmes expressions que la boucle).
    let u2 = ux * ux + uy * uy;
    let c1 = 1.0 - u2 / (2.0 * CS2);
    let f_eq0 = feq_c(W0, 0.0, 0.0, rho, ux, uy);
    let f_eq1 = feq_c(W1, 1.0, 0.0, rho, ux, uy);
    let f_eq2 = feq_c(W1, 0.0, 1.0, rho, ux, uy);
    let f_eq3 = feq_c(W1, -1.0, 0.0, rho, ux, uy);
    let f_eq4 = feq_c(W1, 0.0, -1.0, rho, ux, uy);
    let f_eq5 = feq_c(W5, 1.0, 1.0, rho, ux, uy);
    let f_eq6 = feq_c(W5, -1.0, 1.0, rho, ux, uy);
    let f_eq7 = feq_c(W5, -1.0, -1.0, rho, ux, uy);
    let f_eq8 = feq_c(W5, 1.0, -1.0, rho, ux, uy);
    let g_eq0 = geq_c(W0, 0.0, 0.0, temp, ux, uy);
    let g_eq1 = geq_c(W1, 1.0, 0.0, temp, ux, uy);
    let g_eq2 = geq_c(W1, 0.0, 1.0, temp, ux, uy);
    let g_eq3 = geq_c(W1, -1.0, 0.0, temp, ux, uy);
    let g_eq4 = geq_c(W1, 0.0, -1.0, temp, ux, uy);
    let g_eq5 = geq_c(W5, 1.0, 1.0, temp, ux, uy);
    let g_eq6 = geq_c(W5, -1.0, 1.0, temp, ux, uy);
    let g_eq7 = geq_c(W5, -1.0, -1.0, temp, ux, uy);
    let g_eq8 = geq_c(W5, 1.0, -1.0, temp, ux, uy);

    // Moments non-équilibrés π_neq (avant relaxation) pour |S| — mêmes
    // contributions et ordre que la boucle (i = 0..9 ; les directions
    // cardinales ont cx²=1 ou cy²=1, les 4 diagonales cx²=cy²=1 ; cx·cy est
    // nul sur les cardinales, ±1 sur les diagonales).
    let d0 = f0 - f_eq0;
    let d1 = f1 - f_eq1;
    let d2 = f2 - f_eq2;
    let d3 = f3 - f_eq3;
    let d4 = f4 - f_eq4;
    let d5 = f5 - f_eq5;
    let d6 = f6 - f_eq6;
    let d7 = f7 - f_eq7;
    let d8 = f8 - f_eq8;
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
    // (opp(0)=0, opp(1)=3, opp(2)=4, opp(5)=7, opp(6)=8). Deviations
    // d_i/d_opp identiques à la boucle (Σ f' = Σ f : masse conservée).
    f_col[0u * n + k] = trt_f(W0, 0.0, 0.0, f_in[0u * n + k], d0, d0, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus) - W0;
    f_col[3u * n + k] = trt_f(W1, -1.0, 0.0, f_in[3u * n + k], d3, d1, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus) - W1;
    f_col[1u * n + k] = trt_f(W1, 1.0, 0.0, f_in[1u * n + k], d1, d3, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus) - W1;
    f_col[4u * n + k] = trt_f(W1, 0.0, -1.0, f_in[4u * n + k], d4, d2, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus) - W1;
    f_col[2u * n + k] = trt_f(W1, 0.0, 1.0, f_in[2u * n + k], d2, d4, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus) - W1;
    f_col[7u * n + k] = trt_f(W5, -1.0, -1.0, f_in[7u * n + k], d7, d5, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus) - W5;
    f_col[5u * n + k] = trt_f(W5, 1.0, 1.0, f_in[5u * n + k], d5, d7, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus) - W5;
    f_col[8u * n + k] = trt_f(W5, 1.0, -1.0, f_in[8u * n + k], d8, d6, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus) - W5;
    f_col[6u * n + k] = trt_f(W5, -1.0, 1.0, f_in[6u * n + k], d6, d8, ux, uy,
                              fy_node, a_f, b_f, tau_minus, tau_plus) - W5;

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
    // déroulé, même ordre d'accumulation que la boucle (j = 0..9).
    var rho = 0.0;
    var mx = 0.0;
    var my = 0.0;
    var temp = 0.0;
    rho = rho + (W0 + f_out[0u * n + k]);
    temp = temp + g_out[0u * n + k];
    let f1 = W1 + f_out[1u * n + k];
    let f2 = W1 + f_out[2u * n + k];
    let f3 = W1 + f_out[3u * n + k];
    let f4 = W1 + f_out[4u * n + k];
    let f5 = W5 + f_out[5u * n + k];
    let f6 = W5 + f_out[6u * n + k];
    let f7 = W5 + f_out[7u * n + k];
    let f8 = W5 + f_out[8u * n + k];
    rho = rho + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    mx = mx + f1 - f3 + f5 - f6 - f7 + f8;
    my = my + f2 - f4 + f5 + f6 - f7 - f8;
    temp = temp + g_out[1u * n + k] + g_out[2u * n + k] + g_out[3u * n + k]
        + g_out[4u * n + k] + g_out[5u * n + k] + g_out[6u * n + k]
        + g_out[7u * n + k] + g_out[8u * n + k];
    // Correction de vitesse Guo et al. 2002, éq. (16) : u_y += fy/(2ρ) —
    // fy recalculé localement depuis le stock du temps t (même expression que
    // la collision de CE pas — même F, sémantique CPU `update_macroscopic` ;
    // le buffer fy global est supprimé).
    var ux = 0.0;
    var uy = 0.0;
    let fy_node = rho * P.beta_eff * (temp - P.t_ref);
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

        Ok(Self {
            device,
            queue,
            pipe_collide,
            pipe_stream_macro,
            params_buf,
            f_a,
            g_a,
            f_b,
            g_b,
            f_col,
            g_col,
            nu_t_buf,
            stock_buf,
            read_buf,
            bg_collide_a,
            bg_collide_b,
            bg_stream_macro_a,
            bg_stream_macro_b,
            nx,
            ny,
            time_step: 0,
            even: true,
            batch_enc: None,
            batch_len: 0,
            batch_params: params,
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
        let (bg_collide, bg_stream_macro) = if self.even {
            (&self.bg_collide_a, &self.bg_stream_macro_a)
        } else {
            (&self.bg_collide_b, &self.bg_stream_macro_b)
        };
        self.even = !self.even;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("step_pass"),
                timestamp_writes: None,
            });
            let wg_x = (self.nx as u32).div_ceil(16);
            let wg_y = (self.ny as u32).div_ceil(16);
            pass.set_pipeline(&self.pipe_collide);
            pass.set_bind_group(0, bg_collide, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
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
        Ok(())
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
}
