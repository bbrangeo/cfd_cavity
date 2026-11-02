//! Backend GPU wgpu 3D (Étape V1, plan 3D Trias §3) — solveur D3Q19 TRT +
//! forcing de Guo + Smagorinsky statique, f32, GPU-first (aucun chemin CPU).
//!
//! Architecture identique au 2D validé (`uniform/gpu.rs`) : 2 dispatches
//! ordonnés par pas (un seul compute pass, deux `dispatch_workgroups` — les
//! barrières storage wgpu entre dispatches garantissent l'ordre, aucune
//! course) :
//!
//!   1. `collide`      : collision TRT (Λ = 3/16) + force de Guo (odd/even) +
//!                       Smagorinsky par nœud (ν_t, τ+, τ_t+ dépendent de ν_t)
//!                       → écrit f_col/g_col + nu_t.
//!   2. `stream_macro` : streaming pull + bounce-back (lit f_col/g_col FIGÉS
//!                       du dispatch 1 — même sémantique que le CPU, qui
//!                       clone f/g avant streaming) → écrit f_out/g_out, PUIS
//!                       CL thermiques (horizontales zero-gradient puis
//!                       verticales isothermes — les coins se terminent par
//!                       wall_iso, identité CPU) PUIS macroscopique
//!                       (ρ, u, T) avec correction de vitesse Guo éq. (16) —
//!                       fy recalculé localement depuis le stock du temps t+1
//!                       post-stream (même F que la collision) — no-slip
//!                       périmètre x/y → écrit stock.
//!
//! Conditions aux limites 3D : parois verticales x isothermes (chaude ix=0 /
//! froide ix=nx−1), parois horizontales y adiabatiques (zero-gradient), axe z
//! PÉRIODIQUE (wrap sans bounce-back — Trias Case A, L1 = 2). Les murs x/y
//! sont des nœuds fluides : bounce-back seulement quand la SOURCE du pull est
//! hors domaine.
//!
//! Layout mémoire SoA `[direction][nœud]` (f[i·n + k], k = ix + nx·(iy +
//! ny·iz)) : les 19 accès d'un kernel sont coalescés entre threads adjacents.
//! Les 19 directions sont déroulées (indices littéraux — pas d'indexation
//! dynamique de tableaux privés ; l'indexation dynamique des STORAGE buffers
//! dans `stream_dir`/`wall_grad_dir` est autorisée et coalescée).
//!
//! Précision (parade FluidX3D, P1a — port du kernel 2D validé) : les
//! populations f sont stockées ET COLLIDÉES en DÉCALÉ — f̃_i = f_i − w_i·ρ₀
//! (ρ₀ = 1) — la collision et le macroscopique ne travaillent QUE sur les
//! petits nombres : ρ−1 = Σ f̃_i (densité comme écart à 1), feq_i − w_i =
//! w_i·[(ρ−1) + ρ·A] (A = eu/cs² + eu²/(2cs⁴) − u²/(2cs²)), d_i = f̃_i −
//! (feq_i − w_i), f̃_i' = f̃_i − a·d_i − b·d_opp + S_i (l'offset w_i se
//! simplifie EXACTEMENT — plus aucun W + f̃ ni w·ρ·(1+A) ~0.44 dans le calcul).
//! Le macroscopique accumule les DÉCALAGES : mx/my/mz = Σ c_i·f̃_i EXACT
//! (Σw_i·c_i = 0 par symétrie D3Q19 — poids opposés égaux, table vérifiée du
//! crate). fy garde le ρ pleine magnitude (flottabilité inchangée — le
//! passage à ρ₀ dans la force est P1b, autre lane). g est stocké tel quel
//! (T̄ = 0 ⇒ décalage g_i − w_i^t·T̄ = g_i).
//!
//! Hors spin-up (params constants bit-à-bit), les pas s'enchaînent dans un
//! SEUL command encoder (batch de BATCH_STEPS pas, flushé avant tout
//! readback) pour amortir l'overhead CPU de soumission.

use bytemuck::Pod;
use bytemuck::Zeroable;
use oxiphysics_lbm::lattice::{feq_d3q19, D3Q19_VELOCITIES, D3Q19_WEIGHTS};
use wgpu::util::DeviceExt;

use crate::uniform3d::solver::CavitySolver3D;
use crate::uniform3d::solver::{T_COLD, T_HOT};

/// Paramètres uniformes du pas de temps (passés au shader via uniform buffer).
/// Convertis en f32 à l'interface CPU (WGSL n'a pas de f64 portable).
/// Taille : 11 f32 + 3 u32 + pad [u32;2] = 64 octets.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq)]
struct GpuParams3D {
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
    /// Nœuds en z (périodique).
    nz: u32,
    /// Padding (taille 64 octets, alignement uniform WGSL).
    _pad: [u32; 2],
}

/// Solveur GPU 3D : enveloppe wgpu du pas de temps D3Q19, buffers f32.
pub struct GpuSolver3D {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipe_collide: wgpu::ComputePipeline,
    pipe_stream_macro: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,

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
    /// Viscosité turbulente par nœud (diagnostic).
    nu_t_buf: wgpu::Buffer,
    /// Champs macroscopiques stock (ρ, ux, uy, uz, T) — temps t+1 après
    /// dispatch 2 (stride 5 [nœud][5]).
    stock_buf: wgpu::Buffer,
    /// Staging MAP_READ (stock 5n + nu_t n f32) pour les lectures CPU.
    read_buf: wgpu::Buffer,

    /// Bind groups pré-créés (buffers fixes) — variantes ping-pong A/B.
    bg_collide_a: wgpu::BindGroup,
    bg_collide_b: wgpu::BindGroup,
    bg_stream_macro_a: wgpu::BindGroup,
    bg_stream_macro_b: wgpu::BindGroup,

    nx: usize,
    ny: usize,
    nz: usize,
    /// Pas de temps courant (rampe spin-up, même sémantique que le CPU).
    time_step: usize,
    /// true = le pas suivant lit A et écrit B (ping-pong).
    even: bool,
    /// Encoder batché : hors spin-up (params constants bit-à-bit), les pas
    /// s'enchaînent dans un SEUL command encoder — flushé avant tout readback
    /// et tous BATCH_STEPS.
    batch_enc: Option<wgpu::CommandEncoder>,
    /// Nombre de pas en attente dans l'encoder batché.
    batch_len: usize,
    /// Params du dernier pas du batch (re-vérifiés à chaque pas : le batching
    /// n'est valable que si params reste identique bit-à-bit).
    batch_params: GpuParams3D,
    /// Info adaptateur (affichée au démarrage).
    adapter_info: String,
}

/// Taille du batch de pas (encoder unique) : amortit l'overhead CPU de
/// création/soumission d'encoder — flushé avant tout readback.
const BATCH_STEPS: usize = 64;

const WGSL: &str = r#"
// Paramètres uniformes du pas (uniform buffer, f32) — 64 octets.
struct Params {
    tau_mol: f32,     // 0.5 + 3·max(ν_eff, 0.002/3) — pour |S| (Smagorinsky)
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
    nz: u32,
    pad0: u32,
    pad1: u32,
};

// Layout mémoire SoA [direction][nœud] : f[i·n + k] avec n = nx·ny·nz et
// k = ix + nx·(iy + ny·iz). Les 19 accès d'un kernel sont coalescés entre
// threads adjacents. stock est [nœud][5] (ρ, ux, uy, uz, T lus ensemble).

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> f_in: array<f32>;
@group(0) @binding(2) var<storage, read> g_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> f_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> g_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> stock: array<f32>;
@group(0) @binding(6) var<storage, read_write> f_col: array<f32>;
@group(0) @binding(7) var<storage, read_write> g_col: array<f32>;
@group(0) @binding(8) var<storage, read_write> nu_t: array<f32>;

// D3Q19 (crate oxiphysics-lbm) : 0 = repos ; 1-6 = ±x, ±y, ±z (1=E, 2=W,
// 3=N, 4=S, 5=T(z+), 6=B(z−)) ; 7-10 diag xy (7=[1,1,0], 8=[−1,1,0],
// 9=[1,−1,0], 10=[−1,−1,0]) ; 11-14 diag xz (11=[1,0,1], 12=[−1,0,1],
// 13=[1,0,−1], 14=[−1,0,−1]) ; 15-18 diag yz (15=[0,1,1], 16=[0,−1,1],
// 17=[0,1,−1], 18=[0,−1,−1]). Paires TRT : (0), (1,2), (3,4), (5,6), (7,10),
// (8,9), (11,14), (12,13), (15,18), (16,17).
// Les 19 directions sont DÉROULÉES (indices littéraux — pas de tableaux
// privés indexés dynamiquement) ; les poids et vecteurs sont des constantes.
const W0: f32 = 1.0 / 3.0;  // repos
const W1: f32 = 1.0 / 18.0; // axiaux (1-6)
const W5: f32 = 1.0 / 36.0; // diagonaux (7-18)
const CS2: f32 = 1.0 / 3.0;

// Nombre total de nœuds (stride SoA des directions).
fn n_tot() -> u32 {
    return P.nx * P.ny * P.nz;
}

// Équilibre hydrodynamique en VARIABLE DÉCALÉE (P1a, port du 2D) : feq_i − w_i =
// w_i·[(ρ−1) + ρ·A], A = eu/cs² + eu²/(2cs⁴) − u²/(2cs²). Identité exacte
// (feq_i = w_i·ρ·[1+A] = w_i + w_i·(ρ−1) + w_i·ρ·A, Σw_i = 1) évaluée
// UNIQUEMENT sur les petits nombres : ρ−1 = Σ f̃_i ~ 1e-6 et ρ·A ~ 1e-4,
// jamais le ~0.44 de w_i·ρ. rho_m1 = Σ f̃_i (somme des décalages, sans le 1),
// rho = 1 + ρ−1 (pleine magnitude, uniquement en facteur de ρ·A).
fn feq_off_c(w: f32, cx: f32, cy: f32, cz: f32, rho_m1: f32, rho: f32, ux: f32, uy: f32, uz: f32) -> f32 {
    let eu = cx * ux + cy * uy + cz * uz;
    let u2 = ux * ux + uy * uy + uz * uz;
    let a_term = eu / CS2 + eu * eu / (2.0 * CS2 * CS2) - u2 / (2.0 * CS2);
    return w * (rho_m1 + rho * a_term);
}

// Équilibre thermique geq (crate : w·T·[1 + eu/cs²]).
fn geq_c(w: f32, cx: f32, cy: f32, cz: f32, t: f32, ux: f32, uy: f32, uz: f32) -> f32 {
    let eu = cx * ux + cy * uy + cz * uz;
    return w * t * (1.0 + eu / CS2);
}

// Collision TRT d'une direction avec force de Guo (Guo, Shi & Zheng 2002),
// F = (0, fy, 0) : partie impaire (momentum, facteur 1 − 1/(2τ−)) et partie
// paire (stress, facteur 1 − 1/(2τ+)). P1a (variables décalées, port du 2D) :
// fin et la valeur de retour sont DÉCALÉS (f̃ = f − w_i, f̃' = f' − w_i) —
// l'offset w_i se simplifie EXACTEMENT dans la formule CPU : f' = (w + f̃) −
// a·d − b·d_opp + S = w + [f̃ − a·d − b·d_opp + S] puisque d_i = f̃_i −
// (feq_i − w_i) contient déjà le même w_i de part et d'autre. AUCUN (w + fin)
// réinjecté : le gros nombre (~0.44) ne traverse plus la collision. Sur D3Q19,
// Σe·s_plus = 0 (moment 3ᵉ ordre nul) et Σc·s_minus = fy exact ; à u = 0,
// s_plus = 0 identiquement. (le paramètre w reste pour les termes
// S_i = w_i·[…]·fy, qui ne dépendent pas de f.)
fn trt_f(w: f32, cx: f32, cy: f32, cz: f32, fin: f32, d: f32, d_opp: f32,
         ux: f32, uy: f32, uz: f32, fy_node: f32, a_f: f32, b_f: f32,
         tau_minus: f32, tau_plus: f32) -> f32 {
    let ei_u = cx * ux + cy * uy + cz * uz;
    let s_i = w * ((cy - uy) / CS2 + ei_u * cy / (CS2 * CS2)) * fy_node;
    let s_minus = w * (cy / CS2) * fy_node;
    let s_plus = s_i - s_minus;
    return fin - a_f * d - b_f * d_opp
        + (1.0 - 0.5 / tau_minus) * s_minus
        + (1.0 - 0.5 / tau_plus) * s_plus;
}

// Équilibre thermique de paroi isotherme (u = 0) : g_i = w_i·T — les poids
// seuls suffisent (eu = 0). 19 entrées, indices littéraux, stride SoA.
fn wall_iso(t: f32, k: u32) {
    let n = n_tot();
    g_out[0u * n + k] = W0 * t;
    g_out[1u * n + k] = W1 * t;
    g_out[2u * n + k] = W1 * t;
    g_out[3u * n + k] = W1 * t;
    g_out[4u * n + k] = W1 * t;
    g_out[5u * n + k] = W1 * t;
    g_out[6u * n + k] = W1 * t;
    g_out[7u * n + k] = W5 * t;
    g_out[8u * n + k] = W5 * t;
    g_out[9u * n + k] = W5 * t;
    g_out[10u * n + k] = W5 * t;
    g_out[11u * n + k] = W5 * t;
    g_out[12u * n + k] = W5 * t;
    g_out[13u * n + k] = W5 * t;
    g_out[14u * n + k] = W5 * t;
    g_out[15u * n + k] = W5 * t;
    g_out[16u * n + k] = W5 * t;
    g_out[17u * n + k] = W5 * t;
    g_out[18u * n + k] = W5 * t;
}

// Streaming pull d'une direction : source (ix−cx, iy−cy, wrap(iz−cz)) dans le
// domaine x/y → pull (z TOUJOURS valide : périodique, wrap sans bounce-back),
// sinon bounce-back de la distribution sortante opposée (f → f_col[opp],
// g → g_col[dir] valeur provisoire, remplacée par les CL thermiques).
// Indexation dynamique des STORAGE buffers (autorisée, coalescée) ; les
// arguments cx/cy/cz sont des littéraux aux call sites.
fn stream_dir(dir: u32, opp: u32, cx: i32, cy: i32, cz: i32, ix: u32, iy: u32, iz: u32) {
    let n = n_tot();
    let k = ix + P.nx * (iy + P.ny * iz);
    let szi = (i32(iz) - cz + i32(P.nz)) % i32(P.nz);
    let sz = u32(szi);
    let sx = u32(i32(ix) - cx); // hors domaine → cast u32 énorme → test échoue
    let sy = u32(i32(iy) - cy);
    if (sx < P.nx && sy < P.ny) {
        let src = sx + P.nx * (sy + P.ny * sz);
        f_out[dir * n + k] = f_col[dir * n + src];
        g_out[dir * n + k] = g_col[dir * n + src];
    } else {
        f_out[dir * n + k] = f_col[opp * n + k];
        g_out[dir * n + k] = g_col[dir * n + k];
    }
}

// CL horizontale (adiabatique) : valeur post-stream du nœud intérieur
// adjacent (iyin), RECALCULÉE depuis g_col figé (post-collision, avant tout
// streaming — aucune lecture inter-thread). Si la source est hors domaine
// x/y (coins — écrasées par wall_iso ensuite), bounce-back provisoire.
fn wall_grad_dir(dir: u32, cx: i32, cy: i32, cz: i32, ix: u32, iyin: u32, iz: u32) -> f32 {
    let n = n_tot();
    let kin = ix + P.nx * (iyin + P.ny * iz);
    let szi = (i32(iz) - cz + i32(P.nz)) % i32(P.nz);
    let sz = u32(szi);
    let sx = u32(i32(ix) - cx);
    let sy = u32(i32(iyin) - cy);
    if (sx < P.nx && sy < P.ny) {
        return g_col[dir * n + (sx + P.nx * (sy + P.ny * sz))];
    }
    return g_col[dir * n + kin];
}

// ---------------------------------------------------------------------------
// Dispatch 1 — collision TRT + Smagorinsky + Boussinesq (par nœud, sans
// dépendance voisine). ν_t = (Cs·Δx)²·|S| par nœud, τ+ = 0.5 + 3(ν_eff + ν_t),
// τ− = 0.5 + Λ/(τ+ − 0.5), τ_t+ = 0.5 + 3(α_eff + ν_t/Pr_t), force de Guo
// odd/even avec fy évaluée aux champs du temps t.
// ---------------------------------------------------------------------------
@compute @workgroup_size(16, 4, 4)
fn collide(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    let iy = gid.y;
    let iz = gid.z;
    if (ix >= P.nx || iy >= P.ny || iz >= P.nz) { return; }
    let k = ix + P.nx * (iy + P.ny * iz);
    let n = n_tot();

    // Champs macroscopiques du temps t (stock, écrits par le dispatch 2 du
    // pas précédent — y compris la correction de vitesse Guo et le no-slip).
    let ux = stock[5u * k + 1u];
    let uy = stock[5u * k + 2u];
    let uz = stock[5u * k + 3u];
    let temp = stock[5u * k + 4u];

    // P1a (variables décalées) : ρ−1 = Σ f̃_i, calculé comme la SOMME DES
    // DÉCALAGES (jamais les populations recollées W + f̃) — la physique est
    // dans l'écart f_i − w_i (1e-6..1e-3), la somme des 0.44 annulerait le
    // signal dans f32 (biais d'arrondi systématique, dérive de masse G5 §4 :
    // +1.446e-8/pas en 2D).
    let f0s = f_in[0u * n + k];
    let rho_m1 = f0s + f_in[1u * n + k] + f_in[2u * n + k] + f_in[3u * n + k]
        + f_in[4u * n + k] + f_in[5u * n + k] + f_in[6u * n + k] + f_in[7u * n + k]
        + f_in[8u * n + k] + f_in[9u * n + k] + f_in[10u * n + k] + f_in[11u * n + k]
        + f_in[12u * n + k] + f_in[13u * n + k] + f_in[14u * n + k] + f_in[15u * n + k]
        + f_in[16u * n + k] + f_in[17u * n + k] + f_in[18u * n + k];
    let rho = 1.0 + rho_m1;

    // Force de Boussinesq (P1b, ρ₀ dans la flottabilité) : fy =
    // β_eff·(T − t_ref)·(1 + rho_m1), évaluée aux champs du temps t —
    // utilisée par la collision de CE pas. Identique en arithmétique exacte
    // à l'ancien ρ·β_eff·(T−t_ref) (le terme dominant ρ₀ = 1 absorbé, seul
    // le petit ρ−1 = Σ f̃ multiplie — plus de grosse magnitude ρ dans la
    // force). Le dispatch 2 (correction de vitesse Guo éq. (16)) recalcule
    // localement avec la MÊME expression P1b (même F — sémantique CPU).
    // Formule STRICTEMENT identique au 2D (uniform/gpu.rs : P.beta_eff·
    // (temp − P.t_ref)·(1 + rho_m1)).
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
    let f9s = f_in[9u * n + k];
    let f10s = f_in[10u * n + k];
    let f11s = f_in[11u * n + k];
    let f12s = f_in[12u * n + k];
    let f13s = f_in[13u * n + k];
    let f14s = f_in[14u * n + k];
    let f15s = f_in[15u * n + k];
    let f16s = f_in[16u * n + k];
    let f17s = f_in[17u * n + k];
    let f18s = f_in[18u * n + k];
    let g0 = g_in[0u * n + k];
    let g1 = g_in[1u * n + k];
    let g2 = g_in[2u * n + k];
    let g3 = g_in[3u * n + k];
    let g4 = g_in[4u * n + k];
    let g5 = g_in[5u * n + k];
    let g6 = g_in[6u * n + k];
    let g7 = g_in[7u * n + k];
    let g8 = g_in[8u * n + k];
    let g9 = g_in[9u * n + k];
    let g10 = g_in[10u * n + k];
    let g11 = g_in[11u * n + k];
    let g12 = g_in[12u * n + k];
    let g13 = g_in[13u * n + k];
    let g14 = g_in[14u * n + k];
    let g15 = g_in[15u * n + k];
    let g16 = g_in[16u * n + k];
    let g17 = g_in[17u * n + k];
    let g18 = g_in[18u * n + k];

    // Équilibres en DÉCALÉ (P1a : feq_i − w_i, mêmes expressions que la
    // boucle au terme w_i·ρ·A près — identité exacte). rho_m1 = Σ f̃, rho
    // pleine magnitude en facteur de ρ·A uniquement.
    let f_eq0 = feq_off_c(W0, 0.0, 0.0, 0.0, rho_m1, rho, ux, uy, uz);
    let f_eq1 = feq_off_c(W1, 1.0, 0.0, 0.0, rho_m1, rho, ux, uy, uz);
    let f_eq2 = feq_off_c(W1, -1.0, 0.0, 0.0, rho_m1, rho, ux, uy, uz);
    let f_eq3 = feq_off_c(W1, 0.0, 1.0, 0.0, rho_m1, rho, ux, uy, uz);
    let f_eq4 = feq_off_c(W1, 0.0, -1.0, 0.0, rho_m1, rho, ux, uy, uz);
    let f_eq5 = feq_off_c(W1, 0.0, 0.0, 1.0, rho_m1, rho, ux, uy, uz);
    let f_eq6 = feq_off_c(W1, 0.0, 0.0, -1.0, rho_m1, rho, ux, uy, uz);
    let f_eq7 = feq_off_c(W5, 1.0, 1.0, 0.0, rho_m1, rho, ux, uy, uz);
    let f_eq8 = feq_off_c(W5, -1.0, 1.0, 0.0, rho_m1, rho, ux, uy, uz);
    let f_eq9 = feq_off_c(W5, 1.0, -1.0, 0.0, rho_m1, rho, ux, uy, uz);
    let f_eq10 = feq_off_c(W5, -1.0, -1.0, 0.0, rho_m1, rho, ux, uy, uz);
    let f_eq11 = feq_off_c(W5, 1.0, 0.0, 1.0, rho_m1, rho, ux, uy, uz);
    let f_eq12 = feq_off_c(W5, -1.0, 0.0, 1.0, rho_m1, rho, ux, uy, uz);
    let f_eq13 = feq_off_c(W5, 1.0, 0.0, -1.0, rho_m1, rho, ux, uy, uz);
    let f_eq14 = feq_off_c(W5, -1.0, 0.0, -1.0, rho_m1, rho, ux, uy, uz);
    let f_eq15 = feq_off_c(W5, 0.0, 1.0, 1.0, rho_m1, rho, ux, uy, uz);
    let f_eq16 = feq_off_c(W5, 0.0, -1.0, 1.0, rho_m1, rho, ux, uy, uz);
    let f_eq17 = feq_off_c(W5, 0.0, 1.0, -1.0, rho_m1, rho, ux, uy, uz);
    let f_eq18 = feq_off_c(W5, 0.0, -1.0, -1.0, rho_m1, rho, ux, uy, uz);
    let g_eq0 = geq_c(W0, 0.0, 0.0, 0.0, temp, ux, uy, uz);
    let g_eq1 = geq_c(W1, 1.0, 0.0, 0.0, temp, ux, uy, uz);
    let g_eq2 = geq_c(W1, -1.0, 0.0, 0.0, temp, ux, uy, uz);
    let g_eq3 = geq_c(W1, 0.0, 1.0, 0.0, temp, ux, uy, uz);
    let g_eq4 = geq_c(W1, 0.0, -1.0, 0.0, temp, ux, uy, uz);
    let g_eq5 = geq_c(W1, 0.0, 0.0, 1.0, temp, ux, uy, uz);
    let g_eq6 = geq_c(W1, 0.0, 0.0, -1.0, temp, ux, uy, uz);
    let g_eq7 = geq_c(W5, 1.0, 1.0, 0.0, temp, ux, uy, uz);
    let g_eq8 = geq_c(W5, -1.0, 1.0, 0.0, temp, ux, uy, uz);
    let g_eq9 = geq_c(W5, 1.0, -1.0, 0.0, temp, ux, uy, uz);
    let g_eq10 = geq_c(W5, -1.0, -1.0, 0.0, temp, ux, uy, uz);
    let g_eq11 = geq_c(W5, 1.0, 0.0, 1.0, temp, ux, uy, uz);
    let g_eq12 = geq_c(W5, -1.0, 0.0, 1.0, temp, ux, uy, uz);
    let g_eq13 = geq_c(W5, 1.0, 0.0, -1.0, temp, ux, uy, uz);
    let g_eq14 = geq_c(W5, -1.0, 0.0, -1.0, temp, ux, uy, uz);
    let g_eq15 = geq_c(W5, 0.0, 1.0, 1.0, temp, ux, uy, uz);
    let g_eq16 = geq_c(W5, 0.0, -1.0, 1.0, temp, ux, uy, uz);
    let g_eq17 = geq_c(W5, 0.0, 1.0, -1.0, temp, ux, uy, uz);
    let g_eq18 = geq_c(W5, 0.0, -1.0, -1.0, temp, ux, uy, uz);

    // Moments non-équilibrés π_neq (avant relaxation) pour |S| — P1a :
    // d_i = f̃_i − (feq_i − w_i), calculés UNIQUEMENT sur les petits nombres
    // (hors équilibre, mêmes contributions et ordre que la boucle). 3D :
    // pi_xx = d1+d2+d7..d14 ; pi_yy = d3+d4+d7..d10+d15..d18 ;
    // pi_zz = d5+d6+d11..d18 ; pi_xy = d7−d8−d9+d10 ; pi_xz = d11−d12−d13+d14 ;
    // pi_yz = d15−d16−d17+d18.
    let d0 = f0s - f_eq0;
    let d1 = f1s - f_eq1;
    let d2 = f2s - f_eq2;
    let d3 = f3s - f_eq3;
    let d4 = f4s - f_eq4;
    let d5 = f5s - f_eq5;
    let d6 = f6s - f_eq6;
    let d7 = f7s - f_eq7;
    let d8 = f8s - f_eq8;
    let d9 = f9s - f_eq9;
    let d10 = f10s - f_eq10;
    let d11 = f11s - f_eq11;
    let d12 = f12s - f_eq12;
    let d13 = f13s - f_eq13;
    let d14 = f14s - f_eq14;
    let d15 = f15s - f_eq15;
    let d16 = f16s - f_eq16;
    let d17 = f17s - f_eq17;
    let d18 = f18s - f_eq18;
    let pi_xx = d1 + d2 + d7 + d8 + d9 + d10 + d11 + d12 + d13 + d14;
    let pi_yy = d3 + d4 + d7 + d8 + d9 + d10 + d15 + d16 + d17 + d18;
    let pi_zz = d5 + d6 + d11 + d12 + d13 + d14 + d15 + d16 + d17 + d18;
    let pi_xy = d7 - d8 - d9 + d10;
    let pi_xz = d11 - d12 - d13 + d14;
    let pi_yz = d15 - d16 - d17 + d18;
    // |S| = sqrt(π:π) / (2·cs²·τ_mol·ρ) ; π:π = Πxx² + Πyy² + Πzz²
    //       + 2(Πxy² + Πxz² + Πyz²) (même convention que le CPU 2D).
    let pi2 = pi_xx * pi_xx + pi_yy * pi_yy + pi_zz * pi_zz
        + 2.0 * (pi_xy * pi_xy + pi_xz * pi_xz + pi_yz * pi_yz);
    let s_mag = sqrt(pi2) / (2.0 * P.cs2 * P.tau_mol * max(rho, 1e-12));
    // Viscosité turbulente : ν_t = (Cs·Δx)²·|S| (Δx = 1), par nœud.
    let nu_t_node = P.cs_smag * P.cs_smag * s_mag;
    nu_t[k] = nu_t_node;

    // Temps de relaxation effectifs (par nœud — dépendent de ν_t) :
    // τ+ = 0.5 + 3(ν_eff + ν_t), τ− = 0.5 + Λ/(τ+ − 0.5) ;
    // scalaire : α_t = ν_t/Pr_t, τ_t+ = 0.5 + 3(α_eff + α_t), τ_t− = τ_t+
    // (BGK scalaire, même convention que le 2D validé).
    let tau_plus = 0.5 + 3.0 * (P.nu_eff + nu_t_node);
    let tau_minus = 0.5 + P.lambda_trt / (tau_plus - 0.5);
    let alpha_t = nu_t_node / P.pr_t;
    let tau_t_plus = 0.5 + 3.0 * (P.alpha_eff + alpha_t);
    let a_f = 0.5 * (1.0 / tau_plus + 1.0 / tau_minus);
    let b_f = 0.5 * (1.0 / tau_plus - 1.0 / tau_minus);
    let a_g = 1.0 / tau_t_plus; // τ_t− = τ_t+ ⇒ b_g = 0

    // Collision TRT avec force (Guo) — paires opposées explicites :
    // (0), (1,2), (3,4), (5,6), (7,10), (8,9), (11,14), (12,13), (15,18),
    // (16,17). P1a : trt_f travaille en DÉCALÉ (fin = f̃, retourne f̃') —
    // l'offset w_i se simplifie EXACTEMENT dans l'identité f' = w + f̃' : la
    // partie (w + fin) − ... − w s'annule, plus aucun gros nombre (~0.44)
    // dans la collision. Σ f̃' = Σ f̃ : masse conservée, sémantique TRT
    // identique. Deviations d_i/d_opp en petits nombres (identiques à la
    // boucle au terme w_i près, qui se simplifie exactement).
    f_col[0u * n + k] = trt_f(W0, 0.0, 0.0, 0.0, f_in[0u * n + k], d0, d0, ux, uy, uz,
                              fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[1u * n + k] = trt_f(W1, 1.0, 0.0, 0.0, f_in[1u * n + k], d1, d2, ux, uy, uz,
                              fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[2u * n + k] = trt_f(W1, -1.0, 0.0, 0.0, f_in[2u * n + k], d2, d1, ux, uy, uz,
                              fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[3u * n + k] = trt_f(W1, 0.0, 1.0, 0.0, f_in[3u * n + k], d3, d4, ux, uy, uz,
                              fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[4u * n + k] = trt_f(W1, 0.0, -1.0, 0.0, f_in[4u * n + k], d4, d3, ux, uy, uz,
                              fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[5u * n + k] = trt_f(W1, 0.0, 0.0, 1.0, f_in[5u * n + k], d5, d6, ux, uy, uz,
                              fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[6u * n + k] = trt_f(W1, 0.0, 0.0, -1.0, f_in[6u * n + k], d6, d5, ux, uy, uz,
                              fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[7u * n + k] = trt_f(W5, 1.0, 1.0, 0.0, f_in[7u * n + k], d7, d10, ux, uy, uz,
                              fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[10u * n + k] = trt_f(W5, -1.0, -1.0, 0.0, f_in[10u * n + k], d10, d7, ux, uy, uz,
                               fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[8u * n + k] = trt_f(W5, -1.0, 1.0, 0.0, f_in[8u * n + k], d8, d9, ux, uy, uz,
                              fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[9u * n + k] = trt_f(W5, 1.0, -1.0, 0.0, f_in[9u * n + k], d9, d8, ux, uy, uz,
                              fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[11u * n + k] = trt_f(W5, 1.0, 0.0, 1.0, f_in[11u * n + k], d11, d14, ux, uy, uz,
                               fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[14u * n + k] = trt_f(W5, -1.0, 0.0, -1.0, f_in[14u * n + k], d14, d11, ux, uy, uz,
                               fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[12u * n + k] = trt_f(W5, -1.0, 0.0, 1.0, f_in[12u * n + k], d12, d13, ux, uy, uz,
                               fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[13u * n + k] = trt_f(W5, 1.0, 0.0, -1.0, f_in[13u * n + k], d13, d12, ux, uy, uz,
                               fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[15u * n + k] = trt_f(W5, 0.0, 1.0, 1.0, f_in[15u * n + k], d15, d18, ux, uy, uz,
                               fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[18u * n + k] = trt_f(W5, 0.0, -1.0, -1.0, f_in[18u * n + k], d18, d15, ux, uy, uz,
                               fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[16u * n + k] = trt_f(W5, 0.0, -1.0, 1.0, f_in[16u * n + k], d16, d17, ux, uy, uz,
                               fy_node, a_f, b_f, tau_minus, tau_plus);
    f_col[17u * n + k] = trt_f(W5, 0.0, 1.0, -1.0, f_in[17u * n + k], d17, d16, ux, uy, uz,
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
    g_col[9u * n + k] = g9 - a_g * (g9 - g_eq9);
    g_col[10u * n + k] = g10 - a_g * (g10 - g_eq10);
    g_col[11u * n + k] = g11 - a_g * (g11 - g_eq11);
    g_col[12u * n + k] = g12 - a_g * (g12 - g_eq12);
    g_col[13u * n + k] = g13 - a_g * (g13 - g_eq13);
    g_col[14u * n + k] = g14 - a_g * (g14 - g_eq14);
    g_col[15u * n + k] = g15 - a_g * (g15 - g_eq15);
    g_col[16u * n + k] = g16 - a_g * (g16 - g_eq16);
    g_col[17u * n + k] = g17 - a_g * (g17 - g_eq17);
    g_col[18u * n + k] = g18 - a_g * (g18 - g_eq18);
}

// ---------------------------------------------------------------------------
// Dispatch 2 — streaming pull + bounce-back PUIS CL thermiques PUIS
// macroscopique (2 dispatches/pas). Lit f_col/g_col FIGÉS du dispatch 1
// (même sémantique que le CPU, qui clone f/g avant streaming). z PÉRIODIQUE :
// wrap sans bounce-back. Les CL thermiques recalcule la valeur du nœud
// intérieur DEPUIS g_col figé (post-collision) — aucune lecture inter-thread.
// Ordre CL : horizontales (zero-gradient, lit g_col) PUIS verticales
// (isothermes wall_iso) — les coins x∩y se terminent par wall_iso (identité
// CPU : verticales d'abord, horizontales ensuite, coins écrasées).
// ---------------------------------------------------------------------------
@compute @workgroup_size(16, 4, 4)
fn stream_macro(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ix = gid.x;
    let iy = gid.y;
    let iz = gid.z;
    if (ix >= P.nx || iy >= P.ny || iz >= P.nz) { return; }
    let k = ix + P.nx * (iy + P.ny * iz);
    let n = n_tot();

    // Streaming pull + bounce-back, 19 directions (indices littéraux aux
    // call sites ; z wrap périodique dans stream_dir).
    stream_dir(0u, 0u, 0, 0, 0, ix, iy, iz);
    stream_dir(1u, 2u, 1, 0, 0, ix, iy, iz);
    stream_dir(2u, 1u, -1, 0, 0, ix, iy, iz);
    stream_dir(3u, 4u, 0, 1, 0, ix, iy, iz);
    stream_dir(4u, 3u, 0, -1, 0, ix, iy, iz);
    stream_dir(5u, 6u, 0, 0, 1, ix, iy, iz);
    stream_dir(6u, 5u, 0, 0, -1, ix, iy, iz);
    stream_dir(7u, 10u, 1, 1, 0, ix, iy, iz);
    stream_dir(8u, 9u, -1, 1, 0, ix, iy, iz);
    stream_dir(9u, 8u, 1, -1, 0, ix, iy, iz);
    stream_dir(10u, 7u, -1, -1, 0, ix, iy, iz);
    stream_dir(11u, 14u, 1, 0, 1, ix, iy, iz);
    stream_dir(12u, 13u, -1, 0, 1, ix, iy, iz);
    stream_dir(13u, 12u, 1, 0, -1, ix, iy, iz);
    stream_dir(14u, 11u, -1, 0, -1, ix, iy, iz);
    stream_dir(15u, 18u, 0, 1, 1, ix, iy, iz);
    stream_dir(16u, 17u, 0, -1, 1, ix, iy, iz);
    stream_dir(17u, 16u, 0, 1, -1, ix, iy, iz);
    stream_dir(18u, 15u, 0, -1, -1, ix, iy, iz);

    // --- CL thermiques. Le CPU applique les BC thermiques AVANT
    // update_macroscopic : le T de paroi est calculé sur le g post-CL.
    // Ordre CPU : verticales d'abord, horizontales ensuite (coins écrasées
    // par la copie horizontale → ré-appliquées wall_iso). Ici : horizontales
    // (lit g_col uniquement) PUIS verticales wall_iso — les coins se
    // terminent par wall_iso, identité CPU, aucun ordre d'écriture à
    // synchroniser (tout est recalculé depuis g_col figé).

    // --- Parois horizontales : adiabatiques, g paroi = copie du nœud
    // intérieur adjacent (post-stream, recalculé depuis g_col). Si ce nœud
    // est lui-même un coin (paroi verticale), sa valeur est écrasée par
    // wall_iso ci-dessous (identité CPU).
    if (iy == 0u || iy == P.ny - 1u) {
        let iyin = select(iy + 1u, iy - 1u, iy == P.ny - 1u);
        g_out[0u * n + k] = wall_grad_dir(0u, 0, 0, 0, ix, iyin, iz);
        g_out[1u * n + k] = wall_grad_dir(1u, 1, 0, 0, ix, iyin, iz);
        g_out[2u * n + k] = wall_grad_dir(2u, -1, 0, 0, ix, iyin, iz);
        g_out[3u * n + k] = wall_grad_dir(3u, 0, 1, 0, ix, iyin, iz);
        g_out[4u * n + k] = wall_grad_dir(4u, 0, -1, 0, ix, iyin, iz);
        g_out[5u * n + k] = wall_grad_dir(5u, 0, 0, 1, ix, iyin, iz);
        g_out[6u * n + k] = wall_grad_dir(6u, 0, 0, -1, ix, iyin, iz);
        g_out[7u * n + k] = wall_grad_dir(7u, 1, 1, 0, ix, iyin, iz);
        g_out[8u * n + k] = wall_grad_dir(8u, -1, 1, 0, ix, iyin, iz);
        g_out[9u * n + k] = wall_grad_dir(9u, 1, -1, 0, ix, iyin, iz);
        g_out[10u * n + k] = wall_grad_dir(10u, -1, -1, 0, ix, iyin, iz);
        g_out[11u * n + k] = wall_grad_dir(11u, 1, 0, 1, ix, iyin, iz);
        g_out[12u * n + k] = wall_grad_dir(12u, -1, 0, 1, ix, iyin, iz);
        g_out[13u * n + k] = wall_grad_dir(13u, 1, 0, -1, ix, iyin, iz);
        g_out[14u * n + k] = wall_grad_dir(14u, -1, 0, -1, ix, iyin, iz);
        g_out[15u * n + k] = wall_grad_dir(15u, 0, 1, 1, ix, iyin, iz);
        g_out[16u * n + k] = wall_grad_dir(16u, 0, -1, 1, ix, iyin, iz);
        g_out[17u * n + k] = wall_grad_dir(17u, 0, 1, -1, ix, iyin, iz);
        g_out[18u * n + k] = wall_grad_dir(18u, 0, -1, -1, ix, iyin, iz);
    }
    // --- Parois verticales (chaude à gauche, froide à droite) : isothermes,
    // g paroi = geq(T_wall, u = 0) pour les 19 directions. Appliqué EN
    // DERNIER : les coins x∩y se terminent par wall_iso (ordre CPU).
    if (ix == 0u) {
        wall_iso(P.t_hot, k);
    } else if (ix == P.nx - 1u) {
        wall_iso(P.t_cold, k);
    }
    // --- Axe z : PÉRIODIQUE (wrap dans stream_dir), rien à faire.

    // --- Macroscopique (ρ, u, T) depuis f_out / g_out (post-stream + CL).
    // P1a (variables décalées) : ρ−1 = Σ f̃_i (somme des décalages, SANS
    // recoller les W + f̃ — la somme des 0.44 annulerait le signal f32) ;
    // mx/my/mz = Σ c_i·f_i = Σ c_i·f̃_i EXACTEMENT (Σ c_i·w_i = 0, symétrie
    // D3Q19 — poids opposés égaux, table D3Q19 du crate vérifiée), donc
    // moments en décalé identiques au CPU, sans aucun gros nombre. Déroulé,
    // même ordre d'accumulation que la boucle (j = 0..19).
    let s0 = f_out[0u * n + k];
    let s1 = f_out[1u * n + k];
    let s2 = f_out[2u * n + k];
    let s3 = f_out[3u * n + k];
    let s4 = f_out[4u * n + k];
    let s5 = f_out[5u * n + k];
    let s6 = f_out[6u * n + k];
    let s7 = f_out[7u * n + k];
    let s8 = f_out[8u * n + k];
    let s9 = f_out[9u * n + k];
    let s10 = f_out[10u * n + k];
    let s11 = f_out[11u * n + k];
    let s12 = f_out[12u * n + k];
    let s13 = f_out[13u * n + k];
    let s14 = f_out[14u * n + k];
    let s15 = f_out[15u * n + k];
    let s16 = f_out[16u * n + k];
    let s17 = f_out[17u * n + k];
    let s18 = f_out[18u * n + k];
    var rho_m1 = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9
        + s10 + s11 + s12 + s13 + s14 + s15 + s16 + s17 + s18;
    var rho = 1.0 + rho_m1;
    var mx = s1 - s2 + s7 - s8 + s9 - s10 + s11 - s12 + s13 - s14;
    var my = s3 - s4 + s7 + s8 - s9 - s10 + s15 - s16 + s17 - s18;
    var mz = s5 - s6 + s11 + s12 - s13 - s14 + s15 + s16 - s17 - s18;
    var temp = g_out[0u * n + k] + g_out[1u * n + k] + g_out[2u * n + k]
        + g_out[3u * n + k] + g_out[4u * n + k] + g_out[5u * n + k]
        + g_out[6u * n + k] + g_out[7u * n + k] + g_out[8u * n + k]
        + g_out[9u * n + k] + g_out[10u * n + k] + g_out[11u * n + k]
        + g_out[12u * n + k] + g_out[13u * n + k] + g_out[14u * n + k]
        + g_out[15u * n + k] + g_out[16u * n + k] + g_out[17u * n + k]
        + g_out[18u * n + k];
    // Correction de vitesse Guo et al. 2002, éq. (16) : u_y += fy/(2ρ) —
    // fy recalculé localement avec l'expression P1b (ρ₀ dans la flottabilité,
    // identique en arithmétique exacte à l'ancien ρ·β_eff·(T−t_ref)) depuis
    // g_out (même expression que la collision de CE pas — même F, sémantique
    // CPU `update_macroscopic`). Formule STRICTEMENT identique au 2D. ρ
    // pleine magnitude uniquement au dénominateur (division, inchangée).
    var ux = 0.0;
    var uy = 0.0;
    var uz = 0.0;
    let fy_node = P.beta_eff * (temp - P.t_ref) * (1.0 + rho_m1);
    if (rho > 1e-12) {
        ux = mx / rho;
        uy = my / rho + 0.5 * fy_node / rho;
        uz = mz / rho;
    }
    stock[5u * k + 0u] = rho;
    stock[5u * k + 1u] = ux;
    stock[5u * k + 2u] = uy;
    stock[5u * k + 3u] = uz;
    stock[5u * k + 4u] = temp;

    // --- No-slip : u = 0 sur le périmètre x/y (enforce_wall_velocity du
    // CPU). L'axe z est périodique : aucun mur.
    if (ix == 0u || ix == P.nx - 1u || iy == 0u || iy == P.ny - 1u) {
        stock[5u * k + 1u] = 0.0;
        stock[5u * k + 2u] = 0.0;
        stock[5u * k + 3u] = 0.0;
    }
}
"#;

impl GpuSolver3D {
    /// Initialise le backend GPU depuis le solveur 3D (champs identiques :
    /// nu, alpha, beta_eff, cs, pr_t). Les populations partent des ÉQUILIBRES
    /// reconstruits depuis les champs macroscopiques du CPU (feq_d3q19 / geq
    /// inline, converties f64 → f32 avec décalage f_i − w_i), le
    /// macroscopique part des champs du CPU. La rampe de spin-up est
    /// recalculée côté CPU à chaque pas (uniform buffer réécrit), mêmes
    /// expressions que `params_from`.
    pub fn from_solver(sim: &CavitySolver3D) -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| {
            format!("wgpu : aucun adaptateur GPU disponible ({e}) — le mode uniform3d (Étape V1) est GPU-first, pas de chemin CPU")
        })?;
        let info = adapter.get_info();
        let adapter_info = format!("{} ({:?})", info.name, info.backend);
        // ⚠️ required_limits = limites de l'adaptateur (PAS Limits::default()) :
        // à 96×384×192, chaque buffer population fait ~538 MiB > 128 MiB
        // (max_storage_buffer_binding_size par défaut).
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("cavity-les-gpu-3d"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits().clone(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| format!("wgpu : échec de création du device ({e})"))?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cavity_kernel_3d_d3q19"),
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
        let pipe_collide = mk_pipe("cavity3d_collide", "collide");
        let pipe_stream_macro = mk_pipe("cavity3d_stream_macro", "stream_macro");

        let nx = sim.nx;
        let ny = sim.ny;
        let nz = sim.nz;
        let n = nx * ny * nz;

        // Buffers f/g : populations DÉCALÉES (f_i − w_i·ρ₀, g_i tel quel car
        // T̄ = 0). Conversion f64 → f32 à l'interface (parade FluidX3D).
        // Layout SoA [direction][nœud] : f[i·n + k]. Les populations partent
        // des équilibres (f = feq_d3q19(ρ, u), g = w_i·T·(1 + eu/cs²) inline).
        let cs2_f64 = 1.0f64 / 3.0;
        let mut f_off = vec![0.0f32; 19 * n];
        let mut g_off = vec![0.0f32; 19 * n];
        for k in 0..n {
            let feq = feq_d3q19(sim.rho[k], sim.ux[k], sim.uy[k], sim.uz[k]);
            let (t, ux, uy, uz) =
                (sim.temperature[k], sim.ux[k], sim.uy[k], sim.uz[k]);
            for i in 0..19 {
                f_off[i * n + k] = (feq[i] - D3Q19_WEIGHTS[i]) as f32;
                let cx = D3Q19_VELOCITIES[i][0];
                let cy = D3Q19_VELOCITIES[i][1];
                let cz = D3Q19_VELOCITIES[i][2];
                let eu = cx as f64 * ux + cy as f64 * uy + cz as f64 * uz;
                g_off[i * n + k] = (D3Q19_WEIGHTS[i] * t * (1.0 + eu / cs2_f64)) as f32;
            }
        }
        let mk_pop = |label: &'static str, f: bool| {
            let data: Vec<f32> = if f { f_off.clone() } else { g_off.clone() };
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
        let f_col = Self::empty_storage(&device, 19 * n, "f_col");
        let g_col = Self::empty_storage(&device, 19 * n, "g_col");

        // ν_t : copie de l'état CPU (sanité des lectures avant le 1er pas).
        let nu_t_init: Vec<f32> = sim.nu_t.iter().map(|&v| v as f32).collect();
        let nu_t_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nu_t"),
            contents: bytemuck::cast_slice(&nu_t_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // Stock macroscopique : champs du solveur CPU (temps t initial),
        // stride 5 [ρ, ux, uy, uz, T].
        let mut stock_init = vec![0.0f32; 5 * n];
        for k in 0..n {
            stock_init[5 * k] = sim.rho[k] as f32;
            stock_init[5 * k + 1] = sim.ux[k] as f32;
            stock_init[5 * k + 2] = sim.uy[k] as f32;
            stock_init[5 * k + 3] = sim.uz[k] as f32;
            stock_init[5 * k + 4] = sim.temperature[k] as f32;
        }
        let stock_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stock"),
            contents: bytemuck::cast_slice(&stock_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        // Staging de lecture : stock (5n) + nu_t (n) f32.
        let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read_staging"),
            size: (6 * n * std::mem::size_of::<f32>()) as u64,
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
        let mk_bg = |label: &'static str,
                     layout: &wgpu::BindGroupLayout,
                     entries: &[wgpu::BindGroupEntry]| {
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

        // collide : uniform + f_in + g_in + stock + f_col + g_col + nu_t.
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
                ent(8, ntb.clone()),
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
                ent(8, ntb.clone()),
            ],
        );
        // stream_macro : uniform + f_out + g_out + stock + f_col + g_col
        // (f_out/g_out = buffer partenaire du ping-pong).
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
            nz,
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

    /// Paramètres du pas (rampes spin-up incluses, mêmes expressions que le
    /// 2D — bitwise côté f64, converties f32).
    fn params_from(sim: &CavitySolver3D, step: usize) -> GpuParams3D {
        // Rampe spin-up beta (linéaire).
        let beta_eff = if step < sim.spin_up_steps {
            sim.beta_eff * (step as f64 / sim.spin_up_steps as f64)
        } else {
            sim.beta_eff
        };
        // Rampe viscosité/diffusivité (exponentielle).
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
        // τ moléculaire pour |S| (plancher ν, même marge que le 2D).
        let tau_mol = 0.5 + 3.0 * nu_eff.max(0.002 / 3.0);

        GpuParams3D {
            tau_mol: tau_mol as f32,
            nu_eff: nu_eff as f32,
            alpha_eff: alpha_eff as f32,
            beta_eff: beta_eff as f32,
            t_ref: sim.t_ref as f32,
            cs2: (1.0f64 / 3.0) as f32,
            lambda_trt: (3.0f64 / 16.0) as f32,
            cs_smag: sim.cs as f32,
            pr_t: sim.pr_t as f32,
            t_hot: T_HOT as f32,
            t_cold: T_COLD as f32,
            nx: sim.nx as u32,
            ny: sim.ny as u32,
            nz: sim.nz as u32,
            _pad: [0, 0],
        }
    }

    /// Positionne le compteur de pas (rampe spin-up) — utilisé par les tests
    /// analytiques pour sauter la rampe (beta_eff plein dès le 1er pas).
    pub fn set_time_step(&mut self, step: usize) {
        self.time_step = step;
    }

    /// Un pas de temps GPU : 2 dispatches ordonnés (collide → stream_macro)
    /// dans un seul compute pass — chaque dispatch est un usage-scope wgpu,
    /// les barrières storage sont garanties entre dispatches, aucune race.
    ///
    /// Batching : hors spin-up (params constants bit-à-bit), les pas
    /// s'enchaînent dans un SEUL command encoder, flushé tous les BATCH_STEPS
    /// pas ou avant tout readback. Pendant le spin-up (params varient à
    /// chaque pas), un encoder par pas.
    pub fn step(&mut self, sim: &CavitySolver3D) {
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
                &wgpu::CommandEncoderDescriptor { label: Some("step_batch_3d") },
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
                label: Some("step_pass_3d"),
                timestamp_writes: None,
            });
            // Workgroup 16×4×4 (gid.x → ix, gid.y → iy, gid.z → iz).
            let wg_x = (self.nx as u32).div_ceil(16);
            let wg_y = (self.ny as u32).div_ceil(4);
            let wg_z = (self.nz as u32).div_ceil(4);
            pass.set_pipeline(&self.pipe_collide);
            pass.set_bind_group(0, bg_collide, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, wg_z);
            pass.set_pipeline(&self.pipe_stream_macro);
            pass.set_bind_group(0, bg_stream_macro, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, wg_z);
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

    /// Lecture brute du staging (stock 5n + nu_t n f32) — synchro GPU→CPU.
    /// Précondition : l'appelant a flushé l'encoder batché (sync_into /
    /// read_fields / total_mass appellent `flush()` avant `read_raw`).
    fn read_raw(&self) -> Result<Vec<f32>, String> {
        let n = self.nx * self.ny * self.nz;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("read_3d") });
        encoder.copy_buffer_to_buffer(
            &self.stock_buf,
            0,
            &self.read_buf,
            0,
            (5 * n * std::mem::size_of::<f32>()) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.nu_t_buf,
            0,
            &self.read_buf,
            (5 * n * std::mem::size_of::<f32>()) as u64,
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

    /// Réinjecte le macroscopique GPU (ρ, ux, uy, uz, T) et ν_t dans les
    /// champs du solveur CPU 3D — SANS toucher au pas de temps (il n'y en a
    /// pas côté CPU). Permet de réutiliser les diagnostics du solveur
    /// (nusselt_hot/cold, max_u, total_mass, all_finite).
    pub fn sync_into(&mut self, sim: &mut CavitySolver3D) -> Result<(), String> {
        self.flush(); // le batch en attente doit être soumis avant le readback
        let n = self.nx * self.ny * self.nz;
        let data = self.read_raw()?;
        for k in 0..n {
            sim.rho[k] = data[5 * k] as f64;
            sim.ux[k] = data[5 * k + 1] as f64;
            sim.uy[k] = data[5 * k + 2] as f64;
            sim.uz[k] = data[5 * k + 3] as f64;
            sim.temperature[k] = data[5 * k + 4] as f64;
            sim.nu_t[k] = data[5 * n + k] as f64;
        }
        Ok(())
    }

    /// Lit le buffer stock (ρ, ux, uy, uz, T) — synchro GPU→CPU. Retourne
    /// les champs en f64 (conversions f32→f64 à l'interface), layout
    /// k = ix + nx·(iy + ny·iz).
    pub fn read_fields(
        &mut self,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), String> {
        self.flush();
        let n = self.nx * self.ny * self.nz;
        let data = self.read_raw()?;
        let mut rho = vec![0.0f64; n];
        let mut ux = vec![0.0f64; n];
        let mut uy = vec![0.0f64; n];
        let mut uz = vec![0.0f64; n];
        let mut t = vec![0.0f64; n];
        for k in 0..n {
            rho[k] = data[5 * k] as f64;
            ux[k] = data[5 * k + 1] as f64;
            uy[k] = data[5 * k + 2] as f64;
            uz[k] = data[5 * k + 3] as f64;
            t[k] = data[5 * k + 4] as f64;
        }
        Ok((rho, ux, uy, uz, t))
    }

    /// Masse totale Σρ (lit le stock — diagnostic de conservation).
    pub fn total_mass(&mut self) -> Result<f64, String> {
        let (rho, _, _, _, _) = self.read_fields()?;
        Ok(rho.iter().sum())
    }

    /// Nombre de pas exécutés.
    pub fn time_step(&self) -> usize {
        self.time_step
    }

    /// Dimensions de la grille.
    pub fn dims(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }
}

// ---------------------------------------------------------------------------
// Tests GPU 3D (ignorés par défaut : nécessitent un adaptateur GPU, Metal
// ici). Run : cargo test --release -- --ignored
//
// (a) équilibre exact 1 pas ; (b) streaming champ uniforme 3 pas + masse ;
// (c) conduction pure (T linéaire préservée, u = 0) ; (d) conservation masse
// 10k pas ; (e) impulsion de Guo 1 pas = analytique.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod gpu_tests_3d {
    use super::*;

    /// Petit solveur déterministe (graine 42, mêmes unités réseau que la run
    /// de référence : ν calé → > plancher 0.002).
    fn small_solver() -> CavitySolver3D {
        CavitySolver3D::new(16, 64, 8, 1.0e6, 0.71, 0.10, 0.9, 0.02, 42, 1.0)
    }

    /// Applique une température uniforme (écrase le profil linéaire + bruit).
    fn set_uniform_t(sim: &mut CavitySolver3D, t0: f64) {
        for v in sim.temperature.iter_mut() {
            *v = t0;
        }
    }

    /// (a) Équilibre exact : T = t_ref uniforme, u = 0, ρ = 1 — 1 pas (rampe
    /// sautée : beta plein, mais T = t_ref ⇒ force nulle) laisse l'intérieur
    /// exactement invariant (streaming + collide + CL identité sur l'équilibre).
    #[test]
    #[ignore = "nécessite un adaptateur GPU (Metal) — cargo test --release -- --ignored"]
    fn gpu3d_equilibrium_exact_1_step() {
        let mut sim = small_solver();
        let t0 = sim.t_ref;
        set_uniform_t(&mut sim, t0);
        let mut gpu = GpuSolver3D::from_solver(&sim).expect("init GPU");
        gpu.set_time_step(sim.spin_up_steps); // rampe terminée (force nulle : T = t_ref)
        gpu.step(&sim);
        let (rho, ux, uy, uz, t) = gpu.read_fields().expect("read_fields");

        let (nx, ny, nz) = (sim.nx, sim.ny, sim.nz);
        for iz in 0..nz {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let k = ix + nx * (iy + ny * iz);
                    assert!(
                        (rho[k] - 1.0).abs() < 1e-6,
                        "rho k={k} : {}",
                        rho[k]
                    );
                    assert!(ux[k].abs() < 1e-6, "ux k={k} : {}", ux[k]);
                    assert!(uy[k].abs() < 1e-6, "uy k={k} : {}", uy[k]);
                    assert!(uz[k].abs() < 1e-6, "uz k={k} : {}", uz[k]);
                    assert!(
                        (t[k] - t0).abs() < 1e-6,
                        "T k={k} : {} (t0={t0})",
                        t[k]
                    );
                }
            }
        }
    }

    /// (b) Streaming d'un champ uniforme : 3 pas laissent ρ/u intérieurs
    /// exactement invariants (le bounce-back retourne la valeur uniforme ;
    /// Σw_i = 1 et Σw_i·c_i = 0 analytiques) et la masse constante. Le T
    /// uniforme n'est préservé que dans le bulk : les parois verticales
    /// isothermes (±0.5) diffusent vers l'intérieur à partir du pas 2.
    #[test]
    #[ignore = "nécessite un adaptateur GPU (Metal) — cargo test --release -- --ignored"]
    fn gpu3d_uniform_streaming_3_steps() {
        let mut sim = small_solver();
        let t0 = sim.t_ref;
        set_uniform_t(&mut sim, t0);
        let mut gpu = GpuSolver3D::from_solver(&sim).expect("init GPU");
        let n = sim.nx * sim.ny * sim.nz;
        let m0 = gpu.total_mass().expect("masse initiale");
        for _ in 0..3 {
            gpu.step(&sim);
        }
        let (rho, ux, uy, uz, t) = gpu.read_fields().expect("read_fields");
        let m1: f64 = rho.iter().sum();

        let (nx, ny, nz) = (sim.nx, sim.ny, sim.nz);
        for iz in 0..nz {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let k = ix + nx * (iy + ny * iz);
                    // u : bruit f32 des moments GPU (accumulation 19
                    // populations, mesuré ~5e-8 après 3 pas) — le test porte
                    // sur l'ABSENCE de vitesse spurielle physique (>1e-6).
                    assert!(ux[k].abs() < 1e-6, "ux k={k} : {}", ux[k]);
                    assert!(uy[k].abs() < 1e-6, "uy k={k} : {}", uy[k]);
                    assert!(uz[k].abs() < 1e-6, "uz k={k} : {}", uz[k]);
                    // Bulk (à distance ≥ 3 des parois verticales isothermes) :
                    // le T uniforme est préservé (à ix=2 le front thermique
                    // pénétrant est ~5e-2 après 3 pas, physique).
                    if ix >= 3 && ix < nx - 3 {
                        assert!(
                            (t[k] - t0).abs() < 1e-6,
                            "T bulk k={k} (ix={ix}) : {} (t0={t0})",
                            t[k]
                        );
                    }
                }
            }
        }
        assert!(
            // Dérive f32 par nœud (métrique du test 2D) : mesurée ~2.3e-8
            // (somme f64 de 8192 ρ f32, arrondi ε·√n).
            ((m1 - m0).abs() / n as f64) < 1e-6,
            "masse non conservée : m0={m0} m1={m1} dérive/nœud={:.3e}",
            (m1 - m0).abs() / n as f64
        );
    }

    /// (c) Conduction pure : T linéaire en x, beta_eff = 0 (pas de
    /// flottabilité) — la T linéaire est préservée dans le bulk (premier
    /// moment Σw_i·c_i = 0 : le streaming d'un champ linéaire est exact), la
    /// vitesse reste EXACTEMENT nulle (populations à l'équilibre, offset nul)
    /// et la masse par nœud vaut 1. Déviation murale rapportée (non assertée).
    #[test]
    #[ignore = "nécessite un adaptateur GPU (Metal) — cargo test --release -- --ignored"]
    fn gpu3d_pure_conduction_linear_t() {
        let mut sim = small_solver();
        let (nx, ny, nz) = (sim.nx, sim.ny, sim.nz);
        // T linéaire EXACTE (écrase la perturbation ±1e-3).
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let k = ix + nx * (iy + ny * iz);
                    let xf = ix as f64 / (nx - 1) as f64;
                    sim.temperature[k] = T_HOT + (T_COLD - T_HOT) * xf;
                }
            }
        }
        sim.set_beta_eff(0.0);
        let mut gpu = GpuSolver3D::from_solver(&sim).expect("init GPU");
        let n_steps = 200;
        for _ in 0..n_steps {
            gpu.step(&sim);
        }
        let (rho, ux, uy, uz, t) = gpu.read_fields().expect("read_fields");

        // Profil x du T à iy=2 (diagnostic AVANT asserts : déviation de
        // colonne ix, T uniforme en iy/iz).
        let iy0 = 2;
        for ix in 0..nx {
            let k = ix + nx * (iy0 + ny * 0);
            let xf = ix as f64 / (nx - 1) as f64;
            let t_lin = T_HOT + (T_COLD - T_HOT) * xf;
            eprintln!(
                "colonne ix={ix:2} : T={:.6} t_lin={:.6} dev={:+.3e}",
                t[k],
                t_lin,
                t[k] - t_lin
            );
        }

        let mut max_dev_wall: f64 = 0.0;
        for iz in 0..nz {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let k = ix + nx * (iy + ny * iz);
                    // u : bruit f32 des moments GPU (accumulation 19
                    // populations, mesuré ~5e-8 après 200 pas) — le test
                    // porte sur l'ABSENCE de vitesse spurielle physique
                    // (>1e-6), pas sur le zéro machine.
                    assert!(ux[k].abs() < 1e-6, "ux k={k} : {}", ux[k]);
                    assert!(uy[k].abs() < 1e-6, "uy k={k} : {}", uy[k]);
                    assert!(uz[k].abs() < 1e-6, "uz k={k} : {}", uz[k]);
                    assert!((rho[k] - 1.0).abs() < 1e-9, "rho k={k} : {}", rho[k]);
                    let xf = ix as f64 / (nx - 1) as f64;
                    let t_lin = T_HOT + (T_COLD - T_HOT) * xf;
                    let dev = (t[k] - t_lin).abs();
                    // Bulk : ix ≥ 4 (la couche limite numérique du CL
                    // isotherme discret décroît exponentiellement : 2.5e-2 à
                    // ix=1, 7e-3 à ix=3, 2.7e-3 à ix=4, <1e-3 au-delà —
                    // symétrique, T exact aux murs) et iy ≥ 2 (rangée
                    // adjacente au mur adiabat).
                    if (4..nx - 4).contains(&ix) && iy >= 2 {
                        assert!(
                            dev < 5e-3,
                            "T bulk k={k} (ix={ix}, iy={iy}) : {} vs t_lin={t_lin}",
                            t[k]
                        );
                    } else if dev > max_dev_wall {
                        max_dev_wall = dev;
                    }
                }
            }
        }
        // Profil x du T à iy=2 (diagnostic de la déviation de colonne).
        let iy0 = 2;
        for ix in 0..nx {
            let k = ix + nx * (iy0 + ny * 0);
            let xf = ix as f64 / (nx - 1) as f64;
            let t_lin = T_HOT + (T_COLD - T_HOT) * xf;
            eprintln!(
                "colonne ix={ix:2} : T={:.6} t_lin={:.6} dev={:+.3e}",
                t[k],
                t_lin,
                t[k] - t_lin
            );
        }
        eprintln!(
            "conduction pure : déviation murale max (ix ∈ {{1,2,nx−3,nx−2}} ou iy=1) = {max_dev_wall:.3e} (rapportée, non assertée)"
        );
        // Masse/nœud == 1 (ρ = 1 exact).
        let masse: f64 = rho.iter().sum();
        assert!(
            (masse / (nx * ny * nz) as f64 - 1.0).abs() < 1e-9,
            "masse/nœud = {}",
            masse / (nx * ny * nz) as f64
        );
    }

    /// (d) Conservation de la masse : 10k pas pleins (rampes + Smagorinsky +
    /// flottabilité) sur 16×64×16 — dérive de masse/nœud < 1e-3 (f32 + BB).
    #[test]
    #[ignore = "nécessite un adaptateur GPU (Metal) — cargo test --release -- --ignored"]
    fn gpu3d_mass_conservation_10k_steps() {
        let mut sim = CavitySolver3D::new(16, 64, 16, 1.0e6, 0.71, 0.10, 0.9, 0.02, 42, 1.0);
        let mut gpu = GpuSolver3D::from_solver(&sim).expect("init GPU");
        let n = 16 * 64 * 16;
        let m0 = gpu.total_mass().expect("masse initiale");
        for _ in 0..10_000 {
            gpu.step(&sim);
        }
        let m1 = gpu.total_mass().expect("masse finale");
        let derive = (m1 - m0).abs() / n as f64;
        assert!(
            derive < 1e-3,
            "dérive de masse excessive : m0={m0} m1={m1} dérive/nœud={derive:.3e}"
        );
    }

    /// (e) Impulsion de Guo : T = T₀ = 0.1 uniforme (≠ t_ref), u = 0, ρ = 1,
    /// rampe sautée (beta_eff plein) — après 1 pas, la correction de vitesse
    /// Guo éq. (16) donne EXACTEMENT uy = β_eff·T₀·(3/2 − 1/(2τ−)) aux nœuds
    /// intérieurs (moment y : Σc·s⁻ = fy ; Σw·c² = cs² ; + fy/(2ρ) de Guo),
    /// avec τ+ = 0.5 + 3ν (ν_t = 0 à u = 0), τ− = 0.5 + Λ/(τ+ − 0.5).
    /// ux = uz = 0, ρ = 1, T = T₀.
    #[test]
    #[ignore = "nécessite un adaptateur GPU (Metal) — cargo test --release -- --ignored"]
    fn gpu3d_guo_impulse_analytic() {
        let mut sim = small_solver();
        let t0 = 0.1;
        set_uniform_t(&mut sim, t0);
        let mut gpu = GpuSolver3D::from_solver(&sim).expect("init GPU");
        gpu.set_time_step(sim.spin_up_steps); // rampe terminée : beta_eff plein, ν_eff = ν
        gpu.step(&sim);
        let (rho, ux, uy, uz, t) = gpu.read_fields().expect("read_fields");

        // Valeur analytique (mêmes conventions que params_from/shader).
        let beta = sim.beta_eff;
        let tau_plus = 0.5 + 3.0 * sim.nu; // ν_t = 0 à u = 0
        let tau_minus = 0.5 + (3.0 / 16.0) / (tau_plus - 0.5);
        let uy_exp = beta * t0 * (1.5 - 0.5 / tau_minus);

        let (nx, ny, nz) = (sim.nx, sim.ny, sim.nz);
        for iz in 0..nz {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let k = ix + nx * (iy + ny * iz);
                    assert!(
                        (uy[k] - uy_exp).abs() <= 1e-6 * uy_exp.abs(),
                        "uy k={k} : gpu={} attendu={uy_exp} (rel)",
                        uy[k]
                    );
                    assert!(ux[k].abs() < 1e-12, "ux k={k} : {}", ux[k]);
                    assert!(uz[k].abs() < 1e-12, "uz k={k} : {}", uz[k]);
                    assert!((rho[k] - 1.0).abs() < 1e-6, "rho k={k} : {}", rho[k]);
                    assert!((t[k] - t0).abs() < 1e-6, "T k={k} : {}", t[k]);
                }
            }
        }
    }
}
