//! Solveur LBM D3Q19 3D (Étape V1, plan 3D Trias §3) — cavité
//! différentiellement chauffée AR = 4, profondeur z périodique (L1 = 2).
//!
//! TRT (Λ = 3/16) + forcing de Guo + Smagorinsky statique, GPU-first (wgpu,
//! f32, offset FluidX3D f_i − w_i·ρ₀). Ce module est une extension additive :
//! le chemin 2D (`uniform`) reste bit-à-bit intact.
//!
//! - `solver` : calage unités + champs initiaux uniquement (pas de `step()`
//!   CPU 3D — la validation passe par les tests device 1-pas de `gpu.rs`).
//! - `gpu`    : pas de temps GPU (2 dispatches fusionnés collide →
//!   stream_macro), buffers SoA `[direction][nœud]`, 19 directions déroulées.
//! - `rng`    : PRNG xorshift64* (perturbation initiale).

pub mod gpu;
pub mod rng;
pub mod solver;
