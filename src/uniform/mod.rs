//! # uniform — solveur uniforme de référence (cavité DHC 2D, Ra=1e6)
//!
//! Solveur LBM D2Q9 double distribution avec Smagorinsky LES, construit sur la
//! crate `oxiphysics-lbm` (structure ThermalD2Q9 + équilibres + fonctions
//! physiques), pas de temps / CL / post-traitement maison.
//!
//! Ancien code à la racine du crate (`solver.rs`, `post.rs`, `rng.rs`,
//! `snapshot.rs`, `tests.rs`), déplacé dans ce module derrière le flag
//! `--mode uniform` — par symétrie avec `src/octree/`.

pub mod gpu;
pub mod post;
pub mod rng;
pub mod snapshot;
pub mod solver;

#[cfg(test)]
mod tests;
