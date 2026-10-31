//! # octree — raffinement de maillage octree (grilles cartésiennes 2:1)
//!
//! Extension de la cavité DHC 2D LBM (D2Q9 double distribution, TRT Λ = 3/16)
//! au raffinement de maillage multi-niveaux, d'après `PLAN_OCTREE_REFINEMENT.md`.
//! Ancien crate autonome `cfd_cavity_octree`, fusionné dans `cavity_les`
//! derrière le flag `--mode octree` (mêmes flags, mêmes défauts).
//!
//! - Étape A (porte V1) : quadtree 2 niveaux (ℓ=0 grossier, ℓ=1 fin),
//!   bloc fin rectangulaire intérieur, module interface avec rescaling f^neq
//!   (§1.2 : f^neq_fin = (1/2)·ξ·f^neq_grossier, ξ = τ_fin/τ_grossier sur la
//!   partie paire), tests unitaires V1.
//! - Étape B (porte V2, plan §3.1/§3.3) : cavité DHC 2D Ra=1e6 sur l'octree
//!   (module `cavity` : collision TRT + Boussinesq Guo, CL thermiques, Nusselt,
//!   monitoring bruit acoustique max|ρ−ρ̄| global et aux interfaces).
//!
//! Briques réutilisées à l'identique du solveur uniforme : collision TRT
//! (coefficients a/b, force Guo odd/even), streaming pull avec bounce-back,
//! équilibres D2Q9 double distribution (f vitesse + g scalaire T) de
//! `oxiphysics-lbm` (feq/geq).

// Items exportés par l'ancien crate (lib) mais non utilisés par le binaire
// (solveur octree d'Étape A, collisions bas niveau, etc.) : le binaire n'y
// touche pas, on neutralise les warnings dead_code au niveau du module.
#![allow(dead_code)]

pub mod cavity;
pub mod grid;
pub mod interface;
pub mod lbm;
pub mod smag;
pub mod snapshot;
pub mod solver;

#[cfg(test)]
mod tests;
