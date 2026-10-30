//! # cfd_cavity_octree — raffinement de maillage octree (grilles cartésiennes 2:1)
//!
//! Extension de la cavité DHC 2D LBM (D2Q9 double distribution, TRT Λ = 3/16)
//! au raffinement de maillage multi-niveaux, d'après `PLAN_OCTREE_REFINEMENT.md`.
//!
//! - Étape A (porte V1, ce crate) : quadtree 2 niveaux (ℓ=0 grossier, ℓ=1 fin),
//!   bloc fin rectangulaire intérieur, module interface avec rescaling f^neq
//!   (§1.2 : f^neq_fin = (1/2)·ξ·f^neq_grossier, ξ = τ_fin/τ_grossier sur la
//!   partie paire), tests unitaires V1.
//! - Le crate uniforme validé `cavity_les` n'est PAS modifié (§2 du plan).
//!
//! Briques réutilisées à l'identique du crate uniforme : collision TRT
//! (coefficients a/b, force Guo odd/even), streaming pull avec bounce-back,
//! équilibres D2Q9 double distribution (f vitesse + g scalaire T) de
//! `oxiphysics-lbm` (feq/geq). Pas de Smagorinsky en Étape A : ν constant
//! par niveau (τ⁺/τ⁻ constants par niveau, ξ = rapport des τ).

pub mod grid;
pub mod interface;
pub mod lbm;
pub mod solver;

#[cfg(test)]
mod tests;
