//! Petit générateur pseudo-aléatoire xorshift64* (aucune dépendance).
//! Utilisé uniquement pour la perturbation initiale de température.

/// PRNG xorshift64* déterministe.
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Crée le générateur avec une graine non nulle.
    pub fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    /// Prochain entier 64 bits.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Prochain flottant dans [0, 1).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}
