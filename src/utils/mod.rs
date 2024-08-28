use ark_ff::PrimeField;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

pub fn rand_vec<F: PrimeField>(size: usize) -> Vec<F> {
    (0..size)
        .into_par_iter()
        .map_init(|| ChaCha8Rng::from_entropy(), |rng, _| F::rand(rng))
        .collect()
}
