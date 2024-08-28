use ark_bn254::{Fq, Fr};
use ark_ff::PrimeField;
use ark_std::Zero;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::hint::black_box;
use vectorized_fields::{read_vec_from_file, save_vec_to_file};

use vectorized_fields::benchmarks;

fn main() {
    benchmarks::bench_mul_single_threaded();
    benchmarks::bench_mul_multi_threaded();
    benchmarks::bench_inner_product_multi_threaded();
}
