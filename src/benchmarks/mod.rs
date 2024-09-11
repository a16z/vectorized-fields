use crate::{
    ark::{ark_batch_mul, ark_batch_mul_par, ark_inner_product},
    sub_vec_bn254, sub_vec_par_bn254, utils,
};
use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_std::Zero;
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Duration;

use crate::ark::*;
use crate::{
    add_vec_bn254, add_vec_par_bn254, inner_product_bn254, inner_product_par_bn254, mul_vec_bn254,
    mul_vec_par_bn254, sum_vec_bn254, sum_vec_par_bn254
};

pub fn bench_mul_single_threaded() -> (Duration, Duration) {
    const NUM_OPS: usize = 8 * 4_000_000;

    let x: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let y: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let z: Vec<Fr> = vec![Fr::zero(); NUM_OPS];

    let ark_x: Vec<Fr> = x.clone();
    let ark_y: Vec<Fr> = y.clone();
    let mut ark_z: Vec<Fr> = z.clone();

    let start = std::time::Instant::now();
    ark_batch_mul(&ark_x, &ark_y, ark_z.as_mut_slice());
    let duration_ark = start.elapsed();
    black_box(&ark_z);
    drop((ark_x, ark_y, ark_z));

    let simd_x: Vec<Fr> = x;
    let simd_y: Vec<Fr> = y;
    let mut simd_z: Vec<Fr> = z;

    let start = std::time::Instant::now();
    mul_vec_bn254(&simd_x, &simd_y, simd_z.as_mut_slice());
    let duration_simd = start.elapsed();
    black_box(&simd_z);

    (duration_ark, duration_simd)
}

pub fn bench_mul_multi_threaded() -> (Duration, Duration) {
    const NUM_OPS: usize = 8 * 4_000_000;

    let x: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let y: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let z: Vec<Fr> = vec![Fr::zero(); NUM_OPS];

    let ark_x: Vec<Fr> = x.clone();
    let ark_y: Vec<Fr> = y.clone();
    let mut ark_z: Vec<Fr> = z.clone();

    let start = std::time::Instant::now();
    ark_batch_mul_par(&ark_x, &ark_y, ark_z.as_mut_slice());
    let duration_ark = start.elapsed();
    black_box(&ark_z);
    drop((ark_x, ark_y, ark_z));

    let simd_x: Vec<Fr> = x;
    let simd_y: Vec<Fr> = y;
    let mut simd_z: Vec<Fr> = z;

    let start = std::time::Instant::now();
    mul_vec_par_bn254(&simd_x, &simd_y, simd_z.as_mut_slice());
    let duration_simd = start.elapsed();
    black_box(&simd_z);

    (duration_ark, duration_simd)
}

pub fn bench_inner_product_single_threaded() -> (Duration, Duration) {
    const NUM_OPS: usize = 32 * 4_000_000;

    let x: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let y: Vec<Fr> = utils::rand_vec(NUM_OPS);

    let start = std::time::Instant::now();
    let ark_z = ark_inner_product(&x, &y);
    let duration_ark = start.elapsed();
    black_box(&ark_z);

    let start = std::time::Instant::now();
    let simd_z = inner_product_bn254(&x, &y);
    let duration_simd = start.elapsed();
    black_box(&simd_z);

    assert_eq!(ark_z, simd_z);

    (duration_ark, duration_simd)
}

pub fn bench_inner_product_multi_threaded() -> (Duration, Duration) {
    const NUM_OPS: usize = 32 * 4_000_000;

    let x: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let y: Vec<Fr> = utils::rand_vec(NUM_OPS);

    let start = std::time::Instant::now();
    let ark_z = ark_inner_product_par(&x, &y);
    let duration_ark = start.elapsed();
    black_box(&ark_z);

    let start = std::time::Instant::now();
    let simd_z = inner_product_par_bn254(&x, &y);
    let duration_simd = start.elapsed();
    black_box(&simd_z);

    assert_eq!(ark_z, simd_z);

    (duration_ark, duration_simd)
}

pub fn bench_add_single_threaded() -> (Duration, Duration) {
    const NUM_OPS: usize = 8 * 4_000_000;

    let x: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let y: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let mut z: Vec<Fr> = vec![Fr::zero(); NUM_OPS];

    let start = std::time::Instant::now();
    ark_batch_add(&x, &y, &mut z);
    let duration_ark = start.elapsed();
    black_box(&z);

    let mut simd_z: Vec<Fr> = vec![Fr::zero(); NUM_OPS];

    let start = std::time::Instant::now();
    add_vec_bn254(&x, &y, &mut simd_z);
    let duration_simd = start.elapsed();
    black_box(&simd_z);

    assert_eq!(z, simd_z);

    (duration_ark, duration_simd)
}

pub fn bench_add_multi_threaded() -> (Duration, Duration) {
    const NUM_OPS: usize = 8 * 4_000_000;

    let x: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let y: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let mut z: Vec<Fr> = vec![Fr::zero(); NUM_OPS];

    let start = std::time::Instant::now();
    ark_batch_add_par(&x, &y, &mut z);
    let duration_ark = start.elapsed();
    black_box(&z);

    let mut simd_z: Vec<Fr> = vec![Fr::zero(); NUM_OPS];

    let start = std::time::Instant::now();
    add_vec_par_bn254(&x, &y, &mut simd_z);
    let duration_simd = start.elapsed();
    black_box(&simd_z);

    assert_eq!(z, simd_z);

    (duration_ark, duration_simd)
}

pub fn bench_sub_single_threaded() -> (Duration, Duration) {
    const NUM_OPS: usize = 8 * 4_000_000;

    let x: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let y: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let z: Vec<Fr> = vec![Fr::zero(); NUM_OPS];

    let ark_x: Vec<Fr> = x.clone();
    let ark_y: Vec<Fr> = y.clone();
    let mut ark_z: Vec<Fr> = z.clone();

    let start = std::time::Instant::now();
    ark_batch_sub(&ark_x, &ark_y, ark_z.as_mut_slice());
    let duration_ark = start.elapsed();
    black_box(&ark_z);

    let simd_x: Vec<Fr> = x;
    let simd_y: Vec<Fr> = y;
    let mut simd_z: Vec<Fr> = z;

    let start = std::time::Instant::now();
    sub_vec_bn254(&simd_x, &simd_y, simd_z.as_mut_slice());
    let duration_simd = start.elapsed();
    black_box(&simd_z);

    assert_eq!(ark_z, simd_z);

    (duration_ark, duration_simd)
}

pub fn bench_sub_multi_threaded() -> (Duration, Duration) {
    const NUM_OPS: usize = 8 * 4_000_000;

    let x: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let y: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let mut z: Vec<Fr> = vec![Fr::zero(); NUM_OPS];

    let start = std::time::Instant::now();
    ark_batch_sub_par(&x, &y, &mut z);
    let duration_ark = start.elapsed();
    black_box(&z);

    let mut simd_z: Vec<Fr> = vec![Fr::zero(); NUM_OPS];

    let start = std::time::Instant::now();
    sub_vec_par_bn254(&x, &y, &mut simd_z);
    let duration_simd = start.elapsed();
    black_box(&simd_z);

    assert_eq!(z, simd_z);

    (duration_ark, duration_simd)
}

pub fn bench_sum_single_threaded() -> (Duration, Duration) {
    const NUM_OPS: usize = 8 * 4_000_000;

    let x: Vec<Fr> = utils::rand_vec(NUM_OPS);

    let start = std::time::Instant::now();
    let ark_sum = x.iter().sum::<Fr>();
    let duration_ark = start.elapsed();
    black_box(&ark_sum);

    let start = std::time::Instant::now();
    let simd_sum = sum_vec_bn254(&x);
    let duration_simd = start.elapsed();
    black_box(&simd_sum);

    assert_eq!(ark_sum, simd_sum);

    (duration_ark, duration_simd)
}

pub fn bench_sum_multi_threaded() -> (Duration, Duration) {
    const NUM_OPS: usize = 8 * 4_000_000;

    let x: Vec<Fr> = utils::rand_vec(NUM_OPS);

    let start = std::time::Instant::now();
    let ark_sum = x.par_iter().sum::<Fr>();
    let duration_ark = start.elapsed();
    black_box(&ark_sum);

    let start = std::time::Instant::now();
    let simd_sum = sum_vec_par_bn254(&x);
    let duration_simd = start.elapsed();
    black_box(&simd_sum);

    assert_eq!(ark_sum, simd_sum);

    (duration_ark, duration_simd)
}

pub fn benchmark() {
    let (ark_mul_single, simd_mul_single) = bench_mul_single_threaded();
    let (ark_mul_multi, simd_mul_multi) = bench_mul_multi_threaded();
    let (ark_inner_single, simd_inner_single) = bench_inner_product_single_threaded();
    let (ark_inner_multi, simd_inner_multi) = bench_inner_product_multi_threaded();
    let (ark_add_single, simd_add_single) = bench_add_single_threaded();
    let (ark_add_multi, simd_add_multi) = bench_add_multi_threaded();
    let (ark_sub_single, simd_sub_single) = bench_sub_single_threaded();
    let (ark_sub_multi, simd_sub_multi) = bench_sub_multi_threaded();
    let (ark_sum_single, simd_sum_single) = bench_sum_single_threaded();
    let (ark_sum_multi, simd_sum_multi) = bench_sum_multi_threaded();

    println!("| Benchmark | Arkworks | AVX-512 | Speedup |");
    println!("|-----------|----------|---------|---------|");
    println!("| Mul Single-Threaded | {:?} | {:?} | {:.2}x |", ark_mul_single, simd_mul_single, ark_mul_single.as_secs_f64() / simd_mul_single.as_secs_f64());
    println!("| Mul Multi-Threaded | {:?} | {:?} | {:.2}x |", ark_mul_multi, simd_mul_multi, ark_mul_multi.as_secs_f64() / simd_mul_multi.as_secs_f64());
    println!("| Inner Product Single-Threaded | {:?} | {:?} | {:.2}x |", ark_inner_single, simd_inner_single, ark_inner_single.as_secs_f64() / simd_inner_single.as_secs_f64());
    println!("| Inner Product Multi-Threaded | {:?} | {:?} | {:.2}x |", ark_inner_multi, simd_inner_multi, ark_inner_multi.as_secs_f64() / simd_inner_multi.as_secs_f64());
    println!("| Add Single-Threaded | {:?} | {:?} | {:.2}x |", ark_add_single, simd_add_single, ark_add_single.as_secs_f64() / simd_add_single.as_secs_f64());
    println!("| Add Multi-Threaded | {:?} | {:?} | {:.2}x |", ark_add_multi, simd_add_multi, ark_add_multi.as_secs_f64() / simd_add_multi.as_secs_f64());
    println!("| Sub Single-Threaded | {:?} | {:?} | {:.2}x |", ark_sub_single, simd_sub_single, ark_sub_single.as_secs_f64() / simd_sub_single.as_secs_f64());
    println!("| Sub Multi-Threaded | {:?} | {:?} | {:.2}x |", ark_sub_multi, simd_sub_multi, ark_sub_multi.as_secs_f64() / simd_sub_multi.as_secs_f64());
    println!("| Sum Single-Threaded | {:?} | {:?} | {:.2}x |", ark_sum_single, simd_sum_single, ark_sum_single.as_secs_f64() / simd_sum_single.as_secs_f64());
    println!("| Sum Multi-Threaded | {:?} | {:?} | {:.2}x |", ark_sum_multi, simd_sum_multi, ark_sum_multi.as_secs_f64() / simd_sum_multi.as_secs_f64());
}
