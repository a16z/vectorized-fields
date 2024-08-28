use crate::utils;
use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_std::Zero;
use rayon::prelude::*;
use std::hint::black_box;

use crate::{inner_product_bn254, inner_product_par_bn254, mul_vec_bn254, mul_vec_par_bn254};

fn ark_batch_mul<F: PrimeField>(x: &[F], y: &[F], z: &mut [F]) {
    let len = x.len();
    assert_eq!(len, y.len());
    assert_eq!(len, z.len());

    for i in 0..len {
        z[i] = x[i] * y[i];
    }
}

fn ark_batch_mul_par<F: PrimeField>(x: &[F], y: &[F], z: &mut [F]) {
    let len = x.len();
    assert_eq!(len, y.len());
    assert_eq!(len, z.len());

    x.par_iter()
        .zip(y.par_iter())
        .zip(z.par_iter_mut())
        .for_each(|((xi, yi), zi)| {
            *zi = *xi * *yi;
        });
}

fn ark_inner_product<F: PrimeField>(x: &[F], y: &[F]) -> F {
    let len = x.len();
    assert_eq!(len, y.len());

    let chunk_size = len / rayon::current_num_threads();

    x.par_chunks(chunk_size)
        .zip(y.par_chunks(chunk_size))
        .map(|(chunk_x, chunk_y)| {
            chunk_x
                .iter()
                .zip(chunk_y.iter())
                .map(|(&x, &y)| x * y)
                .sum::<F>()
        })
        .sum()
}

pub fn bench_mul_single_threaded() {
    const NUM_OPS: usize = 8 * 4_000_000;

    let x: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let y: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let z: Vec<Fr> = vec![Fr::zero(); NUM_OPS];

    let total_size = std::mem::size_of_val(&x[..])
        + std::mem::size_of_val(&y[..])
        + std::mem::size_of_val(&z[..]);
    println!("\nBench: Single-threaded multiplication");
    println!(
        "| x + y + z | = {} GB",
        total_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    let ark_x: Vec<Fr> = x.clone();
    let ark_y: Vec<Fr> = y.clone();
    let mut ark_z: Vec<Fr> = z.clone();

    let start = std::time::Instant::now();
    ark_batch_mul(&ark_x, &ark_y, ark_z.as_mut_slice());
    let duration_ark = start.elapsed();
    println!("Arkworks: {:?}", duration_ark);
    black_box(&ark_z);
    drop((ark_x, ark_y, ark_z));

    let simd_x: Vec<Fr> = x;
    let simd_y: Vec<Fr> = y;
    let mut simd_z: Vec<Fr> = z;

    let start = std::time::Instant::now();
    mul_vec_bn254(&simd_x, &simd_y, simd_z.as_mut_slice());
    let duration_simd = start.elapsed();
    println!("AVX-512: {:?}", duration_simd);
    black_box(&simd_z);

    let ark_duration = duration_ark.as_secs_f64();
    let simd_duration = duration_simd.as_secs_f64();
    let speedup = ark_duration / simd_duration;
    println!("AVX-512 {:.2}x speedup", speedup);
}

pub fn bench_mul_multi_threaded() {
    const NUM_OPS: usize = 8 * 4_000_000;

    let x: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let y: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let z: Vec<Fr> = vec![Fr::zero(); NUM_OPS];

    let total_size = std::mem::size_of_val(&x[..])
        + std::mem::size_of_val(&y[..])
        + std::mem::size_of_val(&z[..]);
    println!("\nBench: Multi-threaded multiplication");
    println!(
        "| x + y + z | = {} GB",
        total_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    let ark_x: Vec<Fr> = x.clone();
    let ark_y: Vec<Fr> = y.clone();
    let mut ark_z: Vec<Fr> = z.clone();

    let start = std::time::Instant::now();
    ark_batch_mul_par(&ark_x, &ark_y, ark_z.as_mut_slice());
    let duration_ark = start.elapsed();
    println!("Arkworks: {:?}", duration_ark);
    black_box(&ark_z);
    drop((ark_x, ark_y, ark_z));

    let simd_x: Vec<Fr> = x;
    let simd_y: Vec<Fr> = y;
    let mut simd_z: Vec<Fr> = z;

    let start = std::time::Instant::now();
    mul_vec_par_bn254(&simd_x, &simd_y, simd_z.as_mut_slice());
    let duration_simd = start.elapsed();
    println!("AVX-512: {:?}", duration_simd);
    black_box(&simd_z);

    let ark_duration = duration_ark.as_secs_f64();
    let simd_duration = duration_simd.as_secs_f64();
    let speedup = ark_duration / simd_duration;
    println!("AVX-512 {:.2}x speedup", speedup);
}

pub fn bench_inner_product_multi_threaded() {
    const NUM_OPS: usize = 32 * 4_000_000;

    let x: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let y: Vec<Fr> = utils::rand_vec(NUM_OPS);

    let total_size = std::mem::size_of_val(&x[..]) + std::mem::size_of_val(&y[..]);
    println!("\nBench: Multi-threaded inner product");
    println!(
        "| x + y | = {} GB",
        total_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    let start = std::time::Instant::now();
    let ark_z = ark_inner_product(&x, &y);
    let duration_ark = start.elapsed();
    println!("Arkworks: {:?}", duration_ark);
    black_box(&ark_z);

    let start = std::time::Instant::now();
    let simd_z = inner_product_par_bn254(&x, &y);
    let duration_simd = start.elapsed();
    println!("AVX-512: {:?}", duration_simd);
    black_box(&simd_z);

    let ark_duration = duration_ark.as_secs_f64();
    let simd_duration = duration_simd.as_secs_f64();
    let speedup = ark_duration / simd_duration;
    println!("AVX-512 {:.2}x speedup", speedup);

    assert_eq!(ark_z, simd_z);
}
