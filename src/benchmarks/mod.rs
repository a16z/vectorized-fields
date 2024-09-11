use crate::{
    ark::{ark_batch_mul, ark_batch_mul_par, ark_inner_product},
    sub_vec_bn254, sub_vec_par_bn254, utils,
};
use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_std::Zero;
use rayon::prelude::*;
use std::hint::black_box;

use crate::ark::*;
use crate::{
    add_vec_bn254, add_vec_par_bn254, inner_product_bn254, inner_product_par_bn254, mul_vec_bn254,
    mul_vec_par_bn254,
};

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

pub fn bench_inner_product_single_threaded() {
    const NUM_OPS: usize = 32 * 4_000_000;

    let x: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let y: Vec<Fr> = utils::rand_vec(NUM_OPS);

    let total_size = std::mem::size_of_val(&x[..]) + std::mem::size_of_val(&y[..]);
    println!("\nBench: Single-threaded inner product");
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
    let simd_z = inner_product_bn254(&x, &y);
    let duration_simd = start.elapsed();
    println!("AVX-512: {:?}", duration_simd);
    black_box(&simd_z);

    let ark_duration = duration_ark.as_secs_f64();
    let simd_duration = duration_simd.as_secs_f64();
    let speedup = ark_duration / simd_duration;
    println!("AVX-512 {:.2}x speedup", speedup);

    assert_eq!(ark_z, simd_z);
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
    let ark_z = ark_inner_product_par(&x, &y);
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

pub fn bench_add_single_threaded() {
    const NUM_OPS: usize = 8 * 4_000_000;

    let x: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let y: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let mut z: Vec<Fr> = vec![Fr::zero(); NUM_OPS];

    let total_size = std::mem::size_of_val(&x[..])
        + std::mem::size_of_val(&y[..])
        + std::mem::size_of_val(&z[..]);
    println!("\nBench: Single-threaded vector addition");
    println!(
        "| x + y + z | = {} GB",
        total_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    let start = std::time::Instant::now();
    ark_batch_add(&x, &y, &mut z);
    let duration_ark = start.elapsed();
    println!("Arkworks: {:?}", duration_ark);
    black_box(&z);

    let mut simd_z: Vec<Fr> = vec![Fr::zero(); NUM_OPS];

    let start = std::time::Instant::now();
    add_vec_bn254(&x, &y, &mut simd_z);
    let duration_simd = start.elapsed();
    println!("AVX-512: {:?}", duration_simd);
    black_box(&simd_z);

    let ark_duration = duration_ark.as_secs_f64();
    let simd_duration = duration_simd.as_secs_f64();
    let speedup = ark_duration / simd_duration;
    println!("AVX-512 {:.2}x speedup", speedup);

    assert_eq!(z, simd_z);
}

pub fn bench_add_multi_threaded() {
    const NUM_OPS: usize = 8 * 4_000_000;

    let x: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let y: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let mut z: Vec<Fr> = vec![Fr::zero(); NUM_OPS];

    let total_size = std::mem::size_of_val(&x[..])
        + std::mem::size_of_val(&y[..])
        + std::mem::size_of_val(&z[..]);
    println!("\nBench: Multi-threaded vector addition");
    println!(
        "| x + y + z | = {} GB",
        total_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    let start = std::time::Instant::now();
    ark_batch_add_par(&x, &y, &mut z);
    let duration_ark = start.elapsed();
    println!("Arkworks: {:?}", duration_ark);
    black_box(&z);

    let mut simd_z: Vec<Fr> = vec![Fr::zero(); NUM_OPS];

    let start = std::time::Instant::now();
    add_vec_par_bn254(&x, &y, &mut simd_z);
    let duration_simd = start.elapsed();
    println!("AVX-512: {:?}", duration_simd);
    black_box(&simd_z);

    let ark_duration = duration_ark.as_secs_f64();
    let simd_duration = duration_simd.as_secs_f64();
    let speedup = ark_duration / simd_duration;
    println!("AVX-512 {:.2}x speedup", speedup);

    assert_eq!(z, simd_z);
}

pub fn bench_sub_single_threaded() {
    const NUM_OPS: usize = 8 * 4_000_000;

    let x: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let y: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let z: Vec<Fr> = vec![Fr::zero(); NUM_OPS];

    let total_size = std::mem::size_of_val(&x[..])
        + std::mem::size_of_val(&y[..])
        + std::mem::size_of_val(&z[..]);
    println!("\nBench: Single-threaded subtraction");
    println!(
        "| x + y + z | = {} GB",
        total_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    let ark_x: Vec<Fr> = x.clone();
    let ark_y: Vec<Fr> = y.clone();
    let mut ark_z: Vec<Fr> = z.clone();

    let start = std::time::Instant::now();
    ark_batch_sub(&ark_x, &ark_y, ark_z.as_mut_slice());
    let duration_ark = start.elapsed();
    println!("Arkworks: {:?}", duration_ark);
    black_box(&ark_z);

    let simd_x: Vec<Fr> = x;
    let simd_y: Vec<Fr> = y;
    let mut simd_z: Vec<Fr> = z;

    let start = std::time::Instant::now();
    sub_vec_bn254(&simd_x, &simd_y, simd_z.as_mut_slice());
    let duration_simd = start.elapsed();
    println!("AVX-512: {:?}", duration_simd);
    black_box(&simd_z);

    let ark_duration = duration_ark.as_secs_f64();
    let simd_duration = duration_simd.as_secs_f64();
    let speedup = ark_duration / simd_duration;
    println!("AVX-512 {:.2}x speedup", speedup);

    assert_eq!(ark_z, simd_z);
}

pub fn bench_sub_multi_threaded() {
    const NUM_OPS: usize = 8 * 4_000_000;

    let x: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let y: Vec<Fr> = utils::rand_vec(NUM_OPS);
    let mut z: Vec<Fr> = vec![Fr::zero(); NUM_OPS];

    let total_size = std::mem::size_of_val(&x[..])
        + std::mem::size_of_val(&y[..])
        + std::mem::size_of_val(&z[..]);
    println!("\nBench: Multi-threaded subtraction");
    println!(
        "| x + y + z | = {} GB",
        total_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    let start = std::time::Instant::now();
    ark_batch_sub_par(&x, &y, &mut z);
    let duration_ark = start.elapsed();
    println!("Arkworks: {:?}", duration_ark);
    black_box(&z);

    let mut simd_z: Vec<Fr> = vec![Fr::zero(); NUM_OPS];

    let start = std::time::Instant::now();
    sub_vec_par_bn254(&x, &y, &mut simd_z);
    let duration_simd = start.elapsed();
    println!("AVX-512: {:?}", duration_simd);
    black_box(&simd_z);

    let ark_duration = duration_ark.as_secs_f64();
    let simd_duration = duration_simd.as_secs_f64();
    let speedup = ark_duration / simd_duration;
    println!("AVX-512 {:.2}x speedup", speedup);

    assert_eq!(z, simd_z);
}
