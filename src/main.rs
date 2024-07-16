use std::hint::black_box;
use ark_bn254::Fq;
use ark_ff::PrimeField;
use ark_std::Zero;
use big_field_bench::{read_vec_from_file, save_vec_to_file};
use rayon::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;


extern "C" {
    fn avx512montmul(z: *mut u32, x: *const u32, y: *const u32, m: *const u64);
}

// Base Field
const FP: [u64; 9] = [
    0xD87CFD47, 0x3C208C16, 0x6871CA8D, 0x97816A91, 0x8181585D, 0xB85045B6, 0xE131A029, 0x30644E72, 0xE4866389,
];

fn main() {
    test_memory();
}

/// Version serializes to a file after generating. Takes substantially longer to run due to disk ops
/// but potentially more accurate given both benchmarks have the same amount of RAM available.
fn test_file() {
    const NUM_OPS: usize = 8 * 6_000_000;

    let x: Vec<Fq> = rand_vec(NUM_OPS);
    let y: Vec<Fq> = rand_vec(NUM_OPS);
    let z: Vec<Fq> = vec![Fq::zero(); NUM_OPS];

    let total_size = std::mem::size_of_val(&x[..]) + std::mem::size_of_val(&y[..]) + std::mem::size_of_val(&z[..]);
    println!("| x, y, z | = {} GB", total_size as f64 / (1024.0 * 1024.0 * 1024.0));

    save_vec_to_file("cache/x.ff", &x);
    save_vec_to_file("cache/y.ff", &y);
    save_vec_to_file("cache/z.ff", &z);
    drop((x,y,z));

    let ark_x: Vec<Fq> = read_vec_from_file("cache/x.ff");
    let ark_y: Vec<Fq> = read_vec_from_file("cache/y.ff");
    let mut ark_z: Vec<Fq> = read_vec_from_file("cache/z.ff");
    let start = std::time::Instant::now();
    ark_batch_mul(&ark_x, &ark_y, ark_z.as_mut_slice());
    let duration = start.elapsed();
    println!("ark_batch_mul took: {:?}", duration);
    black_box(&ark_z);
    drop((ark_x, ark_y, ark_z));

    let simd_x: Vec<Fq> = read_vec_from_file("cache/x.ff");
    let simd_y: Vec<Fq> = read_vec_from_file("cache/y.ff");
    let mut simd_z: Vec<Fq> = read_vec_from_file("cache/z.ff");
    let start = std::time::Instant::now();
    simd_batch_mul(&simd_x, &simd_y, simd_z.as_mut_slice());
    let duration = start.elapsed();
    println!("simd_batch_mul took: {:?}", duration);
    black_box(&simd_z);

    // TODO(sragss): Currently this fails most of the time.
    // assert_eq!(simd_z, ark_z); 

}

/// Version keeps the values in memory.
fn test_memory() {
    const NUM_OPS: usize = 8 * 4_000_000;

    let x: Vec<Fq> = rand_vec(NUM_OPS);
    let y: Vec<Fq> = rand_vec(NUM_OPS);
    let z: Vec<Fq> = vec![Fq::zero(); NUM_OPS];

    let total_size = std::mem::size_of_val(&x[..]) + std::mem::size_of_val(&y[..]) + std::mem::size_of_val(&z[..]);
    println!("| x, y, z | = {} GB", total_size as f64 / (1024.0 * 1024.0 * 1024.0));


    let ark_x: Vec<Fq> = x.clone();
    let ark_y: Vec<Fq> = y.clone();
    let mut ark_z: Vec<Fq> = z.clone();
    let start = std::time::Instant::now();
    ark_batch_mul(&ark_x, &ark_y, ark_z.as_mut_slice());
    let duration = start.elapsed();
    println!("ark_batch_mul took: {:?}", duration);
    black_box(&ark_z);
    drop((ark_x, ark_y, ark_z));

    let simd_x: Vec<Fq> = x;
    let simd_y: Vec<Fq> = y;
    let mut simd_z: Vec<Fq> = z; 
    let start = std::time::Instant::now();
    simd_batch_mul(&simd_x, &simd_y, simd_z.as_mut_slice());
    let duration = start.elapsed();
    println!("simd_batch_mul took: {:?}", duration);
    black_box(&simd_z);

    // TODO(sragss): Currently this fails most of the time.
    // assert_eq!(simd_z, ark_z); 
}
fn ark_batch_mul<F: PrimeField>(x: &[F], y: &[F], z: &mut [F]) {
    let len = x.len();
    assert_eq!(len, y.len());
    assert_eq!(len, z.len());

    for i in 0..len {
        z[i] = x[i] * y[i];
    }
}


fn simd_batch_mul(x: &[Fq], y: &[Fq], z: &mut [Fq]) {
    let len = x.len();
    assert_eq!(len, y.len());
    assert_eq!(len, z.len());

    const BATCH_SIZE: usize = 8;

    let simd_x = x.as_ptr() as *const u32;
    let simd_y = y.as_ptr() as *const u32;
    let simd_z = z.as_mut_ptr() as *mut u32;

    for i in 0..len / BATCH_SIZE {
        unsafe {
            let update = i * 8 * BATCH_SIZE;
            avx512montmul(
                simd_z.add(update),
                simd_x.add(update),
                simd_y.add(update),
                FP.as_ptr()
            );
        }
    }
}


fn rand_vec<F: PrimeField>(size: usize) -> Vec<F> {
    (0..size).into_par_iter().map_init(|| ChaCha8Rng::from_entropy(), |rng, _| F::rand(rng)).collect()
}
