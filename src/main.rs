use std::hint::black_box;
use ark_bn254::Fq;
use ark_ff::PrimeField;
use ark_std::Zero;
use vectorized_fields::{read_vec_from_file, save_vec_to_file};
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
    // test_file();
    // test_memory();
    test_memory_par();
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

/// Version keeps the values in memory.
fn test_memory_par() {
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
    ark_batch_mul_par(&ark_x, &ark_y, ark_z.as_mut_slice());
    let duration = start.elapsed();
    println!("ark_batch_mul took: {:?}", duration);
    black_box(&ark_z);
    drop((ark_x, ark_y, ark_z));

    let simd_x: Vec<Fq> = x;
    let simd_y: Vec<Fq> = y;
    let mut simd_z: Vec<Fq> = z; 
    let start = std::time::Instant::now();
    simd_batch_mul_par(&simd_x, &simd_y, simd_z.as_mut_slice());
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

fn simd_batch_mul_par(x: &[Fq], y: &[Fq], z: &mut [Fq]) {
    let len = x.len();
    assert_eq!(len, y.len());
    assert_eq!(len, z.len());

    const BATCH_SIZE: usize = 8;

    let num_threads = rayon::current_num_threads();

    let mut chunk_size = len / num_threads;
    chunk_size = chunk_size + (BATCH_SIZE - chunk_size % BATCH_SIZE); // round to nearest multiple of BATCH_SIZE

    let mut x_chunks: Vec<&[Fq]> = Vec::with_capacity(num_threads);
    let mut y_chunks: Vec<&[Fq]> = Vec::with_capacity(num_threads);
    let mut z_chunks: Vec<&mut [Fq]> = Vec::with_capacity(num_threads);

    let mut x_remainder = x;
    let mut y_remainder = y;
    let mut z_remainder = z;
    for _ in 0..(num_threads - 1) {
        let x_split = x_remainder.split_at(chunk_size);
        let y_split = y_remainder.split_at(chunk_size);
        let z_split = z_remainder.split_at_mut(chunk_size);

        x_chunks.push(x_split.0);
        x_remainder = x_split.1;
        y_chunks.push(y_split.0);
        y_remainder = y_split.1;
        z_chunks.push(z_split.0);
        z_remainder = z_split.1;
    }
    x_chunks.push(x_remainder);
    y_chunks.push(y_remainder);
    z_chunks.push(z_remainder);

    x_chunks.par_iter()
        .zip(y_chunks.par_iter())
        .zip(z_chunks.par_iter_mut())
        .for_each(|((xi, yi), zi)| {
            simd_batch_mul(xi, yi, zi);
        });
}


fn rand_vec<F: PrimeField>(size: usize) -> Vec<F> {
    (0..size).into_par_iter().map_init(|| ChaCha8Rng::from_entropy(), |rng, _| F::rand(rng)).collect()
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_parity() {
        const NUM_OPS: usize = 10_000;
        let x: Vec<Fq> = rand_vec(NUM_OPS);
        let y: Vec<Fq> = rand_vec(NUM_OPS);
        let mut z: Vec<Fq> = vec![Fq::zero(); NUM_OPS];

        let x_par = x.clone();
        let y_par = y.clone();
        let mut z_par = z.clone();

        simd_batch_mul(&x, &y, &mut z);
        simd_batch_mul_par(&x_par, &y_par, &mut z_par);
        assert_eq!(z, z_par);
    }
}