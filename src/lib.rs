use std::{
    fs::{File, OpenOptions},
    io::{BufReader, BufWriter, Read},
};

use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_std::Zero;

use rayon::prelude::*;

mod assembly;
pub mod benchmarks;
mod constants;
mod utils;

use crate::assembly::{modip256_mont, modmul256_mont};

pub fn mul_vec_bn254(x: &[Fr], y: &[Fr], z: &mut [Fr]) {
    let len = x.len();
    assert_eq!(len, y.len());
    assert_eq!(len, z.len());

    let simd_x = x.as_ptr() as *const u64;
    let simd_y = y.as_ptr() as *const u64;
    let simd_z = z.as_mut_ptr() as *mut u64;

    unsafe {
        modmul256_mont(
            simd_z,
            simd_x,
            simd_y,
            len as u64,
            constants::BN254_FR.as_ptr(),
        );
    }
}

pub fn mul_vec_par_bn254(x: &[Fr], y: &[Fr], z: &mut [Fr]) {
    let len = x.len();
    assert_eq!(y.len(), len);
    assert_eq!(z.len(), len);

    let num_threads = rayon::current_num_threads();
    let chunk_size = len / num_threads;

    x.par_chunks(chunk_size)
        .zip(y.par_chunks(chunk_size))
        .zip(z.par_chunks_mut(chunk_size))
        .for_each(|((xi, yi), zi)| {
            mul_vec_bn254(xi, yi, zi);
        });
}

pub fn inner_product_bn254(x: &[Fr], y: &[Fr]) -> Fr {
    assert_eq!(x.len(), y.len());

    let simd_x = x.as_ptr() as *const u64;
    let simd_y = y.as_ptr() as *const u64;
    let mut collect = Fr::zero();
    let mut simd_z = collect.0 .0.as_mut_ptr() as *mut u64;

    let simd_len: u32 = x.len().try_into().unwrap();

    unsafe {
        modip256_mont(
            simd_z,
            simd_x,
            simd_y,
            simd_len,
            constants::BN254_FR.as_ptr(),
        );
    }
    collect
}

pub fn inner_product_par_bn254(x: &[Fr], y: &[Fr]) -> Fr {
    assert_eq!(x.len(), y.len());
    let chunk_size = x.len() / rayon::current_num_threads();

    x.par_chunks(chunk_size)
        .zip(y.par_chunks(chunk_size))
        .map(|(chunk_x, chunk_y)| inner_product_bn254(chunk_x, chunk_y))
        .sum::<Fr>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::rand_vec;
    use ark_std::Zero;

    #[test]
    fn parallel_parity_vec_mul() {
        const NUM_OPS: usize = 10_000;
        let x: Vec<Fr> = rand_vec(NUM_OPS);
        let y: Vec<Fr> = rand_vec(NUM_OPS);
        let mut z: Vec<Fr> = vec![Fr::zero(); NUM_OPS];

        let x_par = x.clone();
        let y_par = y.clone();
        let mut z_par = z.clone();

        mul_vec_bn254(&x, &y, &mut z);
        mul_vec_par_bn254(&x_par, &y_par, &mut z_par);
        assert_eq!(z, z_par);
    }

    #[test]
    fn parallel_parity_inner_product() {
        const NUM_OPS: usize = 10_000;
        let x: Vec<Fr> = rand_vec(NUM_OPS);
        let y: Vec<Fr> = rand_vec(NUM_OPS);

        let z = inner_product_bn254(&x, &y);
        let par_z = inner_product_par_bn254(&x, &y);
        assert_eq!(z, par_z);
    }
}
