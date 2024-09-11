use std::{
    fs::{File, OpenOptions},
    io::{BufReader, BufWriter, Read},
};

use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_std::Zero;

use rayon::prelude::*;

mod ark;
mod assembly;
pub mod benchmarks;
mod constants;
mod utils;

use crate::assembly::{modadd256, modip256_mont, modmul256_mont, modsub256, modsum256};

const PAR_CHUNK_SIZE: usize = 128;

pub fn add_vec_bn254(x: &[Fr], y: &[Fr], z: &mut [Fr]) {
    assert_eq!(x.len(), y.len());

    unsafe {
        modadd256(
            z.as_mut_ptr() as *mut u64,
            x.as_ptr() as *const u64,
            y.as_ptr() as *const u64,
            x.len() as u64,
            constants::BN254_FR.as_ptr(),
        )
    }
}

pub fn add_vec_par_bn254(x: &[Fr], y: &[Fr], z: &mut [Fr]) {
    let len = x.len();
    assert_eq!(y.len(), len);
    assert_eq!(z.len(), len);

    let chunk_size = std::cmp::min(len, PAR_CHUNK_SIZE);

    x.par_chunks(chunk_size)
        .zip(y.par_chunks(chunk_size))
        .zip(z.par_chunks_mut(chunk_size))
        .for_each(|((x_chunk, y_chunk), z_chunk)| {
            add_vec_bn254(x_chunk, y_chunk, z_chunk);
        });
}

pub fn add_vec_inplace_bn254(x: &mut [Fr], y: &[Fr]) {
    assert_eq!(x.len(), y.len());

    unsafe {
        modadd256(
            x.as_mut_ptr() as *mut u64,
            x.as_ptr() as *const u64,
            y.as_ptr() as *const u64,
            x.len() as u64,
            constants::BN254_FR.as_ptr(),
        )
    }
}

pub fn sub_vec_bn254(x: &[Fr], y: &[Fr], z: &mut [Fr]) {
    assert_eq!(x.len(), y.len());

    unsafe {
        modsub256(
            z.as_mut_ptr() as *mut u64,
            x.as_ptr() as *const u64,
            y.as_ptr() as *const u64,
            x.len() as u64,
            constants::BN254_FR.as_ptr(),
        )
    }
}

pub fn sub_vec_par_bn254(x: &[Fr], y: &[Fr], z: &mut [Fr]) {
    let len = x.len();
    assert_eq!(y.len(), len);
    assert_eq!(z.len(), len);

    let chunk_size = std::cmp::min(len, PAR_CHUNK_SIZE);

    x.par_chunks(chunk_size)
        .zip(y.par_chunks(chunk_size))
        .zip(z.par_chunks_mut(chunk_size))
        .for_each(|((x_chunk, y_chunk), z_chunk)| {
            sub_vec_bn254(x_chunk, y_chunk, z_chunk);
        });
}

pub fn sub_vec_inplace_bn254(x: &mut [Fr], y: &[Fr]) {
    assert_eq!(x.len(), y.len());

    unsafe {
        modsub256(
            x.as_mut_ptr() as *mut u64,
            x.as_ptr() as *const u64,
            y.as_ptr() as *const u64,
            x.len() as u64,
            constants::BN254_FR.as_ptr(),
        )
    }
}

pub fn sum_vec_bn254(x: &[Fr]) -> Fr {
    let mut result = Fr::zero();
    let simd_x = x.as_ptr() as *const u64;
    let simd_result = result.0 .0.as_mut_ptr() as *mut u64;

    unsafe {
        modsum256(
            simd_result,
            simd_x,
            x.len() as u32,
            constants::BN254_FR.as_ptr(),
        );
    }

    result
}

pub fn sum_vec_par_bn254(x: &[Fr]) -> Fr {
    let chunk_size = std::cmp::min(x.len(), PAR_CHUNK_SIZE);

    x.par_chunks(chunk_size)
        .map(|chunk_x| sum_vec_bn254(chunk_x))
        .sum::<Fr>()
}

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

    let chunk_size = std::cmp::min(len, PAR_CHUNK_SIZE);

    x.par_chunks(chunk_size)
        .zip(y.par_chunks(chunk_size))
        .zip(z.par_chunks_mut(chunk_size))
        .for_each(|((xi, yi), zi)| {
            mul_vec_bn254(xi, yi, zi);
        });
}

pub fn mul_vec_inplace_bn254(x: &mut [Fr], y: &[Fr]) {
    let len = x.len();
    assert_eq!(len, y.len());

    let simd_x = x.as_ptr() as *const u64;
    let simd_y = y.as_ptr() as *const u64;
    let simd_z = x.as_mut_ptr() as *mut u64;

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

pub fn inner_product_bn254(x: &[Fr], y: &[Fr]) -> Fr {
    assert_eq!(x.len(), y.len());

    let simd_x = x.as_ptr() as *const u64;
    let simd_y = y.as_ptr() as *const u64;
    let mut collect = Fr::zero();
    let simd_z = collect.0 .0.as_mut_ptr() as *mut u64;

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
    let chunk_size = std::cmp::min(x.len(), PAR_CHUNK_SIZE);

    x.par_chunks(chunk_size)
        .zip(y.par_chunks(chunk_size))
        .map(|(chunk_x, chunk_y)| inner_product_bn254(chunk_x, chunk_y))
        .sum::<Fr>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ark::*;
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

    #[test]
    fn parallel_parity_vec_add() {
        const NUM_OPS: usize = 10_000;
        let x: Vec<Fr> = rand_vec(NUM_OPS);
        let y: Vec<Fr> = rand_vec(NUM_OPS);
        let mut z: Vec<Fr> = vec![Fr::zero(); NUM_OPS];

        let x_par = x.clone();
        let y_par = y.clone();
        let mut z_par = z.clone();

        add_vec_bn254(&x, &y, &mut z);
        add_vec_par_bn254(&x_par, &y_par, &mut z_par);
        assert_eq!(z, z_par);
    }

    #[test]
    fn parallel_parity_vec_sub() {
        const NUM_OPS: usize = 10_000;
        let x: Vec<Fr> = rand_vec(NUM_OPS);
        let y: Vec<Fr> = rand_vec(NUM_OPS);
        let mut z: Vec<Fr> = vec![Fr::zero(); NUM_OPS];

        let x_par = x.clone();
        let y_par = y.clone();
        let mut z_par = z.clone();

        sub_vec_bn254(&x, &y, &mut z);
        sub_vec_par_bn254(&x_par, &y_par, &mut z_par);
        assert_eq!(z, z_par);
    }

    #[test]
    fn parallel_parity_vec_sum() {
        const NUM_OPS: usize = 10_000;
        let x: Vec<Fr> = rand_vec(NUM_OPS);

        let x_par = x.clone();

        let z = sum_vec_bn254(&x);
        let z_par = sum_vec_par_bn254(&x_par);
        assert_eq!(z, z_par);
    }

    #[test]
    fn parity_ark_mul() {
        const NUM_OPS: usize = 5;
        let x: Vec<Fr> = rand_vec(NUM_OPS);
        let y: Vec<Fr> = rand_vec(NUM_OPS);
        let mut z: Vec<Fr> = vec![Fr::zero(); NUM_OPS];
        let mut ark_z = z.clone();

        ark_batch_mul(&x, &y, &mut ark_z);
        mul_vec_bn254(&x, &y, &mut z);
        assert_eq!(z, ark_z);
    }

    #[test]
    fn parity_ark_add() {
        const NUM_OPS: usize = 5;
        let x: Vec<Fr> = rand_vec(NUM_OPS);
        let y: Vec<Fr> = rand_vec(NUM_OPS);
        let mut z: Vec<Fr> = vec![Fr::zero(); NUM_OPS];
        let mut ark_z = z.clone();

        ark_batch_add(&x, &y, &mut ark_z);
        add_vec_bn254(&x, &y, &mut z);
        assert_eq!(z, ark_z);
    }

    #[test]
    fn parity_ark_sub() {
        const NUM_OPS: usize = 5;
        let x: Vec<Fr> = rand_vec(NUM_OPS);
        let y: Vec<Fr> = rand_vec(NUM_OPS);
        let mut z: Vec<Fr> = vec![Fr::zero(); NUM_OPS];
        let mut ark_z = z.clone();

        ark_batch_sub(&x, &y, &mut ark_z);
        sub_vec_bn254(&x, &y, &mut z);
        assert_eq!(z, ark_z);
    }

    #[test]
    fn parity_ark_inner_product() {
        const NUM_OPS: usize = 500;
        let x: Vec<Fr> = rand_vec(NUM_OPS);
        let y: Vec<Fr> = rand_vec(NUM_OPS);

        let ark_z = ark_inner_product(&x, &y);
        let z = inner_product_bn254(&x, &y);
        assert_eq!(z, ark_z);
    }

    #[test]
    fn parity_ark_sum() {
        const NUM_OPS: usize = 5;
        let x: Vec<Fr> = rand_vec(NUM_OPS);
        let ark_x = x.clone();

        let ark_z = ark_batch_sum(&ark_x);
        let z = sum_vec_bn254(&x);
        assert_eq!(z, ark_z);
    }
}
