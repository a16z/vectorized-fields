use ark_ff::PrimeField;
use rayon::prelude::*;

pub fn ark_batch_mul<F: PrimeField>(x: &[F], y: &[F], z: &mut [F]) {
    let len = x.len();
    assert_eq!(len, y.len());
    assert_eq!(len, z.len());

    for i in 0..len {
        z[i] = x[i] * y[i];
    }
}

pub fn ark_batch_mul_par<F: PrimeField>(x: &[F], y: &[F], z: &mut [F]) {
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

pub fn ark_inner_product<F: PrimeField>(x: &[F], y: &[F]) -> F {
    assert_eq!(x.len(), y.len());

    let mut result = F::zero();
    for i in 0..x.len() {
        result += x[i] * y[i];
    }
    result
}

pub fn ark_inner_product_par<F: PrimeField>(x: &[F], y: &[F]) -> F {
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

pub fn ark_batch_sum<F: PrimeField>(x: &[F]) -> F {
    let len = x.len();

    let mut z = F::zero();
    for i in 0..len {
        z += x[i];
    }
    z
}

pub fn ark_batch_sum_par<F: PrimeField>(x: &[F], z: &mut [F]) {
    let len = x.len();
    assert_eq!(len, z.len());

    x.par_iter().zip(z.par_iter_mut()).for_each(|(xi, zi)| {
        *zi = *xi + *zi;
    });
}

pub fn ark_batch_add<F: PrimeField>(x: &[F], y: &[F], z: &mut [F]) {
    let len = x.len();
    assert_eq!(len, y.len());
    assert_eq!(len, z.len());

    for i in 0..len {
        z[i] = x[i] + y[i];
    }
}

pub fn ark_batch_add_par<F: PrimeField>(x: &[F], y: &[F], z: &mut [F]) {
    let len = x.len();
    assert_eq!(len, y.len());
    assert_eq!(len, z.len());

    x.par_iter()
        .zip(y.par_iter())
        .zip(z.par_iter_mut())
        .for_each(|((xi, yi), zi)| {
            *zi = *xi + *yi;
        });
}

pub fn ark_batch_sub<F: PrimeField>(x: &[F], y: &[F], z: &mut [F]) {
    let len = x.len();
    assert_eq!(len, y.len());
    assert_eq!(len, z.len());

    for i in 0..len {
        z[i] = x[i] - y[i];
    }
}

pub fn ark_batch_sub_par<F: PrimeField>(x: &[F], y: &[F], z: &mut [F]) {
    let len = x.len();
    assert_eq!(len, y.len());
    assert_eq!(len, z.len());

    x.par_iter()
        .zip(y.par_iter())
        .zip(z.par_iter_mut())
        .for_each(|((xi, yi), zi)| {
            *zi = *xi - *yi;
        });
}
