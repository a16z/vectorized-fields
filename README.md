# vectorized-fields

Vectorized AVX-512 256-bit arithmetic library focused on BN254 field vector operations. Boasts 1-3x speedups over plain CPU arithmetic.

## Features

This library provides optimized vector operations for BN254 field arithmetic, leveraging AVX-512 instructions. The following operations are supported:

- Vector operations: `add`, `sub`, `mul`, `sum`, `inner_product`
- Parallel versions: `add_par`, `sub_par`, `mul_par`, `sum_par`, `inner_product_par`
- In-place operations: `add_inplace`, `sub_inplace`, `mul_inplace`

All operations are suffixed with `_bn254`.

## Elementwise Vector Operations

The library performs elementwise operations on vectors. For example:

### Multiplication

For vectors $\mathbf{x} = \begin{pmatrix} x_0 \\ x_1 \\ x_2 \\ x_3 \end{pmatrix}$ and $\mathbf{y} = \begin{pmatrix} y_0 \\ y_1 \\ y_2 \\ y_3 \end{pmatrix}$, 

the elementwise multiplication is:

$\mathbf{z} = \mathbf{x} \odot \mathbf{y} = \begin{pmatrix} x_0 \cdot y_0 \\ x_1 \cdot y_1 \\ x_2 \cdot y_2 \\ x_3 \cdot y_3 \end{pmatrix}$

### Addition

Similarly, for addition:

$\mathbf{z} = \mathbf{x} + \mathbf{y} = \begin{pmatrix} x_0 + y_0 \\ x_1 + y_1 \\ x_2 + y_2 \\ x_3 + y_3 \end{pmatrix}$

These operations are performed efficiently using AVX-512 instructions on larger vectors.

## Benchmarks (C5-metal)
*Note that the acceleration decreases with the number of threads as RAM gets saturated.*
![Bar Graph](./benchmarks/bar_graph.png)

![Line Graph](./benchmarks/line_graph.png)
