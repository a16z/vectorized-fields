use vectorized_fields::benchmarks;

fn main() {
    benchmarks::bench_mul_single_threaded();
    benchmarks::bench_mul_multi_threaded();

    benchmarks::bench_inner_product_single_threaded();
    benchmarks::bench_inner_product_multi_threaded();

    benchmarks::bench_add_single_threaded();
    benchmarks::bench_add_multi_threaded();

    benchmarks::bench_sub_single_threaded();
    benchmarks::bench_sub_multi_threaded();
}
