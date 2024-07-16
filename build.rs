fn main() {
    cc::Build::new()
        .file("src/avx512montmul.S")
        .compile("avx512montmul");
}