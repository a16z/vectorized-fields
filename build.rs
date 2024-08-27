use std::process::Command;

fn main() {
    // Check for AVX512 support
    let output = Command::new("sh")
        .arg("-c")
        .arg("lscpu | grep avx512")
        .output()
        .expect("Failed to execute command");

    if !output.status.success() || output.stdout.is_empty() {
        panic!("NO AVX_512");
    }
    cc::Build::new()
        .file("src/avx512montmul.S")
        .file("src/innerproduct256.S")
        .file("src/modip256_mont.S")
        .compile("avx512montmul");
}