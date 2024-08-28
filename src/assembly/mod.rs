// TODO(sragss): Clean these up.
extern "C" {
    pub fn avx512montmul(z: *mut u32, x: *const u32, y: *const u32, m: *const u64);

    pub fn modip256_mont(z: *mut u64, x: *const u64, y: *const u64, xy_len: u32, m: *const u64);
    pub fn modmul256_mont(z: *mut u64, x: *const u64, y: *const u64, xy_len: u64, m: *const u64);
}
