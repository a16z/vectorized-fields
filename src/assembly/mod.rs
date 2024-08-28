// TODO(sragss): Clean these up.
extern "C" {
    pub fn modip256_mont(z: *mut u64, x: *const u64, y: *const u64, xy_len: u32, m: *const u64);
    pub fn modmul256_mont(z: *mut u64, x: *const u64, y: *const u64, xy_len: u64, m: *const u64);
    pub fn modsum256(z: *mut u64, x: *const u64, x_len: u32, m: *const u64);
    pub fn modadd256(z: *mut u64, x: *const u64, y: *const u64, xy_len: u64, m: *const u64);
    pub fn modsub256(z: *mut u64, x: *const u64, y: *const u64, xy_len: u64, m: *const u64);
}
