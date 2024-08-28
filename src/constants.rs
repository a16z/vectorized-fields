pub const BN254_FP: [u64; 6] = [
    // Modulus: 0x30644E72E131A029B85045B68181585D97816A916871CA8D3C208C16D87CFD47
    0x3C208C16D87CFD47, 0x97816A916871CA8D, 0xB85045B68181585D, 0x30644E72E131A029,

    // Negative inverse of modulus: 0xF57A22B791888C6BD8AFCBD01833DA809EDE7D651ECA6AC987D20782E4866389
    // The least significant 64-bit word only:
    0x87D20782E4866389,

    // For Barrett: mu = 2^288 / m
    0x000000054A474626
];

pub const BN254_FR: [u64; 6] = [
    // Modulus: 0x30644E72E131A029B85045B68181585D2833E84879B9709143E1F593F0000001
    0x43e1f593f0000001, 0x2833e84879b97091, 0xb85045b68181585d, 0x30644e72e131a029,

    // Negative inverse of modulus: 0x40D019D832A0FCE8AEF9B39374A81A7665DE1528CB3816E9C2E1F593EFFFFFFF
    // The least significant 64-bit word only:
    0xc2e1f593efffffff,

    // For Barrett: mu = 2^288 / m
    0x000000054A474626
];