#include <stdint.h>
#include <stdio.h>
#include <time.h>

// Zero-extended modulus words + inverse

uint64_t fp[9] = {
    0xD87CFD47,
    0x3C208C16,
    0x6871CA8D,
    0x97816A91,
    0x8181585D,
    0xB85045B6,
    0xE131A029,
    0x30644E72,

    0xE4866389
};

uint32_t fp1[8] = {
    // 0E0A77C19A07DF2F666EA36F7879462C0A78EB28F5C70B3DD35D438DC58F0D9D
    // Montgomery representation of 1 in fp
    0xC58F0D9D,
    0xD35D438D,
    0xF5C70B3D,
    0x0A78EB28,
    0x7879462C,
    0x666EA36F,
    0x9A07DF2F,
    0x0E0A77C1
};

uint32_t fp2[8] = {
    // 1C14EF83340FBE5ECCDD46DEF0F28C5814F1D651EB8E167BA6BA871B8B1E1B3A
    // Montgomery representation of 2 in fp
    0x8B1E1B3A,
    0xA6BA871B,
    0xEB8E167B,
    0x14F1D651,
    0xF0F28C58,
    0xCCDD46DE,
    0x340FBE5E,
    0x1C14EF83
};

uint32_t fp4[8] = {
    // 07C5909386EDDC93E16A48076063C052926242126EAA626A115482203DBF392D
    // Montgomery representation of 4 in fp
    0x3DBF392D,
    0x11548220,
    0x6EAA626A,
    0x92624212,
    0x6063C052,
    0xE16A4807,
    0x86EDDC93,
    0x07C59093
};

extern void avx512montmul(uint32_t *z, const uint32_t *x, const uint32_t *y, const uint64_t *m);

int main() {
    uint32_t z[64], x[64], y[64];

    avx512montmul(z, fp1, fp1, fp);

    for (unsigned i = 0; i < 8; ++i) {
        for (unsigned j = 0; j < 8; ++j) {
            printf("%02X", ((unsigned char*)z)[i * 8 * 4 + j * 4 + 3]);
            printf("%02X", ((unsigned char*)z)[i * 8 * 4 + j * 4 + 2]);
            printf("%02X", ((unsigned char*)z)[i * 8 * 4 + j * 4 + 1]);
            printf("%02X", ((unsigned char*)z)[i * 8 * 4 + j * 4 + 0]);
        }
        putchar('\n');
    }
    printf("\nThats one yo\n");

    /* Set test values:
        x = { fp1, fp1, fp2, fp2, fp2, fp2, fp2, fp2 }
        y = { fp1, fp2, fp2, fp2, fp2, fp2, fp2, fp2 }
    */
    for (unsigned i=0; i<8; ++i) {
        for (unsigned j=0; j<8; ++j) {
            x[i*8+j] = fp2[j];
            y[i*8+j] = fp2[j];
        }
        z[i] = 0;
    }
    for (unsigned j=0; j<8; ++j) {
        x[0*8+j] = fp1[j];
        y[0*8+j] = fp1[j];
    }
    for (unsigned j=0; j<8; ++j) {
        x[1*8+j] = fp1[j];
        y[1*8+j] = fp2[j];
    }

    // Before initial transposition

    for (unsigned i=0; i<64; i+=8) {
        printf(" %08X%08X%08X%08X%08X%08X%08X%08X", x[i+7], x[i+6], x[i+5], x[i+4], x[i+3], x[i+2], x[i+1], x[i+0]);
        printf(" %08X%08X%08X%08X%08X%08X%08X%08X", y[i+7], y[i+6], y[i+5], y[i+4], y[i+3], y[i+2], y[i+1], y[i+0]);
        putchar('\n');
    }
    putchar('\n');

    avx512montmul(z, x, y, fp);

    for (unsigned i=0; i<8; ++i) {
        printf(" %08X%08X%08X%08X%08X%08X%08X%08X", z[i*8+7], z[i*8+6], z[i*8+5], z[i*8+4], z[i*8+3], z[i*8+2], z[i*8+1], z[i*8+0]);
        putchar('\n');
    }
    putchar('\n');

    clock_t t0 = clock();

    for (unsigned i=0; i<1000000; ++i)
        avx512montmul(z, x, y, fp);

    clock_t t1 = clock();

    for (unsigned i=0; i<8; ++i) {
        printf(" %08X%08X%08X%08X%08X%08X%08X%08X", z[i*8+7], z[i*8+6], z[i*8+5], z[i*8+4], z[i*8+3], z[i*8+2], z[i*8+1], z[i*8+0]);
        putchar('\n');
    }
    putchar('\n');

    fprintf(stderr, "%f\n", (float)(t1-t0) / CLOCKS_PER_SEC);

    return 0;
}

// vim: ai si sw=4 ts=4 et

