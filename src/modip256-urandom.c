#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

extern void modip256_mont(uint64_t z[4], const uint64_t *x, const uint64_t *y, uint32_t xy_len, const uint64_t m[6]);

// Modulus + inverse (for Montgomery) + mu (for Barrett)

uint64_t fp[6] = {
    // Modulus: 0x30644E72E131A029B85045B68181585D97816A916871CA8D3C208C16D87CFD47
    0x3C208C16D87CFD47, 0x97816A916871CA8D, 0xB85045B68181585D, 0x30644E72E131A029,

    // Negative inverse of modulus: 0xF57A22B791888C6BD8AFCBD01833DA809EDE7D651ECA6AC987D20782E4866389
    // The least significant 64-bit word only:
    0x87D20782E4866389,

    // For Barrett: mu = 2^288 / m
    0x000000054A474626
};

uint64_t fr[6] = {
    // Modulus: 0x30644E72E131A029B85045B68181585D2833E84879B9709143E1F593F0000001
    0x43e1f593f0000001, 0x2833e84879b97091, 0xb85045b68181585d, 0x30644e72e131a029,

    // Negative inverse of modulus: 0x40D019D832A0FCE8AEF9B39374A81A7665DE1528CB3816E9C2E1F593EFFFFFFF
    // The least significant 64-bit word only:
    0xc2e1f593efffffff,

    // For Barrett: mu = 2^288 / m
    0x000000054A474626
};


int main(int argc, char **argv) {
    uint64_t z[4], *x, *y;

    size_t xy_len = 1000000;

    if (argc >= 2)
        sscanf(argv[1], "%zd", &xy_len);

    printf("%zd\n", xy_len);

    x = aligned_alloc(sysconf(_SC_PAGESIZE), xy_len*32);
    y = aligned_alloc(sysconf(_SC_PAGESIZE), xy_len*32);

    if (x == NULL) return -1;
    if (y == NULL) return -1;

    FILE *urandom = fopen("/dev/urandom", "r");

    if (urandom == NULL)
        return -2;

    if (xy_len != fread(x, 32, xy_len, urandom))
        return -3;

    if (xy_len != fread(y, 32, xy_len, urandom))
        return -4;

    fclose(urandom);

    if ((argc >= 3) && (!strcmp(argv[2], "-v")))
    {
        printf("16doi\n0\n");

        for (size_t i=0; i<xy_len; ++i)
        {
            uint64_t *p = x+4*i, *q = y+4*i;
            printf("%016lX%016lX%016lX%016lX ", p[3], p[2], p[1], p[0]);
            printf("%016lX%016lX%016lX%016lX ", q[3], q[2], q[1], q[0]);
            printf("* +\n");
        }

        // R^-1 mod fr
        printf("15EBF95182C5551CC8260DE4AEB85D5D090EF5A9E111EC87DC5BA0056DB1194E*\n");
        printf("%016lX%016lX%016lX%016lX %%p\n", fr[3], fr[2], fr[1], fr[0]);
    }

    fflush(stdout);

    modip256_mont(z, x, y, xy_len, fr);

    clock_t t0 = clock();

    modip256_mont(z, x, y, xy_len, fr);

    clock_t t1 = clock();

    fprintf(stderr, "%f ms\n", (float)(1000*(t1-t0)) / CLOCKS_PER_SEC);

    fflush(stderr);

    // Create output to be piped to dc for validation
    if ((argc >= 3) && (!strcmp(argv[2], "-v")))
        printf("%016lX%016lX%016lX%016lX-pq\n", z[3], z[2], z[1], z[0]);

    return 0;
}

// vim: ai si sw=4 ts=4 et
