/*
 * Montgomery arithmetic for moduli up to 255 bits long
 *
 * AVX-512 implementation of 8-way parallel modular multiplication in assembler
 *
 * Copyright (C) 2024 Dag Arne Osvik
 */

.global avx512montmul
.text
.p2align 6,,63

/*
 * void avx512montmul(uint32_t *z, const uint32_t *x, const uint32_t *y, const uint64_t *m)
 *
 * 8-way parallel montgomery multiplication of x and y with results returned in z.
 *
 * Equivalent to: for (i=0; i<8; ++i) z[i] = montmul(x[i], y[i]);
 *
 * The modulus m must be odd and smaller than 2^255, and input values should be
 * less than 2^128*sqrt(m). This allows this operation to save time by omitting
 * a conditional subtraction of m at the end.
 *
 * Note that this means 0 <= z[i] < 2m, and the results may be larger than m.
 * Therefore, when canonical least-nonnegative residues are required, other
 * operations must perform a conditional subtraction of m.
 *
 * Parameters (AMD64 ABI):
 * rdi	z	output array
 * rsi	x	input array
 * rdx	y	input array
 * rcx	m	modulus and negative inverse mod R
 */

avx512montmul:

	// Load x

	vmovdqu64	0*64(%rsi), %zmm16
	vmovdqu64	1*64(%rsi), %zmm17
	vmovdqu64	2*64(%rsi), %zmm18
	vmovdqu64	3*64(%rsi), %zmm19

	// Load y

	vmovdqu64	0*64(%rdx), %zmm24
	vmovdqu64	1*64(%rdx), %zmm25
	vmovdqu64	2*64(%rdx), %zmm26
	vmovdqu64	3*64(%rdx), %zmm27

	// Create mask for low half of each qword

	vpcmpeqb	%ymm15, %ymm15, %ymm15
	vpmovzxdq	%ymm15, %zmm15

////////////////////////////////////////////////////////////////////////////////

	//// Transpose and expand x and y

	// Step 1

	vshufi64x2	$0x88, %zmm17, %zmm16, %zmm20	// 10 00 10 00: even quarters of each input
	vshufi64x2	$0xdd, %zmm17, %zmm16, %zmm22	// 11 01 11 01: odd quarters of each input
	vshufi64x2	$0x88, %zmm19, %zmm18, %zmm21
	vshufi64x2	$0xdd, %zmm19, %zmm18, %zmm23

	vshufi64x2	$0x88, %zmm25, %zmm24, %zmm28
	vshufi64x2	$0xdd, %zmm25, %zmm24, %zmm30
	vshufi64x2	$0x88, %zmm27, %zmm26, %zmm29
	vshufi64x2	$0xdd, %zmm27, %zmm26, %zmm31

	// Step 2

	vpermq		$0xd8, %zmm20, %zmm20	// swap middle words of each half
	vpermq		$0xd8, %zmm21, %zmm21
	vpermq		$0xd8, %zmm22, %zmm22
	vpermq		$0xd8, %zmm23, %zmm23

	vpermq		$0xd8, %zmm28, %zmm28
	vpermq		$0xd8, %zmm29, %zmm29
	vpermq		$0xd8, %zmm30, %zmm30
	vpermq		$0xd8, %zmm31, %zmm31

	// Step 3

	vshufi64x2	$0xd8, %zmm20, %zmm20, %zmm20	// 11 01 10 00: swap middle words
	vshufi64x2	$0xd8, %zmm21, %zmm21, %zmm21
	vshufi64x2	$0xd8, %zmm22, %zmm22, %zmm22
	vshufi64x2	$0xd8, %zmm23, %zmm23, %zmm23

	vshufi64x2	$0xd8, %zmm28, %zmm28, %zmm28
	vshufi64x2	$0xd8, %zmm29, %zmm29, %zmm29
	vshufi64x2	$0xd8, %zmm30, %zmm30, %zmm30
	vshufi64x2	$0xd8, %zmm31, %zmm31, %zmm31

	// Step 4

	vshufi64x2	$0x44, %zmm21, %zmm20, %zmm16	// 01 00 01 00: low half of each input
	vshufi64x2	$0xee, %zmm21, %zmm20, %zmm18	// 11 10 11 10: high half of each input
	vshufi64x2	$0x44, %zmm23, %zmm22, %zmm20
	vshufi64x2	$0xee, %zmm23, %zmm22, %zmm22

	vshufi64x2	$0x44, %zmm29, %zmm28, %zmm24
	vshufi64x2	$0xee, %zmm29, %zmm28, %zmm26
	vshufi64x2	$0x44, %zmm31, %zmm30, %zmm28
	vshufi64x2	$0xee, %zmm31, %zmm30, %zmm30

	// Step 5

	vpsrlq		$32, %zmm16, %zmm17
	vpsrlq		$32, %zmm18, %zmm19
	vpsrlq		$32, %zmm20, %zmm21
	vpsrlq		$32, %zmm22, %zmm23

	vpsrlq		$32, %zmm24, %zmm25
	vpsrlq		$32, %zmm26, %zmm27
	vpsrlq		$32, %zmm28, %zmm29
	vpsrlq		$32, %zmm30, %zmm31

	vpandq		%zmm15, %zmm16, %zmm16
	vpandq		%zmm15, %zmm18, %zmm18
	vpandq		%zmm15, %zmm20, %zmm20
	vpandq		%zmm15, %zmm22, %zmm22

	vpandq		%zmm15, %zmm24, %zmm24
	vpandq		%zmm15, %zmm26, %zmm26
	vpandq		%zmm15, %zmm28, %zmm28
	vpandq		%zmm15, %zmm30, %zmm30

	// For each 256-bit input value, each zmm register now represents a 32-bit input word zero-extended to 64 bits.

////////////////////////////////////////////////////////////////////////////////

	//// Coarsely integrated operand scanning (CIOS) montgomery multiplication

	// Multiply y by word 0 of x

  	vpmuludq	%zmm16, %zmm24, %zmm0
  	vpmuludq	%zmm16, %zmm25, %zmm1
  	vpmuludq	%zmm16, %zmm26, %zmm2
  	vpmuludq	%zmm16, %zmm27, %zmm3
  	vpmuludq	%zmm16, %zmm28, %zmm4
  	vpmuludq	%zmm16, %zmm29, %zmm5
  	vpmuludq	%zmm16, %zmm30, %zmm6
  	vpmuludq	%zmm16, %zmm31, %zmm7

	// Call multiply-reduce subroutine

					call iter_1	// Skip multiply-add part
	vmovdqa64	%zmm17, %zmm16;	call iter_n
	vmovdqa64	%zmm18, %zmm16;	call iter_n
	vmovdqa64	%zmm19, %zmm16;	call iter_n
	vmovdqa64	%zmm20, %zmm16;	call iter_n
	vmovdqa64	%zmm21, %zmm16;	call iter_n
	vmovdqa64	%zmm22, %zmm16;	call iter_n
	vmovdqa64	%zmm23, %zmm16;	call iter_n

////////////////////////////////////////////////////////////////////////////////

	//// Transpose results back

	vmovdqa64	pattern1(%rip), %zmm11
	vmovdqa64	pattern2(%rip), %zmm12
	vmovdqa64	pattern3(%rip), %zmm13
	vmovdqa64	pattern4(%rip), %zmm14

	// Step 1

	vpsllq		$32, %zmm1, %zmm1;	vporq	%zmm1, %zmm0, %zmm0
	vpsllq		$32, %zmm3, %zmm3;	vporq	%zmm3, %zmm2, %zmm1
	vpsllq		$32, %zmm5, %zmm5;	vporq	%zmm5, %zmm4, %zmm2
	vpsllq		$32, %zmm7, %zmm7;	vporq	%zmm7, %zmm6, %zmm3

	// Step 2

	vmovdqa64	%zmm0, %zmm4
	vmovdqa64	%zmm2, %zmm6

	vpermt2q	%zmm1, %zmm11, %zmm0
	vpermt2q	%zmm4, %zmm12, %zmm1
	vpermt2q	%zmm3, %zmm11, %zmm2
	vpermt2q	%zmm6, %zmm12, %zmm3

	// Step 3

	vmovdqa64	%zmm0, %zmm4
	vmovdqa64	%zmm1, %zmm5

	vpermt2q	%zmm2, %zmm13, %zmm0
	vpermt2q	%zmm4, %zmm14, %zmm2
	vpermt2q	%zmm3, %zmm13, %zmm1
	vpermt2q	%zmm5, %zmm14, %zmm3

	// Save results

	vmovdqu64	%zmm0, 0*64(%rdi)
	vmovdqu64	%zmm2, 1*64(%rdi)
	vmovdqu64	%zmm1, 2*64(%rdi)
	vmovdqu64	%zmm3, 3*64(%rdi)

	movq	%r8, %rax

	ret

////////////////////////////////////////////////////////////////////////////////

//// Multiply-reduce subroutine

.p2align 6,,63
iter_n:
	// Multiply

  	vpmuludq	%zmm16, %zmm24, %zmm10
  	vpmuludq	%zmm16, %zmm25, %zmm11
  	vpmuludq	%zmm16, %zmm26, %zmm12
  	vpmuludq	%zmm16, %zmm27, %zmm13
  	vpmuludq	%zmm16, %zmm28, %zmm14;	vpaddq	%zmm10, %zmm0, %zmm0
  	vpmuludq	%zmm16, %zmm29, %zmm10;	vpaddq	%zmm11, %zmm1, %zmm1
  	vpmuludq	%zmm16, %zmm30, %zmm11;	vpaddq	%zmm12, %zmm2, %zmm2
  	vpmuludq	%zmm16, %zmm31, %zmm12;	vpaddq	%zmm13, %zmm3, %zmm3
						vpaddq	%zmm14, %zmm4, %zmm4
						vpaddq	%zmm10, %zmm5, %zmm5
						vpaddq	%zmm11, %zmm6, %zmm6
						vpaddq	%zmm12, %zmm7, %zmm7

iter_1:

	vpmuludq	8*8(%rcx){1to8}, %zmm0, %zmm9	// Reduction multiplier

	// Propagate carries (use %zmm16 as eight 32-bit carries zero-extended to 64 bits)

	vpsrlq		$32, %zmm0, %zmm16	// Extract carry from %zmm0
	vpandq		%zmm15, %zmm0, %zmm0	// Remove carry from %zmm0
	vpaddq		%zmm16, %zmm1, %zmm1	// Propagate carry to %zmm1

	vpsrlq		$32, %zmm1, %zmm16;	vpandq	%zmm15, %zmm1, %zmm1;	vpaddq	%zmm16, %zmm2, %zmm2
	vpsrlq		$32, %zmm2, %zmm16;	vpandq	%zmm15, %zmm2, %zmm2;	vpaddq	%zmm16, %zmm3, %zmm3
	vpsrlq		$32, %zmm3, %zmm16;	vpandq	%zmm15, %zmm3, %zmm3;	vpaddq	%zmm16, %zmm4, %zmm4
	vpsrlq		$32, %zmm4, %zmm16;	vpandq	%zmm15, %zmm4, %zmm4;	vpaddq	%zmm16, %zmm5, %zmm5
	vpsrlq		$32, %zmm5, %zmm16;	vpandq	%zmm15, %zmm5, %zmm5;	vpaddq	%zmm16, %zmm6, %zmm6
	vpsrlq		$32, %zmm6, %zmm16;	vpandq	%zmm15, %zmm6, %zmm6;	vpaddq	%zmm16, %zmm7, %zmm7
	vpsrlq		$32, %zmm7, %zmm8;	vpandq	%zmm15, %zmm7, %zmm7

	vpandq	%zmm15, %zmm9, %zmm9	// Only low 32 bits of reduction multiplier

	// Reduce

	vpmuludq	0*8(%rcx){1to8}, %zmm9, %zmm10
	vpmuludq	1*8(%rcx){1to8}, %zmm9, %zmm11
	vpmuludq	2*8(%rcx){1to8}, %zmm9, %zmm12
	vpmuludq	3*8(%rcx){1to8}, %zmm9, %zmm13
	vpmuludq	4*8(%rcx){1to8}, %zmm9, %zmm14;	vpaddq	%zmm10, %zmm0, %zmm0
	vpmuludq	5*8(%rcx){1to8}, %zmm9, %zmm10;	vpaddq	%zmm11, %zmm1, %zmm1
	vpmuludq	6*8(%rcx){1to8}, %zmm9, %zmm11;	vpaddq	%zmm12, %zmm2, %zmm2
	vpmuludq	7*8(%rcx){1to8}, %zmm9, %zmm12;	vpaddq	%zmm13, %zmm3, %zmm3
							vpaddq	%zmm14, %zmm4, %zmm4
							vpaddq	%zmm10, %zmm5, %zmm5
							vpaddq	%zmm11, %zmm6, %zmm6
							vpaddq	%zmm12, %zmm7, %zmm7

	vpsrlq		$32, %zmm0, %zmm16;	vpaddq	%zmm16, %zmm1, %zmm1;	vpandq	%zmm15, %zmm1, %zmm0
	vpsrlq		$32, %zmm1, %zmm16;	vpaddq	%zmm16, %zmm2, %zmm2;	vpandq	%zmm15, %zmm2, %zmm1
	vpsrlq		$32, %zmm2, %zmm16;	vpaddq	%zmm16, %zmm3, %zmm3;	vpandq	%zmm15, %zmm3, %zmm2
	vpsrlq		$32, %zmm3, %zmm16;	vpaddq	%zmm16, %zmm4, %zmm4;	vpandq	%zmm15, %zmm4, %zmm3
	vpsrlq		$32, %zmm4, %zmm16;	vpaddq	%zmm16, %zmm5, %zmm5;	vpandq	%zmm15, %zmm5, %zmm4
	vpsrlq		$32, %zmm5, %zmm16;	vpaddq	%zmm16, %zmm6, %zmm6;	vpandq	%zmm15, %zmm6, %zmm5
	vpsrlq		$32, %zmm6, %zmm16;	vpaddq	%zmm16, %zmm7, %zmm7;	vpandq	%zmm15, %zmm7, %zmm6
	vpsrlq		$32, %zmm7, %zmm16;	vpaddq	%zmm16, %zmm8, %zmm8;	vpandq	%zmm15, %zmm8, %zmm7
	vpsrlq		$32, %zmm8, %zmm8;                                  

	ret

////////////////////////////////////////////////////////////////////////////////

//// Set of constants for shuffling AVX-512 register contents using vpermt2q

.p2align 6,,63	// 512-bit alignment

pattern1:
.quad	 0,  8,  1,  9,  2, 10,  3, 11

pattern2:
.quad	12,  4, 13,  5, 14,  6, 15,  7

pattern3:
.quad	 0,  1, 8,  9,  2,  3,  10, 11

pattern4:
.quad	12, 13,  4,  5, 14, 15,  6,  7

////////////////////////////////////////////////////////////////////////////////

