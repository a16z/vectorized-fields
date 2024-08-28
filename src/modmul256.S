/*
 * void modmul256_mont(uint64_t *z, const uint64_t *x, const uint64_t *y, uint64_t xy_len, const uint64_t m[6])
 *
 * Copyright (C) 2024 Dag Arne Osvik
 *
 * Elementwise modular multiplication of vectors.
 * 8-way parallel AVX512 and 8-element sequential integer computation per iteration.
 *
 * Parameters:
 *  z       out  Result. Vector of canonical least non-negative residues.
 *  x,y     in   Pointers to vectors of length xy_len.
 *               Elements must be 256-bit little-endian canonical residues.
 *  xy_len  in   Length of input vectors. Limited to 2^59-1.
 *  m       in   Pointer to 6-qword array containing the modulus m, its negative inverse mod 2^64 and 2^288/m.
 *
 * The implementation is based on manual coarse interleaving of AVX-512 and integer (AMD64) code.
 * Most comments in the interleaved code relate to AVX-512; the integer code is documented under Loop1.
 */

.global modmul256_mont

//////////////////////////////////////////////////
// Register roles
//////////////////////////////////////////////////

//		AMD64	WIN64
//	%rax	PY,Y3	PX
//	%rcx	LEN	PZ, Y1
//	%rdx	MUL	MUL
//	%rbx	PL	PL
//	%rsp	*	*
//	%rbp	PH	PH
//	%rsi	PX	Y0
//	%rdi	PZ,Y2	PM
//	%r8	PM	PY, Y3
//	%r9	Y1	LEN
//	%r10	Z0	Z0
//	%r11	Z1	Z1
//	%r12	Z2	Z2
//	%r13	Z3	Z3
//	%r14	Z4	Z4
//	%r15	Y0	Y2

//	zmm0-zmm7	intermediate result
//	zmm8		2^32-1, mask to extract lower dword from qwords
//	zmm9		reduction multiplier
//	zmm10-zmm17	intermediate results
//	zmm16-zmm23	digits of x
//	zmm24-zmm31	digits of y

#ifndef WIN64
// AMD64 calling convention
# define PZ	%rdi
# define PX     %rsi
# define PY     %rax
# define LEN    %rcx
# define PM     %r8

# define Y3	PY
# define Y2	PZ
# define Y1	%r9
# define Y0	%r15

#else
// X64 calling convention
# define PZ     %rcx
# define PX     %rax
# define PY     %r8
# define LEN    %r9
# define PM     %rax

# define Y0	%rsi
# define Y1	PZ
# define Y2	%r15
# define Y3	PY

#endif

#define MUL	%rdx
#define Z0	%r10
#define Z1	%r11
#define Z2	%r12
#define Z3	%r13
#define Z4	%r14
#define PL	%rbx
#define PH	%rbp

.text
.p2align 6,,63

modmul256_mont:

	//////////////////////////////////////////////////
	// Init
	//////////////////////////////////////////////////

	// Free up rdx to be used as multiplier
	movq	%rdx, %rax

#ifndef WIN64
	pushq	%rbx
	pushq	%rbp
#else
	// Load pointer to the modulus from stack
	movq	5*8(%rsp), PM

	movq	%rbx, 1*8(%rsp)
	movq	%rbp, 2*8(%rsp)
	movq	%rsi, 3*8(%rsp)
	movq	%rdi, 4*8(%rsp)
#endif
	pushq	%r12
	pushq	%r13
	pushq	%r14
	pushq	%r15

	// Change to negatively-indexed pointers

	shlq	$5, LEN	// 32 bytes per element, 16 elements per batch

	addq	LEN, PX
	addq	LEN, PY
	addq	LEN, PZ
	negq	LEN

	// Process inputs one by one until a multiple of 16 is left

	test	$0x1e0, LEN
	jnz	Loop1

Blocksof16:
	test	LEN, LEN
	jz	Done	// Nothing more to do

#ifdef WIN64
	// Save xmm6-xmm15

	subq	$10*16, %rsp

	movdqu	xmm6,  $0*16(%rsp)
	movdqu	xmm7,  $1*16(%rsp)
	movdqu	xmm8,  $2*16(%rsp)
	movdqu	xmm9,  $3*16(%rsp)
	movdqu	xmm10, $4*16(%rsp)
	movdqu	xmm11, $5*16(%rsp)
	movdqu	xmm12, $6*16(%rsp)
	movdqu	xmm13, $7*16(%rsp)
	movdqu	xmm14, $8*16(%rsp)
	movdqu	xmm15, $9*16(%rsp)
#endif
	// Create mask for low dword in each qword

	vpcmpeqb	%ymm8, %ymm8, %ymm8
	vpmovzxdq	%ymm8, %zmm8

	mov	$0x5555, %edx
	kmovd	%edx, %k1

	jmp	Loop16

.p2align 6,,63
Loop16:
	//////////////////////////////////////////////////
	// Load inputs
	//////////////////////////////////////////////////

	// Save registers

	pushq	PY
	pushq	PZ
	pushq	LEN

	// Load x

	vmovdqu64	256+0*64(PX, LEN), %zmm16
	vmovdqu64	256+1*64(PX, LEN), %zmm17
	vmovdqu64	256+2*64(PX, LEN), %zmm18
	vmovdqu64	256+3*64(PX, LEN), %zmm19

	// Load y

	vmovdqu64	256+0*64(PY, LEN), %zmm24
	vmovdqu64	256+1*64(PY, LEN), %zmm25
	vmovdqu64	256+2*64(PY, LEN), %zmm26
	vmovdqu64	256+3*64(PY, LEN), %zmm27

	// Load inputs

	movq	0*8(PX,LEN), MUL
	movq	0*8(PY,LEN), Y0
	movq	1*8(PY,LEN), Y1
	movq	2*8(PY,LEN), Y2
	movq	3*8(PY,LEN), Y3

	//////////////////////////////////////////////////
	// Transpose and expand x and y
	//////////////////////////////////////////////////

	// Step 1

	vshufi64x2	$0x88, %zmm17, %zmm16, %zmm20	// 0x88 = 0b1000_1000: even quarters of each input
	vshufi64x2	$0xdd, %zmm17, %zmm16, %zmm22	// 0xdd = 0b1101_1101: odd quarters of each input
	vshufi64x2	$0x88, %zmm19, %zmm18, %zmm21
	vshufi64x2	$0xdd, %zmm19, %zmm18, %zmm23

	vshufi64x2	$0x88, %zmm25, %zmm24, %zmm28
	vshufi64x2	$0xdd, %zmm25, %zmm24, %zmm30
	vshufi64x2	$0x88, %zmm27, %zmm26, %zmm29
	vshufi64x2	$0xdd, %zmm27, %zmm26, %zmm31

	mulxq	Y0, Z1, Z2
	mulxq	Y1, PL, Z3;		addq	PL, Z2
	mulxq	Y2, PL, Z4;		adcq	PL, Z3
	mulxq	Y3, PL, Z0;		adcq	PL, Z4;	adcq	$0, Z0
	movq	Z1, MUL

	// Step 2

	vpermq		$0xd8, %zmm20, %zmm20	// 0xd8 = 0b11_01_10_00: swap middle words of each half
	vpermq		$0xd8, %zmm21, %zmm21
	vpermq		$0xd8, %zmm22, %zmm22
	vpermq		$0xd8, %zmm23, %zmm23

	mulxq	4*8(PM), MUL, PH
	mulxq	0*8(PM), PL, PH;	addq	PL, Z1;	adcq	PH, Z2
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z3;	adcq	PH, Z4;	adcq	$0, Z0
	mulxq	1*8(PM), PL, PH;	addq	PL, Z2;	adcq	PH, Z3
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1

	movq	1*8(PX, LEN), MUL

	vpermq		$0xd8, %zmm28, %zmm28
	vpermq		$0xd8, %zmm29, %zmm29
	vpermq		$0xd8, %zmm30, %zmm30
	vpermq		$0xd8, %zmm31, %zmm31

	mulxq	Y0, PL, PH;		addq	PL, Z2;	adcq	PH, Z3
	mulxq	Y2, PL, PH;		adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1
	mulxq	Y1, PL, PH;		addq	PL, Z3;	adcq	PH, Z4
	mulxq	Y3, PL, PH;		adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	movq	Z2, MUL

	// Step 3

	vshufi64x2	$0xd8, %zmm20, %zmm20, %zmm20	// 0xd8 = 0b11_01_10_00: swap middle words
	vshufi64x2	$0xd8, %zmm21, %zmm21, %zmm21
	vshufi64x2	$0xd8, %zmm22, %zmm22, %zmm22
	vshufi64x2	$0xd8, %zmm23, %zmm23, %zmm23

	mulxq	4*8(PM), MUL, PH
	mulxq	0*8(PM), PL, PH;	addq	PL, Z2;	adcq	PH, Z3
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1
	mulxq	1*8(PM), PL, PH;	addq	PL, Z3;	adcq	PH, Z4
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2

	movq	2*8(PX, LEN), MUL

	vshufi64x2	$0xd8, %zmm28, %zmm28, %zmm28
	vshufi64x2	$0xd8, %zmm29, %zmm29, %zmm29
	vshufi64x2	$0xd8, %zmm30, %zmm30, %zmm30
	vshufi64x2	$0xd8, %zmm31, %zmm31, %zmm31

	mulxq	Y0, PL, PH;		addq	PL, Z3;	adcq	PH, Z4
	mulxq	Y2, PL, PH;		adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	mulxq	Y1, PL, PH;		addq	PL, Z4;	adcq	PH, Z0
	mulxq	Y3, PL, PH;		adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	movq	Z3, MUL

	// Step 4

	vshufi64x2	$0x44, %zmm21, %zmm20, %zmm16	// 0x44 = 0b01_00_01_00: low half of each input
	vshufi64x2	$0xee, %zmm21, %zmm20, %zmm18	// 0xee = 0b11_10_11_10: high half of each input
	vshufi64x2	$0x44, %zmm23, %zmm22, %zmm20
	vshufi64x2	$0xee, %zmm23, %zmm22, %zmm22

	mulxq	4*8(PM), MUL, PH
	mulxq	0*8(PM), PL, PH;	addq	PL, Z3;	adcq	PH, Z4
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	mulxq	1*8(PM), PL, PH;	addq	PL, Z4;	adcq	PH, Z0
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3

	movq	3*8(PX, LEN), MUL

	vshufi64x2	$0x44, %zmm29, %zmm28, %zmm24
	vshufi64x2	$0xee, %zmm29, %zmm28, %zmm26
	vshufi64x2	$0x44, %zmm31, %zmm30, %zmm28
	vshufi64x2	$0xee, %zmm31, %zmm30, %zmm30

	// Step 5

	vpsrlq		$32, %zmm16, %zmm17
	vpsrlq		$32, %zmm18, %zmm19
	vpsrlq		$32, %zmm20, %zmm21
	vpsrlq		$32, %zmm22, %zmm23

	mulxq	Y0, PL, PH;		addq	PL, Z4;	adcq	PH, Z0
	mulxq	Y2, PL, PH;		adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	mulxq	Y1, PL, PH;		addq	PL, Z0;	adcq	PH, Z1
	mulxq	Y3, PL, PH;		adcq	PL, Z2;	adcq	PH, Z3;	adcq	$0, Z4
	movq	Z4, MUL

	vpsrlq		$32, %zmm24, %zmm25
	vpsrlq		$32, %zmm26, %zmm27
	vpsrlq		$32, %zmm28, %zmm29
	vpsrlq		$32, %zmm30, %zmm31

	mulxq	4*8(PM), MUL, PH
	mulxq	0*8(PM), PL, PH;	addq	PL, Z4;	adcq	PH, Z0
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	mulxq	1*8(PM), PL, PH;	addq	PL, Z0;	adcq	PH, Z1
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z2;	adcq	PH, Z3;	adcq	$0, Z4

	vpandq		%zmm8, %zmm16, %zmm16
	vpandq		%zmm8, %zmm18, %zmm18
	vpandq		%zmm8, %zmm20, %zmm20
	vpandq		%zmm8, %zmm22, %zmm22

	vpandq		%zmm8, %zmm24, %zmm24
	vpandq		%zmm8, %zmm26, %zmm26
	vpandq		%zmm8, %zmm28, %zmm28
	vpandq		%zmm8, %zmm30, %zmm30

	// Conditional subtraction of the modulus

	movq	Z0, Y0
	movq	Z1, Y1
	movq	Z2, Y2
	movq	Z3, Y3

	subq	0*8(PM), Y0
	sbbq	1*8(PM), Y1
	sbbq	2*8(PM), Y2
	sbbq	3*8(PM), Y3

	cmovncq	Y0, Z0
	cmovncq	Y1, Z1
	cmovncq	Y2, Z2
	cmovncq	Y3, Z3

	// Restore registers

	popq	LEN
	mov	0(%rsp), PZ
	mov	8(%rsp), PY

	// Store output

	movq	Z0, 0*8(PZ, LEN)
	movq	Z1, 1*8(PZ, LEN)
	movq	Z2, 2*8(PZ, LEN)
	movq	Z3, 3*8(PZ, LEN)

	addq	$32, LEN

	// Save registers

	pushq	LEN

	// Load inputs

	movq	0*8(PX,LEN), MUL
	movq	0*8(PY,LEN), Y0
	movq	1*8(PY,LEN), Y1
	movq	2*8(PY,LEN), Y2
	movq	3*8(PY,LEN), Y3

	// For each 256-bit input value, each zmm register now represents a 32-bit input word zero-extended to 64 bits.

	//////////////////////////////////////////////////
	// Multiply y by doubleword 0 of x
	//////////////////////////////////////////////////

	vpmuludq	%zmm16, %zmm24, %zmm0
	vpmuludq	%zmm16, %zmm25, %zmm1
	vpmuludq	%zmm16, %zmm26, %zmm2
	vpmuludq	%zmm16, %zmm27, %zmm3
	vpmuludq	%zmm16, %zmm28, %zmm4
	vpmuludq	%zmm16, %zmm29, %zmm5
	vpmuludq	%zmm16, %zmm30, %zmm6
	vpmuludq	%zmm16, %zmm31, %zmm7

	vpmuludq	8*4(PM){1to8}, %zmm0, %zmm9	// Reduction multiplier

	mulxq	Y0, Z1, Z2
	mulxq	Y1, PL, Z3;		addq	PL, Z2
	mulxq	Y2, PL, Z4;		adcq	PL, Z3
	mulxq	Y3, PL, Z0;		adcq	PL, Z4;	adcq	$0, Z0
	movq	Z1, MUL

	vpsrlq		$32, %zmm0, %zmm10;	vpandq	%zmm8, %zmm0, %zmm0;	vpaddq	%zmm10, %zmm1, %zmm1
	vpsrlq		$32, %zmm1, %zmm11;	vpandq	%zmm8, %zmm1, %zmm1;	vpaddq	%zmm11, %zmm2, %zmm2
	vpsrlq		$32, %zmm2, %zmm12;	vpandq	%zmm8, %zmm2, %zmm2;	vpaddq	%zmm12, %zmm3, %zmm3
	vpsrlq		$32, %zmm3, %zmm13;	vpandq	%zmm8, %zmm3, %zmm3;	vpaddq	%zmm13, %zmm4, %zmm4

	mulxq	4*8(PM), MUL, PH
	mulxq	0*8(PM), PL, PH;	addq	PL, Z1;	adcq	PH, Z2
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z3;	adcq	PH, Z4;	adcq	$0, Z0
	mulxq	1*8(PM), PL, PH;	addq	PL, Z2;	adcq	PH, Z3
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1

	movq	1*8(PX, LEN), MUL

	vpsrlq		$32, %zmm4, %zmm14;	vpandq	%zmm8, %zmm4, %zmm4;	vpaddq	%zmm14, %zmm5, %zmm5
	vpsrlq		$32, %zmm5, %zmm15;	vpandq	%zmm8, %zmm5, %zmm5;	vpaddq	%zmm15, %zmm6, %zmm6
	vpsrlq		$32, %zmm6, %zmm16;	vpandq	%zmm8, %zmm6, %zmm6;	vpaddq	%zmm16, %zmm7, %zmm7

	mulxq	Y0, PL, PH;		addq	PL, Z2;	adcq	PH, Z3
	mulxq	Y2, PL, PH;		adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1
	mulxq	Y1, PL, PH;		addq	PL, Z3;	adcq	PH, Z4
	mulxq	Y3, PL, PH;		adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	movq	Z2, MUL

	//////////////////////////////////////////////////
	// Reduce
	//////////////////////////////////////////////////

	vpmuludq	0*4(PM){1to8}, %zmm9, %zmm10;	vpaddq	%zmm10, %zmm0, %zmm0
	vpmuludq	1*4(PM){1to8}, %zmm9, %zmm11;	vpaddq	%zmm11, %zmm1, %zmm1
	vpmuludq	2*4(PM){1to8}, %zmm9, %zmm12;	vpaddq	%zmm12, %zmm2, %zmm2
	vpmuludq	3*4(PM){1to8}, %zmm9, %zmm13;	vpaddq	%zmm13, %zmm3, %zmm3

	mulxq	4*8(PM), MUL, PH
	mulxq	0*8(PM), PL, PH;	addq	PL, Z2;	adcq	PH, Z3
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1
	mulxq	1*8(PM), PL, PH;	addq	PL, Z3;	adcq	PH, Z4
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	movq	2*8(PX, LEN), MUL

	vpmuludq	4*4(PM){1to8}, %zmm9, %zmm14;	vpaddq	%zmm14, %zmm4, %zmm4
	vpmuludq	5*4(PM){1to8}, %zmm9, %zmm15;	vpaddq	%zmm15, %zmm5, %zmm5
	vpmuludq	6*4(PM){1to8}, %zmm9, %zmm16;	vpaddq	%zmm16, %zmm6, %zmm6
	vpmuludq	7*4(PM){1to8}, %zmm9, %zmm10;	vpaddq	%zmm10, %zmm7, %zmm7

	mulxq	Y0, PL, PH;		addq	PL, Z3;	adcq	PH, Z4
	mulxq	Y2, PL, PH;		adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	mulxq	Y1, PL, PH;		addq	PL, Z4;	adcq	PH, Z0
	mulxq	Y3, PL, PH;		adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	movq	Z3, MUL

	vpsrlq	$32, %zmm0, %zmm10;	vpaddq	%zmm10, %zmm1, %zmm1;	vpandq	%zmm8, %zmm1, %zmm0
	vpsrlq	$32, %zmm1, %zmm11;	vpaddq	%zmm11, %zmm2, %zmm2;	vpandq	%zmm8, %zmm2, %zmm1
	vpsrlq	$32, %zmm2, %zmm12;	vpaddq	%zmm12, %zmm3, %zmm3;	vpandq	%zmm8, %zmm3, %zmm2
	vpsrlq	$32, %zmm3, %zmm13;	vpaddq	%zmm13, %zmm4, %zmm4;	vpandq	%zmm8, %zmm4, %zmm3

	mulxq	4*8(PM), MUL, PH
	mulxq	0*8(PM), PL, PH;	addq	PL, Z3;	adcq	PH, Z4
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	mulxq	1*8(PM), PL, PH;	addq	PL, Z4;	adcq	PH, Z0
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	movq	3*8(PX, LEN), MUL

	vpsrlq	$32, %zmm4, %zmm14;	vpaddq	%zmm14, %zmm5, %zmm5;	vpandq	%zmm8, %zmm5, %zmm4
	vpsrlq	$32, %zmm5, %zmm15;	vpaddq	%zmm15, %zmm6, %zmm6;	vpandq	%zmm8, %zmm6, %zmm5
	vpsrlq	$32, %zmm6, %zmm16;	vpaddq	%zmm16, %zmm7, %zmm7;	vpandq	%zmm8, %zmm7, %zmm6
	vpsrlq	$32, %zmm7, %zmm7

	mulxq	Y0, PL, PH;		addq	PL, Z4;	adcq	PH, Z0
	mulxq	Y2, PL, PH;		adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	mulxq	Y1, PL, PH;		addq	PL, Z0;	adcq	PH, Z1
	mulxq	Y3, PL, PH;		adcq	PL, Z2;	adcq	PH, Z3;	adcq	$0, Z4
	movq	Z4, MUL

	//////////////////////////////////////////////////
	// Process doubleword 1 of x
	//////////////////////////////////////////////////

	vpmuludq	%zmm17, %zmm24, %zmm10;		vpaddq	%zmm10, %zmm0, %zmm0;
	vpmuludq	%zmm17, %zmm25, %zmm11;		vpaddq	%zmm11, %zmm1, %zmm1;
	vpmuludq	%zmm17, %zmm26, %zmm12;		vpaddq	%zmm12, %zmm2, %zmm2;
	vpmuludq	%zmm17, %zmm27, %zmm13;		vpaddq	%zmm13, %zmm3, %zmm3;

	mulxq	4*8(PM), MUL, PH
	mulxq	0*8(PM), PL, PH;	addq	PL, Z4;	adcq	PH, Z0
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	mulxq	1*8(PM), PL, PH;	addq	PL, Z0;	adcq	PH, Z1
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z2;	adcq	PH, Z3;	adcq	$0, Z4

	vpmuludq	%zmm17, %zmm28, %zmm14;		vpaddq	%zmm14, %zmm4, %zmm4;
	vpmuludq	%zmm17, %zmm29, %zmm15;		vpaddq	%zmm15, %zmm5, %zmm5;
	vpmuludq	%zmm17, %zmm30, %zmm16;		vpaddq	%zmm16, %zmm6, %zmm6;
	vpmuludq	%zmm17, %zmm31, %zmm17;		vpaddq	%zmm17, %zmm7, %zmm7;

	vpmuludq	8*4(PM){1to8}, %zmm0, %zmm9;	// Compute reduction multipliers

	// Conditional subtraction of the modulus

	movq	Z0, Y0
	movq	Z1, Y1
	movq	Z2, Y2
	movq	Z3, Y3

	subq	0*8(PM), Y0
	sbbq	1*8(PM), Y1
	sbbq	2*8(PM), Y2
	sbbq	3*8(PM), Y3

	cmovncq	Y0, Z0
	cmovncq	Y1, Z1
	cmovncq	Y2, Z2
	cmovncq	Y3, Z3

	// Restore registers

	popq	LEN
	mov	0(%rsp), PZ
	mov	8(%rsp), PY

	// Store output

	movq	Z0, 0*8(PZ, LEN)
	movq	Z1, 1*8(PZ, LEN)
	movq	Z2, 2*8(PZ, LEN)
	movq	Z3, 3*8(PZ, LEN)

	addq	$32, LEN

	// Save registers

	pushq	LEN

	// Load inputs

	movq	0*8(PX,LEN), MUL
	movq	0*8(PY,LEN), Y0
	movq	1*8(PY,LEN), Y1
	movq	2*8(PY,LEN), Y2
	movq	3*8(PY,LEN), Y3

	// Move high dwords to zmm10-16, add each to the corresponding low dword (propagate 32-bit carries)

	vpsrlq		$32, %zmm0, %zmm10;		vpandq	%zmm8, %zmm0, %zmm0;	vpaddq	%zmm10, %zmm1, %zmm1;
	vpsrlq		$32, %zmm1, %zmm11;		vpandq	%zmm8, %zmm1, %zmm1;	vpaddq	%zmm11, %zmm2, %zmm2;
	vpsrlq		$32, %zmm2, %zmm12;		vpandq	%zmm8, %zmm2, %zmm2;	vpaddq	%zmm12, %zmm3, %zmm3;

	mulxq	Y0, Z1, Z2
	mulxq	Y1, PL, Z3;		addq	PL, Z2
	mulxq	Y2, PL, Z4;		adcq	PL, Z3
	mulxq	Y3, PL, Z0;		adcq	PL, Z4;	adcq	$0, Z0
	movq	Z1, MUL

	vpsrlq		$32, %zmm3, %zmm13;		vpandq	%zmm8, %zmm3, %zmm3;	vpaddq	%zmm13, %zmm4, %zmm4;
	vpsrlq		$32, %zmm4, %zmm14;		vpandq	%zmm8, %zmm4, %zmm4;	vpaddq	%zmm14, %zmm5, %zmm5;
	vpsrlq		$32, %zmm5, %zmm15;		vpandq	%zmm8, %zmm5, %zmm5;	vpaddq	%zmm15, %zmm6, %zmm6;
	vpsrlq		$32, %zmm6, %zmm16;		vpandq	%zmm8, %zmm6, %zmm6;	vpaddq	%zmm16, %zmm7, %zmm7;
	// zmm7 keeps all 64 bits

	mulxq	4*8(PM), MUL, PH
	mulxq	0*8(PM), PL, PH;	addq	PL, Z1;	adcq	PH, Z2
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z3;	adcq	PH, Z4;	adcq	$0, Z0
	mulxq	1*8(PM), PL, PH;	addq	PL, Z2;	adcq	PH, Z3
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1

	movq	1*8(PX, LEN), MUL

	vpmuludq	0*4(PM){1to8}, %zmm9, %zmm10;	vpaddq	%zmm10, %zmm0, %zmm0;	// Low dword of zmm0 is zero
	vpmuludq	1*4(PM){1to8}, %zmm9, %zmm11;	vpaddq	%zmm11, %zmm1, %zmm1;
	vpmuludq	2*4(PM){1to8}, %zmm9, %zmm12;	vpaddq	%zmm12, %zmm2, %zmm2;
	vpmuludq	3*4(PM){1to8}, %zmm9, %zmm13;	vpaddq	%zmm13, %zmm3, %zmm3;

	mulxq	Y0, PL, PH;		addq	PL, Z2;	adcq	PH, Z3
	mulxq	Y2, PL, PH;		adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1
	mulxq	Y1, PL, PH;		addq	PL, Z3;	adcq	PH, Z4
	mulxq	Y3, PL, PH;		adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	movq	Z2, MUL

	vpmuludq	4*4(PM){1to8}, %zmm9, %zmm14;	vpaddq	%zmm14, %zmm4, %zmm4;
	vpmuludq	5*4(PM){1to8}, %zmm9, %zmm15;	vpaddq	%zmm15, %zmm5, %zmm5;
	vpmuludq	6*4(PM){1to8}, %zmm9, %zmm16;	vpaddq	%zmm16, %zmm6, %zmm6;
	vpmuludq	7*4(PM){1to8}, %zmm9, %zmm17;	vpaddq	%zmm17, %zmm7, %zmm7;

	mulxq	4*8(PM), MUL, PH
	mulxq	0*8(PM), PL, PH;	addq	PL, Z2;	adcq	PH, Z3
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1
	mulxq	1*8(PM), PL, PH;	addq	PL, Z3;	adcq	PH, Z4
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2

	movq	2*8(PX, LEN), MUL

	// Propagate carries and shift down by one dword

	vpsrlq		$32, %zmm0, %zmm10;		vpaddq	%zmm10, %zmm1, %zmm1;	vpandq	%zmm8, %zmm1, %zmm0
	vpsrlq		$32, %zmm1, %zmm11;		vpaddq	%zmm11, %zmm2, %zmm2;	vpandq	%zmm8, %zmm2, %zmm1
	vpsrlq		$32, %zmm2, %zmm12;		vpaddq	%zmm12, %zmm3, %zmm3;	vpandq	%zmm8, %zmm3, %zmm2
	vpsrlq		$32, %zmm3, %zmm13;		vpaddq	%zmm13, %zmm4, %zmm4;	vpandq	%zmm8, %zmm4, %zmm3

	mulxq	Y0, PL, PH;		addq	PL, Z3;	adcq	PH, Z4
	mulxq	Y2, PL, PH;		adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	mulxq	Y1, PL, PH;		addq	PL, Z4;	adcq	PH, Z0
	mulxq	Y3, PL, PH;		adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	movq	Z3, MUL

	vpsrlq		$32, %zmm4, %zmm14;		vpaddq	%zmm14, %zmm5, %zmm5;	vpandq	%zmm8, %zmm5, %zmm4
	vpsrlq		$32, %zmm5, %zmm15;		vpaddq	%zmm15, %zmm6, %zmm6;	vpandq	%zmm8, %zmm6, %zmm5
	vpsrlq		$32, %zmm6, %zmm16;		vpaddq	%zmm16, %zmm7, %zmm7;	vpandq	%zmm8, %zmm7, %zmm6
	vpsrlq		$32, %zmm7, %zmm7

	mulxq	4*8(PM), MUL, PH
	mulxq	0*8(PM), PL, PH;	addq	PL, Z3;	adcq	PH, Z4
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	mulxq	1*8(PM), PL, PH;	addq	PL, Z4;	adcq	PH, Z0
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3

	movq	3*8(PX, LEN), MUL

	//////////////////////////////////////////////////
	// Process doubleword 2 of x
	//////////////////////////////////////////////////

	vpmuludq	%zmm18, %zmm24, %zmm10;		vpaddq	%zmm10, %zmm0, %zmm0;
	vpmuludq	%zmm18, %zmm25, %zmm11;		vpaddq	%zmm11, %zmm1, %zmm1;
	vpmuludq	%zmm18, %zmm26, %zmm12;		vpaddq	%zmm12, %zmm2, %zmm2;
	vpmuludq	%zmm18, %zmm27, %zmm13;		vpaddq	%zmm13, %zmm3, %zmm3;

	mulxq	Y0, PL, PH;		addq	PL, Z4;	adcq	PH, Z0
	mulxq	Y2, PL, PH;		adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	mulxq	Y1, PL, PH;		addq	PL, Z0;	adcq	PH, Z1
	mulxq	Y3, PL, PH;		adcq	PL, Z2;	adcq	PH, Z3;	adcq	$0, Z4
	movq	Z4, MUL

	vpmuludq	%zmm18, %zmm28, %zmm14;		vpaddq	%zmm14, %zmm4, %zmm4;
	vpmuludq	%zmm18, %zmm29, %zmm15;		vpaddq	%zmm15, %zmm5, %zmm5;
	vpmuludq	%zmm18, %zmm30, %zmm16;		vpaddq	%zmm16, %zmm6, %zmm6;
	vpmuludq	%zmm18, %zmm31, %zmm17;		vpaddq	%zmm17, %zmm7, %zmm7;

	vpmuludq	8*4(PM){1to8}, %zmm0, %zmm9;	// Compute reduction multipliers

	mulxq	4*8(PM), MUL, PH
	mulxq	0*8(PM), PL, PH;	addq	PL, Z4;	adcq	PH, Z0
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	mulxq	1*8(PM), PL, PH;	addq	PL, Z0;	adcq	PH, Z1
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z2;	adcq	PH, Z3;	adcq	$0, Z4

	// Move high dwords to zmm10-16, add each to the corresponding low dword (propagate 32-bit carries)

	vpsrlq		$32, %zmm0, %zmm10;		vpandq	%zmm8, %zmm0, %zmm0;	vpaddq	%zmm10, %zmm1, %zmm1;

	// Conditional subtraction of the modulus

	movq	Z0, Y0
	movq	Z1, Y1
	movq	Z2, Y2
	movq	Z3, Y3

	vpsrlq		$32, %zmm1, %zmm11;		vpandq	%zmm8, %zmm1, %zmm1;	vpaddq	%zmm11, %zmm2, %zmm2;

	subq	0*8(PM), Y0
	sbbq	1*8(PM), Y1
	sbbq	2*8(PM), Y2
	sbbq	3*8(PM), Y3

	vpsrlq		$32, %zmm2, %zmm12;		vpandq	%zmm8, %zmm2, %zmm2;	vpaddq	%zmm12, %zmm3, %zmm3;

	cmovncq	Y0, Z0
	cmovncq	Y1, Z1
	cmovncq	Y2, Z2
	cmovncq	Y3, Z3

	vpsrlq		$32, %zmm3, %zmm13;		vpandq	%zmm8, %zmm3, %zmm3;	vpaddq	%zmm13, %zmm4, %zmm4;

	// Restore registers

	popq	LEN
	mov	0(%rsp), PZ
	mov	8(%rsp), PY

	vpsrlq		$32, %zmm4, %zmm14;		vpandq	%zmm8, %zmm4, %zmm4;	vpaddq	%zmm14, %zmm5, %zmm5;

	// Store output

	movq	Z0, 0*8(PZ, LEN)
	movq	Z1, 1*8(PZ, LEN)
	movq	Z2, 2*8(PZ, LEN)
	movq	Z3, 3*8(PZ, LEN)

	vpsrlq		$32, %zmm5, %zmm15;		vpandq	%zmm8, %zmm5, %zmm5;	vpaddq	%zmm15, %zmm6, %zmm6;

	addq	$32, LEN

	// Save registers

	pushq	LEN

	vpsrlq		$32, %zmm6, %zmm16;		vpandq	%zmm8, %zmm6, %zmm6;	vpaddq	%zmm16, %zmm7, %zmm7;
	// zmm7 keeps all 64 bits

	// Load inputs

	movq	0*8(PX,LEN), MUL
	movq	0*8(PY,LEN), Y0
	movq	1*8(PY,LEN), Y1
	movq	2*8(PY,LEN), Y2
	movq	3*8(PY,LEN), Y3

	vpmuludq	0*4(PM){1to8}, %zmm9, %zmm10;	vpaddq	%zmm10, %zmm0, %zmm0;	// Low dword of zmm0 is zero
	vpmuludq	1*4(PM){1to8}, %zmm9, %zmm11;	vpaddq	%zmm11, %zmm1, %zmm1;
	vpmuludq	2*4(PM){1to8}, %zmm9, %zmm12;	vpaddq	%zmm12, %zmm2, %zmm2;
	vpmuludq	3*4(PM){1to8}, %zmm9, %zmm13;	vpaddq	%zmm13, %zmm3, %zmm3;

	mulxq	Y0, Z1, Z2
	mulxq	Y1, PL, Z3;		addq	PL, Z2
	mulxq	Y2, PL, Z4;		adcq	PL, Z3
	mulxq	Y3, PL, Z0;		adcq	PL, Z4;	adcq	$0, Z0
	movq	Z1, MUL

	vpmuludq	4*4(PM){1to8}, %zmm9, %zmm14;	vpaddq	%zmm14, %zmm4, %zmm4;
	vpmuludq	5*4(PM){1to8}, %zmm9, %zmm15;	vpaddq	%zmm15, %zmm5, %zmm5;
	vpmuludq	6*4(PM){1to8}, %zmm9, %zmm16;	vpaddq	%zmm16, %zmm6, %zmm6;
	vpmuludq	7*4(PM){1to8}, %zmm9, %zmm17;	vpaddq	%zmm17, %zmm7, %zmm7;

	mulxq	4*8(PM), MUL, PH
	mulxq	0*8(PM), PL, PH;	addq	PL, Z1;	adcq	PH, Z2
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z3;	adcq	PH, Z4;	adcq	$0, Z0
	mulxq	1*8(PM), PL, PH;	addq	PL, Z2;	adcq	PH, Z3
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1

	movq	1*8(PX, LEN), MUL

	// Propagate carries and shift down by one dword

	vpsrlq		$32, %zmm0, %zmm10;		vpaddq	%zmm10, %zmm1, %zmm1;	vpandq	%zmm8, %zmm1, %zmm0
	vpsrlq		$32, %zmm1, %zmm11;		vpaddq	%zmm11, %zmm2, %zmm2;	vpandq	%zmm8, %zmm2, %zmm1
	vpsrlq		$32, %zmm2, %zmm12;		vpaddq	%zmm12, %zmm3, %zmm3;	vpandq	%zmm8, %zmm3, %zmm2
	vpsrlq		$32, %zmm3, %zmm13;		vpaddq	%zmm13, %zmm4, %zmm4;	vpandq	%zmm8, %zmm4, %zmm3

	mulxq	Y0, PL, PH;		addq	PL, Z2;	adcq	PH, Z3
	mulxq	Y2, PL, PH;		adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1
	mulxq	Y1, PL, PH;		addq	PL, Z3;	adcq	PH, Z4
	mulxq	Y3, PL, PH;		adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	movq	Z2, MUL

	vpsrlq		$32, %zmm4, %zmm14;		vpaddq	%zmm14, %zmm5, %zmm5;	vpandq	%zmm8, %zmm5, %zmm4
	vpsrlq		$32, %zmm5, %zmm15;		vpaddq	%zmm15, %zmm6, %zmm6;	vpandq	%zmm8, %zmm6, %zmm5
	vpsrlq		$32, %zmm6, %zmm16;		vpaddq	%zmm16, %zmm7, %zmm7;	vpandq	%zmm8, %zmm7, %zmm6
	vpsrlq		$32, %zmm7, %zmm7

	mulxq	4*8(PM), MUL, PH
	mulxq	0*8(PM), PL, PH;	addq	PL, Z2;	adcq	PH, Z3
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1
	mulxq	1*8(PM), PL, PH;	addq	PL, Z3;	adcq	PH, Z4
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2

	movq	2*8(PX, LEN), MUL

	//////////////////////////////////////////////////
	// Process doubleword 3 of x
	//////////////////////////////////////////////////

	vpmuludq	%zmm19, %zmm24, %zmm10;		vpaddq	%zmm10, %zmm0, %zmm0;
	vpmuludq	%zmm19, %zmm25, %zmm11;		vpaddq	%zmm11, %zmm1, %zmm1;
	vpmuludq	%zmm19, %zmm26, %zmm12;		vpaddq	%zmm12, %zmm2, %zmm2;
	vpmuludq	%zmm19, %zmm27, %zmm13;		vpaddq	%zmm13, %zmm3, %zmm3;

	mulxq	Y0, PL, PH;		addq	PL, Z3;	adcq	PH, Z4
	mulxq	Y2, PL, PH;		adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	mulxq	Y1, PL, PH;		addq	PL, Z4;	adcq	PH, Z0
	mulxq	Y3, PL, PH;		adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	movq	Z3, MUL

	vpmuludq	%zmm19, %zmm28, %zmm14;		vpaddq	%zmm14, %zmm4, %zmm4;
	vpmuludq	%zmm19, %zmm29, %zmm15;		vpaddq	%zmm15, %zmm5, %zmm5;
	vpmuludq	%zmm19, %zmm30, %zmm16;		vpaddq	%zmm16, %zmm6, %zmm6;
	vpmuludq	%zmm19, %zmm31, %zmm17;		vpaddq	%zmm17, %zmm7, %zmm7;

	vpmuludq	8*4(PM){1to8}, %zmm0, %zmm9;	// Compute reduction multipliers

	mulxq	4*8(PM), MUL, PH
	mulxq	0*8(PM), PL, PH;	addq	PL, Z3;	adcq	PH, Z4
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	mulxq	1*8(PM), PL, PH;	addq	PL, Z4;	adcq	PH, Z0
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3

	movq	3*8(PX, LEN), MUL

	// Move high dwords to zmm10-16, add each to the corresponding low dword (propagate 32-bit carries)

	vpsrlq		$32, %zmm0, %zmm10;		vpandq	%zmm8, %zmm0, %zmm0;	vpaddq	%zmm10, %zmm1, %zmm1;
	vpsrlq		$32, %zmm1, %zmm11;		vpandq	%zmm8, %zmm1, %zmm1;	vpaddq	%zmm11, %zmm2, %zmm2;
	vpsrlq		$32, %zmm2, %zmm12;		vpandq	%zmm8, %zmm2, %zmm2;	vpaddq	%zmm12, %zmm3, %zmm3;
	vpsrlq		$32, %zmm3, %zmm13;		vpandq	%zmm8, %zmm3, %zmm3;	vpaddq	%zmm13, %zmm4, %zmm4;

	mulxq	Y0, PL, PH;		addq	PL, Z4;	adcq	PH, Z0
	mulxq	Y2, PL, PH;		adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	mulxq	Y1, PL, PH;		addq	PL, Z0;	adcq	PH, Z1
	mulxq	Y3, PL, PH;		adcq	PL, Z2;	adcq	PH, Z3;	adcq	$0, Z4
	movq	Z4, MUL

	vpsrlq		$32, %zmm4, %zmm14;		vpandq	%zmm8, %zmm4, %zmm4;	vpaddq	%zmm14, %zmm5, %zmm5;
	vpsrlq		$32, %zmm5, %zmm15;		vpandq	%zmm8, %zmm5, %zmm5;	vpaddq	%zmm15, %zmm6, %zmm6;
	vpsrlq		$32, %zmm6, %zmm16;		vpandq	%zmm8, %zmm6, %zmm6;	vpaddq	%zmm16, %zmm7, %zmm7;
	// zmm7 keeps all 64 bits

	vpmuludq	0*4(PM){1to8}, %zmm9, %zmm10;	vpaddq	%zmm10, %zmm0, %zmm0;	// Low dword of zmm0 is zero

	mulxq	4*8(PM), MUL, PH
	mulxq	0*8(PM), PL, PH;	addq	PL, Z4;	adcq	PH, Z0
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	mulxq	1*8(PM), PL, PH;	addq	PL, Z0;	adcq	PH, Z1
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z2;	adcq	PH, Z3;	adcq	$0, Z4

	vpmuludq	1*4(PM){1to8}, %zmm9, %zmm11;	vpaddq	%zmm11, %zmm1, %zmm1;

	// Conditional subtraction of the modulus

	movq	Z0, Y0
	movq	Z1, Y1
	movq	Z2, Y2
	movq	Z3, Y3

	vpmuludq	2*4(PM){1to8}, %zmm9, %zmm12;	vpaddq	%zmm12, %zmm2, %zmm2;

	subq	0*8(PM), Y0
	sbbq	1*8(PM), Y1
	sbbq	2*8(PM), Y2
	sbbq	3*8(PM), Y3

	vpmuludq	3*4(PM){1to8}, %zmm9, %zmm13;	vpaddq	%zmm13, %zmm3, %zmm3;

	cmovncq	Y0, Z0
	cmovncq	Y1, Z1
	cmovncq	Y2, Z2
	cmovncq	Y3, Z3

	vpmuludq	4*4(PM){1to8}, %zmm9, %zmm14;	vpaddq	%zmm14, %zmm4, %zmm4;

	// Restore registers

	popq	LEN
	mov	0(%rsp), PZ
	mov	8(%rsp), PY

	vpmuludq	5*4(PM){1to8}, %zmm9, %zmm15;	vpaddq	%zmm15, %zmm5, %zmm5;

	// Store output

	movq	Z0, 0*8(PZ, LEN)
	movq	Z1, 1*8(PZ, LEN)
	movq	Z2, 2*8(PZ, LEN)
	movq	Z3, 3*8(PZ, LEN)

	vpmuludq	6*4(PM){1to8}, %zmm9, %zmm16;	vpaddq	%zmm16, %zmm6, %zmm6;

	addq	$32, LEN

	// Save registers

	pushq	LEN

	vpmuludq	7*4(PM){1to8}, %zmm9, %zmm17;	vpaddq	%zmm17, %zmm7, %zmm7;

	// Load inputs

	movq	0*8(PX,LEN), MUL
	movq	0*8(PY,LEN), Y0
	movq	1*8(PY,LEN), Y1
	movq	2*8(PY,LEN), Y2
	movq	3*8(PY,LEN), Y3

	// Propagate carries and shift down by one dword

	vpsrlq		$32, %zmm0, %zmm10;		vpaddq	%zmm10, %zmm1, %zmm1;	vpandq	%zmm8, %zmm1, %zmm0
	vpsrlq		$32, %zmm1, %zmm11;		vpaddq	%zmm11, %zmm2, %zmm2;	vpandq	%zmm8, %zmm2, %zmm1
	vpsrlq		$32, %zmm2, %zmm12;		vpaddq	%zmm12, %zmm3, %zmm3;	vpandq	%zmm8, %zmm3, %zmm2
	vpsrlq		$32, %zmm3, %zmm13;		vpaddq	%zmm13, %zmm4, %zmm4;	vpandq	%zmm8, %zmm4, %zmm3

	mulxq	Y0, Z1, Z2
	mulxq	Y1, PL, Z3;		addq	PL, Z2
	mulxq	Y2, PL, Z4;		adcq	PL, Z3
	mulxq	Y3, PL, Z0;		adcq	PL, Z4;	adcq	$0, Z0
	movq	Z1, MUL

	vpsrlq		$32, %zmm4, %zmm14;		vpaddq	%zmm14, %zmm5, %zmm5;	vpandq	%zmm8, %zmm5, %zmm4
	vpsrlq		$32, %zmm5, %zmm15;		vpaddq	%zmm15, %zmm6, %zmm6;	vpandq	%zmm8, %zmm6, %zmm5
	vpsrlq		$32, %zmm6, %zmm16;		vpaddq	%zmm16, %zmm7, %zmm7;	vpandq	%zmm8, %zmm7, %zmm6
	vpsrlq		$32, %zmm7, %zmm7

	mulxq	4*8(PM), MUL, PH
	mulxq	0*8(PM), PL, PH;	addq	PL, Z1;	adcq	PH, Z2
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z3;	adcq	PH, Z4;	adcq	$0, Z0
	mulxq	1*8(PM), PL, PH;	addq	PL, Z2;	adcq	PH, Z3
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1

	movq	1*8(PX, LEN), MUL

	//////////////////////////////////////////////////
	// Process doubleword 4 of x
	//////////////////////////////////////////////////

	vpmuludq	%zmm20, %zmm24, %zmm10;		vpaddq	%zmm10, %zmm0, %zmm0;
	vpmuludq	%zmm20, %zmm25, %zmm11;		vpaddq	%zmm11, %zmm1, %zmm1;
	vpmuludq	%zmm20, %zmm26, %zmm12;		vpaddq	%zmm12, %zmm2, %zmm2;
	vpmuludq	%zmm20, %zmm27, %zmm13;		vpaddq	%zmm13, %zmm3, %zmm3;

	mulxq	Y0, PL, PH;		addq	PL, Z2;	adcq	PH, Z3
	mulxq	Y2, PL, PH;		adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1
	mulxq	Y1, PL, PH;		addq	PL, Z3;	adcq	PH, Z4
	mulxq	Y3, PL, PH;		adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	movq	Z2, MUL

	vpmuludq	%zmm20, %zmm28, %zmm14;		vpaddq	%zmm14, %zmm4, %zmm4;
	vpmuludq	%zmm20, %zmm29, %zmm15;		vpaddq	%zmm15, %zmm5, %zmm5;
	vpmuludq	%zmm20, %zmm30, %zmm16;		vpaddq	%zmm16, %zmm6, %zmm6;
	vpmuludq	%zmm20, %zmm31, %zmm17;		vpaddq	%zmm17, %zmm7, %zmm7;

	vpmuludq	8*4(PM){1to8}, %zmm0, %zmm9;	// Compute reduction multipliers

	mulxq	4*8(PM), MUL, PH
	mulxq	0*8(PM), PL, PH;	addq	PL, Z2;	adcq	PH, Z3
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1
	mulxq	1*8(PM), PL, PH;	addq	PL, Z3;	adcq	PH, Z4
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2

	movq	2*8(PX, LEN), MUL

	// Move high dwords to zmm10-16, add each to the corresponding low dword (propagate 32-bit carries)

	vpsrlq		$32, %zmm0, %zmm10;		vpandq	%zmm8, %zmm0, %zmm0;	vpaddq	%zmm10, %zmm1, %zmm1;
	vpsrlq		$32, %zmm1, %zmm11;		vpandq	%zmm8, %zmm1, %zmm1;	vpaddq	%zmm11, %zmm2, %zmm2;
	vpsrlq		$32, %zmm2, %zmm12;		vpandq	%zmm8, %zmm2, %zmm2;	vpaddq	%zmm12, %zmm3, %zmm3;
	vpsrlq		$32, %zmm3, %zmm13;		vpandq	%zmm8, %zmm3, %zmm3;	vpaddq	%zmm13, %zmm4, %zmm4;

	mulxq	Y0, PL, PH;		addq	PL, Z3;	adcq	PH, Z4
	mulxq	Y2, PL, PH;		adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	mulxq	Y1, PL, PH;		addq	PL, Z4;	adcq	PH, Z0
	mulxq	Y3, PL, PH;		adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	movq	Z3, MUL
	mulxq	4*8(PM), MUL, PH

	vpsrlq		$32, %zmm4, %zmm14;		vpandq	%zmm8, %zmm4, %zmm4;	vpaddq	%zmm14, %zmm5, %zmm5;
	vpsrlq		$32, %zmm5, %zmm15;		vpandq	%zmm8, %zmm5, %zmm5;	vpaddq	%zmm15, %zmm6, %zmm6;
	vpsrlq		$32, %zmm6, %zmm16;		vpandq	%zmm8, %zmm6, %zmm6;	vpaddq	%zmm16, %zmm7, %zmm7;
	// zmm7 keeps all 64 bits

	mulxq	0*8(PM), PL, PH;	addq	PL, Z3;	adcq	PH, Z4
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	mulxq	1*8(PM), PL, PH;	addq	PL, Z4;	adcq	PH, Z0
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3

	movq	3*8(PX, LEN), MUL

	vpmuludq	0*4(PM){1to8}, %zmm9, %zmm10;	vpaddq	%zmm10, %zmm0, %zmm0;	// Low dword of zmm0 is zero
	vpmuludq	1*4(PM){1to8}, %zmm9, %zmm11;	vpaddq	%zmm11, %zmm1, %zmm1;
	vpmuludq	2*4(PM){1to8}, %zmm9, %zmm12;	vpaddq	%zmm12, %zmm2, %zmm2;
	vpmuludq	3*4(PM){1to8}, %zmm9, %zmm13;	vpaddq	%zmm13, %zmm3, %zmm3;

	mulxq	Y0, PL, PH;		addq	PL, Z4;	adcq	PH, Z0
	mulxq	Y2, PL, PH;		adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	mulxq	Y1, PL, PH;		addq	PL, Z0;	adcq	PH, Z1
	mulxq	Y3, PL, PH;		adcq	PL, Z2;	adcq	PH, Z3;	adcq	$0, Z4
	movq	Z4, MUL
	mulxq	4*8(PM), MUL, PH

	vpmuludq	4*4(PM){1to8}, %zmm9, %zmm14;	vpaddq	%zmm14, %zmm4, %zmm4;
	vpmuludq	5*4(PM){1to8}, %zmm9, %zmm15;	vpaddq	%zmm15, %zmm5, %zmm5;
	vpmuludq	6*4(PM){1to8}, %zmm9, %zmm16;	vpaddq	%zmm16, %zmm6, %zmm6;
	vpmuludq	7*4(PM){1to8}, %zmm9, %zmm17;	vpaddq	%zmm17, %zmm7, %zmm7;

	mulxq	0*8(PM), PL, PH;	addq	PL, Z4;	adcq	PH, Z0
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	mulxq	1*8(PM), PL, PH;	addq	PL, Z0;	adcq	PH, Z1
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z2;	adcq	PH, Z3;	adcq	$0, Z4

	// Propagate carries and shift down by one dword

	vpsrlq		$32, %zmm0, %zmm10;		vpaddq	%zmm10, %zmm1, %zmm1;	vpandq	%zmm8, %zmm1, %zmm0
	vpsrlq		$32, %zmm1, %zmm11;		vpaddq	%zmm11, %zmm2, %zmm2;	vpandq	%zmm8, %zmm2, %zmm1
	vpsrlq		$32, %zmm2, %zmm12;		vpaddq	%zmm12, %zmm3, %zmm3;	vpandq	%zmm8, %zmm3, %zmm2
	vpsrlq		$32, %zmm3, %zmm13;		vpaddq	%zmm13, %zmm4, %zmm4;	vpandq	%zmm8, %zmm4, %zmm3
	vpsrlq		$32, %zmm4, %zmm14;		vpaddq	%zmm14, %zmm5, %zmm5;	vpandq	%zmm8, %zmm5, %zmm4
	vpsrlq		$32, %zmm5, %zmm15;		vpaddq	%zmm15, %zmm6, %zmm6;	vpandq	%zmm8, %zmm6, %zmm5
	vpsrlq		$32, %zmm6, %zmm16;		vpaddq	%zmm16, %zmm7, %zmm7;	vpandq	%zmm8, %zmm7, %zmm6
	vpsrlq		$32, %zmm7, %zmm7

	// Conditional subtraction of the modulus

	movq	Z0, Y0
	movq	Z1, Y1
	movq	Z2, Y2
	movq	Z3, Y3

	subq	0*8(PM), Y0
	sbbq	1*8(PM), Y1
	sbbq	2*8(PM), Y2
	sbbq	3*8(PM), Y3

	cmovncq	Y0, Z0
	cmovncq	Y1, Z1
	cmovncq	Y2, Z2
	cmovncq	Y3, Z3

	// Restore registers

	popq	LEN
	mov	0(%rsp), PZ
	mov	8(%rsp), PY

	// Store output

	movq	Z0, 0*8(PZ, LEN)
	movq	Z1, 1*8(PZ, LEN)
	movq	Z2, 2*8(PZ, LEN)
	movq	Z3, 3*8(PZ, LEN)

	addq	$32, LEN

	// Save registers

	pushq	LEN

	// Load inputs

	movq	0*8(PX,LEN), MUL
	movq	0*8(PY,LEN), Y0
	movq	1*8(PY,LEN), Y1
	movq	2*8(PY,LEN), Y2
	movq	3*8(PY,LEN), Y3

	//////////////////////////////////////////////////
	// Process doubleword 5 of x
	//////////////////////////////////////////////////

	vpmuludq	%zmm21, %zmm24, %zmm10;		vpaddq	%zmm10, %zmm0, %zmm0;
	vpmuludq	%zmm21, %zmm25, %zmm11;		vpaddq	%zmm11, %zmm1, %zmm1;
	vpmuludq	%zmm21, %zmm26, %zmm12;		vpaddq	%zmm12, %zmm2, %zmm2;
	vpmuludq	%zmm21, %zmm27, %zmm13;		vpaddq	%zmm13, %zmm3, %zmm3;

	mulxq	Y0, Z1, Z2
	mulxq	Y1, PL, Z3;		addq	PL, Z2
	mulxq	Y2, PL, Z4;		adcq	PL, Z3
	mulxq	Y3, PL, Z0;		adcq	PL, Z4;	adcq	$0, Z0
	movq	Z1, MUL
	mulxq	4*8(PM), MUL, PH

	vpmuludq	%zmm21, %zmm28, %zmm14;		vpaddq	%zmm14, %zmm4, %zmm4;
	vpmuludq	%zmm21, %zmm29, %zmm15;		vpaddq	%zmm15, %zmm5, %zmm5;
	vpmuludq	%zmm21, %zmm30, %zmm16;		vpaddq	%zmm16, %zmm6, %zmm6;
	vpmuludq	%zmm21, %zmm31, %zmm17;		vpaddq	%zmm17, %zmm7, %zmm7;

	vpmuludq	8*4(PM){1to8}, %zmm0, %zmm9;	// Compute reduction multipliers

	mulxq	0*8(PM), PL, PH;	addq	PL, Z1;	adcq	PH, Z2
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z3;	adcq	PH, Z4;	adcq	$0, Z0
	mulxq	1*8(PM), PL, PH;	addq	PL, Z2;	adcq	PH, Z3
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1

	movq	1*8(PX, LEN), MUL

	// Move high dwords to zmm10-16, add each to the corresponding low dword (propagate 32-bit carries)

	vpsrlq		$32, %zmm0, %zmm10;		vpandq	%zmm8, %zmm0, %zmm0;	vpaddq	%zmm10, %zmm1, %zmm1;
	vpsrlq		$32, %zmm1, %zmm11;		vpandq	%zmm8, %zmm1, %zmm1;	vpaddq	%zmm11, %zmm2, %zmm2;
	vpsrlq		$32, %zmm2, %zmm12;		vpandq	%zmm8, %zmm2, %zmm2;	vpaddq	%zmm12, %zmm3, %zmm3;
	vpsrlq		$32, %zmm3, %zmm13;		vpandq	%zmm8, %zmm3, %zmm3;	vpaddq	%zmm13, %zmm4, %zmm4;

	mulxq	Y0, PL, PH;		addq	PL, Z2;	adcq	PH, Z3
	mulxq	Y2, PL, PH;		adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1
	mulxq	Y1, PL, PH;		addq	PL, Z3;	adcq	PH, Z4
	mulxq	Y3, PL, PH;		adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	movq	Z2, MUL
	mulxq	4*8(PM), MUL, PH

	vpsrlq		$32, %zmm4, %zmm14;		vpandq	%zmm8, %zmm4, %zmm4;	vpaddq	%zmm14, %zmm5, %zmm5;
	vpsrlq		$32, %zmm5, %zmm15;		vpandq	%zmm8, %zmm5, %zmm5;	vpaddq	%zmm15, %zmm6, %zmm6;
	vpsrlq		$32, %zmm6, %zmm16;		vpandq	%zmm8, %zmm6, %zmm6;	vpaddq	%zmm16, %zmm7, %zmm7;
	// zmm7 keeps all 64 bits

	mulxq	0*8(PM), PL, PH;	addq	PL, Z2;	adcq	PH, Z3
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1
	mulxq	1*8(PM), PL, PH;	addq	PL, Z3;	adcq	PH, Z4
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2

	movq	2*8(PX, LEN), MUL

	vpmuludq	0*4(PM){1to8}, %zmm9, %zmm10;	vpaddq	%zmm10, %zmm0, %zmm0;	// Low dword of zmm0 is zero
	vpmuludq	1*4(PM){1to8}, %zmm9, %zmm11;	vpaddq	%zmm11, %zmm1, %zmm1;
	vpmuludq	2*4(PM){1to8}, %zmm9, %zmm12;	vpaddq	%zmm12, %zmm2, %zmm2;
	vpmuludq	3*4(PM){1to8}, %zmm9, %zmm13;	vpaddq	%zmm13, %zmm3, %zmm3;

	mulxq	Y0, PL, PH;		addq	PL, Z3;	adcq	PH, Z4
	mulxq	Y2, PL, PH;		adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	mulxq	Y1, PL, PH;		addq	PL, Z4;	adcq	PH, Z0
	mulxq	Y3, PL, PH;		adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	movq	Z3, MUL
	mulxq	4*8(PM), MUL, PH

	vpmuludq	4*4(PM){1to8}, %zmm9, %zmm14;	vpaddq	%zmm14, %zmm4, %zmm4;
	vpmuludq	5*4(PM){1to8}, %zmm9, %zmm15;	vpaddq	%zmm15, %zmm5, %zmm5;
	vpmuludq	6*4(PM){1to8}, %zmm9, %zmm16;	vpaddq	%zmm16, %zmm6, %zmm6;
	vpmuludq	7*4(PM){1to8}, %zmm9, %zmm17;	vpaddq	%zmm17, %zmm7, %zmm7;

	mulxq	0*8(PM), PL, PH;	addq	PL, Z3;	adcq	PH, Z4
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	mulxq	1*8(PM), PL, PH;	addq	PL, Z4;	adcq	PH, Z0
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3

	movq	3*8(PX, LEN), MUL

	// Propagate carries and shift down by one dword

	vpsrlq		$32, %zmm0, %zmm10;		vpaddq	%zmm10, %zmm1, %zmm1;	vpandq	%zmm8, %zmm1, %zmm0
	vpsrlq		$32, %zmm1, %zmm11;		vpaddq	%zmm11, %zmm2, %zmm2;	vpandq	%zmm8, %zmm2, %zmm1
	vpsrlq		$32, %zmm2, %zmm12;		vpaddq	%zmm12, %zmm3, %zmm3;	vpandq	%zmm8, %zmm3, %zmm2
	vpsrlq		$32, %zmm3, %zmm13;		vpaddq	%zmm13, %zmm4, %zmm4;	vpandq	%zmm8, %zmm4, %zmm3

	mulxq	Y0, PL, PH;		addq	PL, Z4;	adcq	PH, Z0
	mulxq	Y2, PL, PH;		adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	mulxq	Y1, PL, PH;		addq	PL, Z0;	adcq	PH, Z1
	mulxq	Y3, PL, PH;		adcq	PL, Z2;	adcq	PH, Z3;	adcq	$0, Z4
	movq	Z4, MUL
	mulxq	4*8(PM), MUL, PH

	vpsrlq		$32, %zmm4, %zmm14;		vpaddq	%zmm14, %zmm5, %zmm5;	vpandq	%zmm8, %zmm5, %zmm4
	vpsrlq		$32, %zmm5, %zmm15;		vpaddq	%zmm15, %zmm6, %zmm6;	vpandq	%zmm8, %zmm6, %zmm5
	vpsrlq		$32, %zmm6, %zmm16;		vpaddq	%zmm16, %zmm7, %zmm7;	vpandq	%zmm8, %zmm7, %zmm6
	vpsrlq		$32, %zmm7, %zmm7

	mulxq	0*8(PM), PL, PH;	addq	PL, Z4;	adcq	PH, Z0
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	mulxq	1*8(PM), PL, PH;	addq	PL, Z0;	adcq	PH, Z1
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z2;	adcq	PH, Z3;	adcq	$0, Z4

	//////////////////////////////////////////////////
	// Process doubleword 6 of x
	//////////////////////////////////////////////////

	vpmuludq	%zmm22, %zmm24, %zmm10;		vpaddq	%zmm10, %zmm0, %zmm0;
	vpmuludq	%zmm22, %zmm25, %zmm11;		vpaddq	%zmm11, %zmm1, %zmm1;
	vpmuludq	%zmm22, %zmm26, %zmm12;		vpaddq	%zmm12, %zmm2, %zmm2;
	vpmuludq	%zmm22, %zmm27, %zmm13;		vpaddq	%zmm13, %zmm3, %zmm3;
	vpmuludq	%zmm22, %zmm28, %zmm14;		vpaddq	%zmm14, %zmm4, %zmm4;
	vpmuludq	%zmm22, %zmm29, %zmm15;		vpaddq	%zmm15, %zmm5, %zmm5;
	vpmuludq	%zmm22, %zmm30, %zmm16;		vpaddq	%zmm16, %zmm6, %zmm6;
	vpmuludq	%zmm22, %zmm31, %zmm17;		vpaddq	%zmm17, %zmm7, %zmm7;

	vpmuludq	8*4(PM){1to8}, %zmm0, %zmm9;	// Compute reduction multipliers

	// Conditional subtraction of the modulus

	movq	Z0, Y0
	movq	Z1, Y1
	movq	Z2, Y2
	movq	Z3, Y3

	subq	0*8(PM), Y0
	sbbq	1*8(PM), Y1
	sbbq	2*8(PM), Y2
	sbbq	3*8(PM), Y3

	cmovncq	Y0, Z0
	cmovncq	Y1, Z1
	cmovncq	Y2, Z2
	cmovncq	Y3, Z3

	// Restore registers

	popq	LEN
	mov	0(%rsp), PZ
	mov	8(%rsp), PY

	// Store output

	movq	Z0, 0*8(PZ, LEN)
	movq	Z1, 1*8(PZ, LEN)
	movq	Z2, 2*8(PZ, LEN)
	movq	Z3, 3*8(PZ, LEN)

	addq	$32, LEN

	// Save registers

	pushq	LEN

	// Load inputs

	movq	0*8(PX,LEN), MUL
	movq	0*8(PY,LEN), Y0
	movq	1*8(PY,LEN), Y1
	movq	2*8(PY,LEN), Y2
	movq	3*8(PY,LEN), Y3

	// Move high dwords to zmm10-16, add each to the corresponding low dword (propagate 32-bit carries)

	vpsrlq		$32, %zmm0, %zmm10;		vpandq	%zmm8, %zmm0, %zmm0;	vpaddq	%zmm10, %zmm1, %zmm1;
	vpsrlq		$32, %zmm1, %zmm11;		vpandq	%zmm8, %zmm1, %zmm1;	vpaddq	%zmm11, %zmm2, %zmm2;
	vpsrlq		$32, %zmm2, %zmm12;		vpandq	%zmm8, %zmm2, %zmm2;	vpaddq	%zmm12, %zmm3, %zmm3;
	vpsrlq		$32, %zmm3, %zmm13;		vpandq	%zmm8, %zmm3, %zmm3;	vpaddq	%zmm13, %zmm4, %zmm4;

	mulxq	Y0, Z1, Z2
	mulxq	Y1, PL, Z3;		addq	PL, Z2
	mulxq	Y2, PL, Z4;		adcq	PL, Z3
	mulxq	Y3, PL, Z0;		adcq	PL, Z4;	adcq	$0, Z0
	movq	Z1, MUL
	mulxq	4*8(PM), MUL, PH

	vpsrlq		$32, %zmm4, %zmm14;		vpandq	%zmm8, %zmm4, %zmm4;	vpaddq	%zmm14, %zmm5, %zmm5;
	vpsrlq		$32, %zmm5, %zmm15;		vpandq	%zmm8, %zmm5, %zmm5;	vpaddq	%zmm15, %zmm6, %zmm6;
	vpsrlq		$32, %zmm6, %zmm16;		vpandq	%zmm8, %zmm6, %zmm6;	vpaddq	%zmm16, %zmm7, %zmm7;
	// zmm7 keeps all 64 bits

	mulxq	0*8(PM), PL, PH;	addq	PL, Z1;	adcq	PH, Z2
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z3;	adcq	PH, Z4;	adcq	$0, Z0
	mulxq	1*8(PM), PL, PH;	addq	PL, Z2;	adcq	PH, Z3
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1

	movq	1*8(PX, LEN), MUL

	vpmuludq	0*4(PM){1to8}, %zmm9, %zmm10;	vpaddq	%zmm10, %zmm0, %zmm0;	// Low dword of zmm0 is zero
	vpmuludq	1*4(PM){1to8}, %zmm9, %zmm11;	vpaddq	%zmm11, %zmm1, %zmm1;
	vpmuludq	2*4(PM){1to8}, %zmm9, %zmm12;	vpaddq	%zmm12, %zmm2, %zmm2;
	vpmuludq	3*4(PM){1to8}, %zmm9, %zmm13;	vpaddq	%zmm13, %zmm3, %zmm3;

	mulxq	Y0, PL, PH;		addq	PL, Z2;	adcq	PH, Z3
	mulxq	Y2, PL, PH;		adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1
	mulxq	Y1, PL, PH;		addq	PL, Z3;	adcq	PH, Z4
	mulxq	Y3, PL, PH;		adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	movq	Z2, MUL
	mulxq	4*8(PM), MUL, PH

	vpmuludq	4*4(PM){1to8}, %zmm9, %zmm14;	vpaddq	%zmm14, %zmm4, %zmm4;
	vpmuludq	5*4(PM){1to8}, %zmm9, %zmm15;	vpaddq	%zmm15, %zmm5, %zmm5;
	vpmuludq	6*4(PM){1to8}, %zmm9, %zmm16;	vpaddq	%zmm16, %zmm6, %zmm6;
	vpmuludq	7*4(PM){1to8}, %zmm9, %zmm17;	vpaddq	%zmm17, %zmm7, %zmm7;

	mulxq	0*8(PM), PL, PH;	addq	PL, Z2;	adcq	PH, Z3
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1
	mulxq	1*8(PM), PL, PH;	addq	PL, Z3;	adcq	PH, Z4
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2

	movq	2*8(PX, LEN), MUL

	// Propagate carries and shift down by one dword

	vpsrlq		$32, %zmm0, %zmm10;		vpaddq	%zmm10, %zmm1, %zmm1;	vpandq	%zmm8, %zmm1, %zmm0
	vpsrlq		$32, %zmm1, %zmm11;		vpaddq	%zmm11, %zmm2, %zmm2;	vpandq	%zmm8, %zmm2, %zmm1
	vpsrlq		$32, %zmm2, %zmm12;		vpaddq	%zmm12, %zmm3, %zmm3;	vpandq	%zmm8, %zmm3, %zmm2
	vpsrlq		$32, %zmm3, %zmm13;		vpaddq	%zmm13, %zmm4, %zmm4;	vpandq	%zmm8, %zmm4, %zmm3

	mulxq	Y0, PL, PH;		addq	PL, Z3;	adcq	PH, Z4
	mulxq	Y2, PL, PH;		adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	mulxq	Y1, PL, PH;		addq	PL, Z4;	adcq	PH, Z0
	mulxq	Y3, PL, PH;		adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	movq	Z3, MUL
	mulxq	4*8(PM), MUL, PH

	vpsrlq		$32, %zmm4, %zmm14;		vpaddq	%zmm14, %zmm5, %zmm5;	vpandq	%zmm8, %zmm5, %zmm4
	vpsrlq		$32, %zmm5, %zmm15;		vpaddq	%zmm15, %zmm6, %zmm6;	vpandq	%zmm8, %zmm6, %zmm5
	vpsrlq		$32, %zmm6, %zmm16;		vpaddq	%zmm16, %zmm7, %zmm7;	vpandq	%zmm8, %zmm7, %zmm6
	vpsrlq		$32, %zmm7, %zmm7

	mulxq	0*8(PM), PL, PH;	addq	PL, Z3;	adcq	PH, Z4
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	mulxq	1*8(PM), PL, PH;	addq	PL, Z4;	adcq	PH, Z0
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3

	movq	3*8(PX, LEN), MUL

	//////////////////////////////////////////////////
	// Process doubleword 7 of x
	//////////////////////////////////////////////////

	vpmuludq	%zmm23, %zmm24, %zmm10;		vpaddq	%zmm10, %zmm0, %zmm0;
	vpmuludq	%zmm23, %zmm25, %zmm11;		vpaddq	%zmm11, %zmm1, %zmm1;
	vpmuludq	%zmm23, %zmm26, %zmm12;		vpaddq	%zmm12, %zmm2, %zmm2;
	vpmuludq	%zmm23, %zmm27, %zmm13;		vpaddq	%zmm13, %zmm3, %zmm3;

	mulxq	Y0, PL, PH;		addq	PL, Z4;	adcq	PH, Z0
	mulxq	Y2, PL, PH;		adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	mulxq	Y1, PL, PH;		addq	PL, Z0;	adcq	PH, Z1
	mulxq	Y3, PL, PH;		adcq	PL, Z2;	adcq	PH, Z3;	adcq	$0, Z4
	movq	Z4, MUL
	mulxq	4*8(PM), MUL, PH

	vpmuludq	%zmm23, %zmm28, %zmm14;		vpaddq	%zmm14, %zmm4, %zmm4;
	vpmuludq	%zmm23, %zmm29, %zmm15;		vpaddq	%zmm15, %zmm5, %zmm5;
	vpmuludq	%zmm23, %zmm30, %zmm16;		vpaddq	%zmm16, %zmm6, %zmm6;
	vpmuludq	%zmm23, %zmm31, %zmm17;		vpaddq	%zmm17, %zmm7, %zmm7;

	vpmuludq	8*4(PM){1to8}, %zmm0, %zmm9;	// Compute reduction multipliers

	mulxq	0*8(PM), PL, PH;	addq	PL, Z4;	adcq	PH, Z0
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	mulxq	1*8(PM), PL, PH;	addq	PL, Z0;	adcq	PH, Z1
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z2;	adcq	PH, Z3;	adcq	$0, Z4

	// Move high dwords to zmm10-16, add each to the corresponding low dword (propagate 32-bit carries)

	vpsrlq		$32, %zmm0, %zmm10;		vpandq	%zmm8, %zmm0, %zmm0;	vpaddq	%zmm10, %zmm1, %zmm1;
	vpsrlq		$32, %zmm1, %zmm11;		vpandq	%zmm8, %zmm1, %zmm1;	vpaddq	%zmm11, %zmm2, %zmm2;
	vpsrlq		$32, %zmm2, %zmm12;		vpandq	%zmm8, %zmm2, %zmm2;	vpaddq	%zmm12, %zmm3, %zmm3;
	vpsrlq		$32, %zmm3, %zmm13;		vpandq	%zmm8, %zmm3, %zmm3;	vpaddq	%zmm13, %zmm4, %zmm4;
	vpsrlq		$32, %zmm4, %zmm14;		vpandq	%zmm8, %zmm4, %zmm4;	vpaddq	%zmm14, %zmm5, %zmm5;
	vpsrlq		$32, %zmm5, %zmm15;		vpandq	%zmm8, %zmm5, %zmm5;	vpaddq	%zmm15, %zmm6, %zmm6;
	vpsrlq		$32, %zmm6, %zmm16;		vpandq	%zmm8, %zmm6, %zmm6;	vpaddq	%zmm16, %zmm7, %zmm7;
	// zmm7 keeps all 64 bits

	// Conditional subtraction of the modulus

	movq	Z0, Y0
	movq	Z1, Y1
	movq	Z2, Y2
	movq	Z3, Y3

	subq	0*8(PM), Y0
	sbbq	1*8(PM), Y1
	sbbq	2*8(PM), Y2
	sbbq	3*8(PM), Y3

	cmovncq	Y0, Z0
	cmovncq	Y1, Z1
	cmovncq	Y2, Z2
	cmovncq	Y3, Z3

	// Restore registers

	popq	LEN
	mov	0(%rsp), PZ
	mov	8(%rsp), PY

	// Store output

	movq	Z0, 0*8(PZ, LEN)
	movq	Z1, 1*8(PZ, LEN)
	movq	Z2, 2*8(PZ, LEN)
	movq	Z3, 3*8(PZ, LEN)

	addq	$32, LEN

	// Save registers

	pushq	LEN

	// Load inputs

	movq	0*8(PX,LEN), MUL
	movq	0*8(PY,LEN), Y0
	movq	1*8(PY,LEN), Y1
	movq	2*8(PY,LEN), Y2
	movq	3*8(PY,LEN), Y3

	vpmuludq	0*4(PM){1to8}, %zmm9, %zmm10;	vpaddq	%zmm10, %zmm0, %zmm0;	// Low dword of zmm0 is zero
	vpmuludq	1*4(PM){1to8}, %zmm9, %zmm11;	vpaddq	%zmm11, %zmm1, %zmm1;
	vpmuludq	2*4(PM){1to8}, %zmm9, %zmm12;	vpaddq	%zmm12, %zmm2, %zmm2;
	vpmuludq	3*4(PM){1to8}, %zmm9, %zmm13;	vpaddq	%zmm13, %zmm3, %zmm3;

	mulxq	Y0, Z1, Z2
	mulxq	Y1, PL, Z3;		addq	PL, Z2
	mulxq	Y2, PL, Z4;		adcq	PL, Z3
	mulxq	Y3, PL, Z0;		adcq	PL, Z4;	adcq	$0, Z0
	movq	Z1, MUL
	mulxq	4*8(PM), MUL, PH

	vpmuludq	4*4(PM){1to8}, %zmm9, %zmm14;	vpaddq	%zmm14, %zmm4, %zmm4;
	vpmuludq	5*4(PM){1to8}, %zmm9, %zmm15;	vpaddq	%zmm15, %zmm5, %zmm5;
	vpmuludq	6*4(PM){1to8}, %zmm9, %zmm16;	vpaddq	%zmm16, %zmm6, %zmm6;
	vpmuludq	7*4(PM){1to8}, %zmm9, %zmm17;	vpaddq	%zmm17, %zmm7, %zmm7;

	mulxq	0*8(PM), PL, PH;	addq	PL, Z1;	adcq	PH, Z2
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z3;	adcq	PH, Z4;	adcq	$0, Z0
	mulxq	1*8(PM), PL, PH;	addq	PL, Z2;	adcq	PH, Z3
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1

	movq	1*8(PX, LEN), MUL

	// Propagate carries and shift down by one dword

	vpsrlq		$32, %zmm0, %zmm10;		vpaddq	%zmm10, %zmm1, %zmm1;	vpandq	%zmm8, %zmm1, %zmm0
	vpsrlq		$32, %zmm1, %zmm11;		vpaddq	%zmm11, %zmm2, %zmm2;	vpandq	%zmm8, %zmm2, %zmm1
	vpsrlq		$32, %zmm2, %zmm12;		vpaddq	%zmm12, %zmm3, %zmm3;	vpandq	%zmm8, %zmm3, %zmm2
	vpsrlq		$32, %zmm3, %zmm13;		vpaddq	%zmm13, %zmm4, %zmm4;	vpandq	%zmm8, %zmm4, %zmm3
	vpsrlq		$32, %zmm4, %zmm14;		vpaddq	%zmm14, %zmm5, %zmm5;	vpandq	%zmm8, %zmm5, %zmm4
	vpsrlq		$32, %zmm5, %zmm15;		vpaddq	%zmm15, %zmm6, %zmm6;	vpandq	%zmm8, %zmm6, %zmm5
	vpsrlq		$32, %zmm6, %zmm16;		vpaddq	%zmm16, %zmm7, %zmm7;	vpandq	%zmm8, %zmm7, %zmm6
	vpsrlq		$32, %zmm7, %zmm7

	mulxq	Y0, PL, PH;		addq	PL, Z2;	adcq	PH, Z3
	mulxq	Y2, PL, PH;		adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1
	mulxq	Y1, PL, PH;		addq	PL, Z3;	adcq	PH, Z4
	mulxq	Y3, PL, PH;		adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	movq	Z2, MUL
	mulxq	4*8(PM), MUL, PH
	mulxq	0*8(PM), PL, PH;	addq	PL, Z2;	adcq	PH, Z3
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1
	mulxq	1*8(PM), PL, PH;	addq	PL, Z3;	adcq	PH, Z4
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2

	movq	2*8(PX, LEN), MUL

	//////////////////////////////////////////////////
	// Conditional subtraction of the modulus
	//////////////////////////////////////////////////

	vpermd	0*4(PM){1to16}, %zmm8, %zmm10{%k1}{z}
	vpermd	1*4(PM){1to16}, %zmm8, %zmm11{%k1}{z}
	vpermd	2*4(PM){1to16}, %zmm8, %zmm12{%k1}{z}
	vpermd	3*4(PM){1to16}, %zmm8, %zmm13{%k1}{z}
	vpermd	4*4(PM){1to16}, %zmm8, %zmm14{%k1}{z}
	vpermd	5*4(PM){1to16}, %zmm8, %zmm15{%k1}{z}
	vpermd	6*4(PM){1to16}, %zmm8, %zmm16{%k1}{z}
	vpermd	7*4(PM){1to16}, %zmm8, %zmm17{%k1}{z}

	vpsubq	%zmm10, %zmm0, %zmm10;					vpsrlq	$63, %zmm10, %zmm20;	vpandq	%zmm8, %zmm10, %zmm10
	vpsubq	%zmm11, %zmm1, %zmm11;	vpsubq	%zmm20, %zmm11, %zmm11;	vpsrlq	$63, %zmm11, %zmm21;	vpandq	%zmm8, %zmm11, %zmm11
	vpsubq	%zmm12, %zmm2, %zmm12;	vpsubq	%zmm21, %zmm12, %zmm12;	vpsrlq	$63, %zmm12, %zmm22;	vpandq	%zmm8, %zmm12, %zmm12
	vpsubq	%zmm13, %zmm3, %zmm13;	vpsubq	%zmm22, %zmm13, %zmm13;	vpsrlq	$63, %zmm13, %zmm23;	vpandq	%zmm8, %zmm13, %zmm13
	vpsubq	%zmm14, %zmm4, %zmm14;	vpsubq	%zmm23, %zmm14, %zmm14;	vpsrlq	$63, %zmm14, %zmm24;	vpandq	%zmm8, %zmm14, %zmm14
	vpsubq	%zmm15, %zmm5, %zmm15;	vpsubq	%zmm24, %zmm15, %zmm15;	vpsrlq	$63, %zmm15, %zmm25;	vpandq	%zmm8, %zmm15, %zmm15
	vpsubq	%zmm16, %zmm6, %zmm16;	vpsubq	%zmm25, %zmm16, %zmm16;	vpsrlq	$63, %zmm16, %zmm26;	vpandq	%zmm8, %zmm16, %zmm16
	vpsubq	%zmm17, %zmm7, %zmm17;	vpsubq	%zmm26, %zmm17, %zmm17;

	mulxq	Y0, PL, PH;		addq	PL, Z3;	adcq	PH, Z4
	mulxq	Y2, PL, PH;		adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	mulxq	Y1, PL, PH;		addq	PL, Z4;	adcq	PH, Z0
	mulxq	Y3, PL, PH;		adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	movq	Z3, MUL

	vpmovq2m	%zmm17, %k2

	mulxq	4*8(PM), MUL, PH

	knotb		%k2, %k2

	vmovdqu64	%zmm10, %zmm0{%k2}
	vmovdqu64	%zmm11, %zmm1{%k2}
	vmovdqu64	%zmm12, %zmm2{%k2}
	vmovdqu64	%zmm13, %zmm3{%k2}
	vmovdqu64	%zmm14, %zmm4{%k2}
	vmovdqu64	%zmm15, %zmm5{%k2}
	vmovdqu64	%zmm16, %zmm6{%k2}
	vmovdqu64	%zmm17, %zmm7{%k2}

	mulxq	0*8(PM), PL, PH;	addq	PL, Z3;	adcq	PH, Z4
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	mulxq	1*8(PM), PL, PH;	addq	PL, Z4;	adcq	PH, Z0
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3

	movq	3*8(PX, LEN), MUL

	//////////////////////////////////////////////////
	// Transpose results back
	//////////////////////////////////////////////////

	vmovdqa64	pattern1(%rip), %zmm11
	vmovdqa64	pattern2(%rip), %zmm12
	vmovdqa64	pattern3(%rip), %zmm13
	vmovdqa64	pattern4(%rip), %zmm14

	// Step 1

	vpsllq		$32, %zmm1, %zmm1;	vporq	%zmm1, %zmm0, %zmm0
	vpsllq		$32, %zmm3, %zmm3;	vporq	%zmm3, %zmm2, %zmm1
	vpsllq		$32, %zmm5, %zmm5;	vporq	%zmm5, %zmm4, %zmm2
	vpsllq		$32, %zmm7, %zmm7;	vporq	%zmm7, %zmm6, %zmm3

	mulxq	Y0, PL, PH;		addq	PL, Z4;	adcq	PH, Z0
	mulxq	Y2, PL, PH;		adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	mulxq	Y1, PL, PH;		addq	PL, Z0;	adcq	PH, Z1
	mulxq	Y3, PL, PH;		adcq	PL, Z2;	adcq	PH, Z3;	adcq	$0, Z4
	movq	Z4, MUL

	// Step 2

	vmovdqu64	%zmm0, %zmm4
	vmovdqu64	%zmm2, %zmm6

	vpermt2q	%zmm1, %zmm11, %zmm0
	vpermt2q	%zmm4, %zmm12, %zmm1
	vpermt2q	%zmm3, %zmm11, %zmm2
	vpermt2q	%zmm6, %zmm12, %zmm3

	mulxq	4*8(PM), MUL, PH

	// Step 3

	vmovdqu64	%zmm0, %zmm4
	vmovdqu64	%zmm1, %zmm5

	vpermt2q	%zmm2, %zmm13, %zmm0
	vpermt2q	%zmm4, %zmm14, %zmm2
	vpermt2q	%zmm3, %zmm13, %zmm1
	vpermt2q	%zmm5, %zmm14, %zmm3

	mulxq	0*8(PM), PL, PH;	addq	PL, Z4;	adcq	PH, Z0
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	mulxq	1*8(PM), PL, PH;	addq	PL, Z0;	adcq	PH, Z1
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z2;	adcq	PH, Z3;	adcq	$0, Z4

	// Conditional subtraction of the modulus

	movq	Z0, Y0
	movq	Z1, Y1
	movq	Z2, Y2
	movq	Z3, Y3

	subq	0*8(PM), Y0
	sbbq	1*8(PM), Y1
	sbbq	2*8(PM), Y2
	sbbq	3*8(PM), Y3

	cmovncq	Y0, Z0
	cmovncq	Y1, Z1
	cmovncq	Y2, Z2
	cmovncq	Y3, Z3

	// Restore registers

	popq	LEN
	popq	PZ
	popq	PY

	// Store output

	movq	Z0, 0*8(PZ, LEN)
	movq	Z1, 1*8(PZ, LEN)
	movq	Z2, 2*8(PZ, LEN)
	movq	Z3, 3*8(PZ, LEN)

	addq	$32, LEN

	//////////////////////////////////////////////////
	// Save AVX-512 results
	//////////////////////////////////////////////////

	vmovdqu64	%zmm0, 0*64(PZ, LEN)
	vmovdqu64	%zmm2, 1*64(PZ, LEN)
	vmovdqu64	%zmm1, 2*64(PZ, LEN)
	vmovdqu64	%zmm3, 3*64(PZ, LEN)

	addq	$256, LEN

	jnz	Loop16

Done:
#ifndef WIN64
	popq	%r15
	popq	%r14
	popq	%r13
	popq	%r12

	popq	%rbp
	popq	%rbx
#else
	// Restore xmm6-xmm15

	movdqu	$0*16(%rsp), xmm6
	movdqu	$1*16(%rsp), xmm7
	movdqu	$2*16(%rsp), xmm8
	movdqu	$3*16(%rsp), xmm9
	movdqu	$4*16(%rsp), xmm10
	movdqu	$5*16(%rsp), xmm11
	movdqu	$6*16(%rsp), xmm12
	movdqu	$7*16(%rsp), xmm13
	movdqu	$8*16(%rsp), xmm14
	movdqu	$9*16(%rsp), xmm15

	addq	$10*16, %rsp

	popq	%r15
	popq	%r14
	popq	%r13
	popq	%r12

	movq	1*8(%rsp), %rbx
	movq	2*8(%rsp), %rbp
	movq	3*8(%rsp), %rsi
	movq	4*8(%rsp), %rdi
#endif
	ret

.p2align 6,,63
Loop1:
	//////////////////////////////////////////////////
	// Process 1-15 elements to leave a multiple of 16
	//////////////////////////////////////////////////

	// Save registers

	pushq	PY
	pushq	PZ
	pushq	LEN

	// Load inputs

	movq	0*8(PX,LEN), MUL
	movq	0*8(PY,LEN), Y0
	movq	1*8(PY,LEN), Y1
	movq	2*8(PY,LEN), Y2
	movq	3*8(PY,LEN), Y3

	// Montgomery multiplication

	mulxq	Y0, Z1, Z2
	mulxq	Y1, PL, Z3;		addq	PL, Z2
	mulxq	Y2, PL, Z4;		adcq	PL, Z3
	mulxq	Y3, PL, Z0;		adcq	PL, Z4;	adcq	$0, Z0
	movq	Z1, MUL
	mulxq	4*8(PM), MUL, PH
	mulxq	0*8(PM), PL, PH;	addq	PL, Z1;	adcq	PH, Z2
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z3;	adcq	PH, Z4;	adcq	$0, Z0
	mulxq	1*8(PM), PL, PH;	addq	PL, Z2;	adcq	PH, Z3
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1

	movq	1*8(PX, LEN), MUL

	mulxq	Y0, PL, PH;		addq	PL, Z2;	adcq	PH, Z3
	mulxq	Y2, PL, PH;		adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1
	mulxq	Y1, PL, PH;		addq	PL, Z3;	adcq	PH, Z4
	mulxq	Y3, PL, PH;		adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	movq	Z2, MUL
	mulxq	4*8(PM), MUL, PH
	mulxq	0*8(PM), PL, PH;	addq	PL, Z2;	adcq	PH, Z3
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z4;	adcq	PH, Z0;	adcq	$0, Z1
	mulxq	1*8(PM), PL, PH;	addq	PL, Z3;	adcq	PH, Z4
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2

	movq	2*8(PX, LEN), MUL

	mulxq	Y0, PL, PH;		addq	PL, Z3;	adcq	PH, Z4
	mulxq	Y2, PL, PH;		adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	mulxq	Y1, PL, PH;		addq	PL, Z4;	adcq	PH, Z0
	mulxq	Y3, PL, PH;		adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	movq	Z3, MUL
	mulxq	4*8(PM), MUL, PH
	mulxq	0*8(PM), PL, PH;	addq	PL, Z3;	adcq	PH, Z4
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z0;	adcq	PH, Z1;	adcq	$0, Z2
	mulxq	1*8(PM), PL, PH;	addq	PL, Z4;	adcq	PH, Z0
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3

	movq	3*8(PX, LEN), MUL

	mulxq	Y0, PL, PH;		addq	PL, Z4;	adcq	PH, Z0
	mulxq	Y2, PL, PH;		adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	mulxq	Y1, PL, PH;		addq	PL, Z0;	adcq	PH, Z1
	mulxq	Y3, PL, PH;		adcq	PL, Z2;	adcq	PH, Z3;	adcq	$0, Z4
	movq	Z4, MUL
	mulxq	4*8(PM), MUL, PH
	mulxq	0*8(PM), PL, PH;	addq	PL, Z4;	adcq	PH, Z0
	mulxq	2*8(PM), PL, PH;	adcq	PL, Z1;	adcq	PH, Z2;	adcq	$0, Z3
	mulxq	1*8(PM), PL, PH;	addq	PL, Z0;	adcq	PH, Z1
	mulxq	3*8(PM), PL, PH;	adcq	PL, Z2;	adcq	PH, Z3;	adcq	$0, Z4

	// Conditional subtraction of the modulus

	movq	Z0, Y0
	movq	Z1, Y1
	movq	Z2, Y2
	movq	Z3, Y3

	subq	0*8(PM), Y0
	sbbq	1*8(PM), Y1
	sbbq	2*8(PM), Y2
	sbbq	3*8(PM), Y3
	sbbq	Z4, Z4

	cmovzq	Y0, Z0
	cmovzq	Y1, Z1
	cmovzq	Y2, Z2
	cmovzq	Y3, Z3

	// Restore registers

	popq	LEN
	popq	PZ
	popq	PY

	// Store output

	movq	Z0, 0*8(PZ, LEN)
	movq	Z1, 1*8(PZ, LEN)
	movq	Z2, 2*8(PZ, LEN)
	movq	Z3, 3*8(PZ, LEN)

	// Loop

	addq	$0x020, LEN
	test	$0x1e0, LEN
	jnz	Loop1

	jmp Blocksof16

	//////////////////////////////////////////////////
	// Patterns used to transpose results
	//////////////////////////////////////////////////

.p2align 6,,63	// 512-bit alignment

pattern1:
.quad	 0,  8,  1,  9,  2, 10,  3, 11

pattern2:
.quad	12,  4, 13,  5, 14,  6, 15,  7

pattern3:
.quad	 0,  1, 8,  9,  2,  3,  10, 11

pattern4:
.quad	12, 13,  4,  5, 14, 15,  6,  7

// No executable stack
.section .note.GNU-stack
