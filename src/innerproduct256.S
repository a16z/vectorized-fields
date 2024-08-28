/*
 * void innerproduct256(uint64_t z[4], const uint64_t *x, const uint64_t *y, uint32_t xy_len)
 *
 * Copyright (C) 2024 Dag Arne Osvik
 *
 * AVX-512 assembler implementation of integer inner product.
 *
 * Inputs x and y are two vectors of little-endian 256-bit elements
 * Vector length is up to 2^32-1 elements.
 * Output z is a little-endian 544-bit inner product.
 *
 * z will always be overwritten
 * xy_len can have any 32-bit value
 * x and y must each point to a vector of length xy_len*32 bytes
 */

//////////////////////////////////////////////////
// Register roles
//////////////////////////////////////////////////

// Function parameter registers

#ifndef WIN64
// AMD64 calling convention
# define PZ	%rdi
# define PX	%rsi
# define PY	%rdx
# define LEN	%ecx
#else
// X64 calling convention
# define PZ	%rcx
# define PX	%rdx
# define PY	%r8
# define LEN	%r9d
#endif

// Partial product vector temporaries

#define	PPL	%zmm2
#define	PPH	%zmm3

// Zero-extended multiplicand

#define Y	%zmm4

// Mask for keeping only low 32 bits of each 64-bit word

#define LSW	%zmm5

// NB: Registers zmm6-zmm15 are intentionally not used due to the X64 calling convention designating them as callee-save

// Accumulators for partial product vectors
// NB: in case of changes:
// - ACC and A0L must be the same
// - all references to xmm16 must refer to the low 128 bits of ACC

#define ACC	%zmm16
#define A0L	%zmm16
#define A1L	%zmm17
#define A2L	%zmm18
#define A3L	%zmm19
#define A4L	%zmm20
#define A5L	%zmm21
#define A6L	%zmm22
#define A7L	%zmm23
#define A0H	%zmm24
#define A1H	%zmm25
#define A2H	%zmm26
#define A3H	%zmm27
#define A4H	%zmm28
#define A5H	%zmm29
#define A6H	%zmm30
#define A7H	%zmm31

// Macro for adding up (halves of) sums of (halves of) partial products

#define ADDPP(AxH, AyL, AyH, AzL, I) \
	vpsrlq	$32, ACC, PPL; \
	valignd	$2, ACC, ACC, ACC{%k1}{z}; \
	vpaddq	PPL, ACC, ACC; \
	vpsrlq	$32, AxH, AxH; vpaddq	AxH, ACC, ACC; \
	vpsrlq	$32, AyL, AyL; vpaddq	AyL, ACC, ACC; \
	vpandq	LSW, AyH, PPL; vpaddq	PPL, ACC, ACC; \
	vpandq	LSW, AzL, PPL; vpaddq	PPL, ACC, ACC; \
	valignd	$16-I, ACC, ACC, %zmm0{%k2}; \
	kaddw	%k2, %k2, %k2

.global innerproduct256, innerproduct256_asm
.text
.p2align 6,,63

//////////////////////////////////////////////////
// C callable wrapper for innerproduct256_asm
//////////////////////////////////////////////////

// Return value (544 bits) is in %zmm1 (high 32 bits) and %zmm0 (low 256 bits).

innerproduct256:

	// Pass parameters unchanged

	call 		innerproduct256_asm

	// Save return value to z (unchanged by the call above)

	vmovdqu64	%zmm0, (PZ)
	vmovd		%xmm1, %eax
	mov		%eax, 64(PZ)

	ret

.p2align 4,,15

//////////////////////////////////////////////////
// Assembler core, return value in zmm1:zmm0
//////////////////////////////////////////////////

// Note: Do not call this from C; use the C wrapper.

innerproduct256_asm:

	// Create mask for low dword in each qword

	vpcmpeqb	%ymm0, %ymm0, %ymm0
	vpmovzxdq	%ymm0, LSW

	// Clear accumulator registers

	vpxorq		A0L, A0L, A0L
	vmovdqa64	A0L, A1L
	vmovdqa64	A0L, A2L
	vmovdqa64	A0L, A3L
	vmovdqa64	A0L, A4L
	vmovdqa64	A0L, A5L
	vmovdqa64	A0L, A6L
	vmovdqa64	A0L, A7L
	vmovdqa64	A0L, A0H
	vmovdqa64	A0L, A1H
	vmovdqa64	A0L, A2H
	vmovdqa64	A0L, A3H
	vmovdqa64	A0L, A4H
	vmovdqa64	A0L, A5H
	vmovdqa64	A0L, A6H
	vmovdqa64	A0L, A7H

	// Skip accumulation of partial products if the input length is zero

	test	LEN, LEN
	jz	AddPP

	//////////////////////////////////////////////////
	// Accumulate partial product halves
	//////////////////////////////////////////////////
Loop:
	vpmovzxdq 	(PY), Y
	prefetchnta	8192(PY)
	prefetchnta	8192(PX)

	add	$32, PY

	vpmuludq 0*4(PX){1to8}, Y, PPL;	vpsrlq $32, PPL, PPH;	vpandq LSW, PPL, PPL;	vpaddq PPL, A0L, A0L;	vpaddq PPH, A0H, A0H
	vpmuludq 1*4(PX){1to8}, Y, PPL;	vpsrlq $32, PPL, PPH;	vpandq LSW, PPL, PPL;	vpaddq PPL, A1L, A1L;	vpaddq PPH, A1H, A1H
	vpmuludq 2*4(PX){1to8}, Y, PPL;	vpsrlq $32, PPL, PPH;	vpandq LSW, PPL, PPL;	vpaddq PPL, A2L, A2L;	vpaddq PPH, A2H, A2H
	vpmuludq 3*4(PX){1to8}, Y, PPL;	vpsrlq $32, PPL, PPH;	vpandq LSW, PPL, PPL;	vpaddq PPL, A3L, A3L;	vpaddq PPH, A3H, A3H
	vpmuludq 4*4(PX){1to8}, Y, PPL;	vpsrlq $32, PPL, PPH;	vpandq LSW, PPL, PPL;	vpaddq PPL, A4L, A4L;	vpaddq PPH, A4H, A4H
	vpmuludq 5*4(PX){1to8}, Y, PPL;	vpsrlq $32, PPL, PPH;	vpandq LSW, PPL, PPL;	vpaddq PPL, A5L, A5L;	vpaddq PPH, A5H, A5H
	vpmuludq 6*4(PX){1to8}, Y, PPL;	vpsrlq $32, PPL, PPH;	vpandq LSW, PPL, PPL;	vpaddq PPL, A6L, A6L;	vpaddq PPH, A6H, A6H
	vpmuludq 7*4(PX){1to8}, Y, PPL;	vpsrlq $32, PPL, PPH;	vpandq LSW, PPL, PPL;	vpaddq PPL, A7L, A7L;	vpaddq PPH, A7H, A7H

	add	$32, PX

	dec	LEN
	jnz	Loop

	//////////////////////////////////////////////////
	// Add partial products
	//////////////////////////////////////////////////
AddPP:
	// Load mask register values

	mov		$0x1555, %eax
	kmovd		%eax, %k1

	mov		$1, %eax
	kmovd		%eax, %k2

	// ACC starts with the value of A0L

	valignd		$16, ACC, ACC, %zmm0{%k2}{z}	// Store least significant 32 bits of ACC
	kshiftlw	$1, %k2, %k2

	//////////////////////////////////////////////////

	vpsrlq		$32, ACC, PPL
	valignd		$2, ACC, ACC, ACC{%k1}{z}
	vpaddq		PPL, ACC, ACC

	vpandq		LSW, A0H, PPL
	vpaddq		PPL, ACC, ACC

	vpandq		LSW, A1L, PPL
	vpaddq		PPL, ACC, ACC

	// Word 1 of z is ready
	valignd		$15, ACC, ACC, %zmm0{%k2}
	kshiftlw	$1, %k2, %k2

	//////////////////////////////////////////////////

	ADDPP(A0H, A1L, A1H, A2L, 2);
	ADDPP(A1H, A2L, A2H, A3L, 3);
	ADDPP(A2H, A3L, A3H, A4L, 4);
	ADDPP(A3H, A4L, A4H, A5L, 5);
	ADDPP(A4H, A5L, A5H, A6L, 6);
	ADDPP(A5H, A6L, A6H, A7L, 7);

	//////////////////////////////////////////////////

	vpsrlq		$32, ACC, PPL;
	valignd		$2, ACC, ACC, ACC{%k1}{z};
	vpaddq		PPL, ACC, ACC;
	vpsrlq		$32, A6H, A6H; vpaddq	A6H, ACC, ACC;
	vpsrlq		$32, A7L, A7L; vpaddq	A7L, ACC, ACC;
	vpandq		LSW, A7H, PPL; vpaddq	PPL, ACC, ACC;
	valignd		$16-8, ACC, ACC, %zmm0{%k2}
	kshiftlw	$1, %k2, %k2

	//////////////////////////////////////////////////

	vpsrlq		$32, ACC, PPL;
	valignd		$2, ACC, ACC, ACC{%k1}{z};
	vpaddq		PPL, ACC, ACC;
	vpsrlq		$32, A7H, A7H; vpaddq	A7H, ACC, ACC;
	valignd		$16-9, ACC, ACC, %zmm0{%k2}
	kshiftlw	$1, %k2, %k2

	//////////////////////////////////////////////////

#undef ADDPP
#define ADDPP(I) \
	vpsrlq		$32, ACC, PPL; \
	valignd		$2, ACC, ACC, ACC{%k1}{z}; \
	vpaddq		PPL, ACC, ACC; \
	valignd		$16-I, ACC, ACC, %zmm0{%k2}; \
	kshiftlw	$1, %k2, %k2

	ADDPP(10);
	ADDPP(11);
	ADDPP(12);
	ADDPP(13);
	ADDPP(14);
	ADDPP(15);

	//////////////////////////////////////////////////

	vpsrlq		$32, ACC, PPL;
	valignd		$2, ACC, ACC, ACC{%k1}{z};
	vpaddq		PPL, ACC, ACC;
	vmovdqa64	ACC, %zmm1{%k1}{z}

	ret

	//////////////////////////////////////////////////

// No executable stack
.section .note.GNU-stack
