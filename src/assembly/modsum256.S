/*
 * void modsum256(uint64_t z[4], const uint64_t *x, uint32_t x_len, const uint64_t m[6])
 *
 * Copyright (C) 2024 Dag Arne Osvik
 *
 * Modular sum across components of a vector over finite fields from 225 bits up to 256 bits.
 * Requires AVX-512 support.
 *
 * Parameters:
 *  z       out  Result. Canonical least non-negative residue.
 *  x       in   Pointer to vector of length x_len.
 *  x_len   in   Length of input vector x. Limited to 2^32-1.
 *  m       in   Pointer to 6-qword array containing the modulus m, its negative inverse mod 2^64 and 2^288/m.
 */

.global modsum256
.text
.p2align 6,,63

//////////////////////////////////////////////////
// Register roles
//////////////////////////////////////////////////

#ifndef WIN64
// AMD64 calling convention
# define PZ	%rdi
# define PX     %rsi
# define LEN    %edx
# define PM     %rcx

# define T0	%r8
# define T1	%r9
# define T2	%r10
# define T3	%r11
# define T4	PX

# define PL	%rax
# define PH	%rbx

#else
// X64 calling convention
# define PZ     %rcx
# define PX     %rdx
# define LEN    %r8d
# define PM     %r9

// TODO: verify correctness of allocations below

# define T0	%r8
# define T1	%rbp
# define T2	%r10
# define T3	%r11
# define T4	PZ

# define PL	%rax
# define PH	%rbx
#endif

#define MU	40(PM)

modsum256:
#ifdef WIN64
	mov	PZ,   1*8(%rsp)
	mov	%rbx, 2*8(%rsp)
	mov	%rbp, 3*8(%rsp)
#else
	push	%rbx
#endif
	//////////////////////////////////////////////////
	// Prepare four accumulators
	//////////////////////////////////////////////////

	vpxorq		%zmm0, %zmm0, %zmm0
	vmovdqa64	%zmm0, %zmm1
	vmovdqa64	%zmm0, %zmm2
	vmovdqa64	%zmm0, %zmm3

	xor	%r8d,  %r8d
	xor	%r9d,  %r9d
	xor	%r10d, %r10d
	xor	%r11d, %r11d

	mov	$0x1555, %eax
	kmovw	%eax, %k1

	mov	$0xff80, %eax
	kmovw	%eax, %k2

	mov	$0x01ff, %eax
	kmovw	%eax, %k3

	mov	LEN, %eax
	test	%eax, %eax
	jz	Done

	and	$3, %eax
	shr	$2, LEN

	cmp	$1, %eax
	je	1f

	cmp	$2, %eax
	je	2f

	cmp	$3, %eax
	jne	Loop4

3:	vpmovzxdq 	2*32(PX), %zmm4;	vpaddq	%zmm4, %zmm0, %zmm0
2:	vpmovzxdq 	1*32(PX), %zmm4;	vpaddq	%zmm4, %zmm1, %zmm1
1:	vpmovzxdq 	0*32(PX), %zmm4;	vpaddq	%zmm4, %zmm2, %zmm2

	test	LEN, LEN
	jz	Accumulate

.p2align 6,,63
Loop4:
	//////////////////////////////////////////////////
	// Zero-expand each dword and add it to a qword
	//////////////////////////////////////////////////

	vpmovzxdq 	0*32(PX), %zmm4;	vpaddq	%zmm4, %zmm0, %zmm0
	vpmovzxdq 	1*32(PX), %zmm4;	vpaddq	%zmm4, %zmm1, %zmm1
	vpmovzxdq 	2*32(PX), %zmm4;	vpaddq	%zmm4, %zmm2, %zmm2
	vpmovzxdq 	3*32(PX), %zmm4;	vpaddq	%zmm4, %zmm3, %zmm3

	sub		$-128, PX

	dec		LEN
	jnz		Loop4

Accumulate:
	//////////////////////////////////////////////////
	// Combine accumulators
	//////////////////////////////////////////////////

	vpaddq	%zmm1, %zmm0, %zmm0
	vpaddq	%zmm3, %zmm2, %zmm2
	vpaddq	%zmm2, %zmm0, %zmm0

	//////////////////////////////////////////////////
	// Propagate carries
	//////////////////////////////////////////////////

	mov	$8, %eax

	valignd	$1, %zmm3, %zmm0, %zmm3{%k2}{z}	// Shift lowest dword of zmm0 into zmm3

Propagate:
	vpsrlq	$32, %zmm0, %zmm1		// Zero-expand high dword carries from zmm0 to zmm1
	valignd	$2, %zmm0, %zmm0, %zmm0{%k1}{z}	// Shift low dwords of zmm0 one qword down
	vpaddq	%zmm1, %zmm0, %zmm0		// Add carries
	valignd	$1, %zmm3, %zmm0, %zmm3{%k2}{z}	// Shift lowest dword of zmm0 into zmm3

	dec	%eax
	jnz	Propagate

	//////////////////////////////////////////////////
	// Move intermediate result to integer registers
	//////////////////////////////////////////////////

	// The top 9 dwords of zmm3 now contain the sum. Copy them to the low end of zmm0.

	valignd	$7, %zmm3, %zmm3, %zmm0{%k3}{z}

	// Copy to integer registers

	vmovq	%xmm0, T0;	valignq	$1, %zmm0, %zmm0, %zmm0
	vmovq	%xmm0, T1;	valignq	$1, %zmm0, %zmm0, %zmm0
	vmovq	%xmm0, T2;	valignq	$1, %zmm0, %zmm0, %zmm0
	vmovq	%xmm0, T3;	valignq	$1, %zmm0, %zmm0, %zmm0
	vmovq	%xmm0, T4

	//////////////////////////////////////////////////
	// Reduce using single-word Barrett
	//////////////////////////////////////////////////

	// q1 is low 32 bits of T4 and high 32 bits of T3

	movq	T3, %rax
	shrd	$32, T4, %rax
	mulq	MU		// Multiply by mu. q2 in rdx:rax, q3 in rdx

	// Subtract r2 from r1

	mulx	0*8(PM), PL, PH; sub	PL, T0; sbb	PH, T1;
	mulx	2*8(PM), PL, PH; sbb	PL, T2; sbb	PH, T3;	sbb	$0, T4
	mulx	1*8(PM), PL, PH; sub	PL, T1; sbb	PH, T2;
	mulx	3*8(PM), PL, PH; sbb	PL, T3; sbb	PH, T4

	// Two conditional subtractions to guarantee canonicity of the result

	// Save the result

	mov	T0, 0*8(PZ)
	mov	T1, 1*8(PZ)
	mov	T2, 2*8(PZ)
	mov	T3, 3*8(PZ)

	// Subtract the modulus

	sub	0*8(PM), T0
	sbb	1*8(PM), T1
	sbb	2*8(PM), T2
	sbb	3*8(PM), T3
	sbb	$0, T4

	// If borrow, skip to the end

	jb	Done

	// Save the result

	mov	T0, 0*8(PZ)
	mov	T1, 1*8(PZ)
	mov	T2, 2*8(PZ)
	mov	T3, 3*8(PZ)

	// Subtract the modulus

	sub	0*8(PM), T0
	sbb	1*8(PM), T1
	sbb	2*8(PM), T2
	sbb	3*8(PM), T3
	sbb	$0, T4

	// If borrow, skip to the end

	jb	Done

	// Save the result

	mov	T0, 0*8(PZ)
	mov	T1, 1*8(PZ)
	mov	T2, 2*8(PZ)
	mov	T3, 3*8(PZ)
Done:
	//////////////////////////////////////////////////
	// Cleanup
	//////////////////////////////////////////////////

#ifdef WIN64
	mov	1*8(%rsp), PZ
	mov	2*8(%rsp), %rbx
	mov	3*8(%rsp), %rbp
#else
	pop	%rbx
#endif

	ret

// No executable stack
.section .note.GNU-stack
