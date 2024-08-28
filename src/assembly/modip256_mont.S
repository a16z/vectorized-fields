/*
 * void modip256_mont(uint64_t z[4], const uint64_t *x, const uint64_t *y, uint32_t xy_len, const uint64_t m[6])
 *
 * Copyright (C) 2024 Dag Arne Osvik
 *
 * Modular inner product for vectors over finite fields from 225 bits up to 256 bits using Montgomery form.
 *
 * Dependency: innerproduct256() is used to calculate the integer inner product of the vectors.
 *
 * Parameters:
 *  z       out  Result in Montgomery form. Canonical least non-negative residue.
 *  x,y     in   Pointers to vectors of length xy_len.
 *               Elements are 256-bit little-endian residues in Montgomery form.
 *  xy_len  in   Length of input vectors. Limited to 2^32-1 by innerproduct256().
 *  m       in   Pointer to 6-qword array containing the modulus m, its negative inverse mod 2^64 and 2^288/m.
 */

.global modip256_mont
.extern	innerproduct256_asm
.text
.p2align 6,,63

//////////////////////////////////////////////////
// Register roles
//////////////////////////////////////////////////

#ifndef WIN64
// AMD64 calling convention
# define PZ	%rdi
//define PX     %rsi
//define PY     %rdx
//define LEN    %ecx
# define PM     %r8

#else
// X64 calling convention
# define PZ     %rcx
//define PX     %rdx
//define PY     %r8
//define LEN    %r9d
# define PM     %r8
#endif

#define INV	32(PM)
#define MU	40(PM)

#ifndef WIN64
# define PL	%rax
# define PH	%rcx
# define T0	%r10
# define T1	%r11
# define T2	%r12
# define T3	%r13
# define T4	%r14
#else
# define PL     %rax
# define PH     %r9
# define T0	%r10
# define T1	%r11
# define T2	%r12
# define T3	%r13
# define T4	%r14
#endif

modip256_mont:

	//////////////////////////////////////////////////
	// Get inner product from innerproduct256_asm
	//////////////////////////////////////////////////

	// innerproduct256_asm() modifies LEN, PX and PY, but leaves PZ and PM intact.
	// It returns its 544-bit (72-byte) result in zmm1:zmm0.
	// After innerproduct256_asm() only the modular reduction remains to be computed.

	call	innerproduct256_asm

#ifdef WIN64
	// Load pointer to the modulus from stack
	mov	40(%rsp), PM

	// Save registers

	push	%rbx
#endif
	push	%r14
	push	%r13
	push	%r12

	// Extract the 4 least significant qwords of %zmm0

	vmovq	%xmm0, T1; valignq	$1, %zmm0, %zmm1, %zmm0	// Shift in low word from zmm1
	vmovq	%xmm0, T2; valignq	$1, %zmm0, %zmm0, %zmm0
	vmovq	%xmm0, T3; valignq	$1, %zmm0, %zmm0, %zmm0
	vmovq	%xmm0, T4; valignq	$1, %zmm0, %zmm0, %zmm0
	xorq	T0, T0

	//////////////////////////////////////////////////
	// Montgomery reduction
	//////////////////////////////////////////////////

	// See Handbook of Applied Cryptography, Algorithm 14.32.

	movq	INV, %rdx	// Load negative inverse mod 2^64

	mulx	T1, %rdx, PH

	mulx	0*8(PM), PL, PH; add	PL, T1; adc	PH, T2
	mulx	2*8(PM), PL, PH; adc	PL, T3; adc	PH, T4; adc	$0, T0
	mulx	1*8(PM), PL, PH; add	PL, T2; adc	PH, T3
	mulx	3*8(PM), PL, PH; adc	PL, T4; adc	PH, T0; adc	$0, T1

	movq	INV, %rdx

	mulx	T2, %rdx, PH

	mulx	0*8(PM), PL, PH; add	PL, T2; adc	PH, T3
	mulx	2*8(PM), PL, PH; adc	PL, T4; adc	PH, T0; adc	$0, T1
	mulx	1*8(PM), PL, PH; add	PL, T3; adc	PH, T4
	mulx	3*8(PM), PL, PH; adc	PL, T0; adc	PH, T1; adc	$0, T2

	movq	INV, %rdx

	mulx	T3, %rdx, PH

	mulx	0*8(PM), PL, PH; add	PL, T3; adc	PH, T4
	mulx	2*8(PM), PL, PH; adc	PL, T0; adc	PH, T1; adc	$0, T2
	mulx	1*8(PM), PL, PH; add	PL, T4; adc	PH, T0
	mulx	3*8(PM), PL, PH; adc	PL, T1; adc	PH, T2; adc	$0, T3

	movq	INV, %rdx

	mulx	T4, %rdx, PH

	mulx	0*8(PM), PL, PH; add	PL, T4; adc	PH, T0
	mulx	2*8(PM), PL, PH; adc	PL, T1; adc	PH, T2; adc	$0, T3
	mulx	1*8(PM), PL, PH; add	PL, T0; adc	PH, T1
	mulx	3*8(PM), PL, PH; adc	PL, T2; adc	PH, T3; adc	$0, T4

	// Add the remaining 5 qwords (9 dwords) from zmm0

	vmovq	%xmm0, PL; add	PL, T0;	valignq	$1, %zmm0, %zmm0, %zmm0
	vmovq	%xmm0, PL; adc	PL, T1;	valignq	$1, %zmm0, %zmm0, %zmm0
	vmovq	%xmm0, PL; adc	PL, T2;	valignq	$1, %zmm0, %zmm0, %zmm0
	vmovq	%xmm0, PL; adc	PL, T3;	valignq	$1, %zmm0, %zmm0, %zmm0
	vmovq	%xmm0, PL; adc	PL, T4	// T4 < 2^32

	//////////////////////////////////////////////////
	// Barrett reduction
	//////////////////////////////////////////////////

	// For explanation of mu, q1, q2, q3, r1, r2, see Handbook of
	// Applied Cryptography, Algorithm 14.42.

	// q1 is low 32 bits of T4 and high 32 bits of T3

	movq	T3, %rax
	shrd	$32, T4, %rax	// q1
	mulq	MU		// Multiply by mu. q2 in rdx:rax, q3 in rdx

	// 2^32 < mu < 2^64. 0 <= q1 < 2^64. q2 < 2^128.
	// r1 is in T4:T3:T2:T1:T0

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

	jb	done

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

	jb	done

	// Save result

	mov	T0, 0*8(PZ)
	mov	T1, 1*8(PZ)
	mov	T2, 2*8(PZ)
	mov	T3, 3*8(PZ)

	//////////////////////////////////////////////////
	// Cleanup
	//////////////////////////////////////////////////
done:
	// Restore registers

	pop	%r12
	pop	%r13
	pop	%r14
#ifdef WIN64
	pop	%rbx
#endif

	ret

// No executable stack
.section .note.GNU-stack
