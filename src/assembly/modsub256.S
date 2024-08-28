/*
 * void modsub256(uint64_t *z, const uint64_t *x, const uint64_t *y, uint64_t xy_len, const uint64_t m[6])
 *
 * Copyright (C) 2024 Dag Arne Osvik
 *
 * Modular subtraction of vectors over finite fields up to 256 bits.
 *
 * Parameters:
 *  z       out  Result. Vector of canonical least non-negative residues.
 *  x,y     in   Pointers to vectors of length xy_len.
 *               Elements are 256-bit little-endian canonical residues.
 *  xy_len  in   Length of input vectors. Limited to 2^59-1.
 *  m       in   Pointer to 6-qword array containing the modulus m, its negative inverse mod 2^64 and 2^288/m.
 */

.global modsub256

//////////////////////////////////////////////////
// Register roles
//////////////////////////////////////////////////

//		AMD64	WIN64
//	%rax	X0	X0
//	%rcx	LEN	PZ
//	%rdx	PY	PX
//	%rbx	X1	X1
//	%rsp	-	-
//	%rbp	X2	X2
//	%rsi	PX	X3
//	%rdi	PZ	PM
//	%r8	PM	PY
//	%r9	X3	LEN
//	%r10	X4	X4
//	%r11	T0	T0
//	%r12	T1	T1
//	%r13	T2	T2
//	%r14	T3	T3
//	%r15	-	-

#ifndef WIN64
// AMD64 calling convention
# define PZ	%rdi
# define PX     %rsi
# define PY     %rdx
# define LEN    %rcx
# define PM     %r8

# define X0	%rax
# define X1	%rbx
# define X2	%rbp
# define X3	%r9
# define X4	%r10

#else
// X64 calling convention
# define PZ     %rcx
# define PX     %rdx
# define PY     %r8
# define LEN    %r9
# define PM     %rdi

# define X0	%rax
# define X1	%rbx
# define X2	%rbp
# define X3	%rsi
# define X4	%r10

#endif

# define T0	%r11
# define T1	%r12
# define T2	%r13
# define T3	%r14

.text
.p2align 6,,63
modsub256:

#ifndef WIN64
	pushq	%rbx
	pushq	%rbp
	pushq	%r12
	pushq	%r13
	pushq	%r14
#else
	movq	%rsp, %rax

	pushq	%rbx
	pushq	%rbp
	pushq	%rsi
	pushq	%rdi
	pushq	%r12
	pushq	%r13
	pushq	%r14

	// Load pointer to the modulus from stack
	movq	40(%rax), PM
#endif

	shlq	$5, LEN	// 32 bytes per element

	jz	Done	// Zero-length vector, nothing to do

	// Change to negatively-indexed pointers

	addq	LEN, PX
	addq	LEN, PY
	addq	LEN, PZ
	negq	LEN

	jmp	Loop

.p2align 6,,63
Loop:
	// Clear carry and underflow register

	xorq	X4, X4

	// Load x

	movq	0*8(PX, LEN), X0
	movq	1*8(PX, LEN), X1
	movq	2*8(PX, LEN), X2
	movq	3*8(PX, LEN), X3

	// Subtract y

	subq	0*8(PY, LEN), X0
	sbbq	1*8(PY, LEN), X1
	sbbq	2*8(PY, LEN), X2
	sbbq	3*8(PY, LEN), X3
	sbbq	$0, X4

	// Copy x-y to t

	movq	X0, T0
	movq	X1, T1
	movq	X2, T2
	movq	X3, T3

	// Add m

	addq	0*8(PM), X0
	adcq	1*8(PM), X1
	adcq	2*8(PM), X2
	adcq	3*8(PM), X3

	// Copy x-y+m to t if X4 is nonzero (x-y underflowed)

	testq	X4, X4

	cmovnzq	X0, T0
	cmovnzq	X1, T1
	cmovnzq	X2, T2
	cmovnzq	X3, T3

	// Copy t to z

	movq	T0, 0*8(PZ, LEN)
	movq	T1, 1*8(PZ, LEN)
	movq	T2, 2*8(PZ, LEN)
	movq	T3, 3*8(PZ, LEN)

	addq	$32, LEN
	jnz	Loop

Done:
#ifndef WIN64
	popq	%r14
	popq	%r13
	popq	%r12
	popq	%rbp
	popq	%rbx
#else
	popq	%r14
	popq	%r13
	popq	%r12
	popq	%rdi
	popq	%rsi
	popq	%rbp
	popq	%rbx
#endif

	ret

// No executable stack
.section .note.GNU-stack
