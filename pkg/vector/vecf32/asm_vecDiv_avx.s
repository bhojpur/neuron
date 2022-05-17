// +build avx
// +build amd64

/*
This function adds two []float32 with some SIMD optimizations using AVX.

Instead of doing this:
	for i := 0; i < len(a); i++ {
	    a[i] /= b[i]
	}

Here, I use the term "pairs" to denote an element of `a` and and element of `b` that will be added together. 
a[i], b[i] is a pair.

Using AVX, we can simultaneously add 8 pairs at the same time, which will look something like this:
	for i := 0; i < len(a); i+=8{
	    a[i:i+8] /= b[i:i+8] // this code won't run.
	}

AVX registers are 256 bits, meaning we can put 8 float32s in there. 

These are the registers I use to store the relevant information:
	SI - Used to store the top element of slice A (index 0). This register is incremented every loop
	DI - used to store the top element of slice B. Incremented every loop
	AX - len(a) is stored in here. AX is also used as the "working" count of the length that is decremented.
	Y0, Y1 - YMM registers. 
	X0, X1 - XMM registers.

With regards to VDIVPS and VSUBSS, it turns out that the description of these instructions are:
	VDIVPS ymm1, ymm2, ymm3: Subtract packed double-precision floating-point values in ymm3/mem from ymm2 and stores result in ymm1.[0]

The description is written with intel's syntax (in this form: Dest, Src1, Src2). 
When converting to Go's ASM it becomes: (Src2, Src1, Dest)

This pseudocode best explains the rather simple assembly:

	lenA := len(a)
	i := 0

	loop:
	for {
		a[i:i+8*4] /= b[i:i+8*4]
		lenA -= 8
		i += 8 * 4 // 8 elements, 4 bytes each
		
		if lenA < 0{
			break
		}
	}

	remainder4head:
	lenA += 8
	if lenA == 0 {
		return
	}

	remainder4:
	for {
		a[i:i+4*4] /= b[i:i+4*4]
		lenA -=4
		i += 4 * 4  // 4 elements, 4 bytes each
		
		if lenA < 0{
			break
		}
	}

	remainder1head:
	lenA += 4
	if lenA == 0 {
		return
	}

	remainder1:
	for {
		a[i] /= b[i]
		i+=4 // each element is 4 bytes
		lenA--
	}

	return

Citation
========
[0]http://www.felixcloutier.com/x86/DIVPS.html
*/
#include "textflag.h"

// func divAsm(a, b []float32)
TEXT ·divAsm(SB), NOSPLIT, $0
	MOVQ a_data+0(FP), SI
	MOVQ b_data+24(FP), DI // use destination index register for this

	MOVQ a_len+8(FP), AX  // len(a) into AX - +8, because first 8 is pointer, second 8 is length, third 8 is cap

	SUBQ $8, AX
	JL   remainder

	// each ymm register can take up to 4 float64s.
	// There are 8 ymm registers (8 pairs to do addition) available (TODO: check how to access the other 8 ymm registers without fucking things up)
	// Therefore a total of 16 elements can be processed at a time

loop:
	// a[0] to a[7]
	// VMOVUPS (SI), Y0
	// VMOVUPS (DI), Y1
	// VDIVPS Y1, Y0, Y0
	// VMOVUPS Y0, (SI)
	BYTE $0xc5; BYTE $0xfc; BYTE $0x10; BYTE $0x06 // vmovups (%rsi),%ymm0
	BYTE $0xc5; BYTE $0xfc; BYTE $0x10; BYTE $0x0f // vmovups (%rdi),%ymm1
	BYTE $0xc5; BYTE $0xfc; BYTE $0x5e; BYTE $0xc1 // vdivps %ymm1,%ymm0,%ymm0
	BYTE $0xc5; BYTE $0xfc; BYTE $0x11; BYTE $0x06 // vmovups %ymm0,(%rsi)

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, AX
	JGE  loop

remainder:
	ADDQ $8, AX
	JE   done

	SUBQ $4, AX
	JL   remainder1head

remainder4:
	// VMOVUPS (SI), X0
	// VMOVUPS (DI), X1
	// VDIVPS X1, X0, X0
	// VMOVUPS X0, (SI)
	BYTE $0xc5; BYTE $0xf8; BYTE $0x10; BYTE $0x06 // vmovups (%rsi),%xmm0
	BYTE $0xc5; BYTE $0xf8; BYTE $0x10; BYTE $0x0f // vmovups (%rdi),%xmm1
	BYTE $0xc5; BYTE $0xf8; BYTE $0x5e; BYTE $0xc1 // vdivps %xmm1,%xmm0,%xmm0
	BYTE $0xc5; BYTE $0xf8; BYTE $0x11; BYTE $0x06 // vmovups %xmm0,(%rsi)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $4, AX
	JGE  remainder4

remainder1head:
	ADDQ $4, AX
	JE   done

remainder1:
	// VMOVSS	(SI), X0
	// VMOVSS	(DI), X1
	// VDIVSS	X1, X0, X0
	// VMOVSS	X0, (SI)
	BYTE $0xc5; BYTE $0xfa; BYTE $0x10; BYTE $0x06 // vmovss (%rsi),%xmm0
	BYTE $0xc5; BYTE $0xfa; BYTE $0x10; BYTE $0x0f // vmovss (%rdi),%xmm1
	BYTE $0xc5; BYTE $0xfa; BYTE $0x5e; BYTE $0xc1 // vdivss %xmm1,%xmm0,%xmm0
	BYTE $0xc5; BYTE $0xfa; BYTE $0x11; BYTE $0x06 // vmovss %xmm0,(%rsi)

	// update pointer to the top of the data
	ADDQ $4, SI
	ADDQ $4, DI

	DECQ AX
	JNE  remainder1

done:
	RET