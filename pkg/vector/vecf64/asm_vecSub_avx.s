// +build avx
// +build amd64

/*
Sub subtracts two []float64 with some SIMD optimizations using AVX.

Instead of doing this:
	for i := 0; i < len(a); i++ {
	    a[i] -= b[i]
	}

Here, I use the term "pairs" to denote an element of `a` and and element of `b` that will be added together. 
a[i], b[i] is a pair.

Using AVX, we can simultaneously add 4 pairs at the same time, which will look something like this:
	for i := 0; i < len(a); i+=4{
		a[i:i+4] -= b[i:i+4]	// this code won't run.
	}

These are the registers I use to store the relevant information:
	SI - Used to store the top element of slice A (index 0). This register is incremented every loop
	DI - used to store the top element of slice B. Incremented every loop
	AX - len(a) is stored in here. Volatile register. AX is also used as the "working" count of the length that is decremented.
	AX - len(a) is stored in here. AX is also used as the "working" count of the length that is decremented.
	Y0, Y1 - YMM registers. 
	X0, X1 - XMM registers.

With regards to VSUBPD and VSUBSD, it turns out that the description of these instructions are:
	VSUBPD ymm1, ymm2, ymm3: Subtract packed double-precision floating-point values in ymm3/mem from ymm2 and stores result in ymm1.[0]

The description is written with intel's syntax (in this form: Dest, Src1, Src2). 
When converting to Go's ASM it becomes: (Src2, Src1, Dest)

This pseudocode best explains the rather simple assembly:
	lenA := len(a)
	i := 0
	loop:
	for {
		a[i:i+4*8] -= b[i:i+4*8]
		lenA -= 4
		i += 4*8 // 4 elements, 8 bytes each

		if lenA < 0{
			break
		}
	}

	remainder2head:
	lenA += 4
	if lenA == 0 {
		return
	}

	remainder4:
	for {
		a[i:i+2*4] -= b[i:i+2*4]
		lenA -=2
		i += 2 * 8  // 2 elements, 8 bytes each
		
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
		a[i] -= b[i]
		i+=8 // each element is 8 bytes
		lenA--
	}

	return

Citation
========
[0]http://www.felixcloutier.com/x86/SUBPD.html
*/
#include "textflag.h"

// func subAsm(a, b []float64)
TEXT ·subAsm(SB), NOSPLIT, $0
	MOVQ a_data+0(FP), SI
	MOVQ b_data+24(FP), DI // use destination index register for this

	MOVQ a_len+8(FP), AX  // len(a) into AX
	MOVQ AX, AX           // len(a) into AX for working purposes

	// each ymm register can take up to 4 float64s.
	SUBQ $4, AX
	JL   remainder

loop:
	// a[0] to a[3]
	// VMOVUPD (SI), Y0
	// VMOVUPD (DI), Y1
	// VSUBPD Y1, Y0, Y0
	// VMOVUPD Y0, (SI)
	BYTE $0xc5; BYTE $0xfd; BYTE $0x10; BYTE $0x06 // vmovupd 0(%rsi),%ymm0
	BYTE $0xc5; BYTE $0xfd; BYTE $0x10; BYTE $0x0f // vmovupd 0(%rdi),%ymm1
	BYTE $0xc5; BYTE $0xfd; BYTE $0x5c; BYTE $0xc1 // vsubpd %ymm1,%ymm0,%ymm0
	BYTE $0xc5; BYTE $0xfd; BYTE $0x11; BYTE $0x06 // vmovupd %ymm0, 0(%rsi)

	// 4 elements processed. Each element is 8 bytes. So jump 32 bytes
	ADDQ $32, SI
	ADDQ $32, DI

	SUBQ $4, AX
	JGE  loop

remainder:
	ADDQ $4, AX
	JE   done

	SUBQ $2, AX
	JL   remainder1head

remainder2:
	// VMOVUPD (SI), X0
	// VMOVUPD (DI), X1
	// VSUBPD X1, X0, X0
	// VMOVUPD X0, (SI)
	BYTE $0xc5; BYTE $0xf9; BYTE $0x10; BYTE $0x06 // vmovupd 0(%rsi),%xmm0
	BYTE $0xc5; BYTE $0xf9; BYTE $0x10; BYTE $0x0f // vmovupd 0(%rdi),%xmm1
	BYTE $0xc5; BYTE $0xf9; BYTE $0x5c; BYTE $0xc1 // vsubpd  %xmm1,%xmm0,%xmm0
	BYTE $0xc5; BYTE $0xf9; BYTE $0x11; BYTE $0x06 // vmovupd %xmm0, 0(%rsi)

	ADDQ $16, SI
	ADDQ $16, DI
	SUBQ $2, AX
	JGE  remainder2

remainder1head:
	ADDQ $2, AX
	JE   done

remainder1:
	// copy into the appropriate registers
	// VMOVSD	(SI), X0
	// VMOVSD	(DI), X1
	// VSUBSD	X1, X0, X0
	// VMOVSD	X0, (SI)
	BYTE $0xc5; BYTE $0xfb; BYTE $0x10; BYTE $0x06 // vmovsd 0(%rsi),%xmm0
	BYTE $0xc5; BYTE $0xfb; BYTE $0x10; BYTE $0x0f // vmovsd 0(%rdi),%xmm1
	BYTE $0xc5; BYTE $0xfb; BYTE $0x5c; BYTE $0xc1 // vsubsd %xmm1,%xmm0,%xmm0
	BYTE $0xc5; BYTE $0xfb; BYTE $0x11; BYTE $0x06 // vmovsd %xmm0, 0(%rsi)

	// update pointer to the top of the data
	ADDQ $8, SI
	ADDQ $8, DI

	DECQ AX
	JNE  remainder1

done:
	RET