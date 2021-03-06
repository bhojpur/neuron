// +build sse
// +build amd64
// +build !fastmath

// Copyright (c) 2018 Bhojpur Consulting Private Limited, India. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

/*
InvSqrt is a function that inverse square roots (1/√x) each element in a []float64

The SSE version uses SHUFPD to "broadcast" the 1.0 constant to the X1 register. The rest proceeds as expected
*/
#include "textflag.h"

#define one 0x3ff0000000000000

// func InvSqrt(a []float64)
TEXT ·InvSqrt(SB), NOSPLIT, $0
	MOVQ a_data+0(FP), SI
	MOVQ SI, CX
	MOVQ a_len+8(FP), AX  // len(a) into AX

	// make sure that len(a) >= 1
	XORQ BX, BX
	CMPQ BX, AX
	JGE  done

	MOVQ $one, DX

	SUBQ $2, AX
	JL   remainder

	// back up the first element of the slice
	MOVQ (SI), BX
	MOVQ DX, (SI)

	// broadcast 1.0 to all elements of X1
	// 0x00 shuffles the least significant bits of the X1 reg, which means the first element is repeated
	MOVUPD (SI), X1
	SHUFPD $0x00, X1, X1

	MOVAPD X1, X2 // backup, because X1 will get clobbered in DIVPD

	// restore the first element now we're done
	MOVQ BX, (SI)

loop:
	MOVAPD X2, X1
	SQRTPD (SI), X0
	DIVPD  X0, X1
	MOVUPD X1, (SI)

	// we processed 2 elements. Each element is 8 bytes. So jump 16 ahead
	ADDQ $16, SI

	SUBQ $2, AX
	JGE  loop

remainder:
	ADDQ $2, AX
	JE   done

remainder1:
	MOVQ   DX, X1
	MOVSD  (SI), X0
	SQRTSD X0, X0
	DIVSD  X0, X1
	MOVSD  X1, (SI)

	ADDQ $8, SI
	DECQ AX
	JNE  remainder1

done:
	RET
