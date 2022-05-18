// +build sse
// +build amd64

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

#include "textflag.h"

// func divAsm(a, b []float64)
TEXT ·divAsm(SB), NOSPLIT, $0
	MOVQ a_data+0(FP), SI
	MOVQ b_data+24(FP), DI // use destination index register for this

	MOVQ a_len+8(FP), AX  // len(a) into AX

	// check if there are at least 8 elements
	SUBQ $8, AX
	JL   remainder

loop:

	// a[0]
	MOVAPD (SI), X0
	MOVAPD (DI), X1
	DIVPD  X1, X0
	MOVAPD X0, (SI)

	MOVAPD 16(SI), X2
	MOVAPD 16(DI), X3
	DIVPD  X3, X2
	MOVAPD X2, 16(SI)

	MOVAPD 32(SI), X4
	MOVAPD 32(DI), X5
	DIVPD  X5, X4
	MOVAPD X4, 32(SI)

	MOVAPD 48(SI), X6
	MOVAPD 48(DI), X7
	DIVPD  X7, X6
	MOVAPD X6, 48(SI)

	// update pointers. 4 registers, 2 elements each, 8 bytes per element
	ADDQ $64, SI
	ADDQ $64, DI

	// len(a) is now 4*2 elements less
	SUBQ $8, AX
	JGE  loop

remainder:
	ADDQ $8, AX
	JE   done

remainderloop:

	// copy into the appropriate registers
	MOVSD (SI), X0
	MOVSD (DI), X1
	DIVSD X1, X0

	// save it back
	MOVSD X0, (SI)

	// update pointer to the top of the data
	ADDQ $8, SI
	ADDQ $8, DI

	DECQ AX
	JNE  remainderloop

done:
	RET
