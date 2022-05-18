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
Sqrt takes a []float32 and square roots every element in the slice.
*/
#include "textflag.h"

// func Sqrt(a []float64)
TEXT ·Sqrt(SB), NOSPLIT, $0
	MOVQ a_data+0(FP), SI
	MOVQ SI, CX
	MOVQ a_len+8(FP), AX  // len(a) into AX - +8, because first 8 is pointer, second 8 is length, third 8 is cap

	SUBQ $2, AX
	JL   remainder

loop:
	SQRTPD (SI), X0
	MOVUPD X0, (SI)

	// we processed 2 elements. Each element is 8 bytes. So jump 16 ahead
	ADDQ $16, SI

	SUBQ $2, AX
	JGE  loop

remainder:
	ADDQ $2, AX
	JE   done

remainder1:
	MOVSD  (SI), X0
	SQRTSD X0, X0
	MOVSD  X0, (SI)

	ADDQ $8, SI
	DECQ AX
	JNE  remainder1

done:
	RET
