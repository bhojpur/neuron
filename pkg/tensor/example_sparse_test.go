package tensor

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

import "fmt"

func Example_sparse_basics() {
	xs := []int{1, 2, 6, 8}
	ys := []int{1, 2, 1, 6}
	vals := []float32{3, 1, 4, 1}

	S := CSCFromCoord(Shape{9, 7}, xs, ys, vals)
	T := New(WithShape(9, 7), Of(Float32)) // dense

	Result, _ := Add(S, T)
	fmt.Printf("When adding a sparse tensor to a dense tensor, the result is of %T:\n=============================================================================\n%+#s\n", Result, Result)
	Result, _ = Add(T, S)
	fmt.Printf("And vice versa - %T\n=========================\n%+#s\n", Result, Result)

	// Output:
	// When adding a sparse tensor to a dense tensor, the result is of *tensor.Dense:
	// =============================================================================
	// Matrix (9, 7) [7 1]
	// ⎡0  0  0  0  0  0  0⎤
	// ⎢0  3  0  0  0  0  0⎥
	// ⎢0  0  1  0  0  0  0⎥
	// ⎢0  0  0  0  0  0  0⎥
	// ⎢0  0  0  0  0  0  0⎥
	// ⎢0  0  0  0  0  0  0⎥
	// ⎢0  4  0  0  0  0  0⎥
	// ⎢0  0  0  0  0  0  0⎥
	// ⎣0  0  0  0  0  0  1⎦
	//
	// And vice versa - *tensor.Dense
	// =========================
	// Matrix (9, 7) [7 1]
	// ⎡0  0  0  0  0  0  0⎤
	// ⎢0  3  0  0  0  0  0⎥
	// ⎢0  0  1  0  0  0  0⎥
	// ⎢0  0  0  0  0  0  0⎥
	// ⎢0  0  0  0  0  0  0⎥
	// ⎢0  0  0  0  0  0  0⎥
	// ⎢0  4  0  0  0  0  0⎥
	// ⎢0  0  0  0  0  0  0⎥
	// ⎣0  0  0  0  0  0  1⎦
}

func Example_sparse_advanced() {
	xs := []int{1, 2, 6, 8}
	ys := []int{1, 2, 1, 6}
	vals := []int16{3, 1, 4, 1}

	S := CSCFromCoord(Shape{9, 7}, xs, ys, vals)
	T := New(WithShape(9, 7), Of(Int16))     // dense
	Reuse := New(WithShape(9, 7), Of(Int16)) // reuse must be a *Dense because the result will always be a dense
	Result, _ := Add(S, T, WithReuse(Reuse))
	fmt.Printf("Operations involving sparse tensors also do take the usual function options like Reuse:\n%+#s\nResult == Reuse: %t", Result, Result == Reuse)

	// Output:
	// Operations involving sparse tensors also do take the usual function options like Reuse:
	// Matrix (9, 7) [7 1]
	// ⎡0  0  0  0  0  0  0⎤
	// ⎢0  3  0  0  0  0  0⎥
	// ⎢0  0  1  0  0  0  0⎥
	// ⎢0  0  0  0  0  0  0⎥
	// ⎢0  0  0  0  0  0  0⎥
	// ⎢0  0  0  0  0  0  0⎥
	// ⎢0  4  0  0  0  0  0⎥
	// ⎢0  0  0  0  0  0  0⎥
	// ⎣0  0  0  0  0  0  1⎦
	//
	// Result == Reuse: true
}
