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

func ExampleByIndices() {
	a := New(WithShape(2, 2), WithBacking([]float64{
		100, 200,
		300, 400,
	}))
	indices := New(WithBacking([]int{1, 1, 1, 0, 1}))
	b, err := ByIndices(a, indices, 0) // we select rows 1, 1, 1, 0, 1
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Printf("a:\n%v\nindices: %v\nb:\n%v\n", a, indices, b)

	// Output:
	// a:
	// ⎡100  200⎤
	// ⎣300  400⎦
	//
	// indices: [1  1  1  0  1]
	// b:
	// ⎡300  400⎤
	// ⎢300  400⎥
	// ⎢300  400⎥
	// ⎢100  200⎥
	// ⎣300  400⎦

}

func ExampleByIndicesB() {
	a := New(WithShape(2, 2), WithBacking([]float64{
		100, 200,
		300, 400,
	}))
	indices := New(WithBacking([]int{1, 1, 1, 0, 1}))
	b, err := ByIndices(a, indices, 0) // we select rows 1, 1, 1, 0, 1
	if err != nil {
		fmt.Println(err)
		return
	}

	outGrad := b.Clone().(*Dense)
	outGrad.Memset(1.0)

	grad, err := ByIndicesB(a, outGrad, indices, 0)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Printf("a:\n%v\nindices: %v\nb:\n%v\ngrad:\n%v", a, indices, b, grad)

	// Output:
	// a:
	// ⎡100  200⎤
	// ⎣300  400⎦
	//
	// indices: [1  1  1  0  1]
	// b:
	// ⎡300  400⎤
	// ⎢300  400⎥
	// ⎢300  400⎥
	// ⎢100  200⎥
	// ⎣300  400⎦
	//
	// grad:
	// ⎡1  1⎤
	// ⎣4  4⎦

}
