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

func ExampleSum() {
	T := New(WithBacking([]float64{0, 1, 2, 3}), WithShape(2, 2))
	fmt.Printf("T:\n%v\n", T)

	// sum along axis 0
	summed, _ := Sum(T, 0)
	fmt.Printf("Summed:\n%v\n", summed)

	// to keep dims, simply reshape
	summed.Reshape(1, 2)
	fmt.Printf("Summed (Kept Dims - Shape: %v):\n%v\n\n", summed.Shape(), summed)

	// summing along multiple axes
	summed, _ = Sum(T, 1, 0)
	fmt.Printf("Summed along (1, 0): %v", summed)

	// Output:
	// T:
	// ⎡0  1⎤
	// ⎣2  3⎦
	//
	// Summed:
	// [2  4]
	// Summed (Kept Dims - Shape: (1, 2)):
	// R[2  4]
	//
	// Summed along (1, 0): 6
}

func ExampleSum_sliced() {
	T := New(WithBacking([]float64{0, 1, 2, 3}), WithShape(2, 2))
	fmt.Printf("T:\n%v\n", T)

	V, _ := T.Slice(nil, S(1))
	fmt.Printf("V:\n%v\n", V)

	Σ, _ := Sum(V)
	fmt.Printf("Σ: %v", Σ)

	// Output:
	// T:
	// ⎡0  1⎤
	// ⎣2  3⎦
	//
	// V:
	// [1  3]
	// Σ: 4

}

func ExampleArgmax() {
	T := New(WithBacking([]float64{0, 100, 200, 3}), WithShape(2, 2))
	fmt.Printf("T:\n%v\n", T)

	// argmax along the x-axis
	am, _ := Argmax(T, 0)
	fmt.Printf("Argmax: %v\n", am)
	fmt.Printf("Argmax is %T of %v", am, am.Dtype())

	// Output:
	// T:
	// ⎡  0  100⎤
	// ⎣200    3⎦
	//
	// Argmax: [1  0]
	// Argmax is *tensor.Dense of int
}

func ExampleArgmax_sliced() {
	T := New(WithBacking([]float64{0, 100, 200, 3}), WithShape(2, 2))
	fmt.Printf("T:\n%v\n", T)

	// slice  creates a view
	V, _ := T.Slice(nil, S(1))

	// argmax along the x-axis
	am, _ := Argmax(V, 0)
	fmt.Printf("Argmax: %v\n", am)
	fmt.Printf("Argmax is %T of %v", am, am.Dtype())

	// Output:
	// T:
	// ⎡  0  100⎤
	// ⎣200    3⎦
	//
	// Argmax: 0
	// Argmax is *tensor.Dense of int

}

func ExampleArgmin() {
	T := New(WithBacking([]float64{0, 100, 200, 3}), WithShape(2, 2))
	fmt.Printf("T:\n%v\n", T)

	// argmax along the x-axis
	am, _ := Argmin(T, 0)
	fmt.Printf("Argmin: %v\n", am)
	fmt.Printf("Argmin is %T of %v", am, am.Dtype())

	// Output:
	// T:
	// ⎡  0  100⎤
	// ⎣200    3⎦
	//
	// Argmin: [0  1]
	// Argmin is *tensor.Dense of int
}
