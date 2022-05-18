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

import (
	"fmt"
)

// Data shows how the shape of the *Dense actually affects the return value of .Data().
func ExampleDense_Data() {
	T := New(WithShape(2, 2), WithBacking([]float64{1, 2, 3, 4}))
	fmt.Printf("Basics:\n======\nAny kind of arrays: %v\n", T.Data())

	fmt.Printf("\nScalar-like\n===========\n")
	T = New(WithShape(), FromScalar(3.14))
	fmt.Printf("WithShape(), FromScalar: %v\n", T.Data())

	T = New(WithShape(), WithBacking([]float64{3.14}))
	fmt.Printf("WithShape(), With a slice of 1 as backing: %v\n", T.Data())

	T = New(WithShape(1), FromScalar(3.14))
	fmt.Printf("WithShape(1), With an initial scalar: %v\n", T.Data())

	T = New(WithShape(1, 1), WithBacking([]float64{3.14}))
	fmt.Printf("WithShape(1, 1), With an initial scalar: %v\n", T.Data())

	T = New(WithShape(1, 1), FromScalar(3.14))
	fmt.Printf("WithShape(1, 1), With an initial scalar: %v\n", T.Data())

	T.Reshape()
	fmt.Printf("After reshaping to (): %v\n", T.Data())

	// Output:
	// Basics:
	// ======
	// Any kind of arrays: [1 2 3 4]
	//
	// Scalar-like
	// ===========
	// WithShape(), FromScalar: 3.14
	// WithShape(), With a slice of 1 as backing: 3.14
	// WithShape(1), With an initial scalar: [3.14]
	// WithShape(1, 1), With an initial scalar: [3.14]
	// WithShape(1, 1), With an initial scalar: [3.14]
	// After reshaping to (): 3.14

}
