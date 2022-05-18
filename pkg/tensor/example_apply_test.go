package tensor_test

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

	"github.com/bhojpur/neuron/pkg/tensor"
)

func ExampleDense_Apply() {
	a := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4}))
	cube := func(a float64) float64 { return a * a * a }

	b, err := a.Apply(cube)
	if err != nil {
		fmt.Printf("b is an error %v", err)
	}
	fmt.Printf("a and b are the same object - %t\n", a.Eq(b))
	fmt.Printf("a is unmutated\n%v\n", a)

	c, err := a.Apply(cube, tensor.WithReuse(a))
	if err != nil {
		fmt.Printf("c is an error %v\n", err)
	}
	fmt.Printf("a and c are the same object - %t\n", a.Eq(c))

	fmt.Printf("a is now mutated\n%v\n", a)
	// Output:
	// a and b are the same object - false
	// a is unmutated
	// ⎡1  2⎤
	// ⎣3  4⎦
	//
	// a and c are the same object - true
	// a is now mutated
	// ⎡ 1   8⎤
	// ⎣27  64⎦
}
