package native_test

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
	. "github.com/bhojpur/neuron/pkg/tensor/native"
)

type MyType int

func Example_vector() {
	backing := []MyType{
		0, 1, 2, 3,
	}
	T := tensor.New(tensor.WithShape(4), tensor.WithBacking(backing))
	val, err := Vector(T)
	if err != nil {
		fmt.Printf("error: %v", err)
	}
	it := val.([]MyType)
	fmt.Println(it)

	// Output:
	// [0 1 2 3]
}

func Example_matrix() {
	backing := []MyType{
		0, 1,
		2, 3,
		4, 5,
	}
	T := tensor.New(tensor.WithShape(3, 2), tensor.WithBacking(backing))
	val, err := Matrix(T)
	if err != nil {
		fmt.Printf("error: %v", err)
	}

	it := val.([][]MyType)
	fmt.Println(it)

	// Output:
	// [[0 1] [2 3] [4 5]]
}

func Example_tensor3() {
	backing := []MyType{
		0, 1, 2, 3,
		4, 5, 6, 7,
		8, 9, 10, 11,

		12, 13, 14, 15,
		16, 17, 18, 19,
		20, 21, 22, 23,
	}
	T := tensor.New(tensor.WithShape(2, 3, 4), tensor.WithBacking(backing))
	val, err := Tensor3(T)
	if err != nil {
		fmt.Printf("error: %v", err)
	}
	it := val.([][][]MyType)
	fmt.Println(it)

	//Output:
	// [[[0 1 2 3] [4 5 6 7] [8 9 10 11]] [[12 13 14 15] [16 17 18 19] [20 21 22 23]]]
}
