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

// This is an example of how to use `IteratorFromDense` from a row-major Dense tensor
func Example_iteratorRowmajor() {
	T := New(WithShape(2, 3), WithBacking([]float64{0, 1, 2, 3, 4, 5}))
	it := IteratorFromDense(T)
	fmt.Printf("T:\n%v\n", T)

	for i, err := it.Start(); err == nil; i, err = it.Next() {
		fmt.Printf("i: %d, coord: %v\n", i, it.Coord())
	}

	// Output:
	// T:
	// ⎡0  1  2⎤
	// ⎣3  4  5⎦
	//
	// i: 0, coord: [0 1]
	// i: 1, coord: [0 2]
	// i: 2, coord: [1 0]
	// i: 3, coord: [1 1]
	// i: 4, coord: [1 2]
	// i: 5, coord: [0 0]

}

// This is an example of using `IteratorFromDense` on a col-major Dense tensor. More importantly
// this example shows the order of the iteration.
func Example_iteratorcolMajor() {
	T := New(WithShape(2, 3), WithBacking([]float64{0, 1, 2, 3, 4, 5}), AsFortran(nil))
	it := IteratorFromDense(T)
	fmt.Printf("T:\n%v\n", T)

	for i, err := it.Start(); err == nil; i, err = it.Next() {
		fmt.Printf("i: %d, coord: %v\n", i, it.Coord())
	}

	// Output:
	// T:
	// ⎡0  2  4⎤
	// ⎣1  3  5⎦
	//
	// i: 0, coord: [0 1]
	// i: 2, coord: [0 2]
	// i: 4, coord: [1 0]
	// i: 1, coord: [1 1]
	// i: 3, coord: [1 2]
	// i: 5, coord: [0 0]

}

func ExampleSliceIter() {
	T := New(WithShape(3, 3), WithBacking(Range(Float64, 0, 9)))
	S, err := T.Slice(makeRS(1, 3), makeRS(1, 3))
	if err != nil {
		fmt.Printf("Err %v\n", err)
		return
	}
	fmt.Printf("S (requires iterator? %t)\n%v\n", S.(*Dense).RequiresIterator(), S)
	it := IteratorFromDense(S.(*Dense))
	for i, err := it.Start(); err == nil; i, err = it.Next() {
		fmt.Printf("i %d, coord %v\n", i, it.Coord())
	}

	// Output:
	// S (requires iterator? true)
	// ⎡4  5⎤
	// ⎣7  8⎦
	//
	// i 0, coord [0 1]
	// i 1, coord [1 0]
	// i 3, coord [1 1]
	// i 4, coord [0 0]

}
