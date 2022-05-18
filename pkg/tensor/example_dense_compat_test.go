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

	"github.com/apache/arrow/go/arrow/array"
	"github.com/apache/arrow/go/arrow/memory"
	"github.com/bhojpur/neuron/pkg/tensor"
)

func ExampleDense_Arrow() {
	pool := memory.NewGoAllocator()

	b := array.NewFloat64Builder(pool)
	defer b.Release()

	b.AppendValues(
		[]float64{1, 2, 3, -1, 4, 5},
		[]bool{true, true, true, false, true, true},
	)

	arr := b.NewFloat64Array()
	defer arr.Release()
	fmt.Printf("arrow array = %v\n", arr)

	a := tensor.FromArrowArray(arr)
	fmt.Printf("tensor = %v\n", a)

	// Output:
	// arrow array = [1 2 3 (null) 4 5]
	// tensor = C[ 1   2   3  --   4   5]
}
