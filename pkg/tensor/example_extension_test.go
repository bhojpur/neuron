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
	//"errors"
	"fmt"
	"reflect"

	"github.com/bhojpur/neuron/pkg/tensor"
	"github.com/pkg/errors"
)

// In this example, we want to create and handle a tensor of *MyType

// First, define MyType

// MyType is defined
type MyType struct {
	x, y int
}

func (T MyType) Format(s fmt.State, c rune) { fmt.Fprintf(s, "(%d, %d)", T.x, T.y) }

// MyDtype this the dtype of MyType. This value is populated in the init() function below
var MyDtype tensor.Dtype

// MyEngine supports additions of MyType, as well as other Dtypes
type MyEngine struct {
	tensor.StdEng
}

// For simplicity's sake, we'd only want to handle MyType-MyType or MyType-Int interactions
// Also, we only expect Dense tensors
// You're of course free to define your own rules

// Add adds two tensors
func (e MyEngine) Add(a, b tensor.Tensor, opts ...tensor.FuncOpt) (retVal tensor.Tensor, err error) {
	switch a.Dtype() {
	case MyDtype:
		switch b.Dtype() {
		case MyDtype:
			data := a.Data().([]*MyType)
			datb := b.Data().([]*MyType)
			for i, v := range data {
				v.x += datb[i].x
				v.y += datb[i].y
			}
			return a, nil
		case tensor.Int:
			data := a.Data().([]*MyType)
			datb := b.Data().([]int)
			for i, v := range data {
				v.x += datb[i]
				v.y += datb[i]
			}
			return a, nil
		}
	case tensor.Int:
		switch b.Dtype() {
		case MyDtype:
			data := a.Data().([]int)
			datb := b.Data().([]*MyType)
			for i, v := range datb {
				v.x += data[i]
				v.y += data[i]
			}
		default:
			return e.StdEng.Add(a, b, opts...)
		}
	default:
		return e.StdEng.Add(a, b, opts...)
	}
	return nil, errors.New("Unreachable")
}

func init() {
	MyDtype = tensor.Dtype{reflect.TypeOf(&MyType{})}
}

func Example_extension() {
	T := tensor.New(tensor.WithEngine(MyEngine{}),
		tensor.WithShape(2, 2),
		tensor.WithBacking([]*MyType{
			&MyType{0, 0}, &MyType{0, 1},
			&MyType{1, 0}, &MyType{1, 1},
		}))
	ones := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]int{1, 1, 1, 1}), tensor.WithEngine(MyEngine{}))
	T2, _ := T.Add(ones)

	fmt.Printf("T:\n%+v", T)
	fmt.Printf("T2:\n%+v", T2)

	// output:
	//T:
	// Matrix (2, 2) [2 1]
	// ???(1, 1)  (1, 2)???
	// ???(2, 1)  (2, 2)???
	// T2:
	// Matrix (2, 2) [2 1]
	// ???(1, 1)  (1, 2)???
	// ???(2, 1)  (2, 2)???
}
