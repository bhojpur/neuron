package t2cudnn

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
	cudnn "github.com/bhojpur/neuron/pkg/drivers/cu/dnn"
	"github.com/bhojpur/neuron/pkg/engine"
	"github.com/bhojpur/neuron/pkg/tensor"
)

var (
	_ Tensor = &tensor.Dense{}
	_ Tensor = &engine.Node{}
	_ Tensor = tensor.Tensor(nil)
)

type Tensor interface {
	Shape() tensor.Shape
	Strides() []int
	Dtype() tensor.Dtype
}

// Describe extracts the metadata from a tensor.Dense and returns a cuDNN TensorDescriptor
func Describe(t Tensor) (*cudnn.TensorDescriptor, error) {

	shape := t.Shape().Clone()
	strides := make([]int, len(shape))
	copy(strides, t.Strides())
	switch shape.Dims() {
	case 0:
		// TODO?
	case 1:
		// TODO?
	case 2:
		// take a 2D shape and make it 4D:
		// because Bhojpur Neuron only takes NCHW formats
		// any 2D matrix can be thought of as (1,1 H, W)

		shape = append(shape, 0, 0) // shape after would be (H, W, 0, 0)
		copy(shape[2:], shape[0:])  // shift the shape down by copying: (H, W, H, W)
		shape[0] = 1                // (1, W, H, W)
		shape[1] = 1                // (1,1,H, W)

		strides = append(strides, 0, 0)
		copy(strides[2:], strides[0:])
		strides[0] = strides[2] * shape[2]
		strides[1] = strides[2] * shape[2] // no, this is not a bug.
	case 3:
		shape = append(shape, 0)
		copy(shape[1:], shape[0:])
		shape[0] = 1

		strides = append(strides, 0)
		copy(strides[1:], strides[0:])
		strides[0] = strides[1] * shape[1]
	default:
	}

	return cudnn.NewTensorDescriptor(cudnn.NCHW, Dtype2DataType(t.Dtype()), shape, strides)
}

func DescribeAsFilter(t Tensor, format cudnn.TensorFormat) (*cudnn.Filter, error) {
	shape := t.Shape().Clone()
	return cudnn.NewFilter(Dtype2DataType(t.Dtype()), format, shape)
}

// Dtype2DataType converts a tensor.Dtype to a cudnnDataType.
func Dtype2DataType(t tensor.Dtype) cudnn.DataType {
	switch t.Name() {
	case "float64":
		return cudnn.Double
	case "float32":
		return cudnn.Float
	case "float16":
		return cudnn.Half
	case "int8":
		return cudnn.Int8
	case "int32":
		return cudnn.Int32
	case "int128":
		return cudnn.Int8x4
	}
	panic("Unreachable")
}
