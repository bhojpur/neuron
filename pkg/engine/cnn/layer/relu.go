package layer

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
	"math"

	"github.com/bhojpur/neuron/pkg/engine/cnn/maths"
)

type ReLULayer struct {
	outputDims  []int
	recentInput maths.Tensor
}

func NewReLULayer(inputDims []int) *ReLULayer {
	return &ReLULayer{
		outputDims: inputDims}
}

func (o *ReLULayer) derivatives(input maths.Tensor) *maths.Tensor {
	output := input.Zeroes()

	for i := 0; i < input.Len(); i++ {
		if input.At(i) > 0 {
			output.SetValue(i, 1)
		} else {
			output.SetValue(i, 0)
		}
	}

	return output
}

func (o *ReLULayer) ForwardPropagation(input maths.Tensor) maths.Tensor {
	o.recentInput = input
	output := input.Zeroes()
	for i := 0; i < input.Len(); i++ {
		output.SetValue(i, math.Max(input.At(i), 0))
	}
	return *output
}
func (o *ReLULayer) BackwardPropagation(gradient maths.Tensor, lr float64) maths.Tensor {
	return *gradient.MulElem(o.derivatives(o.recentInput))
}

func (o *ReLULayer) OutputDims() []int {
	return o.outputDims
}
