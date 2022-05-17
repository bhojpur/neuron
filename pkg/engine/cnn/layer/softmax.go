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

type SoftmaxLayer struct {
	outputDims  []int
	recentInput maths.Tensor
}

func NewSoftmaxLayer(inputDims []int) *SoftmaxLayer {
	return &SoftmaxLayer{
		outputDims: inputDims}
}

func (o *SoftmaxLayer) ForwardPropagation(input maths.Tensor) maths.Tensor {
	o.recentInput = input

	output := input.Zeroes()
	expSum := 0.0

	for i := 0; i < input.Len(); i++ {
		output.SetValue(i, math.Exp(input.At(i)))
		expSum += output.At(i)
	}

	output.Apply(func(val float64, idx int) float64 {
		return val / expSum
	})

	return *output
}
func (o *SoftmaxLayer) BackwardPropagation(gradient maths.Tensor, lr float64) maths.Tensor {
	return *gradient.MulElem(o.derivatives(o.recentInput))
}

func (o *SoftmaxLayer) OutputDims() []int {
	return o.outputDims
}

func (o *SoftmaxLayer) derivatives(input maths.Tensor) *maths.Tensor {
	output := make([]float64, input.Len())
	expSum := 0.0

	for i := 0; i < input.Len(); i++ {
		output[i] = math.Exp(input.At(i))
		expSum += output[i]
	}

	for i := 0; i < input.Len(); i++ {
		output[i] *= expSum - output[i]
	}
	return maths.NewTensor(input.Dimensions(), maths.DivideFloat64SliceByFloat64(output, expSum*expSum))
}
