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
	"github.com/bhojpur/neuron/pkg/engine/cnn/maths"
)

type FullyConnectedLayer struct {
	weights maths.Tensor
	biases  []float64

	inputDims  []int
	outputDims []int

	recentOutput []float64
	recentInput  maths.Tensor
}

func NewFullyConnectedLayer(outputLength int, inputDims []int) *FullyConnectedLayer {
	dense := &FullyConnectedLayer{}
	dense.inputDims = inputDims
	dense.recentInput = *maths.NewTensor(inputDims, nil)
	dense.recentOutput = make([]float64, outputLength)
	dense.outputDims = []int{outputLength}

	dense.weights = *maths.NewTensor(append(inputDims, outputLength), nil)
	dense.weights = *dense.weights.Randomize()

	dense.biases = make([]float64, outputLength)

	return dense
}

func (d *FullyConnectedLayer) ForwardPropagation(input maths.Tensor) maths.Tensor {
	d.recentInput = input

	i := maths.NewRegionsIterator(&d.weights, d.inputDims, []int{})
	for i.HasNext() {
		d.recentOutput[i.CoordIterator.GetCurrentCount()] = i.Next().InnerProduct(&input)
	}
	d.recentOutput = func(l, r []float64) []float64 {
		ret := make([]float64, len(l))
		for i := 0; i < len(ret); i++ {
			ret[i] = l[i] + r[i]
		}
		return ret
	}(d.recentOutput, d.biases)

	return *maths.NewTensor([]int{len(d.recentOutput)}, d.recentOutput)
}

func (d *FullyConnectedLayer) BackwardPropagation(gradient maths.Tensor, lr float64) maths.Tensor {
	var weightsGradient *maths.Tensor
	for i := 0; i < len(gradient.Values()); i++ {
		newGrads := d.recentInput.MulScalar(gradient.Values()[i])
		if weightsGradient == nil {
			weightsGradient = newGrads
		} else {
			weightsGradient = weightsGradient.AppendTensor(newGrads, len(d.weights.Dimensions()))
		}
	}

	inputGradient := maths.NewTensor(d.inputDims, nil)
	j := maths.NewRegionsIterator(&d.weights, d.inputDims, []int{})
	for j.HasNext() {
		newGrads := j.Next().MulScalar(gradient.Values()[j.CoordIterator.GetCurrentCount()-1])
		inputGradient = inputGradient.Add(newGrads, 1)
	}

	d.weights = *d.weights.Add(weightsGradient, -1.0*lr)
	d.biases = maths.AddFloat64Slices(d.biases, maths.MulFloat64ToSlice(gradient.Values(), -1.0*lr))

	return *inputGradient
}

func (d *FullyConnectedLayer) OutputDims() []int { return d.outputDims }
