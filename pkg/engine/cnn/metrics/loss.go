package metrics

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

type LossFunction interface {
	CalculateLoss(target, predicted []float64) maths.Tensor
	CalculateLossDerivative(target, predicted []float64) maths.Tensor
}

type CrossEntropyLoss struct{}

func (c *CrossEntropyLoss) CalculateLoss(target, predicted []float64) maths.Tensor {
	lossValues := make([]float64, len(target))
	for i := 0; i < len(lossValues); i++ {
		if target[i] == 1 {
			lossValues[i] = -1.0 * math.Log(predicted[i])
		} else {
			lossValues[i] = -1.0 * math.Log(1-predicted[i])
		}
	}
	return *maths.NewTensor([]int{len(lossValues)}, lossValues)
}

func (c *CrossEntropyLoss) CalculateLossDerivative(target, predicted []float64) maths.Tensor {
	lossDerivatives := make([]float64, len(target))

	for i := 0; i < len(lossDerivatives); i++ {
		if target[i] == 1 {
			lossDerivatives[i] = -1.0 / predicted[i]
		} else {
			lossDerivatives[i] = 1.0 / (1 - predicted[i])
		}
	}
	return *maths.NewTensor([]int{len(lossDerivatives)}, lossDerivatives)
}
