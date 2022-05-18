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
	"fmt"

	"github.com/bhojpur/neuron/pkg/quality"
	qp "github.com/bhojpur/neuron/pkg/quality/plot"
	"github.com/bhojpur/neuron/pkg/tensor"
	"github.com/bhojpur/neuron/pkg/tensor/native"
	"gonum.org/v1/plot"
)

// ConfusionMatrix represents a confusion matrix.
//
// It is ordered in the following shape (predicted, actual)
type ConfusionMatrix struct {
	*tensor.Dense
	classes quality.Classes
	Labels  []string    // optional Labels
	Iter    [][]float64 // a native iterator of the confusion matrix. This allows for quick access.
}

// Confusion creates a Confusion Matrix.
func Confusion(pred, correct quality.Classes) *ConfusionMatrix {
	classes := correct.Clone()
	classes = append(classes, pred...)
	classes = classes.Distinct()

	data := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(len(classes), len(classes)))
	iter, err := native.MatrixF64(data)
	if err != nil {
		panic(err)
	}
	for i := range correct {
		c := correct[i]
		p := pred[i]

		iter[p][c]++
	}
	return &ConfusionMatrix{
		classes: classes,
		Dense:   data,
		Iter:    iter,
	}
}

// Heatmap is a convenience method to create a heatmap.
// It creates a *gonum.org/v1/plot.Plot, which contains a heatmap structure.
// The Palette, X, Y are all accessible for customization.
//
// Not safe to be run concurrently.
func (m *ConfusionMatrix) Heatmap() (*plot.Plot, error) {
	labels := m.Labels
	if len(labels) == 0 {
		labels = make([]string, len(m.classes))
		for i := range labels {
			labels[i] = fmt.Sprintf("Class %d", i)
		}
	}
	return qp.Heatmap(m.Dense, labels)
}
