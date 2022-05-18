package plot

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
	"image/color"
	"math"

	"github.com/bhojpur/neuron/pkg/quality/internal/tensorutils"
	"github.com/bhojpur/neuron/pkg/tensor"
	"github.com/pkg/errors"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/palette"
	"gonum.org/v1/plot/plotter"
)

// internal data structure to hold a heatmap
type heatmap struct {
	data mat.Matrix
}

func (m heatmap) Dims() (c, r int)   { r, c = m.data.Dims(); return c, r }
func (m heatmap) Z(c, r int) float64 { return m.data.At(r, c) }
func (m heatmap) X(c int) float64    { return float64(c) }
func (m heatmap) Y(r int) float64    { return float64(r) }

// internal data structure for heatmap ticks
type ticks []string

func (t ticks) Ticks(min, max float64) []plot.Tick {
	var retVal []plot.Tick
	for i := math.Trunc(min); i <= max; i++ {
		label := ""
		if int(i) < len(t) {
			label = t[int(i)]
		}

		retVal = append(retVal, plot.Tick{Value: i, Label: label})
	}
	return retVal
}

func Heatmap(x tensor.Tensor, labels []string) (p *plot.Plot, err error) {
	pal := palette.Heat(48, 1)
	dense := tensorutils.GetDense(x)
	mat, err := tensor.ToMat64(dense, tensor.UseUnsafe())
	if err != nil {
		return nil, errors.Wrapf(err, "Converting a x (of type %T) to a mat.Dense failed.", x)
	}

	m := heatmap{mat}
	hm := plotter.NewHeatMap(m, pal)
	p = plot.New()
	hm.NaN = color.RGBA{0, 0, 0, 0} // black for NaN

	p.Add(hm)
	p.X.Tick.Label.Rotation = 1.5
	p.X.Tick.Marker = ticks(labels)
	p.Y.Tick.Marker = ticks(labels)

	return
}
