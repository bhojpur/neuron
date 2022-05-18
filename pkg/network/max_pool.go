//go:build !cuda
// +build !cuda

package network

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

	"github.com/bhojpur/neuron/pkg/engine"
	"github.com/bhojpur/neuron/pkg/engine/hm"
	"github.com/bhojpur/neuron/pkg/tensor"
)

// ConsMaxPool is a MaxPool construction function. It takes a engine.Input that has a *engine.Node.
// Defaults:
// 		kernel shape: (2,2)
// 		pad: (0,0)
//		stride: (2,2)
func ConsMaxPool(in engine.Input, opts ...ConsOpt) (retVal Layer, err error) {
	x := in.Node()
	if x == nil {
		return nil, fmt.Errorf("ConsMaxPool expects a *Node. Got input %v of  %T instead", in, in)
	}

	inshape := x.Shape()
	if inshape.Dims() != 4 || inshape.Dims() == 0 {
		return nil, fmt.Errorf("Expected shape is a matrix")
	}

	l, err := NewMaxPool(opts...)
	if err != nil {
		return nil, err
	}

	// prep
	if err = l.Init(x); err != nil {
		return nil, err
	}

	return l, nil
}

// Init will initialize the fully connected layer
func (l *MaxPool) Init(xs ...*engine.Node) (err error) {
	l.initialized = true

	return nil
}

// MaxPool represents a MaxPoololution layer
type MaxPool struct {
	name string
	size int

	kernelShape tensor.Shape
	pad, stride []int

	// optional config
	dropout *float64 // nil when shouldn't be applied

	initialized  bool
	computeFLOPs bool

	flops int
}

func NewMaxPool(opts ...ConsOpt) (*MaxPool, error) {
	l := &MaxPool{
		kernelShape: tensor.Shape{2, 2},
		pad:         []int{0, 0},
		stride:      []int{2, 2},
	}

	for _, opt := range opts {
		var (
			o   Layer
			ok  bool
			err error
		)

		if o, err = opt(l); err != nil {
			return nil, err
		}

		if l, ok = o.(*MaxPool); !ok {
			return nil, fmt.Errorf("Construction Option returned a non MaxPool. Got %T instead", o)
		}
	}
	return l, nil
}

// SetSize sets the size of the layer
func (l *MaxPool) SetSize(s int) error {
	l.size = s
	return nil
}

// SetName sets the name of the layer
func (l *MaxPool) SetName(n string) error {
	l.name = n
	return nil
}

// SetDropout sets the dropout of the layer
func (l *MaxPool) SetDropout(d float64) error {
	l.dropout = &d
	return nil
}

// Model will return the engine.Nodes associated with this MaxPoololution layer
func (l *MaxPool) Model() engine.Nodes {
	return engine.Nodes{}
}

// Fwd runs the equation forwards
func (l *MaxPool) Fwd(x engine.Input) engine.Result {
	if err := engine.CheckOne(x); err != nil {
		return engine.Err(fmt.Errorf("Fwd of MaxPool %v: %w", l.name, err))
	}

	result, err := engine.MaxPool2D(x.Node(), l.kernelShape, l.pad, l.stride)
	if err != nil {
		return wrapErr(l, "applying max pool to %v: %w", x.Node().Shape(), err)
	}

	if l.dropout != nil {
		result, err = engine.Dropout(result, *l.dropout)
		if err != nil {
			return engine.Err(err)
		}
	}

	logf("%T shape %s: %v", l, l.name, result.Shape())

	return result
}

// Type will return the hm.Type of the MaxPoololution layer
func (l *MaxPool) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('b'))
}

// Name will return the name of the MaxPoololution layer
func (l *MaxPool) Name() string {
	return l.name
}

// Describe will describe a MaxPoololution layer
func (l *MaxPool) Describe() {
	panic("not implemented")
}

func (l *MaxPool) SetComputeFLOPs(toCompute bool) error {
	l.computeFLOPs = toCompute
	return nil
}

func (l *MaxPool) FLOPs() int { return l.flops }

func (l *MaxPool) doComputeFLOPs(input tensor.Shape) int {
	copyOps := input.TotalSize()

	if l.dropout != nil {
		return 2 * copyOps // dropout is an elementwise mul
	}
	return copyOps
}

var (
	_ sizeSetter      = &MaxPool{}
	_ namesetter      = &MaxPool{}
	_ dropoutConfiger = &MaxPool{}
)
