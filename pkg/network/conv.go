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

	"github.com/bhojpur/neuron/pkg/engine/hm"
	"github.com/bhojpur/neuron/pkg/tensor"
)

// ConsConv is a Conv construction function. It takes a engine.Input that has a *engine.Node.
// Defaults:
//		activation function: Rectify
// 		kernel shape: (5,5)
// 		pad: (1,1)
//		stride: (1,1)
//		dilation: (1,1)
func ConsConv(in engine.Input, opts ...ConsOpt) (retVal Layer, err error) {
	x := in.Node()
	if x == nil {
		return nil, fmt.Errorf("ConsConv expects a *Node. Got input %v of  %T instead", in, in)
	}

	inshape := x.Shape()
	if inshape.Dims() != 4 || inshape.Dims() == 0 {
		return nil, fmt.Errorf("Expected shape is either a vector or a matrix, got %v", inshape)
	}

	l, err := NewConv(opts...)
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
func (l *Conv) Init(xs ...*engine.Node) (err error) {
	x := xs[0]
	g := x.Graph()
	of := x.Dtype()
	name := l.name + "_w"
	l.w = engine.NewTensor(g, of, 4, engine.WithShape(l.size[0], l.size[1], l.kernelShape[0], l.kernelShape[1]), engine.WithName(name), engine.WithInit(engine.GlorotN(1.0)))

	l.initialized = true

	return nil
}

// Conv represents a convolution layer
type Conv struct {
	w *engine.Node

	name string
	size []int

	kernelShape           tensor.Shape
	pad, stride, dilation []int

	// optional config
	dropout *float64 // nil when shouldn't be applied

	act ActivationFunction

	initialized  bool
	computeFLOPs bool
	flops        int
}

func NewConv(opts ...ConsOpt) (*Conv, error) {
	l := &Conv{
		act:         engine.Rectify,
		kernelShape: tensor.Shape{5, 5},
		pad:         []int{1, 1},
		stride:      []int{1, 1},
		dilation:    []int{1, 1},
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

		if l, ok = o.(*Conv); !ok {
			return nil, fmt.Errorf("Construction Option returned a non Conv. Got %T instead", o)
		}
	}
	return l, nil
}

// SetDropout sets the dropout of the layer
func (l *Conv) SetDropout(d float64) error {
	l.dropout = &d
	return nil
}

// SetSize sets the size of the layer
func (l *Conv) SetSize(s ...int) error {
	l.size = s
	return nil
}

// SetName sets the name of the layer
func (l *Conv) SetName(n string) error {
	l.name = n
	return nil
}

// SetActivationFn sets the activation function of the layer
func (l *Conv) SetActivationFn(act ActivationFunction) error {
	l.act = act
	return nil
}

// Model will return the engine.Nodes associated with this convolution layer
func (l *Conv) Model() engine.Nodes {
	return engine.Nodes{
		l.w,
	}
}

// Fwd runs the equation forwards
func (l *Conv) Fwd(x engine.Input) engine.Result {
	if err := engine.CheckOne(x); err != nil {
		return wrapErr(l, "checking input: %w", err)
	}

	xN := x.Node()
	if !l.initialized {
		if err := l.Init(xN); err != nil {
			return wrapErr(l, "Initializing a previously uninitialized Conv layer: %w", err)
		}
	}

	c, err := engine.Conv2d(xN, l.w, l.kernelShape, l.pad, l.stride, l.dilation)
	if err != nil {
		return wrapErr(l, "applying conv2d %v %v: %w", x.Node().Shape(), l.w.Shape(), err)
	}

	result, err := l.act(c)
	if err != nil {
		return wrapErr(l, "applying activation function: %w", err)
	}

	if l.dropout != nil {
		result, err = engine.Dropout(result, *l.dropout)
		if err != nil {
			return wrapErr(l, "applying dropout: %w", err)
		}
	}

	// Side effects are cool
	if l.computeFLOPs {
		l.flops = l.doComputeFLOPs(xN.Shape())
	}

	logf("%T shape %s: %v", l, l.name, result.Shape())

	return result
}

// Type will return the hm.Type of the convolution layer
func (l *Conv) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('b'))
}

// Shape will return the tensor.Shape of the convolution layer
func (l *Conv) Shape() tensor.Shape {
	return l.w.Shape()
}

// Name will return the name of the convolution layer
func (l *Conv) Name() string {
	return l.name
}

// Describe will describe a convolution layer
func (l *Conv) Describe() {
	panic("not implemented")
}

func (l *Conv) FLOPs() int { return l.flops }

// doComputeFLOPs computes the rough number of floating point operations for this layer.
//
// Adapted from: https://stats.stackexchange.com/a/296793
func (l *Conv) doComputeFLOPs(input tensor.Shape) int {
	shp := l.w.Shape()
	n := shp[1] * shp[2] * shp[3]
	flopsPerInstance := n + 1
	instancesPerFilter := ((input[1] - shp[2] + 2*l.pad[0]) / l.stride[0]) + 1 // rows
	instancesPerFilter *= ((input[2] - shp[3] + 2*l.pad[1]) / l.stride[1]) + 1 // multiplying with cols

	flopsPerFilter := instancesPerFilter * flopsPerInstance
	retVal := flopsPerFilter * shp[0] // multiply with number of filters

	// we assume if there's an activation function, we'll assume there's a multiply and a add.
	if l.act != nil {
		retVal += shp[0] * instancesPerFilter
	}
	return retVal
}

var (
	_ namesetter      = &Conv{}
	_ actSetter       = &Conv{}
	_ dropoutConfiger = &Conv{}
	_ Term            = &Conv{}
)
